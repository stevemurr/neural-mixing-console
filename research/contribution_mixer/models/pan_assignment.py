"""Balance-aware greedy pan assignment — inference-time post-process.

Given PanNet's per-track logits over {L, C, R} and the per-track contribution
vectors C, this picks per-track pan modes that fill the L/R stereo field
rather than collapsing similar tracks to the same side.

Algorithm:

  1. Process mono tracks in order of model confidence (highest first). The
     model's confident commitments get locked in early; uncertain decisions
     happen later when the running balance state can guide them.

  2. Maintain a running `balance` vector — per-bin L-vs-R contribution of
     mono tracks already assigned (positive = left-heavier).

  3. For each track, compute the cost of each choice:

        cost(class) = −log P(class)  +  λ_balance · mean( new_balance(class)² )

     where new_balance simulates the effect of the choice. Pick the min-cost
     class.

  4. Pick the cheapest class; update the balance state.

Why "model confidence first": when the model is uncertain (similar L and R
probabilities — the guitar-pair case), the balance heuristic should decide.
But when the model is confident, that signal should commit early so the
balance state correctly accounts for those tracks before less-certain ones
ask "which side has room?".

We use the log1p-mel C tensor (what the model sees) as the balance state.
It's not strictly linear-additive in magnitude space, but it captures the
right direction: louder tracks shift balance more.
"""

from __future__ import annotations

import torch


def balanced_assignment(
    logits: torch.Tensor,                  # (N, 3) — per-track L/C/R logits
    C: torch.Tensor,                       # (N, n_bins) — log1p-mel C
    is_stereo: torch.Tensor,               # (N,) bool
    track_mask: torch.Tensor | None = None,  # (N,) bool
    lambda_balance: float = 0.5,
) -> torch.Tensor:
    """Greedy balance-aware per-track pan assignment.

    Returns (N,) pan_dir in {-1, 0, +1}. Stereo tracks and padded slots
    always get 0.
    """
    if logits.dim() != 2 or logits.shape[-1] != 3:
        raise ValueError(f"expected logits (N, 3); got {logits.shape}")
    N, _ = logits.shape
    device = logits.device

    pan_dir = torch.zeros(N, device=device)

    probs = torch.softmax(logits, dim=-1).clamp(min=1e-8)
    log_probs = torch.log(probs)

    if track_mask is None:
        track_mask = torch.ones(N, dtype=torch.bool, device=device)
    mono_real = track_mask & ~is_stereo.to(torch.bool)
    mono_idx = torch.where(mono_real)[0]
    if mono_idx.numel() == 0:
        return pan_dir

    # Sort mono tracks by descending model confidence
    confidence = probs[mono_idx].max(dim=-1).values
    order_in_mono = torch.argsort(confidence, descending=True)
    ordered_idx = mono_idx[order_in_mono]

    balance = torch.zeros(C.shape[-1], device=device, dtype=C.dtype)
    class_to_pan = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=C.dtype)

    for t_idx in ordered_idx.tolist():
        c_t = C[t_idx]
        # Hypothetical balance shifts:
        b_after = torch.stack([
            balance + c_t,     # L:  this track adds left bias
            balance,           # C:  no shift
            balance - c_t,     # R:  adds right bias
        ], dim=0)              # (3, n_bins)

        # cost = -log(prob) + lambda * mean(b²)
        nll = -log_probs[t_idx]                 # (3,)
        balance_pen = (b_after ** 2).mean(dim=-1)  # (3,)
        cost = nll + lambda_balance * balance_pen
        choice = int(cost.argmin().item())

        pan_dir[t_idx] = class_to_pan[choice]
        balance = b_after[choice]

    return pan_dir


__all__ = ["balanced_assignment"]


# ---------- Smoke ----------

def _smoke() -> None:
    """Verify: (1) confident L stays L, (2) ambiguous similar pair splits L/R,
    (3) bass-like high-mag track gets pulled toward center by balance,
    (4) stereo tracks stay at 0."""
    torch.manual_seed(0)
    n_bins = 8

    # Test 1: confident model commitments are honored
    logits = torch.tensor([
        [+5.0, -2.0, -2.0],     # very confident L
        [-2.0, -2.0, +5.0],     # very confident R
        [-2.0, +5.0, -2.0],     # very confident C
    ])
    C = torch.ones(3, n_bins)
    is_stereo = torch.zeros(3, dtype=torch.bool)
    pan = balanced_assignment(logits, C, is_stereo, lambda_balance=0.5)
    assert pan.tolist() == [-1.0, 1.0, 0.0], f"confident picks failed: {pan.tolist()}"
    print(f"confident commits honored: {pan.tolist()}")

    # Test 2: ambiguous similar pair (two guitars with same logits and C)
    # Model says ~equally L or R. Balance should split them.
    logits = torch.tensor([
        [+0.5, -0.5, +0.5],     # roughly equal L and R, low C
        [+0.5, -0.5, +0.5],
    ])
    C = torch.ones(2, n_bins)
    is_stereo = torch.zeros(2, dtype=torch.bool)
    pan = balanced_assignment(logits, C, is_stereo, lambda_balance=1.0)
    # Both shouldn't go the same direction; one should be L (-1) and the other R (+1)
    assert sorted(pan.tolist()) == [-1.0, 1.0], \
        f"pair should split L/R; got {pan.tolist()}"
    print(f"ambiguous pair splits: {pan.tolist()}")

    # Test 3: three similar tracks with low lambda — balance should let
    # at least two of them go L/R rather than piling all on one side.
    # (High lambda would force everything to C — that's a calibration
    # concern; tested at audition time with real C scales.)
    logits = torch.tensor([
        [+0.5, +0.0, +0.5],
        [+0.5, +0.0, +0.5],
        [+0.5, +0.0, +0.5],
    ])
    C = torch.ones(3, n_bins)
    is_stereo = torch.zeros(3, dtype=torch.bool)
    pan = balanced_assignment(logits, C, is_stereo, lambda_balance=0.2)
    print(f"three similar tracks (low λ): {pan.tolist()}")
    # Should NOT have all three on the same non-center side
    n_L = int((pan == -1.0).sum())
    n_R = int((pan == 1.0).sum())
    assert not (n_L == 3 or n_R == 3), \
        f"three similar tracks shouldn't all go same side; got {pan.tolist()}"

    # Test 4: stereo tracks stay 0 regardless of logits
    logits = torch.tensor([[+5.0, -5.0, -5.0]])  # strongly says L
    C = torch.ones(1, n_bins)
    is_stereo = torch.tensor([True])
    pan = balanced_assignment(logits, C, is_stereo, lambda_balance=1.0)
    assert pan.tolist() == [0.0], f"stereo should stay at 0; got {pan.tolist()}"
    print(f"stereo bypass: {pan.tolist()}")

    # Test 5: high-magnitude track pulled toward C even if model is split L/R
    # (the cost of off-center is high because |shift|² is large)
    logits = torch.tensor([
        [+0.0, -0.5, +0.0],   # split between L and R, slight anti-C
    ])
    # Make C have huge magnitude for this one track
    C_high_mag = torch.ones(1, n_bins) * 10.0
    is_stereo = torch.zeros(1, dtype=torch.bool)
    pan = balanced_assignment(logits, C_high_mag, is_stereo, lambda_balance=1.0)
    # With λ=1 and big magnitudes, the |shift|² term dominates → should land at C
    print(f"big-mag track with split logits → {pan.tolist()}  (want 0 due to balance cost)")
    # Soft assertion: with high magnitude the balance term should at least be
    # competitive with the NLL term
    assert pan[0].item() == 0.0, "high-magnitude split-logit track should be pulled to C"

    print("\nall balanced_assignment smoke tests passed.")


if __name__ == "__main__":
    _smoke()
