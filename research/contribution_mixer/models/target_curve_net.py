"""Historical v13 entry point."""
import torch
from nmc.eq.target_curve_net import TargetCurveNet


def _smoke() -> None:
    """Verify (1) shapes, (2) bounded output, (3) gradient flow,
    (4) permutation invariance across tracks, (5) padding mask works."""
    torch.manual_seed(0)
    B, N, n_bins = 4, 12, 26
    net = TargetCurveNet(n_bins=n_bins, d_model=192, n_layers=4, n_heads=4)
    print(f"TargetCurveNet params: {net.n_params():,}")

    C = (torch.randn(B, N, n_bins) * 0.5 + 1.0).requires_grad_(True)
    track_mask = torch.ones(B, N, dtype=torch.bool)
    track_mask[0, -3:] = False  # pad last 3 tracks in batch element 0

    # Test 1: shapes + bounded output + gradient flow (train mode)
    out = net(C, track_mask)
    assert out.shape == (B, n_bins), f"bad output shape {out.shape}"
    assert (out.abs() <= net.max_delta_db + 1e-5).all(), \
        "tanh-bounded output should not exceed max_delta_db"
    print(f"shapes ok. output range: [{out.min().item():.2f}, {out.max().item():.2f}] dB")
    out.sum().backward()
    assert C.grad is not None and torch.isfinite(C.grad).all(), "no/NaN grad into C"
    print("gradients flow.")

    # Switch to eval for determinism (dropout off) across the invariance tests.
    net.eval()
    C_orig = C.detach().clone()

    # Test 2: permutation invariance — shuffle real tracks of batch element 1,
    # output for that batch element should be unchanged
    with torch.no_grad():
        perm = torch.randperm(N)
        C_shuf = C_orig.clone()
        C_shuf[1] = C_orig[1, perm]
        out_orig = net(C_orig)
        out_shuf = net(C_shuf)
        diff = (out_orig[1] - out_shuf[1]).abs().max()
        print(f"permutation-invariance max diff: {diff.item():.2e}  (want < 1e-5)")
        assert diff < 1e-5, "permutation invariance violated"

    # Test 3: padding mask — garbage at masked-out positions shouldn't change
    # the [MIX] token's output for that batch element.
    with torch.no_grad():
        out_clean = net(C_orig, track_mask)
        C_garbage = C_orig.clone()
        C_garbage[0, -3:] = 100.0
        out_garbage = net(C_garbage, track_mask)
        mask_diff = (out_garbage[0] - out_clean[0]).abs().max()
        print(f"padded-token invariance max diff: {mask_diff.item():.2e}  (want < 1e-4)")
        assert mask_diff < 1e-4, "padding mask not respected"

    print("\nall TargetCurveNet smoke tests passed.")


if __name__ == "__main__":
    _smoke()
