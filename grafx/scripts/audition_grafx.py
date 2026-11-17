"""Audition a grafx-distill encoder against a few val sessions.

For each session, renders four wavs side by side:
    rendered_student.wav — student's mix from predicted params
    teacher_render.wav  — teacher's labels rendered through our console
    sum_baseline.wav    — naive mask-aware sum of stems (no processing)
    ref_mix.wav         — engineer mix (already aligned via _load_session)

Listen against ref_mix to diagnose what the student is getting wrong:
  - "blanker / less colored than ref" → AF loss ceiling / loss-design issue
  - "wrong instrument too loud" → encoder ID confusion (MERT might help)
  - "right but different style choice" → engineer-style irreducible noise

Usage:
    uv run python scripts/audition_grafx.py \\
        --ckpt dmc-data/checkpoints/grafx-distill-v9/encoder_best.pt \\
        --out-dir dmc-data/audition/v9_best \\
        [--sessions a b c] [--n 3]
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.grafx_console import GrafxMixingConsole
from models.encoders_grafx import MixEncoderGrafx
from training.data_grafx import LabelStore, attach_grafx_labels
from training.losses_grafx import (
    AudioFeatureLoss, make_grafx_prune_recon_loss,
)
from training.train_grafx_distill import (
    PairedMultiSongDataset, collate_pairs, _to_device_params,
    _encoder_forward,
)
from training.train_grafx_multi import (
    _load_session, _example_from_session, collate_multi,
)


def _save_wav(path: Path, audio: torch.Tensor, sr: int) -> None:
    """audio: (2, T) or (1, 2, T) on any device → 24-bit WAV on disk."""
    a = audio.detach().cpu().numpy()
    if a.ndim == 3:
        a = a[0]
    # soundfile wants (T, C); also normalize to ≤ 1.0 to avoid clipping on save
    a = a.T.astype(np.float32, copy=False)
    peak = float(np.abs(a).max()) if a.size else 0.0
    if peak > 0.99:
        a = a / peak * 0.99
    sf.write(str(path), a, sr, subtype="PCM_24")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, help="encoder_best.pt path")
    ap.add_argument("--staging-dir", default="dmc-data/grafx-prune-data-48k")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="Specific session names to render; default = random "
                         "val sessions.")
    ap.add_argument("--n", type=int, default=3,
                    help="Number of val sessions to render (if --sessions not set)")
    ap.add_argument("--audio-len", type=int, default=480_000,
                    help="Samples per render (10 s @ 48 kHz). Longer = more "
                         "context to evaluate. Bounded by max_input_len in "
                         "the console init below.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    # ---- Load ckpt + reconstruct encoder ----
    ckpt = torch.load(args.ckpt, weights_only=False, map_location=device)
    ck_args = ckpt.get("args", {})
    sr = ck_args.get("sample_rate", 48_000)
    n_max = ck_args.get("n_max", 24)
    max_groups = ck_args.get("max_groups", 16)
    d_model = ck_args.get("d_model", 384)
    n_track_layers = ck_args.get("n_track_layers", 4)
    use_bf16 = ck_args.get("use_bf16", False)
    use_af_loss = ck_args.get("use_af_loss", False)
    print(f"ckpt step={ckpt.get('step', '?')}  d_model={d_model}  "
          f"n_layers={n_track_layers}  use_bf16={use_bf16}  use_af={use_af_loss}")

    console = GrafxMixingConsole(
        sample_rate=sr, max_input_len=args.audio_len,
    ).to(device)
    encoder = MixEncoderGrafx(
        console, sample_rate=sr, d_model=d_model,
        n_track_layers=n_track_layers, use_ref_mix=False, mert_dim=0,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    print(f"encoder params: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M")

    # ---- Pick sessions ----
    staging = Path(args.staging_dir).expanduser()
    label_store = LabelStore(staging)
    all_sessions = sorted([d.name for d in staging.iterdir()
                           if d.is_dir() and (d/'correspondence.yaml').is_file()
                           and (d/'mix.wav').is_file() and (d/'stems').is_dir()])
    rng = random.Random(args.seed)
    shuffled = sorted(all_sessions); rng.shuffle(shuffled)
    val_sessions = sorted(shuffled[:max(1, int(round(0.1 * len(all_sessions))))])
    if args.sessions:
        sessions = args.sessions
    else:
        # Pick `n` val sessions with labels for which final_test_loss is finite.
        rng2 = random.Random(args.seed + 1)
        candidates = [s for s in val_sessions
                      if (label_store.get(s) or {}).get("training_meta", {})
                                                   .get("final_test_loss") is not None]
        sessions = rng2.sample(candidates, min(args.n, len(candidates)))
    print(f"rendering {len(sessions)} session(s): {sessions}")

    # ---- Losses for diagnostic numbers ----
    mrstft = make_grafx_prune_recon_loss(sample_rate=sr).to(device)
    af = AudioFeatureLoss(sample_rate=sr).to(device) if use_af_loss else None

    out_root = Path(args.out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []

    with torch.no_grad():
        for sess in sessions:
            out = out_root / sess
            out.mkdir(exist_ok=True)
            try:
                data = _load_session(sess, staging, label_store)
            except Exception as e:
                print(f"  {sess}: load failed: {e}")
                continue
            T = data["T_total"]
            if T < args.audio_len + 1:
                print(f"  {sess}: too short ({T})")
                continue
            n_stems = len(data["stem_filenames"])
            start = max(0, T // 3)
            ex = _example_from_session(data, start, args.audio_len, n_max=n_stems)
            batch = collate_multi([ex])
            labels = label_store.get(sess)
            n_groups = len(labels["groups"]) if labels else max_groups
            batch = attach_grafx_labels(
                batch, label_store, n_max=n_stems, max_groups=n_groups,
                strip_schema=console.strip_param_shapes,
                group_schema=console.group_param_shapes,
            )
            tracks = batch["tracks"].to(device)
            track_mask = batch["track_mask"].to(device)
            group_idx = batch["group_assignments"].to(device)
            mix_ref = batch["mix"].to(device)

            # --- Student render ---
            enc_out = _encoder_forward(
                encoder, tracks, track_mask, group_idx,
                n_groups=n_groups, use_bf16=use_bf16,
            )
            student_mix = console(
                tracks, enc_out["strip_params"], enc_out["group_params"],
                group_idx, track_mask=track_mask, n_groups=n_groups,
            )

            # --- Teacher render ---
            label_strip = _to_device_params(batch["label_strip_params"], device)
            label_group = _to_device_params(batch["label_group_params"], device)
            teacher_mix = console(
                tracks, label_strip, label_group, group_idx,
                track_mask=track_mask, n_groups=n_groups,
            )

            # --- Sum baseline (mask-aware) ---
            sum_baseline = (tracks * track_mask.to(tracks.dtype)
                            .view(1, -1, 1, 1)).sum(dim=1)

            # --- Diagnostic numbers ---
            tloss_recorded = (labels.get("training_meta") or {}).get("final_test_loss") if labels else None
            scores = {
                "session": sess,
                "stem_count": n_stems,
                "teacher_recorded_test_loss": tloss_recorded,
                "student_vs_engineer_mrstft":     mrstft(student_mix, mix_ref)["match/full"].item(),
                "teacher_vs_engineer_mrstft":     mrstft(teacher_mix, mix_ref)["match/full"].item(),
                "sum_vs_engineer_mrstft":         mrstft(sum_baseline, mix_ref)["match/full"].item(),
                "student_vs_teacher_mrstft":      mrstft(student_mix, teacher_mix)["match/full"].item(),
            }
            if af is not None:
                scores.update({
                    "student_vs_engineer_af": af(student_mix, mix_ref)["af/total"].item(),
                    "student_vs_teacher_af":  af(student_mix, teacher_mix)["af/total"].item(),
                })
            scores["amort_gap_mrstft"] = (scores["student_vs_engineer_mrstft"]
                                          - float(tloss_recorded)) if tloss_recorded else None

            print(f"\n{sess}:")
            for k, v in scores.items():
                if isinstance(v, float):
                    print(f"  {k:34s} {v:8.3f}")
                else:
                    print(f"  {k:34s} {v}")

            # --- Save wavs ---
            _save_wav(out / "rendered_student.wav", student_mix, sr)
            _save_wav(out / "teacher_render.wav",   teacher_mix, sr)
            _save_wav(out / "sum_baseline.wav",     sum_baseline, sr)
            _save_wav(out / "ref_mix.wav",          mix_ref, sr)

            summary.append(scores)

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[ok] wrote {len(summary)} session(s) to {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
