#!/usr/bin/env python
"""Evaluate a MixEncoder checkpoint, or aggregate inference parameter dumps.

Two modes:

  eval.py recon  --checkpoint CKPT --shard-dirs DIR [DIR ...] [--split val|test]
                 [--n-batches N] [--loudness-target-dbfs DB] [--out report.json]
      Run the checkpoint on a held-out split. Reports reconstruction losses,
      the loudness match (rendered output level vs target), per-parameter
      distributions (incl. an at-range-edge % — the corner-pinning detector),
      effective-bypass rates, the pan distribution, and a health summary.

  eval.py params --params-glob 'dmc-data/inference/*/params.json' [--out report.json]
      Aggregate already-rendered params.json files (no model needed). Same
      per-parameter / effective-bypass / pan / trim report (minus the recon
      losses, which require running the model).

In both modes a JSON report can be written with --out; a formatted summary is
always printed.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference.param_norm import (
    PARAM_RANGES, normalize, bus_effective_bypass, strip_effective_bypass,
)
from training.data import STRIP_PARAM_KEYS, BUS_PARAM_KEYS


# ---------- per-parameter distribution ----------

def _dist(values: list[float], name: str, edge_frac: float = 0.05) -> dict:
    """Distribution summary for one parameter's values across the eval set.

    Includes `at_min_pct` / `at_max_pct` (% of values within `edge_frac` of the
    normalized [0,1] range edges — the corner-pinning detector) and `norm_std`
    (stdev in normalized space — ~0 means the encoder predicts a constant)."""
    n = len(values)
    if n == 0:
        return {"n": 0}
    vs = sorted(values)
    def q(p: float) -> float:
        return vs[min(n - 1, max(0, int(round(p * (n - 1)))))]
    out = {
        "n": n,
        "mean": st.mean(values), "std": st.pstdev(values),
        "p5": q(0.05), "p50": q(0.50), "p95": q(0.95),
        "min": vs[0], "max": vs[-1],
    }
    if name in PARAM_RANGES:
        norms = [normalize(name, v) for v in values]
        out["norm_mean"] = st.mean(norms)
        out["norm_std"] = st.pstdev(norms)
        out["at_min_pct"] = 100.0 * sum(1 for x in norms if x <= edge_frac) / n
        out["at_max_pct"] = 100.0 * sum(1 for x in norms if x >= 1.0 - edge_frac) / n
    return out


def _param_report(strip_dicts: list[dict], bus_dicts: list[dict],
                  trim_vals: list[float] | None) -> dict:
    """Build the param/effective-bypass/pan/trim portion of an eval report from
    a list of per-track strip dicts and per-mix bus dicts (physical units)."""
    strip_stats = {
        k: _dist([d[k] for d in strip_dicts if k in d], k)
        for k in STRIP_PARAM_KEYS
    }
    bus_stats = {
        k: _dist([d.get(k, d.get(k[len("bus_"):])) for d in bus_dicts
                  if (k in d or k[len("bus_"):] in d)], k)
        for k in BUS_PARAM_KEYS
    }
    # Effective-bypass rates: fraction of tracks / mixes where each processor
    # is effectively at identity.
    n_tr = max(1, len(strip_dicts))
    eb = {p: 0 for p in ("eq_hpf", "eq_ls", "eq_p1", "eq_p2", "eq_hs", "eq_lpf", "comp", "clip")}
    for d in strip_dicts:
        for p, v in strip_effective_bypass(d).items():
            eb[p] += int(v)
    strip_eb_pct = {p: 100.0 * c / n_tr for p, c in eb.items()}
    n_bus = max(1, len(bus_dicts))
    beb = {p: 0 for p in ("bus_eq_low_boost", "bus_eq_low_attn", "bus_eq_mid", "bus_eq_air", "bus_comp")}
    for d in bus_dicts:
        for p, v in bus_effective_bypass(d).items():
            beb[p] += int(v)
    bus_eb_pct = {p: 100.0 * c / n_bus for p, c in beb.items()}

    pan_vals = [d["pan"] for d in strip_dicts if "pan" in d]
    pan_stat = ({"n": len(pan_vals), "mean": st.mean(pan_vals), "std": st.pstdev(pan_vals),
                 "min": min(pan_vals), "max": max(pan_vals)} if pan_vals else {"n": 0})

    report = {
        "n_tracks": len(strip_dicts), "n_mixes": len(bus_dicts),
        "strip_params": strip_stats, "bus_params": bus_stats,
        "strip_effective_bypass_pct": strip_eb_pct, "bus_effective_bypass_pct": bus_eb_pct,
        "pan": pan_stat,
    }
    if trim_vals:
        report["trim_db"] = {"n": len(trim_vals), "mean": st.mean(trim_vals),
                             "std": st.pstdev(trim_vals), "min": min(trim_vals), "max": max(trim_vals)}
    return report


# ---------- health flags ----------

# Params whose identity / "effectively off" state sits at a range edge by
# design — a band the model declines to use lands there, which is a decision
# (already reported in effective_bypass_pct), not a pathology. So a high
# at-edge % on these is NOT flagged as corner-pinning.
_BENIGN_EDGE_PARAMS = {
    # one-directional EQ bands (gain identity at an edge)
    "bus_low_boost_gain", "bus_low_attn_gain", "bus_air_gain",
    # ...and the freq/Q of those bands (irrelevant when the gain is at identity)
    "bus_low_boost_freq", "bus_low_attn_freq", "bus_air_freq",
    # clipper / comp "off" states
    "clip_drive_db", "clip_mix", "ratio", "bus_ratio",
    # filters whose "open" state is a range edge
    "hpf_freq", "lpf_freq",
}


def _health_flags(report: dict, *, trim_max_db: float | None = None,
                  loud_residual_db: float | None = None,
                  edge_pct_flag: float = 50.0) -> list[str]:
    flags: list[str] = []
    for layer, stats in (("strip", report.get("strip_params", {})),
                         ("bus", report.get("bus_params", {}))):
        for k, s in stats.items():
            if not isinstance(s, dict) or s.get("n", 0) == 0:
                continue
            if k in _BENIGN_EDGE_PARAMS:
                continue
            am, ax = s.get("at_min_pct"), s.get("at_max_pct")
            if am is not None and (am >= edge_pct_flag or ax >= edge_pct_flag):
                end = "min" if (am or 0) >= (ax or 0) else "max"
                flags.append(f"CORNER-PINNED  {layer}.{k}: {max(am or 0, ax or 0):.0f}% at range {end}")
    pan = report.get("pan", {})
    if pan.get("n", 0) and abs(pan.get("mean", 0.0)) > 0.1:
        flags.append(f"PAN BIAS       mean pan = {pan['mean']:+.2f} (systematic L/R lean)")
    trim = report.get("trim_db", {})
    if trim.get("n", 0) and trim_max_db and abs(trim.get("mean", 0.0)) > 0.8 * trim_max_db:
        flags.append(f"TRIM NEAR CAP  mean trim = {trim['mean']:+.1f} dB (cap ±{trim_max_db:g})")
    if loud_residual_db is not None and abs(loud_residual_db) > 2.0:
        flags.append(f"LOUDNESS OFF-TARGET  rendered ≈ {loud_residual_db:+.1f} dB vs target")
    return flags


# ---------- pretty-print ----------

def _print_report(report: dict, *, losses: dict | None = None,
                  loudness: dict | None = None, health: list[str] | None = None) -> None:
    if losses:
        print("== reconstruction losses (mean ± std over the eval set) ==")
        for k in ("L_recon", "L_timbre", "L_time", "L_mss", "L_log_mel", "L_stereo_side", "L_loud"):
            if k in losses:
                m, s = losses[k]
                print(f"  {k:16s} {m:8.4f} ± {s:.4f}")
        print()
    if loudness:
        print("== loudness ==")
        print(f"  rendered output (post-trim) RMS: {loudness['rendered_rms_dbfs_mean']:+.2f} "
              f"± {loudness['rendered_rms_dbfs_std']:.2f} dBFS")
        if "target_dbfs" in loudness:
            print(f"  target: {loudness['target_dbfs']:+.2f} dBFS  "
                  f"(residual ≈ {loudness['rendered_rms_dbfs_mean'] - loudness['target_dbfs']:+.2f} dB)")
        print()
    print(f"== per-parameter distributions  (n_tracks={report['n_tracks']}, n_mixes={report['n_mixes']}) ==")
    hdr = f"  {'param':22s} {'mean':>9s} {'std':>8s} {'p5':>9s} {'p50':>9s} {'p95':>9s} {'@min%':>6s} {'@max%':>6s} {'nstd':>6s}"
    for layer in ("strip_params", "bus_params"):
        print(f"  --- {layer.replace('_params','')} ---")
        print(hdr)
        for k, s in report[layer].items():
            if not isinstance(s, dict) or s.get("n", 0) == 0:
                print(f"  {k:22s}  (no data)")
                continue
            am = f"{s['at_min_pct']:6.1f}" if s.get("at_min_pct") is not None else "     -"
            ax = f"{s['at_max_pct']:6.1f}" if s.get("at_max_pct") is not None else "     -"
            ns = f"{s['norm_std']:6.3f}" if s.get("norm_std") is not None else "     -"
            print(f"  {k:22s} {s['mean']:9.3f} {s['std']:8.3f} {s['p5']:9.3f} {s['p50']:9.3f} {s['p95']:9.3f} {am} {ax} {ns}")
    print()
    print("== effective-bypass rates (% of tracks/mixes with the processor at identity) ==")
    for p, v in report["strip_effective_bypass_pct"].items():
        print(f"  strip.{p:10s} {v:6.1f}%")
    for p, v in report["bus_effective_bypass_pct"].items():
        print(f"  {p:16s} {v:6.1f}%")
    print()
    pan = report.get("pan", {})
    if pan.get("n", 0):
        print(f"== pan ==  mean={pan['mean']:+.3f}  std={pan['std']:.3f}  range=[{pan['min']:+.2f}, {pan['max']:+.2f}]  (0=center, ±1=hard L/R)")
    trim = report.get("trim_db", {})
    if trim.get("n", 0):
        print(f"== trim_db ==  mean={trim['mean']:+.2f}  std={trim['std']:.2f}  range=[{trim['min']:+.2f}, {trim['max']:+.2f}] dB")
    print()
    print("== health ==")
    if health:
        for f in health:
            print(f"  ⚠ {f}")
    else:
        print("  (no flags)")


# ---------- mode: aggregate params.json ----------

def _mode_params(args: argparse.Namespace) -> int:
    files = sorted(glob.glob(args.params_glob))
    if not files:
        print(f"no params.json files matched: {args.params_glob}", file=sys.stderr)
        return 1
    strip_dicts, bus_dicts, trim_vals = [], [], []
    for fp in files:
        d = json.loads(Path(fp).read_text())
        for tr in d.get("tracks", []):
            if "strip" in tr:
                strip_dicts.append(tr["strip"])
        if "bus_params" in d:
            bus_dicts.append(d["bus_params"])
        if "trim_db" in d:
            trim_vals.append(float(d["trim_db"]))
    print(f"aggregated {len(files)} params.json files: {len(strip_dicts)} tracks, {len(bus_dicts)} mixes\n")
    report = _param_report(strip_dicts, bus_dicts, trim_vals)
    health = _health_flags(report)
    _print_report(report, health=health)
    if args.out:
        Path(args.out).write_text(json.dumps({"report": report, "health": health,
                                              "source_files": files}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


# ---------- mode: run model on a held-out split ----------

def _mode_recon(args: argparse.Namespace) -> int:
    import torch
    from torch.utils.data import DataLoader
    from models.diff_mixing_graph import DiffMixingGraph
    from models.encoders import MixEncoder
    from training.data import make_stage3_dataset, collate_stage3
    from training.losses import decoupled_recon_loss
    from training.train_stage3 import _build_denorm_table, _denorm, _render_full_mix, MERT_DIM, SAMPLE_RATE

    device = torch.device(args.device)
    enc = MixEncoder(sample_rate=SAMPLE_RATE, use_ref_mix=False, mert_dim=MERT_DIM,
                     trim_max_db=args.trim_max_db).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "encoder_state_dict" in ckpt:
        ckpt = ckpt["encoder_state_dict"]
    missing, unexpected = enc.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)} tensors (non-strict load)")

    graph = DiffMixingGraph(sample_rate=SAMPLE_RATE).to(device).eval()
    for p in graph.parameters():
        p.requires_grad_(False)
    tables = {"strip": _build_denorm_table(device, STRIP_PARAM_KEYS),
              "bus": _build_denorm_table(device, BUS_PARAM_KEYS)}

    ds = make_stage3_dataset(args.shard_dirs, shuffle=32, split=args.split,
                             max_tracks=args.n_max, repeat=False,
                             mert_cache_root=args.mert_cache_root)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=lambda b: collate_stage3(b, n_max=args.n_max, mert_dim=MERT_DIM),
                        drop_last=True)

    loss_keys = ["L_recon", "L_timbre", "L_time", "L_mss", "L_log_mel", "L_stereo_side", "L_loud"]
    loss_acc: dict[str, list[float]] = {k: [] for k in loss_keys}
    rendered_rms_dbfs: list[float] = []
    strip_dicts, bus_dicts, trim_vals = [], [], []

    rms_relative = not args.no_rms_relative_threshold
    nb = 0
    with torch.no_grad():
        for batch in loader:
            if args.n_batches and nb >= args.n_batches:
                break
            tracks = batch["tracks"].to(device, non_blocking=True)
            track_mask = batch["track_mask"].to(device, non_blocking=True)
            ref_mix = batch["mix"].to(device, non_blocking=True)
            mert = batch["mert_embeddings"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = enc(tracks, track_mask, ref_mix=None, mert_embeddings=mert)
            with torch.amp.autocast("cuda", enabled=False):
                pred_mix = _render_full_mix(graph, tracks, track_mask, out, tables,
                                            use_rms_relative_threshold=rms_relative)
                rec = decoupled_recon_loss(pred_mix, ref_mix.float(), out["trim_db"].float(),
                                           w_log_mel=args.w_log_mel, w_loud=args.w_loud,
                                           w_stereo_side=args.w_stereo_side,
                                           loudness_target_dbfs=args.loudness_target_dbfs)
            for k in loss_keys:
                if k in rec:
                    loss_acc[k].append(float(rec[k]))
            # rendered (post-trim) output level
            trimmed = pred_mix * torch.pow(10.0, out["trim_db"].float() / 20.0).view(-1, 1, 1)
            rr = torch.sqrt((trimmed ** 2).mean(dim=(-2, -1)) + 1e-12)
            rendered_rms_dbfs += (20.0 * torch.log10(rr + 1e-12)).cpu().tolist()
            # param collection (active tracks only)
            strip_phys = _denorm(out["track_params"].float(), tables["strip"]).cpu()
            bus_phys = _denorm(out["bus_params"].float(), tables["bus"]).cpu()
            tm = track_mask.cpu()
            B, N = tm.shape
            for b in range(B):
                for kk in range(N):
                    if not tm[b, kk]:
                        continue
                    strip_dicts.append({key: float(strip_phys[b, kk, i]) for i, key in enumerate(STRIP_PARAM_KEYS)})
                bus_dicts.append({key: float(bus_phys[b, i]) for i, key in enumerate(BUS_PARAM_KEYS)})
                trim_vals.append(float(out["trim_db"][b]))
            nb += 1
    if nb == 0:
        print(f"no batches for split={args.split!r} in {args.shard_dirs}", file=sys.stderr)
        return 1

    losses = {k: (st.mean(v), st.pstdev(v)) for k, v in loss_acc.items() if v}
    loudness = {"rendered_rms_dbfs_mean": st.mean(rendered_rms_dbfs),
                "rendered_rms_dbfs_std": st.pstdev(rendered_rms_dbfs)}
    loud_residual = None
    if args.loudness_target_dbfs is not None:
        loudness["target_dbfs"] = args.loudness_target_dbfs
        loud_residual = loudness["rendered_rms_dbfs_mean"] - args.loudness_target_dbfs
    report = _param_report(strip_dicts, bus_dicts, trim_vals)
    health = _health_flags(report, trim_max_db=args.trim_max_db, loud_residual_db=loud_residual)

    print(f"eval: checkpoint={args.checkpoint}  split={args.split}  batches={nb}\n")
    _print_report(report, losses=losses, loudness=loudness, health=health)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "checkpoint": args.checkpoint, "split": args.split, "n_batches": nb,
            "losses": {k: {"mean": m, "std": s} for k, (m, s) in losses.items()},
            "loudness": loudness, "report": report, "health": health,
        }, indent=2))
        print(f"\nwrote {args.out}")
    return 0


# ---------- CLI ----------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    pr = sub.add_parser("recon", help="run a checkpoint on a held-out split and report")
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--shard-dirs", nargs="+", required=True)
    pr.add_argument("--split", default="val", choices=["val", "test", "train"])
    pr.add_argument("--n-batches", type=int, default=50, help="0 = whole split")
    pr.add_argument("--batch-size", type=int, default=4)
    pr.add_argument("--n-max", type=int, default=42)
    pr.add_argument("--num-workers", type=int, default=2)
    pr.add_argument("--mert-cache-root", default="dmc-data/mert_cache")
    pr.add_argument("--device", default="cuda")
    pr.add_argument("--trim-max-db", type=float, default=18.0)
    pr.add_argument("--loudness-target-dbfs", type=float, default=None)
    pr.add_argument("--w-log-mel", type=float, default=0.5)
    pr.add_argument("--w-loud", type=float, default=0.3)
    pr.add_argument("--w-stereo-side", type=float, default=2.5)
    pr.add_argument("--no-rms-relative-threshold", action="store_true")
    pr.add_argument("--out", default="", help="optional path to write the report JSON")

    pp = sub.add_parser("params", help="aggregate already-rendered params.json files")
    pp.add_argument("--params-glob", required=True, help="glob, e.g. 'dmc-data/inference/*/params.json'")
    pp.add_argument("--out", default="", help="optional path to write the report JSON")

    args = ap.parse_args()
    if args.mode == "recon":
        return _mode_recon(args)
    if args.mode == "params":
        return _mode_params(args)
    ap.error(f"unknown mode {args.mode!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
