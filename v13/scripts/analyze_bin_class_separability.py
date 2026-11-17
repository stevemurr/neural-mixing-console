"""Analysis: do mel-bin contribution vectors carry enough instrument-class info
to skip the per-instrument embedding module in the v13 contribution-mixer
architecture?

Pipeline:
  1. For each staged session, read correspondence.yaml to enumerate stems
     and their group labels.
  2. For each stem, load the WAV and compute a 26-band auditory mel-magnitude
     vector, time-averaged in log domain. This is the same C[bin, track]
     that the v13 architecture uses for delta distribution.
  3. Normalize Cambridge's 158 distinct group names → canonical instrument
     classes (Drums / Bass / Guitar / Vox / Synth / Keys / Strings / Other).
  4. Train logistic regression + small MLP classifiers; report per-class
     and overall accuracy.
  5. Visualize: UMAP and t-SNE projections, colored by class.

Decision rule (printed at the end):
  - C alone ≥ 80% → skip Stage 1 (per-instrument prior) entirely in v13.
  - 60-80%        → marginal; skip for v1, revisit if quality plateaus.
  - < 60%         → keep an embedding-conditioned prior.

Usage:
    uv run python v13/scripts/analyze_bin_class_separability.py \
        --staging-dir dmc-data/grafx-prune-data-48k \
        --out-dir analysis/bin_class_separability
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml


# ---------- Class normalization ----------
#
# Cambridge correspondence.yaml uses 158 distinct group names spanning
# spelled-out forms ("Drum", "BackingVox") and 2-3 letter codes
# ("DR", "BV", "EG"). We map all of them to a small set of canonical
# classes for analysis.
#
# Classes are chosen to be:
#   - musically meaningful (different EQ priors per class)
#   - well-populated in the corpus (avoid 1-2 examples per class)
#
# Anything not matched falls into "Other".

CLASS_RULES: list[tuple[str, tuple[str, ...]]] = [
    # canonical_class, list of group-name prefixes (case-insensitive)
    ("Drums", (
        "dr", "drum", "drums", "drumkit", "kick", "snare", "tom", "hat",
        "hi-hat", "hihat", "cymbal", "overhead", "perc", "percussion",
        "congas", "cajon", "loop", "fill", "groove",
    )),
    ("Bass", (
        "ba", "bass", "basssynth", "subbass", "sub",
    )),
    ("Guitar", (
        "eg", "ag", "gtr", "guitar", "elecgtr", "acousticgtr",
        "electricgtr", "ebow",
    )),
    ("LeadVox", (
        "lv", "leadvox", "lead vox", "vox", "vocal", "vocals", "main",
        "lead", "narration",
    )),
    ("BackingVox", (
        "bv", "backingvox", "backing vox", "backupvox", "choir", "harm",
        "harmony", "ah", "ohs",
    )),
    ("Synth", (
        "syn", "synth", "pad", "lead synth", "leadsynth", "arp", "fx",
    )),
    ("Keys", (
        "pn", "piano", "rhodes", "hammond", "organ", "keys", "key",
        "wurli", "epiano",
    )),
    ("Strings", (
        "str", "strings", "violin", "viola", "cello", "pizz", "bow",
    )),
    ("Brass", (
        "brass", "horn", "trumpet", "trombone", "sax", "saxophone",
    )),
]


def classify_group_name(name: str) -> str:
    """Map a raw correspondence.yaml group name to a canonical class."""
    n = name.lower().strip()
    for cls, prefixes in CLASS_RULES:
        for p in prefixes:
            if n == p or n.startswith(p):
                return cls
    return "Other"


# ---------- Mel-bin feature extraction ----------

def build_mel_transform(
    sample_rate: int = 48_000,
    n_mels: int = 26,
    n_fft: int = 8192,
    hop_length: int = 2048,
    f_min: float = 20.0,
    f_max: float = 20_000.0,
) -> torchaudio.transforms.MelSpectrogram:
    """26-band auditory filterbank at 48 kHz — matches the v13 EQ resolution."""
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min, f_max=f_max,
        power=1.0,
        norm=None,
        mel_scale="htk",
    )


def mel_vector(
    wav_path: Path,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    max_seconds: float = 300.0,   # 5 min — capped only to bound memory
    sample_rate: int = 48_000,
    active_pct: float = 0.20,     # use top-20% loudest frames
) -> np.ndarray | None:
    """Return (n_mels,) log-magnitude vector aggregated over active frames.

    Cambridge stems are full-length tracks; many have long silent intros,
    outros, or sparse parts (a vocal that only sings in the chorus). A
    mean over time would be dominated by silence. Instead:
      1. Compute mel-spectrogram over the (capped) full track.
      2. Select frames whose RMS magnitude is in the top `active_pct`.
      3. Time-average log-mel over just those frames.

    This captures the spectrum during the parts of the track where the
    instrument is actually playing — independent of how dense the stem is.
    """
    try:
        max_frames = int(max_seconds * sample_rate)
        audio, sr = sf.read(str(wav_path), dtype="float32",
                            always_2d=True, frames=max_frames)
    except Exception as e:
        logging.warning(f"failed to read {wav_path.name}: {e}")
        return None
    if audio.shape[0] < 4096:
        return None
    if sr != sample_rate:
        logging.warning(f"unexpected sr={sr} in {wav_path.name}")
        return None
    mono = audio.mean(axis=-1).astype(np.float32)
    if np.abs(mono).max() < 1e-6:
        return None
    x = torch.from_numpy(mono).unsqueeze(0)
    with torch.no_grad():
        mel = mel_transform(x).squeeze(0)               # (n_mels, n_frames)
    if mel.shape[-1] < 4:
        return None
    # Per-frame total energy (sum over bins). Pick the top `active_pct`.
    frame_energy = mel.sum(dim=0)                       # (n_frames,)
    k = max(1, int(frame_energy.numel() * active_pct))
    _, top_idx = torch.topk(frame_energy, k, largest=True)
    active_mel = mel[:, top_idx]                        # (n_mels, k)
    feat = torch.log1p(active_mel.mean(dim=-1)).numpy()
    return feat


# ---------- Dataset assembly ----------

def collect_stem_features(
    staging_dir: Path, mel_transform: torchaudio.transforms.MelSpectrogram,
) -> tuple[np.ndarray, np.ndarray, list[str], list[Counter]]:
    """Return (X, y, sessions, raw_group_counts_per_class).

    X: (n_stems, n_mels)
    y: (n_stems,) canonical-class string labels
    sessions: parallel list of session names
    raw_group_counts_per_class: list of Counter mapping
        raw_group_name -> count per canonical class (for inspection).
    """
    sessions_iter = sorted(staging_dir.iterdir())
    feats: list[np.ndarray] = []
    labels: list[str] = []
    sess_per_stem: list[str] = []
    raw_counts: dict[str, Counter] = {cls: Counter() for cls, _ in CLASS_RULES}
    raw_counts["Other"] = Counter()
    skipped = Counter()

    for session_dir in sessions_iter:
        if not session_dir.is_dir():
            continue
        corr_path = session_dir / "correspondence.yaml"
        stems_dir = session_dir / "stems"
        if not corr_path.is_file() or not stems_dir.is_dir():
            continue
        try:
            corr = yaml.safe_load(open(corr_path))
        except Exception as e:
            logging.warning(f"failed to parse {corr_path}: {e}")
            continue
        if not isinstance(corr, dict):
            skipped["bad-yaml-shape"] += 1
            continue

        for group_name, stem_files in corr.items():
            cls = classify_group_name(group_name)
            raw_counts[cls][group_name] += 1
            if not isinstance(stem_files, list):
                continue
            for fname in stem_files:
                wav = stems_dir / fname
                if not wav.is_file():
                    skipped["missing-wav"] += 1
                    continue
                feat = mel_vector(wav, mel_transform)
                if feat is None:
                    skipped["bad-feature"] += 1
                    continue
                feats.append(feat)
                labels.append(cls)
                sess_per_stem.append(session_dir.name)

    X = np.stack(feats, axis=0)
    y = np.asarray(labels)
    logging.info(f"collected {len(y)} stem features; skipped: {dict(skipped)}")
    return X, y, sess_per_stem, raw_counts


# ---------- Classifiers ----------

def train_eval_classifiers(
    X: np.ndarray, y: np.ndarray, sessions: list[str], seed: int = 0,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    classes = sorted(set(y))

    # Session-grouped split — never let stems from the same session leak.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=sessions))
    X_tr, X_te, y_tr, y_te = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    logging.info(f"split: train={len(y_tr)}  test={len(y_te)}  "
                 f"(session-grouped; no leakage)")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    out: dict = {"classes": classes, "n_train": len(y_tr), "n_test": len(y_te)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        lr = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed,
        ).fit(X_tr_s, y_tr)
        lr_pred = lr.predict(X_te_s)
        lr_acc = accuracy_score(y_te, lr_pred)
        out["logreg"] = {
            "accuracy":  float(lr_acc),
            "report":    classification_report(y_te, lr_pred, zero_division=0, output_dict=True),
        }
        logging.info(f"LogisticRegression accuracy: {lr_acc:.3f}")

        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500,
            random_state=seed, early_stopping=True,
        ).fit(X_tr_s, y_tr)
        mlp_pred = mlp.predict(X_te_s)
        mlp_acc = accuracy_score(y_te, mlp_pred)
        out["mlp"] = {
            "accuracy":  float(mlp_acc),
            "report":    classification_report(y_te, mlp_pred, zero_division=0, output_dict=True),
        }
        logging.info(f"MLP accuracy:                {mlp_acc:.3f}")

    return out


# ---------- Visualization ----------

def plot_projection(
    X: np.ndarray, y: np.ndarray, method: str, out_path: Path,
    title: str, seed: int = 0,
) -> None:
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    if method == "umap":
        try:
            import umap
            proj = umap.UMAP(
                n_components=2, n_neighbors=30, min_dist=0.1, random_state=seed,
            ).fit_transform(Xs)
        except ImportError:
            logging.warning("umap-learn not installed; falling back to t-SNE")
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        proj = TSNE(
            n_components=2, perplexity=30, random_state=seed, init="pca",
            learning_rate="auto",
        ).fit_transform(Xs)

    classes = sorted(set(y))
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(classes):
        idx = y == cls
        plt.scatter(
            proj[idx, 0], proj[idx, 1],
            c=[cmap(i)], label=f"{cls} (n={idx.sum()})",
            s=8, alpha=0.6, edgecolors="none",
        )
    plt.legend(loc="best", fontsize=9, framealpha=0.8)
    plt.title(title)
    plt.xlabel(f"{method} dim 1")
    plt.ylabel(f"{method} dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    logging.info(f"saved {out_path}")


# ---------- Main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--staging-dir", default="dmc-data/grafx-prune-data-48k")
    ap.add_argument("--out-dir", default="analysis/bin_class_separability")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-mels", type=int, default=26)
    ap.add_argument("--max-stem-seconds", type=float, default=30.0,
                    help="Read at most this many seconds per stem (the "
                         "time-averaged spectrum is robust to truncation).")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(args.staging_dir)
    if not staging.is_dir():
        ap.error(f"staging dir not found: {staging}")

    # 1. Feature extraction
    logging.info(f"extracting {args.n_mels}-mel features from {staging}")
    mel = build_mel_transform(n_mels=args.n_mels)
    X, y, sessions, raw_counts = collect_stem_features(staging, mel)

    np.save(out_dir / "features.npy", X)
    np.save(out_dir / "labels.npy", y)
    (out_dir / "sessions.json").write_text(json.dumps(sessions))

    # 2. Class distribution
    class_counts = dict(Counter(y))
    logging.info("class distribution:")
    for cls, n in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        logging.info(f"  {cls:12s}  {n:5d}")
    (out_dir / "class_counts.json").write_text(
        json.dumps(class_counts, indent=2),
    )
    # Per-class breakdown of raw group names that landed there.
    (out_dir / "raw_groups_per_class.json").write_text(
        json.dumps(
            {cls: dict(c.most_common()) for cls, c in raw_counts.items()},
            indent=2,
        ),
    )

    # 3. Classifiers
    results = train_eval_classifiers(X, y, sessions, seed=args.seed)
    results["class_counts"] = class_counts
    results["n_mels"] = args.n_mels
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # 4. Visualizations
    plot_projection(
        X, y, "umap", out_dir / "umap.png",
        title=f"{args.n_mels}-mel-bin contribution vectors — UMAP",
        seed=args.seed,
    )
    plot_projection(
        X, y, "tsne", out_dir / "tsne.png",
        title=f"{args.n_mels}-mel-bin contribution vectors — t-SNE",
        seed=args.seed,
    )

    # 5. Decision
    print("\n" + "=" * 70)
    print("SEPARABILITY ANALYSIS RESULT")
    print("=" * 70)
    print(f"Stems analyzed:   {len(y)}")
    print(f"Sessions:         {len(set(sessions))}")
    print(f"Classes:          {len(set(y))}")
    print(f"Mel bins:         {args.n_mels}")
    print()
    print(f"LogReg accuracy:  {results['logreg']['accuracy']:.3f}")
    print(f"MLP  accuracy:    {results['mlp']['accuracy']:.3f}")
    print()
    best = max(results["logreg"]["accuracy"], results["mlp"]["accuracy"])
    if best >= 0.80:
        decision = (
            "SKIP per-instrument embedding for EQ stage. The bin "
            "contribution matrix carries enough instrument-class signal "
            "on its own; closed-form delta distribution is sufficient."
        )
    elif best >= 0.60:
        decision = (
            "MARGINAL. Skip embedding module for v13 v1; revisit if "
            "audition quality plateaus on instrument-conditioned cases."
        )
    else:
        decision = (
            "KEEP embedding module. Instrument identity is not well-captured "
            "by static mel-bin contribution; add MERT/CLAP per-stem prior."
        )
    print("DECISION:")
    print(decision)
    print("=" * 70 + "\n")

    logging.info(f"all outputs written to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
