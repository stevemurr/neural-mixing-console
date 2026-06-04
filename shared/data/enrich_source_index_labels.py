"""Enrich source_audio/source_index.parquet with per-segment instrument labels.

Two source datasets contribute:

  - MUSDB18-HQ: stems are pre-categorized by filename (drums.wav / bass.wav /
    vocals.wav / other.wav). Mapping is direct.
  - MoisesDB: each song has a `data.json` mapping track-UUIDs to detailed
    `trackType` strings. We map those granular types to a coarse vocabulary
    that matches what the Cambridge-MT stage 3 ingest produces.

Labels are used ONLY at synthesis time (sampling biased per-instrument). The
trained model never sees them — it consumes MERT embeddings instead. So the
mapping is purely an internal-pipeline crutch; if it's coarse, the model still
generalizes via MERT at inference.

Output: rewrites source_audio/source_index.parquet with the `instrument_label`
column populated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ===== MUSDB18-HQ ============================================================
# Source paths look like:
#   musdb18hq/train/<song_name>/{drums,bass,vocals,other,mixture}.wav
# We don't ingest mixture.wav (filtered at prepare time).

_MUSDB_STEM_TO_LABEL = {
    "drums":   "drum_other",   # full kit; specific kicks/snares not separable from MUSDB
    "bass":    "bass_electric",
    "vocals":  "vocal_other",  # MUSDB groups lead+bg
    "other":   "unknown",      # heterogeneous: keys, guitar, synth, etc — refuse to guess
}


def musdb_label_from_path(rel_path: str) -> str | None:
    """Infer label for a MUSDB18-HQ source path.

    rel_path examples:
      musdb18hq/train/A Classic Education - NightOwl/drums.wav
      musdb18hq/test/Al James - Schoolboy Facination/vocals.wav
    """
    parts = rel_path.split("/")
    if not parts or parts[0] != "musdb18hq":
        return None
    stem_filename = parts[-1]
    stem_key = stem_filename.lower().rsplit(".", 1)[0]   # "drums.wav" → "drums"
    return _MUSDB_STEM_TO_LABEL.get(stem_key)


# ===== MoisesDB =============================================================
# Granular trackType → coarse label. Matches cambridge stage 3 vocabulary.

_MOISES_TRACKTYPE_TO_LABEL = {
    # Drums
    "kick_drum":              "kick",
    "snare_drum":              "snare",
    "hi_hat":                  "hat",
    "cymbals":                 "cymbal",
    "toms":                    "tom",
    "overheads":               "drum_overhead",
    "full_acoustic_drumkit":   "drum_other",
    "drum_machine":            "drum_other",
    "a-tonal_percussion_(claps,_shakers,_congas,_cowbell_etc)": "drum_other",
    "pitched_percussion_(mallets,_glockenspiel,_...)":          "drum_other",

    # Bass
    "bass_guitar":             "bass_electric",
    "bass_synthesizer_(moog_etc)": "bass_synth",
    "contrabass/double_bass_(bass_of_instrings)": "bass_electric",

    # Guitar
    "acoustic_guitar":         "guitar_acoustic",
    "clean_electric_guitar":   "guitar_electric",
    "distorted_electric_guitar": "guitar_electric",

    # Vocals
    "lead_male_singer":        "vocal_lead",
    "lead_female_singer":      "vocal_lead",
    "background_vocals":       "vocal_bg",
    "other_(vocoder,_beatboxing_etc)": "vocal_other",

    # Keys
    "grand_piano":             "piano",
    "electric_piano_(rhodes,_wurlitzer,_piano_sound_alike)": "keys",
    "organ,_electric_organ":   "keys",

    # Synth
    "synth_lead":              "synth",
    "synth_pad":               "synth",

    # Brass / Wind
    "brass_(trumpet,_trombone,_french_horn,_brass_etc)": "brass",
    "reeds_(saxophone,_clarinets,_oboe,_english_horn,_bagpipe)": "woodwind",
    "flutes_(piccolo,_bamboo_flute,_panpipes,_flutes_etc)": "woodwind",
    "other_wind":              "woodwind",

    # Strings
    "string_section":          "strings",
    "cello_(solo)":            "strings",
    "cello_section":           "strings",
    "viola_(solo)":            "strings",
    "viola_section":           "strings",
    "other_strings":           "strings",
    "banjo,_mandolin,_ukulele,_harp_etc": "strings",

    # FX / other
    "fx/processed_sound,_scratches,_gun_shots,_explosions_etc": "fx",
    "other_sounds_(hapischord,_melotron_etc)": "keys",
}


def _build_moises_track_uuid_to_label(moises_root: Path) -> dict[str, str]:
    """Walk all data.json files; build UUID → coarse label mapping."""
    out: dict[str, str] = {}
    for song_dir in moises_root.iterdir():
        dj = song_dir / "data.json"
        if not dj.exists():
            continue
        try:
            with open(dj) as f:
                song = json.load(f)
        except Exception:
            continue
        for stem in song.get("stems", []):
            for track in stem.get("tracks", []):
                track_type = track.get("trackType", "")
                track_uuid = track.get("id", "")
                label = _MOISES_TRACKTYPE_TO_LABEL.get(track_type, "unknown")
                if track_uuid:
                    out[track_uuid] = label
    return out


def moises_label_from_path(rel_path: str, uuid_to_label: dict[str, str]) -> str | None:
    """rel_path examples:
      moisesdb/moisesdb_v0.1/<song-uuid>/drums/<track-uuid>.wav
      moisesdb/moisesdb/moisesdb_v0.1/<song-uuid>/<stem>/<track-uuid>.wav
    """
    parts = rel_path.split("/")
    if not parts:
        return None
    if "moisesdb" not in parts[0] and not (len(parts) > 0 and parts[0].startswith("moisesdb")):
        return None
    if parts[-1].endswith(".wav"):
        track_uuid = parts[-1].rsplit(".", 1)[0]
        return uuid_to_label.get(track_uuid)
    return None


# ===== Main =================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-index", default="source_audio/source_index.parquet")
    ap.add_argument("--moises-root", default="source_audio/moisesdb/moisesdb/moisesdb_v0.1")
    ap.add_argument("--output", default="",
                    help="default: rewrite the source-index in place")
    args = ap.parse_args()

    src_path = Path(args.source_index)
    moises_root = Path(args.moises_root)

    print(f"loading {src_path}")
    df = pd.read_parquet(src_path)
    print(f"  {len(df)} rows, datasets: {df['dataset'].value_counts().to_dict()}")

    print(f"\nbuilding MoisesDB UUID→label map from {moises_root}")
    uuid_to_label = _build_moises_track_uuid_to_label(moises_root)
    print(f"  {len(uuid_to_label)} track-UUIDs mapped")

    print("\nassigning labels...")
    new_labels: list[str | None] = []
    counts: dict[str, int] = {}
    for _, row in df.iterrows():
        rel = row["source_path"]
        ds = row["dataset"]
        if ds == "musdb18hq":
            label = musdb_label_from_path(rel)
        elif ds == "moisesdb":
            label = moises_label_from_path(rel, uuid_to_label)
        else:
            label = None
        new_labels.append(label)
        counts[str(label)] = counts.get(str(label), 0) + 1

    df["instrument_label"] = new_labels
    print("\nlabel distribution:")
    for lbl, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * n / len(df)
        print(f"  {n:6d}  {pct:5.1f}%  {lbl}")

    out_path = Path(args.output) if args.output else src_path
    print(f"\nwriting → {out_path}")
    df.to_parquet(out_path, index=False)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
