"""Scan extracted Cambridge-MT sessions for bundled reference mix files.

Heuristic only — no audio decoding. We classify candidate "full mix" files
using filename keywords, then *exclude* obvious sub-mix / bus files that
contain instrument-group prefixes (StringsMix, DrumsBus, etc.).

Stereo split files (`.L.wav` / `.R.wav` siblings produced by Pro Tools
exports) are paired into a single logical mix.

Output:
  - source_audio/cambridge_reference_mixes.json
  - stdout summary: how many sessions have a candidate mix
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(
    "/home/murr/Code/neural-mixing-console/source_audio/cambridge-mt"
)
OUT_JSON = Path(
    "/home/murr/Code/neural-mixing-console/source_audio/cambridge_reference_mixes.json"
)

# Keywords that typically mean "this is a full song mix"
MIX_KEYWORDS = [
    "reference mix",
    "ref mix",
    "ruff mix",
    "rough mix",
    "final mix",
    "master mix",
    "mixdown",
    "stereo mix",
    "song mix",
    "full mix",
    "main mix",
    "approved mix",
    "demo mix",
]

# Bus / sub-mix tokens that mean "this is a group, not the full song"
SUBMIX_TOKENS = [
    "drum", "drums", "kick", "snare", "tom", "hat", "ride", "cymbal",
    "bass", "guitar", "gtr", "string", "strings", "horn", "horns",
    "brass", "vocal", "vocals", "vox", "lead", "bgv", "harm",
    "key", "keys", "synth", "piano", "organ", "perc", "percussion",
    "fx", "sfx", "amb", "room", "overhead", "snr", "kik",
    "bus", "group", "sub", "stem",
]

AUDIO_EXTS = {".wav", ".flac", ".aif", ".aiff"}

LR_RE = re.compile(r"\.([LR])(?:\.[^.]+)?$", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z]+")


def looks_like_full_mix(stem_lower: str) -> tuple[bool, str]:
    """(is_candidate, reason) — stem_lower is filename without extension, lowercased."""
    # Pull out word tokens
    tokens = [t for t in re.split(r"[^a-z]+", stem_lower) if t]
    token_set = set(tokens)

    # If a sub-mix token appears, reject — even if "mix" also appears
    if token_set & set(SUBMIX_TOKENS):
        return (False, f"submix token: {sorted(token_set & set(SUBMIX_TOKENS))[0]}")

    # Strong matches: keyword phrases
    for kw in MIX_KEYWORDS:
        if kw in stem_lower:
            return (True, f"keyword: {kw!r}")

    # Standalone "mix" word, no submix token
    if "mix" in token_set:
        return (True, "standalone 'mix'")
    # Standalone "master" (mastered file)
    if "master" in token_set or "mastered" in token_set:
        return (True, "standalone 'master'")

    return (False, "no mix keyword")


def base_for_lr(name: str) -> tuple[str, str | None]:
    """Strip an L/R channel suffix. Returns (base_without_lr, channel_or_None)."""
    m = LR_RE.search(name)
    if m:
        # e.g. "Foo Mix.L.wav" → base = "Foo Mix.wav", channel = "L"
        ch = m.group(1).upper()
        before = name[: m.start()]
        # Re-attach extension
        ext = Path(name).suffix
        if ext.lower() in AUDIO_EXTS:
            return (before + ext, ch)
    return (name, None)


def scan_session(session_dir: Path) -> dict:
    # Auto-flatten single-folder wrappers, same as audition_server
    content_root = session_dir
    for _ in range(4):
        children = [c for c in content_root.iterdir()
                    if not c.name.startswith("._") and c.name != ".DS_Store"]
        if len(children) == 1 and children[0].is_dir():
            content_root = children[0]
        else:
            break

    candidates: list[dict] = []
    seen_bases: dict[str, dict] = {}  # base name → record

    for path in content_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        if path.name.startswith("._"):
            continue

        stem = Path(path.name).stem.lower()
        is_mix, reason = looks_like_full_mix(stem)
        if not is_mix:
            continue

        base_name, channel = base_for_lr(path.name)
        rel = path.relative_to(content_root).as_posix()
        size = path.stat().st_size

        key = base_name.lower()
        if key in seen_bases:
            rec = seen_bases[key]
            rec["channels"][channel or "M"] = rel
            rec["size"] += size
        else:
            rec = {
                "label": Path(base_name).stem,
                "reason": reason,
                "channels": {channel or "M": rel},
                "size": size,
            }
            seen_bases[key] = rec
            candidates.append(rec)

    return {
        "session": session_dir.name,
        "content_root_rel": content_root.relative_to(session_dir).as_posix() or ".",
        "candidates": candidates,
    }


def main() -> int:
    sessions = sorted([p for p in ROOT.iterdir() if p.is_dir()],
                      key=lambda p: p.name.lower())
    rows = [scan_session(s) for s in sessions]

    with_any = [r for r in rows if r["candidates"]]
    without = [r for r in rows if not r["candidates"]]

    OUT_JSON.write_text(json.dumps(rows, indent=2))

    print(f"sessions scanned:    {len(rows)}")
    print(f"  with mix candidate: {len(with_any)}  ({100*len(with_any)/max(len(rows),1):.1f}%)")
    print(f"  without:            {len(without)}")
    print(f"manifest written:    {OUT_JSON}")

    # Show sample reasons distribution
    reason_counts = defaultdict(int)
    for r in rows:
        for c in r["candidates"]:
            reason_counts[c["reason"]] += 1
    print("\nreasons (top):")
    for reason, n in sorted(reason_counts.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {n:>4}  {reason}")

    print("\nsample sessions WITH candidates:")
    for r in with_any[:5]:
        labels = ", ".join(c["label"] for c in r["candidates"][:3])
        print(f"  {r['session']:60s} → {labels}")
    print("\nsample sessions WITHOUT:")
    for r in without[:5]:
        print(f"  {r['session']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
