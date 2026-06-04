"""Canonical instrument-group classification from correspondence.yaml group names.

Cambridge's correspondence.yaml uses 158 distinct group-name spellings spanning
spelled-out forms ("Drum", "BackingVox") and 2-3 letter codes ("DR", "BV",
"EG"). This module maps all of them to a small canonical set used as an input
feature for the v13 panning model (Stage 3) so it can learn engineering
conventions like "drums tend center, guitars tend spread."

The same rules also power the bin-class separability analysis script — both
import from here to stay in sync.
"""

from __future__ import annotations


# Canonical class names in a fixed order. The integer index of each class is
# the embedding lookup key for the panning model. "Unknown" is reserved for
# group names that don't match any rule.
CANONICAL_GROUPS: list[str] = [
    "Drums",
    "Bass",
    "Guitar",
    "LeadVox",
    "BackingVox",
    "Synth",
    "Keys",
    "Strings",
    "Brass",
    "Other",
    "Unknown",
]

GROUP_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(CANONICAL_GROUPS)}
UNKNOWN_IDX: int = GROUP_TO_IDX["Unknown"]

# Each canonical class has a list of group-name prefixes that map to it.
# Matching is case-insensitive and accepts exact match OR prefix match.
CLASS_RULES: list[tuple[str, tuple[str, ...]]] = [
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
    """Map a raw correspondence.yaml group name to a canonical class string."""
    n = name.lower().strip()
    for cls, prefixes in CLASS_RULES:
        for p in prefixes:
            if n == p or n.startswith(p):
                return cls
    return "Other"


def group_name_to_idx(name: str) -> int:
    """Map a raw group name to its canonical class index (for embedding lookup)."""
    return GROUP_TO_IDX[classify_group_name(name)]


def n_canonical_groups() -> int:
    """Total embedding cardinality including the Unknown slot (for padded tracks)."""
    return len(CANONICAL_GROUPS)


__all__ = [
    "CANONICAL_GROUPS",
    "GROUP_TO_IDX",
    "UNKNOWN_IDX",
    "CLASS_RULES",
    "classify_group_name",
    "group_name_to_idx",
    "n_canonical_groups",
]
