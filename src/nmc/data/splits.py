import hashlib


def session_split(
    sessions: list[str], val_frac: float = 0.1, seed: int = 0,
) -> tuple[set[str], set[str]]:
    """Deterministic session-level split using stable hashing on (seed,name).

    The MD5-based bucketing means adding new sessions later doesn't reshuffle
    the existing split — each session keeps the same fold across runs.
    """
    train, val = set(), set()
    for name in sorted(set(sessions)):
        h = hashlib.md5(f"{seed}::{name}".encode()).hexdigest()
        # First 8 hex digits → 32-bit int / 2^32 → uniform in [0, 1)
        x = int(h[:8], 16) / (1 << 32)
        (val if x < val_frac else train).add(name)
    return train, val
