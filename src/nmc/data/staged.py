"""Read historical Cambridge staged audio and group correspondence."""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from nmc.data.groups import group_name_to_idx


def load_session_audio(
    session_dir: Path, sample_rate: int,
) -> dict | None:
    """Load engineer mix + stems for a session, aligned + stereo + (N, 2, T).

    Also returns `stem_groups`: the canonical-group index for each stem,
    derived from the correspondence.yaml group it lives under. This is the
    input feature that lets Stage 3 learn engineering conventions like
    "drum-group tracks tend center, guitar-group tracks tend spread."
    """
    corr_path = session_dir / "correspondence.yaml"
    mix_path = session_dir / "mix.wav"
    if not (corr_path.is_file() and mix_path.is_file()):
        return None
    try:
        corr = yaml.safe_load(corr_path.read_text())
    except Exception:
        return None
    if not isinstance(corr, dict):
        return None
    # Flatten stems while preserving group identity per stem
    stem_filenames: list[str] = []
    stem_groups: list[int] = []
    for group_name, files in corr.items():
        if not isinstance(files, list):
            continue
        g_idx = group_name_to_idx(group_name)
        for f in files:
            stem_filenames.append(f)
            stem_groups.append(g_idx)
    if not stem_filenames:
        return None

    mix_audio, sr = sf.read(str(mix_path), dtype="float32", always_2d=True)
    if sr != sample_rate or mix_audio.shape[0] < 4096:
        return None
    if mix_audio.shape[1] == 1:
        mix_audio = np.repeat(mix_audio, 2, axis=1)
    mix_audio = mix_audio[:, :2]

    stems_list: list[np.ndarray] = []
    fnames: list[str] = []
    fgroups: list[int] = []
    for fname, g_idx in zip(stem_filenames, stem_groups):
        path = session_dir / "stems" / fname
        if not path.is_file():
            continue
        try:
            s, sr_s = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception:
            continue
        if sr_s != sample_rate or s.shape[0] < 4096:
            continue
        if s.shape[1] == 1:
            s = np.repeat(s, 2, axis=1)
        s = s[:, :2]
        stems_list.append(s)
        fnames.append(fname)
        fgroups.append(g_idx)

    if not stems_list:
        return None

    T = min([s.shape[0] for s in stems_list] + [mix_audio.shape[0]])
    mix_audio = mix_audio[:T]
    stems_arr = np.stack([s[:T] for s in stems_list], axis=0)   # (N, T, 2)
    stems = torch.from_numpy(np.ascontiguousarray(stems_arr.transpose(0, 2, 1)))
    mix = torch.from_numpy(np.ascontiguousarray(mix_audio.T))   # (2, T)
    return {
        "stems":          stems,
        "mix":            mix,
        "stem_filenames": fnames,
        "stem_groups":    fgroups,                # (N,) list of canonical group indices
        "T_total":        T,
    }
