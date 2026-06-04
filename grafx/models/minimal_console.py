"""MinimalConsole — strict-minimum DSP graph for v12+.

Pivot from `GrafxMixingConsole` to address the gauge-degeneracy problem
documented in `notes/minimal_console_design_2026-05.md`:

  - GrafxMixingConsole had ~2687 params per strip — dominated by 1024-bin
    FIR EQ and 386-bin reverb impulse magnitudes, both ~10x over-
    parameterized for their perceptual degrees of freedom. The Adam
    search in grafx-prune lands on arbitrary points in those flat
    directions, producing labels with no learnable signal in the high-
    dim params.

  - MinimalConsole drops to ~20 params per strip after v12.1 Path B
    cleanup: RBJ-biquad EQ (14) + Compressor (5 incl. drywet) +
    StereoImager (1) + Gain (2). Each param ≈ 1-to-1 with a perceptual
    degree of freedom.

v12.1 Path B gauge fixes (vs v12.0):
  - Group chain drops `gain_panning` → eliminates strip×group gain
    product gauge (the dominant mean-collapse failure mode).
  - DryWet logits removed from `eq`, `stereo_imager`, `gain_panning` →
    eliminates proc-gain × drywet gauge. Kept on `compressor` only,
    because parallel compression is a genuine engineering DoF.

Strip chain (default): EQ → Compressor → StereoImager → Gain/Pan
Group chain (default): EQ → Compressor → StereoImager   (no gain)

Per processor (post Path B):
  - `eq.{hpf_freq, ls_freq, ls_gain, ls_q, p1_freq, p1_gain, p1_q,
         p2_freq, p2_gain, p2_q, hs_freq, hs_gain, hs_q, lpf_freq}` (14)
  - `compressor.{log_threshold, log_ratio, log_knee, z_alpha_pre,
                 drywet_logit}`                                       (5)
  - `stereo_imager.{log_gain}`                                        (1)
  - `gain_panning.{log_gain}` (log_gain is (2,))                      (2)

Total per strip: 14 + 5 + 1 + 2 = 22 scalars.
Total per group: 14 + 5 + 1     = 20 scalars (no gain).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from grafx import processors as gp

from .fft_eq import FFTEQ


DEFAULT_STRIP_PROCESSOR_ORDER = (
    "eq",            # RBJ-biquad 6-band — replaces grafx ZeroPhaseFIREqualizer
    "compressor",    # grafx Compressor (4 params; already parsimonious)
    "stereo_imager", # grafx SideGainImager (1 param side-gain)
    "gain_panning",  # grafx StereoGain (L/R gain)
)

DEFAULT_GROUP_PROCESSOR_ORDER = (
    "eq",
    "compressor",
    "stereo_imager",
    # No gain_panning: collapses the strip×group gain gauge.
)

# Backwards compat alias — old callers used DEFAULT_PROCESSOR_ORDER as both.
DEFAULT_PROCESSOR_ORDER = DEFAULT_STRIP_PROCESSOR_ORDER

# Procs that get a learnable drywet (true engineering DoF). Others are
# always fully wet to remove the proc-gain × drywet gauge orbit.
_DRYWET_PROCESSORS = frozenset({"compressor"})


# ---------- EQ wrapper: adapt DiffEQ to the kwarg-dict-per-processor schema ----------

class _MinimalEQ(nn.Module):
    """Wrap `FFTEQ` to match the (proc.parameter_size(), proc(x, **params))
    convention used by grafx processors.

    FFTEQ takes a `params=dict_of_(B,)-tensors` argument; the console's
    `_run_chain` calls `proc(x, **kwargs)` with kwargs of (leading, 1)
    shape. We squeeze the trailing dim before delegating.

    Why FFTEQ instead of DiffEQ: torchaudio's `lfilter` backward is
    buggy on aarch64 (Xbyak JIT) and numerically delicate at near-
    degenerate biquad coefficients. FFTEQ applies the same RBJ-biquad
    cascade via FFT/multiply/IFFT, going through robust torch.fft ops.
    Same 14-scalar parameterization. See `models/fft_eq.py`.
    """

    PARAM_NAMES = FFTEQ.PARAM_NAMES  # 14 scalar params

    def __init__(self, sample_rate: int = 48_000):
        super().__init__()
        self.eq = FFTEQ(sample_rate=sample_rate)

    def parameter_size(self) -> dict[str, tuple[int, ...]]:
        return {name: (1,) for name in self.PARAM_NAMES}

    def forward(self, x: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        # Each kwarg is shape (leading, 1). Squeeze to (leading,) for DiffEQ.
        params = {k: kwargs[k].squeeze(-1) for k in self.PARAM_NAMES}
        return self.eq(x, params)


# ---------- DryWet logit wrapper (identical contract to GrafxMixingConsole) ----------

class _DryWetLogitWrap(nn.Module):
    """Linear blend of dry + wet processor output, indexed by a logit.

    `wet = sigmoid(drywet_logit)`; `out = wet * processed + (1-wet) * input`.
    Numerically identical to grafx-prune's DryWet but takes the pre-sigmoid
    logit as input — gradients flow in logit space (no dead-sigmoid risk
    at the encoder head).
    """

    def __init__(self, processor: nn.Module):
        super().__init__()
        self.processor = processor

    def forward(
        self, x: torch.Tensor, drywet_logit: torch.Tensor, **processor_kwargs,
    ) -> torch.Tensor:
        y = self.processor(x, **processor_kwargs)
        if isinstance(y, tuple):
            y = y[0]
        wet = torch.sigmoid(drywet_logit).view(-1, 1, 1)
        return wet * y + (1.0 - wet) * x

    def parameter_size(self) -> dict[str, tuple[int, ...] | int]:
        return {**self.processor.parameter_size(), "drywet_logit": 1}


# ---------- Processor factory ----------

def _make_processor(name: str, sample_rate: int, max_input_len: int) -> nn.Module:
    """Instantiate one processor; wrap in `_DryWetLogitWrap` only for the
    procs in `_DRYWET_PROCESSORS` (currently just `compressor` — see
    module docstring on Path B drywet collapse)."""
    match name:
        case "eq":
            inner = _MinimalEQ(sample_rate=sample_rate)
        case "compressor":
            inner = gp.Compressor(max_input_len=max_input_len)
        case "stereo_imager":
            inner = gp.SideGainImager()
        case "gain_panning":
            inner = gp.StereoGain()
        case _:
            raise ValueError(f"unknown processor: {name}")
    if name in _DRYWET_PROCESSORS:
        return _DryWetLogitWrap(inner)
    return inner


def _normalize_shape(shp) -> tuple[int, ...]:
    if isinstance(shp, int):
        return (shp,)
    return tuple(shp)


# ---------- MinimalConsole ----------

class MinimalConsole(nn.Module):
    """Two-level mixing console: per-track strip -> instrument group -> master sum.

    Same shape contract as `GrafxMixingConsole` so the encoder, losses, and
    trainer integration are drop-in compatible. The schema is much smaller
    though: ~25 scalar params per strip vs ~2687.

    Strip and group both run the same 4-processor chain (EQ -> Comp ->
    StereoImager -> Gain/Pan). Parameters dicts:

        strip_params[processor_name][param_name] : (B, N, *param_shape)
        group_params[processor_name][param_name] : (B, G, *param_shape)

    Output: (B, 2, T) master mix.
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        max_input_len: int = 288_000,
        strip_processors_list: Sequence[str] = DEFAULT_STRIP_PROCESSOR_ORDER,
        group_processors_list: Sequence[str] = DEFAULT_GROUP_PROCESSOR_ORDER,
        # Legacy alias: callers passing `processors_list=` set both.
        processors_list: Sequence[str] | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_input_len = max_input_len
        if processors_list is not None:
            strip_processors_list = processors_list
            group_processors_list = processors_list
        self.strip_processors_list = tuple(strip_processors_list)
        self.group_processors_list = tuple(group_processors_list)
        # Back-compat: some external code reads `processors_list`. Point it
        # at the strip list (the more-complete chain).
        self.processors_list = self.strip_processors_list

        self.strip_processors = nn.ModuleDict({
            name: _make_processor(name, sample_rate, max_input_len)
            for name in self.strip_processors_list
        })
        self.group_processors = nn.ModuleDict({
            name: _make_processor(name, sample_rate, max_input_len)
            for name in self.group_processors_list
        })

    @property
    def strip_param_shapes(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return {
            name: {k: _normalize_shape(v) for k, v in proc.parameter_size().items()}
            for name, proc in self.strip_processors.items()
        }

    @property
    def group_param_shapes(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return {
            name: {k: _normalize_shape(v) for k, v in proc.parameter_size().items()}
            for name, proc in self.group_processors.items()
        }

    @staticmethod
    def _run_chain(
        x: torch.Tensor,
        processors_dict: nn.ModuleDict,
        params_dict: dict[str, dict[str, torch.Tensor]],
        order: Sequence[str],
    ) -> torch.Tensor:
        for name in order:
            proc = processors_dict[name]
            kwargs = params_dict[name]
            y = proc(x, **kwargs)
            if isinstance(y, tuple):
                y = y[0]
            x = y
        return x

    @staticmethod
    def _run_chain_capture(
        x: torch.Tensor,
        processors_dict: nn.ModuleDict,
        params_dict: dict[str, dict[str, torch.Tensor]],
        order: Sequence[str],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        intermediates: dict[str, torch.Tensor] = {}
        for name in order:
            proc = processors_dict[name]
            kwargs = params_dict[name]
            y = proc(x, **kwargs)
            if isinstance(y, tuple):
                y = y[0]
            x = y
            intermediates[name] = x
        return x, intermediates

    def forward(
        self,
        stems: torch.Tensor,                                 # (B, N, 2, T)
        strip_params: dict[str, dict[str, torch.Tensor]],   # per (proc, param): (B, N, *shape)
        group_params: dict[str, dict[str, torch.Tensor]],   # per (proc, param): (B, G, *shape)
        group_assignments: torch.Tensor,                     # (B, N) long
        track_mask: torch.Tensor | None = None,             # (B, N) bool
        n_groups: int | None = None,
    ) -> torch.Tensor:
        if stems.dim() != 4 or stems.shape[2] != 2:
            raise ValueError(f"expected (B, N, 2, T) stems; got {stems.shape}")
        B, N, C, T = stems.shape
        if group_assignments.shape != (B, N):
            raise ValueError(f"group_assignments must be (B={B}, N={N}); got {group_assignments.shape}")
        if n_groups is None:
            n_groups = int(group_assignments.max().item()) + 1

        # Per-track strip
        x = stems.reshape(B * N, C, T)
        flat_strip = {
            proc_name: {k: v.reshape(B * N, *v.shape[2:]) for k, v in pp.items()}
            for proc_name, pp in strip_params.items()
        }
        x = self._run_chain(x, self.strip_processors, flat_strip, self.strip_processors_list)
        x = x.reshape(B, N, C, T)

        if track_mask is not None:
            x = x * track_mask.to(x.dtype).view(B, N, 1, 1)

        # Scatter-sum into groups
        idx = group_assignments.long().view(B, N, 1, 1).expand(B, N, C, T)
        group_out = torch.zeros(B, n_groups, C, T, device=x.device, dtype=x.dtype)
        group_out.scatter_add_(dim=1, index=idx, src=x)

        # Per-group bus
        g = group_out.reshape(B * n_groups, C, T)
        flat_group = {
            proc_name: {k: v.reshape(B * n_groups, *v.shape[2:]) for k, v in pp.items()}
            for proc_name, pp in group_params.items()
        }
        g = self._run_chain(g, self.group_processors, flat_group, self.group_processors_list)
        g = g.reshape(B, n_groups, C, T)

        # Master sum
        return g.sum(dim=1)

    def forward_with_intermediates(
        self,
        stems: torch.Tensor,
        strip_params: dict[str, dict[str, torch.Tensor]],
        group_params: dict[str, dict[str, torch.Tensor]],
        group_assignments: torch.Tensor,
        track_mask: torch.Tensor | None = None,
        n_groups: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Like `forward` but returns per-processor intermediates per level."""
        if stems.dim() != 4 or stems.shape[2] != 2:
            raise ValueError(f"expected (B, N, 2, T) stems; got {stems.shape}")
        B, N, C, T = stems.shape
        if n_groups is None:
            n_groups = int(group_assignments.max().item()) + 1

        x = stems.reshape(B * N, C, T)
        flat_strip = {
            proc_name: {k: v.reshape(B * N, *v.shape[2:]) for k, v in pp.items()}
            for proc_name, pp in strip_params.items()
        }
        x, flat_strip_inter = self._run_chain_capture(
            x, self.strip_processors, flat_strip, self.strip_processors_list,
        )
        strip_inter = {
            name: t.reshape(B, N, C, T) for name, t in flat_strip_inter.items()
        }
        x = x.reshape(B, N, C, T)

        if track_mask is not None:
            x = x * track_mask.to(x.dtype).view(B, N, 1, 1)

        idx = group_assignments.long().view(B, N, 1, 1).expand(B, N, C, T)
        group_out = torch.zeros(B, n_groups, C, T, device=x.device, dtype=x.dtype)
        group_out.scatter_add_(dim=1, index=idx, src=x)

        g = group_out.reshape(B * n_groups, C, T)
        flat_group = {
            proc_name: {k: v.reshape(B * n_groups, *v.shape[2:]) for k, v in pp.items()}
            for proc_name, pp in group_params.items()
        }
        g, flat_group_inter = self._run_chain_capture(
            g, self.group_processors, flat_group, self.group_processors_list,
        )
        group_inter = {
            name: t.reshape(B, n_groups, C, T) for name, t in flat_group_inter.items()
        }
        g = g.reshape(B, n_groups, C, T)

        return g.sum(dim=1), strip_inter, group_inter


__all__ = [
    "MinimalConsole",
    "DEFAULT_STRIP_PROCESSOR_ORDER",
    "DEFAULT_GROUP_PROCESSOR_ORDER",
    "DEFAULT_PROCESSOR_ORDER",  # legacy alias = strip order
]
