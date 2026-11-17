"""Differentiable DSP modules + encoders for the DMC project.

All modules:
- Inherit from `torch.nn.Module`.
- Operate on tensors of shape `(B, C, T)` — batch, channels, time.
- Accept parameters as Tensors of shape `(B,)` per scalar param, or `(B, K)`
  one-hot probability vectors for categoricals (delay subdivision, reverb mode).
- Bypass flags are passed as `bool` Tensors of shape `(B,)` (or `(B, N_bands)`
  for EQ); when True, the corresponding sub-block returns identity.
- All ops are differentiable end-to-end (the parallel-smoothers + min trick
  in the compressor preserves gradients without temperature dials).
"""
