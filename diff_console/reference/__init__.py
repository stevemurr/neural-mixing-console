"""DMC reference DSP — analytical, non-differentiable ground-truth implementations.

These run on CPU with NumPy/SciPy (and Numba where the per-sample state recursion
demands it). The differentiable models are trained to match these outputs.
"""
