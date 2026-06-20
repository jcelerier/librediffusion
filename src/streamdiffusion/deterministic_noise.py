"""Deterministic counter-based PCG32 Gaussian noise — the Python side of a generator that is
bit-identical to the C++/CUDA `launch_randn_fp16` kernel (see librediffusion src/kernels.cu and
src/pcg.hpp). This replaces torch.randn / cuRAND so python<->C++ txt2img (and any generated
noise) match exactly.

librediffusion fork: torch.randn (Philox + torch's uniform->Gaussian transform/layout) and
cuRAND's curandGenerateNormal do NOT agree bit-for-bit even with the same seed. Instead both
sides use ONE explicit counter-based spec defined here and mirrored in CUDA.

SPEC (must match kernels.cu exactly):
  For each flat output element `index`:
    s0 = seed (uint64), s1 = index (uint64)
    seed PCG32 with (s0, s1) using pcg.hpp's seed() procedure
    u1 = (next_u32() + 1) * 2^-32     # uniform in (0,1]
    u2 = (next_u32() + 1) * 2^-32
    z0 = sqrt(-2*ln(u1)) * cos(2*pi*u2)   # Box-Muller, take z0
    out[index] = float16(z0)              # all math in fp32, cast to fp16 last
  index = flat row-major offset into the [B,4,H,W] tensor; the value depends ONLY on
  (seed, index), never on shape/batch/how many elements are requested.
"""
from __future__ import annotations

import math

import numpy as np
import torch

# PCG32 constants (must match src/pcg.hpp)
_PCG_MULT = np.uint64(6364136223846793005)
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _rotr32(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """32-bit rotate-right; x and r are uint32 arrays."""
    x = x.astype(np.uint32)
    r = r.astype(np.uint32) & np.uint32(31)
    # (x >> r) | (x << ((-r) & 31))
    left = (np.uint32(32) - r) & np.uint32(31)
    return (x >> r) | (x << left)


class _PCG32:
    """Vectorized PCG32 over an array of independent streams (one per output element).
    state/inc are uint64 arrays. Mirrors pcg.hpp operator()/seed()."""

    def __init__(self, s0: np.ndarray, s1: np.ndarray):
        # pcg.hpp seed(s0, s1): m_inc=(s1<<1)|1; m_state=0; step(); m_state+=s0; step()
        self.inc = ((s1.astype(np.uint64) << np.uint64(1)) | np.uint64(1)) & _MASK64
        self.state = np.zeros_like(s0, dtype=np.uint64)
        self._step()
        self.state = (self.state + s0.astype(np.uint64)) & _MASK64
        self._step()

    def _step(self) -> np.ndarray:
        """Advance LCG, return the previous-state output (uint32 array)."""
        old = self.state
        self.state = (old * _PCG_MULT + self.inc) & _MASK64
        # xorshifted = uint32(((old >> 18) ^ old) >> 27)
        xorshifted = (((old >> np.uint64(18)) ^ old) >> np.uint64(27)).astype(np.uint32)
        rot = (old >> np.uint64(59)).astype(np.uint32)
        return _rotr32(xorshifted, rot)

    def next_u32(self) -> np.ndarray:
        return self._step()


def pcg32_randn_numpy(seed: int, n: int) -> np.ndarray:
    """Return n fp16 standard-normal samples; element i == randn(seed, i). Returns float16 array."""
    idx = np.arange(n, dtype=np.uint64)
    s0 = np.full(n, np.uint64(seed) & _MASK64, dtype=np.uint64)
    rng = _PCG32(s0, idx)
    # all math in fp32 to match the CUDA kernel's precision path, cast to fp16 last
    two32 = np.float32(2.0) ** np.float32(-32)
    u1 = (rng.next_u32().astype(np.float32) + np.float32(1.0)) * two32
    u2 = (rng.next_u32().astype(np.float32) + np.float32(1.0)) * two32
    r = np.sqrt(np.float32(-2.0) * np.log(u1)).astype(np.float32)
    z0 = (r * np.cos(np.float32(2.0) * np.float32(math.pi) * u2)).astype(np.float32)
    return z0.astype(np.float16)


def pcg32_randn(
    seed: int,
    shape,
    device=None,
    dtype=torch.float16,
) -> torch.Tensor:
    """Deterministic Gaussian noise tensor of the given shape. Flat element i == randn(seed, i).
    Computed as fp16 (matching the C++ kernel), then cast to the requested dtype."""
    n = 1
    for s in shape:
        n *= int(s)
    flat16 = pcg32_randn_numpy(int(seed), n)  # float16, shape-independent by index
    t = torch.from_numpy(flat16.copy()).reshape(tuple(int(s) for s in shape))
    if device is not None:
        t = t.to(device)
    if dtype is not None and dtype != torch.float16:
        t = t.to(dtype)
    return t
