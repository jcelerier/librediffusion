"""Rowwise FP8 linears: E4M3 weights with a per-row fp32 scale, activations quantized per row."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
SCALE_SUFFIX = ".weight_scale"
GEMM_ALIGNMENT = 256  # N and K of the GEMM padded to this
ROW_ALIGNMENT = 128  # activation rows padded to this


def _ceil(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def quantize_rows(value: Tensor) -> tuple[Tensor, Tensor]:
    rows = value.reshape(-1, value.shape[-1]).float()
    scale = (rows.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
    quantized = (rows / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return quantized.reshape_as(value), scale


class FP8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8, device=device))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32, device=device))

    def forward(self, x: Tensor) -> Tensor:
        leading = tuple(x.shape[:-1])
        rows = math.prod(leading)
        flat = x.to(torch.bfloat16).reshape(rows, self.in_features)
        if rows == 0:
            return x.new_zeros(*leading, self.out_features, dtype=torch.bfloat16)
        n, k, m = _ceil(self.out_features, GEMM_ALIGNMENT), _ceil(self.in_features, GEMM_ALIGNMENT), _ceil(rows, ROW_ALIGNMENT)
        weight, scale = self.weight, self.weight_scale
        if (n, k) != (self.out_features, self.in_features):
            weight = F.pad(weight.view(torch.uint8), (0, k - self.in_features, 0, n - self.out_features)).view(FP8)
            scale = F.pad(scale, (0, n - self.out_features), value=1.0)
        if (m, k) != (rows, self.in_features):
            flat = F.pad(flat, (0, k - self.in_features, 0, m - rows))
        q, q_scale = quantize_rows(flat.contiguous())
        out = torch._scaled_mm(q, weight.T, q_scale[:, None].contiguous(), scale[None, :].contiguous(),
                               out_dtype=torch.bfloat16, use_fast_accum=True)
        return out[:rows, : self.out_features].reshape(*leading, self.out_features)
