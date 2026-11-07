"""NVFP4 linears (ModelOpt layout): packed E2M1 weights, E4M3 block scales of 16; needs FlashInfer and Blackwell."""

import math

import torch
from torch import Tensor, nn

PACKED = torch.uint8
FP8 = torch.float8_e4m3fn
BLOCK = 16
SCALE_SUFFIX = ".weight_scale_2"


def _ceil(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _flashinfer():
    try:
        import flashinfer
    except ImportError as e:
        raise RuntimeError("--nvfp4 needs FlashInfer: pip install -e '.[nvfp4]' "
                           "--extra-index-url https://download.pytorch.org/whl/cu130 "
                           "--extra-index-url https://flashinfer.ai/whl/cu130") from e
    return flashinfer


class NVFP4Linear(nn.Module):
    """weight uint8 [N, K/2] (two E2M1 per byte, even element low), weight_scale E4M3 [N, K/16], fp32 weight_scale_2
    (global) and input_scale (static activation scale)."""

    def __init__(self, in_features: int, out_features: int, device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features // 2, dtype=PACKED, device=device))
        self.register_buffer("weight_scale", torch.empty(out_features, in_features // BLOCK, dtype=FP8, device=device))
        self.register_buffer("weight_scale_2", torch.empty((), dtype=torch.float32, device=device))
        self.register_buffer("input_scale", torch.empty((), dtype=torch.float32, device=device))

    def forward(self, x: Tensor) -> Tensor:
        leading = tuple(x.shape[:-1])
        rows = math.prod(leading)
        if rows == 0:
            return x.new_zeros(*leading, self.out_features, dtype=torch.bfloat16)
        flashinfer = _flashinfer()
        flat = x.to(torch.bfloat16).reshape(rows, self.in_features).contiguous()
        inv_a = self.input_scale.float().clamp(min=1e-8).reshape(1)
        x_q, a_sf = flashinfer.nvfp4_quantize(flat, (1.0 / inv_a).reshape(1),
                                              sfLayout=flashinfer.SfLayout.layout_128x4, do_shuffle=False)
        alpha = (self.input_scale * self.weight_scale_2).float().reshape(1)
        out = flashinfer.mm_fp4(x_q, self.weight.T, a_sf, self.weight_scale.T, alpha=alpha,
                                out_dtype=torch.bfloat16, backend="cutlass")
        return out.reshape(*leading, self.out_features)


def swizzle_weight_scales(model: nn.Module) -> None:
    """[N, K/16] block scales -> FlashInfer's 128x4 layout (rows padded to 128, columns to 4), once after loading."""
    interleave = _flashinfer().block_scale_interleave
    for module in model.modules():
        if isinstance(module, NVFP4Linear):
            rows, cols = module.weight_scale.shape
            swizzled = interleave(module.weight_scale.view(torch.uint8).contiguous())
            module.weight_scale = swizzled.reshape(_ceil(rows, 128), _ceil(cols, 4)).view(FP8)
