"""Shared FP8 (e4m3) quantization for SD/SDXL UNet export — the diffusion FP8 recipe.

This generalizes the klein transformer FP8 path (klein/export_klein_fp8_calib.py) to the SD/SDXL
UNet so `train-lora.py --fp8` can weave FP8 directly into the existing ONNX→TRT export, instead of a
separate script. The recipe is NVIDIA demoDiffusion / nvidia-modelopt best-practice:

  - Calibrate amax with modelopt's FP8_DEFAULT_CFG over a handful of REAL forward passes (encoded
    prompts + random latents across several timesteps for range-widening). The amax pass is exact
    regardless of the CUDA ext (modelopt falls back to an exact torch.float8_e4m3fn fake-quant; the
    "simulated" warning is about speed, not correctness).
  - Tag the high-precision (dequant) path with trt_high_precision_dtype so TRT keeps the surrounding
    compute in the model's native float type (SDXL = "Half"/fp16, klein = "BFloat16").
  - After quantize, DISABLE quantizers on the precision-sensitive layers (time/text embedders, the
    AdaGN/time projections, conv_in/conv_out/conv_shortcut) so they stay high-precision in the engine.
    Disable BEFORE export so a stale amax isn't baked in. This is the bit plain FP8_DEFAULT_CFG omits
    and the reason naive FP8 SD/SDXL loses quality.

The heavy compute (attention q/k/v/out + FF Linears + the resnet convs) is what gets FP8 — which the
cross-model profiling showed is 41–58% of SDXL frame time (DRAM-bound tensor-GEMM), the exact slice
weight quantization relieves.
"""
from __future__ import annotations

import copy
import re

import torch

# demoDiffusion's canonical SDXL/SD UNet FP8 sensitive-layer filter: keep these high-precision.
#   time_emb_proj / time_embedding : timestep embedding MLP (tiny, precision-critical)
#   add_embedding                  : SDXL text_embeds+time_ids projection (SDXL only; harmless on SD1.5)
#   conv_in / conv_out             : first/last conv (input/output precision anchors)
#   conv_shortcut                  : resnet skip 1x1 convs (numerically sensitive)
SDXL_SENSITIVE = re.compile(
    r".*(time_emb_proj|time_embedding|add_embedding|conv_in|conv_out|conv_shortcut).*"
)


def fp8_quant_cfg(high_precision: str = "Half"):
    """modelopt FP8_DEFAULT_CFG with the dequant path tagged to the model's native float type.

    high_precision: "Half" for fp16 models (SD/SDXL), "BFloat16" for bf16 models (klein).
    """
    import modelopt.torch.quantization as mtq

    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    for rule in cfg["quant_cfg"].values() if isinstance(cfg["quant_cfg"], dict) else cfg["quant_cfg"]:
        c = rule.get("cfg") if isinstance(rule, dict) else None
        if c is not None and "num_bits" in c:
            c["trt_high_precision_dtype"] = high_precision
    return cfg


def quantize_unet_fp8(module: torch.nn.Module, calib_inputs, high_precision: str = "Half",
                      sensitive: re.Pattern = SDXL_SENSITIVE, quantize_conv: bool = False,
                      quantize_attention: bool = False):
    """FP8-quantize `module` in place (modelopt), then disable quantizers on sensitive layers.

    `module` is the traced export wrapper (e.g. SDXLUNetWrapper) — quantizers are inserted on the
    Linear/Conv submodules it reaches. `calib_inputs` is a list of positional-arg tuples matching
    `module.forward(...)`. Returns (n_enabled_after, n_disabled, disabled_names).

    quantize_conv=False (default): FP8 only the Linear/attention compute (the 41-58% tensor-GEMM the
    profiling found DRAM-bound), keeping ALL Conv2d in fp16. Two reasons: (1) the resnet convs are only
    ~18% of frame time and L2-bound (less to gain from weight quant); (2) modelopt tags the conv weight
    with a custom `trt::TRT_FP8DequantizeLinear` op that the TorchScript ONNX exporter cannot shape-infer
    a convolution kernel through ("convolution of unknown shape"). Disabling the conv quantizers makes
    QuantConv2d export as a plain conv. Set quantize_conv=True only with an ONNX export path that handles
    the custom conv-DQ op.
    """
    import modelopt.torch.quantization as mtq

    module.eval()

    def forward_loop(m):
        with torch.no_grad():
            for ins in calib_inputs:
                m(*ins)

    print(f"[fp8] calibrating ({len(calib_inputs)} passes, high_precision={high_precision})...")
    mtq.quantize(module, fp8_quant_cfg(high_precision), forward_loop)

    def _enabled():
        return sum(1 for _, mod in module.named_modules()
                   if getattr(getattr(mod, "weight_quantizer", None), "is_enabled", False))

    enabled_before = _enabled()
    n_disabled = 0
    disabled_names = []
    n_conv_disabled = 0
    for name, mod in module.named_modules():
        is_conv = isinstance(mod, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Conv1d,
                                   torch.nn.Conv3d))
        # FP8 the Linear/attention compute only: skip sensitive layers AND (unless quantize_conv) all
        # convolutions — the conv weight's custom FP8-DQ op breaks ONNX conv kernel-shape inference.
        drop = sensitive.match(name) or (is_conv and not quantize_conv)
        if not drop:
            continue
        hit = False
        for qn in ("input_quantizer", "weight_quantizer", "output_quantizer"):
            q = getattr(mod, qn, None)
            if q is not None and getattr(q, "is_enabled", False):
                q.disable()
                n_disabled += 1
                hit = True
        if hit:
            if is_conv and not sensitive.match(name):
                n_conv_disabled += 1
            else:
                disabled_names.append(name)
    if n_conv_disabled:
        print(f"[fp8] kept {n_conv_disabled} Conv layers in fp16")

    # Quantized attention (modelopt's FP8SDPA diffusion plugin) builds at 512 but TRT/Myelin FAILS to
    # build the larger attention subgraph at 1024+ ("Assertion g.nodes.size()==0 in setupProxyGraph").
    # Default: keep SDPA in fp16 (disable the attention bmm/softmax quantizers on each QuantAttention so
    # its forward falls back to plain SDPA) — FP8 only the Linear q/k/v/out projections + FF. This
    # matches the klein recipe ("attention is plain Linear q/k/v/out + SDPA, SDPA stays high-precision")
    # and is resolution-robust. Attention is only ~7% of SDXL frame time, so the cost is negligible.
    if not quantize_attention:
        n_attn = 0
        for _, mod in module.named_modules():
            if "Attention" not in type(mod).__name__:
                continue
            for _, child in mod.named_children():
                if type(child).__name__ == "TensorQuantizer" and getattr(child, "is_enabled", False):
                    child.disable()
                    n_attn += 1
        if n_attn:
            print(f"[fp8] disabled {n_attn} attention bmm/softmax quantizers (SDPA stays fp16)")
    enabled_after = _enabled()
    print(f"[fp8] weight_quantizers enabled: {enabled_before} -> {enabled_after} "
          f"(disabled {n_disabled} quantizers across {len(disabled_names)} sensitive layers)")
    # mark so export_onnx/optimize_onnx preserve Q/DQ (no constant-folding, no onnxslim)
    try:
        module._fp8_quantized = True
    except Exception:
        pass
    return enabled_after, n_disabled, disabled_names


def build_sdxl_calib_inputs(pipe, opt_height, opt_width, device, dtype,
                            prompts=None, timesteps=None):
    """Build representative SDXL UNet calibration inputs for amax collection.

    Returns a list of (sample, timestep, encoder_hidden_states, text_embeds, time_ids) tuples matching
    SDXLUNetWrapper.forward — real encoded prompts × a spread of timesteps × random latents.
    """
    prompts = prompts or [
        "a vivid impressionist oil painting, bold brushstrokes, vibrant saturated color",
        "a photorealistic portrait, sharp focus, studio lighting",
        "a sprawling cyberpunk city at night, neon reflections, rain",
        "a serene watercolor landscape, soft gradients, misty mountains",
    ]
    timesteps = timesteps or [999.0, 800.0, 600.0, 400.0, 200.0, 20.0]
    lh, lw = opt_height // 8, opt_width // 8
    # standard SDXL micro-conditioning: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
    time_ids = torch.tensor([[opt_height, opt_width, 0, 0, opt_height, opt_width]],
                            device=device, dtype=dtype)

    conds = []  # (ehs[1,77,2048], pooled[1,1280])
    with torch.no_grad():
        for p in prompts:
            ehs, _neg, pooled, _negp = pipe.encode_prompt(
                prompt=p, prompt_2=None, device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=False)
            conds.append((ehs.to(dtype), pooled.to(dtype)))

    rng = torch.Generator(device=device).manual_seed(52)
    calib = []
    for (ehs, pooled) in conds:
        for t in timesteps:
            sample = torch.randn((1, 4, lh, lw), device=device, dtype=dtype, generator=rng)
            ts = torch.tensor([t], device=device, dtype=torch.float32)
            calib.append((sample, ts, ehs, pooled, time_ids))
    return calib
