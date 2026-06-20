"""FP8-quantize the klein transformer with the PROPER diffusion FP8 recipe (quality fix).

Root cause of the prior ~0.97 step-1 velocity / 19dB e2e: the prior script used plain
modelopt.FP8_DEFAULT_CFG, which FP8-quantizes EVERY Linear including the precision-sensitive
embedders / AdaLN modulation / norm_out / proj_out. demoDiffusion's FLUX FP8 recipe keeps those in
high precision (SD_FP8_BF16_DEFAULT_CONFIG + filter_func_no_proj_out + quant_level), quantizing only the
heavy attention/FF/single-block compute linears, and tags trt_high_precision_dtype="BFloat16".

This script:
  - Calibrates amax over real golden activations (step0+step1) + a few random timesteps (range widening).
    (The amax pass is correct regardless of the CUDA ext: modelopt falls back to fp8_eager which is an
     exact torch.float8_e4m3fn fake-quant; the "simulated" warning is about speed, not correctness.)
  - Uses SD_FP8_BF16_DEFAULT_CONFIG (trt_high_precision_dtype BFloat16, weight/input quantizers).
  - After quantize, DISABLES quantizers on sensitive layers (modulation, embedders, norm_out, proj_out,
    time embedder) so they run bf16 in the engine. Disable BEFORE export so amax isn't baked in.
  - Exports FP8 ONNX (rms_norm decompose, do_constant_folding=False, external data, opset20, dynamo=False).

Output: $KLEIN_ONNX_DIR/transformer_fp8_calib/model.onnx
"""
from __future__ import annotations

import copy
import glob
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _rms_norm_decomposed(x, normalized_shape, weight=None, eps=None):
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    dims = tuple(range(-len(normalized_shape), 0))
    variance = x.to(torch.float32).pow(2).mean(dim=dims, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(weight.dtype) if weight is not None else x
    if weight is not None:
        x = x * weight
    return x


F.rms_norm = _rms_norm_decomposed
torch.rms_norm = _rms_norm_decomposed
torch.nn.functional.rms_norm = _rms_norm_decomposed

import os as _os0


def _resolve_klein_model_dir():
    """Locate the FLUX.2-klein-4B snapshot. Honors KLEIN_MODEL_DIR, else looks it up in the HF cache
    (HF_HOME or the default ~/.cache/huggingface)."""
    explicit = _os0.environ.get("KLEIN_MODEL_DIR")
    if explicit:
        return explicit
    hub = _os0.path.join(_os0.environ.get("HF_HOME", _os0.path.expanduser("~/.cache/huggingface")), "hub")
    cands = glob.glob(_os0.path.join(hub, "models--black-forest-labs--FLUX.2-klein-4B", "snapshots", "*"))
    if not cands:
        raise RuntimeError(
            "FLUX.2-klein-4B not found; set KLEIN_MODEL_DIR or download into the HF cache (HF_HOME)."
        )
    return cands[0]


KLEIN = _resolve_klein_model_dir()
# Calibration goldens dir (npy tensors). Required for this script; set KLEIN_FP8_GOLD to point at it.
GOLD = _os0.environ.get("KLEIN_FP8_GOLD", "")
_SUF = _os0.environ.get("KLEIN_FP8_LEVEL", "all")
_SUF = "" if _SUF == "all" else "_" + _SUF
# honor KLEIN_ONNX_DIR (train-lora --type klein); standalone unchanged.
_ONNX_BASE = _os0.environ.get("KLEIN_ONNX_DIR", "./onnx-klein")
OUT = Path(f"{_ONNX_BASE}/transformer_fp8_calib{_SUF}")
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda"
DT = torch.bfloat16
LP, LT = 720, 512

# Sensitive layers that must stay high-precision (bf16) in the engine.
# Matches klein module names dumped from the model (no diffusers Attention bmm quantizers here:
# attention is plain Linear q/k/v/out + SDPA, SDPA stays bf16).
# QUANT_LEVEL controls which Linears get FP8 (mirrors demoDiffusion quant_level semantics):
#   "all"  -> FP8 on every compute Linear (attn q/k/v/out + ff + single-block fused)  [most aggressive]
#   "noqkv"-> additionally keep attention q/k/v/(add_q/k/v) projections in bf16        [quality-leaning]
#   "ff"   -> FP8 only on FF / output projections; all attention projections bf16      [most conservative]
import os as _os
QUANT_LEVEL = _os.environ.get("KLEIN_FP8_LEVEL", "all")

_BASE_SENSITIVE = (
    r"time_guidance_embed|timestep_embedder|"
    r"x_embedder|context_embedder|"
    r"modulation|norm_out|proj_out"
)
_QKV = r"|to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj"
_ATTN = _QKV + r"|to_out|to_add_out|to_qkv_mlp_proj"
if QUANT_LEVEL == "ff":
    SENSITIVE = re.compile(r".*(" + _BASE_SENSITIVE + _ATTN + r").*")
elif QUANT_LEVEL == "noqkv":
    SENSITIVE = re.compile(r".*(" + _BASE_SENSITIVE + _QKV + r").*")
else:
    SENSITIVE = re.compile(r".*(" + _BASE_SENSITIVE + r").*")
print("KLEIN_FP8_LEVEL =", QUANT_LEVEL)

# Use FP8_DEFAULT_CFG for calibration (it reliably enables + collects amax on 108/109 Linears in this
# modelopt build; the dict-form SD_FP8_BF16 config's *weight_quantizer glob does NOT match here and
# leaves everything disabled -> no amax -> a no-op bf16 engine). We then disable the sensitive layers.
def _fp8_bf16_cfg():
    import modelopt.torch.quantization as mtq
    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    # tag the high-precision (dequant) path as BFloat16 so TRT keeps surrounding compute in bf16
    for rule in cfg["quant_cfg"]:
        c = rule.get("cfg")
        if c is not None and "num_bits" in c:
            c["trt_high_precision_dtype"] = "BFloat16"
    return cfg


class TransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids):
        return self.model(
            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
            timestep=timestep, img_ids=img_ids, txt_ids=txt_ids,
            guidance=None, joint_attention_kwargs=None, return_dict=False,
            kv_cache=None, kv_cache_mode=None, num_ref_tokens=0,
        )[0]


def main():
    import modelopt.torch.quantization as mtq
    from diffusers import Flux2Transformer2DModel

    def gnp(n):
        if not GOLD:
            raise RuntimeError("FP8 calibration goldens required; set KLEIN_FP8_GOLD to the goldens dir.")
        return np.load(f"{GOLD}/{n}.npy")

    print("loading transformer...")
    m = Flux2Transformer2DModel.from_pretrained(KLEIN, subfolder="transformer", torch_dtype=DT).to(DEV).eval()
    w = TransformerWrapper(m).eval()

    ehs = torch.from_numpy(gnp("005_textencode__encoder_hidden_states_7680")).to(DEV, DT)
    img_ids = torch.from_numpy(gnp("031_ids__img_ids")).to(DEV, torch.float32)
    txt_ids = torch.from_numpy(gnp("030_ids__txt_ids")).to(DEV, torch.float32)
    sigmas = gnp("032_ids__timesteps_per_step")
    calib_inputs = []
    for step in range(2):
        in_h = torch.from_numpy(gnp(f"11{step}_transformer__in_hidden_step{step}")).to(DEV, DT)
        ts = torch.tensor([sigmas[step]], dtype=torch.float32, device=DEV)
        calib_inputs.append((in_h, ehs, ts, img_ids, txt_ids))
    # widen amax range with a few extra timesteps along the trajectory + a noise-scale variation
    rng = np.random.default_rng(52)
    for tval in (0.95, 0.65, 0.35):
        in_h = torch.from_numpy(gnp("110_transformer__in_hidden_step0")).to(DEV, DT)
        in_h = in_h + 0.05 * torch.randn_like(in_h)
        ts = torch.tensor([float(tval)], dtype=torch.float32, device=DEV)
        calib_inputs.append((in_h, ehs, ts, img_ids, txt_ids))

    def forward_loop(model):
        with torch.no_grad():
            for ins in calib_inputs:
                model(*ins)

    print("FP8 calibrating (FP8_DEFAULT_CFG + BF16 high-precision)...")
    mtq.quantize(w, _fp8_bf16_cfg(), forward_loop)
    print("FP8 quantize done")

    enabled_before = sum(1 for _, mod in w.named_modules()
                         if getattr(getattr(mod, "weight_quantizer", None), "is_enabled", False))
    # Disable quantizers on sensitive layers (embedders, AdaLN modulation, norm_out, proj_out, time
    # embedder) so they run bf16 in the engine -- this is the diffusion FP8 best-practice that the
    # prior plain-FP8_DEFAULT_CFG run omitted.
    n_disabled = 0
    disabled_names = []
    for name, module in w.named_modules():
        if not isinstance(module, torch.nn.Linear) or not SENSITIVE.match(name):
            continue
        for qn in ("input_quantizer", "weight_quantizer", "output_quantizer"):
            q = getattr(module, qn, None)
            if q is not None and getattr(q, "is_enabled", False):
                q.disable()
                n_disabled += 1
        disabled_names.append(name)
    enabled_after = sum(1 for _, mod in w.named_modules()
                        if getattr(getattr(mod, "weight_quantizer", None), "is_enabled", False))
    print(f"weight_quantizers enabled: {enabled_before} -> {enabled_after} after disabling sensitive")
    print(f"disabled {n_disabled} quantizers across {len(disabled_names)} sensitive Linears:")
    for n in disabled_names:
        print("   -", n)

    path = OUT / "model.onnx"
    print("exporting FP8 ONNX...")
    hs = calib_inputs[0][0]
    torch.onnx.export(
        w, (hs, ehs, calib_inputs[0][2], img_ids, txt_ids), str(path),
        input_names=["hidden_states", "encoder_hidden_states", "timestep", "img_ids", "txt_ids"],
        output_names=["velocity"],
        dynamic_axes={
            "hidden_states": {0: "B", 1: "Lp"}, "encoder_hidden_states": {0: "B", 1: "Lt"},
            "timestep": {0: "B"}, "img_ids": {0: "B", 1: "Lp"}, "txt_ids": {0: "B", 1: "Lt"},
            "velocity": {0: "B", 1: "Lp"},
        },
        opset_version=20, do_constant_folding=False, dynamo=False,
    )
    print("FP8 ONNX ->", path)


if __name__ == "__main__":
    main()
