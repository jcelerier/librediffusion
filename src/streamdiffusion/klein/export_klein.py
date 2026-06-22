"""FLUX.2-klein-4B ONNX export.

Exports three ONNX graphs from diffusers Flux2 modules (option (b): direct torch.onnx export — chosen
over extending demoDiffusion because Flux2 is a brand-new arch with no demo pipeline, and direct export
lets us pin the exact C++ I/O contract):

  transformer : hidden_states[B,Lp,128] bf16, encoder_hidden_states[B,Lt,7680] bf16, timestep[B] fp32,
                img_ids[B,Lp,4] fp32, txt_ids[B,Lt,4] fp32  ->  velocity[B,Lp,128] bf16
                (NO pooled, NO guidance. kv_cache=None.)
  vae_decoder : latent[B,32,h,w] bf16 -> image[B,3,H,W] fp32   (32-ch AutoencoderKLFlux2)
  vae_encoder : image[B,3,H,W] fp32 -> latent_dist mean[B,32,h,w] (argmax/mean used by pipeline)
  qwen3_encoder: handled separately (see note) — emits hidden layers (9,18,27).

The transformer is exported in bf16 (TRT will quantize to FP8 in a later build step via modelopt, OR
we build bf16 first for the correctness gate). VAE in bf16. Qwen in bf16.

Run under the flux venv with PYTHONPATH=klein-pydeps (git diffusers 0.39).
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch
import torch.nn.functional as F


def _rms_norm_decomposed(x, normalized_shape, weight=None, eps=None):
    """Manual RMSNorm so the ONNX exporter doesn't hit aten::rms_norm (unsupported in opset 20)."""
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    dims = tuple(range(-len(normalized_shape), 0))
    variance = x.to(torch.float32).pow(2).mean(dim=dims, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(weight.dtype) if weight is not None else x
    if weight is not None:
        x = x * weight
    return x


# Patch BEFORE any model import builds graphs.
F.rms_norm = _rms_norm_decomposed
torch.rms_norm = _rms_norm_decomposed
torch.nn.functional.rms_norm = _rms_norm_decomposed

# Path overrides (env) so train-lora-daydream.py --type klein can drive this without editing the script.
import os as _os


def _resolve_klein_model_dir():
    """Locate the FLUX.2-klein-4B snapshot. Honors KLEIN_MODEL_DIR, else looks it up in the HF cache
    (HF_HOME or the default ~/.cache/huggingface)."""
    explicit = _os.environ.get("KLEIN_MODEL_DIR")
    if explicit:
        return explicit
    hub = _os.path.join(_os.environ.get("HF_HOME", _os.path.expanduser("~/.cache/huggingface")), "hub")
    cands = glob.glob(_os.path.join(hub, "models--black-forest-labs--FLUX.2-klein-4B", "snapshots", "*"))
    if not cands:
        raise RuntimeError(
            "FLUX.2-klein-4B not found; set KLEIN_MODEL_DIR or download into the HF cache (HF_HOME)."
        )
    return cands[0]


KLEIN = _resolve_klein_model_dir()
OUT = Path(_os.environ.get("KLEIN_ONNX_DIR", "./onnx-klein"))
OUT.mkdir(parents=True, exist_ok=True)

# 320x576 target: latent 36x20 = 720 tokens (H/16, W/16). VAE latent 32ch at 72x40 (H/8,W/8).
W, H = 320, 576
LAT_H, LAT_W = H // 8, W // 8        # 72, 40  (vae latent spatial, 32ch)
TOK_H, TOK_W = LAT_H // 2, LAT_W // 2  # 36, 20 (after 2x2 patchify)
LP = TOK_H * TOK_W                    # 720
LT = 512                             # qwen seq
DEV = "cuda"
DT = torch.bfloat16


class TransformerWrapper(torch.nn.Module):
    """Fixes guidance=None / kv_cache=None, returns the velocity tensor only."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids):
        out = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            joint_attention_kwargs=None,
            return_dict=False,
            kv_cache=None,
            kv_cache_mode=None,
            num_ref_tokens=0,
        )
        return out[0]


def export_transformer():
    from diffusers import Flux2Transformer2DModel
    print("loading transformer...")
    m = Flux2Transformer2DModel.from_pretrained(KLEIN, subfolder="transformer", torch_dtype=DT).to(DEV).eval()
    w = TransformerWrapper(m).eval()

    hs = torch.randn(1, LP, 128, dtype=DT, device=DEV)
    ehs = torch.randn(1, LT, 7680, dtype=DT, device=DEV)
    ts = torch.tensor([1.0], dtype=torch.float32, device=DEV)
    img_ids = torch.zeros(1, LP, 4, dtype=torch.float32, device=DEV)
    txt_ids = torch.zeros(1, LT, 4, dtype=torch.float32, device=DEV)

    with torch.no_grad():
        v = w(hs, ehs, ts, img_ids, txt_ids)
    print("transformer test forward out:", v.shape, v.dtype, "std", float(v.float().std()))

    dst = OUT / "transformer"
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / "model.onnx"
    print("exporting transformer ONNX (external data)...")
    torch.onnx.export(
        w, (hs, ehs, ts, img_ids, txt_ids), str(path),
        input_names=["hidden_states", "encoder_hidden_states", "timestep", "img_ids", "txt_ids"],
        output_names=["velocity"],
        dynamic_axes={
            "hidden_states": {0: "B", 1: "Lp"},
            "encoder_hidden_states": {0: "B", 1: "Lt"},
            "timestep": {0: "B"},
            "img_ids": {0: "B", 1: "Lp"},
            "txt_ids": {0: "B", 1: "Lt"},
            "velocity": {0: "B", 1: "Lp"},
        },
        opset_version=20, do_constant_folding=False, dynamo=False,
    )
    print("transformer ONNX ->", path)


class VaeDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent, return_dict=False)[0]


class VaeEncoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        # mean of the latent dist (argmax/deterministic mode used by the pipeline)
        posterior = self.vae.encode(image).latent_dist
        return posterior.mode()


def export_vae():
    from diffusers import AutoencoderKLFlux2
    print("loading vae...")
    vae = AutoencoderKLFlux2.from_pretrained(KLEIN, subfolder="vae", torch_dtype=DT).to(DEV).eval()

    # decoder: latent [B,32,LAT_H,LAT_W] -> image
    dec = VaeDecoderWrapper(vae).eval()
    lat = torch.randn(1, 32, LAT_H, LAT_W, dtype=DT, device=DEV)
    with torch.no_grad():
        img = dec(lat)
    print("vae decode test:", img.shape, img.dtype)
    ddst = OUT / "vae_decoder"
    ddst.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        dec, (lat,), str(ddst / "model.onnx"),
        input_names=["latent"], output_names=["image"],
        dynamic_axes={"latent": {0: "B", 2: "h", 3: "w"}, "image": {0: "B", 2: "H", 3: "W"}},
        opset_version=20, do_constant_folding=False, dynamo=False,
    )
    print("vae_decoder ONNX done")

    # encoder: image [B,3,H,W] -> latent mean [B,32,LAT_H,LAT_W]
    enc = VaeEncoderWrapper(vae).eval()
    im = torch.randn(1, 3, H, W, dtype=DT, device=DEV)
    with torch.no_grad():
        z = enc(im)
    print("vae encode test:", z.shape, z.dtype)
    edst = OUT / "vae_encoder"
    edst.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        enc, (im,), str(edst / "model.onnx"),
        input_names=["image"], output_names=["latent"],
        dynamic_axes={"image": {0: "B", 2: "H", 3: "W"}, "latent": {0: "B", 2: "h", 3: "w"}},
        opset_version=20, do_constant_folding=False, dynamo=False,
    )
    print("vae_encoder ONNX done")


class Qwen3Wrapper(torch.nn.Module):
    """Emits the 3 stacked hidden layers already mapped to [B, Lt, 7680]."""
    LAYERS = (9, 18, 27)

    def __init__(self, te):
        super().__init__()
        self.te = te

    def forward(self, input_ids, attention_mask):
        out = self.te(input_ids=input_ids, attention_mask=attention_mask,
                      output_hidden_states=True, use_cache=False)
        stacked = torch.stack([out.hidden_states[k] for k in self.LAYERS], dim=1)
        b, nc, sl, hd = stacked.shape
        return stacked.permute(0, 2, 1, 3).reshape(b, sl, nc * hd)


def export_qwen():
    from transformers import Qwen3ForCausalLM
    print("loading qwen3...")
    # attn_implementation="eager": force the plain additive-mask attention path. The SDPA/flash mask
    # builders (transformers >=4.56 use torch.vmap; >=5.x use a fused mask) either (a) can't be traced
    # by the legacy TorchScript ONNX exporter (RuntimeError: unordered_map::at on vmap) or (b) bake a
    # mask that yields NaN through the bf16 TRT engine. Eager masking exports cleanly and stays finite.
    te = Qwen3ForCausalLM.from_pretrained(KLEIN, subfolder="text_encoder", torch_dtype=DT,
                                          attn_implementation="eager").to(DEV).eval()
    w = Qwen3Wrapper(te).eval()
    ids = torch.ones(1, LT, dtype=torch.int64, device=DEV)
    mask = torch.ones(1, LT, dtype=torch.int64, device=DEV)
    with torch.no_grad():
        e = w(ids, mask)
    print("qwen test:", e.shape, e.dtype, "std", float(e.float().std()))
    qdst = OUT / "qwen3_encoder"
    qdst.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        w, (ids, mask), str(qdst / "model.onnx"),
        input_names=["input_ids", "attention_mask"], output_names=["encoder_hidden_states"],
        dynamic_axes={"input_ids": {0: "B", 1: "Lt"}, "attention_mask": {0: "B", 1: "Lt"},
                      "encoder_hidden_states": {0: "B", 1: "Lt"}},
        opset_version=20, do_constant_folding=False, dynamo=False,
    )
    print("qwen3_encoder ONNX done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["all", "transformer", "vae", "qwen"])
    a = ap.parse_args()
    if a.which in ("all", "transformer"):
        export_transformer()
    if a.which in ("all", "vae"):
        export_vae()
    if a.which in ("all", "qwen"):
        export_qwen()
    print("DONE")
