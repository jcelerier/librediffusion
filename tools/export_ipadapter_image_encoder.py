"""Export the on-device IP-Adapter image-conditioning engines (clip_image_encoder + ip_image_proj).

These two engines let the C++ node turn a raw style image into IP-Adapter tokens with no Python:

  1. clip_image_encoder.engine
       CLIPVisionModelWithProjection (CLIP ViT-H/14) from h94/IP-Adapter/models/image_encoder.
       input : pixel_values  [B,3,224,224] fp16  (CLIP-normalized on-device by a CUDA kernel)
       output: image_embeds  [B,1024]      fp16

  2. ip_image_proj.engine
       ImageProjModel from the IP-Adapter checkpoint (image_proj.*):
       Linear(1024 -> num_tokens*cross_attn_dim) + LayerNorm(cross_attn_dim), reshaped to
       [B, num_tokens, cross_attn_dim].
       input : image_embeds  [B,1024]                fp16
       output: ip_tokens     [B,num_tokens,cad]      fp16

    cross_attn_dim = 768 for SD1.5, 2048 for SDXL. --type selects the checkpoint + dim (SDXL uses the
    SAME ViT-H/14 image encoder as SD1.5, so only the projection + checkpoint differ).

The negative tokens are the projection of ZEROS — the C++ runs the SAME ip_image_proj engine on a
[B,1024] zero buffer (encoder is NOT re-run), matching get_image_embeds() for the base IP-Adapter.

TRT 11 is strongly-typed: the ONNX must already be fp16 (no global fp16 builder flag). We export the
modules in fp16 and reuse Engine.build() from the fork's TRT utilities (timing cache, opt level, etc.).

Usage:
    source env.sh
    uv run python tools/export_ipadapter_image_encoder.py --type sdxl \
        --out ./ip-engines
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection

# The TRT utilities live in the librediffusion src/ tree (repo root is this file's parent.parent).
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamdiffusion.acceleration.tensorrt.utilities import Engine  # noqa: E402

CLIP_EMBED_DIM = 1024  # ViT-H/14 projection dim (image_embeds), same for SD1.5 + SDXL


def _ip_snapshot_dir() -> str:
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cands = glob.glob(os.path.join(hf, "hub/models--h94--IP-Adapter/snapshots/*"))
    if not cands:
        raise RuntimeError("h94/IP-Adapter not in HF cache (set HF_HOME)")
    return cands[0]


def _defaults(model_type: str):
    snap = _ip_snapshot_dir()
    encoder = f"{snap}/models/image_encoder"  # ViT-H/14, shared across SD1.5 + SDXL
    if model_type == "sdxl":
        ckpt = f"{snap}/sdxl_models/ip-adapter_sdxl_vit-h.bin"
        cross_attn_dim = 2048
    else:
        ckpt = f"{snap}/models/ip-adapter_sd15.bin"
        cross_attn_dim = 768
    return encoder, ckpt, cross_attn_dim


class ImageProjModel(nn.Module):
    """ImageProjModel from diffusers_ipadapter (base / non-plus): Linear + reshape + LayerNorm."""

    def __init__(self, cross_attention_dim=768, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        x = self.proj(image_embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        return self.norm(x)


def export_clip_image_encoder(onnx_dir: Path, encoder_path: str) -> Path:
    enc = CLIPVisionModelWithProjection.from_pretrained(encoder_path).eval().half().cuda()

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values).image_embeds

    wrap = Wrap(enc).eval()
    sample = torch.randn(1, 3, 224, 224, dtype=torch.float16, device="cuda")
    onnx_path = onnx_dir / "clip_image_encoder.onnx"
    torch.onnx.export(
        wrap, (sample,), str(onnx_path),
        input_names=["pixel_values"], output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "B"}, "image_embeds": {0: "B"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    with torch.no_grad():
        ref = wrap(sample)
    print(f"  clip_image_encoder ONNX -> {onnx_path}  out {tuple(ref.shape)} {ref.dtype}")
    return onnx_path


def export_ip_image_proj(onnx_dir: Path, ckpt: str, cross_attn_dim: int) -> tuple[Path, int]:
    state = torch.load(ckpt, map_location="cpu")["image_proj"]
    # num_tokens inferred from the projection weight: proj.weight is [num_tokens*cad, 1024].
    num_tokens = state["proj.weight"].shape[0] // cross_attn_dim
    proj = ImageProjModel(cross_attn_dim, CLIP_EMBED_DIM, num_tokens)
    proj.load_state_dict(state)
    proj = proj.eval().half().cuda()
    sample = torch.randn(1, CLIP_EMBED_DIM, dtype=torch.float16, device="cuda")
    onnx_path = onnx_dir / "ip_image_proj.onnx"
    torch.onnx.export(
        proj, (sample,), str(onnx_path),
        input_names=["image_embeds"], output_names=["ip_tokens"],
        dynamic_axes={"image_embeds": {0: "B"}, "ip_tokens": {0: "B"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    with torch.no_grad():
        ref = proj(sample)
    print(f"  ip_image_proj ONNX -> {onnx_path}  out {tuple(ref.shape)} {ref.dtype} "
          f"(num_tokens={num_tokens}, cross_attn_dim={cross_attn_dim})")
    return onnx_path, num_tokens


def build_engine(onnx_path: Path, engine_path: Path, input_profile: dict):
    eng = Engine(str(engine_path))
    eng.build(
        str(onnx_path), fp16=True, input_profile=input_profile,
        timing_cache=str(engine_path.with_suffix(".timing.cache")),
    )
    print(f"  built -> {engine_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["sd15", "sdxl"], default="sdxl")
    ap.add_argument("--out", default="./ip-engines")
    ap.add_argument("--encoder", default=None, help="CLIP image encoder dir (default: h94 ViT-H/14)")
    ap.add_argument("--ckpt", default=None, help="IP-Adapter checkpoint (default per --type)")
    ap.add_argument("--max-batch", type=int, default=2)
    args = ap.parse_args()

    enc_default, ckpt_default, cross_attn_dim = _defaults(args.type)
    encoder = args.encoder or enc_default
    ckpt = args.ckpt or ckpt_default

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    onnx_dir = out.parent / "engines_onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting IP image path ({args.type}): encoder={encoder} ckpt={ckpt}")
    enc_onnx = export_clip_image_encoder(onnx_dir, encoder)
    proj_onnx, _num_tokens = export_ip_image_proj(onnx_dir, ckpt, cross_attn_dim)

    print("Building TRT engines ...")
    mb = max(1, args.max_batch)
    build_engine(
        enc_onnx, out / "clip_image_encoder.engine",
        {"pixel_values": [(1, 3, 224, 224), (1, 3, 224, 224), (mb, 3, 224, 224)]},
    )
    build_engine(
        proj_onnx, out / "ip_image_proj.engine",
        {"image_embeds": [(1, CLIP_EMBED_DIM), (1, CLIP_EMBED_DIM), (mb, CLIP_EMBED_DIM)]},
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
