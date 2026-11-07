"""Build FLUX.3-image-klein bf16 TRT engines from export_flux3.py's ONNX.

Profiles (tokens = (H/16)*(W/16)): image MIN_PX..MAX_PX per side, references 0 (text-to-image) up
to one MAX_PX square in total, text 80..640 tokens. --which transformer_fp8 builds
transformer_fp8.plan from export_flux3_fp8.py's ONNX.
Env: FLUX3_ONNX_DIR, FLUX3_ENGINE_DIR, FLUX3_MIN_PX (128), FLUX3_MAX_PX (1024),
FLUX3_OPT_WIDTH / FLUX3_OPT_HEIGHT (FLUX3_OPT_SIZE, 512).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "klein"))
os.environ.setdefault("KLEIN_ENGINE_DIR", os.environ.get("FLUX3_ENGINE_DIR", "./engine-flux3"))
from build_klein_engines import build  # noqa: E402

ONNX = Path(os.environ.get("FLUX3_ONNX_DIR", "./onnx-flux3"))
ENG = Path(os.environ.get("FLUX3_ENGINE_DIR", "./engine-flux3"))
OPT = int(os.environ.get("FLUX3_OPT_SIZE", "512"))
OPT_W = int(os.environ.get("FLUX3_OPT_WIDTH", OPT)) // 16 * 16
OPT_H = int(os.environ.get("FLUX3_OPT_HEIGHT", OPT)) // 16 * 16
PX_MIN = int(os.environ.get("FLUX3_MIN_PX", "128")) // 16 * 16
PX_MAX = int(os.environ.get("FLUX3_MAX_PX", "1024")) // 16 * 16
if not PX_MIN <= min(OPT_W, OPT_H) <= max(OPT_W, OPT_H) <= PX_MAX:
    raise SystemExit(f"opt {OPT_W}x{OPT_H} outside {PX_MIN}..{PX_MAX}")
L_MIN, L_OPT, L_MAX = (PX_MIN // 16) ** 2, (OPT_W // 16) * (OPT_H // 16), (PX_MAX // 16) ** 2
T_MIN, T_OPT, T_MAX = 80, 80, 640


def build_transformer(name="transformer.plan", src="transformer", lc_min=0):
    lc_opt = L_OPT
    p = {
        "img": ((1, L_MIN, 128), (1, L_OPT, 128), (1, L_MAX, 128)),
        "cond": ((1, lc_min, 128), (1, lc_opt, 128), (1, L_MAX, 128)),
        "txt": ((1, T_MIN, 20480), (1, T_OPT, 20480), (1, T_MAX, 20480)),
        "rope_img": ((1, L_MIN, 64, 2, 2), (1, L_OPT, 64, 2, 2), (1, L_MAX, 64, 2, 2)),
        "rope_cond": ((1, lc_min, 64, 2, 2), (1, lc_opt, 64, 2, 2), (1, L_MAX, 64, 2, 2)),
        "rope_txt": ((1, T_MIN, 64, 2, 2), (1, T_OPT, 64, 2, 2), (1, T_MAX, 64, 2, 2)),
        "t": ((1,),) * 3,
        "r": ((1,),) * 3,
    }
    build(ONNX / src / "model.onnx", ENG / name, p, workspace_gb=8)


def build_qwen():
    src = ONNX / "qwen3vl_encoder"
    fixed = src / "model_fixed.onnx"
    if not fixed.exists():
        # same legacy-exporter COMPLEX128 cast artifact as klein.2's Qwen3
        tmp = ONNX / "_qwenfix"
        (tmp / "qwen3_encoder").mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            link = tmp / "qwen3_encoder" / f.name
            if not link.exists():
                link.symlink_to(f.resolve())
        subprocess.check_call([sys.executable, str(Path(__file__).resolve().parent.parent / "klein/fix_qwen_complex.py")],
                              env={**os.environ, "KLEIN_ONNX_DIR": str(tmp)})
        for f in ("model_fixed.onnx", "model_fixed.onnx_data"):
            shutil.move(tmp / "qwen3_encoder" / f, src / f)
        shutil.rmtree(tmp)
    p = {"input_ids": ((1, T_MIN), (1, T_OPT), (1, T_MAX)),
         "attention_mask": ((1, T_MIN), (1, T_OPT), (1, T_MAX))}
    build(fixed, ENG / "qwen.plan", p, workspace_gb=8)


def build_vae():
    lo, hi = PX_MIN // 8, PX_MAX // 8
    build(ONNX / "vae_decoder/model.onnx", ENG / "vae_decoder.plan",
          {"latent": ((1, 32, lo, lo), (1, 32, OPT_H // 8, OPT_W // 8), (1, 32, hi, hi))}, workspace_gb=8)
    build(ONNX / "vae_encoder/model.onnx", ENG / "vae_encoder.plan",
          {"image": ((1, 3, PX_MIN, PX_MIN), (1, 3, OPT_H, OPT_W), (1, 3, PX_MAX, PX_MAX))}, workspace_gb=8)
    for f in ("bn_mean.bin", "bn_std.bin", "tokenizer.json"):
        shutil.copy(ONNX / f, ENG / f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["all", "transformer", "transformer_fp8", "qwen", "vae"])
    a = ap.parse_args()
    ENG.mkdir(parents=True, exist_ok=True)
    if a.which in ("all", "vae"):
        build_vae()
    if a.which in ("all", "qwen"):
        build_qwen()
    if a.which in ("all", "transformer"):
        build_transformer()
    if a.which == "transformer_fp8":
        # export_flux3_fp8.py's ONNX; Q/DQ makes the strongly-typed build FP8 (needs SM 8.9+)
        build_transformer("transformer_fp8.plan", "transformer_fp8")
