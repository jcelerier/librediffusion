"""FP8 (e4m3) FLUX.3-image-klein transformer: same I/O as export_flux3.py's, so the C++ side is unchanged.

Klein.2's diffusion FP8 recipe (modelopt FP8_DEFAULT_CFG, bf16 dequant path): every compute Linear of
the per-mode and single blocks is FP8; the embedders, time MLPs, AdaLN modulations and final layer
stay bf16. amax comes from real sampling traces (make_flux3_golden.py --calib).

  python export_flux3_fp8.py --calib /media/data1/flux3/calib [--check /media/data1/flux3/golden/t2i_512_s4,...]
Writes $FLUX3_ONNX_DIR/transformer_fp8/model.onnx.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "acceleration" / "tensorrt"))
import export_flux3 as ex  # noqa: E402
from fp8_quantize import quantize_unet_fp8  # noqa: E402

SENSITIVE = re.compile(r".*(emb_in|txt_in|time_in|modulations|final_layer).*")


def case_inputs(d: Path):
    """One DenseDiT argument tuple per recorded step of a golden/calib case."""
    g = lambda n: torch.from_numpy(np.load(d / f"{n}.npy")).to(ex.DEV)
    has_cond = (d / "cond_tokens.npy").exists()
    cond = g("cond_tokens")[None].to(ex.DT) if has_cond else torch.zeros(1, 0, 128, device=ex.DEV, dtype=ex.DT)
    rope_cond = ex.rope_table(g("cond_ids") if has_cond else torch.zeros(0, 4, device=ex.DEV))
    txt = g("txt_tokens")[None].to(ex.DT)
    rope_img, rope_txt = ex.rope_table(g("img_ids")), ex.rope_table(g("txt_ids"))
    sched = g("schedule").float()
    out = []
    for i in range(len(sched) - 1):
        x = g(f"step{i}_x")[None].to(ex.DT)
        out.append(((x, cond, txt, rope_img, rope_cond, rope_txt, sched[i:i + 1], sched[i + 1:i + 2]),
                    g(f"step{i}_v")[None].float()))
    return out


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm()))


def check(dense, dirs, tag):
    with torch.inference_mode():
        for d in dirs:
            cs = [cos(dense(*args).float(), ref) for args, ref in case_inputs(d)]
            print(f"[{tag}] {d.name}: velocity cos min {min(cs):.5f} mean {np.mean(cs):.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True, type=Path, help="dir of make_flux3_golden.py --calib cases")
    ap.add_argument("--check", default="", help="comma-separated held-out case dirs: eager FP8 velocity check")
    a = ap.parse_args()
    calib_dirs = sorted(p for p in a.calib.iterdir() if (p / "schedule.npy").exists())
    if not calib_dirs:
        raise SystemExit(f"no calibration cases under {a.calib}")
    check_dirs = [Path(p) for p in a.check.split(",") if p]

    dense = ex.DenseDiT(ex.load_dit(ex.snapshot())).eval()
    calib = [args for d in calib_dirs for args, _ in case_inputs(d)]
    quantize_unet_fp8(dense, calib, high_precision="BFloat16", sensitive=SENSITIVE)
    del calib
    torch.cuda.empty_cache()
    if check_dirs:
        check(dense, check_dirs, "fp8 eager")

    args = (torch.randn(1, ex.LI, 128, device=ex.DEV, dtype=ex.DT),
            torch.randn(1, ex.LI, 128, device=ex.DEV, dtype=ex.DT),
            torch.randn(1, ex.LT, 20480, device=ex.DEV, dtype=ex.DT),
            torch.randn(1, ex.LI, 64, 2, 2, device=ex.DEV), torch.randn(1, ex.LI, 64, 2, 2, device=ex.DEV),
            torch.randn(1, ex.LT, 64, 2, 2, device=ex.DEV),
            torch.tensor([0.9], device=ex.DEV), torch.tensor([0.8], device=ex.DEV))
    names = ["img", "cond", "txt", "rope_img", "rope_cond", "rope_txt", "t", "r"]
    dst = ex.OUT / "transformer_fp8"
    dst.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            dense, args, str(dst / "model.onnx"), input_names=names, output_names=["velocity"],
            dynamic_axes={"img": {1: "Li"}, "cond": {1: "Lc"}, "txt": {1: "Lt"}, "rope_img": {1: "Li"},
                          "rope_cond": {1: "Lc"}, "rope_txt": {1: "Lt"}, "velocity": {1: "Li"}},
            opset_version=20, do_constant_folding=False, dynamo=False)
    print("transformer FP8 ONNX ->", dst / "model.onnx")


if __name__ == "__main__":
    main()
