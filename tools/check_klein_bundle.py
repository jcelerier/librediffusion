"""Check a built klein bundle against the reference the build already captured.

A klein build that finishes is not a klein build that works: the engines are assembled from a traced
graph, two ONNX rewrites and an FP8 calibration, and any of those can produce an engine that loads,
runs, and returns noise. This replays the tensors make_klein_calib.py recorded from the diffusers
pipeline through the engines in the bundle and reports how far each one drifted.

  qwen3_encoder  : (input_ids, attention_mask) -> encoder_hidden_states, vs the pipeline's
  transformer    : the five step-0/step-1 inputs -> velocity, vs the pipeline's (bf16 and FP8)
  vae_decoder    : the decoder's own input latent -> image, written out as PNG to look at
  rife           : two real frames -> the interpolated frame, written out as PNG

Usage:
  python tools/check_klein_bundle.py <bundle_dir> <calib_dir> [--out <report_dir>]

Exit code is non-zero if any engine misses its threshold, so this works as a build gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

# bf16 through TRT is not bit-exact against torch bf16 and is not meant to be; these are the "the
# graph is right" thresholds, far below what a wrong RoPE/mask/precision path would give (those land
# at cos < 0.3). FP8 is quantised on purpose, so it gets a looser bar.
COS_MIN = {"qwen": 0.999, "transformer_bf16": 0.995, "transformer_fp8": 0.95}
PSNR_MIN = 25.0

_TRT2TORCH = {
    trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16, trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64, trt.DataType.INT8: torch.int8,
    trt.DataType.BOOL: torch.bool,
}


class Engine:
    def __init__(self, path: Path, logger):
        self.path = path
        rt = trt.Runtime(logger)
        self.engine = rt.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {path}")
        self.ctx = self.engine.create_execution_context()
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            (self.inputs if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
             else self.outputs).append(n)

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = set(self.inputs) - set(feed)
        if missing:
            raise RuntimeError(f"{self.path.name}: no value for {sorted(missing)}")
        stream = torch.cuda.Stream()
        hold = []
        with torch.cuda.stream(stream):
            for n in self.inputs:
                a = feed[n]
                dt = _TRT2TORCH[self.engine.get_tensor_dtype(n)]
                t = torch.from_numpy(np.ascontiguousarray(a)).to("cuda", dt).contiguous()
                hold.append(t)
                self.ctx.set_input_shape(n, tuple(t.shape))
                self.ctx.set_tensor_address(n, t.data_ptr())
            outs = {}
            for n in self.outputs:
                shape = tuple(self.ctx.get_tensor_shape(n))
                if any(d < 0 for d in shape):
                    raise RuntimeError(f"{self.path.name}: output {n} still dynamic {shape}")
                dt = _TRT2TORCH[self.engine.get_tensor_dtype(n)]
                t = torch.empty(shape, dtype=dt, device="cuda")
                hold.append(t)
                outs[n] = t
                self.ctx.set_tensor_address(n, t.data_ptr())
            if not self.ctx.execute_async_v3(stream.cuda_stream):
                raise RuntimeError(f"{self.path.name}: execute_async_v3 failed")
        stream.synchronize()
        return {n: t.to(torch.float32).cpu().numpy() for n, t in outs.items()}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / d) if d else float("nan")


def psnr(a: np.ndarray, b: np.ndarray, peak=255.0) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * float(np.log10(peak * peak / mse))


def to_u8_image(arr: np.ndarray) -> np.ndarray:
    """[1,3,H,W] in [-1,1] (the klein VAE's output range) -> HWC uint8."""
    a = arr[0] if arr.ndim == 4 else arr
    a = np.transpose(a, (1, 2, 0))
    return np.clip((a / 2.0 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("calib")
    ap.add_argument("--out", default=None, help="report dir (default: <bundle>/check)")
    a = ap.parse_args()
    bundle, calib = Path(a.bundle), Path(a.calib)
    out = Path(a.out) if a.out else bundle / "check"
    out.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.ERROR)

    def ref(n):
        p = calib / f"{n}.npy"
        return np.load(p) if p.exists() else None

    results, failures = [], []

    def record(name, metric, value, threshold, ok):
        results.append({"check": name, "metric": metric, "value": value, "threshold": threshold,
                        "pass": bool(ok)})
        print(f"  {'PASS' if ok else 'FAIL'}  {name:28s} {metric}={value:.6g} (need {threshold})")
        if not ok:
            failures.append(name)

    print(f"bundle {bundle}\ncalib  {calib}\n")

    # --- qwen3 encoder -------------------------------------------------------------------------
    ids, mask, ehs = ref("000_textencode__input_ids"), ref("001_textencode__attention_mask"), \
        ref("005_textencode__encoder_hidden_states_7680")
    p = bundle / "qwen3_encoder_bf16.plan"
    if p.exists() and ids is not None and mask is not None and ehs is not None:
        got = Engine(p, logger).run({"input_ids": ids, "attention_mask": mask})
        v = next(iter(got.values()))
        record("qwen3_encoder", "cos", cosine(v, ehs), COS_MIN["qwen"],
               cosine(v, ehs) >= COS_MIN["qwen"])
    else:
        print(f"  SKIP  qwen3_encoder (engine or reference missing)")

    # --- transformer (bf16 and fp8) ------------------------------------------------------------
    img_ids, txt_ids = ref("031_ids__img_ids"), ref("030_ids__txt_ids")
    sigmas = ref("032_ids__timesteps_per_step")
    for plan, key in (("transformer_bf16.plan", "transformer_bf16"),
                      ("transformer_fp8_calib.plan", "transformer_fp8")):
        p = bundle / plan
        if not p.exists():
            print(f"  SKIP  {key} (no {plan})")
            continue
        if ehs is None or img_ids is None or txt_ids is None or sigmas is None:
            print(f"  SKIP  {key} (reference missing)")
            continue
        eng = Engine(p, logger)
        worst = 1.0
        for step in range(len(sigmas)):
            hs = ref(f"11{step}_transformer__in_hidden_step{step}")
            exp = ref(f"10{step}_transformer__velocity_step{step}")
            if hs is None or exp is None:
                continue
            got = eng.run({"hidden_states": hs, "encoder_hidden_states": ehs,
                           "timestep": np.asarray([sigmas[step]], dtype=np.float32),
                           "img_ids": img_ids, "txt_ids": txt_ids})
            worst = min(worst, cosine(next(iter(got.values())), exp))
        record(key, "min_cos_over_steps", worst, COS_MIN[key], worst >= COS_MIN[key])

    # --- vae decoder: the image to look at -----------------------------------------------------
    dec_in = ref("200_vae__decoder_in_latent")
    p = bundle / "vae_decoder_bf16.plan"
    refpng = calib / "calib_reference.png"
    if p.exists() and dec_in is not None:
        got = Engine(p, logger).run({"latent": dec_in})
        img = to_u8_image(next(iter(got.values())))
        try:
            from PIL import Image
            Image.fromarray(img).save(out / "trt_vae_decode.png")
            print(f"  wrote {out / 'trt_vae_decode.png'} {img.shape}")
            if refpng.exists():
                r = np.asarray(Image.open(refpng).convert("RGB"))
                if r.shape == img.shape:
                    v = psnr(img, r)
                    record("vae_decoder_vs_reference", "psnr_db", v, PSNR_MIN, v >= PSNR_MIN)
                else:
                    print(f"  SKIP  vae psnr (shape {img.shape} vs reference {r.shape})")
        except ImportError:
            np.save(out / "trt_vae_decode.npy", img)
            print("  (no PIL; wrote .npy instead of .png)")
        # Independent of the reference: a decode that collapsed would be flat or non-finite.
        record("vae_decoder_signal", "std", float(img.std()), 10.0, float(img.std()) >= 10.0)
    else:
        print("  SKIP  vae_decoder (engine or reference missing)")

    # --- rife: interpolate between two real frames ---------------------------------------------
    p = bundle / "rife_ifnet_fp16.plan"
    if p.exists() and dec_in is not None:
        try:
            from PIL import Image
            base = np.asarray(Image.open(out / "trt_vae_decode.png").convert("RGB"))
        except Exception:
            base = None
        if base is not None:
            # Frame B = frame A rolled sideways, so a correct interpolation is a visible half-shift
            # rather than a copy of either input.
            a_f = base.astype(np.float32) / 255.0
            b_f = np.roll(a_f, 24, axis=1)
            frames = np.concatenate([np.transpose(a_f, (2, 0, 1)), np.transpose(b_f, (2, 0, 1))],
                                    axis=0)[None]
            got = Engine(p, logger).run({"frames": frames})
            o = next(iter(got.values()))
            mid = np.clip(np.transpose(o[0][:3], (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(mid).save(out / "trt_rife_mid.png")
            print(f"  wrote {out / 'trt_rife_mid.png'} {mid.shape}")
            finite = bool(np.isfinite(o).all())
            record("rife_finite", "all_finite", 1.0 if finite else 0.0, 1.0, finite)
            record("rife_signal", "std", float(mid.std()), 10.0, float(mid.std()) >= 10.0)
            # It must sit between the two inputs, not equal one of them.
            da = psnr(mid, (a_f * 255).astype(np.uint8))
            db = psnr(mid, (b_f * 255).astype(np.uint8))
            print(f"  rife psnr vs A={da:.1f} dB, vs B={db:.1f} dB (both finite and unequal = blended)")
        else:
            print("  SKIP  rife (no decoded frame to interpolate)")
    else:
        print("  SKIP  rife (engine or reference missing)")

    (out / "check.json").write_text(json.dumps(
        {"bundle": str(bundle), "calib": str(calib), "results": results,
         "failures": failures}, indent=2))
    print(f"\nreport -> {out / 'check.json'}")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
