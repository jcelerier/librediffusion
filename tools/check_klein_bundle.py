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
# RIFE: above ~40 dB the output is effectively one of its inputs rather than an interpolation, and a
# true midpoint between A and a shifted A lands at a similar distance from each. Measured on a
# correct engine: 15.4 / 15.2 dB (gap 0.2). An engine echoing frame A: 54.2 / 7.6 dB (gap 46.6).
RIFE_COPY_MAX_DB = 40.0
RIFE_GAP_MAX_DB = 6.0

# getattr rather than attribute access: the enum spelling has moved between TensorRT majors and a
# missing name must not take the whole tool down at import time.
_TRT2TORCH = {getattr(trt.DataType, n): t for n, t in (
    ("FLOAT", torch.float32), ("HALF", torch.float16), ("BF16", torch.bfloat16),
    ("INT32", torch.int32), ("INT64", torch.int64), ("INT8", torch.int8), ("BOOL", torch.bool),
) if hasattr(trt.DataType, n)}


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

    results, failures, skipped = [], [], []

    def record(name, metric, value, threshold, ok, **extra):
        results.append({"check": name, "metric": metric, "value": value, "threshold": threshold,
                        "pass": bool(ok), **extra})
        detail = "".join(f" {k}={v}" for k, v in extra.items())
        print(f"  {'PASS' if ok else 'FAIL'}  {name:28s} {metric}={value:.6g} "
              f"(need {threshold}){detail}")
        if not ok:
            failures.append(name)

    def skip(name, why, fatal):
        """A skip that is allowed to be a pass only when the thing is genuinely optional. Anything
        else counts as a failure: a checker that verifies nothing must not report success."""
        skipped.append({"check": name, "reason": why, "fatal": bool(fatal)})
        print(f"  {'FAIL' if fatal else 'SKIP'}  {name:28s} ({why})")
        if fatal:
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
        skip("qwen3_encoder", "engine or reference missing", fatal=True)

    # --- transformer (bf16 and fp8) ------------------------------------------------------------
    # Which variants exist is the builder's choice: --klein-quality speed ships only the FP8 plan,
    # quality only the bf16 one, both ships both. So neither file is individually required -- what is
    # required is that the bundle carries at least one transformer and that it verifies.
    img_ids, txt_ids = ref("031_ids__img_ids"), ref("030_ids__txt_ids")
    sigmas = ref("032_ids__timesteps_per_step")
    transformers_checked = 0
    for plan, key in (("transformer_bf16.plan", "transformer_bf16"),
                      ("transformer_fp8_calib.plan", "transformer_fp8")):
        p = bundle / plan
        if not p.exists():
            skip(key, f"no {plan}", fatal=False)
            continue
        if ehs is None or img_ids is None or txt_ids is None or sigmas is None:
            skip(key, "reference missing", fatal=True)
            continue
        if len(sigmas) == 0:
            skip(key, "reference has no timesteps", fatal=True)
            continue
        missing = [s for s in range(len(sigmas))
                   if ref(f"11{s}_transformer__in_hidden_step{s}") is None
                   or ref(f"10{s}_transformer__velocity_step{s}") is None]
        if missing:
            # Skipping the steps individually and keeping the running minimum at its 1.0 seed would
            # report a perfect score for an engine that was never executed.
            skip(key, f"reference missing for step(s) {missing}", fatal=True)
            continue
        eng = Engine(p, logger)
        worst = 1.0
        for step in range(len(sigmas)):
            got = eng.run({"hidden_states": ref(f"11{step}_transformer__in_hidden_step{step}"),
                           "encoder_hidden_states": ehs,
                           "timestep": np.asarray([sigmas[step]], dtype=np.float32),
                           "img_ids": img_ids, "txt_ids": txt_ids})
            worst = min(worst, cosine(next(iter(got.values())),
                                      ref(f"10{step}_transformer__velocity_step{step}")))
        transformers_checked += 1
        record(key, "min_cos_over_steps", worst, COS_MIN[key], worst >= COS_MIN[key],
               steps=len(sigmas))
    if transformers_checked == 0:
        skip("transformer", "the bundle carries no transformer that could be verified", fatal=True)

    # --- vae decoder: the image to look at -----------------------------------------------------
    dec_in = ref("200_vae__decoder_in_latent")
    p = bundle / "vae_decoder_bf16.plan"
    refpng = calib / "calib_reference.png"
    if p.exists() and dec_in is not None:
        got = Engine(p, logger).run({"latent": dec_in})
        img = to_u8_image(next(iter(got.values())))
        try:
            from PIL import Image
        except ImportError:
            Image = None
        if Image is None:
            np.save(out / "trt_vae_decode.npy", img)
            # std alone cannot tell a correct decode from structured noise, so without the
            # comparison this check has not established anything.
            skip("vae_decoder_vs_reference", "PIL missing, cannot compare against the frame",
                 fatal=True)
        else:
            Image.fromarray(img).save(out / "trt_vae_decode.png")
            print(f"  wrote {out / 'trt_vae_decode.png'} {img.shape}")
            if not refpng.exists():
                skip("vae_decoder_vs_reference", f"no reference frame at {refpng}", fatal=True)
            else:
                r = np.asarray(Image.open(refpng).convert("RGB"))
                if r.shape != img.shape:
                    skip("vae_decoder_vs_reference",
                         f"shape {img.shape} vs reference {r.shape}", fatal=True)
                else:
                    v = psnr(img, r)
                    record("vae_decoder_vs_reference", "psnr_db", v, PSNR_MIN, v >= PSNR_MIN)
        # Independent of the reference: a decode that collapsed would be flat or non-finite.
        record("vae_decoder_signal", "std", float(img.std()), 10.0, float(img.std()) >= 10.0)
    else:
        skip("vae_decoder", "engine or reference missing", fatal=True)

    # --- vae encoder (the img2img / reference-edit path) ---------------------------------------
    enc_in, enc_out = ref("210_vae__encoder_in_image"), ref("211_vae__encoder_out_latent")
    p = bundle / "vae_encoder_bf16.plan"
    if p.exists() and enc_in is not None and enc_out is not None:
        got = Engine(p, logger).run({"image": enc_in})
        v = next(iter(got.values()))
        c = cosine(v, enc_out)
        record("vae_encoder", "cos", c, COS_MIN["qwen"], c >= COS_MIN["qwen"])
    else:
        skip("vae_encoder", "engine or reference missing", fatal=True)

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
            # Finite and non-flat is also true of an engine that hands back one of its inputs, which
            # is the realistic way RIFE goes wrong. A frame halfway between A and a shifted A sits
            # at a similar distance from both, so require that: neither input may be reproduced
            # (PSNR below RIFE_COPY_MAX_DB), and the two distances must be close to each other.
            da = psnr(mid, (a_f * 255).astype(np.uint8))
            db = psnr(mid, (b_f * 255).astype(np.uint8))
            record("rife_not_a_copy", "max_psnr_vs_input_db", max(da, db), RIFE_COPY_MAX_DB,
                   max(da, db) < RIFE_COPY_MAX_DB, psnr_vs_A=round(da, 1), psnr_vs_B=round(db, 1))
            record("rife_balanced", "abs_psnr_gap_db", abs(da - db), RIFE_GAP_MAX_DB,
                   abs(da - db) < RIFE_GAP_MAX_DB)
        else:
            skip("rife", "no decoded frame to interpolate", fatal=True)
    else:
        skip("rife", "engine or reference missing", fatal=True)

    (out / "check.json").write_text(json.dumps(
        {"bundle": str(bundle), "calib": str(calib), "results": results,
         "skipped": skipped, "failures": failures}, indent=2))
    print(f"\nreport -> {out / 'check.json'}")
    if not results:
        print("FAILED: no check actually ran -- nothing was verified")
        return 1
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print(f"ALL CHECKS PASSED ({len(results)} ran, {len(skipped)} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
