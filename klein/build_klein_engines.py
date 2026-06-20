"""Build FLUX.2-klein-4B TRT engines from the exported ONNX.

bf16 engines for the correctness gate; the transformer also gets an FP8 build path (--fp8) once bf16
validates. Dynamic shape profiles: batch 1, Lp in [720, 1440] (single latent .. +ref tokens),
Lt fixed 512. The wide Lp range gives the variable-query-length the spatial-KV-cache needs.

Run under the flux venv env (klein_env.sh sourced).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import tensorrt as trt

# Path overrides (env) for train-lora-daydream.py --type klein; standalone use unchanged.
ONNX = Path(os.environ.get("KLEIN_ONNX_DIR", "./onnx-klein"))
ENG = Path(os.environ.get("KLEIN_ENGINE_DIR", "./engine-klein"))
ENG.mkdir(parents=True, exist_ok=True)

LOG = trt.Logger(trt.Logger.INFO)


def _apply_hw_compat(config):
    """Apply TensorRT hardware-compatibility from the KLEIN_HW_COMPAT env (none|ampere_plus|same_cc).
    Portable engines run across GPU archs at ~5-15% inference cost. Default none = build-GPU only.
    Shared by build_one_fp8.py via import."""
    hc = os.environ.get("KLEIN_HW_COMPAT", "none").lower()
    lvl = None
    if hc in ("ampere_plus", "ampere", "amperePlus".lower()):
        lvl = trt.HardwareCompatibilityLevel.AMPERE_PLUS
    elif hc in ("same_cc", "same_compute_capability", "same"):
        lvl = trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY
    elif hc not in ("none", ""):
        print(f"[W] unknown KLEIN_HW_COMPAT '{hc}', using NONE")
    if lvl is not None:
        config.hardware_compatibility_level = lvl
        print(f"[I] klein TRT hardware compatibility: {hc} (PORTABLE; ~5-15% slower, larger)")


def build(onnx_path, engine_path, profiles, bf16=True, fp8=False, workspace_gb=12):
    builder = trt.Builder(LOG)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, LOG)
    print(f"parsing {onnx_path} ...")
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read(), path=str(onnx_path))
    if not ok:
        for i in range(parser.num_errors):
            print("PARSE ERR:", parser.get_error(i))
        raise RuntimeError("onnx parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    _apply_hw_compat(config)  # KLEIN_HW_COMPAT env: none|ampere_plus|same_cc (portable engines)
    # strongly-typed network: dtypes come from the ONNX graph; bf16/fp8 are intrinsic.
    profile = builder.create_optimization_profile()
    for name, (mn, opt, mx) in profiles.items():
        profile.set_shape(name, mn, opt, mx)
    config.add_optimization_profile(profile)

    print(f"building engine -> {engine_path} (this can take a while)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build returned None")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"WROTE {engine_path}  ({os.path.getsize(engine_path)/1e6:.0f} MB)")


def build_transformer():
    # Lp: 720 (single img) .. 1440 (latent + 1 ref token block). Lt fixed 512.
    # opt=1440 tunes the engine for the STREAMING ref-edit path (latent 720 + ref 720), which is the
    # real-time use-case; single-image (720) still works (it's within min..max). The previously
    # DEPLOYED transformer_bf16.plan was static-720 (built by an earlier path) and could NOT run the
    # 1440-token ref path -> bf16-quality streaming was blocked. This rebuild fixes that.
    profiles = {
        "hidden_states": ((1, 720, 128), (1, 1440, 128), (1, 1440, 128)),
        "encoder_hidden_states": ((1, 512, 7680), (1, 512, 7680), (1, 512, 7680)),
        "timestep": ((1,), (1,), (1,)),
        "img_ids": ((1, 720, 4), (1, 1440, 4), (1, 1440, 4)),
        "txt_ids": ((1, 512, 4), (1, 512, 4), (1, 512, 4)),
    }
    build(ONNX / "transformer/model.onnx", ENG / "transformer_bf16.plan", profiles, workspace_gb=14)


def build_qwen():
    profiles = {
        "input_ids": ((1, 512), (1, 512), (1, 512)),
        "attention_mask": ((1, 512), (1, 512), (1, 512)),
    }
    build(ONNX / "qwen3_encoder/model_fixed.onnx", ENG / "qwen3_encoder_bf16.plan", profiles, workspace_gb=10)


def build_vae_decoder():
    profiles = {"latent": ((1, 32, 72, 40), (1, 32, 72, 40), (1, 32, 72, 40))}
    build(ONNX / "vae_decoder/model.onnx", ENG / "vae_decoder_bf16.plan", profiles, workspace_gb=8)


def build_vae_encoder():
    profiles = {"image": ((1, 3, 576, 320), (1, 3, 576, 320), (1, 3, 576, 320))}
    build(ONNX / "vae_encoder/model.onnx", ENG / "vae_encoder_bf16.plan", profiles, workspace_gb=8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    choices=["all", "transformer", "qwen", "vae_decoder", "vae_encoder"])
    a = ap.parse_args()
    if a.which in ("all", "vae_decoder"):
        build_vae_decoder()
    if a.which in ("all", "vae_encoder"):
        build_vae_encoder()
    if a.which in ("all", "transformer"):
        build_transformer()
    if a.which in ("all", "qwen"):
        build_qwen()
    print("ALL DONE")
