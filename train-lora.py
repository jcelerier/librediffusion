"""train-lora.py ported to the daydream fork's export API.

This is the NEW exporter, kept SIDE-BY-SIDE with the original `train-lora.orig.py` (which used the
old `accelerate_with_tensorrt`, removed in daydream) so the two can be compared.

Goal: produce the SAME engine set the C++ engine (score-addon-librediffusion) loads:
    <out>/unet.engine
    <out>/vae_encoder.engine
    <out>/vae_decoder.engine
    <out>/clip.engine
    <out>/clip2.engine        (SDXL only)

Key divergence handled here: daydream's EngineManager does NOT build a CLIP engine (it runs the text
encoder in PyTorch). But the C++ side REQUIRES clip.engine (LibreDiffusion.cpp:515). So we build CLIP
ourselves via daydream's EngineBuilder + the CLIP model spec (a local compile_clip ported from the
original), while UNet + VAE encoder/decoder use daydream's compile_* directly.

Usage (mirrors the original):
    source env.sh
    uv run python train-lora-daydream.py --type sd15 --model SimianLuo/LCM_Dreamshaper_v7 --output ./engines
    uv run python train-lora-daydream.py --type sdxl --model stabilityai/stable-diffusion-xl-base-1.0 \
        --lora latent-consistency/lcm-lora-sdxl --output ./engines_sdxl
"""

import argparse
import os
import sys
from dataclasses import dataclass

import torch

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt import (
    TorchVAEEncoder,
    compile_controlnet,
    compile_unet,
    compile_vae_decoder,
    compile_vae_encoder,
)
from streamdiffusion.acceleration.tensorrt.builder import EngineBuilder, create_onnx_path
from streamdiffusion.acceleration.tensorrt.export_wrappers.unet_unified_export import UnifiedExportWrapper
from streamdiffusion.acceleration.tensorrt.export_wrappers.unet_controlnet_export import (
    ControlNetUNetExportWrapper,
)
from streamdiffusion.acceleration.tensorrt.models.controlnet_models import create_controlnet_model
from streamdiffusion.acceleration.tensorrt.models.models import (
    CLIP, CLIPSDXLPooled, VAE, UNet, VAEEncoder, SDXLUNet, SDXLUNetWrapper, SDXLUNetControlWrapper,
    SDXLUNetIPAdapterWrapper,
)


# ---------------------------------------------------------------------------
# CLIP compile — daydream has the CLIP model spec but no compile_clip; port ours.
# ---------------------------------------------------------------------------
def compile_clip(text_encoder, model_data, onnx_path, onnx_opt_path, engine_path,
                 opt_batch_size=1, engine_build_options=None, output_hidden_states=False,
                 penultimate=False):
    engine_build_options = engine_build_options or {}
    text_encoder = text_encoder.to(torch.device("cuda"), dtype=torch.float16)

    # librediffusion fix: SDXL uses the PENULTIMATE hidden layer (hidden_states[-2]) for BOTH
    # text encoders' sequence embeds (not last_hidden_state). penultimate=True for SDXL clip1+clip2.
    if output_hidden_states:
        class CLIPWithHiddenStates(torch.nn.Module):
            def __init__(self, clip_model, penultimate):
                super().__init__()
                self.clip_model = clip_model
                self.penultimate = penultimate

            def forward(self, input_ids):
                outputs = self.clip_model(input_ids, output_hidden_states=True, return_dict=True)
                seq = outputs.hidden_states[-2] if self.penultimate else outputs.last_hidden_state
                return seq, outputs.text_embeds

        export_model = CLIPWithHiddenStates(text_encoder, penultimate)
    elif penultimate:
        # clip1 (SDXL): keep 2 outputs to match the CLIP model spec; swap the first (seq embeds
        # the C++ reads) to the penultimate hidden layer.
        class CLIPPenultimate(torch.nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.clip_model = clip_model

            def forward(self, input_ids):
                outputs = self.clip_model(input_ids, output_hidden_states=True, return_dict=True)
                return outputs.hidden_states[-2], outputs.pooler_output

        export_model = CLIPPenultimate(text_encoder)
    else:
        export_model = text_encoder

    builder = EngineBuilder(model_data, export_model, device=torch.device("cuda"))
    builder.build(onnx_path, onnx_opt_path, engine_path, opt_batch_size=opt_batch_size,
                  **engine_build_options)


@dataclass
class LoraSpec:
    path: str
    weight: float = 1.0
    weight_name: str = None  # HF-repo subfile (e.g. a kohya .safetensors inside a multi-file repo)

    @classmethod
    def parse(cls, spec: str) -> "LoraSpec":
        # librediffusion: "repo|weight_name" picks a specific file inside an HF LoRA repo
        # (e.g. ByteDance/Hyper-SD|Hyper-SD15-4steps-lora.safetensors). load_lora forwards
        # weight_name= to diffusers. Distinct '|' separator avoids the path:floatweight ':' clash.
        if "|" in spec:
            repo, wn = spec.split("|", 1)
            return cls(path=repo, weight=1.0, weight_name=wn)
        if ":" in spec:
            last = spec.rfind(":")
            try:
                return cls(path=spec[:last], weight=float(spec[last + 1:]))
            except ValueError:
                pass
        return cls(path=spec, weight=1.0)


def parse_args():
    p = argparse.ArgumentParser(description="Export TensorRT engines (daydream API) for the C++ engine")
    p.add_argument("-t", "--type", choices=["sd15", "sdxl", "klein", "img2img-turbo"], default="sd15",
                   help="sd15/sdxl: this script's native diffusers export. img2img-turbo: GaParmar "
                        "pix2pix-turbo skip-VAE (1-step, sd-turbo base; --model = a pretrained name "
                        "like edge_to_image, or a path to a *.pkl). klein: FLUX.2-klein-4B "
                        "(dispatches to the vendored klein export scripts in ./src/streamdiffusion/klein/; "
                        "needs the unified venv with diffusers Flux2 + nvidia-modelopt).")
    p.add_argument("--klein-scripts-dir",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "src", "streamdiffusion", "klein"),
                   help="dir holding export_klein.py / build_klein_engines.py / export_klein_fp8_calib.py")
    p.add_argument("--klein-quality", choices=["speed", "quality", "both"], default="both",
                   help="klein only: speed=fp8_calib transformer, quality=bf16, both")
    p.add_argument("--hw-compat", choices=["none", "ampere_plus", "same_cc"], default="none",
                   dest="hw_compat",
                   help="TensorRT hardware compatibility. none=engine locked to the BUILD gpu's arch "
                        "(fastest/smallest, default). ampere_plus=portable across SM 8.0+ GPUs "
                        "(Ampere/Ada/Hopper/Blackwell) at ~5-15%% inference cost + larger engine. "
                        "same_cc=portable within the same compute capability.")
    p.add_argument("-m", "--model", default="SimianLuo/LCM_Dreamshaper_v7")
    p.add_argument("-o", "--output", default="./engines")
    p.add_argument("-l", "--lora", action="append", dest="loras", default=[], metavar="PATH[:WEIGHT]")
    p.add_argument("--lora-scale", type=float, default=1.0)
    p.add_argument("--no-fuse-lora", action="store_true",
                   help="EXPERIMENT: keep the LoRA active as graph ops (unfused) instead of fusing it "
                        "into the weights. The trace then emits lora_A/lora_B matmuls (scale baked at "
                        "1.0) — the basis for a runtime lora_scale engine input.")
    # ControlNet: when set, ALSO export a controlnet.engine (the ControlNet model) AND a
    # controlnet-variant unet.engine that has the input_control_NN / input_control_middle residual
    # inputs (ControlNetUNetExportWrapper). The C++ pipeline runs the controlnet engine, then feeds its
    # residuals into the UNet's extra inputs. SD1.5 only for now.
    p.add_argument("--controlnet", default=None, metavar="HF_REPO",
                   help="HF ControlNet repo (e.g. lllyasviel/control_v11p_sd15_canny) — exports "
                        "controlnet.engine + a control-aware unet.engine")
    # IP-Adapter is BAKED into the UNet engine (to_k_ip/to_v_ip + projection traced in): the unet.engine
    # becomes an IP variant with encoder_hidden_states seq 77+num_tokens + an ipadapter_scale[num_ip_layers]
    # fp32 input. Image tokens are computed HOST-SIDE and fed in at runtime (external).
    p.add_argument("--ipadapter", default=None, metavar="CKPT",
                   help="IP-Adapter checkpoint (.bin/.safetensors path or HF file) — bakes IP attention "
                        "into unet.engine")
    p.add_argument("--ipadapter-encoder", default=None, metavar="DIR",
                   help="CLIP image encoder dir for IP-Adapter (default: h94/IP-Adapter image_encoder)")
    p.add_argument("--v2v", action="store_true",
                   help="StreamV2V: export a kvo_cache extended-self-attention UNet (kvo_cache_in/out_*). "
                        "Each frame banks its K/V; the next frame concatenates the banked K/V into self-"
                        "attention for temporal coherence. SD1.5 only; not combinable with "
                        "--controlnet/--ipadapter/--fp8. The C++ forward_v2v drives this engine.")
    p.add_argument("--v2v-inject", action="store_true",
                   help="StreamV2V FULL: extended self-attention + feature injection. Uses a 3-component "
                        "cache [K,V,O] (also banks attention OUTPUTS) and bakes get_nn_feats into the "
                        "up_blocks.1 / mid_block self-attn layers. Implies --v2v. ToMe bank compaction is "
                        "host-side (C++). SD1.5 only.")
    p.add_argument("--v2v-fi-strength", type=float, default=0.8,
                   help="Feature-injection blend strength baked into the engine (original demo: 0.95).")
    p.add_argument("--v2v-fi-threshold", type=float, default=0.98,
                   help="Feature-injection cosine-NN threshold baked into the engine (original demo: 0.95).")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--max-batch", type=int, default=2)
    p.add_argument("--opt-batch", type=int, default=2)
    p.add_argument("--min-resolution", type=int, default=512)
    p.add_argument("--max-resolution", type=int, default=1024)
    # librediffusion fork: independent W/H ranges for non-square engines. When unset,
    # they default to the square --min/--max-resolution (so square callers are unchanged).
    p.add_argument("--min-width", type=int, default=None)
    p.add_argument("--max-width", type=int, default=None)
    p.add_argument("--min-height", type=int, default=None)
    p.add_argument("--max-height", type=int, default=None)
    p.add_argument("--opt-height", type=int, default=None)
    p.add_argument("--opt-width", type=int, default=None)
    p.add_argument("--builder-optimization-level", type=int, default=5,
                   help="TRT builder optimization level 0-5 (user-facing toggle)")
    # FP8 (e4m3) UNet: modelopt-calibrate + quantize the heavy attention/FF/conv compute to FP8 (the
    # diffusion FP8 recipe — sensitive embedders/conv_in/out stay fp16). Targets the 41-58% tensor-GEMM
    # slice that profiling found DRAM-bound on SDXL. Needs nvidia-modelopt. Engine I/O is unchanged
    # (the C++ node loads it like any unet.engine). SDXL is the intended target; works for SD too.
    p.add_argument("--fp8", action="store_true",
                   help="FP8-quantize the UNet compute (modelopt e4m3, diffusion recipe). SDXL/SD only.")
    a = p.parse_args()
    # librediffusion fork: square --min/--max-resolution act as shortcuts that set both
    # axes when the per-axis flags aren't given.
    a.min_width = a.min_width if a.min_width is not None else a.min_resolution
    a.max_width = a.max_width if a.max_width is not None else a.max_resolution
    a.min_height = a.min_height if a.min_height is not None else a.min_resolution
    a.max_height = a.max_height if a.max_height is not None else a.max_resolution
    a.opt_height = a.opt_height or a.max_height
    a.opt_width = a.opt_width or a.max_width
    a.lora_specs = [LoraSpec.parse(x) for x in a.loras]
    return a


def export_klein(args):
    """FLUX.2-klein-4B export: dispatch to the klein scripts (export ONNX -> build TRT -> FP8 calib ->
    stage a node-ready bundle). Runs in THIS interpreter's venv (the unified venv has diffusers Flux2 +
    nvidia-modelopt), driving the scripts' paths via env-var overrides. --output is the final bundle dir
    (gets the *.plan + tokenizer.json + bn_*.bin + rife the score node loads)."""
    import subprocess, shutil, glob as _glob
    sd = args.klein_scripts_dir
    out = os.path.abspath(args.output)
    onnx_dir = out + "_onnx-klein"
    eng_dir = out + "_engine-klein"
    os.makedirs(out, exist_ok=True)
    _hf_hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    model_dir = args.model if os.path.isdir(args.model) else (
        _glob.glob(os.path.join(_hf_hub, "models--black-forest-labs--FLUX.2-klein-4B", "snapshots", "*")) or [""])[0]
    env = dict(os.environ, KLEIN_MODEL_DIR=model_dir, KLEIN_ONNX_DIR=onnx_dir, KLEIN_ENGINE_DIR=eng_dir,
               KLEIN_HW_COMPAT=args.hw_compat)  # portable-engine flag honored by build_klein_engines/build_one_fp8
    py = sys.executable

    def run(script, *a):
        cmd = [py, os.path.join(sd, script), *a]
        print(f"[klein] $ {' '.join(cmd)}")
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            raise SystemExit(f"[klein] {script} failed (exit {r.returncode})")

    # 1. ONNX export (transformer bf16 + vae + qwen)
    run("export_klein.py", "--which", "all")
    # 1b. fix the qwen ONNX: the legacy TorchScript exporter mis-types the attention sqrt->cast->mul chain
    #     as COMPLEX128, which TRT rejects. fix_qwen_complex.py rewrites model.onnx -> model_fixed.onnx
    #     (which build_klein_engines.py's build_qwen consumes). REQUIRED before step 2.
    run("fix_qwen_complex.py")
    # 1c. fix the transformer ONNX: torch.onnx tracing bakes the concatenated seq len (Lp+Lt=1232) as
    #     50 inline Constant RoPE-reshape literals -> TRT pins Lp=720 STATIC (can't run the 1440-token
    #     ref-edit streaming path). fix_klein_dynamic_seq.py rewrites 1232->0 -> transformer_dynseq/,
    #     which build_klein_engines.py's build_transformer now consumes. REQUIRED or bf16 is static-720.
    run("fix_klein_dynamic_seq.py")
    # 2. bf16 TRT engines (transformer/qwen/vae_decoder/vae_encoder)
    run("build_klein_engines.py", "--which", "all")
    # 3. FP8-calibrated transformer (speed) — modelopt calib -> ONNX, then build_one_fp8.py -> .plan
    if args.klein_quality in ("speed", "both"):
        run("export_klein_fp8_calib.py")
        # same static-720 fix for the fp8-calib ONNX (its own torch.onnx.export bakes 1232 too).
        run("fix_klein_dynamic_seq.py", "transformer_fp8_calib", "transformer_fp8_calib_dynseq")
        fp8_onnx = os.path.join(onnx_dir, "transformer_fp8_calib_dynseq", "model.onnx")
        fp8_plan = os.path.join(eng_dir, "transformer_fp8_calib.plan")
        run("build_one_fp8.py", fp8_onnx, fp8_plan)

    # 4. stage the node-ready bundle: copy the canonical engines + side files into --output.
    # Engines come from eng_dir. The side files (bn_*.bin, tokenizer.json) aren't produced by the engine
    # build — they're VENDORED in klein/assets/ (self-contained; no external /media path needed). RIFE is
    # vendored as ONNX and built into a .plan here (GPU-specific + honors --hw-compat).
    ASSETS = os.path.join(sd, "assets")  # sd = --klein-scripts-dir (the vendored klein/ dir)
    # Optional prebuilt-engine reference dirs (env): only used if a vendored asset is somehow missing.
    REF = os.environ.get("KLEIN_ENGINE_REF", "")
    RIFE_REF = os.environ.get("KLEIN_RIFE_REF", "")
    def cp(src, srcdirs, dst=None):
        for d in srcdirs:
            if not d:
                continue
            s = os.path.join(d, src)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(out, dst or src)); print(f"[klein] staged {dst or src} (from {d})"); return True
        print(f"[klein] WARNING missing {src} (looked in {srcdirs})"); return False
    for f in ["transformer_bf16.plan", "transformer_fp8_calib.plan",
              "qwen3_encoder_bf16.plan", "vae_decoder_bf16.plan", "vae_encoder_bf16.plan"]:
        cp(f, [eng_dir])
    # side files: prefer the freshly-built eng_dir, else the VENDORED assets, else the legacy /media ref.
    for f in ["bn_mean.bin", "bn_std.bin", "tokenizer.json"]:
        cp(f, [eng_dir, ASSETS, REF])
    # RIFE: use a prebuilt .plan if present (eng_dir/legacy ref), else BUILD it from the vendored ONNX so
    # the bundle is self-contained and the engine matches this GPU + --hw-compat.
    rife_plan_out = os.path.join(out, "rife_ifnet_fp16.plan")
    if not cp("rife_ifnet_fp16.plan", [eng_dir, RIFE_REF, REF]):
        rife_onnx = os.path.join(ASSETS, "rife_ifnet_fp16.onnx")
        if os.path.exists(rife_onnx):
            print("[klein] building rife_ifnet_fp16.plan from vendored ONNX ...")
            # build_rife.py: single 'frames'[B,6,H,W] input, profile B[1-7] H/W[64-1024], honors KLEIN_HW_COMPAT.
            run("build_rife.py", rife_onnx, rife_plan_out)
        else:
            print(f"[klein] WARNING no rife .plan and no vendored ONNX at {rife_onnx}")
    # introspection sidecar: the node reads this (no GPU) to show what loaded + its features.
    write_manifest(out, {
        "model_type": "klein",
        "model_family": "flux2",
        "base_model": "black-forest-labs/FLUX.2-klein-4B",
        "resolution": {"width": 320, "height": 576},
        "denoising_steps": 2,
        "precision": ("fp8+bf16" if args.klein_quality == "both"
                      else ("fp8" if args.klein_quality == "speed" else "bf16")),
        "hw_compat": args.hw_compat,
        "features": {"controlnet": False, "ipadapter": False, "v2v": False, "rife": True,
                     "reference_edit": True},
    }, out)
    print(f"[klein] DONE -> bundle at {out}")


def write_manifest(out_dir, meta, list_engines_dir=None):
    """Write <out_dir>/bundle.json — a tiny introspection sidecar the node reads WITHOUT loading any
    engine into VRAM, to show the user what was loaded (model type, resolution, available features:
    controlnet / ipadapter / v2v / rife). Engine I/O can also be re-derived at load time, but this is
    the zero-cost pre-load preview. Schema is intentionally flat + forward-compatible."""
    import json
    eng_dir = list_engines_dir or out_dir
    engines = sorted(f for f in os.listdir(eng_dir)
                     if f.endswith(".engine") or f.endswith(".plan")) if os.path.isdir(eng_dir) else []
    doc = {"schema_version": 1, **meta, "engines": engines}
    path = os.path.join(out_dir, "bundle.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"  wrote {path}")


def export_img2img_turbo(args):
    """GaParmar img2img-turbo (pix2pix-turbo) export -> a COMPLETE bundle:
        <out>/unet.engine          latent[1,4,h,w], ehs[1,77,1024] -> model_pred (t=999 baked)
        <out>/vae_encoder.engine   image -> latent + enc0..enc3 skips      (skip-VAE encoder)
        <out>/vae_decoder.engine   latent_scaled + s0..s3 (skips) -> image (skip-VAE decoder)
        <out>/clip.engine          sd-turbo CLIPTextModel (1024-dim, last_hidden_state)

    The skip-VAE engines come from the vendored img2img_turbo package (raw TRT, static). CLIP reuses
    the shared compile_clip so the C++ node's prompt path (clip_compute_embeddings) is satisfied —
    sd-turbo's text encoder is a plain CLIPTextModel (penultimate=False: pix2pix uses last_hidden_state).
    """
    from streamdiffusion.img2img_turbo.export import load_model, export_onnx, build_engines

    out = os.path.abspath(args.output)
    os.makedirs(out, exist_ok=True)
    res = args.opt_width
    if args.opt_width != args.opt_height:
        raise SystemExit(f"img2img-turbo engines are square; got {args.opt_width}x{args.opt_height}")

    # --model selects the pretrained variant: a known name (edge_to_image / sketch_to_image_stochastic)
    # or a path to a trained *.pkl. The default --model (an HF repo id) is meaningless here -> edge_to_image.
    KNOWN = ("edge_to_image", "sketch_to_image_stochastic")
    if args.model in KNOWN:
        pretrained_name, pretrained_path = args.model, None
    elif os.path.exists(args.model):
        pretrained_name, pretrained_path = None, args.model
    else:
        print(f"[img2img-turbo] --model '{args.model}' not a known name or a path; using edge_to_image")
        pretrained_name, pretrained_path = "edge_to_image", None

    print(f"[img2img-turbo] model={pretrained_name or pretrained_path} res={res} hw_compat={args.hw_compat} -> {out}")
    # ckpt + onnx dirs live OUTSIDE --output: daydream's compile_clip post-build cleanup os.remove()s
    # every non-.engine entry in the engine dir and chokes on a subdirectory.
    model = load_model(pretrained_name, pretrained_path,
                       ckpt_folder=out.rstrip("/") + "_checkpoints-img2img-turbo")

    onnx_dir = out.rstrip("/") + "_onnx-img2img-turbo"
    export_onnx(model, onnx_dir, res)
    build_engines(onnx_dir, out, res, hw_compat=args.hw_compat,
                  builder_optimization_level=args.builder_optimization_level)

    # CLIP engine (the gap that made standalone img2img-turbo bundles incomplete). sd-turbo text encoder
    # = CLIPTextModel, last_hidden_state -> penultimate=False. Build static (batch 1), matching the engines.
    clip_dim = model.text_encoder.config.hidden_size  # 1024 (sd-turbo / SD2.1 text encoder)
    onnx_clip_dir = out.rstrip("/") + "_onnx-img2img-turbo-clip"
    os.makedirs(onnx_clip_dir, exist_ok=True)
    clip_build_opts = {
        "opt_image_height": res, "opt_image_width": res,
        "min_image_resolution": res, "max_image_resolution": res,
        "min_image_height": res, "max_image_height": res,
        "min_image_width": res, "max_image_width": res,
        "build_static_batch": True, "build_dynamic_shape": False,
        "builder_optimization_level": args.builder_optimization_level,
        "hardware_compatibility": args.hw_compat,
    }
    clip_model = CLIP(device=torch.device("cuda"), max_batch_size=1, min_batch_size=1,
                      embedding_dim=clip_dim)
    o = create_onnx_path("clip", onnx_clip_dir, opt=False)
    oo = create_onnx_path("clip", onnx_clip_dir, opt=True)
    compile_clip(model.text_encoder, clip_model, o, oo, f"{out}/clip.engine",
                 opt_batch_size=1, engine_build_options=clip_build_opts,
                 output_hidden_states=False, penultimate=False)

    write_manifest(out, {
        "model_type": "img2img-turbo",
        "model_family": "sd",
        "base_model": "stabilityai/sd-turbo",
        "variant": pretrained_name or os.path.basename(args.model),
        "embedding_dim": clip_dim,
        "resolution": {"width": res, "height": res},
        "denoising_steps": 1,
        "precision": "fp16",
        "features": {"controlnet": False, "ipadapter": False, "v2v": False, "rife": False,
                     "skip_vae": True},
    })
    print(f"\n[img2img-turbo] DONE. Bundle at {out}:")
    for f in sorted(os.listdir(out)):
        if f.endswith(".engine"):
            sz = os.path.getsize(os.path.join(out, f)) / (1024 * 1024)
            print(f"  {f}  ({sz:.0f} MB)")


def main():
    args = parse_args()
    if args.type == "klein":
        export_klein(args)
        return
    if args.type == "img2img-turbo":
        export_img2img_turbo(args)
        return
    device = torch.device("cuda")
    dtype = torch.float16
    is_sdxl = args.type == "sdxl"

    Pipe = StableDiffusionXLPipeline if is_sdxl else StableDiffusionPipeline
    pipe = Pipe.from_pretrained(args.model).to(device=device, dtype=dtype)

    stream = StreamDiffusion(pipe, t_index_list=[30, 45], torch_dtype=dtype, cfg_type="none")

    # LoRAs: load + fuse (same as the original / the matrix path).
    for spec in args.lora_specs:
        # weight_name selects a specific file inside a multi-file HF repo (kohya LoRAs like
        # ByteDance/Hyper-SD); load_lora forwards it to diffusers' load_lora_weights.
        if spec.weight_name:
            stream.load_lora(spec.path, weight_name=spec.weight_name)
        else:
            stream.load_lora(spec.path)
        print(f"Loaded LoRA: {spec.path} (weight_name={spec.weight_name}, scale {args.lora_scale})")
    if args.lora_specs and not args.no_fuse_lora:
        stream.fuse_lora(fuse_unet=True, fuse_text_encoder=True, lora_scale=args.lora_scale,
                         safe_fusing=False)
    elif args.lora_specs and args.no_fuse_lora:
        # Leave the adapter active (default scale 1.0) so the ONNX trace captures lora_A/lora_B as ops.
        print("[no-fuse-lora] LoRA kept UNFUSED (graph ops, scale=1.0) — runtime-scale experiment")

    # TinyVAE (matches the original + what the C++ engine expects).
    taesd = "madebyollin/taesdxl" if is_sdxl else "madebyollin/taesd"
    stream.vae = AutoencoderTiny.from_pretrained(taesd).to(device=device, dtype=dtype)

    os.makedirs(args.output, exist_ok=True)
    # ONNX dir must be OUTSIDE the engine output dir: daydream's builder.build() post-build cleanup
    # os.remove()s every non-.engine entry in the engine dir (and chokes on a subdirectory).
    onnx_dir = args.output.rstrip("/") + "_onnx"
    os.makedirs(onnx_dir, exist_ok=True)

    embedding_dim = 2048 if is_sdxl else stream.text_encoder.config.hidden_size
    # librediffusion fork: a STATIC engine is one where batch and BOTH spatial axes are pinned
    # (min==max==opt per axis). Non-square static engines (e.g. 512x768) are now first-class.
    static = (args.min_batch == args.max_batch
              and args.min_width == args.max_width == args.opt_width
              and args.min_height == args.max_height == args.opt_height)
    build_opts = {
        "opt_image_height": args.opt_height,
        "opt_image_width": args.opt_width,
        # Keep the square keys for backward compat; the per-axis keys below take precedence.
        "min_image_resolution": args.min_resolution,
        "max_image_resolution": args.max_resolution,
        "min_image_height": args.min_height,
        "max_image_height": args.max_height,
        "min_image_width": args.min_width,
        "max_image_width": args.max_width,
        "build_static_batch": static,
        "build_dynamic_shape": not static,
        "builder_optimization_level": args.builder_optimization_level,
        "hardware_compatibility": args.hw_compat,
    }
    print(f"Export: type={args.type} embedding_dim={embedding_dim} static={static} "
          f"hw_compat={args.hw_compat} "
          f"W=[{args.min_width},{args.max_width}] H=[{args.min_height},{args.max_height}] "
          f"opt={args.opt_width}x{args.opt_height} -> {args.output}")

    def onnx_pair(name):
        return create_onnx_path(name, onnx_dir, opt=False), create_onnx_path(name, onnx_dir, opt=True)

    # ---- IP-Adapter: bake the IP attention (to_k_ip/to_v_ip + projection) into stream.unet BEFORE the
    # UNet trace. This installs IPAttnProcessors with loaded weights; the export wrapper then traces them.
    ip_num_tokens = 4
    ip_num_layers = 0
    if args.ipadapter:
        from diffusers_ipadapter.ip_adapter.ip_adapter import IPAdapter
        enc = args.ipadapter_encoder
        if not enc:
            import glob as _glob
            cands = _glob.glob(os.path.join(
                os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                "hub/models--h94--IP-Adapter/snapshots/*/models/image_encoder"))
            if not cands:
                raise RuntimeError("--ipadapter-encoder not given and h94/IP-Adapter image_encoder not "
                                   "in HF cache")
            enc = cands[0]
        print(f"IP-Adapter: loading {args.ipadapter} (encoder {enc})")
        ip = IPAdapter(stream.pipe, args.ipadapter, enc, device=device, dtype=dtype)
        ip_num_tokens = ip.num_tokens
        # num_ip_layers = count of IP cross-attn processors baked onto the UNet.
        from diffusers_ipadapter.ip_adapter.attention_processor import IPAttnProcessor2_0, IPAttnProcessor
        ip_num_layers = sum(
            1 for p in stream.unet.attn_processors.values()
            if isinstance(p, (IPAttnProcessor2_0, IPAttnProcessor)))
        print(f"IP-Adapter: num_tokens={ip_num_tokens} num_ip_layers={ip_num_layers}")

    # ---- StreamV2V kvo: install the extended-self-attention TRT processor on the UNet's self-attn
    # (attn1) modules BEFORE the trace, so the export emits kvo_cache_in_*/kvo_cache_out_* and the
    # attention concatenates the banked K/V. This is the TRT variant (extended-attn only; feature-
    # injection + ToMe are not bakeable into a static graph). SD1.5 only; the C++ forward_v2v drives it.
    kvo_cache_structure = []
    args.v2v = args.v2v or args.v2v_inject   # --v2v-inject implies --v2v
    if args.v2v:
        if is_sdxl or args.controlnet or args.ipadapter or args.fp8:
            raise SystemExit("--v2v is SD1.5-only and not combinable with "
                             "--controlnet/--ipadapter/--fp8 (extended-attn kvo path).")
        from streamdiffusion.acceleration.tensorrt.models.attention_processors import (
            CachedSTAttnProcessor2_0, CachedSTAttnInjectProcessor2_0)
        from streamdiffusion.acceleration.tensorrt.models.utils import get_kvo_cache_info
        from diffusers.models.attention_processor import AttnProcessor2_0
        procs = stream.unet.attn_processors
        n_inst = n_inject = 0
        for name in procs:
            if isinstance(procs[name], AttnProcessor2_0):
                if args.v2v_inject:
                    # feature injection fires on up_blocks.1 / mid_block self-attn (attn1) — the layer
                    # groups the original streamv2v processor gates on. All layers still emit the
                    # 3-component [K,V,O] cache (uniform I/O); only these consume cached_output.
                    inj = name.endswith("attn1.processor") and (
                        "up_blocks.1" in name or "mid_block" in name)
                    procs[name] = CachedSTAttnInjectProcessor2_0(
                        use_feature_injection=inj,
                        feature_injection_strength=args.v2v_fi_strength,
                        feature_similarity_threshold=args.v2v_fi_threshold)
                    n_inject += int(inj)
                else:
                    procs[name] = CachedSTAttnProcessor2_0()
                n_inst += 1
        stream.unet.set_attn_processor(procs)
        _, kvo_cache_structure, kvo_count = get_kvo_cache_info(stream.unet, args.opt_height, args.opt_width)
        print(f"StreamV2V: installed {'INJECT ' if args.v2v_inject else ''}processor on {n_inst} attn "
              f"modules ({n_inject} injection layers); {kvo_count} kvo_cache layers")

    # ---- UNet ----
    if args.v2v:
        # kvo UNet: UNet(use_cached_attn=True) declares kvo_cache_in_*/out_* + the cache input profile;
        # UnifiedExportWrapper traces them via the installed processors. cache_components=3 ([K,V,O]) for
        # the full injection variant (banks attention outputs too), 2 ([K,V]) for extended-attn only.
        unet_model = UNet(stream.unet, fp16=True, device=device, max_batch_size=args.max_batch,
                          min_batch_size=args.min_batch, embedding_dim=embedding_dim,
                          unet_dim=stream.unet.config.in_channels,
                          use_cached_attn=True, image_height=args.opt_height, image_width=args.opt_width,
                          cache_components=3 if args.v2v_inject else 2)
        print(f"StreamV2V UNet ({'inject' if args.v2v_inject else 'extended-attn'}): "
              f"inputs -> {unet_model.get_input_names()}")
        wrapped_unet = UnifiedExportWrapper(stream.unet, use_controlnet=False, use_ipadapter=False,
                                            control_input_names=None, num_tokens=4,
                                            kvo_cache_structure=kvo_cache_structure,
                                            use_v2v_inject=args.v2v_inject)
    elif is_sdxl and args.controlnet:
        # SDXL + ControlNet: SDXLUNet(use_control=True) declares the 9 down + mid input_control_* inputs
        # ALONGSIDE the SDXL text_embeds/time_ids; SDXLUNetControlWrapper feeds them as
        # added_cond_kwargs + down/mid residuals. The C++ pipeline runs the SDXL controlnet engine and
        # injects its 9 down + 1 mid residuals into these inputs.
        unet_model = SDXLUNet(stream.unet, fp16=True, device=device, max_batch_size=args.max_batch,
                              min_batch_size=args.min_batch, embedding_dim=embedding_dim,
                              unet_dim=stream.unet.config.in_channels, use_control=True)
        print(f"SDXL ControlNet UNet: control inputs -> "
              f"{[n for n in unet_model.get_input_names() if n.startswith('input_control')]}")
        wrapped_unet = SDXLUNetControlWrapper(stream.unet)
    elif is_sdxl and args.ipadapter:
        # SDXL + IP-Adapter: UNION of the SDXL spec (text_embeds/time_ids) and the IP-Adapter spec
        # (encoder_hidden_states seq extended to 77+num_tokens + an ipadapter_scale[num_ip_layers] fp32
        # input). The IP attention processors were already BAKED onto stream.unet by the IPAdapter load
        # above; SDXLUNetIPAdapterWrapper converts them to runtime-scale TRT variants (preserving the
        # loaded to_k_ip/to_v_ip weights) and routes ipadapter_scale + added_cond_kwargs.
        unet_model = SDXLUNet(stream.unet, fp16=True, device=device, max_batch_size=args.max_batch,
                              min_batch_size=args.min_batch, embedding_dim=embedding_dim,
                              unet_dim=stream.unet.config.in_channels,
                              use_ipadapter=True, num_image_tokens=ip_num_tokens,
                              num_ip_layers=ip_num_layers)
        print(f"SDXL IP-Adapter UNet: inputs -> {unet_model.get_input_names()} "
              f"(text_maxlen={unet_model.text_maxlen}, num_ip_layers={ip_num_layers})")
        wrapped_unet = SDXLUNetIPAdapterWrapper(stream.unet, num_tokens=ip_num_tokens)
    elif is_sdxl:
        # SDXL needs the extra text_embeds/time_ids inputs. daydream's base UNet spec lacks them →
        # use the ported SDXLUNet spec + SDXLUNetWrapper (takes them as positional args).
        unet_model = SDXLUNet(stream.unet, fp16=True, device=device, max_batch_size=args.max_batch,
                              min_batch_size=args.min_batch, embedding_dim=embedding_dim,
                              unet_dim=stream.unet.config.in_channels)
        wrapped_unet = SDXLUNetWrapper(stream.unet)
    elif args.controlnet:
        # ControlNet-variant UNet: declare the input_control_NN / input_control_middle residual inputs
        # (UNet.use_control + unet_arch) and wrap so the trace takes them as positional graph inputs
        # (ControlNetUNetExportWrapper). The C++ pipeline runs the controlnet engine, then injects its
        # 12 down + 1 mid residuals into these inputs. block_out_channels drives the residual shapes
        # (SD1.5 = 4 blocks (320,640,1280,1280) -> 12 down + mid; see UNet.get_control).
        arch = {"block_out_channels": tuple(stream.unet.config.block_out_channels),
                "layers_per_block": int(stream.unet.config.layers_per_block)}
        unet_model = UNet(stream.unet, fp16=True, device=device, max_batch_size=args.max_batch,
                          min_batch_size=args.min_batch, embedding_dim=embedding_dim,
                          unet_dim=stream.unet.config.in_channels,
                          use_control=True, unet_arch=arch,
                          image_height=args.opt_height, image_width=args.opt_width)
        # control_input_names in the EXACT order the model spec declares them (sorted: 00..11, middle).
        control_input_names = [n for n in unet_model.get_input_names() if n.startswith("input_control")]
        print(f"ControlNet UNet: {len(control_input_names)} control inputs -> {control_input_names}")
        wrapped_unet = ControlNetUNetExportWrapper(stream.unet, control_input_names=control_input_names,
                                                   kvo_cache_structure=[])
    elif args.ipadapter:
        # IP-Adapter-variant UNet: encoder_hidden_states seq extended to 77+num_tokens + an
        # ipadapter_scale[num_ip_layers] fp32 input. Processors were already baked onto stream.unet by
        # the IPAdapter load above -> install_processors=False so the wrapper PRESERVES those weights.
        unet_model = UNet(stream.unet, fp16=True, device=device, max_batch_size=args.max_batch,
                          min_batch_size=args.min_batch, embedding_dim=embedding_dim,
                          unet_dim=stream.unet.config.in_channels,
                          use_ipadapter=True, num_image_tokens=ip_num_tokens, num_ip_layers=ip_num_layers)
        print(f"IP-Adapter UNet: inputs -> {unet_model.get_input_names()} "
              f"(text_maxlen={unet_model.text_maxlen})")
        wrapped_unet = UnifiedExportWrapper(stream.unet, use_controlnet=False, use_ipadapter=True,
                                            control_input_names=None, num_tokens=ip_num_tokens,
                                            kvo_cache_structure=[], install_processors=False)
    else:
        # The forked-diffusers UNet returns (sample, kvo_cache_out); UnifiedExportWrapper normalizes
        # to a single `sample` output (no cached-attn/controlnet/ipadapter here → basic path).
        unet_model = UNet(stream.unet, fp16=True, device=device, max_batch_size=args.max_batch,
                          min_batch_size=args.min_batch, embedding_dim=embedding_dim,
                          unet_dim=stream.unet.config.in_channels)
        wrapped_unet = UnifiedExportWrapper(stream.unet, use_controlnet=False, use_ipadapter=False,
                                            control_input_names=None, num_tokens=4, kvo_cache_structure=[])
    # ---- FP8 (e4m3) UNet quantization (modelopt) — weave FP8 into the trace BEFORE export ----
    # Inserts Q/DQ on the heavy attention/FF/conv compute; sensitive embedders/conv_in-out stay fp16.
    # The TRT-11 build is strongly-typed, so the Q/DQ in the ONNX is honored per-tensor automatically
    # (no builder flag). LRD_FP8_EXPORT tells export_onnx/optimize_onnx to preserve Q/DQ (no
    # constant-folding, no onnxslim). FP8 + ControlNet/IP-Adapter not validated yet -> guard it.
    if args.fp8:
        if args.controlnet or args.ipadapter:
            raise SystemExit("--fp8 with --controlnet/--ipadapter is not supported yet "
                             "(quantize the plain UNet first).")
        from streamdiffusion.acceleration.tensorrt.fp8_quantize import (
            quantize_unet_fp8, build_sdxl_calib_inputs, SDXL_SENSITIVE)
        print("FP8: building calibration inputs + quantizing UNet (e4m3, diffusion recipe)...")
        if is_sdxl:
            calib = build_sdxl_calib_inputs(stream.pipe, args.opt_height, args.opt_width, device, dtype)
        else:
            # SD1.5/SD-turbo: UnifiedExportWrapper.forward(sample, timestep, encoder_hidden_states)
            lh, lw = args.opt_height // 8, args.opt_width // 8
            cdim = stream.text_encoder.config.hidden_size
            calib = []
            for t in (999.0, 800.0, 600.0, 400.0, 200.0, 20.0):
                s = torch.randn((1, 4, lh, lw), device=device, dtype=dtype)
                ehs = torch.randn((1, 77, cdim), device=device, dtype=dtype)
                calib.append((s, torch.tensor([t], device=device, dtype=torch.float32), ehs))
        quantize_unet_fp8(wrapped_unet, calib, high_precision="Half", sensitive=SDXL_SENSITIVE)
        os.environ["LRD_FP8_EXPORT"] = "1"   # gate export_onnx/optimize_onnx to preserve Q/DQ

    o, oo = onnx_pair("unet")
    compile_unet(wrapped_unet, unet_model, o, oo, f"{args.output}/unet.engine",
                 opt_batch_size=args.opt_batch, engine_build_options=build_opts)
    os.environ.pop("LRD_FP8_EXPORT", None)   # scope the flag to the UNet build only

    # ---- VAE decoder ----
    # The decoder engine traces latent(4ch)->image, but vae.forward is encode->decode. EngineManager
    # temporarily sets vae.forward = vae.decode for the trace; replicate that here.
    vae_dec_model = VAE(device=device, max_batch_size=args.max_batch, min_batch_size=args.min_batch)
    o, oo = onnx_pair("vae_decoder")
    stream.vae.forward = stream.vae.decode
    try:
        compile_vae_decoder(stream.vae, vae_dec_model, o, oo, f"{args.output}/vae_decoder.engine",
                            opt_batch_size=args.opt_batch, engine_build_options=build_opts)
    finally:
        if hasattr(stream.vae, "forward"):
            try:
                delattr(stream.vae, "forward")
            except AttributeError:
                pass

    # ---- VAE encoder ----
    vae_enc = TorchVAEEncoder(stream.vae)
    vae_enc_model = VAEEncoder(device=device, max_batch_size=args.max_batch, min_batch_size=args.min_batch)
    o, oo = onnx_pair("vae_encoder")
    compile_vae_encoder(vae_enc, vae_enc_model, o, oo, f"{args.output}/vae_encoder.engine",
                        opt_batch_size=args.opt_batch, engine_build_options=build_opts)

    # ---- CLIP (and CLIP2 for SDXL) — daydream doesn't build these; the C++ engine needs them ----
    clip_dim = stream.text_encoder.config.hidden_size
    clip_model = CLIP(device=device, max_batch_size=args.max_batch, min_batch_size=args.min_batch,
                      embedding_dim=clip_dim)
    o, oo = onnx_pair("clip")
    compile_clip(stream.text_encoder, clip_model, o, oo, f"{args.output}/clip.engine",
                 opt_batch_size=args.opt_batch, engine_build_options=build_opts,
                 output_hidden_states=False,
                 penultimate=is_sdxl)  # SDXL clip1 uses hidden_states[-2] too

    if is_sdxl:
        # daydream's StreamDiffusion only mirrors pipe.text_encoder as stream.text_encoder; the SDXL
        # second encoder (CLIPTextModelWithProjection) lives on the underlying pipe.
        text_encoder_2 = stream.pipe.text_encoder_2
        clip2_dim = text_encoder_2.config.hidden_size  # 1280
        # CLIPSDXLPooled keeps BOTH outputs (hidden_states seq + text_embeddings pooled) — the C++
        # CLIPWrapper::computeEmbeddingsWithPooled requires them; base CLIP strips the pooled one.
        clip2_model = CLIPSDXLPooled(device=device, max_batch_size=args.max_batch,
                                     min_batch_size=args.min_batch, embedding_dim=clip2_dim)
        o, oo = onnx_pair("clip2")
        compile_clip(text_encoder_2, clip2_model, o, oo, f"{args.output}/clip2.engine",
                     opt_batch_size=args.opt_batch, engine_build_options=build_opts,
                     output_hidden_states=True,  # SDXL CLIP2 needs hidden states + pooled
                     penultimate=True)           # ...from the penultimate layer

    # ---- ControlNet engine (separate from the control-aware UNet above) ----
    if args.controlnet:
        from diffusers import ControlNetModel
        print(f"\nControlNet: loading {args.controlnet}")
        controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch.float16).to(device)
        # create_controlnet_model picks ControlNetTRT (sd15: 12 down + mid, inputs sample/timestep/
        # encoder_hidden_states/controlnet_cond — NO conditioning_scale; the C++ applies scale at
        # injection). embedding_dim from the text encoder (768 for SD1.5).
        cn_model = create_controlnet_model("sdxl" if is_sdxl else "sd15", unet=stream.unet,
                                           min_batch_size=args.min_batch, max_batch_size=args.max_batch,
                                           embedding_dim=embedding_dim,
                                           unet_dim=stream.unet.config.in_channels)
        o, oo = onnx_pair("controlnet")
        compile_controlnet(controlnet, cn_model, o, oo, f"{args.output}/controlnet.engine",
                           opt_batch_size=args.opt_batch, engine_build_options=build_opts)

    # introspection sidecar (zero-GPU preview of what the bundle contains)
    write_manifest(args.output, {
        "model_type": args.type,                       # sd15 | sdxl
        "model_family": "sdxl" if is_sdxl else "sd",
        "base_model": args.model,
        "embedding_dim": embedding_dim,
        "resolution": {"min": args.min_resolution, "max": args.max_resolution,
                       "opt_width": args.opt_width, "opt_height": args.opt_height},
        "batch": {"min": args.min_batch, "opt": args.opt_batch, "max": args.max_batch},
        "precision": "fp8" if args.fp8 else "fp16",
        "loras": args.loras,
        "features": {
            "controlnet": bool(args.controlnet),
            "ipadapter": bool(args.ipadapter),
            "v2v": bool(args.v2v),                     # kvo extended-self-attention UNet (--v2v)
            "rife": False,
        },
        "controlnet_repo": args.controlnet,
        "ipadapter_ckpt": args.ipadapter,
    })

    print(f"\nDONE. Engines in {args.output}:")
    for f in sorted(os.listdir(args.output)):
        if f.endswith(".engine"):
            sz = os.path.getsize(os.path.join(args.output, f)) / (1024 * 1024)
            print(f"  {f}  ({sz:.0f} MB)")


if __name__ == "__main__":
    main()
