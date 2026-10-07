"""Produce the klein FP8 calibration tensors from the model itself.

export_klein_fp8_calib.py needs real activations to collect amax over: the Qwen3 encoder output, the
RoPE ids, the sigma schedule and the transformer input at each of the two denoising steps. Those used
to come from $KLEIN_FP8_GOLD -- a dump written by a validation harness that lives outside this repo,
so on a clean checkout the fp8 half of the build could not run at all, and `--klein-quality both`
(the default) died at that step. Every one of those tensors is just what the reference diffusers
pipeline computes, so we run it once here and write them ourselves.

One hook on the transformer's forward captures all of it: the pipeline passes hidden_states,
encoder_hidden_states, timestep, img_ids and txt_ids to it by keyword, which is also the exact I/O
contract export_klein.py traces. Nothing reimplements the text encoder or the ids, so this cannot
drift from the pipeline the way a hand-written dump can.

Also writes the two side files the bundle needs that are properties of the VAE rather than of any
engine: bn_mean/bn_std, the per-channel normalisation the C++ applies around the latent (fp32[128],
NOT a scalar scaling_factor), and a copy of the model's tokenizer.json.

Outputs into $KLEIN_CALIB_DIR (default: alongside $KLEIN_ONNX_DIR), named as export_klein_fp8_calib
loads them.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

STEPS = 2
# Seed/prompt of the reference run the fp8 recipe was tuned against. The prompt only has to put the
# activations in a representative range; override for a model fine-tuned on a very different domain.
SEED = 52
PROMPT = os.environ.get(
    "KLEIN_CALIB_PROMPT",
    "a detailed anime portrait of a woman, masterpiece, best quality, cinematic lighting",
)
HIDDEN_LAYERS = (9, 18, 27)


def _resolve_klein_model_dir():
    import glob

    explicit = os.environ.get("KLEIN_MODEL_DIR")
    if explicit:
        return explicit
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    cands = glob.glob(os.path.join(hub, "models--black-forest-labs--FLUX.2-klein-4B", "snapshots", "*"))
    if not cands:
        raise RuntimeError(
            "FLUX.2-klein-4B not found; set KLEIN_MODEL_DIR or download into the HF cache (HF_HOME)."
        )
    return cands[0]


def main() -> int:
    from diffusers import Flux2KleinPipeline

    model_dir = _resolve_klein_model_dir()
    width = int(os.environ.get("KLEIN_WIDTH", "320"))
    height = int(os.environ.get("KLEIN_HEIGHT", "576"))
    text_len = int(os.environ.get("KLEIN_TEXT_LEN", "512"))
    onnx_base = os.environ.get("KLEIN_ONNX_DIR", "./onnx-klein")
    out = Path(os.environ.get("KLEIN_CALIB_DIR", str(Path(onnx_base).parent / "calib-klein")))
    out.mkdir(parents=True, exist_ok=True)

    print(f"[calib] model={model_dir}")
    print(f"[calib] {width}x{height} Lt={text_len} steps={STEPS} -> {out}")

    pipe = Flux2KleinPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    # The transformer and the Qwen3 encoder are ~7.5 GB each in bf16; holding both resident plus the
    # calibration activations does not fit a 24 GB card next to anything else. This is a two-step run
    # whose wall time is irrelevant, so offload rather than risk an OOM that depends on the GPU.
    pipe.enable_model_cpu_offload()

    # VAE normalisation: a per-channel batch norm over the 128-dim patchified latent, read straight off
    # the module. (x - running_mean) / sqrt(running_var + eps), matching launch_klein_bn in the C++.
    bn_mean = pipe.vae.bn.running_mean.detach().to(torch.float32).cpu().numpy()
    bn_var = pipe.vae.bn.running_var.detach().to(torch.float32).cpu().numpy()
    bn_eps = float(pipe.vae.config.batch_norm_eps)
    bn_std = np.sqrt(bn_var + bn_eps).astype(np.float32)
    bn_mean = bn_mean.astype(np.float32)
    if bn_mean.shape != (128,) or bn_std.shape != (128,):
        raise RuntimeError(f"expected fp32[128] bn stats, got {bn_mean.shape}/{bn_std.shape}")
    (out / "bn_mean.bin").write_bytes(bn_mean.tobytes())
    (out / "bn_std.bin").write_bytes(bn_std.tobytes())
    print(f"[calib] bn_mean[:3]={bn_mean[:3]} bn_std[:3]={bn_std[:3]}")

    tok_src = Path(model_dir) / "tokenizer" / "tokenizer.json"
    if tok_src.exists():
        shutil.copy2(tok_src, out / "tokenizer.json")
        print(f"[calib] tokenizer.json from {tok_src}")
    else:
        print(f"[calib] WARNING no tokenizer.json at {tok_src} (bundle falls back to the vendored copy)")

    # Two hooks, both on the modules the ONNX export traces, so what we record is by construction the
    # engines' own I/O contract: the text encoder's (input_ids, attention_mask) and the transformer's
    # five inputs plus its velocity. check_klein_bundle.py replays these through the built engines.
    cap: dict = {"in_hidden": [], "timesteps": [], "velocity": [], "latents": []}
    real_fwd = pipe.transformer.forward
    real_te = pipe.text_encoder.forward

    def te_hook(*args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        mask = kwargs.get("attention_mask")
        if ids is not None:
            cap.setdefault("input_ids", ids.detach().to(torch.int32).cpu().numpy())
        if mask is not None:
            cap.setdefault("attention_mask", mask.detach().to(torch.int32).cpu().numpy())
        return real_te(*args, **kwargs)

    def fwd_hook(*args, **kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        for key in ("encoder_hidden_states", "img_ids", "txt_ids", "timestep"):
            if kwargs.get(key) is None:
                raise RuntimeError(
                    f"transformer forward did not receive '{key}' by keyword -- diffusers changed its "
                    "Flux2 call convention; the calibration capture must be updated with it."
                )
        cap.setdefault("ehs", kwargs["encoder_hidden_states"].detach().to(torch.float32).cpu().numpy())
        cap.setdefault("img_ids", kwargs["img_ids"].detach().to(torch.float32).cpu().numpy())
        cap.setdefault("txt_ids", kwargs["txt_ids"].detach().to(torch.float32).cpu().numpy())
        cap["in_hidden"].append(hs.detach().to(torch.float32).cpu().numpy())
        cap["timesteps"].append(float(kwargs["timestep"].flatten()[0].item()))
        out = real_fwd(*args, **kwargs)
        vel = out[0] if isinstance(out, tuple) else out.sample
        cap["velocity"].append(vel.detach().to(torch.float32).cpu().numpy())
        return out

    def step_cb(p_, step, t, cbk):
        lat = cbk.get("latents")
        if lat is not None:
            cap["latents"].append(lat.detach().to(torch.float32).cpu().numpy())
        return cbk

    # The decoder's input as the pipeline hands it over, i.e. already unpacked and denormalised. Captured
    # rather than reconstructed so the check exercises the engine instead of a second implementation of
    # the pack/bn math.
    real_dec = pipe.vae.decode

    def dec_hook(*args, **kwargs):
        lat = kwargs.get("z", args[0] if args else None)
        if lat is not None:
            cap.setdefault("dec_in", lat.detach().to(torch.float32).cpu().numpy())
        return real_dec(*args, **kwargs)

    pipe.vae.decode = dec_hook
    pipe.transformer.forward = fwd_hook
    pipe.text_encoder.forward = te_hook
    try:
        # CPU generator so the noise is the same whatever the offload placement does.
        gen = torch.Generator("cpu").manual_seed(SEED)
        image = pipe(
            image=None,
            prompt=PROMPT,
            num_inference_steps=STEPS,
            guidance_scale=1.0,
            height=height,
            width=width,
            generator=gen,
            max_sequence_length=text_len,
            text_encoder_out_layers=HIDDEN_LAYERS,
            output_type="pil",
            callback_on_step_end=step_cb,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
    finally:
        pipe.transformer.forward = real_fwd
        pipe.text_encoder.forward = real_te
        pipe.vae.decode = real_dec

    if len(cap["in_hidden"]) != STEPS:
        raise RuntimeError(f"expected {STEPS} transformer calls, captured {len(cap['in_hidden'])}")

    def save(name, arr):
        arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        np.save(out / f"{name}.npy", arr)
        print(f"[calib] {name}.npy {list(arr.shape)}")

    save("005_textencode__encoder_hidden_states_7680", cap["ehs"])
    save("030_ids__txt_ids", cap["txt_ids"])
    save("031_ids__img_ids", cap["img_ids"])
    save("032_ids__timesteps_per_step", np.asarray(cap["timesteps"], dtype=np.float32))
    for i, h in enumerate(cap["in_hidden"]):
        save(f"11{i}_transformer__in_hidden_step{i}", h)

    # Expected outputs, for check_klein_bundle.py to compare the engines against. int32 on disk because
    # npy keeps the dtype and the engines take int64/int32 depending on the trace; the checker casts.
    for key, name in (("input_ids", "000_textencode__input_ids"),
                      ("attention_mask", "001_textencode__attention_mask")):
        if key in cap:
            np.save(out / f"{name}.npy", np.ascontiguousarray(cap[key]))
            print(f"[calib] {name}.npy {list(cap[key].shape)} {cap[key].dtype}")
    for i, v in enumerate(cap["velocity"]):
        save(f"10{i}_transformer__velocity_step{i}", v)
    for i, s in enumerate(cap["latents"]):
        save(f"12{i}_scheduler__latent_after_step{i}", s)
    if "dec_in" in cap:
        save("200_vae__decoder_in_latent", cap["dec_in"])

    # The txt2img run never encodes, but the img2img / reference-edit path does, so give the encoder
    # engine a reference too: round-trip the frame we just generated through the real VAE encoder.
    # The execution device, NOT next(parameters()).device: under cpu offload the VAE's parameters sit
    # on the CPU between calls and accelerate's pre-hook moves them to the GPU on forward, so reading
    # the parameter device sends the input to the CPU and the conv then gets mismatched operands.
    vae_dev = getattr(pipe, "_execution_device", None) or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    enc_in = (torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0 * 2.0 - 1.0)
              .permute(2, 0, 1)[None])
    with torch.no_grad():
        enc_out = pipe.vae.encode(enc_in.to(vae_dev, pipe.vae.dtype)).latent_dist.mode()
    save("210_vae__encoder_in_image", enc_in)
    save("211_vae__encoder_out_latent", enc_out.detach().to(torch.float32).cpu().numpy())

    # The pipeline's own output: a free end-to-end check that the weights and the venv are sane before
    # hours of engine building, and the reference to compare the finished bundle's frames against.
    image.save(out / "calib_reference.png")
    ref = np.asarray(image).astype(np.float32)
    print(f"[calib] calib_reference.png {image.size} mean={ref.mean():.1f} std={ref.std():.2f}")
    if ref.std() < 1.0:
        raise RuntimeError("reference image is flat -- the pipeline produced no signal")

    lp = (height // 16) * (width // 16)
    (out / "calib.json").write_text(json.dumps({
        "model_dir": str(model_dir), "width": width, "height": height,
        "Lp": lp, "Lt": text_len, "steps": STEPS, "seed": SEED, "prompt": PROMPT,
        "hidden_states_layers": list(HIDDEN_LAYERS), "vae_batch_norm_eps": bn_eps,
        "image_mean": float(ref.mean()), "image_std": float(ref.std()),
    }, indent=2))
    print(f"[calib] DONE -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
