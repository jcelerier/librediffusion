#!/usr/bin/env python3
"""PyTorch reference for one 1-step img2img frame of an SD1.5 / SDXL bundle, to check what the C++
pipeline (validation_cpp/sd_video_harness, the node) computes for the same frame.

Rebuilds the model the bundle was exported from out of its bundle.json (base_model + unet +
unet_timestep_scale; LoRA bundles are not supported) and runs the same math as the C++ single step:

    latent = TAESD(XL).encode(frame)                       noise injection at the noise-level t
    x_t    = sqrt(a_t) * latent + sqrt(1 - a_t) * noise    noise = the C++ init noise (PCG32 replica)
    eps    = unet(x_t, t_unet, prompt)                     t_unet = floor(t * unet_timestep_scale)
    x0     = c_out(t_unet) * (x_t - sqrt(1-a)*eps)/sqrt(a) + c_skip(t_unet) * x_t,   a = a(t_unet)
    image  = TAESD(XL).decode(x0)

The coefficients come from diffusers' LCMScheduler, independently of the C++ tables and of
src/schedulers/schedule.hpp, so a match also checks those.

  bench/reference_img2img.py --bundle B --frame first.png --index 32 --out ref.png [--compare cpp.png]
"""
import argparse, json, math, os, sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from streamdiffusion.deterministic_noise import pcg32_randn_numpy  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--frame", required=True, help="input image, already at the bundle's resolution")
    ap.add_argument("--index", type=int, default=32, help="scheduler table index: t = 999 - 20 * index")
    ap.add_argument("--prompt", default="a vivid impressionist oil painting, bold brushstrokes, vibrant saturated color")
    ap.add_argument("--seed", type=int, default=42, help="the C++ config seed (sd_video_harness default)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", help="C++ output for the same frame: prints PSNR against it")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from diffusers import (AutoencoderTiny, LCMScheduler, StableDiffusionPipeline,
                           StableDiffusionXLPipeline)
    m = json.load(open(os.path.join(a.bundle, "bundle.json")))
    if m.get("loras"):
        sys.exit("LoRA bundles are not supported by this reference")
    sdxl = m["model_type"] == "sdxl"
    dev, dt = a.device, (torch.float16 if a.device == "cuda" else torch.float32)

    Pipe = StableDiffusionXLPipeline if sdxl else StableDiffusionPipeline
    pipe = Pipe.from_pretrained(m["base_model"], torch_dtype=dt).to(dev)
    if m.get("unet"):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_lora", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train-lora.py"))
        tl = importlib.util.module_from_spec(spec)
        sys.argv = sys.argv[:1]
        spec.loader.exec_module(tl)
        tl.load_unet_weights(pipe.unet, m["unet"])
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl" if sdxl else "madebyollin/taesd",
                                          torch_dtype=dt).to(dev)
    sched = LCMScheduler.from_pretrained(m["base_model"], subfolder="scheduler")

    img = Image.open(a.frame).convert("RGB")
    W, H = img.size
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].to(dev, dt) / 127.5 - 1

    t = 999 - 20 * a.index
    t_unet = math.floor(t * float(m.get("unet_timestep_scale") or 1.0))
    ac = sched.alphas_cumprod
    a_noise, a_unet = float(ac[t]), float(ac[t_unet])
    c_skip, c_out = (float(v) for v in sched.get_scalings_for_boundary_condition_discrete(t_unet))
    print(f"t={t} t_unet={t_unet} sqrt(a_noise)={math.sqrt(a_noise):.6f} sqrt(a_unet)={math.sqrt(a_unet):.6f} "
          f"c_skip={c_skip:.3g} c_out={c_out:.6f}")

    with torch.no_grad():
        lat = vae.encode(x).latents
        n = torch.from_numpy(pcg32_randn_numpy(a.seed, lat.numel()).astype(np.float32)).view_as(lat).to(dev, dt)
        x_t = math.sqrt(a_noise) * lat + math.sqrt(1 - a_noise) * n
        if sdxl:
            pe, _, ppe, _ = pipe.encode_prompt(a.prompt, device=dev, do_classifier_free_guidance=False)
            tids = torch.tensor([[H, W, 0, 0, H, W]], device=dev, dtype=dt)
            eps = pipe.unet(x_t, t_unet, encoder_hidden_states=pe,
                            added_cond_kwargs={"text_embeds": ppe, "time_ids": tids}).sample
        else:
            pe, _ = pipe.encode_prompt(a.prompt, dev, 1, False)
            eps = pipe.unet(x_t, t_unet, encoder_hidden_states=pe).sample
        x0 = (x_t - math.sqrt(1 - a_unet) * eps) / math.sqrt(a_unet)
        x0 = c_out * x0 + c_skip * x_t
        out = vae.decode(x0).sample.clamp(-1, 1)
    arr = ((out[0].float().permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
    Image.fromarray(arr).save(a.out)
    print(f"wrote {a.out}")
    if a.compare:
        ref = np.asarray(Image.open(a.compare).convert("RGB")).astype(np.float64)
        mse = ((ref - arr.astype(np.float64)) ** 2).mean()
        print(f"PSNR vs {a.compare}: {10 * math.log10(255 ** 2 / max(mse, 1e-12)):.2f} dB")


if __name__ == "__main__":
    main()
