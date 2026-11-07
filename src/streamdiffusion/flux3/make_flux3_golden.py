"""FLUX.3-image-klein goldens from BFL's reference code (github.com/black-forest-labs/flux-3-image-klein).

Dumps every seam the C++ port has to match, per case, as .npy under OUT/<case>/:
  input_ids, attention_mask       [L] int64   (chat-templated, right-padded to a multiple of 80)
  txt_tokens                      [L,20480]   bf16 stored as fp32
  txt_ids                         [L,4]
  cond_tokens, cond_ids           [Lc,128], [Lc,4]  (edit cases; references concatenated, l = -(1+i))
  ref_rgb / ref{i}_rgb            [3,H,W]     each reference fed to the encoder, in [-1,1]
  ref_hw                          [n,2] int64 (th, tw) per reference
  noise, img_ids                  [Li,128], [Li,4]
  schedule                        [steps+1]
  rope_img/rope_cond/rope_txt     [L,64,4]    fp64-built rotation tables, flattened (2,2)
  step{i}_x, step{i}_v            DiT input / velocity per step
  final_tokens                    [Li,128]
  image.png, image                [3,H,W] decoded, in [-1,1]

Usage: python make_flux3_golden.py --ref /media/data1/flux3/ref --out /media/data1/flux3/golden
       python make_flux3_golden.py --ref ... --out /media/data1/flux3/calib --calib
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CASES = [
    # name, prompt, reference, size, steps, alpha
    ("t2i_512_s16", "a red fox sitting in a snowy forest at dawn, photograph", None, 512, 16, 6.93),
    ("t2i_512_s4", "a red fox sitting in a snowy forest at dawn, photograph", None, 512, 4, 6.93),
    ("t2i_512_s16_a16", "a red fox sitting in a snowy forest at dawn, photograph", None, 512, 16, 16.0),
    ("edit_512_s16", "put a red bandana around the dog's neck", "examples/inputs/3.png", 512, 16, 6.93),
    ("edit_512_s4", "put a red bandana around the dog's neck", "examples/inputs/3.png", 512, 4, 6.93),
    ("edit_512_s1", "put a red bandana around the dog's neck", "examples/inputs/3.png", 512, 1, 6.93),
    # multi-reference: (path, width, height) each, the second non-square
    ("multi_512_s4", "print the dog from the second image onto the mug",
     [("examples/inputs/2.png", 512, 512), ("examples/inputs/3.png", 384, 512)], 512, 4, 6.93),
]
# FP8 calibration traces (--calib), disjoint from the validation cases above
CALIB = [
    ("calib_t2i_city", "a rainy city street at night, neon signs reflected in puddles", None, 512, 8, 6.93),
    ("calib_t2i_portrait", "studio portrait of an old fisherman, dramatic side lighting", None, 512, 8, 6.93),
    ("calib_edit_style", "turn this into a watercolor painting", "examples/inputs/1.png", 512, 8, 6.93),
]
SEED = 1


def snapshot() -> Path:
    if d := os.environ.get("FLUX3_MODEL_DIR"):
        return Path(d)
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    found = glob.glob(str(hub / "models--black-forest-labs--flux-3-image-klein" / "snapshots" / "*"))
    if not found:
        raise SystemExit("flux-3-image-klein not in the HF cache; set FLUX3_MODEL_DIR")
    return Path(found[0])


def save(d: Path, name: str, t) -> None:
    a = t.detach().float().cpu().contiguous().numpy() if torch.is_tensor(t) else np.ascontiguousarray(t)
    if torch.is_tensor(t) and t.dtype == torch.int64:
        a = t.cpu().numpy()
    np.save(d / f"{name}.npy", a)


def center_fit(path: Path, w: int, h: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    s = min(im.width / w, im.height / h)
    cw, ch = round(w * s), round(h * s)
    left, top = (im.width - cw) // 2, (im.height - ch) // 2
    return im.crop((left, top, left + cw, top + ch)).resize((w, h), Image.Resampling.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cases", default="", help="comma-separated subset")
    ap.add_argument("--calib", action="store_true", help="dump the FP8 calibration cases instead")
    a = ap.parse_args()
    sys.path.insert(0, str(a.ref / "src"))
    from flux3.autoencoder import AutoEncoder, latent_ids, latent_shape
    from flux3.checkpoint import load_checkpoint
    from flux3.dit import DiT, IN_CHANNELS, rope_embed
    from flux3.pipeline import pack, scaled_ids, to_pil, to_tensor
    from flux3.sampling import timesteps
    from flux3.text_encoder import TextEncoder

    cases = [c for c in (CALIB if a.calib else CASES) if not a.cases or c[0] in a.cases.split(",")]
    root = snapshot()
    dev = "cuda"
    meta = {"snapshot": str(root), "seed": SEED, "cases": {}}

    te = TextEncoder(str(root / "text_encoder"), dev)
    tok = te.processor.tokenizer
    texts = {}
    for name, prompt, *_ in cases:
        if prompt in texts:
            continue
        chat = te.processor.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                add_generation_prompt=True)
        tokens, ids = te.encode(prompt)
        enc = tok(chat, return_tensors="pt", padding="max_length", truncation=True, max_length=tokens.shape[0])
        texts[prompt] = (chat, enc["input_ids"][0], enc["attention_mask"][0], tokens.cpu(), ids.cpu())
    del te
    gc.collect()
    torch.cuda.empty_cache()

    ae = load_checkpoint(AutoEncoder(), str(root / "ae.safetensors"), dev, torch.float32)
    with torch.device("meta"):
        dit = DiT(flow_map=True)
    dit = load_checkpoint(dit, str(root), dev, torch.bfloat16)

    for name, prompt, ref, size, steps, alpha in cases:
        d = a.out / name
        d.mkdir(parents=True, exist_ok=True)
        chat, input_ids, mask, txt_tokens, txt_ids = texts[prompt]
        save(d, "input_ids", input_ids)
        save(d, "attention_mask", mask)
        save(d, "txt_tokens", txt_tokens)
        save(d, "txt_ids", txt_ids)
        (d / "chat.txt").write_text(chat)

        cond = None
        refs = [] if not ref else [(ref, size, size)] if isinstance(ref, str) else ref
        if refs:
            # pipeline.py:encode_references
            tokens, cids, hw = [], [], []
            for index, (path, rw, rh) in enumerate(refs):
                im = center_fit(a.ref / path, rw, rh)
                im.save(d / ("ref.png" if len(refs) == 1 else f"ref{index}.png"))
                rgb = to_tensor(im).to(dev)
                save(d, "ref_rgb" if len(refs) == 1 else f"ref{index}_rgb", rgb)
                t, i = ae.encode(rgb)
                i = scaled_ids(i)
                i[:, 3] = -(1 + index)
                tokens.append(t.to(torch.bfloat16))
                cids.append(i)
                hw.append((rh // 16, rw // 16))
            cond = (torch.cat(tokens), torch.cat(cids))
            save(d, "cond_tokens", cond[0])
            save(d, "cond_ids", cond[1])
            save(d, "ref_hw", torch.tensor(hw, dtype=torch.int64))

        h, w = latent_shape(size, size)
        noise = torch.randn((h * w, IN_CHANNELS["image"]), generator=torch.Generator().manual_seed(SEED))
        noise = noise.to(torch.bfloat16).float().to(dev)
        ids = latent_ids(h, w, dev)
        save(d, "noise", noise)
        save(d, "img_ids", scaled_ids(ids))

        schedule = timesteps(steps, alpha)
        save(d, "schedule", schedule)
        save(d, "rope_img", rope_embed(scaled_ids(ids)[None])[0, 0].reshape(-1, 64, 4))
        save(d, "rope_txt", rope_embed(txt_ids.to(dev)[None])[0, 0].reshape(-1, 64, 4))
        if cond is not None:
            save(d, "rope_cond", rope_embed(cond[1][None])[0, 0].reshape(-1, 64, 4))
        packed = pack([[(txt_tokens.to(dev), txt_ids.to(dev)), cond, (noise, scaled_ids(ids))]])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(steps):
                t, dt = schedule[s].item(), (schedule[s + 1] - schedule[s]).item()
                packed.timesteps["image"].fill_(t)
                target = schedule[s + 1].reshape(1).to(dev)
                save(d, f"step{s}_x", packed.data["image"])
                v = dit(packed, timesteps_ref=target)
                save(d, f"step{s}_v", v)
                packed.data["image"] = packed.data["image"] + dt * v
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000
        tokens = packed.data["image"]
        save(d, "final_tokens", tokens)
        img = ae.decode(tokens, ids)
        save(d, "image", img)
        to_pil(img).save(d / "image.png")
        meta["cases"][name] = {"prompt": prompt, "ref": ref, "size": size, "steps": steps, "alpha": alpha,
                               "Lt": int(txt_tokens.shape[0]), "Lc": 0 if cond is None else int(cond[0].shape[0]),
                               "Li": int(h * w), "denoise_ms": dt_ms,
                               "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        print(name, meta["cases"][name], flush=True)

    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
