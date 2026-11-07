"""FLUX.3-image-klein ONNX export (distilled flow-map checkpoint, one sample, no CFG).

  transformer : img[1,Li,128] bf16, cond[1,Lc,128] bf16, txt[1,Lt,20480] bf16,
                rope_img[1,Li,64,2,2], rope_cond[1,Lc,64,2,2], rope_txt[1,Lt,64,2,2] fp32,
                t[1], r[1] fp32  ->  velocity[1,Li,128] bf16
                RoPE rotation tables are host-computed in fp64 (the reference builds them in fp64);
                text and cond run at timestep 0, the image at t; r is the flow-map target.
  qwen3vl     : input_ids[1,Lt] int64, attention_mask[1,Lt] int64 -> encoder_hidden_states[1,Lt,20480] bf16
                (named as klein.2's so Qwen3EncoderWrapper binds it unchanged)
                (hidden layers 4..32 of the Qwen3-VL text tower, positions as the reference builds them)
  vae         : same graphs as klein.2 (AutoencoderKLFlux2), new weights.

The DiT is BFL's reference module (bfl/flux3/dit.py, or FLUX3_REF's checkout) re-driven densely for one sample.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# BFL's inference code: the vendored copy, or a checkout of flux-3-image-klein
REF = Path(os.environ.get("FLUX3_REF") or Path(__file__).resolve().parent / "bfl")
sys.path.insert(0, str(REF / "src" if (REF / "src" / "flux3").is_dir() else REF))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "klein"))

from flux3 import dit as fdit  # noqa: E402

OUT = Path(os.environ.get("FLUX3_ONNX_DIR", "./onnx-flux3"))
W = int(os.environ.get("FLUX3_WIDTH", "512"))
H = int(os.environ.get("FLUX3_HEIGHT", "512"))
LT = int(os.environ.get("FLUX3_TEXT_LEN", "160"))
LI = (H // 16) * (W // 16)
DEV = "cuda"
DT = torch.bfloat16
TXT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


def snapshot() -> Path:
    if d := os.environ.get("FLUX3_MODEL_DIR"):
        return Path(d)
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    found = glob.glob(str(hub / "models--black-forest-labs--flux-3-image-klein" / "snapshots" / "*"))
    if not found:
        raise SystemExit("flux-3-image-klein not in the HF cache; set FLUX3_MODEL_DIR")
    return Path(found[0])


def _layernorm_modulation(x, scale, shift):
    return (1.0 + scale) * F.layer_norm(x, (fdit_hidden,), eps=1e-6) + shift


# the reference sizes layer_norm from x.shape[-1], which the TorchScript exporter cannot take
fdit_hidden = fdit.HIDDEN
fdit.layernorm_modulation = _layernorm_modulation


def rope_table(ids: torch.Tensor) -> torch.Tensor:
    """ids [L,4] -> [1,L,64,2,2] fp32: fdit.rope_embed's rotation per token."""
    return fdit.rope_embed(ids[None].to(DEV))[0].contiguous()


def _block(block: fdit.Block, x, freqs, mod):
    shift, scale, gate = mod
    x_mod = fdit.layernorm_modulation(x, scale, shift)
    heads = block.heads
    split = lambda t: t.reshape(1, t.shape[1], heads, t.shape[-1] // heads).transpose(1, 2)
    q, k = block.norm(split(block.q_proj(x_mod)), split(block.k_proj(x_mod)), freqs)
    v = split(block.v_proj(x_mod)).to(q.dtype)
    attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, x.shape[1], x.shape[-1])
    gate_in, up = block.mlp_in(x_mod).chunk(2, dim=-1)
    return x + gate * (block.attn_out(attn) + block.mlp_out(F.silu(gate_in) * up))


class DenseDiT(torch.nn.Module):
    """fdit.DiT.forward for one sample, all tensors [1,L,...]. Matches the reference under bf16 autocast."""

    def __init__(self, dit: fdit.DiT):
        super().__init__()
        self.dit = dit

    def forward(self, img, cond, txt, rope_img, rope_cond, rope_txt, t, r):
        d = self.dit
        with torch.autocast("cuda", dtype=DT):
            tb = t.to(DT)  # the reference stores per-mode timesteps in bf16
            zero = torch.zeros_like(tb)
            ref = d.ref_time_in(fdit.timestep_embedding(r.to(DT)))
            vec = {"image": d.time_in(fdit.timestep_embedding(tb)) + ref}
            vec["image_cond"] = vec["text"] = d.time_in(fdit.timestep_embedding(zero)) + ref
            vec = {m: v[:, None, :] for m, v in vec.items()}
            key = lambda m: "txt" if m == "text" else m
            early = {m: d.early_stream_modulations[key(m)](vec[m]) for m in fdit.MODES}
            single = {m: d.single_stream_modulations[key(m)](vec[m]) for m in fdit.MODES}
            x = {"image": d.emb_in["image"](img), "image_cond": d.emb_in["image_cond"](cond), "text": d.txt_in(txt)}
            # unsqueeze, not reshape: ONNX Reshape reads a 0 (empty reference) as copy-input-dim
            fr = {m: f.unsqueeze(1) for m, f in
                  (("image", rope_img), ("image_cond", rope_cond), ("text", rope_txt))}
            for i in range(fdit.DEPTH):
                for m in ("image", "image_cond"):
                    x[m] = _block(d.content_mode_blocks[m][i], x[m], fr[m], early[m])
                x["text"] = _block(d.txt_mode_blocks[i], x["text"], fr["text"], early["text"])
            lens = [x[m].shape[1] for m in fdit.MODES]
            h = torch.cat([x[m] for m in fdit.MODES], dim=1)
            freqs = torch.cat([fr[m] for m in fdit.MODES], dim=2)
            mods = tuple(torch.cat([single[m][j].expand(-1, n, -1) for m, n in zip(fdit.MODES, lens)], dim=1)
                         for j in range(3))
            for block in d.single_blocks:
                h = _block(block, h, freqs, mods)
            out = d.final_layer["image"](h[:, : lens[0]], vec["image"])
        return out.to(DT)


def load_dit(root: Path) -> fdit.DiT:
    from flux3.checkpoint import load_checkpoint
    with torch.device("meta"):
        m = fdit.DiT(flow_map=True)
    return load_checkpoint(m, str(root), DEV, DT).eval()


def selftest(dense: DenseDiT, golden: Path) -> None:
    """Re-run golden step 0 through DenseDiT and compare with the reference velocity."""
    g = lambda n: torch.from_numpy(np.load(golden / f"{n}.npy")).to(DEV)
    lc = golden / "cond_tokens.npy"
    cond = g("cond_tokens")[None].to(DT) if lc.exists() else torch.zeros(1, 0, 128, device=DEV, dtype=DT)
    cond_ids = g("cond_ids") if lc.exists() else torch.zeros(0, 4, device=DEV)
    sched = g("schedule")
    with torch.inference_mode():
        v = dense(g("step0_x")[None].to(DT), cond, g("txt_tokens")[None].to(DT),
                  rope_table(g("img_ids")), rope_table(cond_ids), rope_table(g("txt_ids")),
                  sched[0:1].float(), sched[1:2].float())
    ref = g("step0_v")[None]
    err = (v.float() - ref).abs()
    print(f"selftest {golden.name}: max {err.max():.4g} mean {err.mean():.4g} ref-std {ref.std():.4g}")


def export_transformer(root: Path, golden: list[Path]) -> None:
    dense = DenseDiT(load_dit(root)).eval()
    for gdir in golden:
        selftest(dense, gdir)
    lc = LI
    args = (torch.randn(1, LI, 128, device=DEV, dtype=DT), torch.randn(1, lc, 128, device=DEV, dtype=DT),
            torch.randn(1, LT, 20480, device=DEV, dtype=DT),
            torch.randn(1, LI, 64, 2, 2, device=DEV), torch.randn(1, lc, 64, 2, 2, device=DEV),
            torch.randn(1, LT, 64, 2, 2, device=DEV),
            torch.tensor([0.9], device=DEV), torch.tensor([0.8], device=DEV))
    names = ["img", "cond", "txt", "rope_img", "rope_cond", "rope_txt", "t", "r"]
    dst = OUT / "transformer"
    dst.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            dense, args, str(dst / "model.onnx"), input_names=names, output_names=["velocity"],
            dynamic_axes={"img": {1: "Li"}, "cond": {1: "Lc"}, "txt": {1: "Lt"}, "rope_img": {1: "Li"},
                          "rope_cond": {1: "Lc"}, "rope_txt": {1: "Lt"}, "velocity": {1: "Li"}},
            opset_version=20, do_constant_folding=False, dynamo=False)
    print("transformer ONNX ->", dst / "model.onnx")


class Qwen3VLText(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, input_ids, attention_mask):
        positions = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 1)
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask,
                      position_ids=positions[None].expand(4, -1, -1), output_hidden_states=True, use_cache=False)
        stacked = torch.stack([out.hidden_states[k] for k in TXT_LAYERS], dim=1)
        b, n, l, hd = stacked.shape
        return stacked.permute(0, 2, 1, 3).reshape(b, l, n * hd)


def export_qwen(root: Path, golden: list[Path]) -> None:
    from transformers import Qwen3VLForConditionalGeneration
    full = Qwen3VLForConditionalGeneration.from_pretrained(
        root / "text_encoder", torch_dtype=DT, attn_implementation="eager").to(DEV).eval()
    lm = full.model.language_model
    # hidden_states[32] must stay a raw layer output: keep layer 33 so the final norm lands on [33].
    lm.layers = lm.layers[: max(TXT_LAYERS) + 1]
    w = Qwen3VLText(lm).eval()
    for gdir in golden:
        ids = torch.from_numpy(np.load(gdir / "input_ids.npy"))[None].to(DEV)
        mask = torch.from_numpy(np.load(gdir / "attention_mask.npy"))[None].to(DEV)
        with torch.inference_mode():
            e = w(ids, mask)[0].float()
        ref = torch.from_numpy(np.load(gdir / "txt_tokens.npy")).to(DEV)
        err = (e - ref).abs()
        print(f"qwen selftest {gdir.name}: max {err.max():.4g} mean {err.mean():.4g} ref-std {ref.std():.4g}")
    ids = torch.ones(1, LT, dtype=torch.int64, device=DEV)
    mask = torch.ones(1, LT, dtype=torch.int64, device=DEV)
    dst = OUT / "qwen3vl_encoder"
    dst.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            w, (ids, mask), str(dst / "model.onnx"), input_names=["input_ids", "attention_mask"],
            output_names=["encoder_hidden_states"],
            dynamic_axes={"input_ids": {1: "Lt"}, "attention_mask": {1: "Lt"}, "encoder_hidden_states": {1: "Lt"}},
            opset_version=20, do_constant_folding=False, dynamo=False)
    print("qwen3vl ONNX ->", dst / "model.onnx")


def export_vae(root: Path) -> None:
    os.environ["KLEIN_MODEL_DIR"] = str(root)
    os.environ["KLEIN_ONNX_DIR"] = str(OUT)
    os.environ["KLEIN_WIDTH"], os.environ["KLEIN_HEIGHT"] = str(W), str(H)
    import export_klein
    export_klein.export_vae()
    from diffusers import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_pretrained(root, subfolder="vae")
    bn = vae.bn
    mean = bn.running_mean.float().numpy()
    std = np.sqrt(bn.running_var.float().numpy() + vae.config.batch_norm_eps)
    mean.astype(np.float32).tofile(OUT / "bn_mean.bin")
    std.astype(np.float32).tofile(OUT / "bn_std.bin")
    import shutil
    shutil.copy(root / "text_encoder/tokenizer.json", OUT / "tokenizer.json")
    print("bn stats + tokenizer ->", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["all", "transformer", "qwen", "vae"])
    ap.add_argument("--golden", default="", help="comma-separated golden case dirs for the self-tests")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    root = snapshot()
    golden = [Path(p) for p in a.golden.split(",") if p]
    print(f"[flux3] {W}x{H} Li={LI} Lt={LT} -> {OUT}")
    if a.which in ("all", "vae"):
        export_vae(root)
    if a.which in ("all", "qwen"):
        export_qwen(root, golden)
    if a.which in ("all", "transformer"):
        export_transformer(root, golden)
    print("DONE")
