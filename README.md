# librediffusion

A C++ / CUDA / TensorRT implementation of StreamDiffusion

Implemented in [ossia score](https://ossia.io)

# Benchmarks

On a RTX 5090 at 1 step: 

SDXL Turbo 1024x1024: stable 26 fps

![sdxl](https://github.com/user-attachments/assets/340bf804-6822-46c6-87d8-bf784722f3b5)

SD Turbo 512x512: stable 96 fps

![sdturbo](https://github.com/user-attachments/assets/10ebc9f7-d5b7-487b-a301-b1b7e2370d55)

SDXS: above 600 fps

![sdxs](https://github.com/user-attachments/assets/9f735f86-d162-4c4c-b781-163cadf166a5)

Models need to be converted to TensorRT through the Python script [train-lora.py] beforehand:

```bash
$ uv run train-lora.py --model stabilityai/sd-turbo --min-batch 1 --max-batch 1 --opt-batch 1 --min-resolution 512 --max-resolution 1024 --output ./engines-sd-turbo
```

## Engine build recipes (validated)

All commands run as `uv run python train-lora.py <args>`. Resolution flags are
`--min-resolution/--max-resolution/--opt-width/--opt-height` (512 for SD1.5, 1024 for SDXL);
add `-l REPO` (or `-l "REPO|file.safetensors"`) to fuse a LoRA, repeatable to stack LoRAs.

### On Windows: pass `--cache-dir` on the build volume

uv's cache defaults to `%LOCALAPPDATA%\uv\cache` on `C:`. The engine-build venv is ~10 GB of
torch/TensorRT wheels, and if you build anywhere other than `C:` — which you will, because the
engines and the HF snapshot need tens of GB — uv cannot hardlink out of its cache across the volume
boundary and copies every file instead:

```
warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
         If the cache and target directories are on different filesystems, hardlinking may not be supported.
```

So you pay for the wheels twice on disk and wait through the copy. Point the cache at the volume you
are building on, and keep passing it on every later `uv` call in the same tree:

```bat
uv sync    --cache-dir D:\tmp\uvcache
uv run     --cache-dir D:\tmp\uvcache python train-lora.py --type sd15 --model stabilityai/sd-turbo ...
```

`set UV_CACHE_DIR=D:\tmp\uvcache` does the same thing for a whole shell session. The HF snapshot has
its own cache: `set HF_HOME=D:\tmp\hf` (or `hf download --cache-dir D:\tmp\hf\hub`).

Koaia's packaged builder already passes `--cache-dir c:\uv` on Windows for both the sync and the
build (`qml/koaia/Views/ModelView.qml`), so engines built through the app land on the same cache; a
short root-level path also stays clear of `MAX_PATH` on machines without long paths enabled. Match
it if you want to share one cache between the app and a command-line build.

### Base models

```bash
# SD1.5 family (--type sd15)
train-lora.py --type sd15 --model stabilityai/sd-turbo                 --min-resolution 512 --max-resolution 512  --output engines/sd-turbo
train-lora.py --type sd15 --model SimianLuo/LCM_Dreamshaper_v7         --min-resolution 512 --max-resolution 512  --output engines/lcm-dreamshaper
train-lora.py --type sd15 --model Lykon/dreamshaper-8-lcm              --min-resolution 512 --max-resolution 512  --output engines/dreamshaper-8-lcm
train-lora.py --type sd15 --model IDKiro/sdxs-512-dreamshaper          --min-resolution 512 --max-resolution 512  --output engines/sdxs
train-lora.py --type sd15 --model runwayml/stable-diffusion-v1-5 -l latent-consistency/lcm-lora-sdv1-5            --min-resolution 512 --max-resolution 512 --output engines/sd15-lcm-lora
train-lora.py --type sd15 --model runwayml/stable-diffusion-v1-5 -l "ByteDance/Hyper-SD|Hyper-SD15-4steps-lora.safetensors" --min-resolution 512 --max-resolution 512 --output engines/hyper-sd15

# SDXL family (--type sdxl)
train-lora.py --type sdxl --model stabilityai/sdxl-turbo                            --min-resolution 1024 --max-resolution 1024 --output engines/sdxl-turbo
train-lora.py --type sdxl --model stabilityai/stable-diffusion-xl-base-1.0 -l latent-consistency/lcm-lora-sdxl   --min-resolution 1024 --max-resolution 1024 --output engines/sdxl-lcm-lora
train-lora.py --type sdxl --model stabilityai/stable-diffusion-xl-base-1.0 -l "ByteDance/Hyper-SD|Hyper-SDXL-4steps-lora.safetensors" --min-resolution 1024 --max-resolution 1024 --output engines/hyper-sdxl
train-lora.py --type sdxl --model stabilityai/stable-diffusion-xl-base-1.0 -l "ByteDance/SDXL-Lightning|sdxl_lightning_4step_lora.safetensors" --min-resolution 1024 --max-resolution 1024 --output engines/sdxl-lightning
train-lora.py --type sdxl --model segmind/Segmind-Vega -l segmind/Segmind-VegaRT    --min-resolution 1024 --max-resolution 1024 --output engines/vega-rt

# img2img-turbo (GaParmar pix2pix-turbo skip-VAE)
train-lora.py --type img2img-turbo --model edge_to_image --min-resolution 512 --max-resolution 512 --output engines/img2img-turbo
```

### FLUX.2-klein-4B (`--type klein`)

Download the weights first; the single-file `.safetensors` is a second copy of the same transformer
that this export does not read, so excluding it saves 7.3 GB of the 23 GB:

```bash
hf download black-forest-labs/FLUX.2-klein-4B --exclude flux-2-klein-4b.safetensors
train-lora.py --type klein --model black-forest-labs/FLUX.2-klein-4B --output engines/klein
```

One command, same as the models above: it runs the reference pipeline once to get the FP8
calibration activations and the VAE normalisation constants, exports the three ONNX graphs, patches
them (`fix_qwen_complex` for the Qwen COMPLEX128 casts, `fix_klein_dynamic_seq` for the baked
sequence length), builds the bf16 and FP8 transformers, the Qwen encoder and both VAE halves, builds
RIFE from the vendored ONNX, and stages a bundle the node loads directly. It refuses to finish if any
piece is missing. Nothing outside this repo and the HF snapshot is needed; `--model` also takes a
local snapshot directory.

Expect ~18 GB of engines (the bf16 transformer alone is 7.8 GB) plus the ONNX intermediates, and a
long build — the transformer and Qwen engines dominate. Steps are idempotent enough to resume: a
rerun reuses an existing calibration directory.

- `--klein-quality speed` builds only the FP8 transformer, `quality` only bf16, `both` (default) both.
  **FP8 needs an Ada (SM 8.9) or newer GPU** — TensorRT refuses to build FP8 Q/DQ engines on Ampere
  and earlier (a 3090 is SM 8.6, a 4090 is 8.9). On such a card the default quietly drops to the bf16
  transformer; pass `--klein-quality quality` to say so explicitly, or `speed`/`both` to make the
  mismatch an error instead.
- Resolution defaults to klein's native 320x576. `--opt-width/--opt-height` (multiples of 16) build
  another geometry; cost scales with `(W/16)*(H/16)` tokens, so 1024x1024 is ~5.7x the 320x576 work.
- `--hw-compat ampere_plus` makes the engines portable across SM 8.0+ instead of locked to the
  build GPU.

### ControlNet (`--controlnet REPO`)

The ControlNet must match the base UNet's block/residual count: standard SD1.5/SDXL ControlNets
for standard models, and the model-native ControlNet for reduced architectures (e.g. sdxs).

```bash
# SD1.5 (12+mid residuals) — works on any standard SD1.5 base (swap --model/-l)
train-lora.py --type sd15 --model SimianLuo/LCM_Dreamshaper_v7 --controlnet lllyasviel/control_v11p_sd15_canny --min-resolution 512 --max-resolution 512 --output engines/lcm-dreamshaper-canny
# other SD1.5 ControlNets: control_v11f1p_sd15_depth, control_v11p_sd15_openpose, control_v11p_sd15_softedge

# SDXL (works on any SDXL base, incl. LoRA-fused)
train-lora.py --type sdxl --model stabilityai/stable-diffusion-xl-base-1.0 -l latent-consistency/lcm-lora-sdxl --controlnet diffusers/controlnet-canny-sdxl-1.0 --min-resolution 1024 --max-resolution 1024 --output engines/sdxl-lcm-lora-canny

# sdxs needs its NATIVE sketch ControlNet (7 residuals) — generic SD1.5 canny (13) is incompatible
train-lora.py --type sd15 --model IDKiro/sdxs-512-dreamshaper --controlnet IDKiro/sdxs-512-dreamshaper-sketch --min-resolution 512 --max-resolution 512 --output engines/sdxs-sketch
```

### IP-Adapter (`--ipadapter CKPT`)

```bash
# SD1.5 (h94 ip-adapter_sd15.bin) — works on standard SD1.5 bases (NOT sdxs, whose attention differs)
train-lora.py --type sd15 --model SimianLuo/LCM_Dreamshaper_v7 --ipadapter /path/to/h94/IP-Adapter/models/ip-adapter_sd15.bin --min-resolution 512 --max-resolution 512 --output engines/lcm-dreamshaper-ip
# SDXL (h94 sdxl_models/ip-adapter_sdxl_vit-h.bin)
train-lora.py --type sdxl --model stabilityai/sdxl-turbo --ipadapter /path/to/h94/IP-Adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin --min-resolution 1024 --max-resolution 1024 --output engines/sdxl-turbo-ip
```

### Merged / stacked LoRAs (`-l` repeated)

Any base + accel LoRA + arbitrary style LoRA(s); LoRAs are fused at export time.

```bash
# accel (LCM) + a style LoRA, stacked
train-lora.py --type sdxl --model stabilityai/stable-diffusion-xl-base-1.0 \
  -l latent-consistency/lcm-lora-sdxl \
  -l "ostris/crayon_style_lora_sdxl|crayons_v1_sdxl.safetensors" \
  --min-resolution 1024 --max-resolution 1024 --output engines/sdxl-lcm-crayon

# single style LoRA on a turbo base
train-lora.py --type sdxl --model stabilityai/sdxl-turbo \
  -l "ostris/ikea-instructions-lora-sdxl|ikea_instructions_xl_v1_5.safetensors" \
  --min-resolution 1024 --max-resolution 1024 --output engines/sdxlturbo-ikea
```

### Optional flags

- `--fp8` — FP8 (e4m3) UNet quantization (SDXL/SD; diffusion recipe).
- `--v2v` / `--v2v-inject` — StreamV2V kvo extended-attention UNet (SD1.5 only).
- `--hw-compat ampere_plus` — portable engine across SM 8.0+ GPUs (~5–15% slower, larger).
- `-l "REPO:WEIGHT"` — set LoRA fusion scale; `--lora-scale` for the global scale.
