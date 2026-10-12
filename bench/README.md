# Fast-model video benchmark

Compares the 1-2 step models librediffusion can build, on real video, at 1 and 2 denoising steps:

| bundle | what | build |
|---|---|---|
| `sdxl-turbo` | baseline (ADD) | `--type sdxl --model stabilityai/sdxl-turbo` |
| `dmd2-sdxl-1step` | DMD2 1-step UNet on SDXL base | `--unet "tianweiy/DMD2\|dmd2_sdxl_1step_unet_fp16.bin"` |
| `dmd2-sdxl-4step` | DMD2 4-step UNet on SDXL base | `--unet "tianweiy/DMD2\|dmd2_sdxl_4step_unet_fp16.bin"` |
| `nitrosd-realism` | NitroFusion Realism UNet | `--unet "ChenDY/NitroFusion\|nitrosd-realism_unet.safetensors" --unet-timestep-scale 0.25` |
| `nitrosd-vibrant` | NitroFusion Vibrant UNet | `--unet "ChenDY/NitroFusion\|nitrosd-vibrant_unet.safetensors" --unet-timestep-scale 0.5` |
| `klein` | FLUX.2-klein-4B baseline | `--type klein` |
| `klein-sprint` | klein + SANA-Sprint LoRA | `--type klein -l radames/FLUX.2-klein-Sana-Sprint` |

All SDXL bundles share one profile (1024x1024, batch 1..2) so both step counts run on the same
engine and the models differ only in their UNet weights. klein runs at its 320x576 streaming geometry.

```bash
# 1. engines (sequential; ~12 min per SDXL bundle on a 4090, klein takes much longer)
HF_HOME=... bench/build_fast_models.sh /media/data2/lrd-engines-fast
ln -s /path/to/an/existing/klein/bundle /media/data2/lrd-engines-fast/klein   # or build it too

# 2. harnesses (validation_cpp: sd_video_harness, klein_fps_harness)
cmake -S validation_cpp -B build-harness -G Ninja -DLIBREDIFFUSION_BUILD_DIR=<lib build dir>
ninja -C build-harness sd_video_harness klein_fps_harness

# 3. run: clips/, compare_*.mp4, results.json, results.md
bench/run_video_bench.py --engines /media/data2/lrd-engines-fast --video dance.mp4 \
    --harness-dir build-harness --out /media/data3/lrd-benchmark/fast-models
```

## What is measured

- SD/SDXL: `sd_video_harness` configures the pipeline the way the node's `createConfiguration` /
  `updateScheduler` do (shared code: `src/bundle_manifest.hpp`, `src/schedulers/schedule.hpp`) and
  times `librediffusion_img2img` per frame: upload, VAE encode, UNet, VAE decode, download, sync.
  Timesteps are the node presets' img2img defaults (`32` for 1 step, `32, 45` for 2, with
  denoise-batch), add-noise off, cfg none, CUDA graph on (`--cuda-graph 0` for the per-call enqueue;
  the output is bit-identical). `--rife PLAN --rife-exp N` adds RIFE after each frame as the node's
  producer does and reports diffused and displayed fps.
- klein: `klein_fps_harness sync-noint` (the node's sync path without RIFE) with the video as the
  edit reference, `LRD_KLEIN_STEPS` = 1 or 2, FP8 transformer.
- Each run is preceded by 10 untimed frames. The output clips are encoded at the source frame rate
  (25 fps) so the motion matches the input; the title shows the measured throughput.

## Model notes

- **NitroFusion** was distilled with a shifted timestep (its `TimestepShiftLCMScheduler`): for a
  noise level `t`, the UNet is fed `t_s = floor(t * 0.25)` (Realism) or `floor(t * 0.5)` (Vibrant)
  and x0 is predicted with the coefficients of `t_s`, while noise is injected (img2img encode,
  between steps) at `t`. The exporter records the factor as `unet_timestep_scale` in bundle.json;
  `src/schedulers/schedule.hpp` turns it into the prepare_scheduler arrays plus a separate
  noise-injection schedule (`librediffusion_prepare_noise_schedule`). The engine is a plain SDXL UNet.
- **DMD2**'s 1-step UNet was trained only from pure noise at t=399 (table index 30): it is a txt2img
  generator and smears img2img input into a soft blur at any strength (reproduced in fp32 PyTorch with
  `bench/reference_img2img.py`, so not an export problem). The 4-step UNet (`999, 749, 499, 249`) is the
  DMD2 to use for video. DMD2 weights are CC-BY-NC-4.0.
- `bench/reference_img2img.py` recomputes one 1-step frame in PyTorch from what bundle.json describes
  (same PCG32 noise as the C++ side, coefficients from diffusers' LCMScheduler) to check an engine's
  output: `--compare cpp_frame.png` prints the PSNR.
- **klein-sprint**: the exporter fuses the LoRA into a transformer-only snapshot next to the bundle
  (`<out>_model-klein-<hash>`, everything else symlinked to the base) and exports from it, including
  the FP8 calibration.
