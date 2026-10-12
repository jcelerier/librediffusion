#!/usr/bin/env bash
# Build the engine bundles compared by bench/run_video_bench.py.
#
#   bench/build_fast_models.sh <out_root> [name...]
#
# The compared SDXL bundles share one profile (1024x1024, batch 1..2, opt 2) so the 1- and 2-step runs
# use the same engine and the models differ only in their UNet weights; dmd2-sdxl-4step-b4 is the
# 4-step variant (not in the default set). With no names, builds all of
# them; a bundle whose bundle.json already exists is skipped.
set -euo pipefail
OUT=${1:?usage: build_fast_models.sh <out_root> [name...]}; shift
HERE=$(cd "$(dirname "$0")/.." && pwd)
SDXL_BASE=stabilityai/stable-diffusion-xl-base-1.0
SDXL_PROFILE=(--min-resolution 1024 --max-resolution 1024 --min-batch 1 --max-batch 2 --opt-batch 2)

declare -A RECIPE=(
  [sdxl-turbo]="--type sdxl --model stabilityai/sdxl-turbo ${SDXL_PROFILE[*]}"
  [dmd2-sdxl-1step]="--type sdxl --model $SDXL_BASE --unet tianweiy/DMD2|dmd2_sdxl_1step_unet_fp16.bin ${SDXL_PROFILE[*]}"
  [dmd2-sdxl-4step]="--type sdxl --model $SDXL_BASE --unet tianweiy/DMD2|dmd2_sdxl_4step_unet_fp16.bin ${SDXL_PROFILE[*]}"
  # DMD2 at its native 4 steps: denoise-batch runs one batch-4 UNet call, so the profile goes to 4.
  [dmd2-sdxl-4step-b4]="--type sdxl --model $SDXL_BASE --unet tianweiy/DMD2|dmd2_sdxl_4step_unet_fp16.bin --min-resolution 1024 --max-resolution 1024 --min-batch 1 --max-batch 4 --opt-batch 4"
  [nitrosd-realism]="--type sdxl --model $SDXL_BASE --unet ChenDY/NitroFusion|nitrosd-realism_unet.safetensors --unet-timestep-scale 0.25 ${SDXL_PROFILE[*]}"
  [nitrosd-vibrant]="--type sdxl --model $SDXL_BASE --unet ChenDY/NitroFusion|nitrosd-vibrant_unet.safetensors --unet-timestep-scale 0.5 ${SDXL_PROFILE[*]}"
  # klein at its native 320x576 streaming geometry, with the 1-2 step SANA-Sprint LoRA fused in.
  [klein-sprint]="--type klein --model black-forest-labs/FLUX.2-klein-4B -l radames/FLUX.2-klein-Sana-Sprint --klein-quality both"
)
ORDER=(sdxl-turbo dmd2-sdxl-1step dmd2-sdxl-4step nitrosd-realism nitrosd-vibrant klein-sprint)
[ $# -gt 0 ] && ORDER=("$@")

for name in "${ORDER[@]}"; do
  args=${RECIPE[$name]:?unknown bundle $name}
  dst=$OUT/$name
  if [ -f "$dst/bundle.json" ]; then echo "[build] $name: exists, skipping"; continue; fi
  echo "[build] $name -> $dst"
  # word-splitting of $args is intended; the '|' inside the --unet specs is not a shell metachar here
  # shellcheck disable=SC2086
  (cd "$HERE" && uv run --no-sync python train-lora.py $args --output "$dst") 2>&1 | tee "$OUT/$name.build.log"
done
