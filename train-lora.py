import os
import sys
import argparse
from dataclasses import dataclass

# Add the src directory to the Python path if it's not already there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt


@dataclass
class LoraSpec:
    """Represents a LoRA with its path and optional weight."""
    path: str
    weight: float = 1.0

    @classmethod
    def parse(cls, spec: str) -> "LoraSpec":
        """
        Parse a LoRA specification string.

        Supports formats:
            - "path/to/lora.safetensors" (weight defaults to 1.0)
            - "path/to/lora.safetensors:0.85" (explicit weight)
        """
        if ":" in spec:
            # Check if the last colon separates a weight value
            last_colon = spec.rfind(":")
            potential_weight = spec[last_colon + 1:]
            try:
                weight = float(potential_weight)
                path = spec[:last_colon]
                return cls(path=path, weight=weight)
            except ValueError:
                # Not a valid weight, treat entire string as path
                pass
        return cls(path=spec, weight=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/accelerate LoRA models with TensorRT for StreamDiffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model SimianLuo/LCM_Dreamshaper_v7 --output ./engines
  %(prog)s --type sdxl --model stabilityai/stable-diffusion-xl-base-1.0 --output ./engines
  %(prog)s --model ./my_model --lora style.safetensors --lora detail.safetensors:0.5
  %(prog)s --model ./model --min-batch 1 --max-batch 4 --min-resolution 512 --max-resolution 1024
        """
    )

    # Model configuration
    parser.add_argument(
        "-t", "--type",
        choices=["sd15", "sdxl"],
        default="sd15",
        help="Model type: sd15 (Stable Diffusion 1.5) or sdxl (default: sd15)"
    )
    parser.add_argument(
        "-m", "--model",
        default="SimianLuo/LCM_Dreamshaper_v7",
        help="Model source path or HuggingFace model ID (default: SimianLuo/LCM_Dreamshaper_v7)"
    )
    parser.add_argument(
        "-o", "--output",
        default="./engines",
        help="Output path for TensorRT engines (default: ./engines)"
    )

    # LoRA configuration
    parser.add_argument(
        "-l", "--lora",
        action="append",
        dest="loras",
        default=[],
        metavar="PATH[:WEIGHT]",
        help="LoRA file to load. Can be specified multiple times. "
             "Optional weight can be appended after colon (e.g., lora.safetensors:0.85)"
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=2.5,
        help="Global LoRA fusion scale (default: 2.5)"
    )

    # Batch size configuration
    parser.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="Minimum batch size for TensorRT optimization (default: 1)"
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=2,
        help="Maximum batch size for TensorRT optimization (default: 2)"
    )
    parser.add_argument(
        "--opt-batch",
        type=int,
        default=2,
        help="Optimal batch size for TensorRT optimization (default: 2)"
    )

    # Resolution configuration
    parser.add_argument(
        "--min-resolution",
        type=int,
        default=1024,
        help="Minimum image resolution for TensorRT optimization (default: 1024)"
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=1024,
        help="Maximum image resolution for TensorRT optimization (default: 1024)"
    )
    parser.add_argument(
        "--opt-height",
        type=int,
        default=None,
        help="Optimal image height (default: same as max-resolution)"
    )
    parser.add_argument(
        "--opt-width",
        type=int,
        default=None,
        help="Optimal image width (default: same as max-resolution)"
    )

    args = parser.parse_args()

    # Set defaults for optional resolution parameters
    if args.opt_height is None:
        args.opt_height = args.max_resolution
    if args.opt_width is None:
        args.opt_width = args.max_resolution

    # Validate batch sizes
    if args.min_batch > args.max_batch:
        parser.error("--min-batch cannot be greater than --max-batch")
    if args.opt_batch < args.min_batch or args.opt_batch > args.max_batch:
        parser.error("--opt-batch must be between --min-batch and --max-batch")

    # Validate resolutions
    if args.min_resolution > args.max_resolution:
        parser.error("--min-resolution cannot be greater than --max-resolution")

    # Parse LoRA specifications
    args.lora_specs = [LoraSpec.parse(lora) for lora in args.loras]

    return args


args = parse_args()


if args.type == "sdxl":
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model).to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained(args.model).to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )

stream = StreamDiffusion(
    pipe,
    t_index_list=[30, 45],
    torch_dtype=torch.float16,
    cfg_type="none",
)

# Load LoRAs with their individual weights
for lora_spec in args.lora_specs:
    stream.load_lora(lora_spec.path)
    print(f"Loaded LoRA: {lora_spec.path} (weight: {lora_spec.weight})")

if len(args.lora_specs) > 0:
    # Calculate per-LoRA weights by multiplying individual weights with global scale
    lora_weights = [spec.weight for spec in args.lora_specs]
    effective_scale = args.lora_scale
    # If multiple LoRAs have different weights, we apply the average weighted by lora_scale
    if len(set(lora_weights)) > 1:
        print(f"Note: Multiple LoRAs with different weights detected.")
        print(f"Individual weights: {lora_weights}, global scale: {args.lora_scale}")
    stream.fuse_lora(
        fuse_unet=True,
        fuse_text_encoder=True,
        lora_scale=effective_scale,
        safe_fusing=False
    )

if args.type == "sd15":
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
        device=pipe.device, dtype=pipe.dtype
    )
else:
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl").to(
        device=pipe.device, dtype=pipe.dtype
    )

engine_build_options = {
    "opt_batch_size": args.opt_batch,
    "min_image_resolution": args.min_resolution,
    "max_image_resolution": args.max_resolution,
    "opt_image_height": args.opt_height,
    "opt_image_width": args.opt_width,
}

# Enable static shapes when all min/max values are equal
use_static_shapes = (
    args.min_batch == args.max_batch and
    args.min_resolution == args.max_resolution and
    args.opt_height == args.opt_width == args.max_resolution
)

print(f"Building TensorRT engine with options:")
print(f"  Batch size: min={args.min_batch}, opt={args.opt_batch}, max={args.max_batch}")
print(f"  Resolution: min={args.min_resolution}, max={args.max_resolution}")
print(f"  Optimal dimensions: {args.opt_width}x{args.opt_height}")
print(f"  Static shapes: {use_static_shapes}")
print(f"  Output: {args.output}")

stream = accelerate_with_tensorrt(
    stream,
    args.output,
    min_batch_size=args.min_batch,
    max_batch_size=args.max_batch,
    engine_build_options=engine_build_options,
    static_shapes=use_static_shapes,
    # use_v2v=True,  # Enable V2V for C++ testing
)

