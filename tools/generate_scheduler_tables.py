#!/usr/bin/env python3
"""
Generate C++ header file with precomputed scheduler parameters.

This script computes scheduler parameters (c_skip, c_out, alpha_prod_t_sqrt, beta_prod_t_sqrt)
for all timesteps and generates a C++ header file with lookup tables.

Usage:
    python generate_scheduler_tables.py [--model MODEL_NAME] [--num-steps NUM_STEPS] [--output OUTPUT_FILE]

Example:
    uv run generate_scheduler_tables.py --model SimianLuo/LCM_Dreamshaper_v7             --num-steps 50 --output lcm_dreamshaper_v7.hpp
    uv run generate_scheduler_tables.py --model stabilityai/stable-diffusion-xl-base-1.0 --num-steps 50 --output stable-diffusion-xl-base-1.0.hpp
    uv run generate_scheduler_tables.py --model stabilityai/sdxl-turbo                   --num-steps 50 --output sdxl-turbo.hpp
    uv run generate_scheduler_tables.py --model Lykon/dreamshaper-8-lcm                  --num-steps 50 --output dreamshaper-8-lcm.hpp
    uv run generate_scheduler_tables.py --model sd-legacy/stable-diffusion-v1-5          --num-steps 50 --output stable-diffusion-v1-5.hpp
    uv run generate_scheduler_tables.py --model stabilityai/sd-turbo                     --num-steps 1 --output sd-turbo.hpp
"""

import argparse
import sys
from pathlib import Path
import torch
from diffusers import LCMScheduler


def sanitize_model_name(model_name: str) -> str:
    """Convert model name to valid C++ identifier."""
    # Replace special characters with underscores
    name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    return name.upper()


def generate_scheduler_header(model_name: str, num_inference_steps: int, output_path: Path):
    """
    Generate C++ header file with precomputed scheduler parameters.

    Args:
        model_name: HuggingFace model identifier (e.g., "SimianLuo/LCM_Dreamshaper_v7")
        num_inference_steps: Number of inference steps to generate parameters for
        output_path: Path to output .hpp file
    """
    print(f"Loading scheduler from model: {model_name}")
    scheduler = LCMScheduler.from_pretrained(model_name, subfolder="scheduler")

    print(f"Setting timesteps: {num_inference_steps}")
    scheduler.set_timesteps(num_inference_steps)

    # Extract timestep values
    timesteps = scheduler.timesteps.cpu().numpy()
    print(f"Generated {len(timesteps)} timesteps")
    print(f"Timestep range: [{timesteps.min()}, {timesteps.max()}]")

    # Compute parameters for each timestep
    print("Computing scheduler parameters...")
    timestep_data = []

    for idx, timestep_tensor in enumerate(scheduler.timesteps):
        timestep_int = int(timestep_tensor.item())

        # Get boundary condition scalings
        c_skip, c_out = scheduler.get_scalings_for_boundary_condition_discrete(timestep_int)

        # Get alpha and beta values
        alpha_prod_t = scheduler.alphas_cumprod[timestep_int]
        alpha_prod_t_sqrt = alpha_prod_t.sqrt()
        beta_prod_t_sqrt = (1 - alpha_prod_t).sqrt()

        timestep_data.append({
            'index': idx,
            'timestep': timestep_int,
            'c_skip': float(c_skip),
            'c_out': float(c_out),
            'alpha_prod_t_sqrt': float(alpha_prod_t_sqrt),
            'beta_prod_t_sqrt': float(beta_prod_t_sqrt),
        })

    # Generate C++ header file
    print(f"Generating C++ header: {output_path}")

    sanitized_name = sanitize_model_name(model_name)
    guard_name = f"STREAMDIFFUSION_SCHEDULER_{sanitized_name}_HPP"

    with open(output_path, 'w') as f:
        f.write(f"""// Auto-generated scheduler parameters for {model_name}
// Generated with num_inference_steps={num_inference_steps}
//
// This file contains precomputed scheduler parameters for LCM scheduling.
// Each timestep has: c_skip, c_out, alpha_prod_t_sqrt, beta_prod_t_sqrt
//
// Usage:
//   #include "scheduler_{sanitized_name.lower()}.hpp"
//   auto params = SCHEDULER_{sanitized_name}::get_params(timestep_index);

#ifndef {guard_name}
#define {guard_name}

#include <cstdint>
#include <stdexcept>
#include "timestep_params.hpp"

namespace streamdiffusion {{

namespace SCHEDULER_{sanitized_name} {{

// Number of timesteps in this scheduler configuration
constexpr int NUM_TIMESTEPS = {len(timestep_data)};

// Mapping from timestep index to actual timestep value
constexpr int TIMESTEP_VALUES[NUM_TIMESTEPS] = {{
""")

        # Write timestep values
        for i, data in enumerate(timestep_data):
            if i % 10 == 0:
                f.write("\n    ")
            f.write(f"{data['timestep']}")
            if i < len(timestep_data) - 1:
                f.write(", ")

        f.write("\n};\n\n")

        # Write precomputed parameters
        f.write(f"""// Precomputed scheduler parameters for each timestep index
constexpr TimestepParams TIMESTEP_PARAMS[NUM_TIMESTEPS] = {{
""")

        for i, data in enumerate(timestep_data):
            f.write(f"    {{ "
                   f"{data['c_skip']:.30f}, "
                   f"{data['c_out']:.30f}, "
                   f"{data['alpha_prod_t_sqrt']:.30f}, "
                   f"{data['beta_prod_t_sqrt']:.30f} "
                   f"}}")

            if i < len(timestep_data) - 1:
                f.write(",")

            f.write(f"  // idx={i}, t={data['timestep']}\n")

        f.write("};\n\n")

        # Add helper functions
        f.write(f"""// Get parameters for a given timestep index (0 to {len(timestep_data)-1})
inline const TimestepParams& get_params(int timestep_index) {{
    if (timestep_index < 0 || timestep_index >= NUM_TIMESTEPS) {{
        throw std::out_of_range("Timestep index out of range");
    }}
    return TIMESTEP_PARAMS[timestep_index];
}}

// Get timestep value for a given index
inline int get_timestep(int timestep_index) {{
    if (timestep_index < 0 || timestep_index >= NUM_TIMESTEPS) {{
        throw std::out_of_range("Timestep index out of range");
    }}
    return TIMESTEP_VALUES[timestep_index];
}}

}}  // namespace SCHEDULER_{sanitized_name}
}}  // namespace streamdiffusion

#endif  // {guard_name}
""")


def main():
    parser = argparse.ArgumentParser(
        description="Generate C++ header with precomputed scheduler parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for LCM Dreamshaper v7 with 50 steps
  python generate_scheduler_tables.py --model SimianLuo/LCM_Dreamshaper_v7 --num-steps 50

  # Generate for custom model with custom output
  python generate_scheduler_tables.py --model my/custom-model --num-steps 100 --output my_scheduler.hpp
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default='SimianLuo/LCM_Dreamshaper_v7',
        help='HuggingFace model identifier (default: SimianLuo/LCM_Dreamshaper_v7)'
    )

    parser.add_argument(
        '--num-steps',
        type=int,
        default=50,
        help='Number of inference steps (default: 50)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output .hpp file path (default: scheduler_<sanitized_model_name>.hpp)'
    )

    args = parser.parse_args()

    # Generate default output filename if not provided
    if args.output is None:
        sanitized = sanitize_model_name(args.model).lower()
        args.output = f"scheduler_{sanitized}.hpp"

    output_path = Path(args.output)

    try:
        generate_scheduler_header(args.model, args.num_steps, output_path)
        print("\n✓ Success!")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
