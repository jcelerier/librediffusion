#!/usr/bin/env python3
"""
Generate CLIP Text Embeddings for Pure C++ Inference

This script precomputes CLIP text embeddings for prompts and saves them in a format
that can be easily loaded by C++ applications.

The embeddings are saved in multiple formats:
1. Raw binary file (.bin) - for direct C++ loading
2. C++ header file (.hpp) - for compile-time embedding
3. NumPy file (.npy) - for Python verification

Usage:
    python generate_prompt_embeddings.py "your prompt here" --model SimianLuo/LCM_Dreamshaper_v7
    python generate_prompt_embeddings.py "cat, 8k, digital art" --output cat_prompt
    python generate_prompt_embeddings.py --prompts-file prompts.txt --output-dir embeddings/
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer


def sanitize_name(text: str) -> str:
    """Convert prompt text to valid C++ identifier."""
    # Remove special characters, replace spaces with underscores
    name = ''.join(c if c.isalnum() or c == '_' else '_' for c in text)
    # Remove consecutive underscores
    while '__' in name:
        name = name.replace('__', '_')
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Ensure it starts with a letter
    if name and name[0].isdigit():
        name = 'prompt_' + name
    # Limit length
    if len(name) > 50:
        name = name[:50]
    return name.upper()


def encode_prompt(prompt: str, model_name: str, device: str = 'cuda', dtype=torch.float16):
    """
    Encode a text prompt using CLIP text encoder.

    Args:
        prompt: Text prompt to encode
        model_name: HuggingFace model identifier
        device: Device to run on ('cuda' or 'cpu')
        dtype: Data type for embeddings (float16 or float32)

    Returns:
        torch.Tensor: Encoded embeddings with shape [1, 77, 768]
    """
    print(f"Loading CLIP model from: {model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_name,
        subfolder="text_encoder"
    ).to(device=device, dtype=dtype)

    print(f"Encoding prompt: '{prompt}'")
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)

    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids)[0]

    print(f"✓ Embeddings shape: {prompt_embeds.shape}")
    print(f"  Data type: {prompt_embeds.dtype}")
    print(f"  Device: {prompt_embeds.device}")
    print(f"  Value range: [{prompt_embeds.min().item():.4f}, {prompt_embeds.max().item():.4f}]")
    print(f"  Mean: {prompt_embeds.mean().item():.4f}, Std: {prompt_embeds.std().item():.4f}")

    return prompt_embeds


def save_binary(embeddings: torch.Tensor, output_path: Path):
    """Save embeddings as raw binary file (fp16)."""
    # Convert to fp16 CPU numpy array
    embeddings_np = embeddings.cpu().half().numpy()

    # Save as raw binary
    embeddings_np.tofile(output_path)

    print(f"✓ Binary file saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size} bytes")
    print(f"  Expected elements: {embeddings_np.size} (1 * 77 * 768 = 59136)")


def save_cpp_header(embeddings: torch.Tensor, prompt: str, output_path: Path):
    """Save embeddings as C++ header file with constexpr array."""
    embeddings_np = embeddings.cpu().half().numpy().flatten()

    var_name = sanitize_name(prompt)
    guard_name = f"STREAMDIFFUSION_PROMPT_{var_name}_HPP"

    with open(output_path, 'w') as f:
        f.write(f"""// Auto-generated CLIP prompt embeddings
// Prompt: "{prompt}"
// Shape: [1, 77, 768] = 59136 elements
// Data type: fp16 (half precision)

#ifndef {guard_name}
#define {guard_name}

#include <cuda_fp16.h>
#include <cstdint>

namespace streamdiffusion {{
namespace prompts {{

// Prompt text: "{prompt}"
constexpr int EMBEDDING_SIZE = 59136;  // 1 * 77 * 768

// Note: Due to C++ constexpr limitations with __half, we store as uint16_t
// and reinterpret_cast to __half* at runtime
constexpr uint16_t {var_name}_RAW[EMBEDDING_SIZE] = {{
""")

        # Write embeddings as uint16_t (fp16 bit pattern)
        # Convert __half to uint16_t for storage
        for i, val in enumerate(embeddings_np):
            # Get bit pattern of fp16 value
            val_uint16 = np.frombuffer(np.array(val).tobytes(), dtype=np.uint16)[0]

            if i % 8 == 0:
                f.write("    ")
            f.write(f"0x{val_uint16:04x}")
            if i < len(embeddings_np) - 1:
                f.write(", ")
            if i % 8 == 7:
                f.write("\n")

        f.write(f"""
}};

// Helper function to get embeddings as __half pointer
inline const __half* get_{var_name.lower()}() {{
    return reinterpret_cast<const __half*>({var_name}_RAW);
}}

// Get embeddings size
inline constexpr int get_{var_name.lower()}_size() {{
    return EMBEDDING_SIZE;
}}

}}  // namespace prompts
}}  // namespace streamdiffusion

#endif  // {guard_name}
""")

    print(f"✓ C++ header saved: {output_path}")
    print(f"  Variable name: {var_name}")
    print(f"  Namespace: streamdiffusion::prompts")


def save_numpy(embeddings: torch.Tensor, output_path: Path):
    """Save embeddings as NumPy file for verification."""
    embeddings_np = embeddings.cpu().half().numpy()
    np.save(output_path, embeddings_np)

    print(f"✓ NumPy file saved: {output_path}")


def load_prompts_from_file(file_path: Path) -> list[str]:
    """Load prompts from text file (one per line)."""
    with open(file_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLIP text embeddings for C++ inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt with default model
  python generate_prompt_embeddings.py "cat, 8k, digital art"

  # Custom model and output name
  python generate_prompt_embeddings.py "landscape painting" --model SimianLuo/LCM_Dreamshaper_v7 --output landscape

  # Multiple prompts from file
  python generate_prompt_embeddings.py --prompts-file prompts.txt --output-dir embeddings/

  # All output formats
  python generate_prompt_embeddings.py "sunset beach" --binary --header --numpy
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'prompt',
        nargs='?',
        type=str,
        help='Text prompt to encode'
    )
    input_group.add_argument(
        '--prompts-file',
        type=str,
        help='File containing prompts (one per line)'
    )

    # Model options
    parser.add_argument(
        '--model',
        type=str,
        default='SimianLuo/LCM_Dreamshaper_v7',
        help='HuggingFace model identifier (default: SimianLuo/LCM_Dreamshaper_v7)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda)'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output file base name (without extension). Default: sanitized prompt'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory (default: current directory)'
    )

    # Output format options
    parser.add_argument(
        '--binary',
        action='store_true',
        help='Generate binary file (.bin)'
    )

    parser.add_argument(
        '--header',
        action='store_true',
        help='Generate C++ header file (.hpp)'
    )

    parser.add_argument(
        '--numpy',
        action='store_true',
        help='Generate NumPy file (.npy) for verification'
    )

    # If no format specified, generate all
    parser.add_argument(
        '--all-formats',
        action='store_true',
        help='Generate all output formats (default if no format specified)'
    )

    args = parser.parse_args()

    # Determine output formats
    if not (args.binary or args.header or args.numpy or args.all_formats):
        # Default: generate all formats
        args.binary = True
        args.header = True
        args.numpy = True

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get prompts list
    if args.prompts_file:
        prompts_file = Path(args.prompts_file)
        if not prompts_file.exists():
            print(f"Error: Prompts file not found: {prompts_file}", file=sys.stderr)
            return 1
        prompts = load_prompts_from_file(prompts_file)
        print(f"Loaded {len(prompts)} prompts from {prompts_file}")
    else:
        prompts = [args.prompt]

    # Process each prompt
    for i, prompt in enumerate(prompts):
        print("\n" + "=" * 80)
        print(f"Processing prompt {i+1}/{len(prompts)}")
        print("=" * 80)

        # Determine output base name
        if args.output:
            base_name = args.output
            if len(prompts) > 1:
                base_name = f"{base_name}_{i:03d}"
        else:
            base_name = sanitize_name(prompt).lower()

        try:
            # Encode prompt
            embeddings = encode_prompt(prompt, args.model, args.device)

            # Save in requested formats
            if args.binary:
                binary_path = output_dir / f"{base_name}.bin"
                save_binary(embeddings, binary_path)

            if args.header:
                header_path = output_dir / f"{base_name}.hpp"
                save_cpp_header(embeddings, prompt, header_path)

            if args.numpy:
                numpy_path = output_dir / f"{base_name}.npy"
                save_numpy(embeddings, numpy_path)

            print("\n✓ All files generated successfully")

        except Exception as e:
            print(f"\n✗ Error processing prompt: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print("=" * 80)
    print("\nC++ Usage (header file):")
    print(f"""
    #include "{base_name}.hpp"

    // Get embeddings pointer
    const __half* embeddings = streamdiffusion::prompts::get_{sanitize_name(prompts[0]).lower()}();
    int size = streamdiffusion::prompts::get_{sanitize_name(prompts[0]).lower()}_size();

    // Copy to device
    __half* d_embeddings;
    cudaMalloc(&d_embeddings, size * sizeof(__half));
    cudaMemcpy(d_embeddings, embeddings, size * sizeof(__half), cudaMemcpyHostToDevice);
    """)

    print("\nC++ Usage (binary file):")
    print(f"""
    // Load from file
    std::ifstream file("{base_name}.bin", std::ios::binary);
    std::vector<__half> embeddings(59136);
    file.read(reinterpret_cast<char*>(embeddings.data()), 59136 * sizeof(__half));

    // Copy to device
    __half* d_embeddings;
    cudaMalloc(&d_embeddings, 59136 * sizeof(__half));
    cudaMemcpy(d_embeddings, embeddings.data(), 59136 * sizeof(__half), cudaMemcpyHostToDevice);
    """)

    return 0


if __name__ == '__main__':
    sys.exit(main())
