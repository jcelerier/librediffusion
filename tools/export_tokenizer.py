#!/usr/bin/env python3
"""
Export CLIP tokenizer to a JSON file that can be loaded by the tokenizers C++ library.
"""

import os
from transformers import CLIPTokenizer

def export_tokenizer(output_path="tokenizer_clip.json"):
    """Export CLIP tokenizer to JSON format."""
    print("=" * 60)
    print("Exporting CLIP Tokenizer")
    print("=" * 60)
    print()

    # Load the tokenizer
    print("[1/2] Loading CLIP tokenizer from SimianLuo/LCM_Dreamshaper_v7...")
    tokenizer = CLIPTokenizer.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        subfolder="tokenizer"
    )
    print("      ✓ Tokenizer loaded")
    print(f"      Vocab size: {tokenizer.vocab_size}")
    print(f"      Model max length: {tokenizer.model_max_length}")
    print()

    # Save as JSON - the tokenizers library can load this
    print(f"[2/2] Exporting tokenizer to {output_path}...")

    # Use the tokenizers library directly to create a compatible tokenizer
    from tokenizers import Tokenizer, models, pre_tokenizers, normalizers, processors
    import json

    # Load vocab and merges from the tokenizer files
    with open("tokenizer_files/vocab.json", "r") as f:
        vocab = json.load(f)

    with open("tokenizer_files/merges.txt", "r") as f:
        merges_lines = f.read().strip().split('\n')[1:]  # Skip header line

    # Convert merges to list of tuples
    merges = [tuple(line.split()) for line in merges_lines]

    # Create a BPE tokenizer
    tokenizer_rust = Tokenizer(models.BPE(vocab=vocab, merges=merges))

    # Add normalizer (lowercase + strip accents for CLIP)
    tokenizer_rust.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents(),
    ])

    # Add pre-tokenizer
    tokenizer_rust.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Add post-processor for CLIP (add start/end tokens)
    tokenizer_rust.post_processor = processors.TemplateProcessing(
        single="<|startoftext|> $A <|endoftext|>",
        special_tokens=[
            ("<|startoftext|>", 49406),
            ("<|endoftext|>", 49407),
        ],
    )

    # Enable padding
    tokenizer_rust.enable_padding(pad_id=49407, pad_token="<|endoftext|>", length=77)

    # Enable truncation
    tokenizer_rust.enable_truncation(max_length=77)

    # Save to JSON
    tokenizer_rust.save(output_path)

    print(f"      ✓ Tokenizer saved to: {output_path}")
    print(f"      File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print()

    # Test the exported tokenizer
    print("Testing exported tokenizer...")
    test_prompt = "cat, 8k, digital art"
    tokens = tokenizer.encode(test_prompt, padding="max_length", max_length=77)
    print(f"  Prompt: \"{test_prompt}\"")
    print(f"  Tokens: {tokens[:10]}... (first 10)")
    print(f"  Total length: {len(tokens)}")
    print()

    print("=" * 60)
    print("✓ Tokenizer export complete!")
    print("=" * 60)
    print()
    print("You can now load this tokenizer in C++ using:")
    print("  #include <tokenizers_c.h>")
    print(f"  auto* tokenizer = tokenizers_from_file(\"{output_path}\");")
    print()

if __name__ == "__main__":
    export_tokenizer()
