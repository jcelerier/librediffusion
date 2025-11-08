#!/usr/bin/env python3
"""
Inspect TensorRT engine to see tensor names and shapes
"""
import sys
sys.path.insert(0, "TensorRT-RTX-1.1.1.26/python")

import tensorrt as trt
import numpy as np

def inspect_engine(engine_path):
    """Inspect a TensorRT engine file"""
    print(f"\n{'='*80}")
    print(f"Inspecting: {engine_path}")
    print(f"{'='*80}\n")

    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine_data = f.read()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)

    if not engine:
        print("Failed to deserialize engine")
        return

    print(f"Number of bindings: {engine.num_io_tensors}")
    print(f"Number of optimization profiles: {engine.num_optimization_profiles}\n")

    # Print all tensor info
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)

        mode_str = "INPUT " if mode == trt.TensorIOMode.INPUT else "OUTPUT"
        print(f"{mode_str} {i}: '{name}'")
        print(f"  dtype: {dtype}")
        print(f"  shape: {shape}")

        # If dynamic shapes, print profile info
        if -1 in shape:
            print(f"  Dynamic shape profiles:")
            for profile_idx in range(engine.num_optimization_profiles):
                min_shape = engine.get_tensor_profile_shape(name, profile_idx)[0]
                opt_shape = engine.get_tensor_profile_shape(name, profile_idx)[1]
                max_shape = engine.get_tensor_profile_shape(name, profile_idx)[2]
                print(f"    Profile {profile_idx}:")
                print(f"      min: {min_shape}")
                print(f"      opt: {opt_shape}")
                print(f"      max: {max_shape}")
        print()

if __name__ == "__main__":
    inspect_engine("engines/vae_encoder.engine")
    inspect_engine("engines/vae_decoder.engine")
    inspect_engine("engines/unet.engine")
    inspect_engine("engines/clip.engine")
    inspect_engine("engines/clip2.engine")
