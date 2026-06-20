#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/utilities.py

#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import gc
import os
from collections import OrderedDict
from typing import *

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from cuda.bindings import runtime as cudart
from PIL import Image
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.backend.trt import util as trt_util

from .models import CLIP, VAE, BaseModel, UNet, VAEEncoder


# Check if TensorRT-RTX should be used
USE_TRT_RTX = os.environ.get("USE_TRT_RTX", "false").lower() in ("true", "1", "yes")

if USE_TRT_RTX:
    try:
        import tensorrt_rtx as trt
        print("Using TensorRT-RTX for acceleration")
    except ImportError as e:
        print(f"Warning: USE_TRT_RTX=true but tensorrt_rtx is not installed: {e}")
        print("Falling back to standard TensorRT")
        import tensorrt as trt
else:
    import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}

# Map TensorRT DataType to numpy dtype (for TensorRT-RTX compatibility)
trt_to_numpy_dtype_dict = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT8: np.int8,
    trt.DataType.INT32: np.int32,
    trt.DataType.BOOL: np.bool_,
}


def trt_dtype_to_np(trt_dtype):
    """
    Convert TensorRT DataType to numpy dtype.
    This provides compatibility between standard TensorRT and TensorRT-RTX.
    """
    # Try the standard TensorRT nptype function first
    try:
        return trt.nptype(trt_dtype)
    except (TypeError, AttributeError, KeyError):
        # Fallback for TensorRT-RTX or when nptype fails
        if trt_dtype in trt_to_numpy_dtype_dict:
            return trt_to_numpy_dtype_dict[trt_dtype]
        else:
            # Try to match by string name as last resort
            dtype_str = str(trt_dtype)
            if "FLOAT" in dtype_str and "HALF" not in dtype_str:
                return np.float32
            elif "HALF" in dtype_str or "FP16" in dtype_str:
                return np.float16
            elif "INT32" in dtype_str:
                return np.int32
            elif "INT8" in dtype_str:
                return np.int8
            elif "BOOL" in dtype_str:
                return np.bool_
            else:
                # Default to float32 if we can't determine
                print(f"Warning: Unknown TensorRT dtype {trt_dtype}, defaulting to float32")
                return np.float32


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class Engine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph
        self.last_shapes = {}  # Cache for input shapes to avoid redundant API calls

    def __del__(self):
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray)]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, onnx_path, onnx_refit_path):
        def convert_int64(arr):
            # TODO: smarter conversion
            if len(arr.shape) == 0:
                return np.int32(arr)
            return arr

        def add_to_map(refit_dict, name, values):
            if name in refit_dict:
                assert refit_dict[name] is None
                if values.dtype == np.int64:
                    values = convert_int64(values)
                refit_dict[name] = values

        print(f"Refitting TensorRT engine with {onnx_refit_path} weights")
        refit_nodes = gs.import_onnx(onnx.load(onnx_refit_path)).toposort().nodes

        # Construct mapping from weight names in refit model -> original model
        name_map = {}
        for n, node in enumerate(gs.import_onnx(onnx.load(onnx_path)).toposort().nodes):
            refit_node = refit_nodes[n]
            assert node.op == refit_node.op
            # Constant nodes in ONNX do not have inputs but have a constant output
            if node.op == "Constant":
                name_map[refit_node.outputs[0].name] = node.outputs[0].name
            # Handle scale and bias weights
            elif node.op == "Conv":
                if node.inputs[1].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTKERNEL"] = node.name + "_TRTKERNEL"
                if node.inputs[2].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTBIAS"] = node.name + "_TRTBIAS"
            # For all other nodes: find node inputs that are initializers (gs.Constant)
            else:
                for i, inp in enumerate(node.inputs):
                    if inp.__class__ == gs.Constant:
                        name_map[refit_node.inputs[i].name] = inp.name

        def map_name(name):
            if name in name_map:
                return name_map[name]
            return name

        # Construct refit dictionary
        refit_dict = {}
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        all_weights = refitter.get_all()
        for layer_name, role in zip(all_weights[0], all_weights[1]):
            # for speciailized roles, use a unique name in the map:
            if role == trt.WeightsRole.KERNEL:
                name = layer_name + "_TRTKERNEL"
            elif role == trt.WeightsRole.BIAS:
                name = layer_name + "_TRTBIAS"
            else:
                name = layer_name

            assert name not in refit_dict, "Found duplicate layer: " + name
            refit_dict[name] = None

        for n in refit_nodes:
            # Constant nodes in ONNX do not have inputs but have a constant output
            if n.op == "Constant":
                name = map_name(n.outputs[0].name)
                print(f"Add Constant {name}\n")
                add_to_map(refit_dict, name, n.outputs[0].values)

            # Handle scale and bias weights
            elif n.op == "Conv":
                if n.inputs[1].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTKERNEL")
                    add_to_map(refit_dict, name, n.inputs[1].values)

                if n.inputs[2].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTBIAS")
                    add_to_map(refit_dict, name, n.inputs[2].values)

            # For all other nodes: find node inputs that are initializers (AKA gs.Constant)
            else:
                for inp in n.inputs:
                    name = map_name(inp.name)
                    if inp.__class__ == gs.Constant:
                        add_to_map(refit_dict, name, inp.values)

        for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
            if weights_role == trt.WeightsRole.KERNEL:
                custom_name = layer_name + "_TRTKERNEL"
            elif weights_role == trt.WeightsRole.BIAS:
                custom_name = layer_name + "_TRTBIAS"
            else:
                custom_name = layer_name

            # Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
            if layer_name.startswith("onnx::Trilu"):
                continue

            if refit_dict[custom_name] is not None:
                refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
            else:
                print(f"[W] No refit weights for layer: {layer_name}")

        if not refitter.refit_cuda_engine():
            print("Failed to refit!")
            exit(0)

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_refit=False,
        enable_all_tactics=True,
        timing_cache=None,
        workspace_size=0,
        builder_optimization_level=5,  # 0-5, higher = more optimizations, slower build
        profiling_verbosity=None,  # None (disabled), 'LAYER_NAMES_ONLY', 'DETAILED', 'NONE'
    ):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}

        if workspace_size > 0:
            config_kwargs["memory_pool_limits"] = {trt.MemoryPoolType.WORKSPACE: workspace_size}

        # ============ PROFILING & LOGGING OPTIMIZATION ============
        # Disable profiling for maximum runtime performance (no overhead from profiling)
        if profiling_verbosity is None or profiling_verbosity == 'NONE':
            # Disable all profiling for best runtime performance
            config_kwargs["profiling_verbosity"] = trt.ProfilingVerbosity.NONE
        elif profiling_verbosity == 'LAYER_NAMES_ONLY':
            config_kwargs["profiling_verbosity"] = trt.ProfilingVerbosity.LAYER_NAMES_ONLY
        elif profiling_verbosity == 'DETAILED':
            config_kwargs["profiling_verbosity"] = trt.ProfilingVerbosity.DETAILED

        # ============ BUILDER OPTIMIZATION LEVEL ============
        # Level 5 = maximum optimizations, slower build time but faster inference
        # Level 0 = minimal optimizations, faster build time
        config_kwargs["builder_optimization_level"] = builder_optimization_level
        config_kwargs["tiling_optimization_level"] = trt.TilingOptimizationLevel.FULL

        # ============ TACTIC SOURCES ============
        # Configure tactic sources for kernel selection
        # CRITICAL: Always enable at least cuBLAS and cuBLAS-LT for optimal performance
        # Setting tactic_sources=[] disables ALL optimized kernels!
        if enable_all_tactics:
            # Enable all available tactics for maximum optimization (slower build, potentially faster runtime)
            tactic_sources = [
                1 << int(trt.TacticSource.CUBLAS),
                1 << int(trt.TacticSource.CUBLAS_LT),
                1 << int(trt.TacticSource.CUDNN),
            ]
            # Add edge mask convolutions if available
            try:
                tactic_sources.append(1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS))
            except AttributeError:
                pass  # Not available in all TensorRT versions
            config_kwargs["tactic_sources"] = tactic_sources
        else:
            # Use standard tactics (cuBLAS, cuBLAS-LT) for reasonable build time and good performance
            config_kwargs["tactic_sources"] = [
                1 << int(trt.TacticSource.CUBLAS),
                1 << int(trt.TacticSource.CUBLAS_LT),
            ]

        network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])

        # TensorRT-RTX does NOT support the FP16 builder flag (models must be pre-converted to FP16)
        # Disable FP16 flag when using TensorRT-RTX
        import os
        use_trt_rtx = os.environ.get("USE_TRT_RTX", "false").lower() in ("true", "1", "yes")
        fp16_flag = False if use_trt_rtx else fp16

        engine = engine_from_network(
            network,
            config=CreateConfig(
                tf32=True, fp16=fp16_flag, refittable=enable_refit, profiles=[p], load_timing_cache=timing_cache, **config_kwargs, 
            ),
            save_timing_cache=timing_cache,
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)
            dtype = trt_dtype_to_np(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[name] = tensor

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        # Memory copy optimization: Only copy when necessary
        # Skip redundant copies when input already shares memory with target buffer
        for name, buf in feed_dict.items():
            target_tensor = self.tensors[name]

            # Skip copy if tensors share the same memory (zero-copy optimization)
            if target_tensor.data_ptr() != buf.data_ptr():
                # Only copy when buffers are different
                target_tensor.copy_(buf)
            # else: data_ptr is same, skip copy entirely

        # For TensorRT-RTX compatibility: ensure input shapes are set before setting addresses
        # TensorRT-RTX returns enums from tensorrt_bindings, not tensorrt_rtx,
        # so we use string comparison as a reliable method to identify inputs
        for name, tensor in self.tensors.items():
            mode = self.engine.get_tensor_mode(name)

            # Check if this is an input tensor
            # Use string comparison for TensorRT-RTX compatibility since enum types differ
            mode_str = str(mode)
            is_input = "INPUT" in mode_str and "OUTPUT" not in mode_str

            if is_input:
                # Only set shape if it changed (avoid redundant API calls)
                if name not in self.last_shapes or self.last_shapes[name] != tensor.shape:
                    self.context.set_input_shape(name, tensor.shape)
                    self.last_shapes[name] = tensor.shape

            self.context.set_tensor_address(name, tensor.data_ptr())

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream.ptr))
                CUASSERT(cudart.cudaStreamSynchronize(stream.ptr))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream.ptr)
                if not noerror:
                    raise ValueError("ERROR: inference failed.")
                # capture cuda graph
                CUASSERT(
                    cudart.cudaStreamBeginCapture(stream.ptr, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
                )
                self.context.execute_async_v3(stream.ptr)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream.ptr))
                self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            noerror = self.context.execute_async_v3(stream.ptr)
            if not noerror:
                raise ValueError("ERROR: inference failed.")

        return self.tensors


def decode_images(images: torch.Tensor):
    images = (
        ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
    )
    return [Image.fromarray(x) for x in images]


def preprocess_image(image: Image.Image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h))
    init_image = np.array(image).astype(np.float32) / 255.0
    init_image = init_image[None].transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image).contiguous()
    return 2.0 * init_image - 1.0


def prepare_mask_and_masked_image(image: Image.Image, mask: Image.Image):
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32).contiguous() / 127.5 - 1.0
    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(dtype=torch.float32).contiguous()

    masked_image = image * (mask < 0.5)

    return mask, masked_image


def create_models(
    model_id: str,
    use_auth_token: Optional[str],
    device: Union[str, torch.device],
    max_batch_size: int,
    unet_in_channels: int = 4,
    embedding_dim: int = 768,
):
    models = {
        "clip": CLIP(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
        "unet": UNet(
            hf_token=use_auth_token,
            fp16=True,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            unet_dim=unet_in_channels,
        ),
        "vae": VAE(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
        "vae_encoder": VAEEncoder(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
    }
    return models


def build_engine(
    engine_path: str,
    onnx_opt_path: str,
    model_data: BaseModel,
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    build_static_batch: bool = False,
    build_dynamic_shape: bool = False,
    build_all_tactics: bool = False,
    build_enable_refit: bool = False,
):
    _, free_mem, _ = cudart.cudaMemGetInfo()
    GiB = 2**30

    # Phase 3.1 Optimization: Enable TF32 for Ampere and newer GPUs (15-25% speedup)
    # TF32 provides ~8x faster matrix multiplication than FP32 with minimal accuracy loss
    # RTX 30/40 series, A100, H100 all support TF32
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] >= 8:  # Ampere (8.x), Ada (8.9), Hopper (9.0)
            # Use new PyTorch 2.9+ API for TF32 settings
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
            device_name = torch.cuda.get_device_name()
            print(f"[I] TF32 enabled for {device_name} (Compute Capability {compute_capability[0]}.{compute_capability[1]})")
        else:
            print(f"[I] TF32 not available on Compute Capability {compute_capability[0]}.{compute_capability[1]} (requires 8.0+)")

    # Optimized workspace allocation - be more aggressive to allow more optimization tactics
    # Old: reserved 4GB (too conservative), limiting optimization options
    # New: use percentage-based allocation for better optimization
    if free_mem > 8 * GiB:
        # High VRAM (>8GB): use 75% for workspace
        max_workspace_size = int(free_mem * 0.75)
        print(f"High VRAM detected ({free_mem / GiB:.1f} GB), using {max_workspace_size / GiB:.2f} GB workspace")
    elif free_mem > 6 * GiB:
        # Medium VRAM (6-8GB): use 70% for workspace
        max_workspace_size = int(free_mem * 0.70)
        print(f"Medium VRAM detected ({free_mem / GiB:.1f} GB), using {max_workspace_size / GiB:.2f} GB workspace")
    elif free_mem > 4 * GiB:
        # Low VRAM (4-6GB): use 60% for workspace
        max_workspace_size = int(free_mem * 0.60)
        print(f"Low VRAM detected ({free_mem / GiB:.1f} GB), using {max_workspace_size / GiB:.2f} GB workspace")
    else:
        # Very low VRAM (<4GB): use 50% for workspace
        max_workspace_size = int(free_mem * 0.50)
        print(f"Very low VRAM detected ({free_mem / GiB:.1f} GB), using {max_workspace_size / GiB:.2f} GB workspace")

    # Create timing cache path for faster subsequent builds
    timing_cache_path = engine_path.replace(".engine", ".timing.cache")

    engine = Engine(engine_path)
    input_profile = model_data.get_input_profile(
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        static_batch=build_static_batch,
        static_shape=not build_dynamic_shape,
    )
    engine.build(
        onnx_opt_path,
        fp16=True,
        input_profile=input_profile,
        enable_refit=build_enable_refit,
        enable_all_tactics=build_all_tactics,
        workspace_size=max_workspace_size,
        timing_cache=timing_cache_path,
    )

    return engine


def export_onnx(
    model,
    onnx_path: str,
    model_data: BaseModel,
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    onnx_opset: int,
):
    with torch.inference_mode(), torch.autocast("cuda"):
        inputs = model_data.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
        torch.onnx.export(
            model,
            inputs,
            onnx_path,
            export_params=True,
            opset_version=onnx_opset,
            do_constant_folding=True,
            input_names=model_data.get_input_names(),
            output_names=model_data.get_output_names(),
            dynamic_axes=model_data.get_dynamic_axes(),
            dynamo=False,  # Use legacy exporter for stability with dynamic_axes
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def optimize_onnx(
    onnx_path: str,
    onnx_opt_path: str,
    model_data: BaseModel,
):
    onnx_opt_graph = model_data.optimize(onnx.load(onnx_path))
    onnx.save(onnx_opt_graph, onnx_opt_path)
    del onnx_opt_graph
    gc.collect()
    torch.cuda.empty_cache()
