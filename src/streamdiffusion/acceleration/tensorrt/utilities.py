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
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart  # cuda-python 13 (cu130): cudart moved under cuda.bindings.runtime
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

from .models.models import CLIP, VAE, BaseModel, UNet, VAEEncoder

# Set up logger for this module
import logging
logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

from ...model_detection import detect_model

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
        
        # Buffer reuse optimization tracking
        self._last_shape_dict = None
        self._last_device = None

    def __del__(self):
        # Check if AttributeError: 'Engine' object has no attribute 'buffers'
        if not hasattr(self, 'buffers'):
            return
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray)]
        
        if hasattr(self, 'cuda_graph_instance') and self.cuda_graph_instance is not None:
            try:
                CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instance))
            except:
                pass
        if hasattr(self, 'graph') and self.graph is not None:
            try:
                CUASSERT(cudart.cudaGraphDestroy(self.graph))
            except:
                pass
        
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

        logger.info(f"Refitting TensorRT engine with {onnx_refit_path} weights")
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
                logger.warning(f"No refit weights for layer: {layer_name}")

        if not refitter.refit_cuda_engine():
            logger.error("Failed to refit!")
            raise RuntimeError("TensorRT engine refit failed")

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_refit=False,
        enable_all_tactics=True,
        timing_cache=None,
        workspace_size=0,
        builder_optimization_level=5,  # 0-5, higher = more optimizations + slower build. Surfaced as a user toggle upstream.
        profiling_verbosity=None,  # None/'NONE' (disabled, fastest runtime), 'LAYER_NAMES_ONLY', 'DETAILED'
        hardware_compatibility=None,  # None/'none' (build-GPU only, fastest) | 'ampere_plus' | 'same_cc'.
                                      # PORTABLE engines that run across GPU archs at ~5-15% inference cost.
    ):
        # NOTE (librediffusion fork): ported Phase-3.x TRT build optimizations from the bundled
        # librediffusion utilities.py — builder_optimization_level, tiling FULL, tf32, explicit
        # tactic sources, profiling verbosity. Upstream daydream had none of these and even
        # *disabled* tactics by default (tactic_sources=[]). The TensorRT-RTX branch is intentionally
        # NOT ported (RTX was slower than plain TRT in practice).
        logger.info(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}

        if workspace_size > 0:
            config_kwargs["memory_pool_limits"] = {trt.MemoryPoolType.WORKSPACE: workspace_size}

        # ---- profiling verbosity: NONE for best runtime perf (no profiling overhead) ----
        if profiling_verbosity is None or profiling_verbosity == "NONE":
            config_kwargs["profiling_verbosity"] = trt.ProfilingVerbosity.NONE
        elif profiling_verbosity == "LAYER_NAMES_ONLY":
            config_kwargs["profiling_verbosity"] = trt.ProfilingVerbosity.LAYER_NAMES_ONLY
        elif profiling_verbosity == "DETAILED":
            config_kwargs["profiling_verbosity"] = trt.ProfilingVerbosity.DETAILED

        # ---- builder optimization level (TODO: surface as a user toggle) ----
        # Level 5 = max optimizations / slower build / faster inference; Level 0 = fast build.
        config_kwargs["builder_optimization_level"] = builder_optimization_level
        config_kwargs["tiling_optimization_level"] = trt.TilingOptimizationLevel.FULL

        # ---- hardware compatibility (portable engines across GPU archs) ----
        # Default (None/'none') = build-GPU-locked (smallest/fastest). 'ampere_plus' makes the engine run
        # on SM 8.0+ (Ampere/Ada/Hopper/Blackwell...) at ~5-15% inference cost + larger engine. polygraphy's
        # CreateConfig sets config.hardware_compatibility_level from this kwarg directly.
        _hc = (hardware_compatibility or "none").lower()
        if _hc in ("ampere_plus", "ampere", "amperePlus".lower()):
            config_kwargs["hardware_compatibility_level"] = trt.HardwareCompatibilityLevel.AMPERE_PLUS
        elif _hc in ("same_cc", "same_compute_capability", "same"):
            config_kwargs["hardware_compatibility_level"] = trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY
        elif _hc not in ("none", ""):
            print(f"[W] unknown hardware_compatibility '{hardware_compatibility}', using NONE (build-GPU only)")
        if "hardware_compatibility_level" in config_kwargs:
            print(f"[I] TensorRT hardware compatibility: {_hc} (PORTABLE engine; ~5-15% slower, larger)")

        # ---- tactic sources ----
        # CRITICAL: tactic_sources=[] (upstream daydream default) disables ALL optimized kernels.
        # Always keep at least cuBLAS + cuBLAS-LT.
        if enable_all_tactics:
            tactic_sources = [
            ]
            # Version-aware: TRT 10 deprecated and TRT 11 REMOVED CUBLAS/CUBLAS_LT/CUDNN tactic
            # sources (only EDGE_MASK_CONVOLUTIONS / JIT_CONVOLUTIONS remain in TRT 11). Build the
            # list from whatever this TRT actually exposes; if none of the legacy GEMM sources exist,
            # DON'T set tactic_sources at all (let TRT use its full default tactic set — the modern
            # kernels are built in). Only restricting would otherwise DISABLE everything.
            for src_name in ("CUBLAS", "CUBLAS_LT", "CUDNN", "EDGE_MASK_CONVOLUTIONS", "JIT_CONVOLUTIONS"):
                src = getattr(trt.TacticSource, src_name, None)
                if src is not None:
                    tactic_sources.append(1 << int(src))
            if tactic_sources:
                config_kwargs["tactic_sources"] = tactic_sources
            # else: leave unset → TRT default (all tactics)
        # When enable_all_tactics is False we intentionally leave tactic_sources UNSET (TRT default),
        # rather than the old [CUBLAS, CUBLAS_LT] which no longer exists in TRT 11.

        # TRT 11 removed the FP16/TF32 BUILDER FLAGS (networks are strongly-typed now: the ONNX is
        # already fp16, so precision is per-tensor, not a global builder flag). polygraphy raises
        # "fp16 in CreateConfig is not available on TensorRT 11" if we pass them. Gate on TRT major.
        # (This generalizes our original code's "TRT-RTX doesn't support the FP16 builder flag" note.)
        trt_major = int(trt.__version__.split(".")[0])
        precision_kwargs = {}
        if trt_major < 11:
            precision_kwargs = {"tf32": True, "fp16": fp16}

        engine = engine_from_network(
            network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
            config=CreateConfig(
                refittable=enable_refit, profiles=[p],
                load_timing_cache=timing_cache, **precision_kwargs, **config_kwargs,
            ),
            save_timing_cache=timing_cache,
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        logger.info(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        # Check if we can reuse existing buffers (OPTIMIZATION)
        if self._can_reuse_buffers(shape_dict, device):
            return
        
        # Clear existing buffers before reallocating
        self.tensors.clear()
        
        # Reset CUDA graph when buffers are reallocated
        # The captured graph becomes invalid with new memory addresses
        if self.cuda_graph_instance is not None:
            CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instance))
            self.cuda_graph_instance = None
            if hasattr(self, 'graph') and self.graph is not None:
                CUASSERT(cudart.cudaGraphDestroy(self.graph))
                self.graph = None
        
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)

            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)

            dtype_np = trt.nptype(self.engine.get_tensor_dtype(name))
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)

            tensor = torch.empty(tuple(shape),
                                 dtype=numpy_to_torch_dtype_dict[dtype_np]) \
                          .to(device=device)
            self.tensors[name] = tensor
        
        # Cache allocation parameters for reuse check
        self._last_shape_dict = shape_dict.copy() if shape_dict else None
        self._last_device = device
    
    def _can_reuse_buffers(self, shape_dict=None, device="cuda"):
        """
        Check if existing buffers can be reused (avoiding expensive reallocation)
        
        Returns:
            bool: True if buffers can be reused, False if reallocation needed
        """
        # No existing tensors - need to allocate
        if not self.tensors:
            return False
        
        # Device changed - need to reallocate
        if not hasattr(self, '_last_device') or self._last_device != device:
            return False
        
        # No cached shape_dict - need to allocate
        if not hasattr(self, '_last_shape_dict'):
            return False
            
        # Compare current vs cached shape_dict
        if shape_dict is None and self._last_shape_dict is None:
            return True
        elif shape_dict is None or self._last_shape_dict is None:
            return False
        
        # Quick check: if tensor counts differ, can't reuse
        if len(shape_dict) != len(self._last_shape_dict):
            return False
        
        # Compare shapes for all tensors in the new shape_dict
        for name, new_shape in shape_dict.items():
            # Check if tensor exists in cached shapes
            cached_shape = self._last_shape_dict.get(name)
            if cached_shape is None:
                return False
            
            # Compare shapes (handle different types consistently)
            if tuple(cached_shape) != tuple(new_shape):
                return False
        
        return True

    def reset_cuda_graph(self):
        if self.cuda_graph_instance is not None:
            CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instance))
            self.cuda_graph_instance = None
        if hasattr(self, 'graph') and self.graph is not None:
            CUASSERT(cudart.cudaGraphDestroy(self.graph))
            self.graph = None

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        # Filter inputs to only those the engine actually exposes to avoid binding errors
        try:
            allowed_inputs = set()
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    allowed_inputs.add(name)

            # Drop any extra keys (e.g., text_embeds/time_ids) that the engine was not built to accept
            if allowed_inputs:
                filtered_feed_dict = {k: v for k, v in feed_dict.items() if k in allowed_inputs}
                if len(filtered_feed_dict) != len(feed_dict):
                    missing = [k for k in feed_dict.keys() if k not in allowed_inputs]
                    if missing:
                        logger.debug(
                            "TensorRT Engine: filtering unsupported inputs %s (allowed=%s)",
                            missing, sorted(list(allowed_inputs))
                        )
                feed_dict = filtered_feed_dict
        except Exception:
            # Be permissive if engine query fails; proceed with original dict
            pass
        
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
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
    builder_optimization_level: int = 5,  # 0-5; surfaced as a user toggle upstream. Higher = faster runtime, slower build.
    hardware_compatibility=None,  # None/'none' | 'ampere_plus' | 'same_cc' — portable cross-GPU engines
):
    # NOTE (librediffusion fork): ported Phase-3.x optimizations from bundled librediffusion —
    # TF32 torch backends (CC>=8.0), VRAM-tiered workspace, per-engine timing cache.
    _, free_mem, _ = cudart.cudaMemGetInfo()
    GiB = 2**30

    # TF32 for Ampere+ (CC>=8.0): ~faster fp32 matmul/conv with minimal accuracy loss (PyTorch 2.9+ API).
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if cc[0] >= 8:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
            logger.info(f"TF32 enabled for {torch.cuda.get_device_name()} (CC {cc[0]}.{cc[1]})")
        else:
            logger.info(f"TF32 not available on CC {cc[0]}.{cc[1]} (requires 8.0+)")

    # Percentage-based workspace (more tactics available than the old fixed 4GiB carveout).
    if free_mem > 8 * GiB:
        max_workspace_size = int(free_mem * 0.75)
    elif free_mem > 6 * GiB:
        max_workspace_size = int(free_mem * 0.70)
    elif free_mem > 4 * GiB:
        max_workspace_size = int(free_mem * 0.60)
    else:
        max_workspace_size = int(free_mem * 0.50)
    logger.info(f"VRAM {free_mem / GiB:.1f} GB free, using {max_workspace_size / GiB:.2f} GB workspace")

    # Timing cache speeds up subsequent rebuilds of the same engine.
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
        builder_optimization_level=builder_optimization_level,
        hardware_compatibility=hardware_compatibility,
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
    # TODO: Not 100% happy about this function - needs refactoring
    
    is_sdxl = False
    is_sdxl_controlnet = False

    # Detect if this is a ControlNet model (vs UNet model)
    is_controlnet = (
        hasattr(model, '__class__') and 'ControlNet' in model.__class__.__name__
    ) or (
        hasattr(model, 'config') and hasattr(model.config, '_class_name') and
        'ControlNet' in model.config._class_name
    )

    # Detect if this is an SDXL model via detect_model
    if hasattr(model, 'unet'):
        detection_result = detect_model(model.unet)
        if detection_result is not None:
            is_sdxl = detection_result.get('is_sdxl', False)
    elif hasattr(model, 'config'):
        detection_result = detect_model(model)
        if detection_result is not None:
            is_sdxl = detection_result.get('is_sdxl', False)
    
    # Detect if this is an SDXL ControlNet
    is_sdxl_controlnet = is_controlnet and (is_sdxl or (
        hasattr(model, 'config') and
        getattr(model.config, 'addition_embed_type', None) == 'text_time'
    ))
    
    wrapped_model = model  # Default: use model as-is

    # librediffusion: if the model already exposes the 5-input SDXL signature (our SDXLUNetWrapper,
    # which feeds text_embeds/time_ids as real graph inputs for the C++ engine), DON'T auto-wrap with
    # SDXLExportWrapper (that expects 3 inputs and zero-fills the conditioning).
    if getattr(model, "_sdxl_export_ready", False):
        is_sdxl = False  # skip the auto-wrap branch below; use the model as-is

    # Apply SDXL wrapper for SDXL models (in practice, always UnifiedExportWrapper)
    if is_sdxl and not is_controlnet:
        embedding_dim = getattr(model_data, 'embedding_dim', 'unknown')
        logger.info(f"Detected SDXL model (embedding_dim={embedding_dim}), using wrapper for ONNX export...")
        from .export_wrappers.unet_sdxl_export import SDXLExportWrapper
        wrapped_model = SDXLExportWrapper(model)
    elif not is_controlnet:
        embedding_dim = getattr(model_data, 'embedding_dim', 'unknown')
        logger.info(f"Detected non-SDXL model (embedding_dim={embedding_dim}), using model as-is for ONNX export...")
    
    # SDXL ControlNet models need special wrapper for added_cond_kwargs
    elif is_sdxl_controlnet:
        logger.info("Detected SDXL ControlNet model, using specialized wrapper...")
        from .export_wrappers.controlnet_export import SDXLControlNetExportWrapper
        wrapped_model = SDXLControlNetExportWrapper(model)
    
    # Regular ControlNet models are exported directly
    elif is_controlnet:
        logger.info("Detected ControlNet model, exporting directly...")
        wrapped_model = model
    
    with torch.inference_mode(), torch.autocast("cuda"):
        inputs = model_data.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
        
        # Determine if we need external data format for large models (like SDXL)
        is_large_model = is_sdxl or (hasattr(model, 'config') and getattr(model.config, 'sample_size', 32) >= 64)
        
        # Export ONNX normally first
        torch.onnx.export(
            wrapped_model,
            inputs,
            onnx_path,
            export_params=True,
            opset_version=onnx_opset,
            do_constant_folding=True,
            input_names=model_data.get_input_names(),
            output_names=model_data.get_output_names(),
            dynamic_axes=model_data.get_dynamic_axes(),
            dynamo=False,
        )
        
        # librediffusion: do NOT manually re-save large models with a separate weights.pb. For >2GB
        # SDXL, torch.onnx.export(export_params=True) already spills weights to a sibling external-data
        # file that the downstream optimize/build path resolves. The old manual re-save wrote
        # location="weights.pb" next to unet.onnx, but optimize_onnx/build read unet.opt.onnx (a
        # different file) → broken external refs → "Inputs available in the TensorRT network: set()".
        # (Our bundled exporter handles SDXL with no manual re-save; match that.)
    del wrapped_model
    gc.collect()
    torch.cuda.empty_cache()


def optimize_onnx(
    onnx_path: str,
    onnx_opt_path: str,
    model_data: BaseModel,
):
    import os

    # librediffusion fork (2026-06-05): ALWAYS load with external data (a no-op if the model has no
    # external refs) and decide the SAVE format by the optimized model's serialized size, not by a
    # fragile filename heuristic.
    #
    # Old code keyed on `*.pb` files existing in onnx_dir. But torch.onnx.export(export_params=True)
    # auto-spills a >2GB model to files named after the TENSORS (unet.add_embedding.linear_1.weight,
    # onnx__Add_NNNN, ...) — NONE end in `.pb`. So for the from-scratch SDXL UNet the heuristic saw
    # `uses_external_data=False`, took the else branch, and called onnx.save() WITHOUT external data on
    # a 5.1GB graph → protobuf's hard 2GB limit truncated/corrupted unet.opt.onnx → TRT failed with
    # "MODEL_DESERIALIZE_FAILED ... Failed to parse the ONNX model". Size-based save format fixes it for
    # any spill convention (torch tensor-named, single weights.pb, or none).
    opt_dir = os.path.dirname(onnx_opt_path)
    os.makedirs(opt_dir, exist_ok=True)
    # Clean stale OPT artifacts only (a prior failed run may have left a truncated unet.opt.onnx /
    # weights.pb). IMPORTANT: opt_dir is often the SAME dir as the input onnx_path, so we must NOT
    # delete the input model (unet.onnx) or its torch-spilled external-data files — only the
    # opt-prefixed outputs. (An earlier version deleted all *.onnx here, including the input, which
    # broke both onnxslim and the fallback with FileNotFoundError: unet.onnx.)
    opt_base = os.path.basename(onnx_opt_path)            # e.g. unet.opt.onnx
    in_base = os.path.basename(onnx_path)                 # e.g. unet.onnx
    if os.path.exists(opt_dir):
        for f in os.listdir(opt_dir):
            full = os.path.join(opt_dir, f)
            if full == os.path.abspath(onnx_path) or f == in_base:
                continue  # never delete the input model
            # opt artifacts: the opt onnx itself + its sibling weights.pb / *.data external file.
            if f == opt_base or f == "weights.pb" or f.endswith('.opt.onnx') or f.endswith('.opt.onnx.data'):
                try:
                    os.remove(full)
                except OSError:
                    pass

    # librediffusion fork (2026-06-05): for LARGE (>2GB) models (SDXL UNet), optimize with ONNXSLIM
    # via its FILE->FILE API. Rationale: the in-memory model_data.optimize() path (onnxsim +
    # onnxoptimizer) CANNOT handle >2GB protos — onnxsim bails on dynamic shapes and onnxoptimizer
    # silently returns an empty graph (protobuf 2GB serialization limit). models.py already SKIPS those
    # passes for >2GB, so SDXL would otherwise get only gs cleanup+fold. onnxslim 0.1.94 loads/saves by
    # PATH with auto external-data, so it round-trips a 5GB model and gives real fusion. We DON'T pass
    # input_shapes (keep the engine's dynamic batch/H/W axes — input_shapes would pin them). If onnxslim
    # is unavailable or errors, fall back to the gs-only path (still correct, just less fused).
    # Engine PERF is unaffected either way (TRT redoes node fusion itself; NVIDIA's demo/Diffusion omits
    # these passes) — onnxslim mainly buys parser robustness + a smaller ONNX artifact.
    # Detect a LARGE model by inspecting the input ONNX proto for EXTERNAL-DATA tensor refs (the
    # definitive >2GB signal — torch.onnx.export spills weights externally only when the model exceeds
    # protobuf's 2GB limit). This is robust regardless of how/where the weights were spilled (a previous
    # directory-size heuristic was unreliable: same dir as opt_dir, files deleted/flushed at scan time).
    use_onnxslim = False
    try:
        meta = onnx.load(onnx_path, load_external_data=False)
        for init in meta.graph.initializer:
            # data_location == EXTERNAL (1) means a weight lives in a sidecar -> the model is >2GB.
            if init.data_location == onnx.TensorProto.EXTERNAL:
                use_onnxslim = True
                break
        del meta
    except Exception:
        use_onnxslim = False
    if use_onnxslim:
        try:
            import onnxslim
            logger.info("ONNX optimize: large model (external-data weights) — using onnxslim "
                        "file->file (auto external data)")
            onnxslim.slim(
                onnx_path,
                onnx_opt_path,
                model_check=False,
                save_as_external_data=True,
            )
            # onnxslim wrote unet.opt.onnx (+ external data). Done — skip the in-memory path.
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("ONNX optimization complete with onnxslim (large model)")
            return
        except ImportError:
            logger.info("onnxslim not installed — falling back to gs-only optimize for the large model")
        except Exception as e:
            logger.warning(f"onnxslim failed ({e}); falling back to gs-only optimize for the large model")
            # Clean any partial onnxslim output before the fallback re-saves.
            for f in os.listdir(opt_dir):
                if f.endswith('.pb') or f.endswith('.onnx') or f.endswith('.data'):
                    try:
                        os.remove(os.path.join(opt_dir, f))
                    except OSError:
                        pass

    onnx_model = onnx.load(onnx_path, load_external_data=True)
    onnx_opt_graph = model_data.optimize(onnx_model)

    # ByteSize() raises/overflows past 2GB on some protobuf builds; guard it.
    try:
        opt_too_large = onnx_opt_graph.ByteSize() > 2147483648
    except (ValueError, Exception):
        opt_too_large = True

    if opt_too_large:
        # Save optimized model with external data (single sibling weights.pb the TRT parser resolves).
        onnx.save_model(
            onnx_opt_graph,
            onnx_opt_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,
        )
        logger.info("ONNX optimization complete with external data (model >2GB)")
    else:
        onnx.save(onnx_opt_graph, onnx_opt_path)
        logger.info("ONNX optimization complete (single file)")

    del onnx_opt_graph
    gc.collect()
    torch.cuda.empty_cache()
