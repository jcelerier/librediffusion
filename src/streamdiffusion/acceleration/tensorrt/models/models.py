#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/models.py

#
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

import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference
from polygraphy.backend.onnx.loader import fold_constants
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

# librediffusion fork: optional advanced ONNX graph optimizers used in BaseModel.optimize.
try:
    import onnxsim
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False
    print("[I] onnxsim not available, skipping ONNX simplification")

try:
    import onnxoptimizer
    ONNXOPTIMIZER_AVAILABLE = True
except ImportError:
    ONNXOPTIMIZER_AVAILABLE = False
    print("[I] onnxoptimizer not available, skipping operator fusion")


class Optimizer:
    def __init__(self, onnx_graph, verbose=False):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            print(f"⚠️ Model size ({onnx_graph.ByteSize() / (1024**3):.2f} GB) exceeds 2GB - this is normal for SDXL models")
            print("🔧 ONNX shape inference will be skipped for large models to avoid memory issues")
            # For large models like SDXL, skip shape inference to avoid memory/size issues
            # The model will still work with TensorRT's own shape inference during engine building
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph


class BaseModel:
    def __init__(
        self,
        fp16=False,
        device="cuda",
        verbose=True,
        max_batch_size=4,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
    ):
        self.name = "SD Model"
        self.fp16 = fp16
        self.device = device
        self.verbose = verbose

        self.min_batch = min_batch_size
        self.max_batch = max_batch_size
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8
        # librediffusion fork: independent latent H/W bounds so a non-square engine
        # (e.g. 512x768) can pin width and height to different values. Default to the
        # shared square bounds so behavior is unchanged unless the builder overrides them.
        self.min_latent_height = self.min_latent_shape
        self.max_latent_height = self.max_latent_shape
        self.min_latent_width = self.min_latent_shape
        self.max_latent_width = self.max_latent_shape

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_model(self):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ": original")
        opt.cleanup()
        opt.info(self.name + ": cleanup")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")

        # librediffusion fork: advanced ONNX optimizations (onnxsim + onnxoptimizer).
        # NOTE: these change graph numerics slightly — re-validate output tolerance after enabling.
        #
        # librediffusion fork (2026-06-05): SKIP these passes for >2GB models (SDXL). They operate on
        # an in-memory ONNX proto (gs.export_onnx), and protobuf cannot serialize/round-trip a >2GB
        # proto without external data. onnxsim then bails on the dynamic `sample` shape (caught), but
        # onnxoptimizer SILENTLY returns an EMPTY graph (0 nodes/0 inputs) for the oversize proto — not
        # an exception — which we'd then import, write as unet.opt.onnx, and feed to TRT → the engine
        # build fails with "Inputs available in the TensorRT network are: set()". fold_constants above
        # (gs/polygraphy) already handles the big graph; infer_shapes() below already self-skips >2GB
        # (same guard). So for SDXL we rely on the gs cleanup+fold path only. The numeric-altering
        # onnxsim/onnxoptimizer fusions are a perf nicety, not required for correctness.
        onnx_too_large = gs.export_onnx(opt.graph).ByteSize() > 2147483648
        if onnx_too_large:
            print(f"[I] {self.name}: model >2GB — skipping onnxsim/onnxoptimizer (in-memory proto "
                  f"passes can't handle >2GB; would empty the graph). Using gs fold/cleanup only.")

        if ONNXSIM_AVAILABLE and not onnx_too_large:
            onnx_model = gs.export_onnx(opt.graph)
            try:
                onnx_model, _check = onnxsim.simplify(
                    onnx_model,
                    check_n=3,
                    perform_optimization=True,
                    skip_fuse_bn=False,
                    skip_shape_inference=False,
                )
                opt.graph = gs.import_onnx(onnx_model)
                opt.info(self.name + ": onnxsim simplified")
            except Exception as e:
                print(f"[W] ONNX simplification failed: {e}, continuing without it")

        if ONNXOPTIMIZER_AVAILABLE and not onnx_too_large:
            onnx_model = gs.export_onnx(opt.graph)
            try:
                passes = [
                    "fuse_bn_into_conv",
                    "fuse_add_bias_into_conv",
                    "fuse_matmul_add_bias_into_gemm",
                    "fuse_consecutive_transposes",
                    "fuse_transpose_into_gemm",
                    "eliminate_nop_transpose",
                    "eliminate_nop_pad",
                    "eliminate_unused_initializer",
                    "eliminate_duplicate_initializer",
                ]
                onnx_model = onnxoptimizer.optimize(onnx_model, passes)
                opt.graph = gs.import_onnx(onnx_model)
                opt.info(self.name + ": onnxoptimizer applied")
            except Exception as e:
                print(f"[W] ONNX optimizer failed: {e}, continuing without it")

        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        # Make batch size check more flexible for ONNX export
        if hasattr(self, '_allow_export_batch_override') and self._allow_export_batch_override:
            # During ONNX export, allow different batch sizes
            effective_min_batch = min(self.min_batch, batch_size)
            effective_max_batch = max(self.max_batch, batch_size)
        else:
            effective_min_batch = self.min_batch
            effective_max_batch = self.max_batch
            
        assert batch_size >= effective_min_batch and batch_size <= effective_max_batch, \
            f"Batch size {batch_size} not in range [{effective_min_batch}, {effective_max_batch}]"
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        # librediffusion fork: independent H/W latent bounds (non-square engine support).
        assert latent_height >= self.min_latent_height and latent_height <= self.max_latent_height
        assert latent_width >= self.min_latent_width and latent_width <= self.max_latent_width
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        # Following ComfyUI TensorRT approach: ensure proper min ≤ opt ≤ max constraints
        # Even with static_batch=True, we need different min/max to avoid TensorRT constraint violations
        
        if static_batch:
            # For static batch, still provide range to avoid min=opt=max constraint violation
            min_batch = max(1, batch_size - 1)  # At least 1, but allow some range
            max_batch = batch_size
        else:
            min_batch = self.min_batch
            max_batch = self.max_batch
        
        latent_height = image_height // 8
        latent_width = image_width // 8

        # librediffusion fork: independent H/W bounds so non-square engines are first-class.
        # The latent bounds are the source of truth (set by the builder from the W/H ranges);
        # image bounds derive as latent*8. When a dim's min==max (e.g. a static non-square
        # engine pinning H=768,W=512) it is naturally pinned to that exact value — no square
        # collapse. Static engines arise when both H and W have min==max.
        min_latent_height = self.min_latent_height
        max_latent_height = self.max_latent_height
        min_latent_width = self.min_latent_width
        max_latent_width = self.max_latent_width
        min_image_height = min_latent_height * 8
        max_image_height = max_latent_height * 8
        min_image_width = min_latent_width * 8
        max_image_width = max_latent_width * 8

        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )


class CLIP(BaseModel):
    def __init__(self, device, max_batch_size, embedding_dim, min_batch_size=1):
        super(CLIP, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
        )
        self.name = "CLIP"

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        return ["text_embeddings", "pooler_output"]

    def get_dynamic_axes(self):
        return {"input_ids": {0: "B"}, "text_embeddings": {0: "B"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape
        )
        return {
            "input_ids": [
                (min_batch, self.text_maxlen),
                (batch_size, self.text_maxlen),
                (max_batch, self.text_maxlen),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph)
        opt.info(self.name + ": original")
        opt.select_outputs([0])
        opt.cleanup()
        opt.info(self.name + ": remove output[1]")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        opt.select_outputs([0], names=["text_embeddings"])
        opt.info(self.name + ": remove output[0]")
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return opt_onnx_graph


class CLIPSDXLPooled(CLIP):
    """SDXL second text encoder (CLIPTextModelWithProjection): keeps BOTH outputs.

    librediffusion fork (2026-06-05): the base CLIP spec strips output[1] (pooler) in optimize() —
    correct for SD1.5 (no pooled needed) but WRONG for SDXL clip2, which the C++ CLIPWrapper::
    computeEmbeddingsWithPooled REQUIRES to expose TWO outputs:
      - hidden_states   : [B, 77, 1280]  (penultimate SEQUENCE embeds; the C++ reads this 3D)
      - text_embeddings : [B, 1280]      (POOLED projection; the C++ detects CLIP2 by this being 2D)
    The compile_clip CLIPWithHiddenStates wrapper returns (seq, text_embeds) in that order, so
    output[0]=hidden_states (seq), output[1]=text_embeddings (pooled). Without this, the from-scratch
    clip2 engine emitted only a single 3D `text_embeddings` (seq) — the C++ then took the CLIP1 path,
    never produced the pooled output, and the harness crashed reading a garbage sdxl_pooled pointer.
    (The prebuilt sdxl-turbo clip2 happened to ship these two outputs already, which is why the
    prebuilt route worked and the from-scratch route exposed this gap.)
    """

    def __init__(self, device, max_batch_size, embedding_dim, min_batch_size=1):
        super().__init__(device, max_batch_size, embedding_dim, min_batch_size=min_batch_size)
        self.name = "CLIP-SDXL-pooled"

    def get_output_names(self):
        # Order matches the export wrapper's return: (penultimate seq, pooled text_embeds).
        return ["hidden_states", "text_embeddings"]

    def get_dynamic_axes(self):
        return {
            "input_ids": {0: "B"},
            "hidden_states": {0: "B"},
            "text_embeddings": {0: "B"},
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            "input_ids": (batch_size, self.text_maxlen),
            "hidden_states": (batch_size, self.text_maxlen, self.embedding_dim),  # 77 x 1280 seq
            "text_embeddings": (batch_size, self.embedding_dim),                  # 1280 pooled
        }

    def optimize(self, onnx_graph):
        # Keep BOTH outputs; just fold/cleanup. Rename to the C++-expected names in wrapper order.
        opt = Optimizer(onnx_graph)
        opt.info(self.name + ": original")
        opt.cleanup()
        opt.info(self.name + ": cleanup")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        opt.select_outputs([0, 1], names=["hidden_states", "text_embeddings"])
        opt.info(self.name + ": keep both outputs")
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return opt_onnx_graph


class SafetyChecker(BaseModel):
    def __init__(self, device, max_batch_size = 1, min_batch_size = 1):
        super(SafetyChecker, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
        )
        self.name = "safety_checker"

    def get_input_names(self):
        return ["clip_input"]

    def get_output_names(self):
        return ["has_nsfw_concepts"]

    def get_dynamic_axes(self):
        return {"clip_input": {0: "B"}}

    def get_input_profile(self, batch_size, *args, **kwargs):
        return {
            "clip_input": [
                (self.min_batch, 3, 224, 224),
                (batch_size, 3, 224, 224),
                (self.max_batch, 3, 224, 224),
            ],
        }

    def get_shape_dict(self, batch_size, *args, **kwargs):
        return {
            "clip_input": (batch_size, 3, 224, 224),
            "has_nsfw_concepts": (batch_size,),
        }

    def get_sample_input(self, batch_size, *args, **kwargs):
        return (
            torch.randn(batch_size, 3, 224, 224, dtype=torch.float16, device=self.device),
        )

class NSFWDetector(BaseModel):
    def __init__(self, device, max_batch_size = 1, min_batch_size = 1):
        super(NSFWDetector, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
        )
        self.name = "nsfw_detector"
    
    def get_input_names(self):
        return ["pixel_values"]
    
    def get_output_names(self):
        return ["logits"]
    
    def get_dynamic_axes(self):
        return {"pixel_values": {0: "B"}}
    
    def get_input_profile(self, batch_size, *args, **kwargs):
        return {
            "pixel_values": [
                (self.min_batch, 3, 448, 448),
                (batch_size, 3, 448, 448),
                (self.max_batch, 3, 448, 448),
            ],
        }
    
    def get_shape_dict(self, batch_size, *args, **kwargs):
        return {
            "pixel_values": (batch_size, 3, 448, 448),
            "logits": (batch_size, 2),
        }
    
    def get_sample_input(self, batch_size, *args, **kwargs):
        return (
            torch.randn(batch_size, 3, 448, 448, dtype=torch.float16, device=self.device),
        )

class UNet(BaseModel):
    def __init__(
        self,
        unet: UNet2DConditionModel = None,
        fp16=False,
        device="cuda",
        max_batch_size=4,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4,
        use_control=False,
        unet_arch=None,
        image_height=512,
        image_width=512,
        use_ipadapter=False,
        num_image_tokens=4,
        num_ip_layers: int = None,
        use_cached_attn: bool = False,
        cache_maxframes: int = 1,
        min_cache_maxframes: int = 1,
        max_cache_maxframes: int = 4,
    ):
        super(UNet, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet = unet
        self.unet_dim = unet_dim
        self.name = "UNet"
        self.image_height = image_height
        self.image_width = image_width
        
        self.use_control = use_control
        self.unet_arch = unet_arch or {}
        self.use_ipadapter = use_ipadapter
        self.num_image_tokens = num_image_tokens
        self.num_ip_layers = num_ip_layers
        
        # Baked-in IPAdapter configuration
        if self.use_ipadapter:
            # With baked-in processors, we extend text_maxlen to include image tokens
            # TODO: Consider making this dynamic instead of fixed per IPAdapter variant
            # Could use dynamic shapes: min=77 (text only), max=93 (text + 16 tokens)
            # This would allow a single engine to handle all IPAdapter types instead of separate engines
            self.text_maxlen = text_maxlen + self.num_image_tokens
            if self.num_ip_layers is None:
                raise ValueError("UNet model requires num_ip_layers when use_ipadapter=True")

        
        if self.use_control and self.unet_arch:
            self.control_inputs = self.get_control(image_height, image_width)
            self._add_control_inputs()
        else:
            self.control_inputs = {}

        self.use_cached_attn = use_cached_attn
        self.cache_maxframes = cache_maxframes
        self.min_cache_maxframes = min_cache_maxframes
        self.max_cache_maxframes = max_cache_maxframes
        if self.use_cached_attn and self.unet is not None:
            from .utils import get_kvo_cache_info
            self.kvo_cache_shapes, self.kvo_cache_structure, self.kvo_cache_count = get_kvo_cache_info(self.unet, image_height, image_width)
            
            self.min_kvo_cache_shapes, _, _ = get_kvo_cache_info(self.unet, image_height, image_width)
            self.max_kvo_cache_shapes, _, _ = get_kvo_cache_info(self.unet, image_height, image_width)

    def get_control(self, image_height: int = 512, image_width: int = 512) -> dict:
        """Generate ControlNet input configurations with dynamic spatial dimensions based on input resolution."""
        block_out_channels = self.unet_arch.get('block_out_channels', (320, 640, 1280, 1280))
        layers_per_block = self.unet_arch.get('layers_per_block', None)

        # Calculate latent space dimensions
        latent_height = image_height // 8
        latent_width = image_width // 8

        control_inputs = {}

        if layers_per_block is not None:
            # GENERIC layout from the real diffusers down_block_res_samples structure — works for
            # any UNet (SD1.5: 4 blocks/lpb2 -> 12; SDXL: 3/lpb2 -> 9; SDXS: 3/lpb1 -> 6). The
            # res_samples are: conv_in (1, factor 1) + per block i: lpb residuals at factor f_i, then
            # (if not last block) 1 downsampled residual at factor 2*f_i. Channels = block_out[i].
            control_tensors = [(block_out_channels[0], 1)]  # conv_in
            f = 1
            n = len(block_out_channels)
            for i in range(n):
                for _ in range(layers_per_block):
                    control_tensors.append((block_out_channels[i], f))
                if i < n - 1:
                    f *= 2
                    control_tensors.append((block_out_channels[i], f))  # downsampler output
            middle_downsample_factor = f
            middle_channels = block_out_channels[-1]
        elif len(block_out_channels) == 3:
            # SDXL architecture: Match UNet's exact down_block_res_samples structure
            # UNet down_block_res_samples = [initial_sample] + [block0_residuals] + [block1_residuals] + [block2_residuals]
            # Pattern: [88x88] + [88x88, 88x88, 44x44] + [44x44, 44x44, 22x22] + [22x22, 22x22]
            # Total: 9 control tensors needed
            control_tensors = [
                # Initial sample (after conv_in: 4->320 channels, no downsampling)
                (block_out_channels[0], 1),  # 320 channels, 88x88
                
                # Block 0 residuals (320 channels)
                (block_out_channels[0], 1),  # 320 channels, 88x88 
                (block_out_channels[0], 1),  # 320 channels, 88x88
                (block_out_channels[0], 2),  # 320 channels, 44x44 (downsampled)
                
                # Block 1 residuals (640 channels) 
                (block_out_channels[1], 2),  # 640 channels, 44x44
                (block_out_channels[1], 2),  # 640 channels, 44x44
                (block_out_channels[1], 4),  # 640 channels, 22x22 (downsampled)
                
                # Block 2 residuals (1280 channels)
                (block_out_channels[2], 4),  # 1280 channels, 22x22
                (block_out_channels[2], 4),  # 1280 channels, 22x22
            ]
        else:
            # SD1.5/SD2.1 architecture: 4 down blocks with 12 control tensors
            control_tensors = [
                # Block 0: No downsampling from latent space (factor = 1)
                (320, 1), (320, 1), (320, 1),
                # Block 1: 2x downsampling from latent space (factor = 2) 
                (320, 2), (640, 2), (640, 2),
                # Block 2: 4x downsampling from latent space (factor = 4)
                (640, 4), (1280, 4), (1280, 4),
                # Block 3: 8x downsampling from latent space (factor = 8)
                (1280, 8), (1280, 8), (1280, 8)
            ]
        
        # Generate control inputs with proper spatial dimensions
        for i, (channels, downsample_factor) in enumerate(control_tensors):
            input_name = f"input_control_{i:02d}"
            
            # Calculate spatial dimensions for this level
            control_height = max(1, latent_height // downsample_factor)
            control_width = max(1, latent_width // downsample_factor)
            
            control_inputs[input_name] = {
                'batch': self.min_batch,
                'channels': channels,
                'height': control_height,
                'width': control_width,
                'downsampling_factor': downsample_factor
            }
        
        # Middle block: use the generic values when computed; else fall back per architecture.
        if layers_per_block is not None:
            pass  # middle_downsample_factor + middle_channels already set generically above
        elif len(block_out_channels) == 3:
            middle_downsample_factor = 4   # SDXL: after 3 down blocks
            middle_channels = 1280
        else:
            middle_downsample_factor = 8   # SD1.5: after 4 down blocks
            middle_channels = 1280

        control_inputs["input_control_middle"] = {
            'batch': self.min_batch,
            'channels': middle_channels,
            'height': max(1, latent_height // middle_downsample_factor),
            'width': max(1, latent_width // middle_downsample_factor),
            'downsampling_factor': middle_downsample_factor
        }
        
        return control_inputs

    def get_kvo_cache_names(self, in_out: str):
        return [f"kvo_cache_{in_out}_{idx}" for idx in range(self.kvo_cache_count)]

    def _add_control_inputs(self):
        """Add ControlNet inputs to the model's input/output specifications"""
        if not self.control_inputs:
            return
        
        self._original_get_input_names = self.get_input_names
        self._original_get_dynamic_axes = self.get_dynamic_axes
        self._original_get_input_profile = self.get_input_profile
        self._original_get_shape_dict = self.get_shape_dict
        self._original_get_sample_input = self.get_sample_input

    def get_input_names(self):
        """Get input names including ControlNet inputs"""
        base_names = ["sample", "timestep", "encoder_hidden_states"]
        if self.use_ipadapter:
            base_names.append("ipadapter_scale")
            try:
                import logging
                logging.getLogger(__name__).debug(f"TRT Models: get_input_names with ipadapter -> {base_names}")
            except Exception:
                pass
        if self.use_control and self.control_inputs:
            control_names = sorted(self.control_inputs.keys())
            base_names = base_names + control_names
        if self.use_cached_attn:
            base_names = base_names + self.get_kvo_cache_names("in")
        return base_names

    def get_output_names(self):
        base_names = ["latent"]
        if self.use_cached_attn:
            base_names = base_names + self.get_kvo_cache_names("out")
        return base_names

    def get_kvo_cache_input_profile(self, min_batch, batch_size, max_batch):
        profiles = []
        for min_shape, shape, max_shape in zip(self.min_kvo_cache_shapes, self.kvo_cache_shapes, self.max_kvo_cache_shapes):
            profile = [(2, self.min_cache_maxframes, min_batch, min_shape[0], min_shape[1]), (2, self.cache_maxframes, batch_size, shape[0], shape[1]), (2, self.max_cache_maxframes, max_batch, max_shape[0], max_shape[1])]
            profiles.append(profile)
        return profiles

    def get_dynamic_axes(self):
        base_axes = {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "timestep": {0: "2B"},
            "encoder_hidden_states": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
        }
        if self.use_ipadapter:
            base_axes["ipadapter_scale"] = {0: "L_ip"}
            try:
                import logging
                logging.getLogger(__name__).debug(f"TRT Models: dynamic axes include ipadapter_scale with L_ip={getattr(self, 'num_ip_layers', None)}")
            except Exception:
                pass
        
        if self.use_control and self.control_inputs:
            for name, shape_spec in self.control_inputs.items():
                height = shape_spec["height"]
                width = shape_spec["width"]
                spatial_suffix = f"{height}x{width}"
                base_axes[name] = {
                    0: "2B", 
                    2: f"H_{spatial_suffix}", 
                    3: f"W_{spatial_suffix}"
                }
        if self.use_cached_attn:
            # hardcoded resolution for now due to VRAM limitations
            for i in range(self.kvo_cache_count):
                base_axes[f"kvo_cache_in_{i}"] = {1: "C", 2: "2B"}
                base_axes[f"kvo_cache_out_{i}"] = {2: "2B"}
        
        return base_axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        
        # Following TensorRT documentation: ensure proper min ≤ opt ≤ max constraints for ALL dimensions
        # Calculate optimal latent dimensions that fall within min/max range
        opt_latent_height = min(max(latent_height, min_latent_height), max_latent_height)
        opt_latent_width = min(max(latent_width, min_latent_width), max_latent_width)
        
        # Ensure no dimension equality that causes constraint violations
        if opt_latent_height == min_latent_height and min_latent_height < max_latent_height:
            opt_latent_height = min(min_latent_height + 8, max_latent_height)  # Add 8 pixels for separation
        if opt_latent_width == min_latent_width and min_latent_width < max_latent_width:
            opt_latent_width = min(min_latent_width + 8, max_latent_width)
        
        # Image dimensions for ControlNet inputs
        # librediffusion fork: derive from independent H/W latent bounds (non-square support).
        min_image_h, max_image_h = self.min_latent_height * 8, self.max_latent_height * 8
        min_image_w, max_image_w = self.min_latent_width * 8, self.max_latent_width * 8
        opt_image_height = min(max(image_height, min_image_h), max_image_h)
        opt_image_width = min(max(image_width, min_image_w), max_image_w)
        
        # Ensure image dimension separation as well
        if opt_image_height == min_image_h and min_image_h < max_image_h:
            opt_image_height = min(min_image_h + 64, max_image_h)  # Add 64 pixels for separation
        if opt_image_width == min_image_w and min_image_w < max_image_w:
            opt_image_width = min(min_image_w + 64, max_image_w)
        
        profile = {
            "sample": [
                (min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (batch_size, self.unet_dim, opt_latent_height, opt_latent_width),
                (max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
            "encoder_hidden_states": [
                (min_batch, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (max_batch, self.text_maxlen, self.embedding_dim),
            ],
        }
        if self.use_ipadapter:
            # scalar per-layer vector, length fixed to num_ip_layers
            profile["ipadapter_scale"] = [
                (1,),
                (self.num_ip_layers,),
                (self.num_ip_layers,),
            ]
            try:
                import logging
                logging.getLogger(__name__).debug(f"TRT Models: profile ipadapter_scale min/opt/max={(1,),(self.num_ip_layers,),(self.num_ip_layers,)}")
            except Exception:
                pass
        
        if self.use_control and self.control_inputs:
            # Use the actual calculated spatial dimensions for each ControlNet input
            # Each control input has its own specific spatial resolution based on UNet architecture
            for name, shape_spec in self.control_inputs.items():
                channels = shape_spec["channels"]
                control_height = shape_spec["height"]
                control_width = shape_spec["width"]
                
                # Create optimization profile with proper spatial dimension scaling
                # Scale the spatial dimensions proportionally with the main latent dimensions
                scale_h = opt_latent_height / latent_height if latent_height > 0 else 1.0
                scale_w = opt_latent_width / latent_width if latent_width > 0 else 1.0
                
                min_control_h = max(1, int(control_height * min_latent_height / latent_height))
                max_control_h = max(min_control_h + 1, int(control_height * max_latent_height / latent_height))
                opt_control_h = max(min_control_h, min(int(control_height * scale_h), max_control_h))
                
                min_control_w = max(1, int(control_width * min_latent_width / latent_width))
                max_control_w = max(min_control_w + 1, int(control_width * max_latent_width / latent_width))
                opt_control_w = max(min_control_w, min(int(control_width * scale_w), max_control_w))
                
                profile[name] = [
                    (min_batch, channels, min_control_h, min_control_w),    # min
                    (batch_size, channels, opt_control_h, opt_control_w),   # opt  
                    (max_batch, channels, max_control_h, max_control_w),    # max
                ]
        if self.use_cached_attn:
            for name, _profile in zip(self.get_kvo_cache_names("in"), self.get_kvo_cache_input_profile(min_batch, batch_size, max_batch)):
                profile[name] = _profile
        
        return profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        shape_dict = {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (2 * batch_size,),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }
        if self.use_ipadapter:
            shape_dict["ipadapter_scale"] = (self.num_ip_layers,)
            try:
                import logging
                logging.getLogger(__name__).debug(f"TRT Models: shape_dict ipadapter_scale={(self.num_ip_layers,)}")
            except Exception:
                pass
        
        if self.use_control and self.control_inputs:
            # Use the actual calculated spatial dimensions for each ControlNet input
            for name, shape_spec in self.control_inputs.items():
                channels = shape_spec["channels"]
                control_height = shape_spec["height"]
                control_width = shape_spec["width"]
                shape_dict[name] = (2 * batch_size, channels, control_height, control_width)

        if self.use_cached_attn:
            for in_name, out_name, shape in zip(self.get_kvo_cache_names("in"), self.get_kvo_cache_names("out"), self.get_kvo_cache_shapes):
                shape_dict[in_name] = (2, self.cache_maxframes, batch_size, shape[0], shape[1])
                shape_dict[out_name] = (2, 1, batch_size, shape[0], shape[1])
        
        return shape_dict

    def get_sample_input(self, batch_size, image_height, image_width):
        # Enable flexible batch size checking for ONNX export
        self._allow_export_batch_override = True
        
        try:
            latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        finally:
            # Clean up the override flag
            if hasattr(self, '_allow_export_batch_override'):
                delattr(self, '_allow_export_batch_override')
        
        dtype = torch.float16 if self.fp16 else torch.float32
        
        # Use smaller batch size for memory efficiency during ONNX export
        export_batch_size = min(batch_size, 1)  # Use batch size 1 for ONNX export to save memory
        
        base_inputs = [
            torch.randn(
                2 * export_batch_size, self.unet_dim, latent_height, latent_width, 
                dtype=torch.float32, device=self.device
            ),
            torch.ones((2 * export_batch_size,), dtype=torch.float32, device=self.device),
            torch.randn(2 * export_batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
        ]
        
        if self.use_ipadapter:
            base_inputs.append(torch.ones(self.num_ip_layers, dtype=torch.float32, device=self.device))
        
        if self.use_control and self.control_inputs:
            control_inputs = []
            
            # Use the ACTUAL calculated spatial dimensions for each control input
            # This ensures each control input matches its expected UNet feature map resolution
            
            for name in sorted(self.control_inputs.keys()):
                shape_spec = self.control_inputs[name]
                channels = shape_spec["channels"]
                
                # KEY FIX: Use the specific spatial dimensions calculated for this control input
                control_height = shape_spec["height"]
                control_width = shape_spec["width"]
                
                control_input = torch.randn(
                    2 * export_batch_size, channels, control_height, control_width, 
                    dtype=dtype, device=self.device
                )
                control_inputs.append(control_input)
                
                # Clear cache periodically to prevent memory buildup
                if len(control_inputs) % 4 == 0:
                    torch.cuda.empty_cache()
            
            base_inputs = base_inputs + control_inputs
        
        if self.use_cached_attn:
            base_inputs = base_inputs + [torch.randn(2, self.cache_maxframes, 2 * export_batch_size, shape[0], shape[1], dtype=torch.float16).to(self.device) for shape in self.kvo_cache_shapes]
        return tuple(base_inputs)


class SDXLUNet(BaseModel):
    """SDXL UNet model spec with the extra text_embeds + time_ids inputs.

    librediffusion fork: ported from our bundled exporter. daydream's `UNet` spec only declares
    sample/timestep/encoder_hidden_states (no SDXL added conditioning), so SDXL UNet ONNX export
    fails with an empty network. This is a self-contained BaseModel subclass (no controlnet/ipadapter/
    kvo machinery) used for plain SDXL exports via SDXLUNetWrapper.
    """

    # SDXL ControlNet residual geometry (9 down + mid). channels + spatial downsample factor, matching
    # ControlNetSDXLTRT.get_output_shapes / UNet.get_control(3-block branch).
    _CTRL_DOWN = [(320, 1), (320, 1), (320, 1), (320, 2), (640, 2), (640, 2),
                  (640, 4), (1280, 4), (1280, 4)]
    _CTRL_MID = (1280, 4)

    def __init__(self, unet=None, fp16=False, device="cuda", max_batch_size=16, min_batch_size=1,
                 embedding_dim=2048, text_maxlen=77, unet_dim=4, pooled_embedding_dim=1280,
                 use_control=False, **kwargs):
        super().__init__(fp16=fp16, device=device, max_batch_size=max_batch_size,
                         min_batch_size=min_batch_size, embedding_dim=embedding_dim,
                         text_maxlen=text_maxlen)
        self.unet = unet
        self.unet_dim = unet_dim
        self.pooled_embedding_dim = pooled_embedding_dim
        self.use_control = use_control
        self.name = "SDXL UNet"

    def _control_names(self):
        # sorted-stable: input_control_00..08 then input_control_middle (matches the C++ binding order
        # and the base UNet's sorted(control_inputs.keys())).
        return [f"input_control_{i:02d}" for i in range(len(self._CTRL_DOWN))] + ["input_control_middle"]

    def get_input_names(self):
        names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]
        if self.use_control:
            names = names + self._control_names()
        return names

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        axes = {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "timestep": {0: "2B"},
            "encoder_hidden_states": {0: "2B"},
            "text_embeds": {0: "2B"},
            "time_ids": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
        }
        if self.use_control:
            # IMPORTANT: control residuals are at DOWNSAMPLED resolutions (H/2, H/4...), so their spatial
            # dims must NOT reuse the sample's "H"/"W" symbols — TRT would force one symbol to two values
            # ("contradictory kMIN/kMAX"). Give each control input its own symbolic spatial dims.
            for n in self._control_names():
                axes[n] = {0: "2B", 2: f"{n}_H", 3: f"{n}_W"}
        return axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (min_batch, max_batch, _, _, _, _,
         min_lh, max_lh, min_lw, max_lw) = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape)
        prof = {
            "sample": [(min_batch, self.unet_dim, min_lh, min_lw),
                       (batch_size, self.unet_dim, latent_height, latent_width),
                       (max_batch, self.unet_dim, max_lh, max_lw)],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
            "encoder_hidden_states": [(min_batch, self.text_maxlen, self.embedding_dim),
                                      (batch_size, self.text_maxlen, self.embedding_dim),
                                      (max_batch, self.text_maxlen, self.embedding_dim)],
            "text_embeds": [(min_batch, self.pooled_embedding_dim),
                            (batch_size, self.pooled_embedding_dim),
                            (max_batch, self.pooled_embedding_dim)],
            "time_ids": [(min_batch, 6), (batch_size, 6), (max_batch, 6)],
        }
        if self.use_control:
            names = self._control_names()
            specs = self._CTRL_DOWN + [self._CTRL_MID]
            for n, (ch, fac) in zip(names, specs):
                prof[n] = [(min_batch, ch, min_lh // fac, min_lw // fac),
                           (batch_size, ch, latent_height // fac, latent_width // fac),
                           (max_batch, ch, max_lh // fac, max_lw // fac)]
        return prof

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        d = {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (2 * batch_size,),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "text_embeds": (2 * batch_size, self.pooled_embedding_dim),
            "time_ids": (2 * batch_size, 6),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }
        if self.use_control:
            names = self._control_names()
            specs = self._CTRL_DOWN + [self._CTRL_MID]
            for n, (ch, fac) in zip(names, specs):
                d[n] = (2 * batch_size, ch, latent_height // fac, latent_width // fac)
        return d

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        inputs = [
            torch.randn(2 * batch_size, self.unet_dim, latent_height, latent_width,
                        dtype=torch.float32, device=self.device),
            torch.ones((2 * batch_size,), dtype=torch.float32, device=self.device),
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
            torch.randn(2 * batch_size, self.pooled_embedding_dim, dtype=dtype, device=self.device),
            torch.randn(2 * batch_size, 6, dtype=dtype, device=self.device),  # time_ids
        ]
        if self.use_control:
            specs = self._CTRL_DOWN + [self._CTRL_MID]
            for ch, fac in specs:
                inputs.append(torch.randn(2 * batch_size, ch, latent_height // fac, latent_width // fac,
                                          dtype=dtype, device=self.device))
        return tuple(inputs)


class SDXLUNetWrapper(torch.nn.Module):
    """Wraps an SDXL UNet so ONNX export takes text_embeds/time_ids as positional args.

    librediffusion fork: ported from our bundled sdxl_unet_wrapper.py. Builds added_cond_kwargs and
    returns the single sample output (also normalizes daydream's (sample, kvo_cache_out) tuple).
    """

    # Marker: tells export_onnx this model ALREADY exposes the 5-input SDXL signature (text_embeds/
    # time_ids as real graph inputs), so it must NOT auto-wrap with SDXLExportWrapper (which expects
    # the 3-input signature and zero-fills the conditioning — wrong for a C++ engine that needs them
    # as inputs).
    _sdxl_export_ready = True

    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.config = unet.config

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        out = self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs, return_dict=False)
        return out[0] if isinstance(out, (tuple, list)) else out


class SDXLUNetControlWrapper(torch.nn.Module):
    """Control-aware SDXL UNet export wrapper: takes the 9 down + mid ControlNet residuals as positional
    graph inputs (after the 5 SDXL inputs) and injects them as down_block_additional_residuals /
    mid_block_additional_residual. Order: sample, timestep, ehs, text_embeds, time_ids, ctrl0..8, mid.
    """

    _sdxl_export_ready = True

    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.config = unet.config

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids, *control):
        down = list(control[:-1])  # 9 down residuals
        mid = control[-1]
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        out = self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs,
                        down_block_additional_residuals=down,
                        mid_block_additional_residual=mid,
                        return_dict=False)
        return out[0] if isinstance(out, (tuple, list)) else out


class VAE(BaseModel):
    def __init__(self, device, max_batch_size, min_batch_size=1):
        super(VAE, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=None,
        )
        self.name = "VAE decoder"

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {
            "latent": {0: "B", 2: "H", 3: "W"},
            "images": {0: "B", 2: "8H", 3: "8W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "latent": [
                (min_batch, 4, min_latent_height, min_latent_width),
                (batch_size, 4, latent_height, latent_width),
                (max_batch, 4, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(
            batch_size,
            4,
            latent_height,
            latent_width,
            dtype=torch.float32,
            device=self.device,
        )


class VAEEncoder(BaseModel):
    def __init__(self, device, max_batch_size, min_batch_size=1):
        super(VAEEncoder, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=None,
        )
        self.name = "VAE encoder"

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "images": {0: "B", 2: "8H", 3: "8W"},
            "latent": {0: "B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            _,
            _,
            _,
            _,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            "images": [
                (min_batch, 3, min_image_height, min_image_width),
                (batch_size, 3, image_height, image_width),
                (max_batch, 3, max_image_height, max_image_width),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "images": (batch_size, 3, image_height, image_width),
            "latent": (batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(
            batch_size,
            3,
            image_height,
            image_width,
            dtype=torch.float32,
            device=self.device,
        )
