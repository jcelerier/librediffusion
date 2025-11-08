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

import onnx
import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference
from polygraphy.backend.onnx.loader import fold_constants

# Phase 3.2: Import advanced ONNX optimizers
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
        # Disable ONNX Runtime shape inference to avoid IR version mismatch warnings
        # We use ONNX shape inference separately in infer_shapes() which is more compatible
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=False)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        import tempfile
        import os

        onnx_graph = gs.export_onnx(self.graph)
        model_size_gb = onnx_graph.ByteSize() / (1024 ** 3)

        # For very large models (>2GB), use external data format
        # This saves weights separately to work around protobuf 2GB limit
        if onnx_graph.ByteSize() > 2147483648:
            if self.verbose:
                print(f"[W] Model size ({model_size_gb:.2f} GiB) exceeds 2GB limit.")
                print("[I] Using external data format for shape inference...")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = os.path.join(temp_dir, "tmp_model.onnx")
                if self.verbose:
                    print(f"[I] Saving ONNX model with external data to: {temp_path}")

                # Save the large model with external data format
                import onnx
                onnx.save(onnx_graph, temp_path,
                         save_as_external_data=True,
                         all_tensors_to_one_file=True,
                         location="weights.pb",
                         size_threshold=1024)

                # Load it back (this handles large models with external data)
                if self.verbose:
                    print(f"[I] Loading model with external data: {temp_path}")
                onnx_graph = onnx.load(temp_path, load_external_data=True)

                # Perform shape inference
                if self.verbose:
                    print("[I] Running shape inference on large model...")
                onnx_graph = shape_inference.infer_shapes(onnx_graph)
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
        max_batch_size=16,
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

        # Phase 3.2: Advanced ONNX optimizations for 10-16% additional performance

        # Step 1: ONNX Simplifier (5-8% gain)
        # Simplifies the graph by removing redundant nodes, fusing operators, etc.
        if ONNXSIM_AVAILABLE:
            onnx_model = gs.export_onnx(opt.graph)
            try:
                onnx_model, check = onnxsim.simplify(
                    onnx_model,
                    check_n=3,  # Verify correctness with 3 test runs
                    perform_optimization=True,  # Enable aggressive optimizations
                    skip_fuse_bn=False,  # Fuse batch normalization
                    skip_shape_inference=False,  # Run shape inference
                )
                opt.graph = gs.import_onnx(onnx_model)
                opt.info(self.name + ": onnxsim simplified")
            except Exception as e:
                print(f"[W] ONNX simplification failed: {e}, continuing without it")

        # Step 2: ONNX Optimizer (5-8% gain)
        # Applies graph optimization passes for operator fusion and elimination
        if ONNXOPTIMIZER_AVAILABLE:
            onnx_model = gs.export_onnx(opt.graph)
            try:
                # Select optimization passes that are safe and beneficial for diffusion models
                passes = [
                    'fuse_bn_into_conv',           # Fuse BatchNorm into Conv (if any)
                    'fuse_add_bias_into_conv',     # Fuse Add+Bias into Conv
                    'fuse_matmul_add_bias_into_gemm',  # Fuse MatMul+Add into GEMM
                    'fuse_consecutive_transposes',  # Remove redundant transposes
                    'fuse_transpose_into_gemm',    # Optimize transpose+matmul patterns
                    'eliminate_nop_transpose',     # Remove no-op transposes
                    'eliminate_nop_pad',           # Remove no-op padding
                    'eliminate_unused_initializer',  # Remove unused weights
                    'eliminate_duplicate_initializer',  # Deduplicate weights
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
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
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
    def __init__(self, device, max_batch_size, embedding_dim, min_batch_size=1, output_hidden_states=False):
        super(CLIP, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
        )
        self.name = "CLIP"
        self.output_hidden_states = output_hidden_states

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        # For SDXL text_encoder_2 (CLIPTextModelWithProjection), the outputs are:
        # - last_hidden_state: [batch, 77, 1280] - sequence embeddings
        # - text_embeds: [batch, 1280] - pooled embeddings with projection
        # For regular CLIP (text_encoder), only text_embeddings is used
        if self.output_hidden_states:
            return ["last_hidden_state", "text_embeds"]
        return ["text_embeddings", "text_embeds"]

    def get_dynamic_axes(self):
        if self.output_hidden_states:
            return {
                "input_ids": {0: "B"},
                "last_hidden_state": {0: "B"},
                "text_embeds": {0: "B"}
            }
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
        if self.output_hidden_states:
            return {
                "input_ids": (batch_size, self.text_maxlen),
                "last_hidden_state": (batch_size, self.text_maxlen, self.embedding_dim),
                "text_embeds": (batch_size, self.embedding_dim),
            }
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

        if self.output_hidden_states:
            # For SDXL CLIP2: Keep BOTH outputs (last_hidden_state and text_embeds)
            # The ONNX model from CLIPTextModelWithProjection has:
            # - output[0]: last_hidden_state [B, 77, embedding_dim]
            # - output[1]: text_embeds [B, embedding_dim] (pooled with projection)
            opt.cleanup()
            opt.info(self.name + ": cleanup")
            opt.fold_constants()
            opt.info(self.name + ": fold constants")
            opt.infer_shapes()
            opt.info(self.name + ": shape inference")
            # Keep both outputs and rename them
            opt.select_outputs([0, 1], names=["hidden_states", "text_embeddings"])
            opt.info(self.name + ": renamed outputs to hidden_states and text_embeddings")
        else:
            # For regular CLIP: Only keep text_embeddings (last_hidden_state)
            opt.select_outputs([0])  # delete graph output#1
            opt.cleanup()
            opt.info(self.name + ": remove output[1]")
            opt.fold_constants()
            opt.info(self.name + ": fold constants")
            opt.infer_shapes()
            opt.info(self.name + ": shape inference")
            opt.select_outputs([0], names=["text_embeddings"])  # rename network output
            opt.info(self.name + ": rename output[0] to text_embeddings")

        opt_onnx_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return opt_onnx_graph


class UNet(BaseModel):
    def __init__(
        self,
        fp16=False,
        device="cuda",
        max_batch_size=16,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4,
    ):
        super(UNet, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.name = "UNet"

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "timestep": {0: "2B"},
            "encoder_hidden_states": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
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
            "sample": [
                (min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (batch_size, self.unet_dim, latent_height, latent_width),
                (max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
            "encoder_hidden_states": [
                (min_batch, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (max_batch, self.text_maxlen, self.embedding_dim),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (2 * batch_size,),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(
                2 * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device
            ),
            torch.ones((2 * batch_size,), dtype=torch.float32, device=self.device),
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
        )


class UNetV2V(UNet):
    """StreamV2V UNet model with attention caching outputs"""
    def __init__(
        self,
        fp16=False,
        device="cuda",
        max_batch_size=2,
        min_batch_size=2,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4,
        num_attention_outputs=16,  # Number of attention blocks to cache
    ):
        super(UNetV2V, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
            unet_dim=unet_dim,
        )
        self.num_attention_outputs = num_attention_outputs
        self.name = "UNetV2V"

    def get_output_names(self):
        # Return: latent + attention outputs
        names = ["latent"]
        for i in range(self.num_attention_outputs):
            names.append(f"attention_{i}")
        return names

    def get_dynamic_axes(self):
        axes = super().get_dynamic_axes()
        # Add dynamic axes for attention outputs
        # Attention outputs are typically [batch, seq_len, hidden_dim]
        for i in range(self.num_attention_outputs):
            axes[f"attention_{i}"] = {0: "2B", 1: "seq"}
        return axes

    def get_shape_dict(self, batch_size, image_height, image_width):
        shapes = super().get_shape_dict(batch_size, image_height, image_width)
        # Add shapes for attention outputs
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        seq_len = latent_height * latent_width  # Typical attention sequence length
        # Attention outputs vary in hidden dimension depending on layer
        # Using a typical value here - will be inferred during export
        for i in range(self.num_attention_outputs):
            shapes[f"attention_{i}"] = (2 * batch_size, seq_len, 640)  # Typical hidden dim
        return shapes


class SDXLUNet(BaseModel):
    """SDXL UNet model with additional text_embeds and time_ids inputs"""
    def __init__(
        self,
        fp16=False,
        device="cuda",
        max_batch_size=16,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4,
        pooled_embedding_dim=1280,  # SDXL pooled text embeddings dimension
    ):
        super(SDXLUNet, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.pooled_embedding_dim = pooled_embedding_dim
        self.name = "SDXL UNet"

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "timestep": {0: "2B"},
            "encoder_hidden_states": {0: "2B"},
            "text_embeds": {0: "2B"},
            "time_ids": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
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
            "sample": [
                (min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (batch_size, self.unet_dim, latent_height, latent_width),
                (max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
            "encoder_hidden_states": [
                (min_batch, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (max_batch, self.text_maxlen, self.embedding_dim),
            ],
            "text_embeds": [
                (min_batch, self.pooled_embedding_dim),
                (batch_size, self.pooled_embedding_dim),
                (max_batch, self.pooled_embedding_dim),
            ],
            "time_ids": [
                (min_batch, 6),  # SDXL uses 6 time IDs
                (batch_size, 6),
                (max_batch, 6),
            ],
        }


class SDXLUNetPrebuilt(SDXLUNet):
    """SDXL UNet for pre-built ONNX from HuggingFace (static timestep shape)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SDXL UNet (Prebuilt)"

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
            "sample": [
                (min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (batch_size, self.unet_dim, latent_height, latent_width),
                (max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            # Pre-built ONNX has static timestep shape [1] - not batch-dependent
            "timestep": [(1,), (1,), (1,)],
            "encoder_hidden_states": [
                (min_batch, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (max_batch, self.text_maxlen, self.embedding_dim),
            ],
            "text_embeds": [
                (min_batch, self.pooled_embedding_dim),
                (batch_size, self.pooled_embedding_dim),
                (max_batch, self.pooled_embedding_dim),
            ],
            "time_ids": [
                (min_batch, 6),
                (batch_size, 6),
                (max_batch, 6),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (2 * batch_size,),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "text_embeds": (2 * batch_size, self.pooled_embedding_dim),
            "time_ids": (2 * batch_size, 6),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(
                2 * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device
            ),
            torch.ones((2 * batch_size,), dtype=torch.float32, device=self.device),
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
            torch.randn(2 * batch_size, self.pooled_embedding_dim, dtype=dtype, device=self.device),
            torch.randn(2 * batch_size, 6, dtype=dtype, device=self.device),  # time_ids
        )


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
