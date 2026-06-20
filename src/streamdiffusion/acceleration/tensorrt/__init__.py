import gc
import os

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from polygraphy import cuda

from ...pipeline import StreamDiffusion
from .builder import EngineBuilder, create_onnx_path
from .engine import AutoencoderKLEngine, UNet2DConditionModelEngine
from .models import VAE, BaseModel, UNet, VAEEncoder, CLIP


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor):
        return retrieve_latents(self.vae.encode(x))


def compile_vae_encoder(
    vae: TorchVAEEncoder,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    # Convert to FP16 for TensorRT-RTX compatibility
    vae = vae.to(torch.device("cuda"), dtype=torch.float16)
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))

    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )


def compile_vae_decoder(
    vae: AutoencoderKL,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    # Convert to FP16 for TensorRT-RTX compatibility
    vae = vae.to(torch.device("cuda"), dtype=torch.float16)
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))

    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )


def compile_unet(
    unet,  # Can be UNet2DConditionModel or SDXLUNetWrapper
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
    is_sdxl: bool = False,
    export_attention_cache: bool = False,
):
    unet = unet.to(torch.device("cuda"), dtype=torch.float16)

    # StreamV2V: Wrap UNet to export attention intermediate outputs
    if export_attention_cache:
        # This wrapper will capture and return attention outputs for caching
        class UNetWithAttentionOutputs(torch.nn.Module):
            def __init__(self, unet_model, num_attention_blocks=16):
                super().__init__()
                self.unet_model = unet_model
                self.num_attention_blocks = num_attention_blocks
                self.attention_outputs = [None] * num_attention_blocks
                self.attention_idx = [0]  # Use list for mutability in closure
                self._register_attention_hooks()

            def _register_attention_hooks(self):
                """Register forward hooks to capture attention outputs"""
                # Find all attention transformer blocks (not sub-modules like to_q, to_k, etc.)
                attention_modules = []
                for name, module in self.unet_model.named_modules():
                    # Match only the main transformer blocks with self-attention
                    if 'transformer_blocks' in name and 'attn1' in name and name.endswith('attn1'):
                        attention_modules.append((name, module))

                # Limit to num_attention_blocks
                attention_modules = attention_modules[:self.num_attention_blocks]

                def create_hook(idx):
                    def hook(module, input, output):
                        # Store the output at the fixed index
                        self.attention_outputs[idx] = output
                    return hook

                for idx, (name, module) in enumerate(attention_modules):
                    module.register_forward_hook(create_hook(idx))

                print(f"[I] Registered {len(attention_modules)} attention hooks")

            def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
                # Reset outputs (fill with zeros to maintain fixed structure)
                for i in range(self.num_attention_blocks):
                    self.attention_outputs[i] = None

                # Run the UNet forward pass (this will populate attention_outputs via hooks)
                latent_out = self.unet_model(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)

                # Handle different return types from unet_model
                if isinstance(latent_out, tuple):
                    latent = latent_out[0]
                else:
                    latent = latent_out

                # Filter out None values and return fixed number of outputs
                valid_outputs = [out for out in self.attention_outputs if out is not None]

                # Pad with zeros if needed to maintain fixed structure for ONNX
                while len(valid_outputs) < self.num_attention_blocks:
                    valid_outputs.append(torch.zeros_like(valid_outputs[0] if valid_outputs else latent))

                return (latent, *valid_outputs[:self.num_attention_blocks])

        unet_wrapped = UNetWithAttentionOutputs(unet, num_attention_blocks=16)
    else:
        unet_wrapped = unet

    builder = EngineBuilder(model_data, unet_wrapped, device=torch.device("cuda"))

    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )


def compile_clip(
    text_encoder,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
    output_hidden_states: bool = False,
):
    # Convert to FP16 for TensorRT-RTX compatibility
    text_encoder = text_encoder.to(torch.device("cuda"), dtype=torch.float16)

    # Wrap the text encoder to output hidden states if needed (for SDXL CLIP2)
    # We need to create a wrapper class to avoid modifying the original model
    if output_hidden_states:
        # Create a wrapper module that outputs both hidden states and text embeds
        class CLIPWithHiddenStates(torch.nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.clip_model = clip_model

            def forward(self, input_ids):
                # Force output_hidden_states=True for ONNX export
                outputs = self.clip_model(input_ids, output_hidden_states=True, return_dict=True)
                # Return (last_hidden_state, text_embeds) for ONNX export
                return outputs.last_hidden_state, outputs.text_embeds

        text_encoder_for_export = CLIPWithHiddenStates(text_encoder)
    else:
        text_encoder_for_export = text_encoder

    builder = EngineBuilder(model_data, text_encoder_for_export, device=torch.device("cuda"))

    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )


def accelerate_with_tensorrt(
    stream: StreamDiffusion,
    engine_dir: str,
    max_batch_size: int = 2,
    min_batch_size: int = 1,
    use_cuda_graph: bool = True,  # Phase 3.1: Enabled by default for 10-15% latency reduction
    engine_build_options: dict = {},
    static_shapes: bool = False,  # Phase 3.1: Changed to False by default for flexibility
    use_v2v: bool = False,  # StreamV2V: Enable attention caching
):
    # Check and notify if TensorRT-RTX is being used
    use_rtx = os.environ.get("USE_TRT_RTX", "false").lower() in ("true", "1", "yes")
    if use_rtx:
        print("=" * 80)
        print("TensorRT-RTX mode enabled (USE_TRT_RTX=true)")
        print("Engines will be built using TensorRT-RTX for improved performance")
        print("=" * 80)

    # Phase 3.3: Memory pool optimization (2-4% gain)
    # Configure PyTorch CUDA memory allocator for better performance
    # - Expandable segments reduce fragmentation
    # - Larger max_split_size reduces allocation overhead
    if torch.cuda.is_available():
        # Enable expandable segments for better memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        print("[I] CUDA memory allocator optimized: expandable_segments=True, max_split_size=128MB")

    # Phase 3.3: Optimize CPU threading for GPU workloads (5-10% additional gain)
    # Reduce OpenMP thread count to minimize barrier synchronization overhead
    # For GPU-bound workloads, excessive CPU threads cause unnecessary waiting
    # Optimal: 2-4 threads for host-side coordination
    optimal_threads = min(4, os.cpu_count() or 4)
    os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
    os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
    torch.set_num_threads(optimal_threads)
    print(f"[I] CPU thread count optimized for GPU workload: {optimal_threads} threads (reduces barrier overhead)")

    # Phase 3.1: Apply static shape optimization for better performance
    # Static shapes enable more aggressive TensorRT optimizations and kernel fusion
    if "build_static_batch" not in engine_build_options:
        engine_build_options["build_static_batch"] = static_shapes
    if "build_dynamic_shape" not in engine_build_options:
        engine_build_options["build_dynamic_shape"] = not static_shapes

    if static_shapes:
        print(f"[I] Static shape optimization enabled - fixed batch={max_batch_size} for maximum performance")
    else:
        print(f"[I] Dynamic shape mode - flexible batch sizes {min_batch_size}-{max_batch_size}")

    # Determine opt_batch_size - prefer user setting, otherwise use max_batch_size
    opt_batch_size = engine_build_options.pop("opt_batch_size", max_batch_size)
    text_encoder = stream.text_encoder
    unet = stream.unet
    vae = stream.vae

    # Save configs and attributes before deleting models
    unet_config = unet.config
    # For SDXL, save add_embedding for validation in prepare()
    unet_add_embedding = unet.add_embedding if hasattr(unet, 'add_embedding') else None
    vae_config = vae.config
    vae_dtype = vae.dtype

    del stream.unet, stream.vae
    if hasattr(stream.pipe, 'unet'):
        del stream.pipe.unet
    if hasattr(stream.pipe, 'vae'):
        del stream.pipe.vae

    unet.to(torch.device("cpu"))
    vae.to(torch.device("cpu"))

    gc.collect()
    torch.cuda.empty_cache()

    onnx_dir = os.path.join(engine_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    unet_engine_path = f"{engine_dir}/unet.engine"
    vae_encoder_engine_path = f"{engine_dir}/vae_encoder.engine"
    vae_decoder_engine_path = f"{engine_dir}/vae_decoder.engine"
    clip_engine_path = f"{engine_dir}/clip.engine"
    clip2_engine_path = f"{engine_dir}/clip2.engine"  # SDXL second CLIP encoder

    # Check if we'll use pre-built ONNX (needed for model selection)
    prebuilt_onnx_dir = os.path.join(os.path.dirname(engine_dir), "engines_sdxl_turbo")
    prebuilt_unet_onnx = os.path.join(prebuilt_onnx_dir, "unetxl.opt", "model.onnx")
    use_prebuilt_onnx = stream.sdxl and os.path.exists(prebuilt_unet_onnx)

    # Use SDXL-specific model configuration if SDXL pipeline
    if stream.sdxl:
        from .models import SDXLUNet, SDXLUNetPrebuilt
        # SDXL uses pooled embeddings from text_encoder_2
        pooled_dim = stream.pipe.text_encoder_2.config.projection_dim
        # SDXL cross attention uses concatenated embeddings (text_encoder + text_encoder_2)
        cross_attention_dim = unet.config.cross_attention_dim

        # Use SDXLUNetPrebuilt for pre-built ONNX (static timestep shape)
        UNetClass = SDXLUNetPrebuilt if use_prebuilt_onnx else SDXLUNet

        unet_model = UNetClass(
            fp16=True,
            device=stream.device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=cross_attention_dim,  # 2048 for SDXL
            unet_dim=unet.config.in_channels,
            pooled_embedding_dim=pooled_dim,  # 1280 for SDXL
        )
    else:
        # Use UNetV2V if StreamV2V mode is enabled
        if use_v2v:
            from .models import UNetV2V
            unet_model = UNetV2V(
                fp16=True,
                device=stream.device,
                max_batch_size=max_batch_size,
                min_batch_size=min_batch_size,
                embedding_dim=text_encoder.config.hidden_size,
                unet_dim=unet.config.in_channels,
                num_attention_outputs=16,  # Number of attention blocks to cache
            )
            print("[I] Using UNetV2V model with attention caching (16 blocks)")
        else:
            unet_model = UNet(
                fp16=True,
                device=stream.device,
                max_batch_size=max_batch_size,
                min_batch_size=min_batch_size,
                embedding_dim=text_encoder.config.hidden_size,
                unet_dim=unet.config.in_channels,
            )
    vae_decoder_model = VAE(
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
    )
    vae_encoder_model = VAEEncoder(
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
    )
    clip_model = CLIP(
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        embedding_dim=text_encoder.config.hidden_size,
    )

    if not os.path.exists(vae_decoder_engine_path):
        vae.forward = vae.decode
        compile_vae_decoder(
            vae,
            vae_decoder_model,
            create_onnx_path("vae_decoder", onnx_dir, opt=False),
            create_onnx_path("vae_decoder", onnx_dir, opt=True),
            vae_decoder_engine_path,
            opt_batch_size=opt_batch_size,
            engine_build_options=engine_build_options,
        )

    if not os.path.exists(vae_encoder_engine_path):
        vae_encoder = TorchVAEEncoder(vae).to(torch.device("cuda"))
        compile_vae_encoder(
            vae_encoder,
            vae_encoder_model,
            create_onnx_path("vae_encoder", onnx_dir, opt=False),
            create_onnx_path("vae_encoder", onnx_dir, opt=True),
            vae_encoder_engine_path,
            opt_batch_size=opt_batch_size,
            engine_build_options=engine_build_options,
        )

    del vae

    if not os.path.exists(clip_engine_path):
        compile_clip(
            text_encoder,
            clip_model,
            create_onnx_path("clip", onnx_dir, opt=False),
            create_onnx_path("clip", onnx_dir, opt=True),
            clip_engine_path,
            opt_batch_size=opt_batch_size,
            engine_build_options=engine_build_options,
        )
    else:
        del text_encoder

    # Build second CLIP encoder for SDXL
    if stream.sdxl and not os.path.exists(clip2_engine_path):
        print("[I] Building second CLIP encoder (clip2) for SDXL...")
        text_encoder_2 = stream.pipe.text_encoder_2

        # CRITICAL: For SDXL CLIP2, we need output_hidden_states=True to export BOTH:
        # - hidden_states [B, 77, 1280] for concatenation with CLIP1
        # - text_embeddings [B, 1280] for pooled conditioning
        clip2_model = CLIP(
            device=stream.device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=text_encoder_2.config.hidden_size,
            output_hidden_states=True,  # CRITICAL: Export both outputs!
        )

        # Build from PyTorch model with output_hidden_states=True
        compile_clip(
            text_encoder_2,
            clip2_model,
            create_onnx_path("clip2", onnx_dir, opt=False),
            create_onnx_path("clip2", onnx_dir, opt=True),
            clip2_engine_path,
            opt_batch_size=opt_batch_size,
            engine_build_options=engine_build_options,
            output_hidden_states=True,  # CRITICAL: Export both outputs!
        )
        print(f"[I] ✓ CLIP2 engine built: {clip2_engine_path}")
        
    if not os.path.exists(unet_engine_path):
        if use_prebuilt_onnx:
            print(f"[I] Using pre-built SDXL UNet ONNX from: {prebuilt_unet_onnx}")
            # Copy pre-built ONNX to our onnx directory for TensorRT build
            os.makedirs(onnx_dir, exist_ok=True)
            onnx_opt_path = create_onnx_path("unet", onnx_dir, opt=True)
   
            # Copy the ONNX file and any external data
            import shutil
            prebuilt_dir = os.path.dirname(prebuilt_unet_onnx)
            shutil.copy(prebuilt_unet_onnx, onnx_opt_path)
   
            # Copy external data file if it exists
            for file in os.listdir(prebuilt_dir):
                if file != "model.onnx" and not file.startswith('.'):
                    src = os.path.join(prebuilt_dir, file)
                    dst = os.path.join(os.path.dirname(onnx_opt_path), file)
                    if os.path.isfile(src):
                        print(f"[I] Copying external data: {file}")
                        shutil.copy(src, dst)
   
            del unet
            # Build TensorRT engine from pre-built ONNX
            from .utilities import build_engine
   
            print(f"[I] Building TensorRT engine from pre-built ONNX...")
            build_engine(
                engine_path=unet_engine_path,
                onnx_opt_path=onnx_opt_path,
                model_data=unet_model,
                opt_image_height=stream.height,
                opt_image_width=stream.width,
                opt_batch_size=opt_batch_size,
                **engine_build_options,
            )
        else:
            # Original path: export and optimize ONNX from PyTorch model
            if stream.sdxl:
                from .sdxl_unet_wrapper import SDXLUNetWrapper
                unet_wrapped = SDXLUNetWrapper(unet)
            else:
                unet_wrapped = unet
   
            compile_unet(
                unet_wrapped,
                unet_model,
                create_onnx_path("unet", onnx_dir, opt=False),
                create_onnx_path("unet", onnx_dir, opt=True),
                unet_engine_path,
                opt_batch_size=opt_batch_size,
                engine_build_options=engine_build_options,
                is_sdxl=stream.sdxl,
                export_attention_cache=use_v2v,  # Pass V2V flag
            )
    else:
        del unet

    # Multi-stream optimization: Create separate CUDA streams for encoder, unet, and decoder
    # This allows:
    # 1. Better kernel scheduling by CUDA driver
    # 2. Potential overlap of memory transfers and computation
    # 3. Pipeline parallelism in streaming scenarios
    encoder_stream = cuda.Stream()
    unet_stream = cuda.Stream()
    decoder_stream = cuda.Stream()

    stream.unet = UNet2DConditionModelEngine(unet_engine_path, unet_stream, use_cuda_graph=use_cuda_graph)
    stream.vae = AutoencoderKLEngine(
        vae_encoder_engine_path,
        vae_decoder_engine_path,
        encoder_stream,  # encoder uses encoder_stream
        decoder_stream,  # decoder uses decoder_stream
        stream.pipe.vae_scale_factor,
        use_cuda_graph=use_cuda_graph,
    )
    # Preserve original configs and attributes for compatibility with StreamDiffusion pipeline
    setattr(stream.unet, "config", unet_config)
    if unet_add_embedding is not None:
        setattr(stream.unet, "add_embedding", unet_add_embedding)
    setattr(stream.vae, "config", vae_config)
    setattr(stream.vae, "dtype", vae_dtype)

    # Store stream references for potential synchronization needs
    stream._trt_encoder_stream = encoder_stream
    stream._trt_unet_stream = unet_stream
    stream._trt_decoder_stream = decoder_stream

    gc.collect()
    torch.cuda.empty_cache()

    return stream
