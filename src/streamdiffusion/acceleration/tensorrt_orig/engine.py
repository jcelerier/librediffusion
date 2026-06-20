from typing import *

import torch
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTinyOutput
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.autoencoders.vae import DecoderOutput
from polygraphy import cuda

from .utilities import Engine


class UNet2DConditionModelEngine:
    def __init__(self, filepath: str, stream: cuda.Stream, use_cuda_graph: bool = False):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph

        self.engine.load()
        self.engine.activate()

        # Cache for shape tracking to avoid unnecessary reallocation
        self.current_shapes = {}
        self.buffers_allocated = False

    def __call__(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ) -> Any:
        if timestep.dtype != torch.float32:
            timestep = timestep.float()

        # Handle SDXL additional conditioning
        added_cond_kwargs = kwargs.get("added_cond_kwargs", {})
        text_embeds = added_cond_kwargs.get("text_embeds", None)
        time_ids = added_cond_kwargs.get("time_ids", None)

        # Build shapes dict - include SDXL inputs if present
        shapes = {
            "sample": latent_model_input.shape,
            "timestep": timestep.shape,
            "encoder_hidden_states": encoder_hidden_states.shape,
            "latent": latent_model_input.shape,
        }

        # Add SDXL-specific inputs if present
        if text_embeds is not None:
            shapes["text_embeds"] = text_embeds.shape
        if time_ids is not None:
            shapes["time_ids"] = time_ids.shape

        if not self.buffers_allocated or shapes != self.current_shapes:
            self.engine.allocate_buffers(
                shape_dict=shapes,
                device=latent_model_input.device,
            )
            self.current_shapes = shapes
            self.buffers_allocated = True

        # Build inference inputs - include SDXL inputs if present
        infer_inputs = {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        if text_embeds is not None:
            infer_inputs["text_embeds"] = text_embeds
        if time_ids is not None:
            infer_inputs["time_ids"] = time_ids

        noise_pred = self.engine.infer(
            infer_inputs,
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]
        return UNet2DConditionOutput(sample=noise_pred)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class AutoencoderKLEngine:
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        encoder_stream: cuda.Stream,
        decoder_stream: cuda.Stream,
        scaling_factor: int,
        use_cuda_graph: bool = False,
    ):
        self.encoder = Engine(encoder_path)
        self.decoder = Engine(decoder_path)
        # Multi-stream optimization: Use separate streams for encoder and decoder
        self.encoder_stream = encoder_stream
        self.decoder_stream = decoder_stream
        self.vae_scale_factor = scaling_factor
        self.use_cuda_graph = use_cuda_graph

        self.encoder.load()
        self.decoder.load()
        self.encoder.activate()
        self.decoder.activate()

        # Cache for shape tracking to avoid unnecessary reallocation
        self.encoder_shapes = {}
        self.decoder_shapes = {}
        self.encoder_buffers_allocated = False
        self.decoder_buffers_allocated = False

    def encode(self, images: torch.Tensor, **kwargs):
        # Only reallocate buffers if shapes changed
        shapes = {
            "images": images.shape,
            "latent": (
                images.shape[0],
                4,
                images.shape[2] // self.vae_scale_factor,
                images.shape[3] // self.vae_scale_factor,
            ),
        }

        if not self.encoder_buffers_allocated or shapes != self.encoder_shapes:
            self.encoder.allocate_buffers(
                shape_dict=shapes,
                device=images.device,
            )
            self.encoder_shapes = shapes
            self.encoder_buffers_allocated = True

        latents = self.encoder.infer(
            {"images": images},
            self.encoder_stream,  # Use encoder-specific stream
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]
        return AutoencoderTinyOutput(latents=latents)

    def decode(self, latent: torch.Tensor, **kwargs):
        # Only reallocate buffers if shapes changed
        shapes = {
            "latent": latent.shape,
            "images": (
                latent.shape[0],
                3,
                latent.shape[2] * self.vae_scale_factor,
                latent.shape[3] * self.vae_scale_factor,
            ),
        }

        if not self.decoder_buffers_allocated or shapes != self.decoder_shapes:
            self.decoder.allocate_buffers(
                shape_dict=shapes,
                device=latent.device,
            )
            self.decoder_shapes = shapes
            self.decoder_buffers_allocated = True

        images = self.decoder.infer(
            {"latent": latent},
            self.decoder_stream,  # Use decoder-specific stream
            use_cuda_graph=self.use_cuda_graph,
        )["images"]
        return DecoderOutput(sample=images)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
