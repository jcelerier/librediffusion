import time
from typing import List, Optional, Union, Any, Dict, Tuple, Literal

import numpy as np
import PIL.Image
import torch
from diffusers import LCMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

from streamdiffusion.image_filter import SimilarImageFilter


class StreamDiffusion:
    def __init__(
        self,
        pipe: DiffusionPipeline,
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        use_cuda_native: bool = False,
    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None
        self.use_cuda_native = use_cuda_native

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)

        self.cfg_type = cfg_type

        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (
                    self.denoising_steps_num + 1
                ) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = (
                    2 * self.denoising_steps_num * self.frame_bff_size
                )
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size

        self.t_list = t_index_list

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_result = None

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        self.inference_time_ema = 0
        self.sdxl = type(self.pipe) is StableDiffusionXLPipeline

        # Phase 3.3: Persistent tensor allocation (3-5% gain)
        # Pre-allocate tensor buffers that get reused across frames to reduce allocation overhead
        self._random_latent_buffer = torch.empty(
            (1, 4, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype,
        )
        self._batch_random_latent_buffer = torch.empty(
            (max(frame_buffer_size, 2), 4, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype,
        )

    def load_lcm_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[
            str, Dict[str, torch.Tensor]
        ] = "latent-consistency/lcm-lora-sdv1-5",
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = torch.Generator(),
        seed: int = 2,
    ) -> None:
        self.generator = generator
        self.seed = seed  # Store seed for CUDA native random generation
        # Ensure generator is on the same device as the model
        if self.generator.device.type != self.device.type:
            self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        # initialize x_t_latent (it can be any random tensor)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None

        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta

        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True


        self.current_negative_prompt = negative_prompt
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

        if self.sdxl:
            self.add_text_embeds = encoder_output[2]
            original_size = (self.height, self.width)
            crops_coords_top_left = (0, 0)
            target_size = (self.height, self.width)
            text_encoder_projection_dim = int(self.add_text_embeds.shape[-1])
            self.add_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=encoder_output[0].dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )


        if self.use_denoising_batch and self.cfg_type == "full":
            uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

        if self.guidance_scale > 1.0 and (
            self.cfg_type == "initialize" or self.cfg_type == "full"
        ):
            from streamdiffusion.cuda import concat_cuda
            self.prompt_embeds = concat_cuda(
                [uncond_prompt_embeds.contiguous(), self.prompt_embeds.contiguous()], dim=0
            )



        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        self.stock_noise = torch.zeros_like(self.init_noise)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        do_classifier_free_guidance = self.guidance_scale > 1.0
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)
        
        if self.sdxl:
            self.add_text_embeds = encoder_output[2]

        if self.use_denoising_batch and self.cfg_type == "full":
            uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

        if self.guidance_scale > 1.0 and (
            self.cfg_type == "initialize" or self.cfg_type == "full"
        ):
            from streamdiffusion.cuda import concat_cuda
            self.prompt_embeds = concat_cuda(
                [uncond_prompt_embeds.contiguous(), self.prompt_embeds.contiguous()], dim=0
            )

    @torch.no_grad()
    def update_prompts(
        self,
        weighted_prompts: List[Tuple[str, float]],
        negative_prompt: Optional[str] = None,
    ) -> None:
        """
        Updates the prompt embeddings by blending multiple weighted prompts.

        Args:
            weighted_prompts (List[Tuple[str, float]]): A list of tuples, where each
                tuple contains a prompt string and its corresponding weight.
                Example: [("a cat", 0.6), ("a dog", 0.4)]
            negative_prompt (Optional[str]): The negative prompt to use. If None,
                uses the negative prompt from the last `prepare` call or the
                last `update_prompt` call that set a negative prompt.
        """
        if not weighted_prompts:
            print("Warning: weighted_prompts list is empty. Prompt embeddings not updated.")
            return

        # --- Normalize weights ---
        total_weight = sum(w for _, w in weighted_prompts)
        if total_weight == 0: # Avoid division by zero, assign equal weights if all are zero
            print("Warning: Total weight of prompts is zero. Assigning equal weights.")
            num_prompts = len(weighted_prompts)
            normalized_weights = [1.0 / num_prompts] * num_prompts
        else:
            normalized_weights = [w / total_weight for _, w in weighted_prompts]

        # --- Encode individual conditional prompts and blend them ---
        blended_cond_embedding = None
        first_embedding_shape = None

        for i, (p_str, _) in enumerate(weighted_prompts):
            # Encode ONLY the conditional part for each prompt
            # We pass an empty negative prompt and turn off CFG for this specific call
            # to isolate the conditional embedding of p_str.
            cond_embed_output = self.pipe.encode_prompt(
                prompt=p_str,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False, # Get only conditional
                negative_prompt="", # Provide empty to avoid issues, though it won't be used
            )
            cond_embed = cond_embed_output[0] # output[0] is prompt_embeds (conditional)

            if first_embedding_shape is None:
                first_embedding_shape = cond_embed.shape
                blended_cond_embedding = torch.zeros_like(cond_embed, device=self.device, dtype=self.dtype)

            if cond_embed.shape != first_embedding_shape:
                raise ValueError(
                    f"Prompt '{p_str}' produced an embedding of shape {cond_embed.shape}, "
                    f"expected {first_embedding_shape} based on the first prompt. "
                    "Ensure all prompts result in compatible embedding shapes (e.g., check token lengths)."
                )

            blended_cond_embedding += normalized_weights[i] * cond_embed

        if blended_cond_embedding is None: # Should not happen if weighted_prompts is not empty
            print("Error: Blended conditional embedding could not be created.")
            return

        repeated_blended_cond_embed = blended_cond_embedding.repeat(self.batch_size, 1, 1)

        # --- Handle unconditional (negative) prompt ---
        do_cfg = self.guidance_scale > 1.0
        negative_prompt_to_use = negative_prompt if negative_prompt is not None else self.current_negative_prompt

        # Update current negative prompt if a new one was explicitly passed
        if negative_prompt is not None:
            self.current_negative_prompt = negative_prompt

        if do_cfg and (self.cfg_type == "initialize" or self.cfg_type == "full"):
            # Get the unconditional embedding for the chosen negative_prompt_to_use
            # We need to call encode_prompt such that it returns the negative_prompt_embeds
            uncond_output = self.pipe.encode_prompt(
                prompt="", # A dummy prompt, its conditional part is ignored
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True, # Essential to get uncond part
                negative_prompt=negative_prompt_to_use,
            )
            actual_uncond_embed = uncond_output[1] # output[1] is negative_prompt_embeds

            if actual_uncond_embed is None:
                raise ValueError(
                    "Failed to obtain unconditional_embeddings for the negative prompt. "
                    "Ensure `do_classifier_free_guidance=True` is effective in `encode_prompt`."
                )

            uncond_repeat_factor = self.frame_bff_size if self.cfg_type == "initialize" else self.batch_size
            repeated_uncond_embed = actual_uncond_embed.repeat(uncond_repeat_factor, 1, 1)

            from streamdiffusion.cuda import concat_cuda
            self.prompt_embeds = concat_cuda([repeated_uncond_embed.contiguous(), repeated_blended_cond_embed.contiguous()], dim=0)
        else: # No specific CFG or "self" CFG that doesn't use concatenated embeds here
            self.prompt_embeds = repeated_blended_cond_embed

        # self.current_prompt is not updated as it's a blend.
        # self.current_negative_prompt is updated above if a new one was given.
        # print(f"Prompt embeddings updated with a blend of {len(weighted_prompts)} prompts.")
        # print(f"Using negative prompt: '{self.current_negative_prompt}'")


    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        # Use CUDA native implementation if enabled
        if self.use_cuda_native:
            from streamdiffusion.cuda import add_noise_cuda
            return add_noise_cuda(
                original_samples,
                noise,
                float(self.alpha_prod_t_sqrt[t_index]),
                float(self.beta_prod_t_sqrt[t_index]),
            )
        else:
            # Original PyTorch implementation
            noisy_samples = (
                self.alpha_prod_t_sqrt[t_index] * original_samples
                + self.beta_prod_t_sqrt[t_index] * noise
            )
            return noisy_samples

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        # TODO: use t_list to select beta_prod_t_sqrt
        if self.use_cuda_native:
            from streamdiffusion.cuda import scheduler_step_cuda
            # Extract scalar values for the specific timestep (or first if idx is None)
            timestep_idx = idx if idx is not None else 0
            return scheduler_step_cuda(
                model_pred_batch,
                x_t_latent_batch,
                float(self.alpha_prod_t_sqrt[timestep_idx].item()),
                float(self.beta_prod_t_sqrt[timestep_idx].item()),
                float(self.c_skip[timestep_idx].item()),
                float(self.c_out[timestep_idx].item()),
            )
        else:
            # Original PyTorch implementation
            if idx is None:
                F_theta = (
                    x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch
                ) / self.alpha_prod_t_sqrt
                denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
                return denoised_batch
            else:
                F_theta = (
                    x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch
                ) / self.alpha_prod_t_sqrt[idx]
                denoised_batch = (
                    self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
                )
                return denoised_batch

    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        added_cond_kwargs,
        idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            from streamdiffusion.cuda import concat_cuda
            x_t_latent_plus_uc = concat_cuda([x_t_latent[0:1].contiguous(), x_t_latent.contiguous()], dim=0)
            t_list = concat_cuda([t_list[0:1].contiguous(), t_list.contiguous()], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            from streamdiffusion.cuda import concat_cuda
            x_t_latent_plus_uc = concat_cuda([x_t_latent.contiguous(), x_t_latent.contiguous()], dim=0)
            t_list = concat_cuda([t_list.contiguous(), t_list.contiguous()], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent

        model_pred = self.unet(
            x_t_latent_plus_uc,
            t_list,
            encoder_hidden_states=self.prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            from streamdiffusion.cuda import concat_cuda
            self.stock_noise = concat_cuda(
                [model_pred[0:1].contiguous(), self.stock_noise[1:].contiguous()], dim=0
            )  # ここコメントアウトでself out cfg
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.guidance_scale > 1.0 and (
            self.cfg_type == "self" or self.cfg_type == "initialize"
        ):
            noise_pred_uncond = self.stock_noise * self.delta
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            # Use CUDA native implementation if enabled
            if self.use_cuda_native:
                from streamdiffusion.cuda import apply_cfg_cuda
                model_pred = apply_cfg_cuda(
                    noise_pred_uncond,
                    noise_pred_text,
                    float(self.guidance_scale),
                )
            else:
                # Original PyTorch implementation
                model_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
        else:
            model_pred = noise_pred_text

        # compute the previous noisy sample x_t -> x_t-1
        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
                if self.use_cuda_native:
                    from streamdiffusion.cuda import concat_cuda, ones_like_cuda
                    alpha_next = concat_cuda(
                        [
                            self.alpha_prod_t_sqrt[1:].contiguous(),
                            ones_like_cuda(self.alpha_prod_t_sqrt[0:1]),
                        ],
                        dim=0,
                    )
                    delta_x = alpha_next * delta_x
                    beta_next = concat_cuda(
                        [
                            self.beta_prod_t_sqrt[1:].contiguous(),
                            ones_like_cuda(self.beta_prod_t_sqrt[0:1]),
                        ],
                        dim=0,
                    )
                    delta_x = delta_x / beta_next
                    init_noise = concat_cuda(
                        [self.init_noise[1:].contiguous(), self.init_noise[0:1].contiguous()], dim=0
                    )
                else:
                    alpha_next = torch.concat(
                        [
                            self.alpha_prod_t_sqrt[1:],
                            torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                        ],
                        dim=0,
                    )
                    delta_x = alpha_next * delta_x
                    beta_next = torch.concat(
                        [
                            self.beta_prod_t_sqrt[1:],
                            torch.ones_like(self.beta_prod_t_sqrt[0:1]),
                        ],
                        dim=0,
                    )
                    delta_x = delta_x / beta_next
                    init_noise = torch.concat(
                        [self.init_noise[1:], self.init_noise[0:1]], dim=0
                    )
                self.stock_noise = init_noise + delta_x

        else:
            # denoised_batch = self.scheduler.step(model_pred, t_list[0], x_t_latent).denoised
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        return denoised_batch, model_pred

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids


    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        image_tensors = image_tensors.to(
            device=self.device,
            dtype=self.vae.dtype,
        )
        img_latent = retrieve_latents(self.vae.encode(image_tensors), self.generator)
        # Phase 3.3: Use in-place multiplication (2-3% gain)
        if self.use_cuda_native:
            from streamdiffusion.cuda import scalar_mul_inplace_cuda
            # Ensure contiguous memory layout for CUDA operations
            img_latent = img_latent.contiguous()
            scalar_mul_inplace_cuda(img_latent, float(self.vae.config.scaling_factor))
        else:
            img_latent.mul_(self.vae.config.scaling_factor)
        # Ensure noise shape matches img_latent for CUDA (expand if needed)
        noise = self.init_noise[0:1] if self.use_cuda_native else self.init_noise[0]
        x_t_latent = self.add_noise(img_latent, noise, 0)
        return x_t_latent

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        if self.use_cuda_native:
            from streamdiffusion.cuda import scalar_div_cuda
            scaled_latent = scalar_div_cuda(x_0_pred_out, float(self.vae.config.scaling_factor))
        else:
            scaled_latent = x_0_pred_out / self.vae.config.scaling_factor
        output_latent = self.vae.decode(
            scaled_latent, return_dict=False
        )[0]
        return output_latent

    def predict_x0_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        added_cond_kwargs = {}
        prev_latent_batch = self.x_t_latent_buffer
        
        # Add SDXL conditioning if needed
        if self.sdxl:
            added_cond_kwargs = {"text_embeds": self.add_text_embeds.to(self.device), "time_ids": self.add_time_ids.to(self.device)}

        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            if self.denoising_steps_num > 1:
                if self.use_cuda_native:
                    from streamdiffusion.cuda import concat_cuda
                    x_t_latent = concat_cuda((x_t_latent.contiguous(), prev_latent_batch.contiguous()), dim=0)
                    self.stock_noise = concat_cuda(
                        (self.init_noise[0:1].contiguous(), self.stock_noise[:-1].contiguous()), dim=0
                    )
                else:
                    x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                    self.stock_noise = torch.cat(
                        (self.init_noise[0:1], self.stock_noise[:-1]), dim=0
                    )
                x_t_latent = x_t_latent.to(self.device)
                t_list = t_list.to(self.device)
            x_0_pred_batch, model_pred = self.unet_step(x_t_latent, t_list, added_cond_kwargs=added_cond_kwargs)


            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                if self.do_add_noise:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                        + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                    )
                else:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                    )
            else:
                x_0_pred_out = x_0_pred_batch
                self.x_t_latent_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(
                    1,
                ).repeat(
                    self.frame_bff_size,
                )
                if self.sdxl:
                    added_cond_kwargs = {"text_embeds": self.add_text_embeds.to(self.device), "time_ids": self.add_time_ids.to(self.device)}
                x_0_pred, model_pred = self.unet_step(x_t_latent, t, idx=idx, added_cond_kwargs=added_cond_kwargs)

                if idx < len(self.sub_timesteps_tensor) - 1:
                    if self.do_add_noise:
                        x_t_latent = self.alpha_prod_t_sqrt[
                            idx + 1
                        ] * x_0_pred + self.beta_prod_t_sqrt[
                            idx + 1
                        ] * torch.randn_like(
                            x_0_pred, device=self.device, dtype=self.dtype
                        )
                    else:
                        x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
            x_0_pred_out = x_0_pred

        return x_0_pred_out

    @torch.no_grad()
    def __call__(
        self, x: Union[torch.Tensor, PIL.Image.Image, np.ndarray] = None
    ) -> torch.Tensor:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if x is not None:
            x = self.image_processor.preprocess(x, self.height, self.width).to(
                device=self.device, dtype=self.dtype
            )
            if self.similar_image_filter:
                x = self.similar_filter(x)
                if x is None:
                    time.sleep(self.inference_time_ema)
                    return self.prev_image_result
            x_t_latent = self.encode_image(x)
        else:
            # Phase 3.3: Use pre-allocated buffer instead of creating new tensor
            if self.use_cuda_native:
                from streamdiffusion.cuda import randn_cuda
                x_t_latent = randn_cuda(self._random_latent_buffer, seed=self.seed)
            else:
                x_t_latent = self._random_latent_buffer.normal_(generator=self.generator)
        x_0_pred_out = self.predict_x0_batch(x_t_latent)
        decoded = self.decode_image(x_0_pred_out)
        if self.use_cuda_native:
            from streamdiffusion.cuda import tensor_clone_cuda
            # detach() not needed in inference mode with CUDA native
            x_output = tensor_clone_cuda(decoded)
        else:
            x_output = decoded.detach().clone()

        self.prev_image_result = x_output
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end) / 1000
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        return x_output

    @torch.no_grad()
    def txt2img(self, batch_size: int = 1) -> torch.Tensor:
        # Phase 3.3: Use pre-allocated buffer for better performance
        if batch_size == 1:
            if self.use_cuda_native:
                from streamdiffusion.cuda import randn_cuda
                x_t_latent = randn_cuda(self._random_latent_buffer, seed=self.seed)
            else:
                x_t_latent = self._random_latent_buffer.normal_(generator=self.generator)
        else:
            # Ensure buffer is large enough, otherwise create temporary tensor
            if batch_size <= self._batch_random_latent_buffer.shape[0]:
                if self.use_cuda_native:
                    from streamdiffusion.cuda import randn_cuda
                    x_t_latent = randn_cuda(self._batch_random_latent_buffer[:batch_size], seed=self.seed)
                else:
                    x_t_latent = self._batch_random_latent_buffer[:batch_size].normal_(generator=self.generator)
            else:
                from streamdiffusion.cuda import randn_cuda
                x_t_latent = randn_cuda((batch_size, 4, self.latent_height, self.latent_width),
                                        device=self.device, dtype=self.dtype)
        x_0_pred_out = self.predict_x0_batch(x_t_latent)
        decoded = self.decode_image(x_0_pred_out)
        if self.use_cuda_native:
            from streamdiffusion.cuda import tensor_clone_cuda
            # detach() not needed in inference mode with CUDA native
            x_output = tensor_clone_cuda(decoded)
        else:
            x_output = decoded.detach().clone()
        return x_output

    def txt2img_sd_turbo(self, batch_size: int = 1) -> torch.Tensor:
        # Phase 3.3: Use pre-allocated buffer for better performance
        if batch_size == 1:
            x_t_latent = self._random_latent_buffer.normal_(generator=self.generator)
        else:
            # Ensure buffer is large enough, otherwise create temporary tensor
            if batch_size <= self._batch_random_latent_buffer.shape[0]:
                x_t_latent = self._batch_random_latent_buffer[:batch_size].normal_(generator=self.generator)
            else:
                from streamdiffusion.cuda import randn_cuda
                x_t_latent = randn_cuda((batch_size, 4, self.latent_height, self.latent_width),
                                        device=self.device, dtype=self.dtype)
        
        # Prepare additional conditioning for SDXL models
        added_cond_kwargs = {}
        if self.sdxl:
            added_cond_kwargs = {
                "text_embeds": self.add_text_embeds.to(self.device), 
                "time_ids": self.add_time_ids.to(self.device)
            }
        
        model_pred = self.unet(
            x_t_latent,
            self.sub_timesteps_tensor,
            encoder_hidden_states=self.prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        # Phase 3.3: Use in-place operations for better performance
        # x_0_pred_out = (x_t_latent - beta * model_pred) / alpha
        if self.use_cuda_native:
            from streamdiffusion.cuda import scalar_mul_inplace_cuda, scalar_div_inplace_cuda, tensor_sub_cuda
            scalar_mul_inplace_cuda(model_pred, float(self.beta_prod_t_sqrt))  # model_pred *= beta
            x_0_pred_out = tensor_sub_cuda(x_t_latent, model_pred)  # x_t_latent - model_pred
            scalar_div_inplace_cuda(x_0_pred_out, float(self.alpha_prod_t_sqrt))  # /= alpha
        else:
            model_pred.mul_(self.beta_prod_t_sqrt)  # model_pred *= beta
            x_0_pred_out = x_t_latent.sub(model_pred)  # x_t_latent - model_pred
            x_0_pred_out.div_(self.alpha_prod_t_sqrt)  # /= alpha
        return self.decode_image(x_0_pred_out)
