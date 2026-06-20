from typing import List, Optional, Dict, Tuple, Literal, Any, Callable
import threading
import torch
import torch.nn.functional as F
import gc

import logging
logger = logging.getLogger(__name__)
from .preprocessing.orchestrator_user import OrchestratorUser

class CacheStats:
    """Helper class to track cache statistics"""
    def __init__(self):
        self.hits = 0
        self.misses = 0

    def record_hit(self):
        self.hits += 1

    def record_miss(self):
        self.misses += 1


class StreamParameterUpdater(OrchestratorUser):
    def __init__(self, stream_diffusion, wrapper=None, normalize_prompt_weights: bool = True, normalize_seed_weights: bool = True):
        self.stream = stream_diffusion
        self.wrapper = wrapper  # Reference to wrapper for accessing pipeline structure
        self.normalize_prompt_weights = normalize_prompt_weights
        self.normalize_seed_weights = normalize_seed_weights
        # Atomic update lock for deterministic, thread-safe runtime updates
        self._update_lock = threading.RLock()
        # Prompt blending caches
        self._prompt_cache: Dict[int, Dict] = {}
        self._current_prompt_list: List[Tuple[str, float]] = []
        self._current_negative_prompt: str = ""
        self._prompt_cache_stats = CacheStats()

        # Seed blending caches
        self._seed_cache: Dict[int, Dict] = {}
        self._current_seed_list: List[Tuple[int, float]] = []
        self._seed_cache_stats = CacheStats()
        
        
        # Attach shared orchestrator once (lazy-creates on stream if absent)
        self.attach_orchestrator(self.stream)

        # IPAdapter embedding preprocessing
        self._embedding_preprocessors = []
        self._embedding_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._current_style_images: Dict[str, Any] = {}
        # Use the shared orchestrator attached via OrchestratorUser
        self._embedding_orchestrator = self._preprocessing_orchestrator
    def get_cache_info(self) -> Dict:
        """Get cache statistics for monitoring performance."""
        total_requests = self._prompt_cache_stats.hits + self._prompt_cache_stats.misses
        hit_rate = self._prompt_cache_stats.hits / total_requests if total_requests > 0 else 0

        total_seed_requests = self._seed_cache_stats.hits + self._seed_cache_stats.misses
        seed_hit_rate = self._seed_cache_stats.hits / total_seed_requests if total_seed_requests > 0 else 0

        return {
            "cached_prompts": len(self._prompt_cache),
            "cache_hits": self._prompt_cache_stats.hits,
            "cache_misses": self._prompt_cache_stats.misses,
            "hit_rate": f"{hit_rate:.2%}",
            "current_prompts": len(self._current_prompt_list),
            "cached_seeds": len(self._seed_cache),
            "seed_cache_hits": self._seed_cache_stats.hits,
            "seed_cache_misses": self._seed_cache_stats.misses,
            "seed_hit_rate": f"{seed_hit_rate:.2%}",
            "current_seeds": len(self._current_seed_list)
        }

    def clear_caches(self) -> None:
        """Clear all caches to free memory."""
        self._prompt_cache.clear()
        self._current_prompt_list.clear()
        self._current_negative_prompt = ""
        self._prompt_cache_stats = CacheStats()

        self._seed_cache.clear()
        self._current_seed_list.clear()
        self._seed_cache_stats = CacheStats()
        
        # Clear embedding caches
        self._embedding_cache.clear()
        self._current_style_images.clear()

    def get_normalize_prompt_weights(self) -> bool:
        """Get the current prompt weight normalization setting."""
        return self.normalize_prompt_weights

    def get_normalize_seed_weights(self) -> bool:
        """Get the current seed weight normalization setting."""
        return self.normalize_seed_weights
    
    # Deprecated enhancer registration removed; embedding composition is handled via stream.embedding_hooks

    def register_embedding_preprocessor(self, preprocessor: Any, style_image_key: str) -> None:
        """
        Register an embedding preprocessor for parallel processing.
        
        Args:
            preprocessor: IPAdapterEmbeddingPreprocessor instance
            style_image_key: Unique key for the style image this preprocessor handles
        """
        if self._embedding_orchestrator is None:
            # Ensure orchestrator is present
            self.attach_orchestrator(self.stream)
            self._embedding_orchestrator = self._preprocessing_orchestrator
        
        self._embedding_preprocessors.append((preprocessor, style_image_key))
    
    def unregister_embedding_preprocessor(self, style_image_key: str) -> None:
        """Unregister an embedding preprocessor by style image key."""
        original_count = len(self._embedding_preprocessors)
        self._embedding_preprocessors = [
            (preprocessor, key) for preprocessor, key in self._embedding_preprocessors 
            if key != style_image_key
        ]
        removed_count = original_count - len(self._embedding_preprocessors)
        
        # Clear cached embeddings for this key
        if style_image_key in self._embedding_cache:
            del self._embedding_cache[style_image_key]
        if style_image_key in self._current_style_images:
            del self._current_style_images[style_image_key]
    
    def update_style_image(self, style_image_key: str, style_image: Any, is_stream: bool = False) -> None:
        """
        Update a style image and trigger embedding preprocessing.
        
        Args:
            style_image_key: Unique key for the style image
            style_image: The style image (PIL Image, path, etc.)
            is_stream: If True, use pipelined processing (1-frame lag, high throughput)
                      If False, use synchronous processing (immediate results, lower throughput)
        """
        # Store the style image
        self._current_style_images[style_image_key] = style_image
        
        # Trigger preprocessing for this style image
        self._preprocess_style_image_parallel(style_image_key, style_image, is_stream)
    
    def _preprocess_style_image_parallel(self, style_image_key: str, style_image: Any, is_stream: bool = False) -> None:
        """
        Preprocessing for a specific style image with mode selection
        
        Args:
            style_image_key: Unique key for the style image
            style_image: The style image to process
            is_stream: If True, use pipelined processing; if False, use synchronous processing
        """
        if not self._embedding_preprocessors or self._embedding_orchestrator is None:
            return
        
        # Find preprocessors for this key
        relevant_preprocessors = [
            preprocessor for preprocessor, key in self._embedding_preprocessors 
            if key == style_image_key
        ]
        
        if not relevant_preprocessors:
            return
        
        # Choose processing mode based on is_stream parameter
        try:
            if is_stream:
                # Pipelined processing - optimized for throughput with 1-frame lag
                embedding_results = self._embedding_orchestrator.process_pipelined(
                    style_image,
                    relevant_preprocessors,
                    None,
                    self.stream.width,
                    self.stream.height,
                    "ipadapter"
                )
            else:
                # Synchronous processing - immediate results for discrete updates
                embedding_results = self._embedding_orchestrator.process_sync(
                    style_image,
                    relevant_preprocessors,
                    None,
                    self.stream.width,
                    self.stream.height,
                    None,
                    "ipadapter"
                )
            
            # Cache results for this style image key
            if embedding_results and embedding_results[0] is not None:
                self._embedding_cache[style_image_key] = embedding_results[0]
            else:
                # This is an error condition - we should always have results
                raise RuntimeError(f"_preprocess_style_image_parallel: Failed to generate embeddings for style image '{style_image_key}'")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    def get_cached_embeddings(self, style_image_key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached embeddings for a style image key"""
        cached_result = self._embedding_cache.get(style_image_key, None)
        return cached_result


    def _normalize_weights(self, weights: List[float], normalize: bool) -> torch.Tensor:
        """Generic weight normalization helper"""
        weights_tensor = torch.tensor(weights, device=self.stream.device, dtype=self.stream.dtype)
        if normalize:
            weights_tensor = weights_tensor / weights_tensor.sum()
        return weights_tensor

    def _validate_index(self, index: int, item_list: List, operation_name: str) -> bool:
        """Generic index validation helper"""
        if not item_list:
            logger.warning(f"{operation_name}: Warning: No current item list")
            return False

        if index < 0 or index >= len(item_list):
            logger.warning(f"{operation_name}: Warning: Index {index} out of range (0-{len(item_list)-1})")
            return False

        return True

    def _reindex_cache(self, cache: Dict[int, Dict], removed_index: int) -> Dict[int, Dict]:
        """Generic cache reindexing helper after item removal"""
        new_cache = {}
        for cache_idx, cache_data in cache.items():
            if cache_idx < removed_index:
                new_cache[cache_idx] = cache_data
            elif cache_idx > removed_index:
                new_cache[cache_idx - 1] = cache_data
        return new_cache

    @torch.no_grad()
    def update_stream_params(
        self,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        delta: Optional[float] = None,
        t_index_list: Optional[List[int]] = None,
        seed: Optional[int] = None,
        prompt_list: Optional[List[Tuple[str, float]]] = None,
        negative_prompt: Optional[str] = None,
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp",
        normalize_prompt_weights: Optional[bool] = None,
        seed_list: Optional[List[Tuple[int, float]]] = None,
        seed_interpolation_method: Literal["linear", "slerp"] = "linear",
        normalize_seed_weights: Optional[bool] = None,
        controlnet_config: Optional[List[Dict[str, Any]]] = None,
        ipadapter_config: Optional[Dict[str, Any]] = None,
        image_preprocessing_config: Optional[List[Dict[str, Any]]] = None,
        image_postprocessing_config: Optional[List[Dict[str, Any]]] = None,
        latent_preprocessing_config: Optional[List[Dict[str, Any]]] = None,
        latent_postprocessing_config: Optional[List[Dict[str, Any]]] = None,
        cache_maxframes: Optional[int] = None,
        cache_interval: Optional[int] = None,
    ) -> None:
        """Update streaming parameters efficiently in a single call."""

        with self._update_lock:
            # First, update num_inference_steps if needed (this changes the timesteps array size)
            if num_inference_steps is not None:
                # Safety check: Ensure num_inference_steps is at least as large as max t_index value
                if t_index_list is None:
                    # Check against current t_list
                    max_t_index = max(self.stream.t_list) if self.stream.t_list else 0
                    if num_inference_steps <= max_t_index:
                        logger.warning(
                            f"update_stream_params: num_inference_steps ({num_inference_steps}) is too small for "
                            f"current t_list (max index: {max_t_index}). Adjusting to {max_t_index + 1}."
                        )
                        num_inference_steps = max_t_index + 1
                else:
                    # Check against provided t_index_list
                    max_t_index = max(t_index_list) if t_index_list else 0
                    if num_inference_steps <= max_t_index:
                        logger.warning(
                            f"update_stream_params: num_inference_steps ({num_inference_steps}) is too small for "
                            f"provided t_index_list (max index: {max_t_index}). Adjusting to {max_t_index + 1}."
                        )
                        num_inference_steps = max_t_index + 1
                
                old_num_steps = len(self.stream.timesteps)
                self.stream.scheduler.set_timesteps(num_inference_steps, self.stream.device)
                self.stream.timesteps = self.stream.scheduler.timesteps.to(self.stream.device)
                
                # If t_index_list wasn't explicitly provided, rescale existing t_list proportionally
                if t_index_list is None and old_num_steps > 0:
                    # Rescale each index proportionally to the new number of steps
                    # e.g., if t_list = [0, 16, 32, 45] with 50 steps -> [0, 3, 6, 8] with 9 steps
                    scale_factor = (num_inference_steps - 1) / (old_num_steps - 1) if old_num_steps > 1 else 1.0
                    t_index_list = [
                        min(round(t * scale_factor), num_inference_steps - 1) 
                        for t in self.stream.t_list
                    ]
            
            # Now update timestep-dependent parameters with the correct t_index_list
            if t_index_list is not None:
                self._recalculate_timestep_dependent_params(t_index_list)

            if guidance_scale is not None:
                if self.stream.cfg_type == "none" and guidance_scale > 1.0:
                    logger.warning("update_stream_params: Warning: guidance_scale > 1.0 with cfg_type='none' will have no effect")
                self.stream.guidance_scale = guidance_scale

            if delta is not None:
                self.stream.delta = delta

            if seed is not None:
                self._update_seed(seed)
            
            if normalize_prompt_weights is not None:
                self.normalize_prompt_weights = normalize_prompt_weights
                logger.info(f"update_stream_params: Prompt weight normalization set to {normalize_prompt_weights}")

            if normalize_seed_weights is not None:
                self.normalize_seed_weights = normalize_seed_weights
                logger.info(f"update_stream_params: Seed weight normalization set to {normalize_seed_weights}")

            # Handle prompt blending if prompt_list is provided
            if prompt_list is not None:
                self._update_blended_prompts(
                    prompt_list=prompt_list,
                    negative_prompt=negative_prompt or self._current_negative_prompt,
                    prompt_interpolation_method=prompt_interpolation_method
                )

            # Handle seed blending if seed_list is provided
            if seed_list is not None:
                self._update_blended_seeds(
                    seed_list=seed_list,
                    interpolation_method=seed_interpolation_method
                )


            # Handle ControlNet configuration updates
            if controlnet_config is not None:
                #TODO: happy path for control images
                self._update_controlnet_config(controlnet_config)
            
            # Handle IPAdapter configuration updates
            if ipadapter_config is not None:
                logger.info(f"update_stream_params: Updating IPAdapter configuration")
                self._update_ipadapter_config(ipadapter_config)
            
            # Handle Hook configuration updates
            if image_preprocessing_config is not None:
                logger.info(f"update_stream_params: Updating image preprocessing configuration with {len(image_preprocessing_config)} processors")
                logger.info(f"update_stream_params: image_preprocessing_config = {image_preprocessing_config}")
                self._update_hook_config('image_preprocessing', image_preprocessing_config)
            
            if image_postprocessing_config is not None:
                logger.info(f"update_stream_params: Updating image postprocessing configuration")
                self._update_hook_config('image_postprocessing', image_postprocessing_config)
            
            if latent_preprocessing_config is not None:
                logger.info(f"update_stream_params: Updating latent preprocessing configuration")
                self._update_hook_config('latent_preprocessing', latent_preprocessing_config)
            
            if latent_postprocessing_config is not None:
                logger.info(f"update_stream_params: Updating latent postprocessing configuration")
                self._update_hook_config('latent_postprocessing', latent_postprocessing_config)

            if self.stream.kvo_cache:
                if cache_interval is not None:
                    self.stream.cache_interval = cache_interval
                    logger.info(f"update_stream_params: Cache interval set to {cache_interval}")

                if cache_maxframes is not None:
                    old_cache_maxframes = self.stream.cache_maxframes
                    self.stream.cache_maxframes = cache_maxframes
                    if old_cache_maxframes != cache_maxframes:
                        for i, cache_tensor in enumerate(self.stream.kvo_cache):
                            current_shape = cache_tensor.shape
                            new_shape = (current_shape[0], cache_maxframes, current_shape[2], current_shape[3], current_shape[4])
                            new_cache_tensor = torch.zeros(
                                new_shape,
                                dtype=cache_tensor.dtype,
                                device=cache_tensor.device
                            )
                    
                            if cache_maxframes > old_cache_maxframes:
                                new_cache_tensor[:, :old_cache_maxframes] = cache_tensor
                            else:
                                new_cache_tensor[:, :] = cache_tensor[:, -cache_maxframes:]
                            
                            self.stream.kvo_cache[i] = new_cache_tensor
                        logger.info(f"update_stream_params: Cache maxframes updated from {old_cache_maxframes} to {cache_maxframes}, kvo_cache tensors resized")
                    else:
                        logger.info(f"update_stream_params: Cache maxframes set to {cache_maxframes}")

    @torch.no_grad()
    def update_prompt_weights(
        self,
        prompt_weights: List[float],
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """Update weights for current prompt list without re-encoding prompts."""
        if not self._current_prompt_list:
            logger.warning("update_prompt_weights: Warning: No current prompt list to update weights for")
            return

        if len(prompt_weights) != len(self._current_prompt_list):
            logger.warning(f"update_prompt_weights: Warning: Weight count {len(prompt_weights)} doesn't match prompt count {len(self._current_prompt_list)}")
            return

        # Update the current prompt list with new weights
        updated_prompt_list = []
        for i, (prompt_text, _) in enumerate(self._current_prompt_list):
            updated_prompt_list.append((prompt_text, prompt_weights[i]))

        self._current_prompt_list = updated_prompt_list

        # Recompute blended embeddings with new weights
        self._apply_prompt_blending(prompt_interpolation_method)

    @torch.no_grad()
    def update_seed_weights(
        self,
        seed_weights: List[float],
        interpolation_method: Literal["linear", "slerp"] = "linear"
    ) -> None:
        """Update weights for current seed list without regenerating noise."""
        if not self._current_seed_list:
            logger.warning("update_seed_weights: Warning: No current seed list to update weights for")
            return

        if len(seed_weights) != len(self._current_seed_list):
            logger.warning(f"update_seed_weights: Warning: Weight count {len(seed_weights)} doesn't match seed count {len(self._current_seed_list)}")
            return

        # Update the current seed list with new weights
        updated_seed_list = []
        for i, (seed_value, _) in enumerate(self._current_seed_list):
            updated_seed_list.append((seed_value, seed_weights[i]))

        self._current_seed_list = updated_seed_list

        # Recompute blended noise with new weights
        self._apply_seed_blending(interpolation_method)

    @torch.no_grad()
    def _update_blended_prompts(
        self,
        prompt_list: List[Tuple[str, float]],
        negative_prompt: str = "",
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """Update prompt embeddings using multiple weighted prompts."""
        # Store current state
        self._current_prompt_list = prompt_list.copy()
        self._current_negative_prompt = negative_prompt

        # Encode any new prompts and cache them
        self._cache_prompt_embeddings(prompt_list, negative_prompt)

        # Apply blending
        self._apply_prompt_blending(prompt_interpolation_method)

    def _cache_prompt_embeddings(
        self,
        prompt_list: List[Tuple[str, float]],
        negative_prompt: str
    ) -> None:
        """Cache prompt embeddings for efficient reuse."""
        for idx, (prompt_text, weight) in enumerate(prompt_list):
            if idx not in self._prompt_cache or self._prompt_cache[idx]['text'] != prompt_text:
                # Cache miss - encode the prompt
                self._prompt_cache_stats.record_miss()
                encoder_output = self.stream.pipe.encode_prompt(
                    prompt=prompt_text,
                    device=self.stream.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=negative_prompt,
                )
                self._prompt_cache[idx] = {
                    'embed': encoder_output[0],
                    'text': prompt_text
                }
            else:
                # Cache hit
                self._prompt_cache_stats.record_hit()

    def _apply_prompt_blending(self, prompt_interpolation_method: Literal["linear", "slerp"]) -> None:
        """Apply weighted blending of cached prompt embeddings."""
        if not self._current_prompt_list:
            return

        embeddings = []
        weights = []

        for idx, (prompt_text, weight) in enumerate(self._current_prompt_list):
            if idx in self._prompt_cache:
                embeddings.append(self._prompt_cache[idx]['embed'])
                weights.append(weight)

        if not embeddings:
            logger.warning("_apply_prompt_blending: Warning: No cached embeddings found")
            return

        # Normalize weights
        weights = self._normalize_weights(weights, self.normalize_prompt_weights)

        # Apply interpolation
        if prompt_interpolation_method == "slerp" and len(embeddings) == 2:
            # Spherical linear interpolation for 2 prompts
            embed1, embed2 = embeddings[0], embeddings[1]
            t = weights[1].item()  # Use second weight as interpolation factor
            combined_embeds = self._slerp(embed1, embed2, t)
        else:
            # Linear interpolation (weighted average)
            combined_embeds = torch.zeros_like(embeddings[0])
            for embed, weight in zip(embeddings, weights):
                combined_embeds += weight * embed

        # Handle CFG properly - prompt_embeds = [uncond, cond] concatenated.
        # librediffusion fork fix: the per-mode repeat counts must match the classic prepare()
        # layout (and our known-good bundled pipeline), which the C++ engine and unet_step's
        # latent concat assume:
        #   - cond  : repeat(batch_size)                       (= denoising_steps_num * frame_bff)
        #   - uncond: repeat(frame_bff_size) for "initialize"  -> unet_step concats [x[0:1], x]
        #             repeat(batch_size)      for "full"        -> unet_step concats [x, x]
        # Upstream daydream used batch_size for BOTH and lumped initialize==full, producing the
        # wrong row count for "initialize" (e.g. 2 steps: 2+2=4 rows vs the 1+2=3 the latent has),
        # causing a UNet cross-attention size mismatch (encoder_hidden_states vs latent batch).
        if self.stream.cfg_type in ["full", "initialize"] and self.stream.guidance_scale > 1.0:
            uncond_repeats = self.stream.batch_size if self.stream.cfg_type == "full" else self.stream.frame_bff_size

            # Get the unconditional (negative) embeddings.
            # librediffusion fork fix: the previous call passed prompt="" with
            # do_classifier_free_guidance=False and took uncond_output[0]. With do_cfg=False,
            # diffusers encode_prompt IGNORES negative_prompt and returns only the positive
            # (here ""), so the negative prompt was SILENTLY DROPPED — the uncond row was always
            # the empty-string encode regardless of negative_prompt. (Caught by the C++ cross-impl
            # harness: withneg cells diverged, error amplified by guidance.) Encode the negative
            # prompt directly as the positive so uncond_embeds = encode(negative_prompt); this
            # honors the negative prompt and is identical to the old behavior when it is "".
            uncond_output = self.stream.pipe.encode_prompt(
                prompt=self._current_negative_prompt,
                device=self.stream.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            uncond_embeds = uncond_output[0].repeat(uncond_repeats, 1, 1)

            # Combine with conditional embeddings
            cond_embeds = combined_embeds.repeat(self.stream.batch_size, 1, 1)
            final_prompt_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)
            final_negative_embeds = None  # CFG mode combines everything into prompt_embeds
        else:
            # No CFG, just use the blended embeddings
            final_prompt_embeds = combined_embeds.repeat(self.stream.batch_size, 1, 1)
            final_negative_embeds = None  # Will be set by enhancers if needed
        
        # Enhancer mechanism removed in favor of embedding_hooks

        # Run embedding hooks to compose final embeddings (e.g., append IP-Adapter tokens)
        try:
            if hasattr(self.stream, 'embedding_hooks') and self.stream.embedding_hooks:
                from .hooks import EmbedsCtx  # local import to avoid cycles
                embeds_ctx = EmbedsCtx(
                    prompt_embeds=final_prompt_embeds,
                    negative_prompt_embeds=final_negative_embeds,
                )
                for hook in self.stream.embedding_hooks:
                    embeds_ctx = hook(embeds_ctx)
                final_prompt_embeds = embeds_ctx.prompt_embeds
                final_negative_embeds = embeds_ctx.negative_prompt_embeds
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"_apply_prompt_blending: embedding hook failed: {e}")
        
        # Set final embeddings on stream
        self.stream.prompt_embeds = final_prompt_embeds
        if final_negative_embeds is not None:
            self.stream.negative_prompt_embeds = final_negative_embeds

    def _slerp(self, embed1: torch.Tensor, embed2: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical linear interpolation between two embeddings."""
        # Handle case where t is 0 or 1
        if t <= 0:
            return embed1
        if t >= 1:
            return embed2

        # SLERP on flattened embeddings but preserve original shape
        original_shape = embed1.shape
        flat1 = embed1.view(-1)
        flat2 = embed2.view(-1)

        # Normalize
        flat1_norm = F.normalize(flat1, dim=0)
        flat2_norm = F.normalize(flat2, dim=0)

        # Calculate angle
        dot_product = torch.clamp(torch.dot(flat1_norm, flat2_norm), -1.0, 1.0)
        theta = torch.acos(dot_product)

        # Handle parallel vectors
        if theta.abs() < 1e-6:
            result = (1 - t) * flat1 + t * flat2
        else:
            # SLERP formula
            sin_theta = torch.sin(theta)
            w1 = torch.sin((1 - t) * theta) / sin_theta
            w2 = torch.sin(t * theta) / sin_theta
            result = w1 * flat1 + w2 * flat2

        return result.view(original_shape)

    @torch.no_grad()
    def _update_blended_seeds(
        self,
        seed_list: List[Tuple[int, float]],
        interpolation_method: Literal["linear", "slerp"] = "linear"
    ) -> None:
        """Update seed tensors using multiple weighted seeds."""
        # Store current state
        self._current_seed_list = seed_list.copy()

        # Cache any new seed noise tensors
        self._cache_seed_noise(seed_list)

        # Apply blending
        self._apply_seed_blending(interpolation_method)

    def _cache_seed_noise(self, seed_list: List[Tuple[int, float]]) -> None:
        """Cache seed noise tensors for efficient reuse."""
        for idx, (seed_value, weight) in enumerate(seed_list):
            if idx not in self._seed_cache or self._seed_cache[idx]['seed'] != seed_value:
                # Cache miss - generate noise for the seed
                self._seed_cache_stats.record_miss()
                generator = torch.Generator(device=self.stream.device)
                generator.manual_seed(seed_value)

                noise = torch.randn(
                    (self.stream.batch_size, 4, self.stream.latent_height, self.stream.latent_width),
                    generator=generator,
                    device=self.stream.device,
                    dtype=self.stream.dtype
                )

                self._seed_cache[idx] = {
                    'noise': noise,
                    'seed': seed_value
                }
            else:
                # Cache hit
                self._seed_cache_stats.record_hit()

    def _apply_seed_blending(self, interpolation_method: Literal["linear", "slerp"]) -> None:
        """Apply weighted blending of cached seed noise tensors."""
        if not self._current_seed_list:
            return

        noise_tensors = []
        weights = []

        for idx, (seed_value, weight) in enumerate(self._current_seed_list):
            if idx in self._seed_cache:
                noise_tensors.append(self._seed_cache[idx]['noise'])
                weights.append(weight)

        if not noise_tensors:
            logger.warning("_apply_seed_blending: Warning: No cached noise tensors found")
            return

        # Normalize weights
        weights = self._normalize_weights(weights, self.normalize_seed_weights)

        # Apply interpolation
        if interpolation_method == "slerp" and len(noise_tensors) == 2:
            # Spherical linear interpolation for 2 seeds
            noise1, noise2 = noise_tensors[0], noise_tensors[1]
            t = weights[1].item()  # Use second weight as interpolation factor
            combined_noise = self._slerp_noise(noise1, noise2, t)
        else:
            # Linear interpolation (weighted average)
            combined_noise = torch.zeros_like(noise_tensors[0])
            for noise, weight in zip(noise_tensors, weights):
                combined_noise += weight * noise
            
            # Preserve noise magnitude when weights are normalized
            if self.normalize_seed_weights and len(noise_tensors) > 1:
                original_magnitude = torch.mean(torch.stack([torch.norm(noise) for noise in noise_tensors]))
                current_magnitude = torch.norm(combined_noise)
                if current_magnitude > 1e-8:  # Avoid division by zero
                    combined_noise = combined_noise * (original_magnitude / current_magnitude)

        # Update stream noise
        self.stream.init_noise = combined_noise
        self.stream.stock_noise = torch.zeros_like(self.stream.init_noise)

    def _slerp_noise(self, noise1: torch.Tensor, noise2: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical linear interpolation between two noise tensors."""
        # Handle case where t is 0 or 1
        if t <= 0:
            return noise1
        if t >= 1:
            return noise2

        # SLERP on flattened noise but preserve original shape
        original_shape = noise1.shape
        flat1 = noise1.view(-1)
        flat2 = noise2.view(-1)

        # Normalize
        flat1_norm = F.normalize(flat1, dim=0)
        flat2_norm = F.normalize(flat2, dim=0)

        # Calculate angle
        dot_product = torch.clamp(torch.dot(flat1_norm, flat2_norm), -1.0, 1.0)
        theta = torch.acos(dot_product)

        # Handle parallel vectors
        if theta.abs() < 1e-6:
            result = (1 - t) * flat1 + t * flat2
        else:
            # SLERP formula
            sin_theta = torch.sin(theta)
            w1 = torch.sin((1 - t) * theta) / sin_theta
            w2 = torch.sin(t * theta) / sin_theta
            result = w1 * flat1 + w2 * flat2

        return result.view(original_shape)

    def _update_seed(self, seed: int) -> None:
        """Update the generator seed and regenerate seed-dependent tensors."""
        if self.stream.generator is None:
            logger.warning("update_stream_params: Warning: generator is None, cannot update seed")
            return

        # Store the current seed value
        self.stream.current_seed = seed

        # Update generator seed
        self.stream.generator.manual_seed(seed)

        # Regenerate init_noise tensor with new seed
        self.stream.init_noise = torch.randn(
            (self.stream.batch_size, 4, self.stream.latent_height, self.stream.latent_width),
            generator=self.stream.generator,
        ).to(device=self.stream.device, dtype=self.stream.dtype)

        # Reset stock_noise to match the new init_noise
        self.stream.stock_noise = torch.zeros_like(self.stream.init_noise)

    def _get_scheduler_scalings(self, timestep):
        """Get LCM/TCD-specific scaling factors for boundary conditions."""
        from diffusers import LCMScheduler
        if isinstance(self.stream.scheduler, LCMScheduler):
            c_skip, c_out = self.stream.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
            return c_skip, c_out
        else:
            # TCD and other schedulers don't use boundary condition scaling like LCM
            # They handle scaling internally in their step() method
            # Return tensors that are compatible with torch.stack()
            c_skip = torch.tensor(1.0, device=self.stream.device, dtype=self.stream.dtype)
            c_out = torch.tensor(1.0, device=self.stream.device, dtype=self.stream.dtype)
            return c_skip, c_out

    def _update_timestep_calculations(self) -> None:
        """Update timestep-dependent calculations based on current t_list."""
        self.stream.sub_timesteps = []
        for t in self.stream.t_list:
            self.stream.sub_timesteps.append(self.stream.timesteps[t])

        sub_timesteps_tensor = torch.tensor(
            self.stream.sub_timesteps, dtype=torch.long, device=self.stream.device
        )
        self.stream.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.stream.frame_bff_size if self.stream.use_denoising_batch else 1,
            dim=0,
        )

        c_skip_list = []
        c_out_list = []
        for timestep in self.stream.sub_timesteps:
            c_skip, c_out = self._get_scheduler_scalings(timestep)
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.stream.c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.stream.t_list), 1, 1, 1)
            .to(dtype=self.stream.dtype, device=self.stream.device)
        )
        self.stream.c_out = (
            torch.stack(c_out_list)
            .view(len(self.stream.t_list), 1, 1, 1)
            .to(dtype=self.stream.dtype, device=self.stream.device)
        )

        if self.stream.use_denoising_batch:
            self.stream.c_skip = torch.repeat_interleave(
                self.stream.c_skip, repeats=self.stream.frame_bff_size, dim=0
            )
            self.stream.c_out = torch.repeat_interleave(
                self.stream.c_out, repeats=self.stream.frame_bff_size, dim=0
            )

        # Update alpha_prod_t_sqrt and beta_prod_t_sqrt
        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.stream.sub_timesteps:
            alpha_prod_t_sqrt = self.stream.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.stream.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)

        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.stream.t_list), 1, 1, 1)
            .to(dtype=self.stream.dtype, device=self.stream.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.stream.t_list), 1, 1, 1)
            .to(dtype=self.stream.dtype, device=self.stream.device)
        )
        self.stream.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.stream.frame_bff_size if self.stream.use_denoising_batch else 1,
            dim=0,
        )
        self.stream.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.stream.frame_bff_size if self.stream.use_denoising_batch else 1,
            dim=0,
        )

    def _update_timestep_values_only(self, t_index_list: List[int]) -> None:
        """Update only timestep-dependent values when t_index_list values change but length stays same.
        This preserves the working branch behavior for value-only changes."""
        self.stream.t_list = t_index_list
        self._update_timestep_calculations()

    def _recalculate_timestep_dependent_params(self, t_index_list: List[int]) -> None:
        """Recalculate all parameters that depend on t_index_list."""
        
        # Check if this is a structural change (length) or just value change
        if len(t_index_list) == len(self.stream.t_list):
            # Same length - only values changed, use lightweight update (working branch behavior)
            self._update_timestep_values_only(t_index_list)
            return
        
        # Length changed - do full recalculation including batch-dependent parameters (broken branch logic - but it works for this case!)
        self.stream.t_list = t_index_list
        self.stream.denoising_steps_num = len(self.stream.t_list)

        old_batch_size = self.stream.batch_size
        
        if self.stream.use_denoising_batch:
            self.stream.batch_size = self.stream.denoising_steps_num * self.stream.frame_bff_size
            if self.stream.cfg_type == "initialize":
                self.stream.trt_unet_batch_size = (
                    self.stream.denoising_steps_num + 1
                ) * self.stream.frame_bff_size
            elif self.stream.cfg_type == "full":
                self.stream.trt_unet_batch_size = (
                    2 * self.stream.denoising_steps_num * self.stream.frame_bff_size
                )
            else:
                self.stream.trt_unet_batch_size = self.stream.denoising_steps_num * self.stream.frame_bff_size
        else:
            self.stream.trt_unet_batch_size = self.stream.frame_bff_size
            self.stream.batch_size = self.stream.frame_bff_size

        if self.stream.denoising_steps_num > 1:
            self.stream.x_t_latent_buffer = torch.zeros(
                (
                    (self.stream.denoising_steps_num - 1) * self.stream.frame_bff_size,
                    4,
                    self.stream.latent_height,
                    self.stream.latent_width,
                ),
                dtype=self.stream.dtype,
                device=self.stream.device,
            )
        else:
            self.stream.x_t_latent_buffer = None

        self.stream.init_noise = torch.randn(
            (self.stream.batch_size, 4, self.stream.latent_height, self.stream.latent_width),
            generator=self.stream.generator,
        ).to(device=self.stream.device, dtype=self.stream.dtype)

        self.stream.stock_noise = torch.zeros_like(self.stream.init_noise)
        self.stream.prompt_embeds = self.stream.prompt_embeds[0].repeat(self.stream.batch_size, 1, 1)

        # Resize kvo_cache tensors if batch size changed
        if self.stream.kvo_cache and old_batch_size != self.stream.batch_size:
            logger.info(f"_recalculate_timestep_dependent_params: Resizing kvo_cache tensors from batch_size {old_batch_size} to {self.stream.batch_size}")
            for i, cache_tensor in enumerate(self.stream.kvo_cache):
                # KVO cache shape: (2, cache_maxframes, batch_size, seq_length, hidden_dim)
                current_shape = cache_tensor.shape
                new_shape = (current_shape[0], current_shape[1], self.stream.batch_size, current_shape[3], current_shape[4])
                new_cache_tensor = torch.zeros(
                    new_shape,
                    dtype=cache_tensor.dtype,
                    device=cache_tensor.device
                )
                
                # Copy over as much data as possible from old cache
                min_batch = min(old_batch_size, self.stream.batch_size)
                new_cache_tensor[:, :, :min_batch, :, :] = cache_tensor[:, :, :min_batch, :, :]
                
                self.stream.kvo_cache[i] = new_cache_tensor
            logger.info(f"_recalculate_timestep_dependent_params: KVO cache tensors resized to new batch_size {self.stream.batch_size}")

        # Update timestep-dependent calculations (shared with value-only path)
        self._update_timestep_calculations()

    def _regenerate_resolution_tensors(self) -> None:
        """This method is no longer used - resolution updates now restart the pipeline"""
        pass

    def _update_controlnet_inputs(self, width: int, height: int) -> None:
        """This method is no longer used - resolution updates now restart the pipeline"""
        pass

    def _recalculate_controlnet_inputs(self, width: int, height: int) -> None:
        """This method is no longer used - resolution updates now restart the pipeline"""
        pass

    @torch.no_grad()
    def update_prompt_at_index(
        self,
        index: int,
        new_prompt: str,
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """Update a single prompt at the specified index without re-encoding others."""
        if not self._validate_index(index, self._current_prompt_list, "update_prompt_at_index"):
            return

        # Update the prompt text while keeping the weight
        old_prompt, weight = self._current_prompt_list[index]
        self._current_prompt_list[index] = (new_prompt, weight)

        # Cache the new prompt embedding
        self._cache_prompt_embeddings([(new_prompt, weight)], self._current_negative_prompt)

        # Update cache index to point to the new prompt
        if index in self._prompt_cache and self._prompt_cache[index]['text'] != new_prompt:
            # Find if this prompt is already cached elsewhere
            existing_cache_key = None
            for cache_idx, cache_data in self._prompt_cache.items():
                if cache_data['text'] == new_prompt:
                    existing_cache_key = cache_idx
                    break

            if existing_cache_key is not None:
                # Reuse existing cached embedding
                self._prompt_cache[index] = self._prompt_cache[existing_cache_key].copy()
                self._prompt_cache_stats.record_hit()
            else:
                # Encode new prompt
                self._prompt_cache_stats.record_miss()
                encoder_output = self.stream.pipe.encode_prompt(
                    prompt=new_prompt,
                    device=self.stream.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=self._current_negative_prompt,
                )
                self._prompt_cache[index] = {
                    'embed': encoder_output[0],
                    'text': new_prompt
                }

        # Recompute blended embeddings with updated prompt
        self._apply_prompt_blending(prompt_interpolation_method)

    @torch.no_grad()
    def get_current_prompts(self) -> List[Tuple[str, float]]:
        """Get the current prompt list with weights."""
        return self._current_prompt_list.copy()

    @torch.no_grad()
    def add_prompt(
        self,
        prompt: str,
        weight: float = 1.0,
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """Add a new prompt to the current list."""
        new_index = len(self._current_prompt_list)
        self._current_prompt_list.append((prompt, weight))


        # Cache the new prompt
        encoder_output = self.stream.pipe.encode_prompt(
            prompt=prompt,
            device=self.stream.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=self._current_negative_prompt,
        )
        self._prompt_cache[new_index] = {
            'embed': encoder_output[0],
            'text': prompt
        }
        self._prompt_cache_stats.record_miss()

        # Recompute blended embeddings
        self._apply_prompt_blending(prompt_interpolation_method)

    @torch.no_grad()
    def remove_prompt_at_index(
        self,
        index: int,
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """Remove a prompt at the specified index."""
        if not self._validate_index(index, self._current_prompt_list, "remove_prompt_at_index"):
            return

        if len(self._current_prompt_list) <= 1:
            logger.warning("remove_prompt_at_index: Warning: Cannot remove last prompt")
            return

        # Remove from current list
        removed_prompt = self._current_prompt_list.pop(index)

        # Remove from cache and reindex
        if index in self._prompt_cache:
            del self._prompt_cache[index]

        # Shift cache indices down
        self._prompt_cache = self._reindex_cache(self._prompt_cache, index)

        # Recompute blended embeddings
        self._apply_prompt_blending(prompt_interpolation_method)

    @torch.no_grad()
    def update_seed_at_index(
        self,
        index: int,
        new_seed: int,
        interpolation_method: Literal["linear", "slerp"] = "linear"
    ) -> None:
        """Update a single seed at the specified index without regenerating others."""
        if not self._validate_index(index, self._current_seed_list, "update_seed_at_index"):
            return

        # Update the seed value while keeping the weight
        old_seed, weight = self._current_seed_list[index]
        self._current_seed_list[index] = (new_seed, weight)


        # Cache the new seed noise
        self._cache_seed_noise([(new_seed, weight)])

        # Update cache index to point to the new seed
        if index in self._seed_cache and self._seed_cache[index]['seed'] != new_seed:
            # Find if this seed is already cached elsewhere
            existing_cache_key = None
            for cache_idx, cache_data in self._seed_cache.items():
                if cache_data['seed'] == new_seed:
                    existing_cache_key = cache_idx
                    break

            if existing_cache_key is not None:
                # Reuse existing cached noise
                self._seed_cache[index] = self._seed_cache[existing_cache_key].copy()
                self._seed_cache_stats.record_hit()
            else:
                # Generate new noise
                self._seed_cache_stats.record_miss()
                generator = torch.Generator(device=self.stream.device)
                generator.manual_seed(new_seed)

                noise = torch.randn(
                    (self.stream.batch_size, 4, self.stream.latent_height, self.stream.latent_width),
                    generator=generator,
                    device=self.stream.device,
                    dtype=self.stream.dtype
                )

                self._seed_cache[index] = {
                    'noise': noise,
                    'seed': new_seed
                }

        # Recompute blended noise with updated seed
        self._apply_seed_blending(interpolation_method)

    @torch.no_grad()
    def get_current_seeds(self) -> List[Tuple[int, float]]:
        """Get the current seed list with weights."""
        return self._current_seed_list.copy()

    @torch.no_grad()
    def add_seed(
        self,
        seed: int,
        weight: float = 1.0,
        interpolation_method: Literal["linear", "slerp"] = "linear"
    ) -> None:
        """Add a new seed to the current list."""
        new_index = len(self._current_seed_list)
        self._current_seed_list.append((seed, weight))

        logger.info(f"add_seed: Added seed {new_index}: {seed} with weight {weight}")

        # Cache the new seed noise
        generator = torch.Generator(device=self.stream.device)
        generator.manual_seed(seed)

        noise = torch.randn(
            (self.stream.batch_size, 4, self.stream.latent_height, self.stream.latent_width),
            generator=generator,
            device=self.stream.device,
            dtype=self.stream.dtype
        )

        self._seed_cache[new_index] = {
            'noise': noise,
            'seed': seed
        }
        self._seed_cache_stats.record_miss()

        # Recompute blended noise
        self._apply_seed_blending(interpolation_method)

    @torch.no_grad()
    def remove_seed_at_index(
        self,
        index: int,
        interpolation_method: Literal["linear", "slerp"] = "linear"
    ) -> None:
        """Remove a seed at the specified index."""
        if not self._validate_index(index, self._current_seed_list, "remove_seed_at_index"):
            return

        if len(self._current_seed_list) <= 1:
            logger.warning("remove_seed_at_index: Warning: Cannot remove last seed")
            return

        # Remove from current list
        removed_seed = self._current_seed_list.pop(index)

        # Remove from cache and reindex
        if index in self._seed_cache:
            del self._seed_cache[index]

        # Shift cache indices down
        self._seed_cache = self._reindex_cache(self._seed_cache, index)

        # Recompute blended noise
        self._apply_seed_blending(interpolation_method)

    def _update_controlnet_config(self, desired_config: List[Dict[str, Any]]) -> None:
        """
        Update ControlNet configuration by diffing current vs desired state.
        
        Args:
            desired_config: Complete ControlNet configuration list defining the desired state.
                           Each dict contains: model_id, preprocessor, conditioning_scale, enabled, etc.
        """
        # Find the ControlNet pipeline/module (module-aware)
        controlnet_pipeline = self._get_controlnet_pipeline()
        if not controlnet_pipeline:
            logger.warning(f"_update_controlnet_config: No ControlNet pipeline found")
            return
        
        current_config = self._get_current_controlnet_config()
        
        # Simple approach: detect what changed and apply minimal updates
        current_models = {i: getattr(cn, 'model_id', f'controlnet_{i}') for i, cn in enumerate(controlnet_pipeline.controlnets)}
        desired_models = {cfg['model_id']: cfg for cfg in desired_config}
        
        # Reorder to match desired order (module supports stable reordering)
        try:
            desired_order = [cfg['model_id'] for cfg in desired_config if 'model_id' in cfg]
            if hasattr(controlnet_pipeline, 'reorder_controlnets_by_model_ids'):
                controlnet_pipeline.reorder_controlnets_by_model_ids(desired_order)
        except Exception:
            pass

        # Recompute current models after potential reorder
        current_models = {i: getattr(cn, 'model_id', f'controlnet_{i}') for i, cn in enumerate(controlnet_pipeline.controlnets)}

        # Remove controlnets not in desired config
        for i in reversed(range(len(controlnet_pipeline.controlnets))):
            model_id = current_models.get(i, f'controlnet_{i}')
            if model_id not in desired_models:
                logger.info(f"_update_controlnet_config: Removing ControlNet {model_id}")
                try:
                    controlnet_pipeline.remove_controlnet(i)
                except Exception:
                    raise
        
        # Add new controlnets and update existing ones
        for desired_cfg in desired_config:
            model_id = desired_cfg['model_id']
            existing_index = next((i for i, mid in current_models.items() if mid == model_id), None)
            
            if existing_index is None:
                # Add new controlnet
                logger.info(f"_update_controlnet_config: Adding ControlNet {model_id}")
                try:
                    # Prefer module path: construct ControlNetConfig
                    try:
                        from .modules.controlnet_module import ControlNetConfig  # type: ignore
                        cn_cfg = ControlNetConfig(
                            model_id=desired_cfg.get('model_id'),
                            preprocessor=desired_cfg.get('preprocessor'),
                            conditioning_scale=desired_cfg.get('conditioning_scale', 1.0),
                            enabled=desired_cfg.get('enabled', True),
                            conditioning_channels=desired_cfg.get('conditioning_channels'),
                            preprocessor_params=desired_cfg.get('preprocessor_params'),
                        )
                        controlnet_pipeline.add_controlnet(cn_cfg, desired_cfg.get('control_image'))
                    except Exception:
                        # No fallback
                        raise
                except Exception as e:
                    logger.error(f"_update_controlnet_config: add_controlnet failed for {model_id}: {e}")
            else:
                # Update existing controlnet
                if 'conditioning_scale' in desired_cfg:
                    current_scale = current_config[existing_index].get('conditioning_scale', 1.0)
                    desired_scale = desired_cfg['conditioning_scale']
                    
                    if current_scale != desired_scale:
                        logger.info(f"_update_controlnet_config: Updating {model_id} scale: {current_scale} → {desired_scale}")
                        if hasattr(controlnet_pipeline, 'controlnet_scales') and 0 <= existing_index < len(controlnet_pipeline.controlnet_scales):
                            controlnet_pipeline.controlnet_scales[existing_index] = float(desired_scale)
                
                # Enable/disable toggle
                if 'enabled' in desired_cfg and hasattr(controlnet_pipeline, 'enabled_list'):
                    if 0 <= existing_index < len(controlnet_pipeline.enabled_list):
                        controlnet_pipeline.enabled_list[existing_index] = bool(desired_cfg['enabled'])

                if 'preprocessor_params' in desired_cfg and hasattr(controlnet_pipeline, 'preprocessors') and controlnet_pipeline.preprocessors[existing_index]:
                    preprocessor = controlnet_pipeline.preprocessors[existing_index]
                    preprocessor.params.update(desired_cfg['preprocessor_params'])
                    for param_name, param_value in desired_cfg['preprocessor_params'].items():
                        if hasattr(preprocessor, param_name):
                            setattr(preprocessor, param_name, param_value)
                
                # Pipeline references are now automatically managed during preprocessor creation
                # No need to manually re-establish pipeline references for pipeline-aware processors


    def _get_controlnet_pipeline(self):
        """
        Get the ControlNet module or legacy pipeline from the structure (module-aware).
        """
        # Module-installed path
        if hasattr(self.stream, '_controlnet_module'):
            return self.stream._controlnet_module
        # Legacy paths
        if hasattr(self.stream, 'controlnets'):
            return self.stream
        if hasattr(self.stream, 'stream') and hasattr(self.stream.stream, 'controlnets'):
            return self.stream.stream
        if self.wrapper and hasattr(self.wrapper, 'stream'):
            if hasattr(self.wrapper.stream, '_controlnet_module'):
                return self.wrapper.stream._controlnet_module
            if hasattr(self.wrapper.stream, 'controlnets'):
                return self.wrapper.stream
            if hasattr(self.wrapper.stream, 'stream') and hasattr(self.wrapper.stream.stream, 'controlnets'):
                return self.wrapper.stream.stream
        return None

    def _get_current_controlnet_config(self) -> List[Dict[str, Any]]:
        """
        Get current ControlNet configuration state.
        
        Returns:
            List of current ControlNet configurations
        """
        controlnet_pipeline = self._get_controlnet_pipeline()
        if not controlnet_pipeline or not hasattr(controlnet_pipeline, 'controlnets') or not controlnet_pipeline.controlnets:
            return []
        
        current_config = []
        for i, controlnet in enumerate(controlnet_pipeline.controlnets):
            model_id = getattr(controlnet, 'model_id', f'controlnet_{i}')
            scale = controlnet_pipeline.controlnet_scales[i] if hasattr(controlnet_pipeline, 'controlnet_scales') and i < len(controlnet_pipeline.controlnet_scales) else 1.0
            enabled_val = True
            try:
                if hasattr(controlnet_pipeline, 'enabled_list') and i < len(controlnet_pipeline.enabled_list):
                    enabled_val = bool(controlnet_pipeline.enabled_list[i])
            except Exception:
                enabled_val = True
            config = {
                'model_id': model_id,
                'conditioning_scale': scale,
                'preprocessor_params': getattr(controlnet_pipeline.preprocessors[i], 'params', {}) if hasattr(controlnet_pipeline, 'preprocessors') and controlnet_pipeline.preprocessors[i] else {},
                'enabled': enabled_val,
            }
            current_config.append(config)
        
        return current_config

    def _update_ipadapter_config(self, desired_config: Dict[str, Any]) -> None:
        """
        Update IPAdapter configuration.
        
        Args:
            desired_config: IPAdapter configuration dict containing: 
                           ipadapter_model_path, image_encoder_path, style_image, scale, enabled, etc.
        """
        # Find the IPAdapter pipeline
        ipadapter_pipeline = self._get_ipadapter_pipeline()
        
        if not ipadapter_pipeline:
            logger.warning(f"_update_ipadapter_config: No IPAdapter pipeline found")
            return
        
        if 'scale' in desired_config and desired_config['scale'] is not None:
            desired_scale = float(desired_config['scale'])
            # Get current scale from IPAdapter instance
            current_scale = getattr(self.stream.ipadapter, 'scale', 1.0) if hasattr(self.stream, 'ipadapter') else 1.0
            
            if current_scale != desired_scale:
                logger.info(f"_update_ipadapter_config: Updating scale: {current_scale} → {desired_scale}")
                
                # Get weight_type from IPAdapter instance
                weight_type = getattr(self.stream.ipadapter, 'weight_type', None) if hasattr(self.stream, 'ipadapter') else None
                
                # Apply scale with weight type consideration
                if weight_type is not None and hasattr(self.stream, 'ipadapter'):
                    try:
                        from diffusers_ipadapter.ip_adapter.attention_processor import build_layer_weights
                        ip_procs = [p for p in self.stream.pipe.unet.attn_processors.values() if hasattr(p, "_ip_layer_index")]
                        num_layers = len(ip_procs)
                        weights = build_layer_weights(num_layers, desired_scale, weight_type)
                        if weights is not None:
                            self.stream.ipadapter.set_scale(weights)
                        else:
                            self.stream.ipadapter.set_scale(desired_scale)
                        # Update our tracking attribute
                        setattr(self.stream.ipadapter, 'scale', desired_scale)
                    except Exception:
                        # Do not add fallback mechanisms
                        raise
                else:
                    # Simple uniform scale
                    if hasattr(self.stream, 'ipadapter'):
                        # Tell diffusers_ipadapter to set the scale
                        self.stream.ipadapter.set_scale(desired_scale)
                        # Update our tracking attribute
                        setattr(self.stream.ipadapter, 'scale', desired_scale)
        

        # Update enabled state if provided
        if 'enabled' in desired_config and desired_config['enabled'] is not None:
            enabled_state = bool(desired_config['enabled'])
            # Update IPAdapter instance
            if hasattr(self.stream, 'ipadapter'):
                current_enabled = getattr(self.stream.ipadapter, 'enabled', True)
                if current_enabled != enabled_state:
                    logger.info(f"_update_ipadapter_config: Updating enabled state: {current_enabled} → {enabled_state}")
                    setattr(self.stream.ipadapter, 'enabled', enabled_state)

        # Update weight type if provided (affects per-layer distribution and/or per-step factor)
        if 'weight_type' in desired_config and desired_config['weight_type'] is not None:
            weight_type = desired_config['weight_type']
            # Update IPAdapter instance
            if hasattr(self.stream, 'ipadapter'):
                setattr(self.stream.ipadapter, 'weight_type', weight_type)
                
                # For PyTorch UNet, immediately apply a per-layer scale vector so layers reflect selection types
                try:
                    is_tensorrt_engine = hasattr(self.stream.unet, 'engine') and hasattr(self.stream.unet, 'stream')
                    if not is_tensorrt_engine:
                        # Compute per-layer vector using Diffusers_IPAdapter helper
                        from diffusers_ipadapter.ip_adapter.attention_processor import build_layer_weights
                        # Count installed IP layers by scanning processors with _ip_layer_index
                        ip_procs = [p for p in self.stream.pipe.unet.attn_processors.values() if hasattr(p, "_ip_layer_index")]
                        num_layers = len(ip_procs)
                        # Get base weight from IPAdapter instance
                        base_weight = float(getattr(self.stream.ipadapter, 'scale', 1.0))
                        weights = build_layer_weights(num_layers, base_weight, weight_type)
                        # If None, keep uniform base scale; else set per-layer vector
                        if weights is not None:
                            self.stream.ipadapter.set_scale(weights)
                        else:
                            self.stream.ipadapter.set_scale(base_weight)
                        # Keep our tracking attribute in sync
                        setattr(self.stream.ipadapter, 'scale', base_weight)
                except Exception:
                    # Do not add fallback mechanisms
                    raise

    def _get_ipadapter_pipeline(self):
        """
        Get the IPAdapter pipeline from the pipeline structure (following ControlNet pattern).
        
        Returns:
            IPAdapter pipeline object or None if not found
        """
        # Check if stream is IPAdapter pipeline directly
        if hasattr(self.stream, 'ipadapter'):
            return self.stream
            
        # Check if stream has nested stream (ControlNet wrapper)
        if hasattr(self.stream, 'stream') and hasattr(self.stream.stream, 'ipadapter'):
            return self.stream.stream
        
        # Check if we have a wrapper reference and can access through it
        if self.wrapper and hasattr(self.wrapper, 'stream'):
            if hasattr(self.wrapper.stream, 'ipadapter'):
                return self.wrapper.stream
            elif hasattr(self.wrapper.stream, 'stream') and hasattr(self.wrapper.stream.stream, 'ipadapter'):
                return self.wrapper.stream.stream
        
        return None

    def _get_current_ipadapter_config(self) -> Optional[Dict[str, Any]]:
        """
        Get current IPAdapter configuration by introspecting the IPAdapter instance.
        
        Returns:
            Current IPAdapter configuration dict or None if no IPAdapter
        """
        # Get config from IPAdapter instance
        if hasattr(self.stream, 'ipadapter') and self.stream.ipadapter is not None:
            ipadapter = self.stream.ipadapter
            
            config = {
                'scale': getattr(ipadapter, 'scale', 1.0),
                'weight_type': getattr(ipadapter, 'weight_type', None),
                'enabled': getattr(ipadapter, 'enabled', True),  # Check actual enabled state
            }
            
            # Add static initialization fields
            if hasattr(self.stream, '_ipadapter_module'):
                module_config = self.stream._ipadapter_module.config
                config.update({
                    'style_image_key': module_config.style_image_key,
                    'num_image_tokens': module_config.num_image_tokens,
                    'type': module_config.type.value,
                })
            
            # Check if style image is set
            ipadapter_pipeline = self._get_ipadapter_pipeline()
            if ipadapter_pipeline and hasattr(ipadapter_pipeline, 'style_image') and ipadapter_pipeline.style_image:
                config['has_style_image'] = True
            else:
                config['has_style_image'] = False
            
            return config
        
        # No IPAdapter instance found
        return None

    def _get_current_hook_config(self, hook_type: str) -> List[Dict[str, Any]]:
        """
        Get current hook configuration by introspecting the hook module state.
        
        Args:
            hook_type: Type of hook (image_preprocessing, image_postprocessing, etc.)
            
        Returns:
            List of processor configurations or empty list if no module
        """
        # Get the hook module
        module_attr_name = f"_{hook_type}_module"
        hook_module = getattr(self.stream, module_attr_name, None)
        
        if not hook_module:
            return []
        
        # Get processors from the module
        processors = getattr(hook_module, 'processors', [])
        
        config = []
        for i, processor in enumerate(processors):
            proc_config = {
                'type': getattr(processor, '__class__').__name__,
                'order': getattr(processor, 'order', i),
                'enabled': getattr(processor, 'enabled', True),
            }
            
            # Try to get processor parameters
            if hasattr(processor, 'params'):
                proc_config['params'] = dict(processor.params)
            
            config.append(proc_config)
        
        return config

    def _update_hook_config(self, hook_type: str, desired_config: List[Dict[str, Any]]) -> None:
        """
        Update hook configuration by modifying existing processors in-place instead of recreating them.
        
        Args:
            hook_type: Type of hook (image_preprocessing, image_postprocessing, etc.)
            desired_config: List of processor configurations
        """
        logger.info(f"_update_hook_config: Updating {hook_type} with {len(desired_config)} processors")
        
        # Get or create the hook module
        module_attr_name = f"_{hook_type}_module"
        hook_module = getattr(self.stream, module_attr_name, None)
        
        if not hook_module:
            logger.info(f"_update_hook_config: No existing {hook_type} module, creating new one")
            # Create the appropriate hook module
            try:
                if hook_type in ["image_preprocessing", "image_postprocessing"]:
                    from streamdiffusion.modules.image_processing_module import ImagePreprocessingModule, ImagePostprocessingModule
                    if hook_type == "image_preprocessing":
                        hook_module = ImagePreprocessingModule()
                    else:
                        hook_module = ImagePostprocessingModule()
                elif hook_type in ["latent_preprocessing", "latent_postprocessing"]:
                    from streamdiffusion.modules.latent_processing_module import LatentPreprocessingModule, LatentPostprocessingModule
                    if hook_type == "latent_preprocessing":
                        hook_module = LatentPreprocessingModule()
                    else:
                        hook_module = LatentPostprocessingModule()
                else:
                    raise ValueError(f"Unknown hook type: {hook_type}")
                
                # Install the module
                hook_module.install(self.stream)
                setattr(self.stream, module_attr_name, hook_module)
                logger.info(f"_update_hook_config: Created and installed {hook_type} module")
                
            except Exception as e:
                logger.error(f"_update_hook_config: Failed to create {hook_type} module: {e}")
                return
        
        logger.info(f"_update_hook_config: Found existing {hook_type} module with {len(hook_module.processors)} processors")
        
        # Modify existing processors in-place instead of clearing and recreating
        for i, proc_config in enumerate(desired_config):
            processor_type = proc_config.get('type', 'unknown')
            enabled = proc_config.get('enabled', True)
            params = proc_config.get('params', {})
            
            logger.info(f"_update_hook_config: Processing config {i}: type={processor_type}, enabled={enabled}")
            
            if i < len(hook_module.processors):
                # Modify existing processor
                existing_processor = hook_module.processors[i]
                
                # Get the current processor type from registry name if available, otherwise use class name
                current_type = existing_processor.params.get('_registry_name') if hasattr(existing_processor, 'params') else None
                if not current_type:
                    current_type = existing_processor.__class__.__name__
                
                logger.info(f"_update_hook_config: Modifying existing processor {i}: {current_type} -> {processor_type}")
                
                # If processor type changed, replace it
                if current_type.lower() != processor_type.lower():
                    logger.info(f"_update_hook_config: Type changed, replacing processor {i}")
                    try:
                        from streamdiffusion.preprocessing.processors import get_preprocessor
                        
                        # Determine normalization context from hook type
                        if 'latent' in hook_type:
                            normalization_context = 'latent'
                        else:
                            # Image preprocessing/postprocessing uses 'pipeline' context
                            normalization_context = 'pipeline'
                        
                        new_processor = get_preprocessor(
                            processor_type, 
                            pipeline_ref=getattr(self, 'stream', None),
                            normalization_context=normalization_context
                        )
                        
                        # Copy attributes from old processor
                        setattr(new_processor, 'order', getattr(existing_processor, 'order', i))
                        setattr(new_processor, 'enabled', enabled)
                        
                        # Set parameters
                        if hasattr(new_processor, 'params'):
                            new_processor.params.update(params)
                        
                        hook_module.processors[i] = new_processor
                        logger.info(f"_update_hook_config: Successfully replaced processor {i} with {processor_type}")
                    except Exception as e:
                        logger.error(f"_update_hook_config: Failed to replace processor {i}: {e}")
                else:
                    # Same type, just update attributes
                    logger.info(f"_update_hook_config: Same type, updating attributes for processor {i}")
                    setattr(existing_processor, 'enabled', enabled)
                    
                    # Update parameters
                    if hasattr(existing_processor, 'params'):
                        existing_processor.params.update(params)
                    for param_name, param_value in params.items():
                        if hasattr(existing_processor, param_name):
                            setattr(existing_processor, param_name, param_value)
                    
                    logger.info(f"_update_hook_config: Updated processor {i} enabled={enabled}, params={params}")
            else:
                # Add new processor
                logger.info(f"_update_hook_config: Adding new processor {i}: {processor_type}")
                try:
                    hook_module.add_processor(proc_config)
                    logger.info(f"_update_hook_config: Successfully added processor {i}: {processor_type}")
                except Exception as e:
                    logger.error(f"_update_hook_config: Failed to add processor {i}: {e}")
        
        # Remove extra processors if config is shorter
        while len(hook_module.processors) > len(desired_config):
            removed_idx = len(hook_module.processors) - 1
            removed_processor = hook_module.processors.pop()
            logger.info(f"_update_hook_config: Removed extra processor {removed_idx}: {removed_processor.__class__.__name__}")
        
        logger.info(f"_update_hook_config: Finished updating {hook_type}, now has {len(hook_module.processors)} processors")

