from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.models import ControlNetModel
import logging

from streamdiffusion.hooks import StepCtx, UnetKwargsDelta, UnetHook
from streamdiffusion.preprocessing.preprocessing_orchestrator import (
    PreprocessingOrchestrator,
)
from streamdiffusion.preprocessing.orchestrator_user import OrchestratorUser


@dataclass
class ControlNetConfig:
    model_id: str
    preprocessor: Optional[str] = None
    conditioning_scale: float = 1.0
    enabled: bool = True
    conditioning_channels: Optional[int] = None
    preprocessor_params: Optional[Dict[str, Any]] = None


class ControlNetModule(OrchestratorUser):
    """ControlNet module that provides a UNet hook for residual conditioning.

    Responsibilities in this step (3):
    - Manage a collection of ControlNet models, their scales, and current images
    - Provide a UNet hook that computes down/mid residuals for active ControlNets
    - Reuse the existing preprocessing orchestrator for control images
    - Do not alter the wrapper or pipeline call sites (registration happens via install())
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16) -> None:
        self.device = device
        self.dtype = dtype

        self.controlnets: List[Optional[ControlNetModel]] = []
        self.controlnet_images: List[Optional[torch.Tensor]] = []
        self.controlnet_scales: List[float] = []
        self.preprocessors: List[Optional[Any]] = []
        self.enabled_list: List[bool] = []

        self._collections_lock = threading.RLock()
        self._preprocessing_orchestrator: Optional[PreprocessingOrchestrator] = None

        self._stream = None  # set in install
        # Per-frame prepared tensor cache to avoid per-step device/dtype alignment and batch repeats
        self._prepared_tensors: List[Optional[torch.Tensor]] = []
        self._prepared_device: Optional[torch.device] = None
        self._prepared_dtype: Optional[torch.dtype] = None
        self._prepared_batch: Optional[int] = None
        self._images_version: int = 0
        
        # Cache expensive lookups to avoid repeated hasattr/getattr calls
        self._engines_by_id: Dict[str, Any] = {}
        self._engines_cache_valid: bool = False
        self._is_sdxl: Optional[bool] = None
        self._expected_text_len: int = 77
        
        # SDXL-specific caching for performance optimization
        self._sdxl_conditioning_cache: Optional[Dict[str, torch.Tensor]] = None
        self._sdxl_conditioning_valid: bool = False
        
        # Cache engine type detection to avoid repeated hasattr calls
        self._engine_type_cache: Dict[str, bool] = {}

    # ---------- Public API (used by wrapper in a later step) ----------
    def install(self, stream) -> None:
        self._stream = stream
        if self._preprocessing_orchestrator is None:
            # Enforce shared orchestrator via base helper (raises if missing)
            self.attach_orchestrator(stream)
        # Register UNet hook
        stream.unet_hooks.append(self.build_unet_hook())
        # Expose controlnet collections so existing updater can find them
        setattr(stream, 'controlnets', self.controlnets)
        setattr(stream, 'controlnet_scales', self.controlnet_scales)
        setattr(stream, 'preprocessors', self.preprocessors)
        # Reset prepared tensors on install
        self._prepared_tensors = []
        self._prepared_device = None
        self._prepared_dtype = None
        self._prepared_batch = None
        # Invalidate caches on install
        self._engines_cache_valid = False
        self._is_sdxl = None
        self._sdxl_conditioning_valid = False
        self._engine_type_cache.clear()

    def add_controlnet(self, cfg: ControlNetConfig, control_image: Optional[Union[str, Any, torch.Tensor]] = None) -> None:
        model = self._load_pytorch_controlnet_model(cfg.model_id, cfg.conditioning_channels)

        preproc = None
        if cfg.preprocessor:
            from streamdiffusion.preprocessing.processors import get_preprocessor
            preproc = get_preprocessor(cfg.preprocessor, pipeline_ref=self._stream, normalization_context='controlnet', params=cfg.preprocessor_params)
            # Apply provided parameters to the preprocessor instance
            if cfg.preprocessor_params:
                params = cfg.preprocessor_params or {}
                # If the preprocessor exposes a 'params' dict, update it
                if hasattr(preproc, 'params') and isinstance(getattr(preproc, 'params'), dict):
                    preproc.params.update(params)
                # Also set attributes directly when they exist
                for name, value in params.items():
                    try:
                        if hasattr(preproc, name):
                            setattr(preproc, name, value)
                    except Exception:
                        pass


            # Align preprocessor target size with stream resolution once (avoid double-resize later)
            try:
                if hasattr(preproc, 'params') and isinstance(getattr(preproc, 'params'), dict):
                    preproc.params['image_width'] = int(self._stream.width)
                    preproc.params['image_height'] = int(self._stream.height)
                if hasattr(preproc, 'image_width'):
                    setattr(preproc, 'image_width', int(self._stream.width))
                if hasattr(preproc, 'image_height'):
                    setattr(preproc, 'image_height', int(self._stream.height))
            except Exception:
                pass

        image_tensor: Optional[torch.Tensor] = None
        if control_image is not None and self._preprocessing_orchestrator is not None:
            image_tensor = self._prepare_control_image(control_image, preproc)

        with self._collections_lock:
            self.controlnets.append(model)
            self.controlnet_images.append(image_tensor)
            self.controlnet_scales.append(float(cfg.conditioning_scale))
            self.preprocessors.append(preproc)
            self.enabled_list.append(bool(cfg.enabled))
            # Invalidate prepared tensors and bump version when graph changes
            self._prepared_tensors = []
            self._images_version += 1
            # Invalidate SDXL conditioning cache when ControlNet configuration changes
            self._sdxl_conditioning_valid = False

    def update_control_image_efficient(self, control_image: Union[str, Any, torch.Tensor], index: Optional[int] = None) -> None:
        if self._preprocessing_orchestrator is None:
            return
        with self._collections_lock:
            if not self.controlnets:
                return
            total = len(self.controlnets)
            # Build active scales, respecting enabled_list if present
            scales = [
                (self.controlnet_scales[i] if i < len(self.controlnet_scales) else 1.0)
                for i in range(total)
            ]
            if hasattr(self, 'enabled_list') and self.enabled_list and len(self.enabled_list) == total:
                scales = [sc if bool(self.enabled_list[i]) else 0.0 for i, sc in enumerate(scales)]
            preprocessors = [self.preprocessors[i] if i < len(self.preprocessors) else None for i in range(total)]

        # Single-index fast path
        if index is not None:
            results = self._preprocessing_orchestrator.process_sync(
                control_image,
                preprocessors,
                scales,
                self._stream.width,
                self._stream.height,
                index
            )
            processed = results[index] if results and len(results) > index else None
            with self._collections_lock:
                if processed is not None and index < len(self.controlnet_images):
                    self.controlnet_images[index] = processed
                    # Invalidate prepared tensors and bump version for per-frame reuse
                    self._prepared_tensors = []
                    self._images_version += 1
                    # Invalidate SDXL conditioning cache
                    self._sdxl_conditioning_valid = False
                    self.prepare_frame_tensors(self.device, self.dtype, 1)
            return

        # Use intelligent pipelining (automatically detects feedback preprocessors and switches to sync)
        processed_images = self._preprocessing_orchestrator.process_pipelined(
            control_image,
            preprocessors,
            scales,
            self._stream.width,
            self._stream.height
        )

        # If orchestrator returns empty list, it indicates no update needed for this frame
        if processed_images is None or (isinstance(processed_images, list) and len(processed_images) == 0):
            return

        # Assign results
        with self._collections_lock:
            for i, img in enumerate(processed_images):
                if img is not None and i < len(self.controlnet_images):
                    self.controlnet_images[i] = img
            # Invalidate prepared cache and bump version after bulk update
            self._prepared_tensors = []
            self._images_version += 1
            # Invalidate SDXL conditioning cache
            self._sdxl_conditioning_valid = False
            self.prepare_frame_tensors(self.device, self.dtype, 1)

    def update_controlnet_scale(self, index: int, scale: float) -> None:
        with self._collections_lock:
            if 0 <= index < len(self.controlnet_scales):
                self.controlnet_scales[index] = float(scale)

    def update_controlnet_enabled(self, index: int, enabled: bool) -> None:
        with self._collections_lock:
            if 0 <= index < len(self.enabled_list):
                self.enabled_list[index] = bool(enabled)

    def remove_controlnet(self, index: int) -> None:
        with self._collections_lock:
            if 0 <= index < len(self.controlnets):
                del self.controlnets[index]
                if index < len(self.controlnet_images):
                    del self.controlnet_images[index]
                if index < len(self.controlnet_scales):
                    del self.controlnet_scales[index]
                if index < len(self.preprocessors):
                    del self.preprocessors[index]
                if index < len(self.enabled_list):
                    del self.enabled_list[index]
                # Invalidate prepared tensors and bump version
                self._prepared_tensors = []
                self._images_version += 1
                # Invalidate SDXL conditioning cache
                self._sdxl_conditioning_valid = False

    def reorder_controlnets_by_model_ids(self, desired_model_ids: List[str]) -> None:
        """Reorder internal collections to match the desired model_id order.

        Any controlnet whose model_id is not present in desired_model_ids retains its
        relative order after those that are specified.
        """
        with self._collections_lock:
            # Build current mapping from model_id to index
            current_ids: List[str] = []
            for i, cn in enumerate(self.controlnets):
                model_id = getattr(cn, 'model_id', f'controlnet_{i}')
                current_ids.append(model_id)

            # Compute new index order
            picked = set()
            new_order: List[int] = []
            for mid in desired_model_ids:
                if mid in current_ids:
                    idx = current_ids.index(mid)
                    new_order.append(idx)
                    picked.add(idx)
            # Append remaining indices (not specified) preserving order
            for i in range(len(self.controlnets)):
                if i not in picked:
                    new_order.append(i)

            if new_order == list(range(len(self.controlnets))):
                return  # Already in desired order

            def reindex(lst: List[Any]) -> List[Any]:
                return [lst[i] for i in new_order]

            self.controlnets = reindex(self.controlnets)
            self.controlnet_images = reindex(self.controlnet_images)
            self.controlnet_scales = reindex(self.controlnet_scales)
            self.preprocessors = reindex(self.preprocessors)
            self.enabled_list = reindex(self.enabled_list)

    def get_current_config(self) -> List[Dict[str, Any]]:
        cfg: List[Dict[str, Any]] = []
        with self._collections_lock:
            for i, cn in enumerate(self.controlnets):
                model_id = getattr(cn, 'model_id', f'controlnet_{i}')
                scale = self.controlnet_scales[i] if i < len(self.controlnet_scales) else 1.0
                preproc_params = getattr(self.preprocessors[i], 'params', {}) if i < len(self.preprocessors) and self.preprocessors[i] else {}
                cfg.append({
                    'model_id': model_id,
                    'conditioning_scale': scale,
                    'preprocessor_params': preproc_params,
                    'enabled': (self.enabled_list[i] if i < len(self.enabled_list) else True),
                })
        return cfg

    def prepare_frame_tensors(self, device: torch.device, dtype: torch.dtype, batch_size: int) -> None:
        """Prepare control image tensors for the current frame.
        
        This method is called once per frame to prepare all control images with the correct
        device, dtype, and batch size. This avoids redundant operations during each denoising step.
        
        Args:
            device: Target device for tensors
            dtype: Target dtype for tensors
            batch_size: Target batch size
        """
        with self._collections_lock:
            # Check if we need to re-prepare tensors
            cache_valid = (
                self._prepared_device == device and
                self._prepared_dtype == dtype and
                self._prepared_batch == batch_size and
                len(self._prepared_tensors) == len(self.controlnet_images)
            )
            
            if cache_valid:
                return
            
            # Prepare tensors for current frame
            self._prepared_tensors = []
            for img in self.controlnet_images:
                if img is None:
                    self._prepared_tensors.append(None)
                    continue
                
                # Prepare tensor with correct batch size
                prepared = img
                if prepared.dim() == 4 and prepared.shape[0] != batch_size:
                    if prepared.shape[0] == 1:
                        prepared = prepared.repeat(batch_size, 1, 1, 1)
                    else:
                        repeat_factor = max(1, batch_size // prepared.shape[0])
                        prepared = prepared.repeat(repeat_factor, 1, 1, 1)[:batch_size]
                
                # Move to correct device and dtype
                prepared = prepared.to(device=device, dtype=dtype)
                self._prepared_tensors.append(prepared)
            
            # Update cache state
            self._prepared_device = device
            self._prepared_dtype = dtype
            self._prepared_batch = batch_size

    def _get_cached_sdxl_conditioning(self, ctx: 'StepCtx') -> Optional[Dict[str, torch.Tensor]]:
        """Get cached SDXL conditioning to avoid repeated preparation"""
        if not self._is_sdxl or ctx.sdxl_cond is None:
            return None
            
        # Check if cache is valid
        if self._sdxl_conditioning_valid and self._sdxl_conditioning_cache is not None:
            cached = self._sdxl_conditioning_cache
            # Verify batch size matches current context
            if ('text_embeds' in cached and 
                cached['text_embeds'].shape[0] == ctx.x_t_latent.shape[0]):
                return cached
        
        # Cache miss or invalid - prepare new conditioning
        try:
            conditioning = {}
            if 'text_embeds' in ctx.sdxl_cond:
                text_embeds = ctx.sdxl_cond['text_embeds']
                batch_size = ctx.x_t_latent.shape[0]
                
                # Optimize batch expansion for SDXL text embeddings
                if text_embeds.shape[0] != batch_size:
                    if text_embeds.shape[0] == 1:
                        conditioning['text_embeds'] = text_embeds.repeat(batch_size, 1)
                    else:
                        conditioning['text_embeds'] = text_embeds[:batch_size]
                else:
                    conditioning['text_embeds'] = text_embeds
            
            if 'time_ids' in ctx.sdxl_cond:
                time_ids = ctx.sdxl_cond['time_ids']
                batch_size = ctx.x_t_latent.shape[0]
                
                # Optimize batch expansion for SDXL time IDs
                if time_ids.shape[0] != batch_size:
                    if time_ids.shape[0] == 1:
                        conditioning['time_ids'] = time_ids.repeat(batch_size, 1)
                    else:
                        conditioning['time_ids'] = time_ids[:batch_size]
                else:
                    conditioning['time_ids'] = time_ids
            
            # Cache the prepared conditioning
            self._sdxl_conditioning_cache = conditioning
            self._sdxl_conditioning_valid = True
            return conditioning
            
        except Exception:
            # Fallback to original conditioning on any error
            return ctx.sdxl_cond

    # ---------- Internal helpers ----------
    def build_unet_hook(self) -> UnetHook:
        def _unet_hook(ctx: StepCtx) -> UnetKwargsDelta:
            # Compute residuals under lock, using only original text tokens for ControlNet encoding
            x_t = ctx.x_t_latent
            t_list = ctx.t_list

            with self._collections_lock:
                if not self.controlnets:
                    return UnetKwargsDelta()

                # Single pass to collect active ControlNet data
                active_data = []
                enabled_flags = self.enabled_list if len(self.enabled_list) == len(self.controlnets) else None
                
                for i, (cn, img, scale) in enumerate(zip(self.controlnets, self.controlnet_images, self.controlnet_scales)):
                    if cn is not None and img is not None and scale > 0:
                        enabled = enabled_flags[i] if enabled_flags else True
                        if enabled:
                            active_data.append((cn, img, scale, i))

                if not active_data:
                    return UnetKwargsDelta()

            # Cache TRT engines lookup to avoid rebuilding every frame
            if not self._engines_cache_valid:
                self._engines_by_id.clear()
                try:
                    if hasattr(self._stream, 'controlnet_engines') and isinstance(self._stream.controlnet_engines, list):
                        for eng in self._stream.controlnet_engines:
                            mid = getattr(eng, 'model_id', None)
                            if mid:
                                self._engines_by_id[mid] = eng
                    self._engines_cache_valid = True
                except Exception:
                    pass

            # Cache SDXL detection to avoid repeated hasattr calls
            if self._is_sdxl is None:
                try:
                    self._is_sdxl = getattr(self._stream, 'is_sdxl', False)
                except Exception:
                    self._is_sdxl = False

            encoder_hidden_states = self._stream.prompt_embeds[:, :self._expected_text_len, :]

            base_kwargs: Dict[str, Any] = {
                'sample': x_t,
                'timestep': t_list,
                'encoder_hidden_states': encoder_hidden_states,
                'return_dict': False,
            }

            down_samples_list: List[List[torch.Tensor]] = []
            mid_samples_list: List[torch.Tensor] = []

            # Ensure tensors are prepared for this frame
            # This should have been called earlier, but we call it here as a safety net
            if (self._prepared_device != x_t.device or 
                self._prepared_dtype != x_t.dtype or 
                self._prepared_batch != x_t.shape[0]):
                self.prepare_frame_tensors(x_t.device, x_t.dtype, x_t.shape[0])
            
            # Use pre-prepared tensors
            prepared_images = self._prepared_tensors

            for cn, img, scale, idx_i in active_data:
                # Swap to TRT engine if available for this model_id (use cached lookup)
                model_id = getattr(cn, 'model_id', None)
                if model_id and model_id in self._engines_by_id:
                    cn = self._engines_by_id[model_id]
                
                # Use pre-prepared tensor
                current_img = prepared_images[idx_i] if idx_i < len(prepared_images) else img
                if current_img is None:
                    continue

                # Check if this is TensorRT engine (use cached result to avoid repeated hasattr calls)
                cache_key = id(cn)  # Use object id as unique identifier
                if cache_key in self._engine_type_cache:
                    is_trt_engine = self._engine_type_cache[cache_key]
                else:
                    is_trt_engine = hasattr(cn, 'engine') and hasattr(cn, 'stream')
                    self._engine_type_cache[cache_key] = is_trt_engine
                
                # Get optimized SDXL conditioning (uses caching to avoid repeated tensor operations)
                added_cond_kwargs = self._get_cached_sdxl_conditioning(ctx)
                
                try:
                    if is_trt_engine:
                        # TensorRT engine path
                        if added_cond_kwargs:
                            down_samples, mid_sample = cn(
                                sample=x_t,
                                timestep=t_list,
                                encoder_hidden_states=encoder_hidden_states,
                                controlnet_cond=current_img,
                                conditioning_scale=float(scale),
                                **added_cond_kwargs
                            )
                        else:
                            down_samples, mid_sample = cn(
                                sample=x_t,
                                timestep=t_list,
                                encoder_hidden_states=encoder_hidden_states,
                                controlnet_cond=current_img,
                                conditioning_scale=float(scale)
                            )
                    else:
                        # PyTorch ControlNet path
                        if added_cond_kwargs:
                            down_samples, mid_sample = cn(
                                sample=x_t,
                                timestep=t_list,
                                encoder_hidden_states=encoder_hidden_states,
                                controlnet_cond=current_img,
                                conditioning_scale=float(scale),
                                return_dict=False,
                                added_cond_kwargs=added_cond_kwargs
                            )
                        else:
                            down_samples, mid_sample = cn(
                                sample=x_t,
                                timestep=t_list,
                                encoder_hidden_states=encoder_hidden_states,
                                controlnet_cond=current_img,
                                conditioning_scale=float(scale),
                                return_dict=False
                            )
                except Exception as e:
                    import traceback
                    __import__('logging').getLogger(__name__).error("ControlNetModule: controlnet forward failed: %s", e)
                    try:
                        __import__('logging').getLogger(__name__).error("ControlNetModule: call_summary: cond_shape=%s, img_shape=%s, scale=%s, is_sdxl=%s, is_trt=%s",
                                     (tuple(encoder_hidden_states.shape) if isinstance(encoder_hidden_states, torch.Tensor) else None),
                                     (tuple(current_img.shape) if isinstance(current_img, torch.Tensor) else None),
                                     scale,
                                     self._is_sdxl,
                                     is_trt_engine)
                    except Exception:
                        pass
                    __import__('logging').getLogger(__name__).error(traceback.format_exc())
                    continue
                down_samples_list.append(down_samples)
                mid_samples_list.append(mid_sample)

            if not down_samples_list:
                return UnetKwargsDelta()

            if len(down_samples_list) == 1:
                return UnetKwargsDelta(
                    down_block_additional_residuals=down_samples_list[0],
                    mid_block_additional_residual=mid_samples_list[0],
                )

            # Merge multiple ControlNet residuals
            merged_down = down_samples_list[0]
            merged_mid = mid_samples_list[0]
            for ds, ms in zip(down_samples_list[1:], mid_samples_list[1:]):
                for j in range(len(merged_down)):
                    merged_down[j] = merged_down[j] + ds[j]
                merged_mid = merged_mid + ms

            return UnetKwargsDelta(
                down_block_additional_residuals=merged_down,
                mid_block_additional_residual=merged_mid,
            )

        return _unet_hook

    def _prepare_control_image(self, control_image: Union[str, Any, torch.Tensor], preprocessor: Optional[Any]) -> torch.Tensor:
        if self._preprocessing_orchestrator is None:
            raise RuntimeError("ControlNetModule: preprocessing orchestrator is not initialized")
        # Reuse orchestrator API used by BaseControlNetPipeline
        images = self._preprocessing_orchestrator.process_sync(
            control_image,
            [preprocessor],
            [1.0],
            self._stream.width,
            self._stream.height,
            0
        )
        # API returns a list; pick first if present
        return images[0] if images else None

    #FIXME: more robust model management is needed in general.
    def _load_pytorch_controlnet_model(self, model_id: str, conditioning_channels: Optional[int] = None) -> ControlNetModel:
        from pathlib import Path
        import logging
        import os
        logger = logging.getLogger(__name__)
        
        try:
            # Prepare loading kwargs
            load_kwargs = {"torch_dtype": self.dtype}
            if conditioning_channels is not None:
                load_kwargs["conditioning_channels"] = conditioning_channels
            
            # Check if offline mode is enabled via environment variables
            is_offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
            
            if Path(model_id).exists():
                model_path = Path(model_id)
                
                # Check if it's a direct file path to a safetensors/ckpt file
                if model_path.is_file() and model_path.suffix in ['.safetensors', '.ckpt', '.bin']:
                    logger.info(f"ControlNetModule._load_pytorch_controlnet_model: Loading ControlNet from single file: {model_path} (channels={conditioning_channels})")
                    # Try loading from single file (works for most ControlNet models)
                    try:
                        controlnet = ControlNetModel.from_single_file(str(model_path), **load_kwargs)
                    except Exception as e:
                        logger.warning(f"ControlNetModule._load_pytorch_controlnet_model: Single file loading failed: {e}")
                        # Fallback: try pretrained loading in case it's in a proper directory structure
                        load_kwargs["local_files_only"] = True
                        controlnet = ControlNetModel.from_pretrained(str(model_path.parent), **load_kwargs)
                else:
                    # It's a directory path
                    load_kwargs["local_files_only"] = True
                    controlnet = ControlNetModel.from_pretrained(model_id, **load_kwargs)
            else:
                # Loading from HuggingFace Hub - respect offline mode
                if is_offline:
                    load_kwargs["local_files_only"] = True
                    logger.info(f"ControlNetModule._load_pytorch_controlnet_model: Offline mode enabled, loading '{model_id}' from cache only")
                
                if "/" in model_id and model_id.count("/") > 1:
                    parts = model_id.split("/")
                    repo_id = "/".join(parts[:2])
                    subfolder = "/".join(parts[2:])
                    controlnet = ControlNetModel.from_pretrained(
                        repo_id, subfolder=subfolder, **load_kwargs
                    )
                else:
                    controlnet = ControlNetModel.from_pretrained(model_id, **load_kwargs)
            controlnet = controlnet.to(device=self.device, dtype=self.dtype)
            # Track model_id for updater diffing
            try:
                setattr(controlnet, 'model_id', model_id)
            except Exception:
                pass
            return controlnet
        except Exception as e:
            import logging, traceback
            logger = logging.getLogger(__name__)
            logger.error(f"ControlNetModule: failed to load model '{model_id}': {e}")
            logger.error(traceback.format_exc())
            raise



