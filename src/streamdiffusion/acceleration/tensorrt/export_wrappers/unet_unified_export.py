import torch
from diffusers import UNet2DConditionModel
from typing import Optional, List
from .unet_controlnet_export import create_controlnet_wrapper
from .unet_ipadapter_export import create_ipadapter_wrapper
from ..models.utils import convert_list_to_structure

class UnifiedExportWrapper(torch.nn.Module):
    """
    Unified wrapper that composes wrappers for conditioning modules. 
    """
    
    def __init__(self, 
                 unet: UNet2DConditionModel, 
                 use_controlnet: bool = False,
                 use_ipadapter: bool = False,
                 control_input_names: Optional[List[str]] = None,
                 num_tokens: int = 4,
                 kvo_cache_structure: List[int] = [],
                 **kwargs):
        super().__init__()
        self.use_controlnet = use_controlnet
        self.use_ipadapter = use_ipadapter
        self.controlnet_wrapper = None
        self.ipadapter_wrapper = None
        self.unet = unet
        self.kvo_cache_structure = kvo_cache_structure
        
        # Apply IPAdapter first (installs processors into UNet)
        if use_ipadapter:
            ipadapter_kwargs = {k: v for k, v in kwargs.items() if k in ['install_processors']}
            if 'install_processors' not in ipadapter_kwargs:
                ipadapter_kwargs['install_processors'] = True
            

            self.ipadapter_wrapper = create_ipadapter_wrapper(unet, num_tokens=num_tokens, **ipadapter_kwargs)
            self.unet = self.ipadapter_wrapper.unet
        
        # Apply ControlNet second (wraps whatever UNet we have)
        if use_controlnet and control_input_names:
            controlnet_kwargs = {k: v for k, v in kwargs.items() if k in ['num_controlnets', 'conditioning_scales']}

            self.controlnet_wrapper = create_controlnet_wrapper(self.unet, control_input_names, kvo_cache_structure, **controlnet_kwargs)
        
    def _basic_unet_forward(self, sample, timestep, encoder_hidden_states, *kvo_cache, **kwargs):
        """Basic UNet forward that passes through all parameters to handle any model type"""
        formatted_kvo_cache = []
        if len(kvo_cache) > 0:
            formatted_kvo_cache = convert_list_to_structure(kvo_cache, self.kvo_cache_structure)

        unet_kwargs = {
            'sample': sample,
            'timestep': timestep,
            'encoder_hidden_states': encoder_hidden_states,
            'return_dict': False,
            'kvo_cache': formatted_kvo_cache,
            **kwargs  # Pass through all additional parameters (SDXL, future model types, etc.)
        }
        res = self.unet(**unet_kwargs)
        if len(kvo_cache) > 0:
            return res
        else:
            return res[0]
        
    def forward(self, 
                sample: torch.Tensor,
                timestep: torch.Tensor, 
                encoder_hidden_states: torch.Tensor,
                *args,
                **kwargs) -> torch.Tensor:
        """Forward pass that handles any UNet parameters via **kwargs passthrough"""
        # Handle IP-Adapter runtime scale vector as a positional argument placed before control tensors
        if self.use_ipadapter and self.ipadapter_wrapper is not None:
            # ipadapter_scale is appended as the first extra positional input after the 3 base inputs
            if len(args) == 0:
                import logging
                logging.getLogger(__name__).error("UnifiedExportWrapper: ipadapter_scale missing; required when use_ipadapter=True")
                raise RuntimeError("UnifiedExportWrapper: ipadapter_scale tensor is required when use_ipadapter=True")
            ipadapter_scale = args[0]
            if not isinstance(ipadapter_scale, torch.Tensor):
                import logging
                logging.getLogger(__name__).error(f"UnifiedExportWrapper: ipadapter_scale wrong type: {type(ipadapter_scale)}")
                raise TypeError("ipadapter_scale must be a torch.Tensor")
            try:
                import logging
                logging.getLogger(__name__).debug(f"UnifiedExportWrapper: ipadapter_scale shape={tuple(ipadapter_scale.shape)}, dtype={ipadapter_scale.dtype}")
            except Exception:
                pass
            # assign per-layer scale tensors into processors
            self.ipadapter_wrapper.set_ipadapter_scale(ipadapter_scale)
            # remove it from control args before passing to controlnet wrapper
            args = args[1:]

        if self.controlnet_wrapper:
            # ControlNet wrapper handles the UNet call with all parameters
            return self.controlnet_wrapper(sample, timestep, encoder_hidden_states, *args, **kwargs)
        else:
            # Basic UNet call with all parameters passed through
            return self._basic_unet_forward(sample, timestep, encoder_hidden_states, *args, **kwargs) 