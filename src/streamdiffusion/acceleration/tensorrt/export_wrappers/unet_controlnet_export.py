"""ControlNet-aware UNet wrapper for ONNX export"""

import torch
from typing import List, Optional, Dict, Any
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from ..models.utils import convert_list_to_structure


class ControlNetUNetExportWrapper(torch.nn.Module):
    """Wrapper that combines UNet with ControlNet inputs for ONNX export"""
    
    def __init__(self, unet: UNet2DConditionModel, control_input_names: List[str], kvo_cache_structure: List[int]):
        super().__init__()
        self.unet = unet
        self.control_input_names = control_input_names
        self.kvo_cache_structure = kvo_cache_structure
        
        self.control_names = []
        for name in control_input_names:
            if "input_control" in name or "output_control" in name or "middle_control" in name:
                self.control_names.append(name)
        
        self.num_controlnet_args = len(self.control_names)
        
        # Detect if this is SDXL based on UNet config
        self.is_sdxl = self._detect_sdxl_architecture(unet)

        # The number of down-block control residuals is whatever get_control declared (== the
        # count of input_control_* names, excluding the middle). This is the single source of
        # truth and works for every arch: SD1.5=12, SDXL=9, SDXS(pruned, lpb1)=6. Do NOT hardcode
        # per-architecture — SDXS and SDXL are both 3-block but differ (6 vs 9) by layers_per_block.
        self.expected_down_blocks = sum(
            1 for n in self.control_names if "input_control" in n and "middle" not in n
        )
        if self.expected_down_blocks == 0:  # fallback for legacy naming
            self.expected_down_blocks = 9 if self.is_sdxl else 12
    
    def _detect_sdxl_architecture(self, unet):
        """Detect if UNet is SDXL based on architecture"""
        if hasattr(unet, 'config'):
            config = unet.config
            # SDXL has 3 down blocks vs SD1.5's 4 down blocks
            block_out_channels = getattr(config, 'block_out_channels', None)
            if block_out_channels and len(block_out_channels) == 3:
                return True
        return False
    
    def forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        """Forward pass that organizes control inputs and calls UNet"""
        
        control_args = args[:self.num_controlnet_args]
        kvo_cache = args[self.num_controlnet_args:]
        
        down_block_controls = []
        mid_block_control = None
        
        if control_args:
            all_control_tensors = []
            middle_tensor = None
            
            for tensor, name in zip(control_args, self.control_names):
                if "input_control" in name:
                    if "middle" in name:
                        middle_tensor = tensor
                    else:
                        all_control_tensors.append(tensor)
                elif "middle_control" in name:
                    middle_tensor = tensor
            
            if len(all_control_tensors) == self.expected_down_blocks:
                down_block_controls = all_control_tensors
                mid_block_control = middle_tensor
            else:
                # Try to adapt the available tensors
                if len(all_control_tensors) > 0:
                    if len(all_control_tensors) > self.expected_down_blocks:
                        # Too many tensors - take the first expected_down_blocks
                        down_block_controls = all_control_tensors[:self.expected_down_blocks]
                    else:
                        # Too few tensors - use what we have
                        down_block_controls = all_control_tensors
                    mid_block_control = middle_tensor
                else:
                    # No control tensors available - skip ControlNet
                    down_block_controls = None
                    mid_block_control = None
        
        formatted_kvo_cache = []
        if len(kvo_cache) > 0:
            formatted_kvo_cache = convert_list_to_structure(kvo_cache, self.kvo_cache_structure)

        unet_kwargs = {
            'sample': sample,
            'timestep': timestep,
            'encoder_hidden_states': encoder_hidden_states,
            'kvo_cache': formatted_kvo_cache,
            'return_dict': False,
        }
        
        # Pass through all additional kwargs (for SDXL models)
        unet_kwargs.update(kwargs)
        
        if down_block_controls:
            # Adapt control tensor shapes for SDXL if needed
            adapted_controls = self._adapt_control_tensors(down_block_controls, sample)

            # Control tensors are now generated in the correct order to match UNet's down_block_res_samples
            # For SDXL: [88x88, 88x88, 88x88, 44x44, 44x44, 44x44, 22x22, 22x22, 22x22]
            # This directly aligns with UNet's: [initial_sample] + [block0_residuals] + [block1_residuals] + [block2_residuals]
            unet_kwargs['down_block_additional_residuals'] = adapted_controls
        
        if mid_block_control is not None:
            # Adapt middle control tensor shape if needed
            adapted_mid_control = self._adapt_middle_control_tensor(mid_block_control, sample)
            unet_kwargs['mid_block_additional_residual'] = adapted_mid_control
        
        try:
            res = self.unet(**unet_kwargs)
            if len(kvo_cache) > 0:
                return res
            else:
                return res[0]
        except Exception as e:
            print(f"❌ DEBUG: UNet forward failed: {e}")
            raise
    
    def _adapt_control_tensors(self, control_tensors, sample):
        """Pass control residuals through unchanged.

        The control inputs are declared by UNet.get_control() at EXACTLY the spatial/channel
        shapes of the diffusers UNet's down_block_res_samples (verified against the real
        ControlNetModel trace for SD1.5/SDXL/SDXS). The sample tensors fed during ONNX export
        already carry those correct shapes, so NO interpolation is needed — and arch-guessed
        per-index factor lists were WRONG for pruned arches (e.g. SDXS 3-block/lpb1 has factors
        [1,1,2,2,4,4], not SDXL's [1,1,1,2,2,2,4,4,4]), which corrupted residual[i] to the wrong
        resolution and broke the element-wise add in unet_2d_condition.forward. Keep this a
        no-op: down_block_additional_residuals[i] must equal down_block_res_samples[i] as-is.
        """
        return list(control_tensors)

    def _adapt_middle_control_tensor(self, mid_control, sample):
        """Pass the mid-block control residual through unchanged (see _adapt_control_tensors)."""
        return mid_control


class MultiControlNetUNetExportWrapper(torch.nn.Module):
    """Advanced wrapper for multiple ControlNets with different scales"""
    
    def __init__(self, 
                 unet: UNet2DConditionModel, 
                 control_input_names: List[str],
                 kvo_cache_structure: List[int],
                 num_controlnets: int = 1,
                 conditioning_scales: Optional[List[float]] = None):
        super().__init__()
        self.unet = unet
        self.control_input_names = control_input_names
        self.num_controlnets = num_controlnets
        self.conditioning_scales = conditioning_scales or [1.0] * num_controlnets
        self.kvo_cache_structure = kvo_cache_structure
        
        self.control_names = []
        for name in control_input_names:
            if "input_control" in name or "output_control" in name or "middle_control" in name:
                self.control_names.append(name)
        
        self.num_controlnet_args = len(self.control_names)

        self.controlnet_indices = []
        controls_per_net = self.num_controlnet_args // num_controlnets
        
        for cn_idx in range(num_controlnets):
            start_idx = cn_idx * controls_per_net
            end_idx = start_idx + controls_per_net
            self.controlnet_indices.append(list(range(start_idx, end_idx)))
    
    def forward(self, sample, timestep, encoder_hidden_states, *args):
        """Forward pass for multiple ControlNets"""
        control_args = args[:self.num_controlnet_args]
        kvo_cache = args[self.num_controlnet_args:]

        combined_down_controls = None
        combined_mid_control = None
        
        for cn_idx, indices in enumerate(self.controlnet_indices):
            scale = self.conditioning_scales[cn_idx]
            if scale == 0:
                continue
            
            cn_controls = [control_args[i] for i in indices if i < len(control_args)]
            
            if not cn_controls:
                continue
            
            num_down = len(cn_controls) - 1
            down_controls = cn_controls[:num_down]
            mid_control = cn_controls[num_down] if num_down < len(cn_controls) else None
            
            scaled_down = [ctrl * scale for ctrl in down_controls]
            scaled_mid = mid_control * scale if mid_control is not None else None
            
            if combined_down_controls is None:
                combined_down_controls = scaled_down
                combined_mid_control = scaled_mid
            else:
                for i in range(min(len(combined_down_controls), len(scaled_down))):
                    combined_down_controls[i] += scaled_down[i]
                if scaled_mid is not None and combined_mid_control is not None:
                    combined_mid_control += scaled_mid
        
        formatted_kvo_cache = []
        if len(kvo_cache) > 0:
            formatted_kvo_cache = convert_list_to_structure(kvo_cache, self.kvo_cache_structure)

        unet_kwargs = {
            'sample': sample,
            'timestep': timestep,
            'encoder_hidden_states': encoder_hidden_states,
            'kvo_cache': formatted_kvo_cache,
            'return_dict': False,
        }
        
        if combined_down_controls:
            unet_kwargs['down_block_additional_residuals'] = list(reversed(combined_down_controls))
        if combined_mid_control is not None:
            unet_kwargs['mid_block_additional_residual'] = combined_mid_control
        
        res = self.unet(**unet_kwargs)
        if len(kvo_cache) > 0:
            return res
        else:
            return res[0]
        return res


def create_controlnet_wrapper(unet: UNet2DConditionModel, 
                            control_input_names: List[str],
                            kvo_cache_structure: List[int],
                            num_controlnets: int = 1,
                            conditioning_scales: Optional[List[float]] = None) -> torch.nn.Module:
    """Factory function to create appropriate ControlNet wrapper"""
    if num_controlnets == 1:
        return ControlNetUNetExportWrapper(unet, control_input_names, kvo_cache_structure)
    else:
        return MultiControlNetUNetExportWrapper(
            unet, control_input_names, kvo_cache_structure, num_controlnets, conditioning_scales
        )


def organize_control_tensors(control_tensors: List[torch.Tensor], 
                           control_input_names: List[str]) -> Dict[str, List[torch.Tensor]]:
    """Organize control tensors by type (input, output, middle)"""
    organized = {'input': [], 'output': [], 'middle': []}
    
    for tensor, name in zip(control_tensors, control_input_names):
        if "input_control" in name:
            organized['input'].append(tensor)
        elif "output_control" in name:
            organized['output'].append(tensor)
        elif "middle_control" in name:
            organized['middle'].append(tensor)
    
    return organized 