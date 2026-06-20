import torch
import torch.nn as nn
from typing import Optional, Dict


class SDXLUNetWrapper(nn.Module):
    """Wrapper for SDXL UNet to handle additional conditioning during ONNX export"""
    
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        text_embeds: torch.Tensor,
        time_ids: torch.Tensor,
    ):
        # Create added_cond_kwargs dict for SDXL
        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }
        
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]