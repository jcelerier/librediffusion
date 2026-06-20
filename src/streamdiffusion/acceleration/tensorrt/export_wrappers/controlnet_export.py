import torch

class SDXLControlNetExportWrapper(torch.nn.Module):
    """Wrapper for SDXL ControlNet models to handle added_cond_kwargs properly during ONNX export"""
    
    def __init__(self, controlnet_model):
        super().__init__()
        self.controlnet = controlnet_model
        
        # Get device and dtype from model
        if hasattr(controlnet_model, 'device'):
            self.device = controlnet_model.device
        else:
            # Try to infer from first parameter
            try:
                self.device = next(controlnet_model.parameters()).device
            except:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if hasattr(controlnet_model, 'dtype'):
            self.dtype = controlnet_model.dtype
        else:
            # Try to infer from first parameter
            try:
                self.dtype = next(controlnet_model.parameters()).dtype
            except:
                self.dtype = torch.float16
    
    def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond, conditioning_scale, text_embeds, time_ids):
        """Forward pass that handles SDXL ControlNet requirements and produces 9 down blocks"""
        # Use the provided SDXL conditioning
        added_cond_kwargs = {
            'text_embeds': text_embeds,
            'time_ids': time_ids
        }
        
        # Call the ControlNet with proper arguments including conditioning_scale
        result = self.controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False
        )
        
        # Extract down blocks and mid block from result
        if isinstance(result, tuple) and len(result) >= 2:
            down_block_res_samples, mid_block_res_sample = result[0], result[1]
        elif hasattr(result, 'down_block_res_samples') and hasattr(result, 'mid_block_res_sample'):
            down_block_res_samples = result.down_block_res_samples
            mid_block_res_sample = result.mid_block_res_sample
        else:
            raise ValueError(f"Unexpected ControlNet output format: {type(result)}")
        
        # SDXL ControlNet should have exactly 9 down blocks
        if len(down_block_res_samples) != 9:
            raise ValueError(f"SDXL ControlNet expected 9 down blocks, got {len(down_block_res_samples)}")
        
        # Return 9 down blocks + 1 mid block with explicit names matching UNet pattern
        # Following the pattern from controlnet_wrapper.py and models.py:
        # down_block_00: Initial sample (320 channels)
        # down_block_01-03: Block 0 residuals (320 channels) 
        # down_block_04-06: Block 1 residuals (640 channels)
        # down_block_07-08: Block 2 residuals (1280 channels)
        down_block_00 = down_block_res_samples[0]  # Initial: 320 channels, 88x88
        down_block_01 = down_block_res_samples[1]  # Block0: 320 channels, 88x88
        down_block_02 = down_block_res_samples[2]  # Block0: 320 channels, 88x88  
        down_block_03 = down_block_res_samples[3]  # Block0: 320 channels, 44x44
        down_block_04 = down_block_res_samples[4]  # Block1: 640 channels, 44x44
        down_block_05 = down_block_res_samples[5]  # Block1: 640 channels, 44x44
        down_block_06 = down_block_res_samples[6]  # Block1: 640 channels, 22x22
        down_block_07 = down_block_res_samples[7]  # Block2: 1280 channels, 22x22
        down_block_08 = down_block_res_samples[8]  # Block2: 1280 channels, 22x22
        mid_block = mid_block_res_sample            # Mid: 1280 channels, 22x22
        
        # Return as individual tensors to preserve names in ONNX
        return (down_block_00, down_block_01, down_block_02, down_block_03, 
                down_block_04, down_block_05, down_block_06, down_block_07, 
                down_block_08, mid_block)