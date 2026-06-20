"""ControlNet TensorRT model definitions for compilation"""

from typing import List, Dict, Optional
from .models import BaseModel
from ..export_wrappers.unet_sdxl_export import SDXLConditioningHandler, get_sdxl_tensorrt_config
from ....model_detection import detect_model
import torch


class ControlNetTRT(BaseModel):
    """TensorRT model definition for ControlNet compilation"""
    
    def __init__(self, 
                 fp16: bool = True,
                 device: str = "cuda",
                 min_batch_size: int = 1,
                 max_batch_size: int = 4,
                 embedding_dim: int = 768,
                 unet_dim: int = 4,
                 conditioning_channels: int = 3,
                 **kwargs):
        super().__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            **kwargs
        )
        self.unet_dim = unet_dim
        self.conditioning_channels = conditioning_channels
        self.name = "ControlNet"
        
    def get_input_names(self) -> List[str]:
        """Get input names for ControlNet TensorRT engine"""
        return [
            "sample",
            "timestep",
            "encoder_hidden_states",
            "controlnet_cond"
        ]
        
    def get_output_names(self) -> List[str]:
        """Get output names for ControlNet TensorRT engine"""
        down_names = [f"down_block_{i:02d}" for i in range(12)]
        return down_names + ["mid_block"]
        
    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        """Get dynamic axes configuration for variable input shapes"""
        return {
            "sample": {0: "B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "B"},
            "timestep": {0: "B"},
            "controlnet_cond": {0: "B", 2: "H_ctrl", 3: "W_ctrl"},
            **{f"down_block_{i:02d}": {0: "B", 2: "H", 3: "W"} for i in range(12)},
            "mid_block": {0: "B", 2: "H", 3: "W"}
        }
    
    def get_input_profile(self, batch_size, image_height, image_width, 
                         static_batch, static_shape):
        """Generate TensorRT input profiles for ControlNet with dynamic 384-1024 range"""
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        
        # Force dynamic shapes for universal engines (384-1024 range)
        min_ctrl_h = 384  # Changed from 256 to 512 to match min resolution
        max_ctrl_h = 1024
        min_ctrl_w = 384  # Changed from 256 to 512 to match min resolution
        max_ctrl_w = 1024
        
        # Use a flexible optimal resolution that's in the middle of the range
        # This allows the engine to handle both smaller and larger resolutions
        opt_ctrl_h = 704  # Middle of 512-1024 range
        opt_ctrl_w = 704  # Middle of 512-1024 range
        
        # Calculate latent dimensions
        min_latent_h = min_ctrl_h // 8  # 64
        max_latent_h = max_ctrl_h // 8  # 128
        min_latent_w = min_ctrl_w // 8  # 64
        max_latent_w = max_ctrl_w // 8  # 128
        opt_latent_h = opt_ctrl_h // 8  # 96
        opt_latent_w = opt_ctrl_w // 8  # 96
        
        profile = {
            "sample": [
                (min_batch, self.unet_dim, min_latent_h, min_latent_w),
                (batch_size, self.unet_dim, opt_latent_h, opt_latent_w),
                (max_batch, self.unet_dim, max_latent_h, max_latent_w),
            ],
            "timestep": [
                (min_batch,), (batch_size,), (max_batch,)
            ],
            "encoder_hidden_states": [
                (min_batch, 77, self.embedding_dim),
                (batch_size, 77, self.embedding_dim),
                (max_batch, 77, self.embedding_dim),
            ],
            "controlnet_cond": [
                (min_batch, self.conditioning_channels, min_ctrl_h, min_ctrl_w),
                (batch_size, self.conditioning_channels, opt_ctrl_h, opt_ctrl_w),
                (max_batch, self.conditioning_channels, max_ctrl_h, max_ctrl_w),
            ],
        }
        
        return profile
    
    def get_sample_input(self, batch_size, image_height, image_width):
        """Generate sample inputs for ONNX export"""
        latent_height = image_height // 8
        latent_width = image_width // 8
        dtype = torch.float16 if self.fp16 else torch.float32
        
        return (
            torch.randn(batch_size, self.unet_dim, latent_height, latent_width, 
                       dtype=dtype, device=self.device),
            torch.ones(batch_size, dtype=torch.float32, device=self.device),
            torch.randn(batch_size, 77, self.embedding_dim, 
                       dtype=dtype, device=self.device),
            torch.randn(batch_size, self.conditioning_channels, image_height, image_width, 
                       dtype=dtype, device=self.device)
        )


class ControlNetSDXLTRT(ControlNetTRT):
    """SDXL-specific ControlNet TensorRT model definition"""
    
    def __init__(self, unet=None, model_path="", **kwargs):
        # Use new model detection if UNet provided
        if unet is not None:
            # Use the new detection function
            detection_result = detect_model(unet)

            # Create a config dict compatible with SDXLConditioningHandler
            config = {
                'is_sdxl': detection_result['is_sdxl'],
                'has_time_cond': detection_result['architecture_details']['has_time_conditioning'],
                'has_addition_embed': detection_result['architecture_details']['has_addition_embeds'],
                'model_type': detection_result['model_type'],
                'is_turbo': detection_result['is_turbo'],
                'is_sd3': detection_result['is_sd3'],
                'confidence': detection_result['confidence'],
                'architecture_details': detection_result['architecture_details'],
                'compatibility_info': detection_result['compatibility_info']
            }
            conditioning_handler = SDXLConditioningHandler(config)
            conditioning_spec = conditioning_handler.get_conditioning_spec()
            
            # Set embedding_dim from sophisticated detection
            kwargs.setdefault('embedding_dim', conditioning_spec['context_dim'])
        
        # Set SDXL-specific defaults
        kwargs.setdefault('embedding_dim', 2048)  # SDXL uses 2048-dim embeddings
        kwargs.setdefault('unet_dim', 4)          # SDXL latent channels
        
        super().__init__(**kwargs)
        
        # SDXL ControlNet output specifications - 9 down blocks + 1 mid block
        # Following the pattern from UNet implementation:
        self.sdxl_output_channels = {
            # Initial sample
            'down_block_00': 320,   # Initial: 320 channels
            # Block 0 residuals  
            'down_block_01': 320,   # Block0: 320 channels
            'down_block_02': 320,   # Block0: 320 channels
            'down_block_03': 320,   # Block0: 320 channels
            # Block 1 residuals
            'down_block_04': 640,   # Block1: 640 channels
            'down_block_05': 640,   # Block1: 640 channels
            'down_block_06': 640,   # Block1: 640 channels
            # Block 2 residuals
            'down_block_07': 1280,  # Block2: 1280 channels
            'down_block_08': 1280,  # Block2: 1280 channels
            # Mid block
            'mid_block': 1280       # Mid: 1280 channels
        }
    
    def get_shape_dict(self, batch_size, image_height, image_width):
        """Override to provide SDXL-specific output shapes for 9 down blocks"""
        # Get base input shapes
        base_shapes = super().get_shape_dict(batch_size, image_height, image_width)
        
        # Add conditioning_scale to input shapes (scalar tensor)
        base_shapes["conditioning_scale"] = ()  # Scalar tensor has empty shape
        
        # Calculate latent dimensions
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        
        # SDXL output shapes matching UNet pattern:
        # Pattern: [88x88] + [88x88, 88x88, 44x44] + [44x44, 44x44, 22x22] + [22x22, 22x22]
        sdxl_output_shapes = {
            # Initial sample (no downsampling)
            'down_block_00': (batch_size, 320, latent_height, latent_width),        # 88x88
            # Block 0 residuals
            'down_block_01': (batch_size, 320, latent_height, latent_width),        # 88x88  
            'down_block_02': (batch_size, 320, latent_height, latent_width),        # 88x88
            'down_block_03': (batch_size, 320, latent_height // 2, latent_width // 2),  # 44x44 (downsampled)
            # Block 1 residuals
            'down_block_04': (batch_size, 640, latent_height // 2, latent_width // 2),  # 44x44
            'down_block_05': (batch_size, 640, latent_height // 2, latent_width // 2),  # 44x44
            'down_block_06': (batch_size, 640, latent_height // 4, latent_width // 4),  # 22x22 (downsampled)
            # Block 2 residuals  
            'down_block_07': (batch_size, 1280, latent_height // 4, latent_width // 4), # 22x22
            'down_block_08': (batch_size, 1280, latent_height // 4, latent_width // 4), # 22x22
            # Mid block
            'mid_block': (batch_size, 1280, latent_height // 4, latent_width // 4),     # 22x22
        }
        
        # Combine base inputs with SDXL outputs
        base_shapes.update(sdxl_output_shapes)
        return base_shapes
    
    def get_sample_input(self, batch_size, image_height, image_width):
        """Override to provide SDXL-specific sample tensors with correct input format"""
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        
        # SDXL ControlNet inputs (wrapper expects 7 inputs including SDXL conditioning)
        base_inputs = (
            torch.randn(batch_size, self.unet_dim, latent_height, latent_width, 
                       dtype=dtype, device=self.device),  # sample
            torch.ones(batch_size, dtype=torch.float32, device=self.device),  # timestep
            torch.randn(batch_size, self.text_maxlen, self.embedding_dim, 
                       dtype=dtype, device=self.device),  # encoder_hidden_states
            torch.randn(batch_size, self.conditioning_channels, image_height, image_width, 
                       dtype=dtype, device=self.device),  # controlnet_cond
            torch.tensor(1.0, dtype=torch.float32, device=self.device),  # conditioning_scale
            torch.randn(batch_size, 1280, dtype=dtype, device=self.device),  # text_embeds
            torch.randn(batch_size, 6, dtype=dtype, device=self.device),     # time_ids
        )
        
        return base_inputs
    
    def get_input_names(self):
        """Override to provide SDXL-specific input names"""
        return ["sample", "timestep", "encoder_hidden_states", "controlnet_cond", "conditioning_scale", "text_embeds", "time_ids"]
    
    def get_output_names(self):
        """Override to provide SDXL-specific output names that match wrapper return format"""
        return ["down_block_00", "down_block_01", "down_block_02", "down_block_03", 
                "down_block_04", "down_block_05", "down_block_06", "down_block_07", 
                "down_block_08", "mid_block"]

    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        """Get dynamic axes configuration for variable input shapes"""
        return {
            "sample": {0: "B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "B"},
            "timestep": {0: "B"},
            "controlnet_cond": {0: "B", 2: "H_ctrl", 3: "W_ctrl"},
            "text_embeds": {0: "B"},
            "time_ids": {0: "B"},
            **{f"down_block_{i:02d}": {0: "B", 2: "H", 3: "W"} for i in range(9)},
            "mid_block": {0: "B", 2: "H", 3: "W"}
        }
    
    def get_input_profile(self, batch_size, image_height, image_width, 
                         static_batch, static_shape):
        """Override to provide SDXL-specific input profiles including text_embeds and time_ids"""
        # Get base profiles from parent class
        profile = super().get_input_profile(batch_size, image_height, image_width, 
                                           static_batch, static_shape)
        
        # Add SDXL-specific input profiles with dynamic batch dimension
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        
        # conditioning_scale is a scalar (empty shape)
        profile["conditioning_scale"] = [
            (),  # min
            (),  # opt
            (),  # max
        ]
        
        # text_embeds has shape (batch, 1280)
        profile["text_embeds"] = [
            (min_batch, 1280),      # min
            (batch_size, 1280),     # opt
            (max_batch, 1280),      # max
        ]
        
        # time_ids has shape (batch, 6)
        profile["time_ids"] = [
            (min_batch, 6),         # min
            (batch_size, 6),        # opt
            (max_batch, 6),         # max
        ]
        
        return profile


def create_controlnet_model(model_type: str = "sd15", 
                           unet=None, model_path: str = "",
                           conditioning_channels: int = 3,
                           **kwargs) -> ControlNetTRT:
    """Factory function to create appropriate ControlNet TensorRT model"""
    if model_type.lower() in ["sdxl"]:
        return ControlNetSDXLTRT(unet=unet, model_path=model_path, 
                                conditioning_channels=conditioning_channels, **kwargs)
    else:
        return ControlNetTRT(conditioning_channels=conditioning_channels, **kwargs) 