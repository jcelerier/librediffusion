import torch
import torch.nn.functional as F
from PIL import Image
from typing import Literal
from .base import BasePreprocessor


class UpscalePreprocessor(BasePreprocessor):
    """
    Image upscaling preprocessor with multiple interpolation algorithms.
    Supports bilinear, lanczos, bicubic, and nearest neighbor upscaling.
    """
    
    @classmethod 
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Upscale",
            "description": "Image upscaling with multiple interpolation algorithms for different quality/speed tradeoffs.",
            "parameters": {
                "scale_factor": {
                    "type": "float",
                    "default": 2.0,
                    "range": [1.0, 4.0],
                    "description": "Upscaling factor"
                },
                "algorithm": {
                    "type": "str",
                    "default": "bilinear",
                    "options": ["bilinear", "lanczos", "bicubic", "nearest"],
                    "description": "Interpolation algorithm: bilinear (fast), lanczos (high quality), bicubic (balanced), nearest (pixel art)"
                }
            },
            "use_cases": ["Real-time upscaling", "Image enhancement", "Resolution conversion"]
        }
    
    def __init__(self, scale_factor: float = 2.0, algorithm: Literal["bilinear", "lanczos", "bicubic", "nearest"] = "bilinear", **kwargs):
        super().__init__(scale_factor=scale_factor, algorithm=algorithm, **kwargs)
        self.scale_factor = scale_factor
        self.algorithm = algorithm
        
        # Map algorithm names to PIL and PyTorch modes
        self.pil_resample_map = {
            "bilinear": Image.BILINEAR,
            "lanczos": Image.LANCZOS,
            "bicubic": Image.BICUBIC,
            "nearest": Image.NEAREST
        }
        
        self.torch_mode_map = {
            "bilinear": "bilinear",
            "lanczos": "bicubic",  # PyTorch doesn't have lanczos, use bicubic as closest
            "bicubic": "bicubic",
            "nearest": "nearest"
        }
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """PIL-based upscaling"""
        target_width, target_height = self.get_target_dimensions()
        resample_method = self.pil_resample_map.get(self.algorithm, Image.BILINEAR)
        return image.resize((target_width, target_height), resample_method)
    
    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """Tensor-based upscaling"""
        target_width, target_height = self.get_target_dimensions()
        
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        mode = self.torch_mode_map.get(self.algorithm, "bilinear")
        
        if mode in ["bilinear", "bicubic"]:
            return F.interpolate(tensor, size=(target_height, target_width), 
                               mode=mode, align_corners=False)
        else:  # nearest
            return F.interpolate(tensor, size=(target_height, target_width), 
                               mode=mode)
    
    def get_target_dimensions(self):
        """Handle scale factor for dimensions"""
        width = self.params.get('image_width')
        height = self.params.get('image_height')
        
        if width is not None and height is not None:
            return (int(width * self.scale_factor), int(height * self.scale_factor))
        
        base_resolution = self.params.get('image_resolution', 512)
        target_resolution = int(base_resolution * self.scale_factor)
        return (target_resolution, target_resolution)



