import cv2
import numpy as np
from PIL import Image
import torch
from typing import Union
from .base import BasePreprocessor

#TODO provide gpu native edge detection
class CannyPreprocessor(BasePreprocessor):
    """
    Canny edge detection preprocessor for ControlNet
    
    Detects edges in the input image using the Canny edge detection algorithm.
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Canny Edge Detection",
            "description": "Detects edges in the input image using the Canny edge detection algorithm. Good for line art and architectural images.",
            "parameters": {
                "low_threshold": {
                    "type": "int",
                    "default": 100,
                    "range": [1, 255],
                    "description": "Lower threshold for edge detection. Lower values detect more edges."
                },
                "high_threshold": {
                    "type": "int", 
                    "default": 200,
                    "range": [1, 255],
                    "description": "Upper threshold for edge detection. Higher values are more selective."
                }
            },
            "use_cases": ["Line art", "Architecture", "Technical drawings", "Clean edge detection"]
        }
    
    def __init__(self, low_threshold: int = 100, high_threshold: int = 200, **kwargs):
        """
        Initialize Canny preprocessor
        
        Args:
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            **kwargs: Additional parameters
        """
        super().__init__(
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            **kwargs
        )
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Apply Canny edge detection to the input image
        """
        image_np = np.array(image)
        
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        low_threshold = self.params.get('low_threshold', 100)
        high_threshold = self.params.get('high_threshold', 200)
        
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(edges_rgb)
    
    def _process_tensor_core(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor directly on GPU for Canny edge detection
        """
        if image_tensor.shape[0] == 3:
            gray_tensor = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
        else:
            gray_tensor = image_tensor[0] if image_tensor.shape[0] == 1 else image_tensor
        
        gray_cpu = gray_tensor.cpu()
        gray_np = (gray_cpu * 255).clamp(0, 255).to(torch.uint8).numpy()
        
        low_threshold = self.params.get('low_threshold', 100)
        high_threshold = self.params.get('high_threshold', 200)
        
        edges = cv2.Canny(gray_np, low_threshold, high_threshold)
        
        edges_tensor = torch.from_numpy(edges).float() / 255.0
        edges_tensor = edges_tensor.to(device=self.device, dtype=self.dtype)
        
        edges_rgb = edges_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        return edges_rgb 