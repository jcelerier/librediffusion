import numpy as np
import torch
from PIL import Image
from typing import Union, Optional, Dict, Any
from .base import BasePreprocessor


class ExternalPreprocessor(BasePreprocessor):
    """
    External source preprocessor for client-processed control data
    
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "External",
            "description": "Allows using external preprocessing tools or custom processing pipelines.",
            "parameters": {

            },
            "use_cases": ["Custom processing", "Third-party tools integration", "Pre-processed control images"]
        }
    
    def __init__(self,
                 image_resolution: int = 512,
                 validate_input: bool = True,
                 **kwargs):
        """
        Initialize external source preprocessor
        
        Args:
            image_resolution: Target output resolution
            validate_input: Whether to validate the control image format
            **kwargs: Additional parameters
        """
        super().__init__(
            image_resolution=image_resolution,
            validate_input=validate_input,
            **kwargs
        )
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Process client-preprocessed control image
        
        Applies minimal server-side validation to control images
        that have already been processed by external sources.
        """
        # Optional validation of control image format
        if self.params.get('validate_input', True):
            image = self._validate_control_image(image)
        
        return image
    
    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor directly (optimized path for external sources)
        
        For external sources, tensor input likely comes from client WebGL/Canvas
        processing, so minimal processing needed.
        """
        return tensor
    
    def _validate_control_image(self, image: Image.Image) -> Image.Image:
        """
        Validate that the control image is in proper format
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Basic validation - check if image has content
        # (not completely black, which might indicate processing failure)
        img_array = np.array(image)
        brightness = np.mean(img_array)
        
        if brightness < 1.0:  # Very dark image, might be processing error
            print("ExternalPreprocessor._validate_control_image: Warning - control image appears very dark")
        
        return image
    
    
    def __call__(self, image: Union[Image.Image, np.ndarray, torch.Tensor], **kwargs) -> Image.Image:
        """
        Process control image (convenience method)
        """
        # Store any client metadata if provided
        client_metadata = kwargs.get('client_metadata', {})
        if client_metadata:
            source = client_metadata.get('source', 'unknown')
            control_type = client_metadata.get('type', 'unknown')
            print(f"ExternalPreprocessor: Received {control_type} control from {source}")
        
        return super().__call__(image, **kwargs) 