import torch
import numpy as np
from PIL import Image
from typing import Union, Optional
from .base import BasePreprocessor

try:
    from controlnet_aux import HEDdetector
    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False
    print("HEDPreprocessor: controlnet_aux not available - please install it with: pip install controlnet_aux")


class HEDPreprocessor(BasePreprocessor):
    """
    HED (Holistically-Nested Edge Detection) preprocessor
    
    Uses controlnet_aux HEDdetector for high-quality edge detection.
    """
    
    _model_cache = {}
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "HED Edge Detection",
            "description": "Holistically-Nested Edge Detection for clean, structured edge maps.",
            "parameters": {
                "safe": {
                    "type": "bool",
                    "default": True,
                    "description": "Whether to use safe mode for edge detection"
                }
            },
            "use_cases": ["Structured edge detection", "Clean architectural edges", "Line art generation"]
        }
    
    def __init__(self, safe: bool = True, **kwargs):
        if not CONTROLNET_AUX_AVAILABLE:
            raise ImportError("controlnet_aux is required for HED preprocessor. Install with: pip install controlnet_aux")
        
        super().__init__(**kwargs)
        self.safe = safe
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load controlnet_aux HED model with caching"""
        cache_key = f"hed_{self.device}"
        
        if cache_key in self._model_cache:
            self.model = self._model_cache[cache_key]
            return
        
        print("HEDPreprocessor: Loading controlnet_aux HED model")
        try:
            # Initialize HED detector
            self.model = HEDdetector.from_pretrained("lllyasviel/Annotators")
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            # Cache the model
            self._model_cache[cache_key] = self.model
            print(f"HEDPreprocessor: Successfully loaded model on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load HED model: {e}")
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """Apply HED edge detection to the input image"""
        # Get target dimensions
        target_width, target_height = self.get_target_dimensions()
        
        # Process with controlnet_aux
        result = self.model(image, output_type="pil")
        
        # Ensure result is PIL Image
        if not isinstance(result, Image.Image):
            if isinstance(result, np.ndarray):
                result = Image.fromarray(result)
            else:
                raise ValueError(f"Unexpected result type: {type(result)}")
        
        # Resize to target size if needed
        if result.size != (target_width, target_height):
            result = result.resize((target_width, target_height), Image.LANCZOS)
        
        return result
    
    def _process_tensor_core(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        GPU-optimized HED processing using tensors
        
        Note: controlnet_aux doesn't support direct tensor input, so we convert to PIL and back.
        This is still reasonably fast due to optimized conversions in the base class.
        """
        # Convert tensor to PIL, process, then back to tensor
        pil_image = self.tensor_to_pil(image_tensor)
        processed_pil = self._process_core(pil_image)
        return self.pil_to_tensor(processed_pil)
    
    
    @classmethod
    def create_optimized(cls, device: str = 'cuda', dtype: torch.dtype = torch.float16, **kwargs):
        """
        Create an optimized HED preprocessor
        
        Args:
            device: Target device ('cuda' or 'cpu')
            dtype: Data type for inference
            **kwargs: Additional parameters
            
        Returns:
            Optimized HEDPreprocessor instance
        """
        return cls(
            device=device,
            dtype=dtype,
            safe=True,
            **kwargs
        ) 