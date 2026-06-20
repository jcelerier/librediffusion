from typing import Union, Tuple, Optional, Any
import torch
from PIL import Image
from .base import BasePreprocessor


class IPAdapterEmbeddingPreprocessor(BasePreprocessor):
    """
    Preprocessor that generates IPAdapter embeddings instead of spatial conditioning.
    Leverages existing preprocessing infrastructure for parallel IPAdapter embedding generation.
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "IPAdapter Embedding",
            "description": "Generates IPAdapter embeddings for style transfer and image conditioning instead of spatial control maps.",
            "parameters": {},
            "use_cases": ["Style transfer", "Image conditioning", "Semantic control", "Content-aware generation"]
        }
    
    def __init__(self, ipadapter: Any, **kwargs):
        super().__init__(**kwargs)
        self.ipadapter = ipadapter
        # Verify the ipadapter has the required method
        if not hasattr(ipadapter, 'get_image_embeds'):
            raise ValueError("IPAdapterEmbeddingPreprocessor: ipadapter must have 'get_image_embeds' method")
        
        # Create dedicated CUDA stream for IPAdapter processing to avoid TensorRT conflicts
        self._ipadapter_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
    def _process_core(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (positive_embeds, negative_embeds) instead of processed image"""
        if self._ipadapter_stream is not None:
            # Use dedicated stream to avoid TensorRT stream capture conflicts
            with torch.cuda.stream(self._ipadapter_stream):
                image_embeds, negative_embeds = self.ipadapter.get_image_embeds(images=[image])
                
            # Wait for stream completion and move tensors to default stream
            self._ipadapter_stream.synchronize()
            
            # Ensure tensors are accessible from default stream
            if hasattr(image_embeds, 'record_stream'):
                image_embeds.record_stream(torch.cuda.current_stream())
            if hasattr(negative_embeds, 'record_stream'):
                negative_embeds.record_stream(torch.cuda.current_stream())
        else:
            # Fallback for non-CUDA environments
            image_embeds, negative_embeds = self.ipadapter.get_image_embeds(images=[image])
            
        return image_embeds, negative_embeds
        
    def _process_tensor_core(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU-optimized path for tensor inputs"""
        # Convert tensor to PIL for IPAdapter processing
        pil_image = self.tensor_to_pil(tensor)
        return self._process_core(pil_image)
    
    def process(self, image: Union[Image.Image, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override base process to return embeddings tuple instead of PIL Image"""
        if isinstance(image, torch.Tensor):
            result = self._process_tensor_core(image)
        else:
            image = self.validate_input(image)
            result = self._process_core(image)
        
        return result
    
    def process_tensor(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override base process_tensor to return embeddings tuple"""
        tensor = self.validate_tensor_input(image_tensor)
        return self._process_tensor_core(tensor)
