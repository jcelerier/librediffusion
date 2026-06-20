from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


class BasePreprocessor(ABC):
    """
    Base class for ControlNet preprocessors with template method pattern
    """
    
    
    def __init__(self, normalization_context: str = 'controlnet', **kwargs):
        """
        Initialize the preprocessor
        
        Args:
            normalization_context: Context for normalization handling.
                - 'controlnet': Expects/produces [0,1] range for ControlNet conditioning
                - 'pipeline': Expects/produces [-1,1] range for pipeline image processing
                - 'latent': Works in latent space (no normalization needed)
            **kwargs: Preprocessor-specific parameters
        """
        self.params = kwargs
        self.normalization_context = normalization_context
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = kwargs.get('dtype', torch.float16)
    
    @classmethod
    def get_preprocessor_metadata(cls) -> Dict[str, Any]:
        """
        Get comprehensive metadata for this preprocessor.
        Subclasses should override this to define their specific metadata.
        
        Returns:
            Dictionary containing:
            - display_name: Human-readable name
            - description: Detailed description
            - parameters: User-configurable parameters with metadata
            - use_cases: List of common use cases
        """
        return {
            "display_name": cls.__name__.replace("Preprocessor", ""),
            "description": f"Preprocessor for {cls.__name__.replace('Preprocessor', '').lower()}",
            "parameters": {},
            "use_cases": []
        }
    
    def process(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """
        Template method - handles all common operations
        """
        image = self.validate_input(image)
        processed = self._process_core(image)
        return self._ensure_target_size(processed)
    
    def process_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Template method for GPU tensor processing
        """
        tensor = self.validate_tensor_input(image_tensor)
        processed = self._process_tensor_core(tensor)
        return self._ensure_target_size_tensor(processed)
    
    @abstractmethod
    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Subclasses implement ONLY their specific algorithm
        """
        pass
    
    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optional GPU processing (fallback to PIL if not overridden)
        """
        pil_image = self.tensor_to_pil(tensor)
        processed_pil = self._process_core(pil_image)
        return self.pil_to_tensor(processed_pil)
    
    def _ensure_target_size(self, image: Image.Image) -> Image.Image:
        """
        Centralized PIL resize logic
        """
        target_width, target_height = self.get_target_dimensions()
        if image.size != (target_width, target_height):
            return image.resize((target_width, target_height), Image.LANCZOS)
        return image
    
    def _ensure_target_size_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Centralized tensor resize logic
        """
        target_width, target_height = self.get_target_dimensions()
        current_size = tensor.shape[-2:]
        target_size = (target_height, target_width)
        
        if current_size != target_size:
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            tensor = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
        return tensor
    
    def validate_tensor_input(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Validate and normalize tensor input for processing
        
        Args:
            image_tensor: Input tensor
            
        Returns:
            Tensor in CHW format, on correct device
            Range: [0,1] if input was [0,255], otherwise preserves input range
            
        Note: This preserves [-1,1] tensors (from pipeline) since max() <= 1.0
        """
        # Handle batch dimension
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]  # Take first image from batch
        
        # Convert to CHW format if needed
        if image_tensor.dim() == 3 and image_tensor.shape[0] not in [1, 3]:
            # Likely HWC format, convert to CHW
            image_tensor = image_tensor.permute(2, 0, 1)
        
        # Ensure correct device and dtype
        image_tensor = image_tensor.to(device=self.device, dtype=self.dtype)
        
        # Normalize to [0,1] range only if tensor is in [0,255] uint8 range
        # Preserves [-1,1] and [0,1] ranges (max <= 1.0)
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        return image_tensor
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert tensor to PIL Image (minimize CPU transfers)
        
        Args:
            tensor: Input tensor
            
        Returns:
            PIL Image
        """
        # Ensure tensor is in correct format
        if tensor.dim() == 4:
            tensor = tensor[0]
        if tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
            tensor = tensor.permute(1, 2, 0)
        
        # Convert to numpy (unavoidable for PIL)
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Convert to uint8
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
        else:
            tensor = tensor.clamp(0, 255).to(torch.uint8)
        
        array = tensor.numpy()
        
        if array.shape[-1] == 3:
            return Image.fromarray(array, 'RGB')
        elif array.shape[-1] == 1:
            return Image.fromarray(array.squeeze(-1), 'L')
        else:
            return Image.fromarray(array)
    
    def pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        Convert PIL Image to tensor on GPU
        
        Args:
            image: PIL Image
            
        Returns:
            Tensor on correct device
        """
        # Convert to numpy first
        array = np.array(image)
        
        # Convert to tensor
        if len(array.shape) == 2:  # Grayscale
            tensor = torch.from_numpy(array).float() / 255.0
            tensor = tensor.unsqueeze(0)  # Add channel dimension
        else:  # RGB
            tensor = torch.from_numpy(array).float() / 255.0
            tensor = tensor.permute(2, 0, 1)  # HWC to CHW
        
        # Move to device
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def validate_input(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """
        Convert input to PIL Image for processing
        
        Args:
            image: Input image in various formats
            
        Returns:
            PIL Image
        """
        if isinstance(image, torch.Tensor):
            # Use tensor_to_pil method for better handling
            return self.tensor_to_pil(image)
                
        if isinstance(image, np.ndarray):
            # Ensure uint8 format
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Convert to PIL Image
            if len(image.shape) == 3:
                image = Image.fromarray(image, 'RGB')
            else:
                image = Image.fromarray(image, 'L')
                
        if not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        return image
    
    def get_target_dimensions(self) -> Tuple[int, int]:
        """
        Get target output dimensions (width, height)
        """
        # Check for explicit width/height parameters first
        width = self.params.get('image_width')
        height = self.params.get('image_height')
        
        if width is not None and height is not None:
            return (width, height)
        
        # Fallback to square resolution for backwards compatibility
        resolution = self.params.get('image_resolution', 512)
        return (resolution, resolution)
    
    def __call__(self, image: Union[Image.Image, np.ndarray, torch.Tensor], **kwargs) -> Image.Image:
        """
        Process an image (convenience method)
        
        Args:
            image: Input image
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Processed PIL Image
        """
        # Update parameters for this call
        params = {**self.params, **kwargs}
        
        # Store original params and update
        original_params = self.params
        self.params = params
        
        try:
            result = self.process(image)
        finally:
            # Restore original params
            self.params = original_params
            
        return result


class PipelineAwareProcessor(BasePreprocessor):
    """
    Abstract base class for processors that need access to pipeline state (previous outputs).
    
    This base class marks processors as requiring synchronous processing to avoid 
    temporal artifacts and ensures they have access to pipeline references.
    
    Usage:
        class MyProcessor(PipelineAwareProcessor):
            pass
    
    Examples:
    - FeedbackPreprocessor: Needs previous diffusion output
    - TemporalNetPreprocessor: Needs previous frame for optical flow
    """
    
    # Class attribute to mark processors as requiring sync processing
    requires_sync_processing = True
    
    def __init__(self, pipeline_ref: Any, normalization_context: str = 'controlnet', **kwargs):
        """
        Initialize pipeline-aware functionality
        
        Args:
            pipeline_ref: Reference to the StreamDiffusion pipeline instance (required)
            normalization_context: Context for normalization handling
            **kwargs: Additional parameters passed to base class
        """
        if pipeline_ref is None:
            raise ValueError(f"{self.__class__.__name__} requires a pipeline_ref")
        super().__init__(normalization_context=normalization_context, **kwargs)
        self.pipeline_ref = pipeline_ref 