import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union
from .base import BasePreprocessor


class BlurPreprocessor(BasePreprocessor):
    """
    Gaussian blur preprocessor for ControlNet
    
    Applies Gaussian blur to the input image using GPU-accelerated operations.
    Useful for creating soft, dreamy effects or reducing image detail.
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Gaussian Blur",
            "description": "Applies Gaussian blur to the input image using GPU-accelerated operations. Useful for creating soft, dreamy effects or reducing image detail.",
            "parameters": {
                "blur_intensity": {
                    "type": "float",
                    "default": 2.0,
                    "range": [0.1, 10.0],
                    "description": "Intensity of the blur effect. Higher values create stronger blur."
                },
                "kernel_size": {
                    "type": "int",
                    "default": 15,
                    "range": [3, 51],
                    "description": "Size of the blur kernel. Must be odd. Larger values create smoother blur."
                }
            },
            "use_cases": ["Soft focus effects", "Background blur", "Artistic rendering", "Detail reduction"]
        }
    
    def __init__(self, blur_intensity: float = 2.0, kernel_size: int = 15, **kwargs):
        """
        Initialize Blur preprocessor
        
        Args:
            blur_intensity: Standard deviation for Gaussian kernel (higher = more blur)
            kernel_size: Size of the blur kernel (must be odd)
            **kwargs: Additional parameters
        """
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        super().__init__(
            blur_intensity=blur_intensity,
            kernel_size=kernel_size,
            **kwargs
        )
        
        # Cache the Gaussian kernel for efficiency
        self._cached_kernel = None
        self._cached_kernel_size = None
        self._cached_intensity = None
    
    def _create_gaussian_kernel(self, kernel_size: int, intensity: float) -> torch.Tensor:
        """
        Create a 2D Gaussian kernel for blurring
        
        Args:
            kernel_size: Size of the kernel (must be odd)
            intensity: Standard deviation of the Gaussian
            
        Returns:
            2D Gaussian kernel tensor
        """
        # Create coordinate grids
        coords = torch.arange(kernel_size, dtype=self.dtype, device=self.device)
        coords = coords - (kernel_size - 1) / 2
        
        # Create 2D coordinate grids
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')
        
        # Calculate Gaussian values
        gaussian = torch.exp(-(x_grid**2 + y_grid**2) / (2 * intensity**2))
        
        # Normalize to sum to 1
        gaussian = gaussian / gaussian.sum()
        
        return gaussian
    
    def _get_gaussian_kernel(self, kernel_size: int, intensity: float) -> torch.Tensor:
        """
        Get cached Gaussian kernel or create new one
        """
        if (self._cached_kernel is None or 
            self._cached_kernel_size != kernel_size or 
            self._cached_intensity != intensity):
            
            self._cached_kernel = self._create_gaussian_kernel(kernel_size, intensity)
            self._cached_kernel_size = kernel_size
            self._cached_intensity = intensity
            
        return self._cached_kernel
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Apply Gaussian blur to the input image using PIL/numpy fallback
        """
        # Convert to tensor for processing
        tensor = self.pil_to_tensor(image)
        tensor = tensor.squeeze(0)  # Remove batch dimension
        
        # Process on GPU
        blurred = self._process_tensor_core(tensor)
        
        # Convert back to PIL
        return self.tensor_to_pil(blurred)
    
    def _process_tensor_core(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor directly on GPU for Gaussian blur
        """
        blur_intensity = self.params.get('blur_intensity', 2.0)
        kernel_size = self.params.get('kernel_size', 15)
        
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Get the Gaussian kernel
        kernel = self._get_gaussian_kernel(kernel_size, blur_intensity)
        
        # Ensure tensor has batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Ensure tensor is on the correct device and dtype
        image_tensor = image_tensor.to(device=self.device, dtype=self.dtype)
        
        # Reshape kernel for conv2d: (out_channels, in_channels/groups, H, W)
        # We'll apply the same kernel to each channel separately
        num_channels = image_tensor.shape[1]
        kernel_conv = kernel.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1)
        
        # Apply Gaussian blur using conv2d with groups=num_channels for per-channel convolution
        padding = kernel_size // 2
        blurred = F.conv2d(
            image_tensor,
            kernel_conv,
            padding=padding,
            groups=num_channels  # Apply kernel separately to each channel
        )
        
        return blurred
