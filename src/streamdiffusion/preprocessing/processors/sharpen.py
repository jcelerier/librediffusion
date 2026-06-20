import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union
from .base import BasePreprocessor


class SharpenPreprocessor(BasePreprocessor):
    """
    GPU-heavy image sharpening preprocessor using unsharp masking and edge enhancement
    
    Applies sophisticated sharpening using multiple Gaussian operations:
    - Multi-scale unsharp masking
    - Edge-preserving enhancement
    - Laplacian-based detail enhancement
    - All operations performed on GPU for maximum performance
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "GPU Sharpen",
            "description": "GPU-accelerated image sharpening using multi-scale unsharp masking and edge enhancement. Computationally intensive for maximum quality.",
            "parameters": {
                "sharpen_intensity": {
                    "type": "float",
                    "default": 1.5,
                    "range": [0.1, 5.0],
                    "description": "Overall sharpening intensity. Higher values create stronger effects."
                },
                "unsharp_radius": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.1, 5.0],
                    "description": "Radius for unsharp masking blur. Affects detail scale."
                },
                "edge_enhancement": {
                    "type": "float",
                    "default": 0.5,
                    "range": [0.0, 2.0],
                    "description": "Edge enhancement factor. Emphasizes image boundaries."
                },
                "detail_boost": {
                    "type": "float",
                    "default": 0.3,
                    "range": [0.0, 1.0],
                    "description": "Fine detail enhancement using Laplacian filtering."
                },
                "noise_reduction": {
                    "type": "float",
                    "default": 0.1,
                    "range": [0.0, 0.5],
                    "description": "Mild noise reduction to prevent amplification."
                },
                "multi_scale": {
                    "type": "bool",
                    "default": True,
                    "description": "Use multi-scale processing for better quality (more GPU intensive)."
                }
            },
            "use_cases": ["Detail enhancement", "Photo sharpening", "Edge definition", "Clarity improvement"]
        }
    
    def __init__(self, 
                 sharpen_intensity: float = 1.5,
                 unsharp_radius: float = 1.0,
                 edge_enhancement: float = 0.5,
                 detail_boost: float = 0.3,
                 noise_reduction: float = 0.1,
                 multi_scale: bool = True,
                 **kwargs):
        """
        Initialize Sharpen preprocessor
        
        Args:
            sharpen_intensity: Overall sharpening strength
            unsharp_radius: Blur radius for unsharp masking
            edge_enhancement: Edge enhancement factor
            detail_boost: Fine detail enhancement
            noise_reduction: Noise reduction factor
            multi_scale: Enable multi-scale processing
            **kwargs: Additional parameters
        """
        super().__init__(
            sharpen_intensity=sharpen_intensity,
            unsharp_radius=unsharp_radius,
            edge_enhancement=edge_enhancement,
            detail_boost=detail_boost,
            noise_reduction=noise_reduction,
            multi_scale=multi_scale,
            **kwargs
        )
        
        # Cache kernels for efficiency
        self._cached_gaussian_kernels = {}
        self._cached_laplacian_kernel = None
        self._cached_edge_kernels = None
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel"""
        coords = torch.arange(size, dtype=self.dtype, device=self.device)
        coords = coords - (size - 1) / 2
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')
        gaussian = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        return gaussian / gaussian.sum()
    
    def _get_gaussian_kernel(self, sigma: float) -> torch.Tensor:
        """Get cached Gaussian kernel"""
        # Calculate appropriate kernel size (6 sigma rule)
        size = max(3, int(6 * sigma + 1))
        if size % 2 == 0:
            size += 1
        
        key = (size, sigma)
        if key not in self._cached_gaussian_kernels:
            self._cached_gaussian_kernels[key] = self._create_gaussian_kernel(size, sigma)
        
        return self._cached_gaussian_kernels[key]
    
    def _create_laplacian_kernel(self) -> torch.Tensor:
        """Create Laplacian kernel for edge detection"""
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=self.dtype, device=self.device)
        return kernel
    
    def _get_laplacian_kernel(self) -> torch.Tensor:
        """Get cached Laplacian kernel"""
        if self._cached_laplacian_kernel is None:
            self._cached_laplacian_kernel = self._create_laplacian_kernel()
        return self._cached_laplacian_kernel
    
    def _create_edge_kernels(self) -> tuple:
        """Create Sobel edge detection kernels"""
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=self.dtype, device=self.device)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=self.dtype, device=self.device)
        
        return sobel_x, sobel_y
    
    def _get_edge_kernels(self) -> tuple:
        """Get cached edge kernels"""
        if self._cached_edge_kernels is None:
            self._cached_edge_kernels = self._create_edge_kernels()
        return self._cached_edge_kernels
    
    def _apply_kernel(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply convolution kernel to image"""
        num_channels = image.shape[1]
        padding = kernel.shape[-1] // 2
        
        # Expand kernel for all channels
        kernel_conv = kernel.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1)
        
        return F.conv2d(image, kernel_conv, padding=padding, groups=num_channels)
    
    def _gaussian_blur(self, image: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur"""
        kernel = self._get_gaussian_kernel(sigma)
        return self._apply_kernel(image, kernel)
    
    def _unsharp_mask(self, image: torch.Tensor, radius: float, intensity: float) -> torch.Tensor:
        """Apply unsharp masking"""
        # Create blurred version
        blurred = self._gaussian_blur(image, radius)
        
        # Create mask (original - blurred)
        mask = image - blurred
        
        # Apply sharpening
        sharpened = image + intensity * mask
        
        return torch.clamp(sharpened, 0, 1)
    
    def _edge_enhancement(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """Enhance edges using Sobel operators"""
        sobel_x, sobel_y = self._get_edge_kernels()
        
        # Calculate gradients
        grad_x = self._apply_kernel(image, sobel_x)
        grad_y = self._apply_kernel(image, sobel_y)
        
        # Calculate edge magnitude
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Enhance edges
        enhanced = image + strength * edge_magnitude
        
        return torch.clamp(enhanced, 0, 1)
    
    def _detail_enhancement(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """Enhance fine details using Laplacian"""
        laplacian = self._get_laplacian_kernel()
        
        # Apply Laplacian filter
        details = self._apply_kernel(image, laplacian)
        
        # Add details back to image
        enhanced = image + strength * details
        
        return torch.clamp(enhanced, 0, 1)
    
    def _noise_reduction_light(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """Light noise reduction using small Gaussian blur"""
        if strength <= 0:
            return image
        
        # Very light blur to reduce noise
        noise_reduced = self._gaussian_blur(image, 0.3)
        
        # Blend with original
        return (1 - strength) * image + strength * noise_reduced
    
    def _multi_scale_sharpen(self, image: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale sharpening for better quality"""
        sharpen_intensity = self.params.get('sharpen_intensity', 1.5)
        unsharp_radius = self.params.get('unsharp_radius', 1.0)
        
        # Multiple scales for better quality
        scales = [unsharp_radius * 0.5, unsharp_radius, unsharp_radius * 2.0]
        weights = [0.3, 0.5, 0.2]
        
        result = image.clone()
        
        for scale, weight in zip(scales, weights):
            # Apply unsharp mask at this scale
            sharpened_scale = self._unsharp_mask(image, scale, sharpen_intensity * weight)
            
            # Blend with result
            result = result + weight * (sharpened_scale - image)
        
        return torch.clamp(result, 0, 1)
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """Apply sharpening using PIL/numpy fallback"""
        # Convert to tensor for GPU processing
        tensor = self.pil_to_tensor(image)
        tensor = tensor.squeeze(0)  # Remove batch dimension
        
        # Process on GPU
        sharpened = self._process_tensor_core(tensor)
        
        # Convert back to PIL
        return self.tensor_to_pil(sharpened)
    
    def _process_tensor_core(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """GPU-intensive sharpening processing"""
        # Ensure batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Ensure correct device and dtype
        image_tensor = image_tensor.to(device=self.device, dtype=self.dtype)
        
        # Get parameters
        sharpen_intensity = self.params.get('sharpen_intensity', 1.5)
        unsharp_radius = self.params.get('unsharp_radius', 1.0)
        edge_enhancement = self.params.get('edge_enhancement', 0.5)
        detail_boost = self.params.get('detail_boost', 0.3)
        noise_reduction = self.params.get('noise_reduction', 0.1)
        multi_scale = self.params.get('multi_scale', True)
        
        result = image_tensor.clone()
        
        # Step 1: Light noise reduction (prevent amplification)
        if noise_reduction > 0:
            result = self._noise_reduction_light(result, noise_reduction)
        
        # Step 2: Main sharpening
        if multi_scale:
            # Multi-scale processing (more GPU intensive)
            result = self._multi_scale_sharpen(result)
        else:
            # Single-scale unsharp masking
            result = self._unsharp_mask(result, unsharp_radius, sharpen_intensity)
        
        # Step 3: Edge enhancement
        if edge_enhancement > 0:
            result = self._edge_enhancement(result, edge_enhancement)
        
        # Step 4: Fine detail enhancement
        if detail_boost > 0:
            result = self._detail_enhancement(result, detail_boost)
        
        # Final clamp to ensure valid range
        result = torch.clamp(result, 0, 1)
        
        return result
