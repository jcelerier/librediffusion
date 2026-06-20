import numpy as np
import torch
import cv2
from PIL import Image
from typing import Union, Optional, List, Tuple
from .base import BasePreprocessor

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class MediaPipeSegmentationPreprocessor(BasePreprocessor):
    """
    MediaPipe-based segmentation preprocessor for ControlNet
    
    Uses MediaPipe's Selfie Segmentation model to create accurate person segmentation masks.
    Outputs binary masks suitable for ControlNet conditioning.
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "MediaPipe Segmentation",
            "description": "MediaPipe-based image segmentation for precise object control using Selfie Segmentation model.",
            "parameters": {
                "model_selection": {
                    "type": "int",
                    "default": 1,
                    "range": [0, 1],
                    "description": "Model type (0=general/faster, 1=landscape/better quality)"
                },
                "threshold": {
                    "type": "float",
                    "default": 0.5,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Confidence threshold for segmentation"
                },
                "blur_radius": {
                    "type": "int",
                    "default": 0,
                    "range": [0, 20],
                    "description": "Blur radius for mask smoothing (0=no blur)"
                },
                "invert_mask": {
                    "type": "bool",
                    "default": False,
                    "description": "Whether to invert the segmentation mask"
                }
            },
            "use_cases": ["Precise object control", "Background replacement", "Person segmentation", "Mask generation"]
        }
    
    def __init__(self,
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 model_selection: int = 1,  # 0 for general model, 1 for landscape model
                 threshold: float = 0.5,
                 blur_radius: int = 0,
                 invert_mask: bool = False,
                 output_mode: str = "binary",  # "binary", "alpha", "background"
                 background_color: Tuple[int, int, int] = (0, 0, 0),
                 **kwargs):
        """
        Initialize MediaPipe segmentation preprocessor
        
        Args:
            detect_resolution: Resolution for segmentation processing
            image_resolution: Output image resolution
            model_selection: 0 for general model (faster), 1 for landscape model (better quality)
            threshold: Confidence threshold for segmentation (0.0-1.0)
            blur_radius: Blur radius for mask smoothing (0 = no blur)
            invert_mask: Whether to invert the segmentation mask
            output_mode: Output format - "binary" (white/black), "alpha" (transparent), "background" (remove background)
            background_color: Background color when using "background" mode
            **kwargs: Additional parameters
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is required for MediaPipe segmentation preprocessing. "
                "Install it with: pip install mediapipe"
            )
        
        super().__init__(
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            model_selection=model_selection,
            threshold=threshold,
            blur_radius=blur_radius,
            invert_mask=invert_mask,
            output_mode=output_mode,
            background_color=background_color,
            **kwargs
        )
        
        self._segmentor = None
        self._current_options = None
    
    @property
    def segmentor(self):
        """Lazy loading of the MediaPipe Selfie Segmentation model"""
        new_options = {
            'model_selection': self.params.get('model_selection', 1),
        }
        
        # Initialize or update segmentor if needed
        if self._segmentor is None or self._current_options != new_options:
            if self._segmentor is not None:
                self._segmentor.close()
                
            print(f"MediaPipeSegmentationPreprocessor.segmentor: Initializing MediaPipe Selfie Segmentation model")
            self._segmentor = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=new_options['model_selection']
            )
            self._current_options = new_options
            
        return self._segmentor
    
    def _apply_mask_smoothing(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply smoothing to the segmentation mask
        
        Args:
            mask: Input segmentation mask
            
        Returns:
            Smoothed mask
        """
        blur_radius = self.params.get('blur_radius', 0)
        
        if blur_radius > 0:
            # Apply Gaussian blur for smoother edges
            kernel_size = blur_radius * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        return mask
    
    def _threshold_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply threshold to segmentation mask
        
        Args:
            mask: Input segmentation mask (0.0-1.0)
            
        Returns:
            Binary mask
        """
        threshold = self.params.get('threshold', 0.5)
        invert_mask = self.params.get('invert_mask', False)
        
        # Apply threshold
        binary_mask = (mask > threshold).astype(np.uint8)
        
        # Invert if requested
        if invert_mask:
            binary_mask = 1 - binary_mask
        
        return binary_mask
    
    def _create_output_image(self, original_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Create final output image based on output mode
        
        Args:
            original_image: Original input image
            mask: Segmentation mask
            
        Returns:
            Output image
        """
        output_mode = self.params.get('output_mode', 'binary')
        
        if output_mode == 'binary':
            # Create binary black/white mask
            binary_mask = self._threshold_mask(mask)
            output = np.stack([binary_mask * 255] * 3, axis=-1)
            
        elif output_mode == 'alpha':
            # Create RGBA output with alpha channel
            if len(original_image.shape) == 3:
                alpha = (mask * 255).astype(np.uint8)
                output = np.concatenate([original_image, alpha[..., np.newaxis]], axis=-1)
            else:
                output = original_image
                
        elif output_mode == 'background':
            # Replace background with solid color
            background_color = self.params.get('background_color', (0, 0, 0))
            binary_mask = self._threshold_mask(mask)
            
            output = original_image.copy()
            # Apply background where mask is 0
            for i in range(3):
                output[..., i] = np.where(binary_mask, output[..., i], background_color[i])
                
        else:
            raise ValueError(f"Unknown output_mode: {output_mode}")
        
        return output
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Apply MediaPipe segmentation to the input image
        """
        detect_resolution = self.params.get('detect_resolution', 512)
        image_resized = image.resize((detect_resolution, detect_resolution), Image.LANCZOS)
        
        rgb_image = cv2.cvtColor(np.array(image_resized), cv2.COLOR_BGR2RGB)
        
        results = self.segmentor.process(rgb_image)
        
        if results.segmentation_mask is not None:
            mask = results.segmentation_mask
            
            mask = self._apply_mask_smoothing(mask)
            
            output_image = self._create_output_image(rgb_image, mask)
        else:
            output_mode = self.params.get('output_mode', 'binary')
            if output_mode == 'binary':
                output_image = np.zeros((detect_resolution, detect_resolution, 3), dtype=np.uint8)
            else:
                output_image = rgb_image
        
        if output_image.shape[-1] == 4:
            result_pil = Image.fromarray(output_image, 'RGBA')
        else:
            result_pil = Image.fromarray(output_image, 'RGB')
        
        return result_pil
    
    def _process_tensor_core(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor directly on GPU to avoid unnecessary CPU transfers
        """
        pil_image = self.tensor_to_pil(image_tensor)
        processed_pil = self._process_core(pil_image)
        return self.pil_to_tensor(processed_pil)
    
    def __del__(self):
        """Cleanup MediaPipe segmentor"""
        if hasattr(self, '_segmentor') and self._segmentor is not None:
            self._segmentor.close() 