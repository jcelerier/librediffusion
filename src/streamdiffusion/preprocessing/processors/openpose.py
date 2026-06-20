import numpy as np
from PIL import Image, ImageDraw
from typing import Union, Optional, List, Tuple
from .base import BasePreprocessor

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from controlnet_aux import OpenposeDetector
    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False


class OpenPosePreprocessor(BasePreprocessor):
    """
    OpenPose human pose detection preprocessor for ControlNet
    
    Detects human poses and creates stick figure representations.
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "OpenPose",
            "description": "Human pose estimation using OpenPose. Detects body keypoints and skeleton structure.",
            "parameters": {
                "include_hands": {
                    "type": "bool",
                    "default": False,
                    "description": "Whether to include hand keypoints in detection"
                },
                "include_face": {
                    "type": "bool",
                    "default": False,
                    "description": "Whether to include face keypoints in detection"
                }
            },
            "use_cases": ["Human pose control", "Dance movements", "Character poses"]
        }
    
    def __init__(self, 
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 include_hands: bool = False,
                 include_face: bool = False,
                 **kwargs):
        """
        Initialize OpenPose preprocessor
        
        Args:
            detect_resolution: Resolution for pose detection
            image_resolution: Output image resolution
            include_hands: Whether to include hand keypoints
            include_face: Whether to include face keypoints
            **kwargs: Additional parameters
        """
        super().__init__(
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            include_hands=include_hands,
            include_face=include_face,
            **kwargs
        )
        
        self._detector = None
    
    @property
    def detector(self):
        """Lazy loading of the OpenPose detector"""
        if self._detector is None:
            if CONTROLNET_AUX_AVAILABLE:
                print("Loading OpenPose detector from controlnet_aux")
                self._detector = OpenposeDetector.from_pretrained('lllyasviel/Annotators')
            else:
                print("Warning: controlnet_aux not available, using fallback OpenPose implementation")
                self._detector = self._create_fallback_detector()
        return self._detector
    
    def _create_fallback_detector(self):
        """Create a simple fallback detector if controlnet_aux is not available"""
        class FallbackDetector:
            def __call__(self, image, include_hands=False, include_face=False):
                # Simple fallback: return a blank image with some basic pose lines
                width, height = image.size
                pose_image = Image.new('RGB', (width, height), (0, 0, 0))
                draw = ImageDraw.Draw(pose_image)
                
                # Draw a basic stick figure in the center
                center_x, center_y = width // 2, height // 2
                
                # Head
                head_radius = min(width, height) // 20
                draw.ellipse([
                    center_x - head_radius, center_y - height // 4 - head_radius,
                    center_x + head_radius, center_y - height // 4 + head_radius
                ], outline=(255, 255, 255), width=2)
                
                # Body
                body_top = center_y - height // 4 + head_radius
                body_bottom = center_y + height // 6
                draw.line([center_x, body_top, center_x, body_bottom], fill=(255, 255, 255), width=2)
                
                # Arms
                arm_length = width // 6
                arm_y = body_top + (body_bottom - body_top) // 3
                draw.line([center_x - arm_length, arm_y, center_x + arm_length, arm_y], fill=(255, 255, 255), width=2)
                
                # Legs
                leg_length = height // 8
                draw.line([center_x, body_bottom, center_x - leg_length//2, body_bottom + leg_length], fill=(255, 255, 255), width=2)
                draw.line([center_x, body_bottom, center_x + leg_length//2, body_bottom + leg_length], fill=(255, 255, 255), width=2)
                
                return pose_image
        
        return FallbackDetector()
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Apply OpenPose detection to the input image
        """
        detect_resolution = self.params.get('detect_resolution', 512)
        image_resized = image.resize((detect_resolution, detect_resolution), Image.LANCZOS)
        
        include_hands = self.params.get('include_hands', False)
        include_face = self.params.get('include_face', False)
        
        if CONTROLNET_AUX_AVAILABLE and hasattr(self.detector, '__call__'):
            try:
                pose_image = self.detector(
                    image_resized,
                    hand_and_face=include_hands or include_face
                )
            except Exception as e:
                print(f"Warning: OpenPose detection failed, using fallback: {e}")
                pose_image = self._create_fallback_detector()(image_resized, include_hands, include_face)
        else:
            pose_image = self.detector(image_resized, include_hands, include_face)
        
        return pose_image 