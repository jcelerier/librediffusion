import numpy as np
from PIL import Image
from typing import Union, Optional
import time
from .base import BasePreprocessor

try:
    from controlnet_aux import LineartDetector, LineartAnimeDetector
    CONTROLNET_AUX_AVAILABLE = True
    print("LineartPreprocessor: controlnet_aux successfully imported")
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False
    raise ImportError("LineartPreprocessor: controlnet_aux is required for real-time optimization. Install with: pip install controlnet_aux")

#TODO provide gpu native lineart detection
class LineartPreprocessor(BasePreprocessor):
    """
    Real-time optimized Lineart detection preprocessor for ControlNet
    
    Extracts line art from input images using controlnet_aux line art detection models.
    Supports both realistic and anime-style line art extraction.
    Optimized for real-time performance - no fallbacks.
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Line Art Detection",
            "description": "Detects line art and sketches from input images. Good for converting photos to line drawings.",
            "parameters": {
                "coarse": {
                    "type": "bool",
                    "default": True,
                    "description": "Whether to use coarse line art detection (faster but less detailed)"
                },
                "anime_style": {
                    "type": "bool",
                    "default": False,
                    "description": "Whether to use anime-style line art detection"
                }
            },
            "use_cases": ["Sketch to image", "Line art generation", "Clean line extraction"]
        }
    
    def __init__(self, 
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 coarse: bool = True,
                 anime_style: bool = False,
                 **kwargs):
        """
        Initialize Lineart preprocessor
        
        Args:
            detect_resolution: Resolution for line art detection
            image_resolution: Output image resolution
            coarse: Whether to use coarse line art detection
            anime_style: Whether to use anime-style line art detection
            **kwargs: Additional parameters
        """
        print(f"LineartPreprocessor.__init__: Initializing with detect_resolution={detect_resolution}, image_resolution={image_resolution}, coarse={coarse}, anime_style={anime_style}")
        
        super().__init__(
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            coarse=coarse,
            anime_style=anime_style,
            **kwargs
        )
        
        self._detector = None
        print("LineartPreprocessor.__init__: Initialization complete")
    
    @property
    def detector(self):
        """Lazy loading of the line art detector - controlnet_aux only"""
        if self._detector is None:
            start_time = time.time()
            anime_style = self.params.get('anime_style', False)
            
            print(f"LineartPreprocessor.detector: Loading {'Anime' if anime_style else 'Realistic'} Lineart detector from controlnet_aux")
            
            if anime_style:
                self._detector = LineartAnimeDetector.from_pretrained('lllyasviel/Annotators')
                print("LineartPreprocessor.detector: LineartAnimeDetector loaded successfully")
            else:
                self._detector = LineartDetector.from_pretrained('lllyasviel/Annotators')
                print("LineartPreprocessor.detector: LineartDetector loaded successfully")
            
            load_time = time.time() - start_time
            print(f"LineartPreprocessor.detector: Detector loaded in {load_time:.3f}s")
            
        return self._detector
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Apply line art detection to the input image
        """
        start_time = time.time()
        print("LineartPreprocessor.process: Starting line art detection")
        
        detect_resolution = self.params.get('detect_resolution', 512)
        coarse = self.params.get('coarse', False)
        
        print(f"LineartPreprocessor.process: Using detect_resolution={detect_resolution}, coarse={coarse}")
        
        if image.size != (detect_resolution, detect_resolution):
            image_resized = image.resize((detect_resolution, detect_resolution), Image.LANCZOS)
            resize_time = time.time()
            print(f"LineartPreprocessor.process: Image resized from {image.size} to {image_resized.size} in {resize_time - start_time:.3f}s")
        else:
            image_resized = image
            print("LineartPreprocessor.process: No resizing needed")
        
        detection_start = time.time()
        print("LineartPreprocessor.process: Starting controlnet_aux line art detection")
        
        lineart_image = self.detector(
            image_resized,
            detect_resolution=detect_resolution,
            image_resolution=detect_resolution,
            coarse=coarse
        )
        
        detection_time = time.time() - detection_start
        print(f"LineartPreprocessor.process: Line art detection completed in {detection_time:.3f}s")
        
        total_time = time.time() - start_time
        print(f"LineartPreprocessor.process: Total processing time: {total_time:.3f}s")
        
        return lineart_image 