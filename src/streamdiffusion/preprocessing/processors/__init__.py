from .base import BasePreprocessor, PipelineAwareProcessor
from typing import Any
from .canny import CannyPreprocessor
from .depth import DepthPreprocessor
from .openpose import OpenPosePreprocessor
from .lineart import LineartPreprocessor
from .standard_lineart import StandardLineartPreprocessor
from .passthrough import PassthroughPreprocessor
from .external import ExternalPreprocessor
from .soft_edge import SoftEdgePreprocessor
from .hed import HEDPreprocessor
from .ipadapter_embedding import IPAdapterEmbeddingPreprocessor
from .faceid_embedding import FaceIDEmbeddingPreprocessor
from .feedback import FeedbackPreprocessor
from .latent_feedback import LatentFeedbackPreprocessor
from .sharpen import SharpenPreprocessor
from .upscale import UpscalePreprocessor
from .blur import BlurPreprocessor
from .realesrgan_trt import RealESRGANProcessor

# Try to import TensorRT preprocessors - might not be available on all systems
try:
    from .depth_tensorrt import DepthAnythingTensorrtPreprocessor
    DEPTH_TENSORRT_AVAILABLE = True
except ImportError:
    DepthAnythingTensorrtPreprocessor = None
    DEPTH_TENSORRT_AVAILABLE = False

try:
    from .pose_tensorrt import YoloNasPoseTensorrtPreprocessor
    POSE_TENSORRT_AVAILABLE = True
except ImportError:
    YoloNasPoseTensorrtPreprocessor = None
    POSE_TENSORRT_AVAILABLE = False

try:
    from .temporal_net_tensorrt import TemporalNetTensorRTPreprocessor
    TEMPORAL_NET_TENSORRT_AVAILABLE = True
except ImportError:
    TemporalNetTensorRTPreprocessor = None
    TEMPORAL_NET_TENSORRT_AVAILABLE = False

try:
    from .mediapipe_pose import MediaPipePosePreprocessor
    MEDIAPIPE_POSE_AVAILABLE = True
except ImportError:
    MediaPipePosePreprocessor = None
    MEDIAPIPE_POSE_AVAILABLE = False

try:
    from .mediapipe_segmentation import MediaPipeSegmentationPreprocessor
    MEDIAPIPE_SEGMENTATION_AVAILABLE = True
except ImportError:
    MediaPipeSegmentationPreprocessor = None
    MEDIAPIPE_SEGMENTATION_AVAILABLE = False

# Registry for easy lookup
_preprocessor_registry = {
    "canny": CannyPreprocessor,
    "depth": DepthPreprocessor,
    "openpose": OpenPosePreprocessor,
    "lineart": LineartPreprocessor,
    "standard_lineart": StandardLineartPreprocessor,
    "passthrough": PassthroughPreprocessor,
    "external": ExternalPreprocessor,
    "soft_edge": SoftEdgePreprocessor,
    "hed": HEDPreprocessor,
    "feedback": FeedbackPreprocessor,
    "latent_feedback": LatentFeedbackPreprocessor,
    "sharpen": SharpenPreprocessor,
    "upscale": UpscalePreprocessor,
    "blur": BlurPreprocessor,
    "realesrgan_trt": RealESRGANProcessor,
}   

# Add TensorRT preprocessors if available
if DEPTH_TENSORRT_AVAILABLE:
    _preprocessor_registry["depth_tensorrt"] = DepthAnythingTensorrtPreprocessor

if POSE_TENSORRT_AVAILABLE:
    _preprocessor_registry["pose_tensorrt"] = YoloNasPoseTensorrtPreprocessor

if TEMPORAL_NET_TENSORRT_AVAILABLE:
    _preprocessor_registry["temporal_net_tensorrt"] = TemporalNetTensorRTPreprocessor

# Add MediaPipe preprocessors if available
if MEDIAPIPE_POSE_AVAILABLE:
    _preprocessor_registry["mediapipe_pose"] = MediaPipePosePreprocessor

if MEDIAPIPE_SEGMENTATION_AVAILABLE:
    _preprocessor_registry["mediapipe_segmentation"] = MediaPipeSegmentationPreprocessor


def get_preprocessor_class(name: str) -> type:
    """
    Get a preprocessor class by name
    
    Args:
        name: Name of the preprocessor
        
    Returns:
        Preprocessor class
        
    Raises:
        ValueError: If preprocessor name is not found
    """
    if name not in _preprocessor_registry:
        available = ", ".join(_preprocessor_registry.keys())
        raise ValueError(f"Unknown preprocessor '{name}'. Available: {available}")
    
    return _preprocessor_registry[name]


def get_preprocessor(name: str, pipeline_ref: Any = None, normalization_context: str = 'controlnet', params: Any = None) -> BasePreprocessor:
    """
    Get a preprocessor by name
    
    Args:
        name: Name of the preprocessor
        pipeline_ref: Pipeline reference for pipeline-aware processors (required for some processors)
        normalization_context: Context for normalization handling
            - 'controlnet': Expects/produces [0,1] range for ControlNet conditioning
            - 'pipeline': Expects/produces [-1,1] range for pipeline image processing
            - 'latent': Works in latent space (no normalization needed)
        
    Returns:
        Preprocessor instance
        
    Raises:
        ValueError: If preprocessor name is not found or pipeline_ref missing for pipeline-aware processor
    """
    processor_class = get_preprocessor_class(name)
    
    # Check if this is a pipeline-aware processor
    if hasattr(processor_class, 'requires_sync_processing') and processor_class.requires_sync_processing:
        if pipeline_ref is None:
            raise ValueError(f"Processor '{name}' requires a pipeline_ref")
        return processor_class(pipeline_ref=pipeline_ref, normalization_context=normalization_context, _registry_name=name, **(params or {}))
    else:
        return processor_class(normalization_context=normalization_context, _registry_name=name, **(params or {}))


def register_preprocessor(name: str, preprocessor_class):
    """
    Register a new preprocessor
    
    Args:
        name: Name to register under
        preprocessor_class: Preprocessor class
    """
    _preprocessor_registry[name] = preprocessor_class


def list_preprocessors():
    """List all available preprocessors"""
    return list(_preprocessor_registry.keys())


__all__ = [
    "BasePreprocessor",
    "PipelineAwareProcessor",
    "CannyPreprocessor",
    "DepthPreprocessor", 
    "OpenPosePreprocessor",
    "LineartPreprocessor",
    "StandardLineartPreprocessor",
    "PassthroughPreprocessor",
    "ExternalPreprocessor",
    "SoftEdgePreprocessor",
    "HEDPreprocessor",
    "IPAdapterEmbeddingPreprocessor",
    "FaceIDEmbeddingPreprocessor",
    "FeedbackPreprocessor",
    "LatentFeedbackPreprocessor",
    "get_preprocessor",
    "get_preprocessor_class",
    "register_preprocessor",
    "list_preprocessors",
]

if DEPTH_TENSORRT_AVAILABLE:
    __all__.append("DepthAnythingTensorrtPreprocessor")

if POSE_TENSORRT_AVAILABLE:
    __all__.append("YoloNasPoseTensorrtPreprocessor")

if TEMPORAL_NET_TENSORRT_AVAILABLE:
    __all__.append("TemporalNetTensorRTPreprocessor")

if MEDIAPIPE_POSE_AVAILABLE:
    __all__.append("MediaPipePosePreprocessor")

if MEDIAPIPE_SEGMENTATION_AVAILABLE:
    __all__.append("MediaPipeSegmentationPreprocessor") 