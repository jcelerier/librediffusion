#NOTE: ported from https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt

import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from typing import Union, Optional
from .base import BasePreprocessor

try:
    import tensorrt as trt
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import engine_from_bytes
    from collections import OrderedDict
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool


class TensorRTEngine:
    """Simplified TensorRT engine wrapper for depth estimation inference (optimized)"""
    
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.tensors = OrderedDict()
        self._cuda_stream = None  # Cache CUDA stream

    def load(self):
        """Load TensorRT engine from file"""
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self):
        """Create execution context"""
        self.context = self.engine.create_execution_context()
        # Cache CUDA stream for reuse
        self._cuda_stream = torch.cuda.current_stream().cuda_stream

    def allocate_buffers(self, device="cuda"):
        """Allocate input/output buffers"""
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[name] = tensor

    def infer(self, feed_dict, stream=None):
        """Run inference with optional stream parameter"""
        # Use cached stream if none provided
        if stream is None:
            stream = self._cuda_stream
            
        # Copy input data to tensors
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        # Set tensor addresses
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        
        # Execute inference
        success = self.context.execute_async_v3(stream)
        if not success:
            raise ValueError("ERROR: TensorRT inference failed.")
        
        return self.tensors


class DepthAnythingTensorrtPreprocessor(BasePreprocessor):
    """
    Depth Anything TensorRT preprocessor for ControlNet
    
    Uses TensorRT-optimized Depth Anything model for fast depth estimation.
    """
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Depth Estimation (TensorRT)",
            "description": "Fast TensorRT-optimized depth estimation using Depth Anything model. Significantly faster than standard depth estimation.",
            "parameters": {
               
            },
            "use_cases": ["High-performance depth estimation", "Real-time applications", "3D-aware generation"]
        }
    def __init__(self, 
                 engine_path: str = None,
                 detect_resolution: int = 518,
                 image_resolution: int = 512,
                 **kwargs):
        """
        Initialize TensorRT depth preprocessor
        
        Args:
            engine_path: Path to TensorRT engine file
            detect_resolution: Resolution for depth detection (should match engine input)
            image_resolution: Output image resolution
            **kwargs: Additional parameters
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT and polygraphy libraries are required for TensorRT depth preprocessing. "
                "Install them with: pip install tensorrt polygraphy"
            )
        
        super().__init__(
            engine_path=engine_path,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            **kwargs
        )
        
        self._engine = None
    
    @property
    def engine(self):
        """Lazy loading of the TensorRT engine"""
        if self._engine is None:
            engine_path = self.params.get('engine_path')
            if engine_path is None:
                raise ValueError(
                    "engine_path is required for TensorRT depth preprocessing. "
                    "Please provide it in the preprocessor_params config."
                )
            
            if not os.path.exists(engine_path):
                raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
            
            print(f"Loading TensorRT depth estimation engine: {engine_path}")
            
            self._engine = TensorRTEngine(engine_path)
            self._engine.load()
            self._engine.activate()
            self._engine.allocate_buffers()
            
        return self._engine
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Apply TensorRT depth estimation to the input image
        """
        detect_resolution = self.params.get('detect_resolution', 518)
        
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        
        image_resized = F.interpolate(
            image_tensor, 
            size=(detect_resolution, detect_resolution), 
            mode='bilinear', 
            align_corners=False
        )
        
        if torch.cuda.is_available():
            image_resized = image_resized.cuda()
        
        cuda_stream = torch.cuda.current_stream().cuda_stream
        result = self.engine.infer({"input": image_resized}, cuda_stream)
        depth = result['output']
        
        depth = np.reshape(depth.cpu().numpy(), (detect_resolution, detect_resolution))
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        original_size = image.size
        depth = cv2.resize(depth, original_size)
        
        depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        result = Image.fromarray(depth_rgb)
        
        return result
    
    def _process_tensor_core(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor directly on GPU to avoid CPU transfers
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if not image_tensor.is_cuda:
            image_tensor = image_tensor.cuda()
        
        detect_resolution = self.params.get('detect_resolution', 518)
        
        image_resized = torch.nn.functional.interpolate(
            image_tensor, size=(detect_resolution, detect_resolution), 
            mode='bilinear', align_corners=False
        )
        
        cuda_stream = torch.cuda.current_stream().cuda_stream
        result = self.engine.infer({"input": image_resized}, cuda_stream)
        depth_tensor = result['output']
        
        depth_tensor = depth_tensor.squeeze() if depth_tensor.dim() > 2 else depth_tensor
        depth_min, depth_max = depth_tensor.min(), depth_tensor.max()
        depth_normalized = (depth_tensor - depth_min) / (depth_max - depth_min)
        
        return depth_normalized.repeat(3, 1, 1).unsqueeze(0) 