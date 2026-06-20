# NOTE: ported from https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt

import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import requests
from tqdm import tqdm
import hashlib
import logging
from pathlib import Path
from collections import OrderedDict

from .base import BasePreprocessor

# Try to import spandrel for model loading
try:
    from spandrel import ModelLoader
    SPANDREL_AVAILABLE = True
except ImportError:
    SPANDREL_AVAILABLE = False

# Try to import TensorRT dependencies
try:
    import tensorrt as trt
    from streamdiffusion.acceleration.tensorrt.utilities import engine_from_bytes, bytes_from_path
    TRT_AVAILABLE = True
    
    # Numpy to PyTorch dtype mapping (same as depth_tensorrt.py)
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
    
    # Handle bool type for numpy compatibility (same as depth_tensorrt.py)
    if np.version.full_version >= "1.24.0":
        numpy_to_torch_dtype_dict[np.bool_] = torch.bool
    else:
        numpy_to_torch_dtype_dict[np.bool] = torch.bool
        
except ImportError:
    TRT_AVAILABLE = False


class RealESRGANEngine:
    """TensorRT engine wrapper for RealESRGAN inference (following depth_tensorrt pattern)"""
    
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.tensors = OrderedDict()
        
        import threading
        self._inference_lock = threading.Lock()

    def load(self):
        """Load TensorRT engine from file"""
        # Ensure clean CUDA context before loading TensorRT engine
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self):
        """Create execution context"""
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, input_shape, device="cuda"):
        """Allocate input/output buffers for given input shape"""
        # Set input shape for dynamic sizing
        input_name = "input"
        self.context.set_input_shape(input_name, input_shape)
        
        # Allocate tensors for all bindings
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            # Convert numpy dtype to torch dtype
            if dtype == np.float32:
                torch_dtype = torch.float32
            elif dtype == np.float16:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            tensor = torch.empty(tuple(shape), dtype=torch_dtype, device=device)
            self.tensors[name] = tensor

    def infer(self, feed_dict, stream=None):
        """Run inference with consistent stream usage"""
        # Use provided stream or current stream context
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        
        # Copy input data to tensors
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        # Set tensor addresses
        for name, tensor in self.tensors.items():
            addr = tensor.data_ptr()
            self.context.set_tensor_address(name, addr)
        
        with self._inference_lock:
            success = self.context.execute_async_v3(stream)
            
            if not success:
                raise RuntimeError("RealESRGANEngine: TensorRT inference failed")
            
            torch.cuda.synchronize()
        
        return self.tensors

logger = logging.getLogger(__name__)

class RealESRGANProcessor(BasePreprocessor):
    """
    RealESRGAN 2x upscaling processor with automatic model download, ONNX export, and TensorRT acceleration.
    """
    
    MODEL_URL = "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth?download=true"
    
    @classmethod 
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "RealESRGAN 2x",
            "description": "High-quality 2x image upscaling using RealESRGAN with TensorRT acceleration",
            "parameters": {
                "enable_tensorrt": {
                    "type": "bool",
                    "default": True,
                    "description": "Use TensorRT acceleration for faster inference"
                },
                "force_rebuild": {
                    "type": "bool", 
                    "default": False,
                    "description": "Force rebuild TensorRT engine even if it exists"
                }
            },
            "use_cases": ["High-quality upscaling", "Real-time 2x enlargement", "Image enhancement"]
        }
    
    def __init__(self, enable_tensorrt: bool = True, force_rebuild: bool = False, **kwargs):
        super().__init__(enable_tensorrt=enable_tensorrt, force_rebuild=force_rebuild, **kwargs)
        self.enable_tensorrt = enable_tensorrt and TRT_AVAILABLE
        self.force_rebuild = force_rebuild
        self.scale_factor = 2  # RealESRGAN 2x model
        
        # Model paths
        self.models_dir = Path("models") / "realesrgan"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / "RealESRGAN_x2.pth"
        self.onnx_path = self.models_dir / "RealESRGAN_x2.onnx"
        self.engine_path = self.models_dir / f"RealESRGAN_x2_{trt.__version__ if TRT_AVAILABLE else 'notrt'}.trt"
        
        # Model state
        self.pytorch_model = None
        self._engine = None  # Lazy loading like depth processor
        
        # Thread safety for engine initialization
        import threading
        self._engine_lock = threading.Lock()
        
        # Initialize
        self._ensure_model_ready()
    
    @property
    def engine(self):
        """Lazy loading of the TensorRT engine"""
        if self._engine is None:
            if not self.engine_path.exists():
                raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
            
            self._engine = RealESRGANEngine(str(self.engine_path))
            self._engine.load()
            self._engine.activate()
            
            # Allocate buffers for standard input size (will be reallocated as needed)
            standard_shape = (1, 3, 512, 512)
            self._engine.allocate_buffers(standard_shape, device=self.device)
        
        return self._engine
    
    def _download_file(self, url: str, save_path: Path):
        """Download file with progress bar"""
        if save_path.exists():
            return
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as file, tqdm(
            desc=f"Downloading {save_path.name}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            colour='green'
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
    
    def _ensure_model_ready(self):
        """Ensure PyTorch model is downloaded and loaded"""
        # Download model if needed
        if not self.model_path.exists():
            self._download_file(self.MODEL_URL, self.model_path)
        
        # Load PyTorch model
        if self.pytorch_model is None:
            self._load_pytorch_model()
        
        # Setup TensorRT if enabled
        if self.enable_tensorrt:
            self._setup_tensorrt()
    
    def _load_pytorch_model(self):
        """Load PyTorch model from file"""
        if not SPANDREL_AVAILABLE:
            # Fallback loading without spandrel
            state_dict = torch.load(self.model_path, map_location=self.device)
            # This is a simplified approach - real implementation would need model architecture
            return
        
        model_descriptor = ModelLoader().load_from_file(str(self.model_path))
        # Don't force dtype conversion as it can cause type mismatches
        # Let the model keep its native dtype and convert inputs as needed
        self.pytorch_model = model_descriptor.model.eval().to(device=self.device)
        model_dtype = next(self.pytorch_model.parameters()).dtype
    
    def _export_to_onnx(self):
        """Export PyTorch model to ONNX format"""
        if self.onnx_path.exists() and not self.force_rebuild:
            return
        
        if self.pytorch_model is None:
            self._load_pytorch_model()
        
        if self.pytorch_model is None:
            return
        
        # Test with small input for export
        test_input = torch.randn(1, 3, 256, 256).to(self.device)
        
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        }
        
        with torch.no_grad():
            torch.onnx.export(
                self.pytorch_model,
                test_input,
                str(self.onnx_path),
                verbose=False,
                input_names=['input'],
                output_names=['output'],
                opset_version=17,
                export_params=True,
                dynamic_axes=dynamic_axes,
            )
    
    def _setup_tensorrt(self):
        """Setup TensorRT engine"""
        if not TRT_AVAILABLE:
            return
        
        # Export to ONNX first if needed
        if not self.onnx_path.exists():
            self._export_to_onnx()
        
        # Build/load TensorRT engine
        self._load_tensorrt_engine()
    
    def _load_tensorrt_engine(self):
        """Load or build TensorRT engine"""
        if self.engine_path.exists() and not self.force_rebuild:
            self._load_existing_engine()
        else:
            self._build_tensorrt_engine()
    
    def _load_existing_engine(self):
        """Load existing TensorRT engine (now handled by lazy loading property)"""
        # Engine loading is now handled by the lazy loading 'engine' property
        # This method is kept for compatibility but does nothing
        pass
    
    def _build_tensorrt_engine(self):
        """Build TensorRT engine from ONNX model"""
        if not self.onnx_path.exists():
            return
        
        try:
            # Create builder and network
            builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
            
            # Parse ONNX model
            with open(self.onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        pass
                    return
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 for better performance
            
            # Set optimization profile for dynamic shapes
            profile = builder.create_optimization_profile()
            min_shape = (1, 3, 256, 256)
            opt_shape = (1, 3, 512, 512)
            max_shape = (1, 3, 1024, 1024)
            profile.set_shape("input", min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            
            # Build engine
            engine = builder.build_serialized_network(network, config)
            
            if engine is None:
                return
            
            # Save engine
            with open(self.engine_path, 'wb') as f:
                f.write(engine)
            
            # Load the built engine
            self._load_existing_engine()
        
        except Exception as e:
            pass
    
    def _process_with_tensorrt(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process tensor using TensorRT engine (following depth_tensorrt pattern)"""
        batch_size, channels, height, width = tensor.shape
        input_shape = (batch_size, channels, height, width)
        
        # Ensure buffers are allocated for this input shape
        if not hasattr(self.engine, 'tensors') or len(self.engine.tensors) == 0:
            self.engine.allocate_buffers(input_shape, device=self.device)
        else:
            # Check if we need to reallocate for different input shape
            input_tensor_shape = self.engine.tensors.get("input", torch.empty(0)).shape
            if input_tensor_shape != input_shape:
                self.engine.allocate_buffers(input_shape, device=self.device)
        
        # Prepare input tensor
        input_tensor = tensor.contiguous()
        if input_tensor.dtype != self.engine.tensors["input"].dtype:
            input_tensor = input_tensor.to(dtype=self.engine.tensors["input"].dtype)
        
        # Use engine inference with current stream context for proper synchronization
        cuda_stream = torch.cuda.current_stream().cuda_stream
        result = self.engine.infer({"input": input_tensor}, cuda_stream)
        output_tensor = result['output']
        
        # Ensure output is properly clamped to [0, 1] range for RealESRGAN
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
        
        return output_tensor.clone()
    
    def _process_with_pytorch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process tensor using PyTorch model"""
        if self.pytorch_model is None:
            raise RuntimeError("_process_with_pytorch: PyTorch model not loaded")
        
        # Ensure model and input tensor have compatible dtypes
        model_dtype = next(self.pytorch_model.parameters()).dtype
        original_dtype = tensor.dtype
        if tensor.dtype != model_dtype:
            tensor = tensor.to(dtype=model_dtype)
        
        with torch.no_grad():
            result = self.pytorch_model(tensor)
            
            # Ensure output is properly clamped to [0, 1] range for RealESRGAN
            result = torch.clamp(result, 0.0, 1.0)
            
            # Convert result to the desired output dtype (self.dtype)
            if result.dtype != self.dtype:
                result = result.to(dtype=self.dtype)
                
            return result
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """Core processing using PIL Image"""
        # Convert to tensor for processing
        tensor = self.pil_to_tensor(image)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        # Process with available backend
        if self.enable_tensorrt and TRT_AVAILABLE and self.engine_path.exists():
            try:
                output_tensor = self._process_with_tensorrt(tensor)
            except Exception as e:
                output_tensor = self._process_with_pytorch(tensor)
        elif self.pytorch_model is not None:
            output_tensor = self._process_with_pytorch(tensor)
        else:
            # Fallback to simple upscaling if no model is available
            target_width, target_height = self.get_target_dimensions()
            return image.resize((target_width, target_height), Image.LANCZOS)
        
        # Convert back to PIL
        if output_tensor.dim() == 4:
            output_tensor = output_tensor.squeeze(0)
        
        result_image = self.tensor_to_pil(output_tensor)
        
        return result_image
    
    def _ensure_target_size(self, image: Image.Image) -> Image.Image:
        """
        Override base class method - for upscaling, we want to keep the upscaled size
        Don't resize back to original dimensions
        """
        return image
    
    def _ensure_target_size_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Override base class method - for upscaling, we want to keep the upscaled size
        Don't resize back to original dimensions
        """
        return tensor
    
    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """Core tensor processing"""
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Process with available backend
        if self.enable_tensorrt and TRT_AVAILABLE and self.engine_path.exists():
            try:
                output_tensor = self._process_with_tensorrt(tensor)
            except Exception as e:
                output_tensor = self._process_with_pytorch(tensor)
        elif self.pytorch_model is not None:
            output_tensor = self._process_with_pytorch(tensor)
        else:
            # Fallback using interpolation
            output_tensor = torch.nn.functional.interpolate(
                tensor, 
                scale_factor=self.scale_factor,
                mode='bicubic',
                align_corners=False
            )
        
        if squeeze_output:
            output_tensor = output_tensor.squeeze(0)
        
        return output_tensor
    
    def get_target_dimensions(self) -> Tuple[int, int]:
        """Get target output dimensions (width, height) - 2x upscaled"""
        width = self.params.get('image_width')
        height = self.params.get('image_height')
        
        if width is not None and height is not None:
            target_dims = (width * self.scale_factor, height * self.scale_factor)
            return target_dims
        
        # Fallback to square resolution
        resolution = self.params.get('image_resolution', 512)
        target_resolution = resolution * self.scale_factor
        target_dims = (target_resolution, target_resolution)
        return target_dims
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_engine') and self._engine is not None:
            # Cleanup dedicated stream if it exists
            if hasattr(self._engine, '_dedicated_stream'):
                torch.cuda.synchronize()
                del self._engine._dedicated_stream
            del self._engine
