import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import logging
from pathlib import Path
from typing import Any
from .base import PipelineAwareProcessor

# Try to import TensorRT dependencies
try:
    import tensorrt as trt
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import engine_from_bytes
    from collections import OrderedDict
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# Try to import torchvision for RAFT model
try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    from torchvision.utils import flow_to_image
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

logger = logging.getLogger(__name__)

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
    """TensorRT engine wrapper for RAFT optical flow inference"""
    
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.tensors = OrderedDict()
        self._cuda_stream = None

    def load(self):
        """Load TensorRT engine from file"""
        logger.info(f"TensorRTEngine.load: Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self):
        """Create execution context"""
        self.context = self.engine.create_execution_context()
        self._cuda_stream = torch.cuda.current_stream().cuda_stream

    def allocate_buffers(self, device="cuda", input_shape=None):
        """
        Allocate input/output buffers
        
        Args:
            device: Device to allocate tensors on
            input_shape: Shape for input tensors (B, C, H, W). Required for engines with dynamic shapes.
        """
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                # For dynamic shapes, use provided input_shape
                if input_shape is not None and any(dim == -1 for dim in shape):
                    shape = input_shape
                self.context.set_input_shape(name, shape)
                # Update shape after setting it
                shape = self.context.get_tensor_shape(name)
            else:
                # For output tensors, get shape after input shapes are set
                shape = self.context.get_tensor_shape(name)
            
            # Verify shape has no dynamic dimensions
            if any(dim == -1 for dim in shape):
                raise RuntimeError(
                    f"Tensor '{name}' still has dynamic dimensions {shape} after setting input shapes. "
                    f"Please provide input_shape parameter to allocate_buffers()."
                )
            
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[name] = tensor

    def infer(self, feed_dict, stream=None):
        """Run inference with optional stream parameter"""
        if stream is None:
            stream = self._cuda_stream
        
        # Check if we need to update tensor shapes for dynamic dimensions
        need_realloc = False
        for name, buf in feed_dict.items():
            if name in self.tensors:
                if self.tensors[name].shape != buf.shape:
                    need_realloc = True
                    break
        
        # Reallocate buffers if input shape changed
        if need_realloc:
            # Update input shapes
            for name, buf in feed_dict.items():
                # Check if this tensor is an input tensor
                try:
                    if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        self.context.set_input_shape(name, buf.shape)
                except:
                    # Tensor name might not be in engine, skip
                    pass
            
            # Reallocate all tensors with new shapes
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)
                shape = self.context.get_tensor_shape(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                
                tensor = torch.empty(
                    tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
                ).to(device=self.tensors[name].device)
                self.tensors[name] = tensor
            
        # Copy input data to tensors
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        # Set tensor addresses
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        
        # Execute inference
        success = self.context.execute_async_v3(stream)
        if not success:
            raise ValueError("TensorRT inference failed.")
        
        return self.tensors


class TemporalNetTensorRTPreprocessor(PipelineAwareProcessor):
    """
    TensorRT-accelerated TemporalNet preprocessor for temporal consistency using optical flow visualization.
    
    This preprocessor uses TensorRT to accelerate RAFT optical flow computation and creates a 6-channel
    control tensor by concatenating the previous input frame (RGB) with a colorized optical flow
    visualization (RGB) computed between the previous and current input frames.
    
    Output: [prev_input_RGB, flow_RGB(prev_input → current_input)]
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "TemporalNet TensorRT",
            "description": "TensorRT-accelerated optical flow visualization for temporal consistency. Outputs [prev_input_RGB, flow_RGB].",
            "parameters": {
                "engine_path": {
                    "type": "str",
                    "default": None,
                    "description": "Path to pre-built TensorRT engine file. Use compile_raft_tensorrt.py to build one."
                },
                "flow_strength": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.0, 2.0],
                    "step": 0.1,
                    "description": "Strength multiplier for optical flow visualization (1.0 = normal, higher = more pronounced flow)"
                },
                "height": {
                    "type": "int",
                    "default": 512,
                    "range": [256, 1024],
                    "step": 64,
                    "description": "Height for optical flow computation (must be within engine's height range)"
                },
                "width": {
                    "type": "int",
                    "default": 512,
                    "range": [256, 1024],
                    "step": 64,
                    "description": "Width for optical flow computation (must be within engine's width range)"
                },
                "output_format": {
                    "type": "str", 
                    "default": "concat",
                    "options": ["concat", "warped_only"],
                    "description": "Output format: 'concat' for 6-channel (prev_input+flow_RGB), 'warped_only' for 3-channel flow RGB only"
                }
            },
            "use_cases": ["High-performance video generation", "Real-time temporal consistency", "GPU-optimized motion control"]
        }
    
    def __init__(self, 
                 pipeline_ref: Any,
                 engine_path: str = None,
                 height: int = 512,
                 width: int = 512,
                 flow_strength: float = 1.0,
                 output_format: str = "concat",
                 **kwargs):
        """
        Initialize TensorRT TemporalNet preprocessor
        
        Args:
            pipeline_ref: Reference to the StreamDiffusion pipeline instance (required)
            engine_path: Path to pre-built TensorRT engine file (required). 
                        Build one using: python -m streamdiffusion.tools.compile_raft_tensorrt
            height: Height for optical flow computation (must be within engine's height range)
            width: Width for optical flow computation (must be within engine's width range)
            flow_strength: Strength multiplier for optical flow visualization
            output_format: "concat" for 6-channel [prev_input+flow_RGB], "warped_only" for 3-channel flow RGB only
            **kwargs: Additional parameters passed to BasePreprocessor
        """
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required for TemporalNet preprocessing. "
                "Install it with: pip install torchvision"
            )
        
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT and polygraphy are required for TensorRT acceleration. "
                "Install them with: python -m streamdiffusion.tools.install-tensorrt"
            )
        if engine_path is None:
            raise ValueError(
                "engine_path is required for TemporalNetTensorRTPreprocessor. "
                "Build a TensorRT engine using:\n"
                "  python -m streamdiffusion.tools.compile_raft_tensorrt --min_resolution 512x512 --max_resolution 1024x1024 --output_dir ./models/temporal_net\n"
                "Then pass the engine path to this preprocessor."
            )
        
        super().__init__(
            pipeline_ref=pipeline_ref,
            height=height,
            width=width,
            engine_path=engine_path,
            flow_strength=flow_strength,
            output_format=output_format,
            **kwargs
        )
        
        self.flow_strength = max(0.0, min(2.0, flow_strength))
        self.height = height
        self.width = width
        self._first_frame = True
        
        # Store previous input frame for flow computation
        self.prev_input = None
        
        # Engine path
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"TensorRT engine not found at: {self.engine_path}\n"
                f"Build one using:\n"
                f"  python -m streamdiffusion.tools.compile_raft_tensorrt --min_resolution {height}x{width} --max_resolution {height}x{width} --output_dir {self.engine_path.parent}"
            )
        
        # Model state
        self.trt_engine = None
        
        # Cached tensors for performance
        self._grid_cache = {}
        self._tensor_cache = {}
        
        # Load TensorRT engine
        self._load_tensorrt_engine()
    
    def _load_tensorrt_engine(self):
        """Load pre-built TensorRT engine"""
        logger.info(f"_load_tensorrt_engine: Loading TensorRT engine: {self.engine_path}")
        try:
            self.trt_engine = TensorRTEngine(str(self.engine_path))
            self.trt_engine.load()
            self.trt_engine.activate()
            
            # For dynamic shapes, provide the input shape based on image dimensions
            input_shape = (1, 3, self.height, self.width)
            self.trt_engine.allocate_buffers(device=self.device, input_shape=input_shape)
            
            logger.info(f"_load_tensorrt_engine: TensorRT engine loaded successfully from {self.engine_path}")
            logger.info(f"_load_tensorrt_engine: Using resolution: {self.height}x{self.width}")
        except Exception as e:
            logger.error(f"_load_tensorrt_engine: Failed to load TensorRT engine: {e}")
            self.trt_engine = None
            raise RuntimeError(
                f"Failed to load TensorRT engine from {self.engine_path}: {e}\n"
                f"Make sure the engine was built with a resolution range that includes {self.height}x{self.width}.\n"
                f"For example: python -m streamdiffusion.tools.compile_raft_tensorrt --min_resolution 512x512 --max_resolution 1024x1024"
            )
    

    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Process using TensorRT-accelerated optical flow warping
        
        Args:
            image: Current input image
            
        Returns:
            Warped previous frame for temporal guidance, or fallback for first frame
        """
        # Convert to tensor and use tensor processing path for efficiency
        tensor = self.pil_to_tensor(image)
        result_tensor = self._process_tensor_core(tensor)
        return self.tensor_to_pil(result_tensor)
    
    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process using TensorRT-accelerated optical flow computation (GPU-optimized path)
        
        Args:
            tensor: Current input tensor
            
        Returns:
            Concatenated tensor: [prev_input_RGB, flow_RGB] for temporal guidance
        """
        
        # Normalize input tensor
        input_tensor = tensor
        if input_tensor.max() > 1.0:
            input_tensor = input_tensor / 255.0
        
        # Ensure consistent format
        if input_tensor.dim() == 4 and input_tensor.shape[0] == 1:
            input_tensor = input_tensor[0]
        
        # Check if we have a previous input frame
        if self.prev_input is not None and not self._first_frame:
            try:
                # Compute optical flow between prev_input -> current_input
                flow_rgb_tensor = self._compute_flow_to_rgb_tensor(self.prev_input, input_tensor)
                
                # Check output format
                output_format = self.params.get('output_format', 'concat')
                if output_format == "concat":
                    # Concatenate prev_input + flow_RGB for TemporalNet2 (6 channels)
                    result_tensor = self._concatenate_frames_tensor(self.prev_input, flow_rgb_tensor)
                else:
                    # Return only flow RGB (3 channels)
                    result_tensor = flow_rgb_tensor
                
                # Ensure correct output format
                if result_tensor.dim() == 3:
                    result_tensor = result_tensor.unsqueeze(0)
                
                result = result_tensor.to(device=self.device, dtype=self.dtype)
            except Exception as e:
                logger.error(f"_process_tensor_core: TensorRT optical flow failed: {e}")
                output_format = self.params.get('output_format', 'concat')
                if output_format == "concat":
                    # Create 6-channel fallback by concatenating prev_input with itself
                    result_tensor = self._concatenate_frames_tensor(self.prev_input, self.prev_input)
                    if result_tensor.dim() == 3:
                        result_tensor = result_tensor.unsqueeze(0)
                    result = result_tensor.to(device=self.device, dtype=self.dtype)
                else:
                    # Fallback: return prev_input as 3-channel
                    result_tensor = self.prev_input
                    if result_tensor.dim() == 3:
                        result_tensor = result_tensor.unsqueeze(0)
                    result = result_tensor.to(device=self.device, dtype=self.dtype)
        else:
            # First frame or no previous input available
            self._first_frame = False
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            
            # Handle 6-channel output for first frame
            output_format = self.params.get('output_format', 'concat')
            if output_format == "concat":
                # For first frame, concatenate current frame with zeros (no flow)
                if tensor.dim() == 4 and tensor.shape[0] == 1:
                    current_tensor = tensor[0]
                else:
                    current_tensor = tensor
                
                # Create zero tensor for flow (same shape as current_tensor)
                zero_flow = torch.zeros_like(current_tensor, device=self.device, dtype=current_tensor.dtype)
                
                result_tensor = self._concatenate_frames_tensor(current_tensor, zero_flow)
                if result_tensor.dim() == 3:
                    result_tensor = result_tensor.unsqueeze(0)
                result = result_tensor.to(device=self.device, dtype=self.dtype)
            else:
                # Return zeros as 3-channel (no flow for first frame)
                if tensor.dim() == 4 and tensor.shape[0] == 1:
                    current_tensor = tensor[0]
                else:
                    current_tensor = tensor
                result_tensor = torch.zeros_like(current_tensor, device=self.device, dtype=current_tensor.dtype)
                if result_tensor.dim() == 3:
                    result_tensor = result_tensor.unsqueeze(0)
                result = result_tensor.to(device=self.device, dtype=self.dtype)
        
        # Store current input as previous for next frame
        self.prev_input = input_tensor.clone()
        
        return result
    
    def _compute_flow_to_rgb_tensor(self, prev_input_tensor: torch.Tensor, current_input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute optical flow between prev_input -> current_input and convert to RGB visualization
        
        Args:
            prev_input_tensor: Previous input frame tensor (CHW format, [0,1]) on GPU
            current_input_tensor: Current input frame tensor (CHW format, [0,1]) on GPU
            
        Returns:
            Flow visualization as RGB tensor (CHW format, [0,1]) on GPU
        """
        target_width, target_height = self.get_target_dimensions()
        
        # Convert to float32 for TensorRT processing
        prev_tensor = prev_input_tensor.to(device=self.device, dtype=torch.float32)
        current_tensor = current_input_tensor.to(device=self.device, dtype=torch.float32)
        
        # Resize for flow computation if needed (keep on GPU)
        if current_tensor.shape[-1] != self.width or current_tensor.shape[-2] != self.height:
            prev_resized = F.interpolate(
                prev_tensor.unsqueeze(0),
                size=(self.height, self.width), 
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            current_resized = F.interpolate(
                current_tensor.unsqueeze(0), 
                size=(self.height, self.width),
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        else:
            prev_resized = prev_tensor
            current_resized = current_tensor
        
        # Compute optical flow using TensorRT: prev_input -> current_input
        flow = self._compute_optical_flow_tensorrt(prev_resized, current_resized)
        
        # Apply flow strength scaling (GPU operation)
        flow_strength = self.params.get('flow_strength', 1.0)
        if flow_strength != 1.0:
            flow = flow * flow_strength
        
        # Convert flow to RGB visualization using torchvision's flow_to_image
        # flow_to_image expects (2, H, W) and returns (3, H, W) in range [0, 255]
        flow_rgb = flow_to_image(flow)  # Returns uint8 tensor [0, 255]
        
        # Convert to float [0, 1] range
        flow_rgb = flow_rgb.float() / 255.0
        
        # Resize back to target resolution if needed (keep on GPU)
        if flow_rgb.shape[-1] != target_width or flow_rgb.shape[-2] != target_height:
            flow_rgb = F.interpolate(
                flow_rgb.unsqueeze(0),
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Convert to processor's dtype only at the very end
        result = flow_rgb.to(dtype=self.dtype)
        
        return result
    
    def _compute_optical_flow_tensorrt(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """
        Compute optical flow between two frames using TensorRT-accelerated RAFT
        
        Args:
            frame1: First frame tensor (CHW format, [0,1])
            frame2: Second frame tensor (CHW format, [0,1])
            
        Returns:
            Optical flow tensor (2HW format)
        """
        
        if self.trt_engine is None:
            raise RuntimeError("_compute_optical_flow_tensorrt: TensorRT engine not loaded")
        
        # Prepare inputs for TensorRT
        frame1_batch = frame1.unsqueeze(0)
        frame2_batch = frame2.unsqueeze(0)
        
        # Apply RAFT preprocessing if available
        weights = Raft_Small_Weights.DEFAULT
        if hasattr(weights, 'transforms') and weights.transforms is not None:
            transforms = weights.transforms()
            frame1_batch, frame2_batch = transforms(frame1_batch, frame2_batch)
        
        # Run TensorRT inference
        feed_dict = {
            'frame1': frame1_batch,
            'frame2': frame2_batch
        }
        
        cuda_stream = torch.cuda.current_stream().cuda_stream
        result = self.trt_engine.infer(feed_dict, cuda_stream)
        flow = result['flow'][0]  # Remove batch dimension
        
        return flow
    

    
    def _warp_frame_tensor(self, frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp frame using optical flow with cached coordinate grids
        
        Args:
            frame: Frame to warp (CHW format)
            flow: Optical flow (2HW format)
            
        Returns:
            Warped frame tensor
        """
        H, W = frame.shape[-2:]
        
        # Use cached grid if available
        grid_key = (H, W)
        if grid_key not in self._grid_cache:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=self.device, dtype=torch.float32),
                torch.arange(W, device=self.device, dtype=torch.float32),
                indexing='ij'
            )
            self._grid_cache[grid_key] = (grid_x, grid_y)
        else:
            grid_x, grid_y = self._grid_cache[grid_key]
        
        # Apply flow to coordinates
        new_x = grid_x + flow[0]
        new_y = grid_y + flow[1]
        
        # Normalize coordinates to [-1, 1] for grid_sample
        new_x = 2.0 * new_x / (W - 1) - 1.0
        new_y = 2.0 * new_y / (H - 1) - 1.0
        
        # Create sampling grid (HW2 format for grid_sample)
        grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)
        
        # Warp frame
        warped_batch = F.grid_sample(
            frame.unsqueeze(0), 
            grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        result = warped_batch.squeeze(0)
        
        return result
    
    def _concatenate_frames(self, current_image: Image.Image, warped_image: Image.Image) -> Image.Image:
        """Concatenate current frame and warped previous frame for TemporalNet2 (6-channel input)"""
        # Convert to tensors and use tensor concatenation for consistency
        current_tensor = self.pil_to_tensor(current_image).squeeze(0)
        warped_tensor = self.pil_to_tensor(warped_image).squeeze(0)
        result_tensor = self._concatenate_frames_tensor(current_tensor, warped_tensor)
        return self.tensor_to_pil(result_tensor)
    
    def _concatenate_frames_tensor(self, current_tensor: torch.Tensor, warped_tensor: torch.Tensor) -> torch.Tensor:
        """
        Concatenate current frame and warped previous frame tensors for TemporalNet2 (6-channel input)
        
        Args:
            current_tensor: Current input frame tensor (CHW format)
            warped_tensor: Warped previous frame tensor (CHW format)
            
        Returns:
            Concatenated tensor (6CHW format)
        """
        # Ensure same size
        if current_tensor.shape != warped_tensor.shape:
            target_width, target_height = self.get_target_dimensions()
            
            if current_tensor.shape[-2:] != (target_height, target_width):
                current_tensor = F.interpolate(
                    current_tensor.unsqueeze(0),
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            if warped_tensor.shape[-2:] != (target_height, target_width):
                warped_tensor = F.interpolate(
                    warped_tensor.unsqueeze(0),
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
        
        # Concatenate along channel dimension: [current_R, current_G, current_B, warped_R, warped_G, warped_B]
        concatenated = torch.cat([current_tensor, warped_tensor], dim=0)
        
        return concatenated
    
    def reset(self):
        """
        Reset the preprocessor state (useful for new sequences)
        """
        self._first_frame = True
        self.prev_input = None
        # Clear caches to free memory
        self._grid_cache.clear()
        self._tensor_cache.clear()
        torch.cuda.empty_cache()