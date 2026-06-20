#NOTE: ported from https://github.com/yuvraj108c/ComfyUI-YoloNasPose-Tensorrt

import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from typing import Union, Optional, List, Tuple
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
    """Simplified TensorRT engine wrapper for pose estimation inference (optimized)"""
    
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.tensors = OrderedDict()
        self._cuda_stream = None  # Cache CUDA stream

    def load(self):
        """Load TensorRT engine from file"""
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
            raise ValueError("TensorRT inference failed.")
        
        return self.tensors


class PoseVisualization:
    """Pose drawing utilities ported from ComfyUI YoloNasPose node"""
    
    @staticmethod
    def draw_skeleton(image, keypoints, edge_links, edge_colors, joint_thickness=10, keypoint_radius=10):
        """Draw pose skeleton on image"""
        overlay = image.copy()
        
        # Draw edges/links between keypoints
        for (kp1, kp2), color in zip(edge_links, edge_colors):
            if kp1 < len(keypoints) and kp2 < len(keypoints):
                # Check if both keypoints are valid (confidence > threshold)
                if len(keypoints[kp1]) >= 3 and len(keypoints[kp2]) >= 3:
                    conf1, conf2 = keypoints[kp1][2], keypoints[kp2][2]
                    if conf1 > 0.5 and conf2 > 0.5:
                        p1 = (int(keypoints[kp1][0]), int(keypoints[kp1][1]))
                        p2 = (int(keypoints[kp2][0]), int(keypoints[kp2][1]))
                        cv2.line(overlay, p1, p2, color=color, thickness=joint_thickness, lineType=cv2.LINE_AA)
        
        # Draw keypoints
        for keypoint in keypoints:
            if len(keypoint) >= 3 and keypoint[2] > 0.5:  # confidence threshold
                x, y = int(keypoint[0]), int(keypoint[1])
                cv2.circle(overlay, (x, y), keypoint_radius, (0, 255, 0), -1, cv2.LINE_AA)
        
        return cv2.addWeighted(overlay, 0.75, image, 0.25, 0)

    @staticmethod
    def draw_poses(image, poses, edge_links, edge_colors, joint_thickness=10, keypoint_radius=10):
        """Draw multiple poses on image"""
        result = image.copy()
        
        for pose in poses:
            result = PoseVisualization.draw_skeleton(
                result, pose, edge_links, edge_colors, joint_thickness, keypoint_radius
            )
        
        return result


def iterate_over_batch_predictions(predictions, batch_size):
    """Process batch predictions from TensorRT output"""
    num_detections, batch_boxes, batch_scores, batch_joints = predictions
    
    for image_index in range(batch_size):
        num_detection_in_image = int(num_detections[image_index, 0])

        # Handle case where no detections are found
        if num_detection_in_image == 0:
            pred_scores = np.array([])
            pred_boxes = np.array([]).reshape(0, 4)
            pred_joints = np.array([]).reshape(0, 17, 3)
        else:
            pred_scores = batch_scores[image_index, :num_detection_in_image]
            pred_boxes = batch_boxes[image_index, :num_detection_in_image]
            pred_joints = batch_joints[image_index, :num_detection_in_image].reshape(
                (num_detection_in_image, -1, 3))

        yield image_index, pred_boxes, pred_scores, pred_joints

# precompute edge links define skeleton connections (COCO format)
edge_links = [[0, 17], [13, 15], [14, 16], [12, 14], [12, 17], [5, 6], 
                [11, 13], [7, 9], [5, 7], [17, 11], [6, 8], [8, 10], 
                [1, 3], [0, 1], [0, 2], [2, 4]]

edge_colors = [
    [255, 0, 0], [255, 85, 0], [170, 255, 0], [85, 255, 0], [85, 255, 0], 
    [85, 0, 255], [255, 170, 0], [0, 177, 58], [0, 179, 119], [179, 179, 0], 
    [0, 119, 179], [0, 179, 179], [119, 0, 179], [179, 0, 179], [178, 0, 118], [178, 0, 118]
]
def show_predictions_from_batch_format(predictions):
    """Convert predictions to pose visualization format"""
    try:
        image_index, pred_boxes, pred_scores, pred_joints = next(
            iter(iterate_over_batch_predictions(predictions, 1)))
    except Exception as e:
        raise RuntimeError(f"show_predictions_from_batch_format: Error in iterate_over_batch_predictions: {e}")


    
    # Handle case where no poses are detected
    if pred_joints.shape[0] == 0:
        return np.zeros((640, 640, 3))
    
    # Add middle joint between shoulders (keypoints 5 and 6)
    try:
        # Calculate middle joints for all poses at once
        middle_joints = (pred_joints[:, 5] + pred_joints[:, 6]) / 2
        # Add middle joint as keypoint 17 to all poses
        new_pred_joints = np.concatenate([pred_joints, middle_joints[:, np.newaxis]], axis=1)
    except Exception as e:
        raise RuntimeError(f"show_predictions_from_batch_format: Error processing poses: {e}")
    
    # Create black background for pose visualization
    black_image = np.zeros((640, 640, 3))
    
    try:
        image = PoseVisualization.draw_poses(
            image=black_image, 
            poses=new_pred_joints, 
            edge_links=edge_links, 
            edge_colors=edge_colors, 
            joint_thickness=10,
            keypoint_radius=10
        )
    except Exception as e:
        raise RuntimeError(f"show_predictions_from_batch_format: Error in pose drawing: {e}")
    
    return image


class YoloNasPoseTensorrtPreprocessor(BasePreprocessor):
    """
    YoloNas Pose TensorRT preprocessor for ControlNet
    
    Uses TensorRT-optimized YoloNas Pose model for fast pose estimation.
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Pose Detection (TensorRT)",
            "description": "Fast TensorRT-optimized pose detection using YOLO-NAS Pose model. Detects human pose keypoints with high performance.",
            "parameters": {},
            "use_cases": ["Human pose control", "Character animation", "Pose-guided generation", "Real-time pose detection"]
        }
    
    def __init__(self, 
                 engine_path: str = None,
                 detect_resolution: int = 640,
                 image_resolution: int = 512,
                 **kwargs):
        """
        Initialize TensorRT pose preprocessor
        
        Args:
            engine_path: Path to TensorRT engine file
            detect_resolution: Resolution for pose detection (should match engine input)
            image_resolution: Output image resolution
            **kwargs: Additional parameters
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT and polygraphy libraries are required for TensorRT pose preprocessing. "
                "Install them with: pip install tensorrt polygraphy"
            )
        
        super().__init__(
            engine_path=engine_path,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            **kwargs
        )
        
        self._engine = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_cuda_available = torch.cuda.is_available()
    
    @property
    def engine(self):
        """Lazy loading of the TensorRT engine"""
        if self._engine is None:
            engine_path = self.params.get('engine_path')
            if engine_path is None:
                raise ValueError(
                    "engine_path is required for TensorRT pose preprocessing. "
                    "Please provide it in the preprocessor_params config."
                )
            
            if not os.path.exists(engine_path):
                raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
            
            self._engine = TensorRTEngine(engine_path)
            self._engine.load()
            self._engine.activate()
            self._engine.allocate_buffers()
            
        return self._engine
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Apply TensorRT pose estimation to the input image
        """
        detect_resolution = self.params.get('detect_resolution', 640)
        
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        
        image_resized = F.interpolate(
            image_tensor, 
            size=(detect_resolution, detect_resolution), 
            mode='bilinear', 
            align_corners=False
        )
        
        image_resized_uint8 = (image_resized * 255.0).type(torch.uint8)
        
        if self._is_cuda_available:
            image_resized_uint8 = image_resized_uint8.cuda()
        
        cuda_stream = torch.cuda.current_stream().cuda_stream
        result = self.engine.infer({"input": image_resized_uint8}, cuda_stream)
        
        predictions = [result[key].cpu().numpy() for key in result.keys() if key != 'input']
        
        try:
            pose_image = show_predictions_from_batch_format(predictions)
        except Exception:
            # Fallback to black image on error
            pose_image = np.zeros((detect_resolution, detect_resolution, 3))
        
        pose_image = pose_image.clip(0, 255).astype(np.uint8)
        pose_image = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
        
        result = Image.fromarray(pose_image)
        
        return result
    
    def _process_tensor_core(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor directly on GPU to avoid CPU transfers
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if not image_tensor.is_cuda:
            image_tensor = image_tensor.cuda()
        
        detect_resolution = self.params.get('detect_resolution', 640)
        
        image_resized = torch.nn.functional.interpolate(
            image_tensor, size=(detect_resolution, detect_resolution), 
            mode='bilinear', align_corners=False
        )
        
        image_resized_uint8 = (image_resized * 255.0).type(torch.uint8)
        
        cuda_stream = torch.cuda.current_stream().cuda_stream
        result = self.engine.infer({"input": image_resized_uint8}, cuda_stream)
        
        predictions = [result[key].cpu().numpy() for key in result.keys() if key != 'input']
        
        try:
            pose_image = show_predictions_from_batch_format(predictions)
            pose_image = pose_image.clip(0, 255).astype(np.uint8)
            pose_image = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
            
            pose_tensor = torch.from_numpy(pose_image).float() / 255.0
            pose_tensor = pose_tensor.permute(2, 0, 1).unsqueeze(0).cuda()
            
        except Exception:
            # Fallback to black tensor on error
            pose_tensor = torch.zeros(1, 3, detect_resolution, detect_resolution).cuda()
        
        return pose_tensor 