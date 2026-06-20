"""ControlNet TensorRT Engine with PyTorch fallback"""

import torch
import logging
from typing import List, Optional, Tuple, Dict
from polygraphy import cuda

from ..utilities import Engine

# Set up logger for this module
logger = logging.getLogger(__name__)


class ControlNetModelEngine:
    """TensorRT-accelerated ControlNet inference engine"""
    
    def __init__(self, engine_path: str, stream: 'cuda.Stream', use_cuda_graph: bool = False, model_type: str = "sd15"):
        """Initialize ControlNet TensorRT engine"""
        self.engine = Engine(engine_path)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph
        self.model_type = model_type.lower()
        
        self.engine.load()
        self.engine.activate()
        
        self._input_names = None
        self._output_names = None
        
        # Pre-compute model-specific values to eliminate runtime branching
        if self.model_type in ["sdxl", "sdxl_turbo"]:
            self.max_blocks = 9
            self.down_block_configs = [
                (320, 1), (320, 1), (320, 1), (320, 2),
                (640, 2), (640, 2), (640, 4),
                (1280, 4), (1280, 4)
            ]
            self.mid_block_channels = 1280
            self.mid_downsample_factor = 4
        else:
            self.max_blocks = 12
            self.down_block_configs = [
                (320, 1), (320, 1), (320, 1), (320, 2), (640, 2), (640, 2),
                (640, 4), (1280, 4), (1280, 4), (1280, 8), (1280, 8), (1280, 8)
            ]
            self.mid_block_channels = 1280
            self.mid_downsample_factor = 8
        
        self._shape_cache = {}
    
    def _resolve_output_shapes(self, batch_size: int, latent_height: int, latent_width: int) -> Dict[str, Tuple[int, ...]]:
        """Optimized shape resolution using pre-computed configurations"""
        cache_key = (batch_size, latent_height, latent_width)
        if cache_key in self._shape_cache:
            return self._shape_cache[cache_key]
        
        output_shapes = {}
        
        # Generate down block shapes using pre-computed configs
        for i, (channels, factor) in enumerate(self.down_block_configs):
            output_name = f"down_block_{i:02d}"
            h = max(1, latent_height // factor)
            w = max(1, latent_width // factor)
            output_shapes[output_name] = (batch_size, channels, h, w)
        
        # Generate mid block shape
        mid_h = max(1, latent_height // self.mid_downsample_factor)
        mid_w = max(1, latent_width // self.mid_downsample_factor)
        output_shapes["mid_block"] = (batch_size, self.mid_block_channels, mid_h, mid_w)
        
        self._shape_cache[cache_key] = output_shapes
        return output_shapes

    def __call__(self, 
                 sample: torch.Tensor,
                 timestep: torch.Tensor, 
                 encoder_hidden_states: torch.Tensor,
                 controlnet_cond: torch.Tensor,
                 conditioning_scale: float = 1.0,
                 text_embeds: Optional[torch.Tensor] = None,
                 time_ids: Optional[torch.Tensor] = None,
                 **kwargs) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass through TensorRT ControlNet engine"""
        if timestep.dtype != torch.float32:
            timestep = timestep.float()
        
        input_dict = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "controlnet_cond": controlnet_cond,
            "conditioning_scale": torch.tensor(conditioning_scale, dtype=torch.float32, device=sample.device)
        }
        
        if text_embeds is not None:
            input_dict["text_embeds"] = text_embeds
        if time_ids is not None:
            input_dict["time_ids"] = time_ids
        

        
        shape_dict = {name: tensor.shape for name, tensor in input_dict.items()}
        
        batch_size = sample.shape[0]
        latent_height = sample.shape[2]
        latent_width = sample.shape[3]
        
        output_shapes = self._resolve_output_shapes(batch_size, latent_height, latent_width)
        shape_dict.update(output_shapes)
        
        self.engine.allocate_buffers(shape_dict=shape_dict, device=sample.device)
        
        outputs = self.engine.infer(
            input_dict,
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        
        self.stream.synchronize()
        
        down_blocks, mid_block = self._extract_controlnet_outputs(outputs)
        
        return down_blocks, mid_block
    
    def _extract_controlnet_outputs(self, outputs: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Extract and organize ControlNet outputs from engine results"""
        down_blocks = []
        
        for i in range(self.max_blocks):
            output_name = f"down_block_{i:02d}"
            if output_name in outputs:
                tensor = outputs[output_name]
                down_blocks.append(tensor)
        
        mid_block = outputs.get("mid_block")
        return down_blocks, mid_block
    




 