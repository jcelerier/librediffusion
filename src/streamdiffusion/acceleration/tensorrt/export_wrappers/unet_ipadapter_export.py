import torch
from diffusers import UNet2DConditionModel
from typing import Optional, Dict, Any, List

from ....model_detection import detect_model, detect_model_from_diffusers_unet
from diffusers_ipadapter.ip_adapter.attention_processor import TRTIPAttnProcessor, TRTIPAttnProcessor2_0


class IPAdapterUNetExportWrapper(torch.nn.Module):
    """
    Wrapper that bakes IPAdapter attention processors into the UNet for ONNX export.
    
    This approach installs IPAdapter attention processors before ONNX export,
    allowing the specialized attention logic to be compiled into TensorRT.
    The UNet expects concatenated embeddings (text + image) as encoder_hidden_states.
    """
    
    def __init__(self, unet: UNet2DConditionModel, cross_attention_dim: int, num_tokens: int = 4, install_processors: bool = True):
        super().__init__()
        self.unet = unet
        self.num_image_tokens = num_tokens  # 4 for standard, 16 for plus
        self.cross_attention_dim = cross_attention_dim  # 768 for SD1.5, 2048 for SDXL
        self.install_processors = install_processors
        
        # Convert to float32 BEFORE installing processors (to avoid resetting them)
        self.unet = self.unet.to(dtype=torch.float32)
        
        # Track installed TRT processors
        self._ip_trt_processors: List[torch.nn.Module] = []
        self.num_ip_layers: int = 0

        # Check if IPAdapter processors are already installed (from pre-loading)
        if self._has_ipadapter_processors():
            self._ensure_processor_dtype_consistency()
        elif install_processors:
            # Install IPAdapter processors AFTER dtype conversion
            self._install_ipadapter_processors()
        else:
            print("IPAdapterUNetExportWrapper: WARNING - UNet will not have IPAdapter functionality without processors!")
    
    def _has_ipadapter_processors(self) -> bool:
        """Check if the UNet already has IPAdapter processors installed"""
        try:
            processors = self.unet.attn_processors
            for name, processor in processors.items():
                # Check for IPAdapter processor class names
                processor_class = processor.__class__.__name__
                if 'IPAttn' in processor_class or 'IPAttnProcessor' in processor_class:
                    return True
            return False
        except Exception as e:
            print(f"IPAdapterUNetExportWrapper: Error checking existing processors: {e}")
            return False
    
    def _ensure_processor_dtype_consistency(self):
        """Ensure existing IPAdapter processors have correct dtype for ONNX export"""
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            from diffusers.models.attention_processor import AttnProcessor2_0 as AttnProcessor
            IPProcClass = TRTIPAttnProcessor2_0
        else:
            from diffusers.models.attention_processor import AttnProcessor
            IPProcClass = TRTIPAttnProcessor
        try:
            processors = self.unet.attn_processors
            updated_processors = {}
            self._ip_trt_processors = []
            ip_layer_index = 0
            
            for name, processor in processors.items():
                processor_class = processor.__class__.__name__
                if 'TRTIPAttn' in processor_class:
                    # Already TRT processors: ensure dtype and record
                    proc = processor.to(dtype=torch.float32)
                    proc._scale_index = ip_layer_index
                    self._ip_trt_processors.append(proc)
                    ip_layer_index += 1
                    updated_processors[name] = proc
                elif 'IPAttn' in processor_class or 'IPAttnProcessor' in processor_class:
                    # Replace standard processors with TRT variants, preserving weights where applicable
                    hidden_size = getattr(processor, 'hidden_size', None)
                    cross_attention_dim = getattr(processor, 'cross_attention_dim', None)
                    num_tokens = getattr(processor, 'num_tokens', self.num_image_tokens)
                    proc = IPProcClass(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens)
                    # Copy IP projection weights if present
                    if hasattr(processor, 'to_k_ip') and hasattr(processor, 'to_v_ip') and hasattr(proc, 'to_k_ip'):
                        with torch.no_grad():
                            proc.to_k_ip.weight.copy_(processor.to_k_ip.weight.to(dtype=torch.float32))
                            proc.to_v_ip.weight.copy_(processor.to_v_ip.weight.to(dtype=torch.float32))
                    proc = proc.to(self.unet.device, dtype=torch.float32)
                    proc._scale_index = ip_layer_index
                    self._ip_trt_processors.append(proc)
                    ip_layer_index += 1
                    updated_processors[name] = proc
                else:
                    updated_processors[name] = AttnProcessor()
            
            # Update all processors to ensure consistency
            self.unet.set_attn_processor(updated_processors)
            self.num_ip_layers = len(self._ip_trt_processors)
                
        except Exception as e:
            print(f"IPAdapterUNetExportWrapper: Error updating processor dtypes: {e}")
            import traceback
            traceback.print_exc()
    
    def _install_ipadapter_processors(self):
        """
        Install IPAdapter attention processors that will be baked into ONNX.
        These processors handle the internal splitting and processing of concatenated embeddings.
        """
        # Import IPAdapter attention processors from installed package
        try:
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                from diffusers.models.attention_processor import AttnProcessor2_0 as AttnProcessor
                IPProcClass = TRTIPAttnProcessor2_0
            else:
                from diffusers.models.attention_processor import AttnProcessor
                IPProcClass = TRTIPAttnProcessor
            
            # Install attention processors with proper configuration
            processor_names = list(self.unet.attn_processors.keys())
            
            attn_procs = {}
            ip_layer_index = 0
            for name in processor_names:
                cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
                
                # Determine hidden_size based on processor location
                hidden_size = None
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]
                else:
                    # Fallback for any unexpected processor names
                    hidden_size = self.unet.config.block_out_channels[0]  # Use first block size as fallback
                
                if cross_attention_dim is None:
                    # Self-attention layers use standard processors
                    attn_procs[name] = AttnProcessor()
                else:
                    # Cross-attention layers use TRTIPAttn processors (runtime scale tensor)
                    proc = IPProcClass(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        num_tokens=self.num_image_tokens,
                    ).to(self.unet.device, dtype=torch.float32)
                    # record mapping index
                    proc._scale_index = ip_layer_index
                    self._ip_trt_processors.append(proc)
                    ip_layer_index += 1
                    attn_procs[name] = proc
            
            self.unet.set_attn_processor(attn_procs)
            self.num_ip_layers = len(self._ip_trt_processors)
            

            
        except Exception as e:
            print(f"IPAdapterUNetExportWrapper: ERROR - Could not install IPAdapter processors: {e}")
            print(f"IPAdapterUNetExportWrapper: Exception type: {type(e).__name__}")
            print("IPAdapterUNetExportWrapper: IPAdapter functionality will not work without processors!")
            import traceback
            traceback.print_exc()
            raise e
    
    def set_ipadapter_scale(self, ipadapter_scale: torch.Tensor) -> None:
        """Assign per-layer scale tensor to installed TRTIPAttn processors."""
        if not isinstance(ipadapter_scale, torch.Tensor):
            import logging
            logging.getLogger(__name__).error(f"IPAdapterUNetExportWrapper: ipadapter_scale wrong type: {type(ipadapter_scale)}")
            raise TypeError("ipadapter_scale must be a torch.Tensor")
        if self.num_ip_layers <= 0 or not self._ip_trt_processors:
            raise RuntimeError("No TRTIPAttn processors installed")
        if ipadapter_scale.ndim != 1 or ipadapter_scale.shape[0] != self.num_ip_layers:
            import logging
            logging.getLogger(__name__).error(f"IPAdapterUNetExportWrapper: ipadapter_scale has wrong shape {tuple(ipadapter_scale.shape)}, expected=({self.num_ip_layers},)")
            raise ValueError(f"ipadapter_scale must have shape [{self.num_ip_layers}]")

        # Ensure float32 for ONNX export stability
        scale_vec = ipadapter_scale.to(dtype=torch.float32)
        try:
            import logging
            logging.getLogger(__name__).debug(f"IPAdapterUNetExportWrapper: scale_vec min={scale_vec.min()}, max={scale_vec.max()}")
        except Exception:
            pass
        for proc in self._ip_trt_processors:
            proc._scale_tensor = scale_vec[proc._scale_index]

    def forward(self, sample, timestep, encoder_hidden_states, ipadapter_scale: torch.Tensor = None):
        """
        Forward pass with concatenated embeddings (text + image).
        
        The IPAdapter processors installed in the UNet will automatically:
        1. Split the concatenated embeddings into text and image parts
        2. Process image tokens with separate attention computation
        3. Apply scaling and blending between text and image attention
        
        Args:
            sample: Latent input tensor
            timestep: Timestep tensor  
            encoder_hidden_states: Concatenated embeddings [text_tokens + image_tokens, cross_attention_dim]
            
        Returns:
            UNet output (noise prediction)
        """
        # Validate input shapes
        batch_size, seq_len, embed_dim = encoder_hidden_states.shape
        
        # Check that we have the expected number of image tokens
        if embed_dim != self.cross_attention_dim:
            raise ValueError(f"Embedding dimension {embed_dim} doesn't match expected {self.cross_attention_dim}")
        
        # Ensure dtype consistency for ONNX export
        if encoder_hidden_states.dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.to(torch.float32)

        # Set per-layer scale tensor
        if ipadapter_scale is None:
            raise RuntimeError("IPAdapterUNetExportWrapper.forward requires ipadapter_scale tensor")
        self.set_ipadapter_scale(ipadapter_scale)
        
        # Pass concatenated embeddings to UNet with baked-in IPAdapter processors
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )


def create_ipadapter_wrapper(unet: UNet2DConditionModel, num_tokens: int = 4, install_processors: bool = True) -> IPAdapterUNetExportWrapper:
    """
    Create an IPAdapter wrapper with automatic architecture detection and baked-in processors.
    
    Handles both cases:
    1. UNet with pre-loaded IPAdapter processors (preserves existing weights)
    2. UNet without IPAdapter processors (installs new ones if install_processors=True)
    
    Args:
        unet: UNet2DConditionModel to wrap
        num_tokens: Number of image tokens (4 for standard, 16 for plus)
        install_processors: Whether to install IPAdapter processors if none exist
        
    Returns:
        IPAdapterUNetExportWrapper with baked-in IPAdapter attention processors
    """
    # Detect model architecture
    try:
        model_type = detect_model_from_diffusers_unet(unet)
        cross_attention_dim = unet.config.cross_attention_dim
        
        # Check if UNet already has IPAdapter processors installed
        existing_processors = unet.attn_processors
        has_ipadapter = any('IPAttn' in proc.__class__.__name__ or 'IPAttnProcessor' in proc.__class__.__name__ 
                           for proc in existing_processors.values())
        
        # Validate expected dimensions
        expected_dims = {
            "SD15": 768,
            "SDXL": 2048, 
            "SD21": 1024
        }
        
        expected_dim = expected_dims.get(model_type)
        
        return IPAdapterUNetExportWrapper(unet, cross_attention_dim, num_tokens, install_processors)
        
    except Exception as e:
        print(f"create_ipadapter_wrapper: Error during model detection: {e}")
        return IPAdapterUNetExportWrapper(unet, 768, num_tokens, install_processors) 