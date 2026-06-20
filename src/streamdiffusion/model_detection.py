"""Comprehensive model detection for TensorRT and pipeline support"""

from typing import Dict, Tuple, Optional, Any, List

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

# Gracefully import the SD3 model class; it might not exist in older diffusers versions.
try:
    from diffusers.models.transformers.mm_dit import MMDiTTransformer2DModel
    HAS_MMDIT = True
except ImportError:
    # Create a dummy class if the import fails to prevent runtime errors.
    MMDiTTransformer2DModel = type("MMDiTTransformer2DModel", (torch.nn.Module,), {})
    HAS_MMDIT = False

import logging
logger = logging.getLogger(__name__)


def detect_model(model: torch.nn.Module, pipe: Optional[Any] = None) -> Dict[str, Any]:
    """
    Comprehensive and robust model detection using definitive architectural features.

    This function replaces heuristic-based analysis with a deterministic,
    rule-based approach by first inspecting the model's class and then its key 
    configuration parameters that define the architecture.

    Args:
        model: The model to analyze (e.g., UNet or MMDiT).
        pipe: Optional pipeline for additional context (e.g., detecting Turbo via scheduler).

    Returns:
        A dictionary with detailed information about the detected model.
    """
    model_type = "Unknown"
    is_turbo = False
    is_sdxl = False
    is_sd3 = False
    confidence = 0.0

    # 1. SD3 Detection (based on MMDiT Architecture)
    # NOTE THAT THIS IS NOT IMPLEMNTED AT THIS TIME
    if HAS_MMDIT and isinstance(model, MMDiTTransformer2DModel):
        config = model.config
        is_sd3 = True
        # Definitive fingerprints for SD3 Medium
        if config.get("in_channels") == 16 and config.get("joint_attention_dim") == 4096:
            model_type = "SD3"
            confidence = 1.0
            # Differentiating SD3 vs. SD3-Turbo from the MMDiT config alone is currently
            # speculative. A check on the pipeline's scheduler is a reasonable proxy.
            if pipe and hasattr(pipe, 'scheduler'):
                scheduler_name = getattr(pipe.scheduler.config, '_class_name', '').lower()
                if 'lcm' in scheduler_name or 'turbo' in scheduler_name:
                    is_turbo = True
                    model_type = "SD3-Turbo"
        else:
            model_type = "Unknown MMDiT"
            confidence = 0.6

    # 2. UNet-based Model Detection (SDXL, SD2.1, SD1.5)
    elif isinstance(model, UNet2DConditionModel):
        config = model.config
        
        # 2a. SDXL vs. non-SDXL
        # The `addition_embed_type` is the clearest indicator for the SDXL architecture.
        if config.get("addition_embed_type") is not None:
            model_type = "SDXL"
            is_sdxl = True
            confidence = 1.0
            # Differentiate SDXL-Base from SDXL-Turbo.
            # Base SDXL has `time_cond_proj_dim` (e.g., 256), while Turbo has it set to `None`.
            if config.get("time_cond_proj_dim") is None:
                is_turbo = True
        
        # 2b. SD2.1 vs. SD1.5 (if not SDXL)
        # Differentiate based on the text encoder's projection dimension.
        else:
            cross_attention_dim = config.get("cross_attention_dim")
            if cross_attention_dim == 1024:
                model_type = "SD2.1"
                confidence = 1.0
            elif cross_attention_dim == 768:
                model_type = "SD1.5"
                confidence = 1.0
            else:
                # Fallback for fine-tunes with non-standard dimensions.
                model_type = "SD-finetune"
                confidence = 0.7

    # 3. ControlNet Model Detection (detect underlying architecture)
    elif hasattr(model, 'config') and hasattr(model.config, 'cross_attention_dim'):
        # ControlNet models have UNet-like configs, detect their base architecture
        config = model.config
        
        # Apply same detection logic as UNet models
        if config.get("addition_embed_type") is not None:
            model_type = "SDXL"
            is_sdxl = True
            confidence = 0.95  # Slightly lower confidence for ControlNet
            if config.get("time_cond_proj_dim") is None:
                is_turbo = True
        else:
            cross_attention_dim = config.get("cross_attention_dim")
            if cross_attention_dim == 1024:
                model_type = "SD2.1"
                confidence = 0.95
            elif cross_attention_dim == 768:
                model_type = "SD1.5" 
                confidence = 0.95
            else:
                model_type = "SD-finetune"
                confidence = 0.7
    
    else:
        # The model is not a known UNet or MMDiT class.
        confidence = 0.0
        model_type = f"Unknown ({model.__class__.__name__})"

    # Populate architecture and compatibility details (can be expanded as needed)
    architecture_details = {
        'model_class': model.__class__.__name__,
        'in_channels': getattr(model.config, 'in_channels', 'N/A'),
        'cross_attention_dim': getattr(model.config, 'cross_attention_dim', 'N/A'),
        'block_out_channels': getattr(model.config, 'block_out_channels', 'N/A'),
    }
    
    # For UNet models, add detailed characteristics that SDXL code expects
    if isinstance(model, UNet2DConditionModel):
        unet_chars = detect_unet_characteristics(model)
        architecture_details.update({
            'has_time_conditioning': unet_chars['has_time_cond'],
            'has_addition_embeds': unet_chars['has_addition_embed'],
        })
    
    # For ControlNet models, add similar characteristics
    elif hasattr(model, 'config') and hasattr(model.config, 'cross_attention_dim'):
        # ControlNet models have similar config structure to UNet
        config = model.config
        has_addition_embed = config.get("addition_embed_type") is not None
        has_time_cond = hasattr(config, 'time_cond_proj_dim') and config.time_cond_proj_dim is not None
        
        architecture_details.update({
            'has_time_conditioning': has_time_cond,
            'has_addition_embeds': has_addition_embed,
        })

    compatibility_info = {
        'notes': f"Detected as {model_type} with {confidence:.2f} confidence based on architecture."
    }

    result = {
        'model_type': model_type,
        'is_turbo': is_turbo,
        'is_sdxl': is_sdxl,
        'is_sd3': is_sd3,
        'confidence': confidence,
        'architecture_details': architecture_details,
        'compatibility_info': compatibility_info,
    }

    return result


def detect_unet_characteristics(unet: UNet2DConditionModel) -> Dict[str, any]:
    """Detect detailed UNet characteristics including SDXL-specific features"""
    config = unet.config
    
    # Get cross attention dimensions to detect model type
    cross_attention_dim = getattr(config, 'cross_attention_dim', None)
    
    # Detect SDXL by multiple indicators
    is_sdxl = False
    
    # Check cross attention dimension
    if isinstance(cross_attention_dim, (list, tuple)):
        # SDXL typically has [1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280]
        is_sdxl = any(dim >= 1280 for dim in cross_attention_dim)
    elif isinstance(cross_attention_dim, int):
        # Single value - SDXL uses 2048 for concatenated embeddings, or 1280+ for individual encoders
        is_sdxl = cross_attention_dim >= 1280
    
    # Check addition_embed_type for SDXL detection (strong indicator)
    addition_embed_type = getattr(config, 'addition_embed_type', None)
    has_addition_embed = addition_embed_type is not None
    
    if addition_embed_type in ['text_time', 'text_time_guidance']:
        is_sdxl = True  # This is a definitive SDXL indicator
    
    # Check if model has time conditioning projection (SDXL feature)
    has_time_cond = hasattr(config, 'time_cond_proj_dim') and config.time_cond_proj_dim is not None
    
    # Additional SDXL detection checks
    if hasattr(config, 'num_class_embeds') and config.num_class_embeds is not None:
        is_sdxl = True  # SDXL often has class embeddings
        
    # Check sample size (SDXL typically uses 128 vs 64 for SD1.5)
    sample_size = getattr(config, 'sample_size', 64)
    if sample_size >= 128:
        is_sdxl = True
    
    return {
        'is_sdxl': is_sdxl,
        'has_time_cond': has_time_cond, 
        'has_addition_embed': has_addition_embed,
        'cross_attention_dim': cross_attention_dim,
        'addition_embed_type': addition_embed_type,
        'in_channels': getattr(config, 'in_channels', 4),
        'sample_size': getattr(config, 'sample_size', 64 if not is_sdxl else 128),
        'block_out_channels': tuple(getattr(config, 'block_out_channels', [])),
        'attention_head_dim': getattr(config, 'attention_head_dim', None)
    }

# This is used for controlnet/ipadapter model detection - can be deprecated (along with detect_unet_characteristics)
def detect_model_from_diffusers_unet(unet: UNet2DConditionModel) -> str:
    """Detect model type from diffusers UNet configuration"""
    characteristics = detect_unet_characteristics(unet)
    
    in_channels = characteristics['in_channels']
    block_out_channels = characteristics['block_out_channels']
    cross_attention_dim = characteristics['cross_attention_dim']
    is_sdxl = characteristics['is_sdxl']
    
    # Use enhanced SDXL detection
    if is_sdxl:
        return "SDXL"
    
    # Original detection logic for other models
    if (cross_attention_dim == 768 and 
        block_out_channels == (320, 640, 1280, 1280) and
        in_channels == 4):
        return "SD15"
    
    elif (cross_attention_dim == 1024 and 
          block_out_channels == (320, 640, 1280, 1280) and
          in_channels == 4):
        return "SD21"
    
    elif cross_attention_dim == 768 and in_channels == 4:
        return "SD15"
    elif cross_attention_dim == 1024 and in_channels == 4:
        return "SD21"
    
    if cross_attention_dim == 768:
        print(f"detect_model_from_diffusers_unet: Unknown SD1.5-like model with channels {block_out_channels}, defaulting to SD15")
        return "SD15"
    elif cross_attention_dim == 1024:
        print(f"detect_model_from_diffusers_unet: Unknown SD2.1-like model with channels {block_out_channels}, defaulting to SD21")
        return "SD21"
    else:
        raise ValueError(
            f"Unknown model architecture: "
            f"cross_attention_dim={cross_attention_dim}, "
            f"block_out_channels={block_out_channels}, "
            f"in_channels={in_channels}"
        )


def extract_unet_architecture(unet: UNet2DConditionModel) -> Dict[str, Any]:
    """
    Extract UNet architecture details needed for TensorRT engine building.
    
    This function provides the essential architecture information needed
    for TensorRT engine compilation in a clean, structured format.
    
    Args:
        unet: The UNet model to analyze
        
    Returns:
        Dict with architecture parameters for TensorRT engine building
    """
    config = unet.config
    
    # Basic model parameters
    model_channels = config.block_out_channels[0] if config.block_out_channels else 320
    block_out_channels = tuple(config.block_out_channels)
    channel_mult = tuple(ch // model_channels for ch in block_out_channels)
    
    # Resolution blocks
    if hasattr(config, 'layers_per_block'):
        if isinstance(config.layers_per_block, (list, tuple)):
            num_res_blocks = tuple(config.layers_per_block)
        else:
            num_res_blocks = tuple([config.layers_per_block] * len(block_out_channels))
    else:
        num_res_blocks = tuple([2] * len(block_out_channels))
    
    # Attention and context dimensions
    context_dim = config.cross_attention_dim
    in_channels = config.in_channels
    
    # Attention head configuration
    attention_head_dim = getattr(config, 'attention_head_dim', 8)
    if isinstance(attention_head_dim, (list, tuple)):
        attention_head_dim = attention_head_dim[0]
    
    # Transformer depth
    transformer_depth = getattr(config, 'transformer_layers_per_block', 1)
    if isinstance(transformer_depth, (list, tuple)):
        transformer_depth = tuple(transformer_depth)
    else:
        transformer_depth = tuple([transformer_depth] * len(block_out_channels))
    
    # Time embedding
    time_embed_dim = getattr(config, 'time_embedding_dim', None)
    if time_embed_dim is None:
        time_embed_dim = model_channels * 4
    
    # Build architecture dictionary
    architecture_dict = {
        "model_channels": model_channels,
        "in_channels": in_channels,
        "out_channels": getattr(config, 'out_channels', in_channels),
        "num_res_blocks": num_res_blocks,
        "channel_mult": channel_mult,
        "context_dim": context_dim,
        "attention_head_dim": attention_head_dim,
        "transformer_depth": transformer_depth,
        "time_embed_dim": time_embed_dim,
        "block_out_channels": block_out_channels,
        
        # Additional configuration
        "use_linear_in_transformer": getattr(config, 'use_linear_in_transformer', False),
        "conv_in_kernel": getattr(config, 'conv_in_kernel', 3),
        "conv_out_kernel": getattr(config, 'conv_out_kernel', 3),
        "resnet_time_scale_shift": getattr(config, 'resnet_time_scale_shift', 'default'),
        "class_embed_type": getattr(config, 'class_embed_type', None),
        "num_class_embeds": getattr(config, 'num_class_embeds', None),
        
        # Block types
        "down_block_types": getattr(config, 'down_block_types', []),
        "up_block_types": getattr(config, 'up_block_types', []),
    }
    
    return architecture_dict


def validate_architecture(arch_dict: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """
    Validate and fix architecture dictionary using model type presets.
    
    Ensures that all required architecture parameters are present and
    have reasonable values for the specified model type.
    
    Args:
        arch_dict: Architecture dictionary to validate
        model_type: Expected model type for validation
        
    Returns:
        Validated and corrected architecture dictionary
    """
    
    # Check for required keys
    required_keys = [
        "model_channels", "channel_mult", "num_res_blocks", 
        "context_dim", "in_channels", "block_out_channels"
    ]
    
    for key in required_keys:
        if key not in arch_dict:
            raise ValueError(f"Missing required architecture parameter: {key}")
    
    # Ensure tuple format for sequence parameters
    for key in ["channel_mult", "num_res_blocks", "transformer_depth", "block_out_channels"]:
        if key in arch_dict and not isinstance(arch_dict[key], tuple):
            if isinstance(arch_dict[key], (list, int)):
                if isinstance(arch_dict[key], int):
                    arch_dict[key] = tuple([arch_dict[key]] * len(arch_dict["channel_mult"]))
                else:
                    arch_dict[key] = tuple(arch_dict[key])
            else:
                arch_dict[key] = preset[key]
    
    # Validate sequence lengths match
    expected_levels = len(arch_dict["channel_mult"])
    for key in ["num_res_blocks", "transformer_depth"]:
        if key in arch_dict and len(arch_dict[key]) != expected_levels:
            arch_dict[key] = preset[key]
    
    return arch_dict

