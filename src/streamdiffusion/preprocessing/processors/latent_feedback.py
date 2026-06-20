import torch
from typing import Optional, Any
from .base import PipelineAwareProcessor


class LatentFeedbackPreprocessor(PipelineAwareProcessor):
    """
    Latent domain feedback preprocessor
    
    Creates a configurable blend between the current input latent and the previous frame's latent output.
    This creates a feedback loop in latent space where each generated latent influences the next generation,
    providing temporal consistency without the overhead of VAE encoding/decoding.
    
    Formula: output = (1 - feedback_strength) * input_latent + feedback_strength * previous_latent
    
    Examples:
    - feedback_strength = 0.0: Pure passthrough (input only)
    - feedback_strength = 0.15: Default safe blend
    - feedback_strength = 0.40: Maximum safe feedback (values > 0.4 produce garbage)
    
    The preprocessor accesses the pipeline's prev_latent_result to get the previous latent output.
    For the first frame (when no previous output exists), it falls back to the input latent.
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Latent Feedback Loop",
            "description": "Creates a configurable blend between the current input latent and the previous frame's latent output for temporal consistency in latent space.",
            "parameters": {
                "feedback_strength": {
                    "type": "float",
                    "default": 0.15,
                    "range": [0.0, 0.40],
                    "step": 0.01,
                    "description": "Strength of latent feedback blend (0.0 = pure input, .40 = more feedback)"
                }
            },
            "use_cases": ["Latent temporal consistency", "Latent space transitions", "Efficient feedback", "Latent preprocessing", "Temporal stability"]
        }
    
    def __init__(self, 
                 pipeline_ref: Any,
                 normalization_context: str = 'latent',
                 feedback_strength: float = 0.15,
                 **kwargs):
        """
        Initialize latent feedback preprocessor
        
        Args:
            pipeline_ref: Reference to the StreamDiffusion pipeline instance (required)
            normalization_context: Context for normalization handling (latent space doesn't need normalization)
            feedback_strength: Strength of feedback blend (0.0 = pure input, 0.40 = max safe value, values > 0.4 produce garbage)
            **kwargs: Additional parameters passed to BasePreprocessor
        """
        super().__init__(
            pipeline_ref=pipeline_ref,
            normalization_context=normalization_context,
            feedback_strength=feedback_strength,
            **kwargs
        )
        self.feedback_strength = max(0.0, min(0.40, feedback_strength))  # Clamp to [0, 0.40] - values > 0.4 produce garbage
        self._first_frame = True
    
    def _get_previous_data(self):
        """Get previous frame latent data from pipeline"""
        if self.pipeline_ref is not None:
            # Get previous OUTPUT latent (after diffusion), not input latent
            # Check for prev_latent_result (the actual attribute name used by the pipeline)
            if hasattr(self.pipeline_ref, 'prev_latent_result'):
                if self.pipeline_ref.prev_latent_result is not None and not self._first_frame:
                    return self.pipeline_ref.prev_latent_result
        return None
    
    #TODO: eventually, these processors should be divided by input and output domain rather than overriding image-first basec class
    def validate_tensor_input(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """
        Validate latent tensor input - preserve batch dimensions for latent processing
        
        Args:
            latent_tensor: Input latent tensor in format [B, C, H/8, W/8]
            
        Returns:
            Validated latent tensor with preserved batch dimension
        """
        # For latent processing, we want to preserve the batch dimension
        # Only ensure correct device and dtype
        latent_tensor = latent_tensor.to(device=self.device, dtype=self.dtype)
        return latent_tensor
        
    #TODO: eventually, these processors should be divided by input and output domain rather than overriding image-first basec class
    def _ensure_target_size_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Override base class resize logic - latent tensors should NOT be resized to image dimensions
        
        For latent domain processing, we want to preserve the latent space dimensions,
        not resize to image target dimensions like image-domain processors.
        """
        # For latent feedback, just return the tensor as-is without any resizing
        return tensor
    
    def _process_core(self, image):
        """
        For latent feedback, we don't process PIL images directly.
        This method should not be called in normal latent preprocessing workflows.
        """
        raise NotImplementedError(
            "LatentFeedbackPreprocessor is designed for latent domain processing. "
            "Use _process_tensor_core or process_tensor for latent tensors."
        )
    
    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process latent tensor with feedback blending
        
        Args:
            tensor: Current input latent tensor in format [B, C, H/8, W/8]
            
        Returns:
            Blended latent tensor (blend strength controlled by feedback_strength), or input tensor for first frame
        """
        # Get previous frame latent data using mixin method
        prev_latent = self._get_previous_data()
        
        if prev_latent is not None:
            input_latent = tensor
            
            # Ensure both tensors have the same batch size for element-wise blending
            # If batch sizes differ, expand the smaller one to match
            if prev_latent.shape[0] != input_latent.shape[0]:
                if prev_latent.shape[0] == 1:
                    # Expand previous latent to match input batch size
                    prev_latent = prev_latent.expand(input_latent.shape[0], -1, -1, -1)
                elif input_latent.shape[0] == 1:
                    # Expand input latent to match previous batch size
                    input_latent = input_latent.expand(prev_latent.shape[0], -1, -1, -1)
                else:
                    # Different non-unit batch sizes - use minimum to avoid errors
                    min_batch = min(prev_latent.shape[0], input_latent.shape[0])
                    prev_latent = prev_latent[:min_batch]
                    input_latent = input_latent[:min_batch]
            
            # Resize spatial dimensions if they don't match (though this should be rare in latent space)
            if prev_latent.shape[2:] != input_latent.shape[2:]:
                target_size = input_latent.shape[2:]  # Get H, W from input
                prev_latent = torch.nn.functional.interpolate(
                    prev_latent, size=target_size, mode='bilinear', align_corners=False
                )
            
            # Blend current latent with previous latent for temporal consistency
            # Higher feedback_strength = more influence from previous frame
            blended_latent = (1 - self.feedback_strength) * input_latent + self.feedback_strength * prev_latent
            
            # Add safety measures for latent values to prevent extreme outputs
            # Clamp to reasonable range based on typical latent distributions
            blended_latent = torch.clamp(blended_latent, min=-10.0, max=10.0)
            
            # Ensure correct device and dtype
            blended_latent = blended_latent.to(device=self.device, dtype=self.dtype)
            return blended_latent
        else:
            # First frame, no pipeline ref, or no previous latent available - use input tensor
            self._first_frame = False
            result = tensor.to(device=self.device, dtype=self.dtype)
            return result
