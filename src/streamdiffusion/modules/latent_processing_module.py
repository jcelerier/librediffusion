from typing import List, Optional, Any, Dict
import torch

from ..preprocessing.orchestrator_user import OrchestratorUser
from ..hooks import LatentCtx, LatentHook


class LatentProcessingModule(OrchestratorUser):
    """
    Shared base class for latent domain processing modules.
    
    Handles sequential chain execution for both preprocessing and postprocessing
    timing variants. Processing domain is always latent tensors.
    """
    
    def __init__(self):
        """Initialize latent processing module."""
        self.processors = []
        
    def _process_latent_chain(self, input_latent: torch.Tensor) -> torch.Tensor:
        """Execute sequential chain of processors in latent domain.
        
        Uses the shared orchestrator's sequential chain processing.
        """
        if not self.processors:
            return input_latent
            
        ordered_processors = self._get_ordered_processors()
        return self._preprocessing_orchestrator.execute_pipeline_chain(
            input_latent, ordered_processors, processing_domain="latent"
        )
    
    def add_processor(self, proc_config: Dict[str, Any]) -> None:
        """Add a processor using the existing registry, following ControlNet pattern."""
        from streamdiffusion.preprocessing.processors import get_preprocessor
        
        processor_type = proc_config.get('type')
        if not processor_type:
            raise ValueError("Processor config missing 'type' field")
        
        # Check if processor is enabled (default to True, same as ControlNet)
        enabled = proc_config.get('enabled', True)
        
        # Create processor using existing registry (same as ControlNet)
        # LatentProcessingModule uses 'latent' normalization context (works in latent space)
        processor = get_preprocessor(processor_type, pipeline_ref=self._stream, normalization_context='latent')
        
        # Apply parameters (same pattern as ControlNet)
        processor_params = proc_config.get('params', {})
        if processor_params:
            if hasattr(processor, 'params') and isinstance(getattr(processor, 'params'), dict):
                processor.params.update(processor_params)
            for name, value in processor_params.items():
                try:
                    if hasattr(processor, name):
                        setattr(processor, name, value)
                except Exception:
                    pass
        
        # Set order for sequential execution
        order = proc_config.get('order', len(self.processors))
        setattr(processor, 'order', order)
        
        # Set enabled state
        setattr(processor, 'enabled', enabled)
        
        # Pipeline reference is now automatically handled by the factory function
        
        self.processors.append(processor)
    
    def _get_ordered_processors(self) -> List[Any]:
        """Return enabled processors in execution order based on their order attribute."""
        # Filter for enabled processors first, then sort by order
        enabled_processors = [p for p in self.processors if getattr(p, 'enabled', True)]
        return sorted(enabled_processors, key=lambda p: getattr(p, 'order', 0))


class LatentPreprocessingModule(LatentProcessingModule):
    """
    Latent domain preprocessing module - executes after VAE encoding, before diffusion.
    
    Timing: After encode_image(), before predict_x0_batch()
    """
    
    def install(self, stream) -> None:
        """Install module by registering hook with stream and attaching orchestrator."""
        self.attach_orchestrator(stream)
        self._stream = stream  # Store stream reference like ControlNet module does
        stream.latent_preprocessing_hooks.append(self.build_latent_hook())
    
    def build_latent_hook(self) -> LatentHook:
        """Build hook function that processes latent context."""
        def hook(ctx: LatentCtx) -> LatentCtx:
            ctx.latent = self._process_latent_chain(ctx.latent)
            return ctx
        return hook


class LatentPostprocessingModule(LatentProcessingModule):
    """
    Latent domain postprocessing module - executes after diffusion, before VAE decoding.
    
    Timing: After predict_x0_batch(), before decode_image()
    """
    
    def install(self, stream) -> None:
        """Install module by registering hook with stream and attaching orchestrator."""
        self.attach_orchestrator(stream)
        self._stream = stream  # Store stream reference like ControlNet module does
        stream.latent_postprocessing_hooks.append(self.build_latent_hook())
    
    def build_latent_hook(self) -> LatentHook:
        """Build hook function that processes latent context."""
        def hook(ctx: LatentCtx) -> LatentCtx:
            ctx.latent = self._process_latent_chain(ctx.latent)
            return ctx
        return hook
