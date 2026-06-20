from typing import List, Optional, Any, Dict
import torch

from ..preprocessing.orchestrator_user import OrchestratorUser
from ..preprocessing.pipeline_preprocessing_orchestrator import PipelinePreprocessingOrchestrator
from ..hooks import ImageCtx, ImageHook


class ImageProcessingModule(OrchestratorUser):
    """
    Shared base class for image domain processing modules.
    
    Handles sequential chain execution for both preprocessing and postprocessing
    timing variants. Processing domain is always image tensors.
    """
    
    def __init__(self):
        """Initialize image processing module."""
        self.processors = []
        
    def _process_image_chain(self, input_image: torch.Tensor) -> torch.Tensor:
        """Execute sequential chain of processors in image domain.
        
        Uses the shared orchestrator's sequential chain processing.
        """
        if not self.processors:
            return input_image
            
        ordered_processors = self._get_ordered_processors()
        return self._preprocessing_orchestrator.execute_pipeline_chain(
            input_image, ordered_processors, processing_domain="image"
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
        # ImageProcessingModule uses 'pipeline' normalization context
        processor = get_preprocessor(processor_type, pipeline_ref=getattr(self, '_stream', None), normalization_context='pipeline')
        
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
        
        # Align preprocessor target size with stream resolution (same as ControlNet)
        if hasattr(self, '_stream'):
            try:
                if hasattr(processor, 'params') and isinstance(getattr(processor, 'params'), dict):
                    processor.params['image_width'] = int(self._stream.width)
                    processor.params['image_height'] = int(self._stream.height)
                if hasattr(processor, 'image_width'):
                    setattr(processor, 'image_width', int(self._stream.width))
                if hasattr(processor, 'image_height'):
                    setattr(processor, 'image_height', int(self._stream.height))
            except Exception:
                pass
        
        self.processors.append(processor)
    
    def _get_ordered_processors(self) -> List[Any]:
        """Return enabled processors in execution order based on their order attribute."""
        # Filter for enabled processors first, then sort by order
        enabled_processors = [p for p in self.processors if getattr(p, 'enabled', True)]
        return sorted(enabled_processors, key=lambda p: getattr(p, 'order', 0))


class ImagePreprocessingModule(ImageProcessingModule):
    """
    Image domain preprocessing module - executes before VAE encoding.
    
    Timing: After image_processor.preprocess(), before similar_image_filter
    Uses pipelined processing for performance optimization.
    """
    
    def install(self, stream) -> None:
        """Install module by registering hook with stream and attaching orchestrators."""
        self._stream = stream  # Store stream reference for dimension access
        self.attach_orchestrator(stream)  # For sequential chain processing (fallback)
        self.attach_pipeline_preprocessing_orchestrator(stream)  # For pipelined processing
        stream.image_preprocessing_hooks.append(self.build_image_hook())
    
    def build_image_hook(self) -> ImageHook:
        """Build hook function that processes image context with pipelined processing."""
        def hook(ctx: ImageCtx) -> ImageCtx:
            ctx.image = self._process_image_pipelined(ctx.image)
            return ctx
        return hook
    
    def _process_image_pipelined(self, input_image: torch.Tensor) -> torch.Tensor:
        """Execute pipelined processing of preprocessors for performance.
        
        Uses PipelinePreprocessingOrchestrator for Frame N-1 results while starting Frame N processing.
        Falls back to synchronous processing when needed.
        """
        if not self.processors:
            return input_image
        
        ordered_processors = self._get_ordered_processors()
        
        # Use pipelined pipeline preprocessing orchestrator for performance
        return self._pipeline_preprocessing_orchestrator.process_pipelined(
            input_image, ordered_processors
        )


class ImagePostprocessingModule(ImageProcessingModule):
    """
    Image domain postprocessing module - executes after VAE decoding.
    
    Timing: After decode_image(), before returning final output
    Uses pipelined processing for performance optimization.
    """
    
    def install(self, stream) -> None:
        """Install module by registering hook with stream and attaching orchestrators."""
        self._stream = stream  # Store stream reference for dimension access
        self.attach_preprocessing_orchestrator(stream)  # For sequential chain processing (fallback)
        self.attach_postprocessing_orchestrator(stream)  # For pipelined processing
        stream.image_postprocessing_hooks.append(self.build_image_hook())
    
    def build_image_hook(self) -> ImageHook:
        """Build hook function that processes image context with pipelined processing."""
        def hook(ctx: ImageCtx) -> ImageCtx:
            ctx.image = self._process_image_pipelined(ctx.image)
            return ctx
        return hook
    
    def _process_image_pipelined(self, input_image: torch.Tensor) -> torch.Tensor:
        """Execute pipelined processing of postprocessors for performance.
        
        Uses PostprocessingOrchestrator for Frame N-1 results while starting Frame N processing.
        Falls back to synchronous processing when needed.
        """
        if not self.processors:
            return input_image
        
        ordered_processors = self._get_ordered_processors()
        
        # Use pipelined postprocessing orchestrator for performance
        return self._postprocessing_orchestrator.process_pipelined(
            input_image, ordered_processors
        )
