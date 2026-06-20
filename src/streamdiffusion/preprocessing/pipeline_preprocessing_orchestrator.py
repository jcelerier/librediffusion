import torch
from typing import List, Dict, Any, Optional
import logging
from .base_orchestrator import BaseOrchestrator

logger = logging.getLogger(__name__)

class PipelinePreprocessingOrchestrator(BaseOrchestrator[torch.Tensor, torch.Tensor]):
    """
    Orchestrates pipeline input preprocessing with parallelization and pipelining.
    
    Handles preprocessing of input tensors before they enter the diffusion pipeline.
    
    Tensor ranges:
    - Input: Receives [-1, 1] tensors from image_processor.preprocess()
    - Processors: Work in [-1, 1] space when normalization_context='pipeline'
    - Output: Returns [-1, 1] tensors for pipeline processing
    
    Note: Processors created with normalization_context='pipeline' expect and preserve
    [-1, 1] range. No automatic conversion happens in this orchestrator.
    """
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16, max_workers: int = 4, pipeline_ref: Optional[Any] = None):
        # Pipeline preprocessing: 10ms timeout for responsive processing
        super().__init__(device, dtype, max_workers, timeout_ms=10.0, pipeline_ref=pipeline_ref)
        
        # Pipeline preprocessing specific state
        self._current_input_tensor = None  # For BaseOrchestrator fallback logic
    
    def _should_use_sync_processing(self, *args, **kwargs) -> bool:
        """
        Determine if synchronous processing should be used instead of pipelined.
        
        For pipeline preprocessing, we typically use pipelined processing since most
        pipeline preprocessors are stateless and don't have temporal feedback requirements.
        
        Returns:
            False - pipeline preprocessing can typically always use pipelined processing
        """
        # Pipeline preprocessing generally doesn't require sync processing
        # Most processors are stateless and work well with pipelining
        return False
    
    def process_pipelined(self, 
                        input_tensor: torch.Tensor,
                        processors: List[Any],
                        *args, **kwargs) -> torch.Tensor:
        """
        Process input with intelligent pipelining.
        
        Overrides base method to store current input tensor for fallback logic.
        """
        # Store current input for fallback logic
        self._current_input_tensor = input_tensor
        
        # RACE CONDITION FIX: Check if there are actually enabled processors
        # Filter to only enabled processors (same logic as _get_ordered_processors)
        enabled_processors = [p for p in processors if getattr(p, 'enabled', True)] if processors else []
        
        if not enabled_processors:
            return input_tensor
        
        # Call parent implementation
        return super().process_pipelined(input_tensor, processors, *args, **kwargs)
    
    def process_sync(self, 
                   input_tensor: torch.Tensor,
                   processors: List[Any]) -> torch.Tensor:
        """
        Process pipeline input tensor synchronously through preprocessors.
        
        Implementation of BaseOrchestrator.process_sync for pipeline preprocessing.
        
        Args:
            input_tensor: Input tensor to preprocess (already normalized)
            processors: List of preprocessor instances
            
        Returns:
            Preprocessed tensor ready for pipeline processing
        """
        if not processors:
            return input_tensor
        
        # Sequential application of processors
        current_tensor = input_tensor
        for processor in processors:
            if processor is not None:
                current_tensor = self._apply_single_processor(current_tensor, processor)
        
        return current_tensor
    
    def _process_frame_background(self, 
                                input_tensor: torch.Tensor,
                                processors: List[Any]) -> Dict[str, Any]:
        """
        Process a frame in the background thread.
        
        Implementation of BaseOrchestrator._process_frame_background for pipeline preprocessing.
        
        Returns:
            Dictionary containing processing results and status
        """
        try:
            # Set CUDA stream for background processing
            original_stream = self._set_background_stream_context()
            
            if not processors:
                return {
                    'result': input_tensor,
                    'status': 'success'
                }
            
            # Process processors sequentially (most pipeline preprocessing is dependent)
            current_tensor = input_tensor
            for processor in processors:
                if processor is not None:
                    current_tensor = self._apply_single_processor(current_tensor, processor)
            
            return {
                'result': current_tensor,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"PipelinePreprocessingOrchestrator: Background processing failed: {e}")
            # Return original input tensor on error
            return {
                'result': input_tensor,
                'error': str(e),
                'status': 'error'
            }
        finally:
            # Restore original CUDA stream
            self._restore_stream_context(original_stream)
    
    
    
    def _apply_single_processor(self, 
                              input_tensor: torch.Tensor, 
                              processor: Any) -> torch.Tensor:
        """
        Apply a single processor to the input tensor.
        
        Args:
            input_tensor: Input tensor to process
            processor: Processor instance
            
        Returns:
            Processed tensor
        """
        try:
            # Apply processor
            if hasattr(processor, 'process_tensor'):
                # Prefer tensor processing method
                result = processor.process_tensor(input_tensor)
            elif hasattr(processor, 'process'):
                # Use general process method
                result = processor.process(input_tensor)
            elif callable(processor):
                # Treat as callable
                result = processor(input_tensor)
            else:
                logger.warning(f"PipelinePreprocessingOrchestrator: Unknown processor type: {type(processor)}")
                return input_tensor
            
            # Ensure result is a tensor
            if isinstance(result, torch.Tensor):
                return result
            else:
                logger.warning(f"PipelinePreprocessingOrchestrator: Processor returned non-tensor: {type(result)}")
                return input_tensor
                
        except Exception as e:
            logger.error(f"PipelinePreprocessingOrchestrator: Processor failed: {e}")
            return input_tensor  # Return original on error
    
    def clear_cache(self) -> None:
        """Clear preprocessing cache"""
        pass
