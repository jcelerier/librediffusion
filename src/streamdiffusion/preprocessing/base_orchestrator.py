import torch
from typing import List, Optional, Union, Dict, Any, Tuple, Callable, TypeVar, Generic
from abc import ABC, abstractmethod
import numpy as np
import concurrent.futures
import logging

logger = logging.getLogger(__name__)

# Type variables for generic orchestrator
T = TypeVar('T')  # Input type (e.g., ControlImage for preprocessing)
R = TypeVar('R')  # Result type (e.g., List[torch.Tensor] for preprocessing)


class BaseOrchestrator(Generic[T, R], ABC):
    """
    Generic base orchestrator for parallelized and pipelined processing.
    
    Handles thread pool management, pipeline state, and inter-frame pipelining
    while leaving domain-specific processing logic to subclasses.
    
    Type Parameters:
        T: Input type for processing operations
        R: Result type returned from processing operations
    """
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16, max_workers: int = 4, timeout_ms: float = 10.0, pipeline_ref: Optional[Any] = None):
        self.device = device
        self.dtype = dtype
        self.timeout_ms = timeout_ms
        self.pipeline_ref = pipeline_ref
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Pipeline state for pipelined processing
        self._next_frame_future = None
        self._next_frame_result = None
        
        # CUDA stream for background processing to avoid GPU contention
        self._background_stream = None
        device_str = str(device)
        if device_str.startswith("cuda") and torch.cuda.is_available():
            self._background_stream = torch.cuda.Stream()
    

    
    def cleanup(self) -> None:
        """Cleanup thread pool and CUDA stream resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
        
        # Cleanup CUDA stream if it exists
        if hasattr(self, '_background_stream') and self._background_stream is not None:
            # Synchronize the stream before cleanup
            torch.cuda.synchronize()
            self._background_stream = None
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.cleanup()
        except:
            pass
    
    @abstractmethod
    def _should_use_sync_processing(self, *args, **kwargs) -> bool:
        """
        Determine if synchronous processing should be used instead of pipelined.
        
        Subclasses implement domain-specific logic (e.g., feedback preprocessor detection).
        
        Returns:
            True if sync processing should be used, False for pipelined processing
        """
        pass
    
    @abstractmethod
    def _process_frame_background(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Process a frame in the background thread.
        
        Subclasses implement their specific processing logic here.
        
        Returns:
            Dictionary containing processing results and status
        """
        pass
    
    def process_pipelined(self, input_data: T, *args, **kwargs) -> R:
        """
        Process input with intelligent pipelining.
        
        Automatically falls back to sync processing when required by domain logic,
        otherwise uses pipelined processing for performance.
        
        Args:
            input_data: Input data to process
            *args, **kwargs: Additional arguments passed to processing methods
        
        Returns:
            Processing results
        """
        # Check if sync processing is required (domain-specific logic)
        if self._should_use_sync_processing(*args, **kwargs):
            return self.process_sync(input_data, *args, **kwargs)
        
        # Use pipelined processing
        # Wait for previous frame processing; non-blocking with short timeout
        self._wait_for_previous_processing()
        
        # Start next frame processing in background
        self._start_next_frame_processing(input_data, *args, **kwargs)
        
        # Apply current frame processing results if available; otherwise signal no update
        return self._apply_current_frame_processing(*args, **kwargs)
    
    @abstractmethod
    def process_sync(self, input_data: T, *args, **kwargs) -> R:
        """
        Process input synchronously.
        
        Subclasses implement their specific synchronous processing logic.
        
        Args:
            input_data: Input data to process
            *args, **kwargs: Additional arguments passed to processing methods
        
        Returns:
            Processing results
        """
        pass
    
    def _start_next_frame_processing(self, input_data: T, *args, **kwargs) -> None:
        """Start processing for next frame in background thread"""
        # Submit background processing
        self._next_frame_future = self._executor.submit(
            self._process_frame_background, input_data, *args, **kwargs
        )
    
    def _wait_for_previous_processing(self) -> None:
        """Wait for previous frame processing with configurable timeout"""
        if hasattr(self, '_next_frame_future') and self._next_frame_future is not None:
            try:
                # Use configurable timeout based on orchestrator type
                self._next_frame_result = self._next_frame_future.result(timeout=self.timeout_ms / 1000.0)
            except concurrent.futures.TimeoutError:
                # Non-blocking: skip applying results this frame
                self._next_frame_result = None
            except Exception as e:
                logger.error(f"BaseOrchestrator: Processing error: {e}")
                self._next_frame_result = None
        else:
            self._next_frame_result = None
    
    def _apply_current_frame_processing(self, processors=None, *args, **kwargs) -> R:
        """
        Apply processing results from previous iteration.
        
        Default implementation provides common fallback logic for tensor-to-tensor orchestrators.
        Subclasses can override this method for specialized behavior.
        
        Args:
            processors: List of processors/postprocessors to apply (parameter name varies by subclass)
            *args, **kwargs: Additional arguments
            
        Returns:
            Processing results, or processed current input if no results available
        """
        if not hasattr(self, '_next_frame_result') or self._next_frame_result is None:
            # First frame or no background results - process current input synchronously
            if hasattr(self, '_current_input_tensor') and self._current_input_tensor is not None:
                if processors:
                    return self.process_sync(self._current_input_tensor, processors)
                else:
                    return self._current_input_tensor
            
            # If we don't have current input stored, we have an issue
            class_name = self.__class__.__name__
            logger.error(f"{class_name}: No background results and no current input tensor available")
            raise RuntimeError(f"{class_name}: No processing results available")
        
        result = self._next_frame_result
        if result['status'] != 'success':
            class_name = self.__class__.__name__
            logger.warning(f"{class_name}: Background processing failed: {result.get('error', 'Unknown error')}")
            # Process current input synchronously on error
            if hasattr(self, '_current_input_tensor') and self._current_input_tensor is not None:
                if processors:
                    return self.process_sync(self._current_input_tensor, processors)
                else:
                    return self._current_input_tensor
            raise RuntimeError(f"{class_name}: Background processing failed and no fallback available")
        
        return result['result']
    
    def _set_background_stream_context(self):
        """
        Set CUDA stream context for background processing.
        
        Returns:
            The original stream to restore later, or None if no background stream
        """
        if self._background_stream is not None:
            original_stream = torch.cuda.current_stream()
            torch.cuda.set_stream(self._background_stream)
            return original_stream
        return None
    
    def _restore_stream_context(self, original_stream):
        """
        Restore the original CUDA stream context.
        
        Args:
            original_stream: The stream to restore, or None to do nothing
        """
        if self._background_stream is not None and original_stream is not None:
            torch.cuda.set_stream(original_stream)
