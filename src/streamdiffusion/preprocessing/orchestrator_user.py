from __future__ import annotations

from typing import Optional

from .preprocessing_orchestrator import PreprocessingOrchestrator
from .postprocessing_orchestrator import PostprocessingOrchestrator
from .pipeline_preprocessing_orchestrator import PipelinePreprocessingOrchestrator


class OrchestratorUser:
    """
    Base class to attach shared orchestrators from the stream.
    Supports both preprocessing and postprocessing orchestrators.
    """

    _preprocessing_orchestrator: Optional[PreprocessingOrchestrator] = None
    _postprocessing_orchestrator: Optional[PostprocessingOrchestrator] = None
    _pipeline_preprocessing_orchestrator: Optional[PipelinePreprocessingOrchestrator] = None

    def attach_orchestrator(self, stream) -> None:
        """Attach preprocessing orchestrator (backward compatibility)."""
        self.attach_preprocessing_orchestrator(stream)
    
    def attach_preprocessing_orchestrator(self, stream) -> None:
        """Attach shared preprocessing orchestrator from stream."""
        orchestrator = getattr(stream, 'preprocessing_orchestrator', None)
        if orchestrator is None:
            # Lazy-create on stream once, on first user that needs it
            orchestrator = PreprocessingOrchestrator(device=stream.device, dtype=stream.dtype, max_workers=4, pipeline_ref=stream)
            setattr(stream, 'preprocessing_orchestrator', orchestrator)
        self._preprocessing_orchestrator = orchestrator
    
    def attach_postprocessing_orchestrator(self, stream) -> None:
        """Attach shared postprocessing orchestrator from stream."""
        orchestrator = getattr(stream, 'postprocessing_orchestrator', None)
        if orchestrator is None:
            # Lazy-create on stream once, on first user that needs it
            orchestrator = PostprocessingOrchestrator(device=stream.device, dtype=stream.dtype, max_workers=4, pipeline_ref=stream)
            setattr(stream, 'postprocessing_orchestrator', orchestrator)
        self._postprocessing_orchestrator = orchestrator
    
    def attach_pipeline_preprocessing_orchestrator(self, stream) -> None:
        """Attach shared pipeline preprocessing orchestrator from stream."""
        orchestrator = getattr(stream, 'pipeline_preprocessing_orchestrator', None)
        if orchestrator is None:
            # Lazy-create on stream once, on first user that needs it
            orchestrator = PipelinePreprocessingOrchestrator(device=stream.device, dtype=stream.dtype, max_workers=4, pipeline_ref=stream)
            setattr(stream, 'pipeline_preprocessing_orchestrator', orchestrator)
        self._pipeline_preprocessing_orchestrator = orchestrator


