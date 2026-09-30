"""Shared singleton pipeline (avoids duplicate STT/TTS models in memory).

Previously ``app.api.routes`` and ``app.api.websocket`` each constructed
their own ``AssistantPipeline`` at import time, loading Whisper + Kokoro
twice (≈2x RAM, 2x NPU compile). All server entrypoints must use
:func:`get_pipeline` so models are loaded and compiled ONCE and reused.
"""

from __future__ import annotations

from typing import Optional

from app.core.pipeline import AssistantPipeline

_instance: Optional[AssistantPipeline] = None


def get_pipeline() -> AssistantPipeline:
    """Returns the process-wide shared AssistantPipeline (lazy singleton)."""
    global _instance
    if _instance is None:
        _instance = AssistantPipeline()
    return _instance


def set_pipeline(pipeline: AssistantPipeline) -> None:
    """Overrides the shared instance (tests / embedding)."""
    global _instance
    _instance = pipeline
