"""Core subpackage containing STT, LLM, TTS, Audio, and Pipeline components."""

from app.core.audio import AudioProcessor, StreamAudioPlayer
from app.core.conversation import ConversationManager
from app.core.llm import LMStudioClient, split_into_sentence_chunks
from app.core.metrics import LatencyTracker, StageTimestamps, TurnTelemetry
from app.core.pipeline import (
    AssistantPipeline,
    AssistantResponse,
    AssistantStreamChunk,
    PipelineMetrics,
)
from app.core.sentences import SentenceBuffer, clean_text_for_speech
from app.core.stt import OpenVINOWhisperSTT
from app.core.tts import QwenTTSEngine

__all__ = [
    "AudioProcessor",
    "StreamAudioPlayer",
    "ConversationManager",
    "OpenVINOWhisperSTT",
    "LMStudioClient",
    "split_into_sentence_chunks",
    "SentenceBuffer",
    "clean_text_for_speech",
    "LatencyTracker",
    "StageTimestamps",
    "TurnTelemetry",
    "QwenTTSEngine",
    "AssistantPipeline",
    "AssistantResponse",
    "AssistantStreamChunk",
    "PipelineMetrics",
]
