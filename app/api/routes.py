"""REST API endpoints for the Personal Assistant service.

Provides HTTP routes for speech-to-text transcription, LLM chat completion,
text-to-speech synthesis, and complete end-to-end voice assistant interaction.
"""

import asyncio
import base64
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.core.audio import AudioProcessor
from app.core.pipeline import AssistantPipeline, PipelineMetrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Assistant"])

# Shared singleton pipeline (STT/TTS models loaded ONCE process-wide).
# See app/core/shared.py: routes + websocket must share one instance to
# avoid duplicate Whisper/Kokoro models in 16 GB RAM.
from app.core.shared import get_pipeline

pipeline = get_pipeline()


# ============================================================================
# Pydantic Schemas
# ============================================================================


class LLMModelInfo(BaseModel):
    """Schema for individual LM Studio model description."""

    id: str
    name: str
    loaded: bool = False
    params: Optional[str] = ""
    architecture: Optional[str] = ""
    size_bytes: Optional[int] = 0
    size_formatted: Optional[str] = ""
    type: str = "llm"
    supports_thinking: bool = False


class ThinkModeRequest(BaseModel):
    """Request schema for toggling LLM thinking mode."""

    think_mode: bool = Field(..., description="Whether to enable or disable thinking mode")
    reasoning_effort: Optional[str] = Field(
        None, description="Reasoning effort level: 'low', 'medium', 'high', 'max'"
    )


class ThinkModeResponse(BaseModel):
    """Response schema for LLM thinking mode status."""

    status: str = "success"
    think_mode: bool
    reasoning_effort: str
    supports_thinking: bool
    current_model: str
    message: str


class ModelsResponse(BaseModel):
    """Response schema listing available LM Studio models."""

    status: str = "success"
    current_model: str
    models: List[LLMModelInfo]
    loaded_models: List[str] = Field(default_factory=list)


class SelectModelRequest(BaseModel):
    """Request schema for selecting the active LM Studio model."""

    model: str = Field(..., min_length=1, description="LM Studio model identifier")
    load: bool = Field(default=True, description="Whether to preload model via LMS")


class SelectModelResponse(BaseModel):
    """Response schema for model selection."""

    status: str = "success"
    model: str
    loaded: bool = False
    ejected_models: List[str] = Field(default_factory=list)
    message: str


class EjectModelRequest(BaseModel):
    """Request schema for model ejection."""

    model: Optional[str] = Field(None, description="Model to unload, or null for all")


class EjectModelResponse(BaseModel):
    """Response schema for model ejection."""

    status: str = "success"
    ejected_models: List[str] = Field(default_factory=list)
    message: str


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    stt_model: str
    stt_device: str
    openvino_devices: List[str]
    lm_studio_connected: bool
    lm_studio_model: str
    lm_studio_models: List[str] = Field(default_factory=list)
    lm_studio_loaded_models: List[str] = Field(default_factory=list)
    think_mode: bool = False
    reasoning_effort: Optional[str] = "medium"
    current_model_supports_thinking: bool = False
    tts_model: str
    tts_speakers: List[str]
    tts_device: str = "cpu"
    tts_effective_device: Optional[str] = None
    tts_available_devices: List[str] = Field(default_factory=list)


class TTSDeviceRequest(BaseModel):
    """Request schema for selecting the active Kokoro TTS compute processing unit."""

    device: str = Field(..., min_length=1, description="Compute device ('cpu' or 'npu')")


class TTSDeviceResponse(BaseModel):
    """Response schema for Kokoro TTS compute device status."""

    status: str = "success"
    device: str
    effective_device: Optional[str] = None
    backend: Optional[str] = None
    available_devices: List[str] = Field(default_factory=list)
    message: str


class TranscribeResponse(BaseModel):
    """Transcription response schema."""

    transcription: str
    duration_seconds: Optional[float] = None


class ChatRequest(BaseModel):
    """Chat completion request schema."""

    message: str = Field(..., min_length=1, description="User prompt text")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    history: Optional[List[Dict[str, str]]] = Field(
        None, description="Chat history"
    )
    session_id: str = Field(default="default", description="Session key for history isolation")
    model: Optional[str] = Field(None, description="Optional model override")
    think_mode: Optional[bool] = Field(None, description="Optional think mode override")


class ChatResponse(BaseModel):
    """Chat completion response schema."""

    response: str
    model: str
    thought: Optional[str] = None


class TTSRequest(BaseModel):
    """Text-to-speech request schema."""

    text: str = Field(..., min_length=1, description="Text to synthesize")
    speaker: Optional[str] = Field(None, description="Voice persona")
    language: Optional[str] = Field(None, description="Spoken language")
    device: Optional[str] = Field(None, description="Compute device override ('cpu' or 'npu')")
    play_audio: bool = Field(False, description="Whether to play output on server speakers")


class InteractResponse(BaseModel):
    """Full assistant loop response schema."""

    user_text: str
    assistant_text: str
    audio_base64: str
    sample_rate: int
    metrics: Dict[str, Any]
    assistant_thought: Optional[str] = None



class RecordRequest(BaseModel):
    """Microphone record request schema."""

    duration_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    speaker: Optional[str] = None
    language: Optional[str] = None
    tts_device: Optional[str] = None
    play_audio: bool = False
    session_id: str = "default"


def _metrics_dict(metrics: Any) -> Dict[str, Any]:
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        return dict(metrics)
    base: Dict[str, Any] = {
        "stt_ms": getattr(metrics, "stt_latency_ms", 0.0),
        "llm_ms": getattr(metrics, "llm_latency_ms", 0.0),
        "tts_ms": getattr(metrics, "tts_latency_ms", 0.0),
        "tts_synth_ms": getattr(metrics, "tts_synth_ms", 0.0),
        "ttfa_ms": getattr(metrics, "ttfa_ms", 0.0),
        "total_ms": getattr(metrics, "total_latency_ms", 0.0),
    }
    # Detailed T0..T8 + throughput/resource telemetry (best-effort).
    for attr, key in (
        ("llm_ttft_ms", "llm_ttft_ms"),
        ("sentence_latency_ms", "sentence_latency_ms"),
        ("voice_latency_ms", "voice_latency_ms"),
        ("tokens_per_sec", "tokens_per_sec"),
        ("tts_realtime_factor", "tts_realtime_factor"),
        ("audio_queue_depth", "audio_queue_depth"),
        ("system_ram_mb", "system_ram_mb"),
        ("cpu_pct", "cpu_pct"),
        ("gpu_util_pct", "gpu_util_pct"),
        ("gpu_vram_mb", "gpu_vram_mb"),
        ("npu_status", "npu_status"),
    ):
        try:
            val = getattr(metrics, attr, None)
            if val is not None:
                base[key] = val
        except Exception:
            pass
    return base


# ============================================================================
# API Endpoints
# ============================================================================


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Returns the operational status of all assistant subsystems."""
    lm_status = await pipeline.llm.check_health()
    devices = pipeline.stt.get_openvino_devices()
    speakers = pipeline.tts.get_supported_speakers()

    lm_models: List[str] = []
    lm_loaded: List[str] = []
    if lm_status:
        try:
            available = await pipeline.llm.list_available_models()
            lm_models = [m["id"] for m in available]
            lm_loaded = [m["id"] for m in available if m.get("loaded")]
        except Exception:
            lm_models = [pipeline.llm.model]
            lm_loaded = [pipeline.llm.model]

    tts_info = pipeline.get_tts_device()
    think_info = pipeline.get_think_mode()

    return HealthResponse(
        status="healthy",
        stt_model=pipeline.stt.model_id,
        stt_device=pipeline.stt.device,
        openvino_devices=devices,
        lm_studio_connected=lm_status,
        lm_studio_model=pipeline.llm.model,
        lm_studio_models=lm_models,
        lm_studio_loaded_models=lm_loaded,
        think_mode=think_info.get("think_mode", False),
        reasoning_effort=think_info.get("reasoning_effort", "medium"),
        current_model_supports_thinking=think_info.get("supports_thinking", False),
        tts_model=pipeline.tts.model_id,
        tts_speakers=speakers,
        tts_device=tts_info.get("device", "cpu"),
        tts_effective_device=tts_info.get("effective_device"),
        tts_available_devices=tts_info.get("available_devices", ["cpu", "npu"]),
    )


@router.get("/llm/think-mode", response_model=ThinkModeResponse)
async def get_think_mode() -> ThinkModeResponse:
    """Returns the current LLM thinking/reasoning mode configuration."""
    info = pipeline.get_think_mode()
    return ThinkModeResponse(
        status="success",
        think_mode=info.get("think_mode", False),
        reasoning_effort=info.get("reasoning_effort", "medium"),
        supports_thinking=info.get("supports_thinking", False),
        current_model=info.get("current_model", pipeline.llm.model),
        message=f"Think mode is {'enabled' if info.get('think_mode') else 'disabled'}.",
    )


@router.post("/llm/think-mode", response_model=ThinkModeResponse)
async def set_think_mode(request: ThinkModeRequest) -> ThinkModeResponse:
    """Configures the LLM thinking/reasoning mode."""
    res = pipeline.set_think_mode(request.think_mode, request.reasoning_effort)
    return ThinkModeResponse(
        status="success",
        think_mode=res.get("think_mode", request.think_mode),
        reasoning_effort=res.get("reasoning_effort", "medium"),
        supports_thinking=res.get("supports_thinking", False),
        current_model=pipeline.llm.model,
        message=res.get("message", "Think mode updated."),
    )



@router.get("/llm/models", response_model=ModelsResponse)
@router.get("/models", response_model=ModelsResponse)
async def get_llm_models() -> ModelsResponse:
    """Returns all available and running LLM models from LM Studio."""
    try:
        models = await pipeline.llm.list_available_models()
        loaded = [m["id"] for m in models if m.get("loaded")]
        return ModelsResponse(
            status="success",
            current_model=pipeline.llm.model,
            models=models,
            loaded_models=loaded,
        )
    except Exception as exc:
        logger.error("Failed to list LM Studio models: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query LM Studio models: {exc}",
        )


@router.post("/llm/model", response_model=SelectModelResponse)
@router.post("/models/select", response_model=SelectModelResponse)
async def select_llm_model(req: SelectModelRequest) -> SelectModelResponse:
    """Switches the active LM Studio LLM model used by the assistant.

    Crucially ejects any previously loaded models from memory before loading
    the requested model.
    """
    try:
        res = await pipeline.set_model(req.model, load=req.load)
        return SelectModelResponse(
            status="success",
            model=res["model"],
            loaded=res.get("loaded", False),
            ejected_models=res.get("ejected_models", []),
            message=res.get("message", f"Switched to {req.model}"),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Failed to select model '%s': %s", req.model, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to switch model: {exc}",
        )


@router.post("/llm/eject", response_model=EjectModelResponse)
@router.post("/llm/unload", response_model=EjectModelResponse)
async def eject_llm_model(req: Optional[EjectModelRequest] = None) -> EjectModelResponse:
    """Ejects loaded models from LM Studio memory to free RAM and VRAM."""
    model_id = req.model if req else None
    try:
        res = await pipeline.eject_model(model_id)
        return EjectModelResponse(
            status="success",
            ejected_models=res.get("ejected_models", []),
            message=res.get("message", "Model(s) ejected successfully"),
        )
    except Exception as exc:
        logger.error("Failed to eject models: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to eject model(s): {exc}",
        )


@router.get("/devices")
async def get_devices() -> Dict[str, Any]:
    """Lists available OpenVINO hardware and audio input/output devices."""
    return {
        "openvino_devices": pipeline.stt.get_openvino_devices(),
        "audio_devices": AudioProcessor.get_audio_devices(),
    }


@router.get("/speakers")
async def get_speakers() -> Dict[str, List[str]]:
    """Returns the supported voice personas and languages for Kokoro-82M TTS."""
    return {
        "speakers": pipeline.tts.get_supported_speakers(),
        "languages": pipeline.tts.get_supported_languages(),
    }


@router.get("/tts/device", response_model=TTSDeviceResponse)
async def get_tts_device() -> TTSDeviceResponse:
    """Returns the currently active Kokoro TTS compute processing unit (cpu or npu)."""
    info = pipeline.get_tts_device()
    return TTSDeviceResponse(
        status="success",
        device=info["device"],
        effective_device=info.get("effective_device"),
        backend=info.get("backend"),
        available_devices=info.get("available_devices", ["cpu", "npu"]),
        message=f"Current Kokoro TTS processing unit: {info['device'].upper()}",
    )


@router.post("/tts/device", response_model=TTSDeviceResponse)
async def set_tts_device(req: TTSDeviceRequest) -> TTSDeviceResponse:
    """Switches the Kokoro TTS compute processing unit (cpu or npu)."""
    try:
        info = await asyncio.to_thread(pipeline.set_tts_device, req.device)
        return TTSDeviceResponse(
            status="success",
            device=info["device"],
            effective_device=info.get("effective_device"),
            backend=info.get("backend"),
            available_devices=info.get("available_devices", ["cpu", "npu"]),
            message=f"Kokoro TTS processing unit set to {info['device'].upper()} ({info.get('effective_device', info['device'].upper())})",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Failed to switch TTS processing unit to '%s': %s", req.device, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to switch TTS processing unit: {exc}",
        )


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
) -> TranscribeResponse:
    """Transcribes an uploaded audio file using OpenVINO Whisper Base INT8.

    Args:
        file: Multipart audio file upload.

    Returns:
        TranscribeResponse containing the transcribed text.
    """
    try:
        content = await file.read()
        audio_array, sr = AudioProcessor.load_from_bytes(
            content, target_sr=16000
        )
        duration = len(audio_array) / sr
        text = await asyncio.to_thread(
            pipeline.stt.transcribe, audio_array, sample_rate=sr
        )
        return TranscribeResponse(transcription=text, duration_seconds=round(duration, 2))
    except Exception as exc:
        logger.error("API transcribe error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {exc}",
        ) from exc


@router.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest) -> ChatResponse:
    """Sends a conversational prompt to the LM Studio LLM.

    Args:
        request: Chat message request payload.

    Returns:
        ChatResponse containing the generated assistant response.
    """
    try:
        if request.model and request.model != pipeline.llm.model:
            await pipeline.set_model(request.model, load=False)

        # Maintain multi-turn conversational context if history not explicitly passed.
        # Uses per-session isolated history to avoid cross-talk between clients.
        if request.history is not None:
            history = request.history
        else:
            history = await pipeline._get_history_slice(request.session_id)
        response_text = await pipeline.llm.generate_response(
            prompt=request.message,
            system_prompt=request.system_prompt,
            history=history,
        )

        # Update per-session history (bounded).
        await pipeline._append_history(request.session_id, request.message, response_text)

        return ChatResponse(response=response_text, model=pipeline.llm.model)
    except Exception as exc:
        logger.error("API chat error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM generation failed: {exc}",
        ) from exc


@router.post("/chat/stream")
async def chat_streaming(request: ChatRequest) -> StreamingResponse:
    """Streams sentence chunks from LM Studio LLM using Server-Sent Events (SSE).

    Args:
        request: Chat message request payload.

    Returns:
        SSE text/event-stream yielding sentence chunks as they complete.
    """
    if request.history is not None:
        history = request.history
    else:
        history = await pipeline._get_history_slice(request.session_id)

    async def _event_generator():
        import json

        full_reply_parts = []
        try:
            chunk_idx = 0
            async for sentence in pipeline.llm.stream_sentence_chunks(
                prompt=request.message,
                system_prompt=request.system_prompt,
                history=history,
            ):
                full_reply_parts.append(sentence)
                payload = json.dumps({"index": chunk_idx, "sentence": sentence})
                yield f"data: {payload}\n\n"
                chunk_idx += 1

            # Update per-session history on completion
            complete_text = " ".join(full_reply_parts).strip()
            await pipeline._append_history(
                request.session_id, request.message, complete_text
            )
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("SSE stream error: %s", exc)
            err_data = json.dumps({"error": str(exc)})
            yield f"data: {err_data}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@router.post("/chat/tokens")
async def chat_token_stream(request: ChatRequest) -> StreamingResponse:
    """Streams raw LLM token deltas for word-by-word text display.

    Emits SSE ``token`` events for generated tokens as they arrive from
    LM Studio (with type='thought' for reasoning or type='content' for answers),
    ending with ``[DONE]``. On successful completion the reply is appended
    to session history without thoughts.

    Args:
        request: Chat message request payload.

    Returns:
        SSE text/event-stream yielding token deltas.
    """
    if request.history is not None:
        history = request.history
    else:
        history = await pipeline._get_history_slice(request.session_id)

    async def _event_generator():
        import json

        full_reply_parts = []
        try:
            stream_fn = getattr(pipeline.llm, "stream_tokens", None)
            if stream_fn is not None and callable(stream_fn):
                token_stream = stream_fn(
                    prompt=request.message,
                    system_prompt=request.system_prompt,
                    history=history,
                    think_mode=request.think_mode,
                )
                async for token_type, token in token_stream:
                    if not token:
                        continue
                    if token_type == "thought":
                        payload = json.dumps({"token": token, "thought": token, "type": "thought"})
                    else:
                        full_reply_parts.append(token)
                        payload = json.dumps({"token": token, "type": "content"})
                    yield f"data: {payload}\n\n"
            else:
                async for token in pipeline.llm.stream_response(
                    prompt=request.message,
                    system_prompt=request.system_prompt,
                    history=history,
                ):
                    if not token:
                        continue
                    full_reply_parts.append(token)
                    payload = json.dumps({"token": token, "type": "content"})
                    yield f"data: {payload}\n\n"
            complete_text = "".join(full_reply_parts)
            await pipeline._append_history(
                request.session_id, request.message, complete_text
            )
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("Token SSE stream error: %s", exc)
            err_data = json.dumps({"error": str(exc)})
            yield f"data: {err_data}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@router.post("/tts")
async def synthesize_speech(request: TTSRequest) -> Response:
    """Synthesizes text into speech using Kokoro-82M and streams the WAV audio.

    Args:
        request: Text to synthesize and optional speaker voice / device / playback flag.

    Returns:
        Streaming WAV audio file response.
    """
    try:
        if request.play_audio:
            res = await pipeline.process_direct_tts(
                text=request.text,
                speaker=request.speaker,
                language=request.language,
                play_audio=True,
                tts_device=request.device,
            )
            return Response(content=res.audio_bytes, media_type="audio/wav")

        wav_bytes = await asyncio.to_thread(
            pipeline.tts.synthesize_to_wav_bytes,
            text=request.text,
            speaker=request.speaker,
            language=request.language,
            device=request.device,
        )
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as exc:
        logger.error("API TTS error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis failed: {exc}",
        ) from exc


@router.post("/tts/generate", response_model=InteractResponse)
async def generate_tts_speech(request: TTSRequest) -> InteractResponse:
    """Synthesizes text directly into speech and returns base64 audio and latency metrics.

    Args:
        request: Text to synthesize, optional voice persona, device, and playback options.

    Returns:
        InteractResponse with user text, assistant text, base64 audio, and timing metrics.
    """
    try:
        res = await pipeline.process_direct_tts(
            text=request.text,
            speaker=request.speaker,
            language=request.language,
            play_audio=request.play_audio,
            tts_device=request.device,
        )
        return InteractResponse(
            user_text=res.user_text,
            assistant_text=res.assistant_text,
            assistant_thought=res.assistant_thought,
            audio_base64=base64.b64encode(res.audio_bytes).decode("utf-8"),
            sample_rate=res.sample_rate,
            metrics=_metrics_dict(res.metrics),
        )
    except Exception as exc:
        logger.error("API TTS generate error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS generation failed: {exc}",
        ) from exc


@router.post("/interact", response_model=InteractResponse)
async def interact_voice(
    file: UploadFile = File(..., description="Voice recording audio file"),
    speaker: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    tts_device: Optional[str] = Form(None),
    play_audio: bool = Form(False),
    session_id: str = Form("default"),
    think_mode: Optional[bool] = Form(None),
) -> InteractResponse:
    """Full assistant flow: Audio in -> OpenVINO STT -> LM Studio LLM -> Kokoro-82M TTS.

    Args:
        file: Uploaded audio recording from user.
        speaker: Voice persona name.
        language: Language identifier.
        tts_device: Optional compute processing unit for Kokoro TTS ('cpu' or 'npu').
        play_audio: Whether the server should output audio to its speakers.
        session_id: Session key for history isolation.
        think_mode: Optional boolean flag to enable/disable reasoning thinking mode.

    Returns:
        InteractResponse with transcription, bot reply, thoughts, audio, and metrics.
    """
    try:
        audio_bytes = await file.read()
        res = await pipeline.process_audio_bytes(
            audio_bytes=audio_bytes,
            speaker=speaker,
            language=language,
            play_audio=play_audio,
            session_id=session_id,
            tts_device=tts_device,
            think_mode=think_mode,
        )
        b64_audio = base64.b64encode(res.audio_bytes).decode("utf-8")

        return InteractResponse(
            user_text=res.user_text,
            assistant_text=res.assistant_text,
            assistant_thought=res.assistant_thought,
            audio_base64=b64_audio,
            sample_rate=res.sample_rate,
            metrics=_metrics_dict(res.metrics),
        )
    except Exception as exc:
        logger.error("API interact error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Interaction cycle failed: {exc}",
        ) from exc


@router.post("/interact/stream")
async def interact_voice_stream(
    file: UploadFile = File(..., description="Voice recording audio file"),
    speaker: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    tts_device: Optional[str] = Form(None),
    play_audio: bool = Form(False),
    session_id: str = Form("default"),
) -> StreamingResponse:
    """Streams voice assistant interaction: STT -> LLM tokens -> sentence audio chunks over SSE.

    Args:
        file: Uploaded audio file.
        speaker: Voice persona name.
        language: Synthesis language.
        tts_device: Optional compute processing unit for Kokoro TTS ('cpu' or 'npu').
        play_audio: Whether to play output audio on host device speakers.
        session_id: Session key for history isolation.

    Returns:
        SSE text/event-stream delivering transcription, sentence audio chunks, and result frame.
    """
    import json

    audio_bytes = await file.read()
    queue: asyncio.Queue = asyncio.Queue(maxsize=32)

    async def _on_transcription(user_text: str):
        await queue.put({"type": "transcription", "user_text": user_text})

    async def _on_chunk(chunk):
        await queue.put(
            {
                "type": "chunk",
                "index": chunk.sentence_index,
                "text": chunk.text,
                "audio_base64": base64.b64encode(chunk.audio_bytes).decode("utf-8"),
                "sample_rate": chunk.sample_rate,
            }
        )

    async def _on_token(token_text):
        await queue.put({"type": "token", "text": token_text})

    async def _runner():
        try:
            res = await pipeline.process_audio_bytes(
                audio_bytes=audio_bytes,
                speaker=speaker,
                language=language,
                play_audio=play_audio,
                on_chunk=_on_chunk,
                on_token=_on_token,
                on_transcription=_on_transcription,
                session_id=session_id,
                tts_device=tts_device,
            )
            await queue.put(
                {
                    "type": "result",
                    "user_text": res.user_text,
                    "assistant_text": res.assistant_text,
                    "audio_base64": base64.b64encode(res.audio_bytes).decode("utf-8"),
                    "sample_rate": res.sample_rate,
                    "metrics": _metrics_dict(res.metrics),
                }
            )
        except Exception as exc:
            logger.error("API interact stream error: %s", exc)
            await queue.put({"type": "error", "message": str(exc)})
        finally:
            await queue.put(None)

    async def _event_stream():
        task = asyncio.create_task(_runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                payload = json.dumps(item)
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            await task

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@router.post("/record", response_model=InteractResponse)
async def record_from_microphone(request: RecordRequest) -> InteractResponse:
    """Captures microphone audio on the server host and runs the assistant loop.

    Args:
        request: Recording parameters (duration, speaker, play_audio).

    Returns:
        InteractResponse with the results of the voice loop.
    """
    try:
        recorded_audio = await asyncio.to_thread(
            AudioProcessor.record_microphone,
            request.duration_seconds,
            16000,
        )
        wav_bytes = AudioProcessor.to_wav_bytes(recorded_audio, 16000)

        res = await pipeline.process_audio_bytes(
            audio_bytes=wav_bytes,
            speaker=request.speaker,
            language=request.language,
            play_audio=request.play_audio,
            session_id=request.session_id,
            tts_device=request.tts_device,
        )
        b64_audio = base64.b64encode(res.audio_bytes).decode("utf-8")

        return InteractResponse(
            user_text=res.user_text,
            assistant_text=res.assistant_text,
            audio_base64=b64_audio,
            sample_rate=res.sample_rate,
            metrics=_metrics_dict(res.metrics),
        )
    except Exception as exc:
        logger.error("API record error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Microphone interaction failed: {exc}",
        ) from exc


class InterruptRequest(BaseModel):
    """User-interruption request schema."""

    session_id: str = Field(default="default", description="Session to interrupt")


@router.post("/interrupt")
async def interrupt_session(request: InterruptRequest) -> Dict[str, Any]:
    """Immediately stops LLM/TTS/playback for a session (barge-in).

    Stops pending TTS jobs, clears the audio queue, and aborts the in-flight
    LLM stream so a new user utterance can start without hearing stale audio.

    Args:
        request: Session key to interrupt.

    Returns:
        Dict with interruption acknowledgement.
    """
    try:
        pipeline.cancel(request.session_id)
        return {"interrupted": True, "session_id": request.session_id}
    except Exception as exc:
        logger.error("API interrupt error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Interrupt failed: {exc}",
        ) from exc


@router.get("/diagnostics")
async def get_diagnostics() -> Dict[str, Any]:
    """Returns runtime-confirmed OpenVINO/Kokoro device diagnostics.

    Reports available devices, OpenVINO version, Kokoro requested vs
    effective (compiled) device, per-stage NPU placement, and NPU plugin
    state. Never claims NPU execution unless the runtime confirms it.
    """
    try:
        from app.core.diagnostics import (
            get_device_details,
            get_kokoro_device_report,
            get_openvino_version,
        )

        kokoro = get_kokoro_device_report(pipeline.tts)
        return {
            "openvino_version": get_openvino_version(),
            "openvino_devices": kokoro.get("available_devices", []),
            "device_details": get_device_details(),
            "kokoro": kokoro,
            "stt_device": pipeline.stt.device,
            "stt_model": pipeline.stt.model_id,
        }
    except Exception as exc:
        logger.error("Diagnostics error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagnostics failed: {exc}",
        ) from exc


class BenchmarkRequest(BaseModel):
    """Kokoro benchmark request schema."""

    text: str = Field(
        default="Hello there, how are you today? This is a fixed benchmark sentence for measuring Kokoro speech synthesis speed.",
        description="Fixed text synthesized on each device",
    )
    devices: Optional[List[str]] = Field(
        default=None, description="Subset of [CPU, GPU, NPU] to benchmark"
    )
    warmup_iters: int = Field(default=2, ge=1, le=10)
    measure_iters: int = Field(default=3, ge=1, le=10)
    include_turbo_comparison: bool = Field(
        default=True, description="Benchmark NPU_TURBO off vs on when supported"
    )


@router.post("/benchmark/kokoro")
async def benchmark_kokoro_endpoint(request: BenchmarkRequest) -> Dict[str, Any]:
    """Benchmarks Kokoro TTS across CPU/GPU/NPU with the same fixed text.

    Steady-state generation excludes model init (warm-up iters run first).
    Reports compile time, warm-up, generation mean/p50, audio duration, and
    RTFx (audio_s / gen_s) per device, plus NPU turbo off vs on rows.
    """
    try:
        from app.core.benchmark import benchmark_all_devices, results_to_dicts

        results = await asyncio.to_thread(
            benchmark_all_devices,
            request.text,
            request.devices,
            request.warmup_iters,
            request.measure_iters,
            request.include_turbo_comparison,
        )
        return {"results": results_to_dicts(results)}
    except Exception as exc:
        logger.error("Kokoro benchmark error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark failed: {exc}",
        ) from exc
