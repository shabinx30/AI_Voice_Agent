"""WebSocket streaming interface for real-time voice interaction.

Enables bidirectional low-latency audio transmission and event-driven updates
between web/mobile frontends and the voice assistant server.
"""

import base64
import json
import logging
import uuid
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.audio import AudioProcessor
from app.core.pipeline import AssistantPipeline

logger = logging.getLogger(__name__)

ws_router = APIRouter(tags=["WebSocket"])
# Shared singleton with REST routes: one Whisper + one Kokoro in memory.
from app.core.shared import get_pipeline

ws_pipeline = get_pipeline()


def _metrics_dict(metrics) -> dict:
    base = {
        "stt_ms": metrics.stt_latency_ms,
        "llm_ms": metrics.llm_latency_ms,
        "tts_ms": metrics.tts_latency_ms,
        "tts_synth_ms": getattr(metrics, "tts_synth_ms", 0.0),
        "ttfa_ms": metrics.ttfa_ms,
        "total_ms": metrics.total_latency_ms,
    }
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
            base[key] = getattr(metrics, attr)
        except Exception:
            pass
    return base


@ws_router.websocket("/ws/assistant")
async def websocket_assistant_endpoint(websocket: WebSocket) -> None:
    """Manages full-duplex WebSocket connection for voice interaction.

    Protocol Messages:
        Inbound:
            {"type": "audio", "data": "<base64_audio_data>", "speaker": "ryan"}
            {"type": "text", "prompt": "Hello assistant"}
            {"type": "ping"}
        Outbound:
            {"type": "status", "stage": "stt|llm|tts", "message": "..."}
            {"type": "transcription", "user_text": "..."}
            {"type": "token", "text": "<raw LLM token delta>"}
            {"type": "chunk", "index": 0, "text": "...", "audio_base64": "...", "sample_rate": 24000}
            {"type": "result", "user_text": "...", "assistant_text": "...", "audio": "<base64>"}
            {"type": "error", "message": "..."}
    """
    await websocket.accept()
    logger.info("WebSocket client connected.")
    # Per-connection session isolates history + playback interruption so
    # concurrent browser tabs don't share context or cut each other's audio.
    session_id = f"ws-{uuid.uuid4().hex[:12]}"

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                data = json.loads(raw_message)
            except Exception:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON frame."}
                )
                continue

            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type == "interrupt":
                # Barge-in: user started speaking while assistant talks.
                try:
                    ws_pipeline.cancel(session_id)
                except Exception:
                    pass
                await websocket.send_json(
                    {"type": "interrupted", "session_id": session_id}
                )
                continue

            if msg_type == "audio":
                audio_b64 = data.get("data", "")
                speaker = data.get("speaker")
                play_host = data.get("play_audio", False)

                if not audio_b64:
                    await websocket.send_json(
                        {"type": "error", "message": "Missing audio data."}
                    )
                    continue

                try:
                    await websocket.send_json(
                        {
                            "type": "status",
                            "stage": "stt",
                            "message": "Transcribing speech with OpenVINO Whisper...",
                        }
                    )

                    async def _send_transcription(transcribed_text: str) -> None:
                        await websocket.send_json(
                            {
                                "type": "transcription",
                                "user_text": transcribed_text,
                            }
                        )

                    async def _send_audio_chunk(chunk) -> None:
                        await websocket.send_json(
                            {
                                "type": "chunk",
                                "index": chunk.sentence_index,
                                "text": chunk.text,
                                "audio_base64": base64.b64encode(
                                    chunk.audio_bytes
                                ).decode("utf-8"),
                                "sample_rate": chunk.sample_rate,
                            }
                        )

                    async def _send_token(token_text: str) -> None:
                        await websocket.send_json(
                            {
                                "type": "token",
                                "text": token_text,
                            }
                        )

                    audio_bytes = base64.b64decode(audio_b64)
                    res = await ws_pipeline.process_audio_bytes(
                        audio_bytes=audio_bytes,
                        speaker=speaker,
                        play_audio=play_host,
                        on_chunk=_send_audio_chunk,
                        on_token=_send_token,
                        on_transcription=_send_transcription,
                        session_id=session_id,
                    )

                    await websocket.send_json(
                        {
                            "type": "result",
                            "user_text": res.user_text,
                            "assistant_text": res.assistant_text,
                            "audio_base64": base64.b64encode(
                                res.audio_bytes
                            ).decode("utf-8"),
                            "metrics": _metrics_dict(res.metrics),
                        }
                    )
                except Exception as proc_exc:
                    logger.error("WebSocket audio cycle error: %s", proc_exc)
                    await websocket.send_json(
                        {"type": "error", "message": str(proc_exc)}
                    )

            elif msg_type == "text":
                prompt = data.get("prompt", "")
                speaker = data.get("speaker")
                play_host = data.get("play_audio", False)

                try:
                    await websocket.send_json(
                        {
                            "type": "status",
                            "stage": "llm",
                            "message": "Streaming LM Studio LLM and synthesizing speech...",
                        }
                    )

                    async def _send_text_chunk(chunk) -> None:
                        await websocket.send_json(
                            {
                                "type": "chunk",
                                "index": chunk.sentence_index,
                                "text": chunk.text,
                                "audio_base64": base64.b64encode(
                                    chunk.audio_bytes
                                ).decode("utf-8"),
                                "sample_rate": chunk.sample_rate,
                            }
                        )

                    async def _send_token(token_text: str) -> None:
                        await websocket.send_json(
                            {
                                "type": "token",
                                "text": token_text,
                            }
                        )

                    res = await ws_pipeline.process_text_prompt(
                        prompt=prompt,
                        speaker=speaker,
                        play_audio=play_host,
                        on_chunk=_send_text_chunk,
                        on_token=_send_token,
                        session_id=session_id,
                    )

                    await websocket.send_json(
                        {
                            "type": "result",
                            "user_text": res.user_text,
                            "assistant_text": res.assistant_text,
                            "audio_base64": base64.b64encode(
                                res.audio_bytes
                            ).decode("utf-8"),
                            "metrics": _metrics_dict(res.metrics),
                        }
                    )
                except Exception as proc_exc:
                    logger.error("WebSocket text cycle error: %s", proc_exc)
                    await websocket.send_json(
                        {"type": "error", "message": str(proc_exc)}
                    )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.error("WebSocket unhandled exception: %s", exc)
    finally:
        # Free per-session history + stop any lingering playback.
        try:
            ws_pipeline.clear_history(session_id=session_id)
        except Exception:
            pass
