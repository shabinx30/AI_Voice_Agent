"""WebSocket streaming interface for real-time voice interaction.

Enables bidirectional low-latency audio transmission and event-driven updates
between web/mobile frontends and the voice assistant server.
"""

import base64
import json
import logging
import uuid
from typing import Any, Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.audio import AudioProcessor
from app.core.pipeline import AssistantPipeline

logger = logging.getLogger(__name__)

ws_router = APIRouter(tags=["WebSocket"])
# Shared singleton with REST routes: one Whisper + one Kokoro in memory.
from app.core.shared import get_pipeline

ws_pipeline = get_pipeline()


def _metrics_dict(metrics: Any) -> dict:
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        return dict(metrics)
    base = {
        "stt_ms": getattr(metrics, "stt_latency_ms", 0.0),
        "llm_ms": getattr(metrics, "llm_latency_ms", 0.0),
        "tts_ms": getattr(metrics, "tts_latency_ms", 0.0),
        "tts_synth_ms": getattr(metrics, "tts_synth_ms", 0.0),
        "ttfa_ms": getattr(metrics, "ttfa_ms", 0.0),
        "total_ms": getattr(metrics, "total_latency_ms", 0.0),
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
            val = getattr(metrics, attr, None)
            if val is not None:
                base[key] = val
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

            if msg_type == "get_models":
                try:
                    available = await ws_pipeline.get_available_models()
                    await websocket.send_json({
                        "type": "models",
                        "current_model": ws_pipeline.llm.model,
                        "models": available,
                    })
                except Exception as exc:
                    await websocket.send_json({"type": "error", "message": f"Could not list models: {exc}"})
                continue

            if msg_type == "set_model":
                new_model = data.get("model")
                if not new_model:
                    await websocket.send_json({"type": "error", "message": "Missing model field."})
                    continue
                try:
                    res = await ws_pipeline.set_model(new_model, load=data.get("load", True))
                    await websocket.send_json({
                        "type": "model_changed",
                        "status": "success",
                        "model": res["model"],
                        "loaded": res.get("loaded", False),
                        "message": res.get("message", f"Switched to {new_model}"),
                    })
                except Exception as exc:
                    await websocket.send_json({"type": "error", "message": f"Could not switch model: {exc}"})
                continue

            if msg_type == "get_tts_device":
                try:
                    info = ws_pipeline.get_tts_device()
                    await websocket.send_json({
                        "type": "tts_device",
                        "device": info["device"],
                        "effective_device": info.get("effective_device"),
                        "available_devices": info.get("available_devices", ["cpu", "npu"]),
                    })
                except Exception as exc:
                    await websocket.send_json({"type": "error", "message": f"Could not get TTS device: {exc}"})
                continue

            if msg_type == "set_tts_device":
                new_device = data.get("device")
                if not new_device:
                    await websocket.send_json({"type": "error", "message": "Missing device field."})
                    continue
                try:
                    info = ws_pipeline.set_tts_device(new_device)
                    await websocket.send_json({
                        "type": "tts_device_changed",
                        "status": "success",
                        "device": info["device"],
                        "effective_device": info.get("effective_device"),
                        "available_devices": info.get("available_devices", ["cpu", "npu"]),
                        "message": f"Kokoro TTS processing unit set to {info['device'].upper()} ({info.get('effective_device', info['device'].upper())})",
                    })
                except Exception as exc:
                    await websocket.send_json({"type": "error", "message": f"Could not set TTS device: {exc}"})
                continue

            if msg_type == "get_think_mode":
                try:
                    info = ws_pipeline.get_think_mode()
                    await websocket.send_json({
                        "type": "think_mode",
                        "think_mode": info["think_mode"],
                        "reasoning_effort": info["reasoning_effort"],
                        "supports_thinking": info["supports_thinking"],
                    })
                except Exception as exc:
                    await websocket.send_json({"type": "error", "message": f"Could not get think mode: {exc}"})
                continue

            if msg_type == "set_think_mode":
                think_mode_val = data.get("think_mode")
                effort_val = data.get("reasoning_effort")
                try:
                    info = ws_pipeline.set_think_mode(
                        enabled=think_mode_val,
                        effort=effort_val,
                    )
                    await websocket.send_json({
                        "type": "think_mode_changed",
                        "status": "success",
                        "think_mode": info["think_mode"],
                        "reasoning_effort": info["reasoning_effort"],
                        "supports_thinking": info["supports_thinking"],
                        "message": f"Think mode set to {'enabled' if info['think_mode'] else 'disabled'} (effort: {info['reasoning_effort']})",
                    })
                except Exception as exc:
                    await websocket.send_json({"type": "error", "message": f"Could not set think mode: {exc}"})
                continue

            if msg_type == "audio":
                req_model = data.get("model")
                if req_model and req_model != ws_pipeline.llm.model:
                    try:
                        await ws_pipeline.set_model(req_model, load=False)
                    except Exception:
                        pass
                req_tts_device = data.get("tts_device")
                if req_tts_device and str(req_tts_device).lower() != str(ws_pipeline.tts.device).lower():
                    try:
                        ws_pipeline.set_tts_device(req_tts_device)
                    except Exception:
                        pass
                req_think_mode = data.get("think_mode")
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

                    async def _send_thought(thought_text: str) -> None:
                        await websocket.send_json(
                            {
                                "type": "thought",
                                "text": thought_text,
                            }
                        )

                    audio_bytes = base64.b64decode(audio_b64)
                    res = await ws_pipeline.process_audio_bytes(
                        audio_bytes=audio_bytes,
                        speaker=speaker,
                        play_audio=play_host,
                        on_chunk=_send_audio_chunk,
                        on_token=_send_token,
                        on_thought=_send_thought,
                        on_transcription=_send_transcription,
                        think_mode=req_think_mode,
                        session_id=session_id,
                    )

                    await websocket.send_json(
                        {
                            "type": "result",
                            "user_text": res.user_text,
                            "assistant_text": res.assistant_text,
                            "assistant_thought": res.assistant_thought,
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
                req_model = data.get("model")
                if req_model and req_model != ws_pipeline.llm.model:
                    try:
                        await ws_pipeline.set_model(req_model, load=False)
                    except Exception:
                        pass
                req_tts_device = data.get("tts_device")
                if req_tts_device and str(req_tts_device).lower() != str(ws_pipeline.tts.device).lower():
                    try:
                        ws_pipeline.set_tts_device(req_tts_device)
                    except Exception:
                        pass
                req_think_mode = data.get("think_mode")
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

                    async def _send_text_thought(thought_text: str) -> None:
                        await websocket.send_json(
                            {
                                "type": "thought",
                                "text": thought_text,
                            }
                        )

                    res = await ws_pipeline.process_text_prompt(
                        prompt=prompt,
                        speaker=speaker,
                        play_audio=play_host,
                        on_chunk=_send_text_chunk,
                        on_token=_send_token,
                        on_thought=_send_text_thought,
                        think_mode=req_think_mode,
                        session_id=session_id,
                    )

                    await websocket.send_json(
                        {
                            "type": "result",
                            "user_text": res.user_text,
                            "assistant_text": res.assistant_text,
                            "assistant_thought": res.assistant_thought,
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

            elif msg_type == "tts":
                req_tts_device = data.get("tts_device")
                if req_tts_device and str(req_tts_device).lower() != str(ws_pipeline.tts.device).lower():
                    try:
                        ws_pipeline.set_tts_device(req_tts_device)
                    except Exception:
                        pass
                text = data.get("text") or data.get("prompt", "")
                speaker = data.get("speaker")
                play_host = data.get("play_audio", False)

                if not text or not str(text).strip():
                    await websocket.send_json(
                        {"type": "error", "message": "Missing or empty text for TTS synthesis."}
                    )
                    continue

                try:
                    await websocket.send_json(
                        {
                            "type": "status",
                            "stage": "tts",
                            "message": "Synthesizing speech with Kokoro TTS...",
                        }
                    )

                    async def _send_tts_chunk(chunk) -> None:
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

                    res = await ws_pipeline.process_direct_tts(
                        text=text,
                        speaker=speaker,
                        play_audio=play_host,
                        on_chunk=_send_tts_chunk,
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
                    logger.error("WebSocket TTS cycle error: %s", proc_exc)
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
