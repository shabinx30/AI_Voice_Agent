"""True-overlapped streaming executor: LLM tokens -> sentences -> TTS -> audio.

Why this module exists (latency rationale):
    The previous ``AssistantPipeline._execute_streaming_pipeline`` collected
    *all* LLM sentences first (``await _produce()``) and only then drained
    TTS. TTFA therefore included the *full* LLM response time. This module
    instead runs producer (LLM), workers (Kokoro), and emitter (playback) as
    concurrent asyncio tasks sharing bounded queues, so:

    * Kokoro starts sentence 1 while Qwen still generates sentence 2,
    * audio playback starts while later sentences still synthesize,
    * a bounded sentence queue applies backpressure instead of accumulating
      unbounded audio (prevents TTS overproduction),
    * per-session generation counters abort stale turns instantly on barge-in.

Pipeline shape::

    LLM token stream
        -> SentenceBuffer (incremental, abbreviation-safe)
        -> sentence_queue (bounded, max tts_queue_max)
        -> N TTS workers (Kokoro, semaphore-capped)
        -> reorder buffer (dict idx -> audio)
        -> emitter (in-order play + on_chunk + WAV bytes)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


async def run_streaming_pipeline(
    pipeline: Any,
    prompt: str,
    speaker: Optional[str] = None,
    language: Optional[str] = None,
    play_audio: Optional[bool] = None,
    on_chunk: Optional[Callable[[Any], Any]] = None,
    on_token: Optional[Callable[[str], Any]] = None,
    session_id: str = "default",
) -> Tuple[str, bytes, int, Any]:
    """Executes one overlapped LLM->TTS->audio turn.

    Args:
        pipeline: Owning ``AssistantPipeline`` (provides stt/llm/tts, history,
            synth helpers, player registry, cancellation counters).
        prompt: User message text.
        speaker: Voice persona override.
        language: Language override.
        play_audio: Whether to play on host speakers (defaults to settings).
        on_chunk: Per-sentence audio callback.
        on_token: Per-token text callback.
        session_id: Session key for history + interruption isolation.

    Returns:
        Tuple (assistant_text, wav_bytes, sample_rate, metrics).
    """
    from app.core.audio import AudioProcessor
    from app.core.metrics import LatencyTracker
    from app.core.pipeline import PipelineMetrics
    from app.core.sentences import SentenceBuffer, clean_text_for_speech

    pipeline_start = time.perf_counter()
    tracker = LatencyTracker()
    tracker.stages.t0_mic_end = pipeline_start
    metrics = PipelineMetrics()

    should_play = play_audio if play_audio is not None else settings.auto_play_audio
    my_gen = await pipeline._next_generation(session_id)

    def _cancelled() -> bool:
        try:
            return pipeline._is_cancelled(session_id, my_gen)
        except Exception:
            return False

    # Register live tasks so cancel() can abort a blocked LLM await instantly.
    import asyncio as _aio

    _task_registry = getattr(pipeline, "_session_tasks", None)
    if _task_registry is None:
        pipeline._session_tasks = {}
        _task_registry = pipeline._session_tasks
    _my_tasks: set = set()
    _task_registry[session_id] = _my_tasks

    # Stop previous playback for this session only; new turn owns the player.
    prev_player = pipeline._players.pop(session_id, None)
    if prev_player is not None:
        try:
            prev_player.stop()
        except Exception:
            pass

    audio_qmax = max(2, int(getattr(settings, "audio_queue_max", 8)))
    t8_fired = {"done": False}

    def _on_first_play() -> None:
        tracker.mark("t8_speaker_start")
        t8_fired["done"] = True

    stream_player = None
    if should_play:
        stream_player = AudioProcessor.create_stream_player(
            max_queue=audio_qmax,
            on_first_play=_on_first_play,
        )
        pipeline._players[session_id] = stream_player

    history_slice = await pipeline._get_history_slice(session_id)

    min_chars = int(getattr(settings, "sentence_min_chars", 12))
    max_chars = int(getattr(settings, "sentence_max_chars", 160))
    qmax = max(1, int(getattr(settings, "tts_queue_max", 3)))
    n_workers = max(1, int(getattr(settings, "tts_max_workers", 2)))

    # Single token stream drives both callbacks + sentence split (one LLM pass).
    # _token_tap_stream already forwards on_token; it returns None for
    # non-streaming test doubles, in which case we fall back below.
    token_gen = pipeline._token_tap_stream(prompt, history_slice, on_token)
    sentence_stream = None
    use_tokens = token_gen is not None
    if not use_tokens:
        sentence_stream = pipeline._resolve_stream(prompt, history_slice)

    if token_gen is None and sentence_stream is None:
        return await _run_non_streaming(
            pipeline, prompt, speaker, language, history_slice,
            should_play, stream_player, session_id, my_gen,
            pipeline_start, tracker, metrics, on_chunk,
        )

    sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=qmax)
    results: Dict[int, Tuple[str, np.ndarray, int, float]] = {}
    emit_event = asyncio.Event()
    state = {
        "producer_done": False,
        "producer_error": None,
        "total": 0,
        "display_texts": {},
        "llm_start": 0.0,
        "first_token_seen": False,
        "first_sentence_seen": False,
        "tts_wall_start": None,
        "tts_wall_end": None,
        "tts_synth_total": 0.0,
        "tts_sr": getattr(pipeline.tts, "sample_rate", 24000),
    }

    async def _put_sentence(display: str, synth_text: str) -> None:
        idx = state["total"]
        state["total"] += 1
        state["display_texts"][idx] = display
        # Bounded put = backpressure: slows TTS submission when consumers lag
        # instead of accumulating unbounded audio (prevents overproduction).
        while True:
            if _cancelled():
                return
            try:
                sentence_queue.put_nowait((idx, synth_text))
                break
            except asyncio.QueueFull:
                await asyncio.sleep(0.01)

    async def _producer() -> None:
        tracker.mark("t3_llm_send")
        state["llm_start"] = time.perf_counter()
        try:
            if use_tokens:
                assert token_gen is not None
                buf = SentenceBuffer(min_chars=min_chars, max_chars=max_chars)
                async for token in token_gen:
                    if _cancelled():
                        break
                    if not token:
                        continue
                    if not state["first_token_seen"]:
                        state["first_token_seen"] = True
                        tracker.mark("t4_first_token")
                    tracker.note_token(token)
                    try:
                        chunks = buf.feed(token)
                    except Exception:
                        chunks = []
                    for raw in chunks:
                        display = raw.strip()
                        if not display:
                            continue
                        cleaned = clean_text_for_speech(display)
                        synth_text = cleaned if cleaned else display
                        if not state["first_sentence_seen"]:
                            state["first_sentence_seen"] = True
                            tracker.mark("t5_first_sentence")
                        await _put_sentence(display, synth_text)
                tail = buf.flush()
                if tail and not _cancelled():
                    display = tail.strip()
                    if display and any(c.isalnum() for c in display):
                        cleaned = clean_text_for_speech(display)
                        await _put_sentence(display, cleaned if cleaned else display)
                        if not state["first_sentence_seen"]:
                            tracker.mark("t5_first_sentence")
            else:
                assert sentence_stream is not None
                async for sentence in sentence_stream:
                    if _cancelled():
                        break
                    display = sentence.strip() if isinstance(sentence, str) else ""
                    if not display:
                        continue
                    if not state["first_token_seen"]:
                        # No token timing available; approximate TTFT with
                        # first-sentence arrival.
                        state["first_token_seen"] = True
                        tracker.mark("t4_first_token")
                        tracker.telemetry.llm_tokens += max(1, len(display) // 4)
                    if not state["first_sentence_seen"]:
                        state["first_sentence_seen"] = True
                        tracker.mark("t5_first_sentence")
                    tracker.telemetry.llm_chars += len(display)
                    cleaned = clean_text_for_speech(display)
                    await _put_sentence(display, cleaned if cleaned else display)
        except Exception as exc:
            logger.error("LLM streaming producer error: %s", exc)
            state["producer_error"] = exc
        finally:
            state["producer_done"] = True
            # Wake workers + emitter; one sentinel per worker.
            for _ in range(n_workers):
                try:
                    sentence_queue.put_nowait(None)
                except asyncio.QueueFull:
                    # Queue full with real work; workers will drain then see
                    # done flag. Put sentinels lazily via emitter fallback.
                    break
            emit_event.set()

    async def _worker() -> None:
        while True:
            if _cancelled():
                # Drain to avoid deadlocking the producer on a full queue.
                try:
                    item = sentence_queue.get_nowait()
                    sentence_queue.task_done()
                    if item is None:
                        break
                    continue
                except asyncio.QueueEmpty:
                    break
            try:
                item = await asyncio.wait_for(sentence_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                if state["producer_done"]:
                    break
                continue
            try:
                if item is None:
                    break
                idx, synth_text = item
                if state["tts_wall_start"] is None:
                    tracker.mark("t6_tts_start")
                    state["tts_wall_start"] = time.perf_counter()
                try:
                    audio_data, sr, synth_ms = await pipeline._synthesize_one(
                        synth_text, speaker, language
                    )
                except Exception as exc:
                    logger.error("TTS synth error (sentence %d): %s", idx, exc)
                    results[idx] = (state["display_texts"].get(idx, synth_text),
                                    np.zeros(0, dtype=np.float32),
                                    state["tts_sr"], 0.0)
                    emit_event.set()
                    continue
                state["tts_synth_total"] += synth_ms
                state["tts_wall_end"] = time.perf_counter()
                if tracker.stages.t7_first_audio == 0.0:
                    tracker.mark("t7_first_audio")
                try:
                    audio_len_s = len(audio_data) / float(sr) if sr else 0.0
                except Exception:
                    audio_len_s = 0.0
                tracker.note_synth(audio_len_s, synth_ms / 1000.0)
                state["tts_sr"] = sr
                results[idx] = (state["display_texts"].get(idx, synth_text),
                                audio_data, sr, synth_ms)
                emit_event.set()
            finally:
                try:
                    sentence_queue.task_done()
                except Exception:
                    pass

    all_waveforms: List[np.ndarray] = []
    ordered_texts: Dict[int, str] = {}
    emitted: List[Tuple[int, str]] = []
    first_emit_done = {"done": False}

    async def _emitter() -> None:
        next_idx = 0
        while True:
            if _cancelled():
                break
            if not emit_event.is_set():
                try:
                    await asyncio.wait_for(emit_event.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(0)
            emit_event.clear()
            # Emit everything available in order.
            while next_idx in results:
                display, audio_data, sr, _synth_ms = results.pop(next_idx)
                ordered_texts[next_idx] = display
                if len(audio_data) > 0:
                    all_waveforms.append(audio_data)
                    if should_play and stream_player is not None:
                        try:
                            stream_player.play_chunk(audio_data, sr)
                        except Exception as exc:
                            logger.warning("Playback enqueue warning: %s", exc)
                    if tracker.stages.t8_speaker_start == 0.0 and not should_play:
                        # No host playback; T8 ~= first chunk ready.
                        tracker.mark("t8_speaker_start")
                    if not first_emit_done["done"]:
                        first_emit_done["done"] = True
                        ttfa = round((time.perf_counter() - pipeline_start) * 1000, 2)
                        metrics.ttfa_ms = ttfa
                        logger.info(
                            "Time-To-First-Audio (TTFA): %.1fms (Sentence 0: '%s')",
                            ttfa, display[:50],
                        )
                # Chunk WAV bytes: single encode per sentence (no full-reply
                # re-encode here; final WAV encoded once below).
                try:
                    chunk_wav = AudioProcessor.to_wav_bytes(audio_data, sr) \
                        if len(audio_data) > 0 else b""
                except Exception:
                    chunk_wav = b""
                total = state["total"]
                prod_done = state["producer_done"]
                is_final = prod_done and (next_idx == total - 1)
                # When total unknown yet (producer running), is_final False;
                # fixed up after loop for the true last chunk.
                from app.core.pipeline import AssistantStreamChunk as _Chunk
                await pipeline._emit_chunk(
                    on_chunk,
                    _Chunk(sentence_index=next_idx, text=display,
                           audio_bytes=chunk_wav, sample_rate=sr,
                           is_final=is_final),
                )
                emitted.append((next_idx, display))
                next_idx += 1
            if state["producer_done"] and next_idx >= state["total"]:
                # Ensure at least one drain attempt after sentinels when the
                # queue momentarily held no sentinel (full-queue edge).
                if not results:
                    break
        # Fix is_final on the true last chunk is handled by consumers via
        # index; re-emit not needed (result frame carries full text).

    producer_task = asyncio.create_task(_producer())
    worker_tasks = [asyncio.create_task(_worker()) for _ in range(n_workers)]
    emitter_task = asyncio.create_task(_emitter())
    _my_tasks.update([producer_task, *worker_tasks, emitter_task])

    try:
        await producer_task
    except asyncio.CancelledError:
        state["producer_done"] = True
        emit_event.set()
    except Exception as exc:
        state["producer_error"] = exc
    # If producer filled the queue without room for sentinels, ensure workers exit.
    if state["producer_done"]:
        for _ in range(n_workers):
            try:
                sentence_queue.put_nowait(None)
            except asyncio.QueueFull:
                break
        emit_event.set()
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    # Final wake for emitter to flush trailing in-order results.
    emit_event.set()
    try:
        await asyncio.wait_for(emitter_task, timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("Emitter drain timed out; cancelling.")
        emitter_task.cancel()
    except asyncio.CancelledError:
        pass
    finally:
        try:
            if _task_registry.get(session_id) is _my_tasks:
                _task_registry.pop(session_id, None)
        except Exception:
            pass

    llm_elapsed = max(0.0, time.perf_counter() - (state["llm_start"] or pipeline_start))
    metrics.llm_latency_ms = round(llm_elapsed * 1000, 2)
    tracker.telemetry.llm_elapsed_s = llm_elapsed
    if state["tts_wall_start"] is not None and state["tts_wall_end"] is not None:
        metrics.tts_latency_ms = round(
            (state["tts_wall_end"] - state["tts_wall_start"]) * 1000, 2)
    metrics.tts_synth_ms = round(state["tts_synth_total"], 2)
    if should_play and stream_player is not None:
        try:
            stream_player.finish(wait=False)
        except Exception:
            pass

    if state["producer_error"] is not None and not ordered_texts:
        raise state["producer_error"]
    if _cancelled():
        # Interrupted: return partial work so callers can discard quickly.
        metrics.total_latency_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
        _fill_telemetry(pipeline, tracker, metrics, stream_player)
        pipeline.last_metrics = metrics
        partial = " ".join(ordered_texts[i] for i in sorted(ordered_texts)).strip()
        return partial, b"", state["tts_sr"], metrics

    if not ordered_texts:
        fallback_text = "I didn't receive a response. Please try again."
        try:
            audio_data, sr = await asyncio.to_thread(
                pipeline.tts.synthesize, text=fallback_text,
                speaker=speaker, language=language)
        except Exception:
            audio_data, sr = np.zeros(0, dtype=np.float32), state["tts_sr"]
        all_waveforms.append(audio_data) if len(audio_data) else None
        ordered_texts[0] = fallback_text
        if metrics.ttfa_ms == 0.0:
            metrics.ttfa_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
        if tracker.stages.t8_speaker_start == 0.0:
            tracker.mark("t8_speaker_start")

    # Assistant text preserves readable order.
    assistant_reply = " ".join(
        ordered_texts[i] for i in sorted(ordered_texts)).strip()
    await pipeline._append_history(session_id, prompt, assistant_reply)

    if all_waveforms:
        combined = np.concatenate(all_waveforms, axis=0).astype(np.float32)
        wav_bytes = AudioProcessor.to_wav_bytes(combined, state["tts_sr"])
    else:
        wav_bytes = b""

    # Stage-derived headline latencies (T0..T8) + resource telemetry.
    lats = tracker.stages.as_latencies_ms()
    metrics.llm_ttft_ms = lats["llm_ttft_ms"]
    metrics.sentence_latency_ms = lats["sentence_latency_ms"]
    if metrics.ttfa_ms == 0.0:
        metrics.ttfa_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
    metrics.voice_latency_ms = lats["voice_latency_ms"] or metrics.ttfa_ms
    _fill_telemetry(pipeline, tracker, metrics, stream_player)
    pipeline.last_metrics = metrics
    return assistant_reply, wav_bytes, state["tts_sr"], metrics


def _fill_telemetry(pipeline: Any, tracker: Any, metrics: Any, stream_player: Any) -> None:
    """Copies throughput + host counters into ``metrics`` (best-effort)."""
    try:
        depth = stream_player.depth if stream_player is not None else 0
    except Exception:
        depth = 0
    try:
        tel = tracker.finalize_system_stats(audio_queue_depth=depth)
    except Exception:
        return
    try:
        metrics.tokens_per_sec = tel.tokens_per_sec
        metrics.tts_realtime_factor = tel.tts_realtime_factor
        metrics.audio_queue_depth = int(tel.audio_queue_depth)
        metrics.system_ram_mb = float(tel.system_ram_mb)
        metrics.cpu_pct = float(tel.cpu_pct)
        metrics.gpu_util_pct = float(tel.gpu_util_pct)
        metrics.gpu_vram_mb = float(tel.gpu_vram_mb)
        metrics.npu_status = str(tel.npu_status)
    except Exception:
        pass


async def _run_non_streaming(
    pipeline: Any, prompt: str, speaker: Any, language: Any,
    history_slice: List[Dict[str, str]], should_play: bool,
    stream_player: Any, session_id: str, my_gen: int,
    pipeline_start: float, tracker: Any, metrics: Any,
    on_chunk: Any,
) -> Tuple[str, bytes, int, Any]:
    """Fallback for LLM clients without any stream (single generate + synth)."""
    import numpy as np
    from app.core.audio import AudioProcessor
    from app.core.sentences import clean_text_for_speech

    tracker.mark("t3_llm_send")
    llm_start = time.perf_counter()
    reply = await pipeline.llm.generate_response(prompt=prompt, history=history_slice)
    metrics.llm_latency_ms = round((time.perf_counter() - llm_start) * 1000, 2)
    tracker.mark("t4_first_token")
    tracker.mark("t5_first_sentence")
    tracker.telemetry.llm_tokens += max(1, len(reply) // 4)
    tracker.telemetry.llm_chars += len(reply)
    tracker.telemetry.llm_elapsed_s = max(0.01, time.perf_counter() - llm_start)

    synth_text = clean_text_for_speech(reply) or reply
    tracker.mark("t6_tts_start")
    tts_start = time.perf_counter()
    audio_data, sr = await asyncio.to_thread(
        pipeline.tts.synthesize, text=synth_text, speaker=speaker, language=language)
    wall = (time.perf_counter() - tts_start) * 1000
    metrics.tts_latency_ms = round(wall, 2)
    metrics.tts_synth_ms = round(wall, 2)
    tracker.mark("t7_first_audio")
    try:
        tracker.note_synth(len(audio_data) / float(sr), wall / 1000.0)
    except Exception:
        pass
    metrics.ttfa_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
    if should_play and stream_player is not None:
        try:
            stream_player.play_chunk(audio_data, sr)
            stream_player.finish(wait=False)
        except Exception:
            pass
    else:
        tracker.mark("t8_speaker_start")
    lats = tracker.stages.as_latencies_ms()
    metrics.llm_ttft_ms = lats["llm_ttft_ms"]
    metrics.sentence_latency_ms = lats["sentence_latency_ms"]
    metrics.voice_latency_ms = lats["voice_latency_ms"] or metrics.ttfa_ms
    _fill_telemetry(pipeline, tracker, metrics, stream_player)

    await pipeline._append_history(session_id, prompt, reply)
    if len(audio_data) > 0:
        wav_bytes = AudioProcessor.to_wav_bytes(audio_data, sr)
        from app.core.pipeline import AssistantStreamChunk as _Chunk
        await pipeline._emit_chunk(
            on_chunk, _Chunk(sentence_index=0, text=reply,
                             audio_bytes=wav_bytes, sample_rate=sr, is_final=True))
    else:
        wav_bytes = b""
    pipeline.last_metrics = metrics
    return reply, wav_bytes, sr, metrics
