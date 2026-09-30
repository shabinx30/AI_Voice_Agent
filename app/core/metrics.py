"""Detailed latency + resource telemetry for the voice pipeline.

Tracks the T0..T8 stage timestamps required by the latency spec:

    T0 = microphone input ends (audio bytes ready)
    T1 = Whisper starts
    T2 = Whisper finishes (STT latency = T2 - T1)
    T3 = request sent to LM Studio
    T4 = first LLM token received (TTFT = T4 - T3)
    T5 = first complete sentence detected (sentence = T5 - T4)
    T6 = Kokoro starts (first synth)
    T7 = first audio generated (TTS = T7 - T6)
    T8 = speaker starts (voice latency = T8 - T0)

Plus throughput counters (tokens/sec, TTS realtime factor) and host
resource gauges (RAM, CPU, GPU VRAM where queryable, NPU status string,
audio queue depth). All helpers are best-effort and never raise: telemetry
must never break a voice turn.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class StageTimestamps:
    """Wall-clock/perf-counter marks for one voice turn (seconds)."""

    t0_mic_end: float = 0.0
    t1_stt_start: float = 0.0
    t2_stt_end: float = 0.0
    t3_llm_send: float = 0.0
    t4_first_token: float = 0.0
    t5_first_sentence: float = 0.0
    t6_tts_start: float = 0.0
    t7_first_audio: float = 0.0
    t8_speaker_start: float = 0.0

    def as_latencies_ms(self) -> Dict[str, float]:
        """Derives the headline latency numbers in milliseconds."""
        def _ms(a: float, b: float) -> float:
            if a <= 0.0 or b <= 0.0:
                return 0.0
            return round((b - a) * 1000.0, 2)

        return {
            "stt_latency_ms": _ms(self.t1_stt_start, self.t2_stt_end),
            "llm_ttft_ms": _ms(self.t3_llm_send, self.t4_first_token),
            "sentence_latency_ms": _ms(self.t4_first_token, self.t5_first_sentence),
            "tts_latency_ms": _ms(self.t6_tts_start, self.t7_first_audio),
            "voice_latency_ms": _ms(self.t0_mic_end, self.t8_speaker_start),
        }


@dataclass
class TurnTelemetry:
    """Throughput + resource snapshot attached to a pipeline turn."""

    llm_tokens: int = 0
    llm_chars: int = 0
    llm_elapsed_s: float = 0.0
    tts_audio_s: float = 0.0
    tts_synth_s: float = 0.0
    audio_queue_depth: int = 0
    system_ram_mb: float = 0.0
    system_ram_pct: float = 0.0
    cpu_pct: float = 0.0
    gpu_util_pct: float = 0.0
    gpu_vram_mb: float = 0.0
    npu_status: str = "unknown"

    @property
    def tokens_per_sec(self) -> float:
        """LLM decode throughput (tokens/sec, 0 when unknown)."""
        if self.llm_elapsed_s <= 0.0:
            return 0.0
        return round(self.llm_tokens / self.llm_elapsed_s, 2)

    @property
    def tts_realtime_factor(self) -> float:
        """Audio-seconds generated per synthesis-second (>1 = faster than realtime)."""
        if self.tts_synth_s <= 0.0:
            return 0.0
        return round(self.tts_audio_s / self.tts_synth_s, 2)

    def as_dict(self) -> Dict[str, float | str]:
        """Flattens telemetry into a JSON-serializable dict."""
        return {
            "tokens_per_sec": self.tokens_per_sec,
            "tts_realtime_factor": self.tts_realtime_factor,
            "llm_tokens": self.llm_tokens,
            "audio_queue_depth": self.audio_queue_depth,
            "system_ram_mb": round(self.system_ram_mb, 1),
            "system_ram_pct": round(self.system_ram_pct, 1),
            "cpu_pct": round(self.cpu_pct, 1),
            "gpu_util_pct": round(self.gpu_util_pct, 1),
            "gpu_vram_mb": round(self.gpu_vram_mb, 1),
            "npu_status": self.npu_status,
        }


def sample_system_stats() -> Dict[str, float | str]:
    """Samples host RAM/CPU/GPU counters on a best-effort basis.

    Returns:
        Dict with system_ram_mb, system_ram_pct, cpu_pct, gpu_util_pct,
        gpu_vram_mb, npu_status. Missing providers yield 0.0 / "unknown".
    """
    stats: Dict[str, float | str] = {
        "system_ram_mb": 0.0,
        "system_ram_pct": 0.0,
        "cpu_pct": 0.0,
        "gpu_util_pct": 0.0,
        "gpu_vram_mb": 0.0,
        "npu_status": "unknown",
    }
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        stats["system_ram_mb"] = float(vm.used / (1024 * 1024))
        stats["system_ram_pct"] = float(vm.percent)
        try:
            stats["cpu_pct"] = float(psutil.cpu_percent(interval=None))
        except Exception:
            pass
        try:
            proc = psutil.Process(os.getpid())
            stats["process_ram_mb"] = float(proc.memory_info().rss / (1024 * 1024))
        except Exception:
            pass
    except Exception as exc:
        logger.debug("psutil stats unavailable: %s", exc)

    # NVIDIA GPUs via nvidia-smi (Windows + Linux). Intel Arc has no stable
    # CLI query here; leave 0.0 rather than guessing.
    try:
        import subprocess

        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            first = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 2:
                stats["gpu_util_pct"] = float(parts[0])
                stats["gpu_vram_mb"] = float(parts[1])
    except Exception as exc:
        logger.debug("nvidia-smi query skipped: %s", exc)

    # NPU presence via OpenVINO device enumeration (cheap, cached by caller).
    try:
        import openvino as ov  # type: ignore

        devices = ov.Core().available_devices
        stats["npu_status"] = "available" if "NPU" in devices else "unavailable"
    except Exception:
        pass
    return stats


@dataclass
class LatencyTracker:
    """Mutable helper accumulating T0..T8 marks plus throughput counters."""

    stages: StageTimestamps = field(default_factory=StageTimestamps)
    telemetry: TurnTelemetry = field(default_factory=TurnTelemetry)
    _start: float = field(default_factory=time.perf_counter)

    def mark(self, name: str) -> float:
        """Records ``time.perf_counter()`` under stage ``name``.

        Args:
            name: One of t0..t8 field names (e.g. "t4_first_token").

        Returns:
            Recorded timestamp value.
        """
        now = time.perf_counter()
        if hasattr(self.stages, name):
            setattr(self.stages, name, now)
        return now

    def note_token(self, token: str) -> None:
        """Accounts one raw LLM token delta toward tokens/sec."""
        if token:
            self.telemetry.llm_tokens += 1
            self.telemetry.llm_chars += len(token)

    def note_synth(self, audio_seconds: float, synth_seconds: float) -> None:
        """Accumulates TTS audio/ synth time for realtime-factor math."""
        self.telemetry.tts_audio_s += max(0.0, float(audio_seconds))
        self.telemetry.tts_synth_s += max(0.0, float(synth_seconds))

    def finalize_system_stats(self, audio_queue_depth: int = 0) -> TurnTelemetry:
        """Samples host counters into telemetry (call once per turn)."""
        self.telemetry.audio_queue_depth = int(audio_queue_depth)
        for key, value in sample_system_stats().items():
            if hasattr(self.telemetry, key):
                try:
                    setattr(self.telemetry, key, value)
                except Exception:
                    pass
        return self.telemetry
