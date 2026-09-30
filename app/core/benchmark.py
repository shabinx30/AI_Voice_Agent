"""Kokoro TTS device benchmark: CPU vs GPU vs NPU (same fixed text).

Measures, per device:
  - model compilation/load time (cold, reported separately)
  - warm-up inference time
  - steady-state generation time (mean/p50 over measured iters)
  - generated audio duration + RTFx (audio_s / gen_s)
  - RAM / CPU telemetry (best-effort via psutil)

Steady-state numbers NEVER include model init: several warm-up iterations
run first, then multiple measured iterations. NPU turbo OFF vs ON is
benchmarked when the installed NPU plugin advertises NPU_TURBO; the faster
setting wins (never assumed).

GPU note: the torch Kokoro backend has no CUDA/XPU provider in this env, so
a "gpu" request falls back to optimized CPU execution (logged explicitly).
The OpenVINO backend can still target GPU via the OV plugin when available.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_BENCH_TEXT = (
    "Hello there, how are you today? This is a fixed benchmark sentence "
    "for measuring Kokoro speech synthesis speed."
)


@dataclass
class KokoroBenchResult:
    """One device's benchmark numbers."""

    device: str
    backend: str
    effective_device: str
    compile_s: float
    warmup_s: float
    gen_mean_s: float
    gen_p50_s: float
    audio_s: float
    rtfx: float
    iters: int
    stage_devices: Optional[Dict[str, str]] = None
    turbo: Optional[str] = None
    fallback_reason: str = ""
    system_ram_mb: float = 0.0
    cpu_pct: float = 0.0


def _sample_stats() -> Dict[str, float]:
    try:
        from app.core.metrics import sample_system_stats

        s = sample_system_stats()
        return {
            "system_ram_mb": float(s.get("system_ram_mb", 0.0)),
            "cpu_pct": float(s.get("cpu_pct", 0.0)),
        }
    except Exception:
        return {"system_ram_mb": 0.0, "cpu_pct": 0.0}


def _make_engine(device: str, turbo: Optional[bool] = None):
    """Builds a KokoroTTSEngine for ``device`` with explicit backend choice."""
    from app.config import settings
    from app.core.tts import KokoroTTSEngine

    dev = device.lower()
    if dev == "npu":
        eng = KokoroTTSEngine(device="npu")
        eng.backend = "openvino"  # explicit: torch has no NPU provider
    elif dev == "gpu":
        # Prefer OpenVINO GPU when the INT8 export is usable; fall back to
        # torch (which logs its own CPU fallback) only if OV GPU fails.
        # For a fair latency comparison we benchmark the torch path here
        # (legacy behavior) and let operators compare with TTS_BACKEND=openvino.
        eng = KokoroTTSEngine(device="gpu")
    else:
        eng = KokoroTTSEngine(device="cpu")
    if turbo is not None and hasattr(eng, "_npu_encoder"):
        pass  # turbo is read from settings at encoder build time
    return eng


def benchmark_one_device(
    device: str,
    text: str = DEFAULT_BENCH_TEXT,
    warmup_iters: int = 2,
    measure_iters: int = 5,
    speaker: Optional[str] = None,
    turbo: Optional[bool] = None,
) -> KokoroBenchResult:
    """Benchmarks one device end-to-end (fresh engine instance)."""
    import os

    # Turbo is a compile-time property: set env override for this engine.
    # KokoroNPUEncoder reads settings.tts_npu_turbo at build time.
    prev_turbo = None
    if turbo is not None and device.lower() == "npu":
        from app.config import settings as _s

        prev_turbo = getattr(_s, "tts_npu_turbo", "auto")
        try:
            object.__setattr__(_s, "tts_npu_turbo", "on" if turbo else "off")
        except Exception:
            pass

    try:
        eng = _make_engine(device, turbo)
        # Honest steady-state: bypass the production LRU phrase cache, or
        # repeated fixed-text iters would hit cache (≈0ms) and fake RTFx.
        try:
            eng._cache_max = 0
            eng._cache.clear()
        except Exception:
            pass
        t0 = time.perf_counter()
        eng.load_model()
        compile_s = time.perf_counter() - t0

        # Warm-up (excluded from steady-state): also triggers lazy NPU compile.
        tw0 = time.perf_counter()
        sr = 24000
        for _ in range(max(1, warmup_iters)):
            audio, sr = eng.synthesize(text, speaker=speaker)
        warmup_s = (time.perf_counter() - tw0) / max(1, warmup_iters)

        audio_s = len(audio) / float(sr) if len(audio) else 0.0

        gens: List[float] = []
        for _ in range(max(1, measure_iters)):
            tg0 = time.perf_counter()
            audio, sr = eng.synthesize(text, speaker=speaker)
            gens.append(time.perf_counter() - tg0)
        audio_s = len(audio) / float(sr) if len(audio) else audio_s
        gens_sorted = sorted(gens)
        gen_mean = sum(gens) / len(gens)
        gen_p50 = gens_sorted[len(gens_sorted) // 2]
        rtfx = audio_s / gen_mean if gen_mean > 0 else 0.0
        stats = _sample_stats()
        stage_devices = None
        try:
            enc = getattr(eng, "_npu_encoder", None)
            if enc is not None and getattr(enc, "stage_devices", None):
                stage_devices = dict(enc.stage_devices)
        except Exception:
            pass
        fallback = ""
        eff = str(getattr(eng, "effective_device", "") or "")
        if device.lower() == "npu" and "NPU" not in eff:
            fallback = (
                "NPU requested but runtime placed stages on CPU "
                f"(effective={eff}, stages={stage_devices})."
            )
        if device.lower() == "gpu" and eff.upper() == "CPU":
            fallback = (
                "GPU requested but torch backend has no CUDA/XPU provider; "
                "ran on optimized CPU."
            )
        return KokoroBenchResult(
            device=device.upper(),
            backend=str(getattr(eng, "backend", "?")),
            effective_device=eff or "unknown",
            compile_s=round(compile_s, 3),
            warmup_s=round(warmup_s, 3),
            gen_mean_s=round(gen_mean, 3),
            gen_p50_s=round(gen_p50, 3),
            audio_s=round(audio_s, 3),
            rtfx=round(rtfx, 2),
            iters=max(1, measure_iters),
            stage_devices=stage_devices,
            turbo=("on" if turbo else "off") if turbo is not None else None,
            fallback_reason=fallback,
            system_ram_mb=round(stats["system_ram_mb"], 1),
            cpu_pct=round(stats["cpu_pct"], 1),
        )
    finally:
        if prev_turbo is not None:
            try:
                from app.config import settings as _s

                object.__setattr__(_s, "tts_npu_turbo", prev_turbo)
            except Exception:
                pass


def npu_turbo_supported() -> bool:
    """True when the installed NPU plugin advertises NPU_TURBO."""
    try:
        from app.core.diagnostics import get_npu_supported_properties

        return "NPU_TURBO" in get_npu_supported_properties()
    except Exception:
        return False


def benchmark_all_devices(
    text: str = DEFAULT_BENCH_TEXT,
    devices: Optional[List[str]] = None,
    warmup_iters: int = 2,
    measure_iters: int = 3,
    include_turbo_comparison: bool = True,
) -> List[KokoroBenchResult]:
    """Benchmarks CPU/GPU/NPU (whichever are available) + NPU turbo on/off."""
    from app.core.diagnostics import get_available_devices

    available = set(d.upper() for d in get_available_devices())
    wanted = [d.upper() for d in (devices or ["CPU", "GPU", "NPU"])]
    results: List[KokoroBenchResult] = []
    for dev in wanted:
        if dev not in available and dev != "CPU":
            logger.warning("Device %s not available; skipping.", dev)
            continue
        if dev == "NPU" and include_turbo_comparison and npu_turbo_supported():
            # Benchmark TURBO OFF vs ON, keep both rows; caller picks winner.
            for turbo in (False, True):
                try:
                    results.append(
                        benchmark_one_device(
                            dev, text, warmup_iters, measure_iters, turbo=turbo
                        )
                    )
                except Exception as exc:
                    logger.error("NPU turbo=%s bench failed: %s", turbo, exc)
        else:
            try:
                results.append(
                    benchmark_one_device(dev, text, warmup_iters, measure_iters)
                )
            except Exception as exc:
                logger.error("Bench %s failed: %s", dev, exc)
    return results


def format_results_table(results: List[KokoroBenchResult]) -> str:
    """Formats Device | Gen | Audio | RTFx table (steady-state only)."""
    header = (
        f"{'Device':<10} {'Backend':<9} {'Eff':<9} {'Compile':>8} "
        f"{'Warm':>7} {'Gen':>7} {'Audio':>7} {'RTFx':>7}  Notes"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        notes = ""
        if r.turbo:
            notes += f"turbo={r.turbo} "
        if r.stage_devices:
            notes += f"{r.stage_devices} "
        if r.fallback_reason:
            notes += f"[{r.fallback_reason[:90]}]"
        lines.append(
            f"{r.device:<10} {r.backend:<9} {r.effective_device:<9} "
            f"{r.compile_s:>8.3f} {r.warmup_s:>7.3f} {r.gen_mean_s:>7.3f} "
            f"{r.audio_s:>7.2f} {r.rtfx:>7.2f}x  {notes}"
        )
    lines.append("")
    lines.append("RTFx = audio_duration / generation_time (higher = faster).")
    lines.append("Compile/warm excluded from steady-state Gen.")
    return "\n".join(lines)


def results_to_dicts(results: List[KokoroBenchResult]) -> List[Dict[str, Any]]:
    """JSON-serializable benchmark rows (for API responses)."""
    return [asdict(r) for r in results]
