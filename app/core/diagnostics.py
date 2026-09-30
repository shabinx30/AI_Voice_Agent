"""OpenVINO / Kokoro device diagnostics (uses the installed OpenVINO API).

Never claims NPU execution unless the runtime confirms it. All property
probes go through ``ov.Core().get_property`` with per-key try/except so the
module works across OpenVINO versions without hardcoding properties that
may not exist.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_openvino_version() -> str:
    """Returns the installed OpenVINO runtime version string."""
    try:
        import openvino as ov

        return str(getattr(ov, "__version__", "unknown"))
    except Exception as exc:
        return f"unavailable ({exc})"


def get_available_devices() -> List[str]:
    """Returns OpenVINO available devices (e.g. ['CPU', 'GPU', 'NPU'])."""
    try:
        import openvino as ov

        return list(ov.Core().available_devices)
    except Exception as exc:
        logger.warning("Failed to query OpenVINO devices: %s", exc)
        return ["CPU"]


def _safe_device_prop(core: Any, device: str, prop: str) -> Optional[Any]:
    try:
        return core.get_property(device, prop)
    except Exception:
        return None


def get_device_details() -> Dict[str, Dict[str, Any]]:
    """Returns per-device full name, arch, capabilities (best-effort)."""
    details: Dict[str, Dict[str, Any]] = {}
    try:
        import openvino as ov

        core = ov.Core()
        for dev in core.available_devices:
            info: Dict[str, Any] = {}
            for prop in (
                "FULL_DEVICE_NAME",
                "DEVICE_ARCHITECTURE",
                "DEVICE_TYPE",
                "OPTIMIZATION_CAPABILITIES",
                "PERFORMANCE_HINT",
                "RANGE_FOR_ASYNC_INFER_REQUESTS",
                "RANGE_FOR_STREAMS",
            ):
                val = _safe_device_prop(core, dev, prop)
                if val is not None:
                    info[prop.lower()] = str(val)[:300]
            details[dev] = info
    except Exception as exc:
        logger.debug("Device details probe failed: %s", exc)
    return details


def get_npu_supported_properties() -> Dict[str, str]:
    """Returns NPU SUPPORTED_PROPERTIES keys (empty when NPU absent)."""
    try:
        import openvino as ov

        core = ov.Core()
        if "NPU" not in core.available_devices:
            return {}
        props = core.get_property("NPU", "SUPPORTED_PROPERTIES")
        if isinstance(props, dict):
            return {k: str(v) for k, v in props.items()}
        return {}
    except Exception as exc:
        logger.debug("NPU SUPPORTED_PROPERTIES probe failed: %s", exc)
        return {}


def get_npu_runtime_state() -> Dict[str, Any]:
    """Returns live NPU property values for the keys we care about."""
    state: Dict[str, Any] = {}
    try:
        import openvino as ov

        core = ov.Core()
        if "NPU" not in core.available_devices:
            state["available"] = False
            state["reason"] = "NPU not in available_devices"
            return state
        state["available"] = True
        for prop in (
            "FULL_DEVICE_NAME",
            "DEVICE_ARCHITECTURE",
            "PERFORMANCE_HINT",
            "PERFORMANCE_HINT_NUM_REQUESTS",
            "OPTIMAL_NUMBER_OF_INFER_REQUESTS",
            "NUM_STREAMS",
            "CACHE_DIR",
            "CACHE_MODE",
            "NPU_TURBO",
            "NPU_COMPILATION_MODE_PARAMS",
            "INFERENCE_PRECISION_HINT",
            "EXECUTION_MODE_HINT",
            "WORKLOAD_TYPE",
        ):
            val = _safe_device_prop(core, "NPU", prop)
            if val is not None:
                state[prop.lower()] = str(val)
    except Exception as exc:
        state["available"] = False
        state["reason"] = str(exc)[:300]
    return state


def build_npu_compile_config(
    cache_dir: str = "",
    performance_hint: str = "LATENCY",
    turbo: Optional[bool] = None,
    num_requests: int = 1,
) -> Dict[str, Any]:
    """Builds an NPU compile config using ONLY supported properties.

    Queries ``SUPPORTED_PROPERTIES`` on the installed runtime and drops any
    key the NPU plugin does not advertise, so callers never hit
    "unsupported property" errors on other OpenVINO versions.

    Args:
        cache_dir: On-disk compiler cache (enables fast warm loads).
        performance_hint: LATENCY (TTFA-optimized) or THROUGHPUT.
        turbo: True/False to force NPU_TURBO, None to leave at driver default.
        num_requests: PERFORMANCE_HINT_NUM_REQUESTS (1 = lowest latency).

    Returns:
        Dict safe to pass as ``compile_model(..., config)``.
    """
    supported = get_npu_supported_properties()
    config: Dict[str, Any] = {}
    if not supported:
        return config
    if "CACHE_DIR" in supported and cache_dir:
        config["CACHE_DIR"] = cache_dir
    if "PERFORMANCE_HINT" in supported and performance_hint:
        config["PERFORMANCE_HINT"] = performance_hint
    if (
        "PERFORMANCE_HINT_NUM_REQUESTS" in supported
        and num_requests is not None
    ):
        config["PERFORMANCE_HINT_NUM_REQUESTS"] = str(int(num_requests))
    if "NPU_TURBO" in supported and turbo is not None:
        # Verified on OV 2026.4: bool and "YES"/"NO" both accepted.
        config["NPU_TURBO"] = bool(turbo)
    return config


def get_kokoro_device_report(tts_engine: Any = None) -> Dict[str, Any]:
    """Reports which device Kokoro is ACTUALLY using (runtime-confirmed).

    Never reports NPU unless the engine's compiled stages confirm it.
    Falls back to an explicit reason string when NPU was requested but
    the runtime placed stages on CPU.
    """
    from app.config import settings

    report: Dict[str, Any] = {
        "openvino_version": get_openvino_version(),
        "available_devices": get_available_devices(),
        "requested_device": str(getattr(settings, "tts_device", "unknown")),
        "backend": str(getattr(settings, "tts_backend", "auto")),
        "npu_plugin_available": "NPU" in get_available_devices(),
    }
    if tts_engine is not None:
        report["backend"] = str(getattr(tts_engine, "backend", report["backend"]))
        report["requested_device"] = str(
            getattr(tts_engine, "device", report["requested_device"])
        )
        eff = getattr(tts_engine, "effective_device", None)
        report["effective_device"] = eff or "not-yet-loaded"
        stage_devices = getattr(
            getattr(tts_engine, "_npu_encoder", None), "stage_devices", None
        )
        if stage_devices:
            report["stage_devices"] = dict(stage_devices)
            npu_stages = sum(1 for d in stage_devices.values() if d == "NPU")
            report["npu_stage_count"] = npu_stages
            if npu_stages == 0 and str(
                getattr(tts_engine, "device", "")
            ).upper() == "NPU":
                report["fallback_reason"] = (
                    "NPU requested but all encoder stages compiled on CPU; "
                    "see logs for per-stage compile errors."
                )
        npu_attempted = getattr(tts_engine, "_npu_attempted", False)
        report["npu_encoder_attempted"] = bool(npu_attempted)
        if eff and "NPU" in str(eff) and stage_devices:
            report["confirmed_on_npu"] = any(
                d == "NPU" for d in stage_devices.values()
            )
        elif eff:
            report["confirmed_on_npu"] = "NPU" in str(eff)
        else:
            report["confirmed_on_npu"] = False
    report["npu_runtime"] = get_npu_runtime_state()
    return report


def format_diagnostic_text(report: Dict[str, Any]) -> str:
    """Formats a human-readable diagnostic block (CLI / logs)."""
    lines = [
        f"OpenVINO version: {report.get('openvino_version')}",
        f"Available devices: {', '.join(report.get('available_devices', []))}",
        f"Kokoro requested device: {report.get('requested_device')}",
        f"Kokoro backend: {report.get('backend')}",
        f"Kokoro effective device: {report.get('effective_device', 'unknown')}",
    ]
    if "stage_devices" in report:
        lines.append(f"Kokoro stage devices: {report['stage_devices']}")
    if "fallback_reason" in report:
        lines.append(f"Fallback reason: {report['fallback_reason']}")
    npu = report.get("npu_runtime", {})
    if npu.get("available"):
        lines.append(
            "NPU: available "
            f"({npu.get('full_device_name', '')} "
            f"{npu.get('device_architecture', '')}) "
            f"TURBO={npu.get('npu_turbo', '?')} "
            f"PERF_HINT={npu.get('performance_hint', '?')}"
        )
    else:
        lines.append(f"NPU: unavailable ({npu.get('reason', 'unknown')})")
    return "\n".join(lines)
