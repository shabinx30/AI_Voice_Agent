"""Tests for Kokoro-82M speech synthesis module."""

import numpy as np
import pytest
from app.core.tts import KokoroTTSEngine, QwenTTSEngine


def test_tts_engine_initialization() -> None:
    """Verifies that KokoroTTSEngine initializes with expected voice personas."""
    engine = KokoroTTSEngine(
        model_id="hexgrad/Kokoro-82M",
        speaker="af_heart",
    )
    assert engine.speaker == "af_heart"
    assert engine.sample_rate == 24000
    speakers = engine.get_supported_speakers()
    assert "af_heart" in speakers
    assert "am_adam" in speakers
    assert "af_bella" in speakers


def test_qwen_tts_engine_alias() -> None:
    """Verifies QwenTTSEngine backward compatibility alias."""
    assert QwenTTSEngine is KokoroTTSEngine


def test_tts_npu_selects_openvino_backend() -> None:
    """Verifies an NPU request routes Kokoro to the OpenVINO backend."""
    engine = KokoroTTSEngine(device="npu", speaker="af_heart")
    assert engine.backend == "openvino"
    assert engine.device == "npu"
    assert engine.effective_device is None  # Set during load_model()
    assert "Kokoro" in engine._ov_model_id


def test_tts_cpu_keeps_torch_backend(monkeypatch) -> None:
    """Verifies CPU/GPU requests preserve the legacy PyTorch backend under auto."""
    from app.config import settings

    monkeypatch.setattr(settings, "tts_backend", "auto")
    assert KokoroTTSEngine(device="cpu").backend == "torch"
    assert KokoroTTSEngine(device="gpu").backend == "torch"
    assert KokoroTTSEngine(device="NPU").backend == "openvino"


def test_tts_empty_text_raises() -> None:
    """Verifies empty input text raises ValueError."""
    engine = KokoroTTSEngine()
    with pytest.raises(ValueError):
        engine.synthesize("   ")


def test_fallback_synthesize() -> None:
    """Tests the fallback speech synthesis mechanism."""
    engine = KokoroTTSEngine()
    data, sr = engine._fallback_synthesize("Fallback test")
    assert isinstance(data, np.ndarray)
    assert sr > 0
    assert len(data) > 0


def test_tts_set_device_cpu_npu() -> None:
    """Tests dynamic switching between CPU and NPU processing units."""
    from unittest.mock import MagicMock

    engine = KokoroTTSEngine(device="cpu")
    engine.load_model = MagicMock()
    info = engine.get_device_info()
    assert info["device"] == "cpu"
    assert "cpu" in info["available_devices"]

    # Switch to NPU
    res_npu = engine.set_device("NPU")
    assert res_npu["device"] == "npu"
    assert engine.device == "npu"
    assert engine.backend == "openvino"

    # Switch back to CPU
    res_cpu = engine.set_device("cpu")
    assert res_cpu["device"] == "cpu"
    assert engine.device == "cpu"
    assert res_cpu["effective_device"] == "CPU"


def test_tts_set_device_invalid_raises() -> None:
    """Verifies that setting an unsupported device raises ValueError."""
    engine = KokoroTTSEngine(device="cpu")
    with pytest.raises(ValueError, match="Unsupported TTS device"):
        engine.set_device("quantum")

