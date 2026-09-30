"""Tests for OpenVINO Whisper Base INT8 Speech-to-Text module."""

import numpy as np
import pytest
from app.core.stt import OpenVINOWhisperSTT


def test_stt_initialization() -> None:
    """Verifies that OpenVINOWhisperSTT initializes with expected attributes."""
    stt = OpenVINOWhisperSTT(
        model_id="OpenVINO/whisper-base-int8-ov",
        device="CPU",
    )
    assert stt.model_id == "OpenVINO/whisper-base-int8-ov"
    assert stt.device == "CPU"
    assert not stt.is_loaded


def test_stt_get_devices() -> None:
    """Tests discovery of OpenVINO accelerator devices."""
    stt = OpenVINOWhisperSTT()
    devices = stt.get_openvino_devices()
    assert isinstance(devices, list)
    assert len(devices) > 0
    assert "CPU" in devices


def test_stt_empty_audio_raises() -> None:
    """Verifies that attempting transcription on empty array raises ValueError."""
    stt = OpenVINOWhisperSTT()
    # Mock loaded state to trigger validation check
    stt._is_loaded = True
    stt.model = object()  # type: ignore
    stt.processor = object()  # type: ignore

    with pytest.raises(ValueError):
        stt.transcribe(np.array([], dtype=np.float32))
