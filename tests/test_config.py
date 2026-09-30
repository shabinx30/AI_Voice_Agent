"""Tests for application settings and configuration."""

import pytest
from app.config import Settings


def test_default_settings() -> None:
    """Verifies default configuration values conform to requirements."""
    settings = Settings()
    assert settings.stt_model_id == "OpenVINO/whisper-base-int8-ov"
    assert settings.stt_device.upper() in ["CPU", "GPU", "NPU", "AUTO"]
    assert settings.lm_studio_base_url.startswith("http")
    assert "qwen" in settings.lm_studio_model.lower()
    assert "kokoro" in settings.tts_model_id.lower()
    assert settings.audio_sample_rate == 16000


def test_custom_settings_override() -> None:
    """Tests environment variable or argument overrides on settings."""
    settings = Settings(
        stt_device="GPU",
        server_port=9000,
        tts_speaker="vivian",
    )
    assert settings.stt_device == "GPU"
    assert settings.server_port == 9000
    assert settings.tts_speaker == "vivian"
