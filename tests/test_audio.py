"""Tests for audio loading, conversion, and playback utilities."""

import numpy as np
import pytest
from app.core.audio import AudioProcessor


def test_audio_to_wav_and_back() -> None:
    """Tests round-trip encoding and decoding of audio numpy arrays."""
    sample_rate = 16000
    duration = 0.5
    # Generate 0.5s 440 Hz test tone
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    original_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    wav_bytes = AudioProcessor.to_wav_bytes(original_audio, sample_rate)
    assert len(wav_bytes) > 0
    assert wav_bytes.startswith(b"RIFF")

    decoded_audio, decoded_sr = AudioProcessor.load_from_bytes(
        wav_bytes, target_sr=16000
    )
    assert decoded_sr == 16000
    assert len(decoded_audio) == len(original_audio)
    assert np.allclose(decoded_audio, original_audio, atol=1e-3)


def test_audio_empty_bytes_raises() -> None:
    """Verifies that empty audio bytes raise a ValueError."""
    with pytest.raises(ValueError):
        AudioProcessor.load_from_bytes(b"")


def test_query_audio_devices() -> None:
    """Tests querying audio devices on the host."""
    dev_info = AudioProcessor.get_audio_devices()
    assert isinstance(dev_info, dict)
    assert "devices" in dev_info


def test_stream_audio_player_queue() -> None:
    """Tests StreamAudioPlayer queueing, finish, and stop operations."""
    from unittest.mock import patch
    from app.core.audio import StreamAudioPlayer

    player = StreamAudioPlayer()
    assert not player.is_playing

    fake_samples = np.zeros(2400, dtype=np.float32)
    with patch("sounddevice.play") as mock_play, patch("sounddevice.wait"):
        player.play_chunk(fake_samples, 24000)
        player.finish(wait=True)
        assert mock_play.called

    player.stop()
    assert not player.is_playing

