"""Tests for FastAPI HTTP routes."""

import pytest
from starlette.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Tests the /api/health status endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "OpenVINO" in data["stt_model"]
    assert "Kokoro" in data["tts_model"]
    assert isinstance(data["tts_speakers"], list)


def test_devices_endpoint() -> None:
    """Tests the /api/devices discovery endpoint."""
    response = client.get("/api/devices")
    assert response.status_code == 200
    data = response.json()
    assert "openvino_devices" in data
    assert "audio_devices" in data


def test_speakers_endpoint() -> None:
    """Tests the /api/speakers endpoint."""
    response = client.get("/api/speakers")
    assert response.status_code == 200
    data = response.json()
    assert "speakers" in data
    assert "languages" in data
    assert "af_heart" in data["speakers"]


def test_static_index() -> None:
    """Tests that the frontend HTML dashboard is served at root."""
    response = client.get("/")
    assert response.status_code == 200
    assert "NexusVoice" in response.text


def test_chat_tokens_endpoint() -> None:
    """Tests POST /api/chat/tokens streams raw token deltas via SSE."""
    import json
    from unittest.mock import patch
    from app.api import routes

    async def fake_tokens(*args, **kwargs):
        yield "Hello "
        yield "world!"

    with patch.object(routes.pipeline.llm, "stream_response", new=fake_tokens):
        with client.stream(
            "POST", "/api/chat/tokens", json={"message": "Hi"}
        ) as response:
            assert response.status_code == 200
            tokens = []
            for line in response.iter_lines():
                if not line:
                    continue
                text = line.strip()
                if not text.startswith("data:"):
                    continue
                data = text[5:].strip()
                if data == "[DONE]":
                    continue
                event = json.loads(data)
                if "token" in event:
                    tokens.append(event["token"])

    assert tokens == ["Hello ", "world!"]


def test_websocket_token_streaming() -> None:
    """Tests /ws/assistant emits word-level token frames before chunks."""
    from unittest.mock import patch
    import numpy as np
    from app.api.websocket import ws_pipeline

    async def fake_tokens(*args, **kwargs):
        yield "Hello there. "
        yield "How are you?"

    with patch.object(
        ws_pipeline.llm, "stream_response", new=fake_tokens
    ), patch.object(
        ws_pipeline.tts,
        "synthesize",
        return_value=(np.zeros(24000, dtype=np.float32), 24000),
    ):
        with client.websocket_connect("/ws/assistant") as websocket:
            websocket.send_json({"type": "text", "prompt": "Hi", "play_audio": False})
            frames = []
            while True:
                frame = websocket.receive_json()
                frames.append(frame)
                if frame.get("type") in ("result", "error"):
                    break

            types = [f["type"] for f in frames]
            assert "token" in types
            token_text = "".join(f["text"] for f in frames if f["type"] == "token")
            assert "Hello there." in token_text
            assert "How are you?" in token_text
            assert "chunk" in types
            assert "result" in types


def test_websocket_chat_streaming() -> None:
    """Tests /ws/assistant streaming WebSocket endpoint."""
    from unittest.mock import patch
    import numpy as np
    from app.api.websocket import ws_pipeline

    async def fake_tokens(*args, **kwargs):
        yield "Hello there. "
        yield "Streaming is active."

    with patch.object(
        ws_pipeline.llm, "stream_response", new=fake_tokens
    ), patch.object(
        ws_pipeline.tts,
        "synthesize",
        return_value=(np.zeros(24000, dtype=np.float32), 24000),
    ):
        with client.websocket_connect("/ws/assistant") as websocket:
            websocket.send_json({"type": "text", "prompt": "Hi", "play_audio": False})
            frames = []
            while True:
                frame = websocket.receive_json()
                frames.append(frame)
                if frame.get("type") in ("result", "error"):
                    break

            types = [f["type"] for f in frames]
            assert "chunk" in types
            assert "result" in types
            chunks = [f for f in frames if f["type"] == "chunk"]
            assert len(chunks) == 2
            assert chunks[0]["text"] == "Hello there."
            assert chunks[1]["text"] == "Streaming is active."
            assert len(chunks[0]["audio_base64"]) > 0

