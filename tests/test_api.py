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
    assert "tts_device" in data
    assert "tts_effective_device" in data
    assert "tts_available_devices" in data
    assert "cpu" in data["tts_available_devices"]
    assert "npu" in data["tts_available_devices"]


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


def test_interact_response_validation_with_string_metrics() -> None:
    """Verifies InteractResponse schema accepts string values in metrics dictionary."""
    from app.api.routes import InteractResponse

    resp = InteractResponse(
        user_text="Hi",
        assistant_text="Hello",
        audio_base64="AAAA",
        sample_rate=24000,
        metrics={
            "stt_ms": 12.5,
            "total_ms": 100.0,
            "npu_status": "available",
            "audio_queue_depth": 0,
        },
    )
    assert resp.metrics["npu_status"] == "available"
    assert resp.metrics["audio_queue_depth"] == 0


def test_interact_endpoint_with_npu_status() -> None:
    """Tests /api/interact endpoint returns 200 when metrics include string npu_status."""
    from unittest.mock import AsyncMock, patch
    from app.api import routes
    from app.core.pipeline import AssistantResponse, PipelineMetrics

    mock_metrics = PipelineMetrics(
        stt_latency_ms=10.0,
        llm_latency_ms=20.0,
        tts_latency_ms=30.0,
        tts_synth_ms=15.0,
        ttfa_ms=25.0,
        total_latency_ms=60.0,
        npu_status="available",
    )
    mock_resp = AssistantResponse(
        user_text="Hello assistant",
        assistant_text="Hi there!",
        audio_bytes=b"dummy_wav_bytes",
        sample_rate=24000,
        metrics=mock_metrics,
    )

    with patch.object(
        routes.pipeline, "process_audio_bytes", new=AsyncMock(return_value=mock_resp)
    ):
        response = client.post(
            "/api/interact",
            files={"file": ("test.wav", b"dummy_audio_bytes", "audio/wav")},
            data={"speaker": "ryan", "play_audio": "false"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_text"] == "Hello assistant"
        assert data["assistant_text"] == "Hi there!"
        assert data["metrics"]["npu_status"] == "available"
        assert data["metrics"]["total_ms"] == 60.0


def test_record_endpoint_with_npu_status() -> None:
    """Tests /api/record endpoint handles npu_status in metrics."""
    from unittest.mock import AsyncMock, patch
    import numpy as np
    from app.api import routes
    from app.core.audio import AudioProcessor
    from app.core.pipeline import AssistantResponse, PipelineMetrics

    mock_metrics = PipelineMetrics(
        stt_latency_ms=12.0,
        total_latency_ms=50.0,
        npu_status="available",
    )
    mock_resp = AssistantResponse(
        user_text="Microphone speech",
        assistant_text="Microphone reply",
        audio_bytes=b"dummy_wav_bytes",
        sample_rate=24000,
        metrics=mock_metrics,
    )

    with patch.object(
        AudioProcessor, "record_microphone", return_value=np.zeros(16000, dtype=np.float32)
    ), patch.object(
        routes.pipeline, "process_audio_bytes", new=AsyncMock(return_value=mock_resp)
    ):
        response = client.post(
            "/api/record",
            json={"duration_seconds": 1.0, "play_audio": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["metrics"]["npu_status"] == "available"


def test_get_models_endpoint() -> None:
    """Tests GET /api/llm/models and GET /api/models."""
    from unittest.mock import patch, AsyncMock
    from app.api import routes

    mock_models = [
        {
            "id": "qwen2.5-0.5b-instruct",
            "name": "Qwen2.5 0.5B Instruct",
            "loaded": True,
            "params": "630M",
            "architecture": "qwen2",
            "size_bytes": 675710816,
            "size_formatted": "644.4 MB",
            "type": "llm",
        },
        {
            "id": "qwen2.5-3b-instruct",
            "name": "Qwen2.5 3B Instruct",
            "loaded": False,
            "params": "3.4B",
            "architecture": "qwen2",
            "size_bytes": 2104932768,
            "size_formatted": "1.96 GB",
            "type": "llm",
        },
    ]

    with patch.object(
        routes.pipeline.llm, "list_available_models", new=AsyncMock(return_value=mock_models)
    ):
        res1 = client.get("/api/llm/models")
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["status"] == "success"
        assert len(data1["models"]) == 2
        assert "loaded_models" in data1
        assert "qwen2.5-0.5b-instruct" in data1["loaded_models"]

        res2 = client.get("/api/models")
        assert res2.status_code == 200
        assert res2.json()["current_model"] == routes.pipeline.llm.model


def test_select_model_endpoint() -> None:
    """Tests POST /api/llm/model switching active model."""
    from unittest.mock import patch, AsyncMock
    from app.api import routes

    mock_result = {
        "status": "success",
        "model": "qwen2.5-3b-instruct",
        "loaded": True,
        "message": "Switched to qwen2.5-3b-instruct",
    }

    with patch.object(
        routes.pipeline, "set_model", new=AsyncMock(return_value=mock_result)
    ):
        res = client.post("/api/llm/model", json={"model": "qwen2.5-3b-instruct", "load": True})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["model"] == "qwen2.5-3b-instruct"
        assert data["loaded"] is True


def test_get_tts_device_endpoint() -> None:
    """Tests GET /api/tts/device returns device information."""
    response = client.get("/api/tts/device")
    assert response.status_code == 200
    data = response.json()
    assert "device" in data
    assert "effective_device" in data
    assert "backend" in data
    assert "available_devices" in data
    assert "cpu" in data["available_devices"]
    assert "npu" in data["available_devices"]


def test_set_tts_device_endpoint() -> None:
    """Tests POST /api/tts/device updates active processing unit."""
    from unittest.mock import patch
    from app.api import routes

    mock_resp = {
        "status": "success",
        "device": "npu",
        "effective_device": "npu",
        "backend": "openvino",
        "message": "TTS device set to npu (effective: npu)",
    }

    with patch.object(routes.pipeline, "set_tts_device", return_value=mock_resp):
        response = client.post("/api/tts/device", json={"device": "npu"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["device"] == "npu"
        assert data["effective_device"] == "npu"

    mock_resp_full = {
        "status": "success",
        "device": "npu_only",
        "effective_device": "NPU (Full)",
        "backend": "openvino",
        "message": "TTS device set to npu_only",
    }
    with patch.object(routes.pipeline, "set_tts_device", return_value=mock_resp_full):
        response = client.post("/api/tts/device", json={"device": "npu_only"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["device"] == "npu_only"
        assert data["effective_device"] == "NPU (Full)"



def test_set_tts_device_invalid_device() -> None:
    """Tests POST /api/tts/device returns 400 for unsupported hardware device."""
    response = client.post("/api/tts/device", json={"device": "invalid_accelerator"})
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported TTS device" in data["detail"]



def test_websocket_tts_device_handling() -> None:
    """Tests /ws/assistant get_tts_device and set_tts_device operations."""
    from unittest.mock import patch
    from app.api.websocket import ws_pipeline

    mock_info = {
        "device": "cpu",
        "effective_device": "cpu",
        "backend": "openvino",
        "available_devices": ["cpu", "npu"],
    }
    mock_set_res = {
        "status": "success",
        "device": "npu",
        "effective_device": "npu",
        "backend": "openvino",
        "message": "TTS device set to npu",
    }

    with patch.object(
        ws_pipeline, "get_tts_device", return_value=mock_info
    ), patch.object(
        ws_pipeline, "set_tts_device", return_value=mock_set_res
    ):
        with client.websocket_connect("/ws/assistant") as websocket:
            # Query device
            websocket.send_json({"type": "get_tts_device"})
            resp1 = websocket.receive_json()
            assert resp1["type"] == "tts_device"
            assert resp1["device"] == "cpu"
            assert "available_devices" in resp1

            # Switch device
            websocket.send_json({"type": "set_tts_device", "device": "npu"})
            resp2 = websocket.receive_json()
            assert resp2["type"] == "tts_device_changed"
            assert resp2["device"] == "npu"


def test_tts_stream_endpoint() -> None:
    """Tests POST /api/tts returns streaming WAV bytes."""
    from unittest.mock import patch
    from app.api import routes

    fake_wav = b"RIFF....WAVEfmt ...."
    with patch.object(routes.pipeline.tts, "synthesize_to_wav_bytes", return_value=fake_wav):
        response = client.post("/api/tts", json={"text": "Hello world from TTS"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content == fake_wav


def test_tts_generate_endpoint() -> None:
    """Tests POST /api/tts/generate returns structured JSON with base64 audio and metrics."""
    from unittest.mock import patch
    import numpy as np
    from app.api import routes

    fake_waveform = np.zeros(24000, dtype=np.float32)
    with patch.object(routes.pipeline.tts, "synthesize", return_value=(fake_waveform, 24000)):
        response = client.post(
            "/api/tts/generate",
            json={"text": "Hello world from direct TTS", "speaker": "af_bella"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_text"] == "Hello world from direct TTS"
        assert data["assistant_text"] == "Hello world from direct TTS"
        assert len(data["audio_base64"]) > 0
        assert data["sample_rate"] == 24000
        assert "metrics" in data
        assert "tts_ms" in data["metrics"]


def test_websocket_direct_tts() -> None:
    """Tests /ws/assistant direct speech synthesis via type: 'tts'."""
    from unittest.mock import patch
    import numpy as np
    from app.api.websocket import ws_pipeline

    fake_waveform = np.zeros(24000, dtype=np.float32)
    with patch.object(ws_pipeline.tts, "synthesize", return_value=(fake_waveform, 24000)):
        with client.websocket_connect("/ws/assistant") as websocket:
            websocket.send_json({
                "type": "tts",
                "text": "Direct WebSocket TTS speech",
                "speaker": "af_heart",
            })

            # First message should be status
            msg1 = websocket.receive_json()
            assert msg1["type"] == "status"
            assert msg1["stage"] == "tts"

            # Subsequent messages: chunk and/or result
            received_result = False
            for _ in range(5):
                msg = websocket.receive_json()
                if msg["type"] == "result":
                    received_result = True
                    assert msg["user_text"] == "Direct WebSocket TTS speech"
                    assert msg["assistant_text"] == "Direct WebSocket TTS speech"
                    assert len(msg["audio_base64"]) > 0
                    assert "metrics" in msg
                    break

            assert received_result, "Did not receive final result frame from WebSocket TTS"


def test_think_mode_api_endpoints() -> None:
    """Verifies GET and POST /api/llm/think-mode endpoints."""
    # GET think mode
    resp = client.get("/api/llm/think-mode")
    assert resp.status_code == 200
    data = resp.json()
    assert "think_mode" in data
    assert "supports_thinking" in data
    assert "reasoning_effort" in data

    # POST think mode enable
    resp = client.post("/api/llm/think-mode", json={"think_mode": True, "reasoning_effort": "high"})
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["think_mode"] is True
    assert res_data["reasoning_effort"] == "high"

    # POST think mode disable
    resp = client.post("/api/llm/think-mode", json={"think_mode": False})
    assert resp.status_code == 200
    assert resp.json()["think_mode"] is False


def test_websocket_think_mode_control() -> None:
    """Verifies WebSocket get_think_mode and set_think_mode frames."""
    with client.websocket_connect("/ws/assistant") as websocket:
        # Request current think mode
        websocket.send_json({"type": "get_think_mode"})
        msg = websocket.receive_json()
        assert msg["type"] == "think_mode"
        assert "think_mode" in msg
        assert "supports_thinking" in msg

        # Update think mode
        websocket.send_json({"type": "set_think_mode", "think_mode": True, "reasoning_effort": "medium"})
        msg = websocket.receive_json()
        assert msg["type"] == "think_mode_changed"
        assert msg["think_mode"] is True
        assert msg["reasoning_effort"] == "medium"





