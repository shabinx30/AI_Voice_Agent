"""Tests for AssistantPipeline orchestrator."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest
from app.core.pipeline import AssistantPipeline, PipelineMetrics


def test_pipeline_process_text_prompt() -> None:
    """Tests end-to-end text prompt processing through the pipeline."""
    async def _test() -> None:
        mock_stt = MagicMock()
        mock_llm = MagicMock()
        mock_tts = MagicMock()

        mock_llm.generate_response = AsyncMock(
            return_value="This is a test response from the assistant."
        )

        sample_rate = 24000
        test_waveform = np.zeros(sample_rate, dtype=np.float32)
        mock_tts.synthesize = MagicMock(return_value=(test_waveform, sample_rate))

        pipeline = AssistantPipeline(
            stt_engine=mock_stt,
            llm_client=mock_llm,
            tts_engine=mock_tts,
        )

        res = await pipeline.process_text_prompt(
            prompt="Hello!",
            play_audio=False,
        )

        assert res.user_text == "Hello!"
        assert res.assistant_text == "This is a test response from the assistant."
        assert len(res.audio_bytes) > 0
        assert res.sample_rate == sample_rate
        assert isinstance(res.metrics, PipelineMetrics)

    asyncio.run(_test())


def test_pipeline_process_audio_bytes() -> None:
    """Tests end-to-end audio byte processing through the pipeline."""
    async def _test() -> None:
        mock_stt = MagicMock()
        mock_llm = MagicMock()
        mock_tts = MagicMock()

        mock_stt.transcribe = MagicMock(return_value="What time is it?")
        mock_llm.generate_response = AsyncMock(
            return_value="It is currently morning."
        )

        sample_rate = 24000
        test_waveform = np.zeros(sample_rate, dtype=np.float32)
        mock_tts.synthesize = MagicMock(return_value=(test_waveform, sample_rate))

        pipeline = AssistantPipeline(
            stt_engine=mock_stt,
            llm_client=mock_llm,
            tts_engine=mock_tts,
        )

        from app.core.audio import AudioProcessor

        fake_audio = np.zeros(16000, dtype=np.float32)
        fake_wav_bytes = AudioProcessor.to_wav_bytes(fake_audio, 16000)

        res = await pipeline.process_audio_bytes(
            audio_bytes=fake_wav_bytes,
            play_audio=False,
        )

        assert res.user_text == "What time is it?"
        assert res.assistant_text == "It is currently morning."
        assert len(res.audio_bytes) > 0

    asyncio.run(_test())


def test_pipeline_streaming_sentence_to_tts() -> None:
    """Verifies that pipeline streams sentences and hands them over to TTS concurrently."""
    async def _test() -> None:
        mock_stt = MagicMock()
        mock_llm = MagicMock()
        mock_tts = MagicMock()

        async def fake_sentence_stream(*args, **kwargs):
            yield "Hello! How are you?"
            yield "I am doing well, thank you."

        mock_llm.stream_sentence_chunks = fake_sentence_stream

        sample_rate = 24000
        test_waveform = np.zeros(sample_rate, dtype=np.float32)
        synthesized_sentences = []

        def fake_synthesize(text, *args, **kwargs):
            synthesized_sentences.append(text)
            return (test_waveform, sample_rate)

        mock_tts.synthesize = MagicMock(side_effect=fake_synthesize)
        mock_tts.sample_rate = sample_rate

        pipeline = AssistantPipeline(
            stt_engine=mock_stt,
            llm_client=mock_llm,
            tts_engine=mock_tts,
        )

        received_chunks = []

        async def on_chunk(chunk):
            received_chunks.append(chunk)

        res = await pipeline.process_text_prompt(
            prompt="Hi",
            play_audio=False,
            on_chunk=on_chunk,
        )

        assert res.user_text == "Hi"
        assert "Hello! How are you?" in res.assistant_text
        assert "I am doing well, thank you." in res.assistant_text
        assert len(synthesized_sentences) == 2
        assert synthesized_sentences == [
            "Hello! How are you?",
            "I am doing well, thank you.",
        ]
        assert len(received_chunks) == 2
        assert received_chunks[0].text == "Hello! How are you?"
        assert received_chunks[1].text == "I am doing well, thank you."
        assert len(res.audio_bytes) > 0

    asyncio.run(_test())


def test_pipeline_token_tap_forwards_tokens() -> None:
    """Verifies raw LLM tokens reach on_token while TTS still gets sentences."""
    async def _test() -> None:
        mock_stt = MagicMock()
        mock_llm = MagicMock()
        mock_tts = MagicMock()

        async def fake_token_stream(*args, **kwargs):
            yield "Hello there! "
            yield "How are you today?"

        mock_llm.stream_response = fake_token_stream

        sample_rate = 24000
        test_waveform = np.zeros(sample_rate, dtype=np.float32)
        synthesized = []
        mock_tts.sample_rate = sample_rate

        def fake_synthesize(text, *args, **kwargs):
            synthesized.append(text)
            return (test_waveform, sample_rate)

        mock_tts.synthesize = MagicMock(side_effect=fake_synthesize)

        pipeline = AssistantPipeline(
            stt_engine=mock_stt,
            llm_client=mock_llm,
            tts_engine=mock_tts,
        )

        seen_tokens = []

        async def on_token(token: str) -> None:
            seen_tokens.append(token)

        res = await pipeline.process_text_prompt(
            prompt="Hi",
            play_audio=False,
            on_token=on_token,
        )

        assert seen_tokens == ["Hello there! ", "How are you today?"]
        assert "Hello there!" in res.assistant_text
        assert "How are you today?" in res.assistant_text
        assert len(synthesized) == 2
        assert len(res.audio_bytes) > 0

    asyncio.run(_test())


def test_pipeline_token_tap_falls_back_without_token_stream() -> None:
    """Verifies sentence-only streaming when the LLM has no token stream."""
    async def _test() -> None:
        class _StubLLM:
            async def generate_response(self, prompt, history=None):
                return "Stub reply."

            async def stream_sentence_chunks(self, *args, **kwargs):
                yield "Stub reply sentence here."

        assert not hasattr(_StubLLM(), "stream_response")

        mock_stt = MagicMock()
        mock_tts = MagicMock()
        sample_rate = 24000
        test_waveform = np.zeros(sample_rate, dtype=np.float32)
        mock_tts.synthesize = MagicMock(return_value=(test_waveform, sample_rate))
        mock_tts.sample_rate = sample_rate

        pipeline = AssistantPipeline(
            stt_engine=mock_stt,
            llm_client=_StubLLM(),
            tts_engine=mock_tts,
        )

        seen_tokens = []
        res = await pipeline.process_text_prompt(
            prompt="Hi",
            play_audio=False,
            on_token=seen_tokens.append,
        )

        assert seen_tokens == []
        assert "Stub reply sentence here." in res.assistant_text

    asyncio.run(_test())


def test_pipeline_stream_text_prompt_generator() -> None:
    """Verifies stream_text_prompt async generator yields sentence chunks."""
    async def _test() -> None:
        mock_stt = MagicMock()
        mock_llm = MagicMock()
        mock_tts = MagicMock()

        async def fake_sentence_stream(*args, **kwargs):
            yield "First sentence."
            yield "Second sentence."

        mock_llm.stream_sentence_chunks = fake_sentence_stream

        sample_rate = 24000
        test_waveform = np.zeros(sample_rate, dtype=np.float32)
        mock_tts.synthesize = MagicMock(return_value=(test_waveform, sample_rate))
        mock_tts.sample_rate = sample_rate

        pipeline = AssistantPipeline(
            stt_engine=mock_stt,
            llm_client=mock_llm,
            tts_engine=mock_tts,
        )

        chunks = []
        async for chunk in pipeline.stream_text_prompt("Hello", play_audio=False):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].text == "First sentence."
        assert chunks[1].text == "Second sentence."

    asyncio.run(_test())


def test_pipeline_tts_device_management() -> None:
    """Verifies that AssistantPipeline delegates TTS device get and set calls."""
    mock_stt = MagicMock()
    mock_llm = MagicMock()
    mock_tts = MagicMock()

    mock_tts.get_device_info = MagicMock(
        return_value={
            "device": "cpu",
            "effective_device": "cpu",
            "backend": "openvino",
            "available_devices": ["cpu", "npu"],
        }
    )
    mock_tts.set_device = MagicMock(
        return_value={
            "status": "success",
            "device": "npu",
            "effective_device": "npu",
            "backend": "openvino",
            "message": "TTS device set to npu",
        }
    )

    pipeline = AssistantPipeline(
        stt_engine=mock_stt,
        llm_client=mock_llm,
        tts_engine=mock_tts,
    )

    info = pipeline.get_tts_device()
    assert info["device"] == "cpu"
    mock_tts.get_device_info.assert_called_once()

    set_res = pipeline.set_tts_device("npu")
    assert set_res["status"] == "success"
    assert set_res["device"] == "npu"
    mock_tts.set_device.assert_called_once_with("npu")


def test_pipeline_preserves_paragraphs_in_assistant_text() -> None:
    """Verifies that multi-paragraph LLM responses retain their newlines/paragraphs upon completion."""
    async def _test() -> None:
        mock_stt = MagicMock()
        mock_llm = MagicMock()
        mock_tts = MagicMock()

        async def fake_token_stream(*args, **kwargs):
            yield "Paragraph one line.\n\n"
            yield "Paragraph two line.\n\n"
            yield "Paragraph three line."

        mock_llm.stream_response = fake_token_stream

        sample_rate = 24000
        test_waveform = np.zeros(sample_rate, dtype=np.float32)
        mock_tts.sample_rate = sample_rate
        mock_tts.synthesize = MagicMock(return_value=(test_waveform, sample_rate))

        pipeline = AssistantPipeline(
            stt_engine=mock_stt,
            llm_client=mock_llm,
            tts_engine=mock_tts,
        )

        res = await pipeline.process_text_prompt(
            prompt="Tell me about Docker",
            play_audio=False,
        )

        expected = "Paragraph one line.\n\nParagraph two line.\n\nParagraph three line."
        assert res.assistant_text == expected
        assert "\n\n" in res.assistant_text

    asyncio.run(_test())


def test_pipeline_process_direct_tts() -> None:
    """Verifies that process_direct_tts synthesizes text directly bypassing STT and LLM."""
    async def _test() -> None:
        mock_stt = MagicMock()
        mock_llm = MagicMock()
        mock_tts = MagicMock()

        sample_rate = 24000
        test_waveform = np.zeros(sample_rate, dtype=np.float32)
        mock_tts.sample_rate = sample_rate
        mock_tts.speaker = "af_heart"
        mock_tts.language = "a"
        mock_tts.synthesize = MagicMock(return_value=(test_waveform, sample_rate))

        pipeline = AssistantPipeline(
            stt_engine=mock_stt,
            llm_client=mock_llm,
            tts_engine=mock_tts,
        )

        emitted_chunks = []
        def _on_chunk(chunk):
            emitted_chunks.append(chunk)

        res = await pipeline.process_direct_tts(
            text="Hello from direct TTS!",
            speaker="af_bella",
            play_audio=False,
            on_chunk=_on_chunk,
        )

        # STT and LLM must NOT have been called
        mock_stt.transcribe.assert_not_called()
        mock_llm.generate_response.assert_not_called()
        if hasattr(mock_llm, "stream_sentence_chunks") and isinstance(mock_llm.stream_sentence_chunks, MagicMock):
            mock_llm.stream_sentence_chunks.assert_not_called()

        # TTS must have been called with target speaker
        mock_tts.synthesize.assert_called_once()
        args, kwargs = mock_tts.synthesize.call_args
        assert "Hello from direct TTS" in (args[0] if args else kwargs.get("text"))
        assert kwargs.get("speaker") == "af_bella"

        assert res.user_text == "Hello from direct TTS!"
        assert res.assistant_text == "Hello from direct TTS!"
        assert len(res.audio_bytes) > 0
        assert res.sample_rate == sample_rate
        assert res.metrics.tts_latency_ms >= 0
        assert res.metrics.stt_latency_ms == 0.0
        assert res.metrics.llm_latency_ms == 0.0
        assert len(emitted_chunks) == 1
        assert emitted_chunks[0].text == "Hello from direct TTS!"
        assert emitted_chunks[0].is_final is True

    asyncio.run(_test())


def test_pipeline_process_direct_tts_empty_raises() -> None:
    """Verifies that process_direct_tts with empty text raises ValueError."""
    async def _test() -> None:
        pipeline = AssistantPipeline(
            stt_engine=MagicMock(),
            llm_client=MagicMock(),
            tts_engine=MagicMock(),
        )
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            await pipeline.process_direct_tts(text="   ")

def test_pipeline_process_direct_tts_multi_sentence() -> None:
    """Verifies that process_direct_tts with multiple sentences streams sentence-by-sentence chunks."""
    async def _test() -> None:
        mock_stt = MagicMock()
        mock_llm = MagicMock()
        mock_tts = MagicMock()

        sample_rate = 24000
        test_waveform = np.zeros(sample_rate, dtype=np.float32)
        mock_tts.sample_rate = sample_rate
        mock_tts.speaker = "af_heart"
        mock_tts.language = "a"
        mock_tts.synthesize = MagicMock(return_value=(test_waveform, sample_rate))

        pipeline = AssistantPipeline(
            stt_engine=mock_stt,
            llm_client=mock_llm,
            tts_engine=mock_tts,
        )

        emitted_chunks = []
        def _on_chunk(chunk):
            emitted_chunks.append(chunk)

        res = await pipeline.process_direct_tts(
            text="First sentence here! Second sentence follows. Third sentence ends.",
            speaker="af_heart",
            play_audio=False,
            on_chunk=_on_chunk,
        )

        assert len(emitted_chunks) == 3
        assert emitted_chunks[0].text == "First sentence here!"
        assert emitted_chunks[0].sentence_index == 0
        assert emitted_chunks[0].is_final is False

        assert emitted_chunks[1].text == "Second sentence follows."
        assert emitted_chunks[1].sentence_index == 1
        assert emitted_chunks[1].is_final is False

        assert emitted_chunks[2].text == "Third sentence ends."
        assert emitted_chunks[2].sentence_index == 2
        assert emitted_chunks[2].is_final is True

        assert mock_tts.synthesize.call_count == 3
        assert len(res.audio_bytes) > 0

    asyncio.run(_test())
