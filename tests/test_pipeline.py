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

