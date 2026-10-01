"""Tests for LM Studio LLM client."""

import asyncio
import json
import subprocess
from unittest.mock import AsyncMock, patch
import httpx
import pytest
from app.core.llm import LMStudioClient



def test_llm_client_initialization() -> None:
    """Verifies LM Studio client attributes."""
    async def _test() -> None:
        client = LMStudioClient(
            base_url="http://127.0.0.1:1234/v1",
            model="qwen2.5-0.5b-instruct",
        )
        assert client.base_url == "http://127.0.0.1:1234/v1"
        assert client.model == "qwen2.5-0.5b-instruct"
        await client.close()

    asyncio.run(_test())


def test_llm_generate_response_mocked() -> None:
    """Tests response parsing from LM Studio API."""
    async def _test() -> None:
        client = LMStudioClient()

        mock_response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello, I am your voice assistant.",
                        }
                    }
                ]
            },
            request=httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions"),
        )

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response
            reply = await client.generate_response("Hello")
            assert reply == "Hello, I am your voice assistant."

        await client.close()

    asyncio.run(_test())


def test_split_into_sentence_chunks_fullstops() -> None:
    """Verifies splitting token stream into sentence chunks on fullstops and punctuation."""
    async def _test() -> None:
        from app.core.llm import split_into_sentence_chunks

        async def token_gen():
            tokens = ["Hello", " world", ".", " How", " are", " you", "?"]
            for t in tokens:
                yield t

        chunks = [c async for c in split_into_sentence_chunks(token_gen())]
        assert chunks == ["Hello world.", "How are you?"]

    asyncio.run(_test())


def test_split_into_sentence_chunks_newlines() -> None:
    """Verifies splitting token stream into sentence chunks on newlines."""
    async def _test() -> None:
        from app.core.llm import split_into_sentence_chunks

        async def token_gen():
            tokens = ["First sentence.\n", "Second sentence.\n", "Third sentence."]
            for t in tokens:
                yield t

        chunks = [c async for c in split_into_sentence_chunks(token_gen())]
        assert chunks == ["First sentence.", "Second sentence.", "Third sentence."]

    asyncio.run(_test())


def test_split_into_sentence_chunks_protects_abbreviations_and_decimals() -> None:
    """Verifies abbreviations and decimal numbers are not incorrectly split."""
    async def _test() -> None:
        from app.core.llm import split_into_sentence_chunks

        async def token_gen():
            tokens = [
                "Dr", ". Smith", " purchased", " items", " for", " $3", ".50", ".",
                " He", " was", " very", " pleased", "."
            ]
            for t in tokens:
                yield t

        chunks = [c async for c in split_into_sentence_chunks(token_gen())]
        assert len(chunks) == 2
        assert chunks[0] == "Dr. Smith purchased items for $3.50."
        assert chunks[1] == "He was very pleased."

    asyncio.run(_test())


def test_llm_stream_sentence_chunks_mocked() -> None:
    """Tests stream_sentence_chunks with a mocked stream_response generator."""
    async def _test() -> None:
        client = LMStudioClient()

        async def fake_tokens(*args, **kwargs):
            tokens = ["The weather", " is", " clear", " today", ".\n", "Enjoy", " your day", "!"]
            for t in tokens:
                yield t

        client.stream_response = fake_tokens

        sentences = [s async for s in client.stream_sentence_chunks("Tell me the weather")]
        assert sentences == ["The weather is clear today.", "Enjoy your day!"]
        await client.close()

    asyncio.run(_test())


def test_llm_list_available_models_mocked() -> None:
    """Tests list_available_models with mocked HTTP response."""
    async def _test() -> None:
        client = LMStudioClient()

        mock_http_response = httpx.Response(
            200,
            json={
                "data": [
                    {"id": "qwen2.5-0.5b-instruct"},
                    {"id": "text-embedding-nomic-embed-text-v1.5"},
                    {"id": "qwen2.5-3b-instruct"},
                ]
            },
            request=httpx.Request("GET", "http://127.0.0.1:1234/v1/models"),
        )

        with patch("subprocess.run") as mock_run, patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock
        ) as mock_get:
            # Simulate 'lms' CLI unavailable so HTTP path is exercised
            mock_run.side_effect = FileNotFoundError("lms not found")
            mock_get.return_value = mock_http_response

            models = await client.list_available_models()
            model_ids = [m["id"] for m in models]
            assert "qwen2.5-0.5b-instruct" in model_ids
            assert "qwen2.5-3b-instruct" in model_ids
            # Embedding model should be filtered out
            assert "text-embedding-nomic-embed-text-v1.5" not in model_ids

        await client.close()

    asyncio.run(_test())


def test_llm_set_model() -> None:
    """Tests set_model ejecting old model and switching client model."""
    async def _test() -> None:
        client = LMStudioClient(model="qwen2.5-0.5b-instruct")
        assert client.model == "qwen2.5-0.5b-instruct"

        with patch.object(
            client, "eject_other_models", return_value=["qwen2.5-0.5b-instruct"]
        ) as mock_eject, patch.object(
            client, "_try_lms_load", return_value=True
        ) as mock_load:
            res = await client.set_model("qwen2.5-3b-instruct", load_into_memory=True)
            assert res["status"] == "success"
            assert res["model"] == "qwen2.5-3b-instruct"
            assert client.model == "qwen2.5-3b-instruct"
            assert res["loaded"] is True
            assert res["ejected_models"] == ["qwen2.5-0.5b-instruct"]

            # Assert eject was called with target model
            mock_eject.assert_called_once_with("qwen2.5-3b-instruct")
            # Assert load was called after
            mock_load.assert_called_once_with("qwen2.5-3b-instruct")

        await client.close()

    asyncio.run(_test())


def test_llm_eject_other_models_mocked() -> None:
    """Verifies eject_other_models unloads non-matching loaded models."""
    client = LMStudioClient()

    fake_ps = json.dumps([
        {"identifier": "qwen2.5-0.5b-instruct", "modelKey": "qwen2.5-0.5b-instruct"},
        {"identifier": "qwen2.5-3b-instruct", "modelKey": "qwen2.5-3b-instruct"},
    ])

    with patch("subprocess.run") as mock_run:
        # Mock lms ps
        mock_ps_res = subprocess.CompletedProcess(
            args=["lms", "ps", "--json"], returncode=0, stdout=fake_ps, stderr=""
        )
        # Mock lms unload
        mock_unload_res = subprocess.CompletedProcess(
            args=["lms", "unload"], returncode=0, stdout="Unloaded", stderr=""
        )

        def _side_effect(cmd, *args, **kwargs):
            if "ps" in cmd:
                return mock_ps_res
            return mock_unload_res

        mock_run.side_effect = _side_effect

        # When keeping qwen2.5-3b-instruct, qwen2.5-0.5b-instruct must be ejected
        ejected = client.eject_other_models(keep_model_id="qwen2.5-3b-instruct")
        assert ejected == ["qwen2.5-0.5b-instruct"]


def test_llm_eject_model_mocked() -> None:
    """Verifies eject_model endpoint functionality."""
    async def _test() -> None:
        client = LMStudioClient()
        with patch.object(client, "_try_lms_unload", return_value=True):
            res = await client.eject_model("qwen2.5-0.5b-instruct")
            assert res["status"] == "success"
            assert "qwen2.5-0.5b-instruct" in res["ejected_models"]
        await client.close()

    asyncio.run(_test())





