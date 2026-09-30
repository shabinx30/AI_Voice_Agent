"""LM Studio LLM client module.

This module interfaces with an LM Studio instance hosting local language models
via an OpenAI-compatible HTTP REST API endpoint.
"""

import asyncio
import logging
import subprocess
import time
from typing import AsyncGenerator, Dict, List, Optional, Set
import httpx

from app.config import settings
from app.core.sentences import (
    COMMON_ABBREVIATIONS as _SENT_COMMON_ABBREVS,
    clean_text_for_speech as _clean_text_for_speech,
    is_abbreviation as _is_abbreviation_impl,
    split_into_sentence_chunks as _split_impl,
)

logger = logging.getLogger(__name__)

# Re-exported for backward compatibility (tests import from app.core.llm).
COMMON_ABBREVIATIONS: Set[str] = _SENT_COMMON_ABBREVS


def is_abbreviation(text: str) -> bool:
    """Checks whether the trailing word in text matches a known abbreviation.

    Args:
        text: Text string to inspect.

    Returns:
        True if the trailing token represents an abbreviation, False otherwise.
    """
    return _is_abbreviation_impl(text)


async def split_into_sentence_chunks(
    token_stream: AsyncGenerator[str, None],
    min_chunk_length: int = 12,
) -> AsyncGenerator[str, None]:
    """Splits an asynchronous token stream into sentence-sized chunks.

    Buffers incoming tokens and yields complete sentences whenever a full stop ('.')
    or newline ('\\n') boundary is reached (along with '!' and '?'), while avoiding
    premature splits on decimal numbers (e.g. 3.14), abbreviations (e.g. Dr.,
    U.S.A., e.g.), or incomplete words.

    Args:
        token_stream: Async generator yielding string token deltas from LLM.
        min_chunk_length: Minimum characters required to emit a sentence chunk.

    Yields:
        Sentence strings ready for immediate text-to-speech synthesis.
    """
    async for chunk in _split_impl(token_stream, min_chunk_length=min_chunk_length):
        yield chunk


class LMStudioClient:
    """Client for querying local LLM models served by LM Studio.

    Attributes:
        base_url: Base URL of LM Studio (e.g. 'http://127.0.0.1:1234/v1').
        model: Target model identifier configured or loaded in LM Studio.
        temperature: Generation temperature setting.
        top_p: Nucleus sampling cutoff.
        max_tokens: Maximum response tokens.
        system_prompt: Base system persona prompt.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Initializes the LM Studio client.

        Args:
            base_url: LM Studio OpenAI-compatible endpoint URL.
            model: Model name/identifier. Defaults to settings.lm_studio_model.
            temperature: Sampling temperature. Defaults to settings.lm_studio_temperature.
            top_p: Nucleus cutoff. Defaults to settings.lm_studio_top_p.
            max_tokens: Max tokens. Defaults to settings.lm_studio_max_tokens.
            system_prompt: System instructions. Defaults to settings.lm_studio_system_prompt.
        """
        self.base_url = (base_url or settings.lm_studio_base_url).rstrip("/")
        self.model = model or settings.lm_studio_model
        self.temperature = (
            temperature
            if temperature is not None
            else settings.lm_studio_temperature
        )
        _top_p_default = getattr(settings, "lm_studio_top_p", 0.9)
        self.top_p = top_p if top_p is not None else _top_p_default
        self.max_tokens = max_tokens or settings.lm_studio_max_tokens
        self.system_prompt = system_prompt or settings.lm_studio_system_prompt
        # Reuse a single pooled client for all calls (keep-alive). Creating a
        # new AsyncClient per request wastes TCP/TLS handshakes and adds
        # 5-50ms latency per LLM call.
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        self._closed = False

    async def check_health(self) -> bool:
        """Verifies connectivity to the LM Studio server.

        Returns:
            True if the server is accessible and responding with 200, False otherwise.
        """
        try:
            response = await self._http_client.get("/models")
            return response.status_code == 200
        except Exception as exc:
            logger.debug("LM Studio health check failed: %s", exc)
            return False

    def ensure_server_running(self) -> bool:
        """Attempts to start the LM Studio server via CLI if currently offline.

        Returns:
            True if the server is now running or was started, False on failure.
        """
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/models")
                if res.status_code == 200:
                    return True
        except Exception:
            pass

        logger.info("LM Studio server appears offline. Attempting to start via 'lms' CLI...")
        try:
            result = subprocess.run(
                ["lms", "server", "start"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
            logger.info("LM Studio start output: %s", result.stdout.strip())
            time.sleep(2)
            return True
        except Exception as exc:
            logger.warning("Could not automatically start LM Studio server: %s", exc)
            return False

    async def list_models(self) -> List[str]:
        """Fetches the list of models currently registered in LM Studio.

        Returns:
            List of model IDs available on the server.
        """
        try:
            response = await self._http_client.get("/models")
            if response.status_code == 200:
                data = response.json()
                return [m["id"] for m in data.get("data", [])]
            return []
        except Exception as exc:
            logger.error("Failed to retrieve LM Studio models: %s", exc)
            return []

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generates a text completion from the LM Studio model.

        Args:
            prompt: User message content.
            system_prompt: Optional override for the system prompt.
            history: Optional list of previous chat messages.

        Returns:
            Assistant response text.

        Raises:
            RuntimeError: If the API request fails or returns an error.
        """
        messages: List[Dict[str, str]] = []
        effective_system = system_prompt or self.system_prompt
        if effective_system:
            messages.append({"role": "system", "content": effective_system})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        logger.debug("Dispatching LLM prompt to %s/chat/completions", self.base_url)
        start_time = time.perf_counter()

        try:
            response = await self._http_client.post(
                "/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            reply = data["choices"][0]["message"]["content"].strip()
            elapsed = time.perf_counter() - start_time
            logger.info("LLM generated response in %.3fs: '%s'", elapsed, reply)
            return reply

        except httpx.ConnectError as exc:
            logger.error(
                "Unable to connect to LM Studio at %s. Ensure LM Studio server is running: %s",
                self.base_url,
                exc,
            )
            raise RuntimeError(
                f"LM Studio server unreachable at {self.base_url}. "
                "Please verify that LM Studio is open and the local server is started (lms server start)."
            ) from exc
        except Exception as exc:
            logger.error("LM Studio completion failed: %s", exc)
            raise RuntimeError(f"LM Studio API request error: {exc}") from exc

    async def stream_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Streams text chunks from the LM Studio LLM asynchronously.

        Args:
            prompt: User input prompt.
            system_prompt: Optional system instruction override.
            history: Optional conversation context.

        Yields:
            Generated text tokens/chunks as they arrive.
        """
        messages: List[Dict[str, str]] = []
        effective_system = system_prompt or self.system_prompt
        if effective_system:
            messages.append({"role": "system", "content": effective_system})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        try:
            async with self._http_client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
        except httpx.ConnectError as exc:
            logger.error(
                "Unable to connect to LM Studio at %s: %s", self.base_url, exc
            )
            raise RuntimeError(
                f"LM Studio server unreachable at {self.base_url}. "
                "Please verify that LM Studio is running (lms server start)."
            ) from exc
        except Exception as exc:
            logger.error("LLM stream error: %s", exc)
            raise RuntimeError(f"Error during LLM streaming: {exc}") from exc

    async def stream_sentence_chunks(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        min_chunk_length: int = 12,
    ) -> AsyncGenerator[str, None]:
        """Streams complete sentence chunks from LM Studio for TTS handover.

        Splits incoming LLM tokens into natural sentence chunks by detecting fullstops,
        newlines, question marks, or exclamation marks, while protecting abbreviations
        and decimal numbers from premature splitting.

        Args:
            prompt: User message prompt.
            system_prompt: Optional system prompt override.
            history: Optional conversation context.
            min_chunk_length: Minimum character length for a chunk.

        Yields:
            Sentence strings ready for text-to-speech synthesis.
        """
        token_stream = self.stream_response(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
        )
        async for sentence in split_into_sentence_chunks(
            token_stream, min_chunk_length=min_chunk_length
        ):
            yield sentence

    async def close(self) -> None:
        """Closes the underlying HTTP client session."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._http_client.aclose()
        except Exception:
            pass
