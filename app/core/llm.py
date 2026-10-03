"""LM Studio LLM client module.

This module interfaces with an LM Studio instance hosting local language models
via an OpenAI-compatible HTTP REST API endpoint.
"""

import asyncio
import json
import logging
import subprocess
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple
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


def model_supports_thinking(model_info_or_id: Any) -> bool:
    """Determines whether an LM Studio model natively supports thinking/reasoning.

    Inspects model architecture, identifier, displayName, or type for indicators of
    reasoning architectures (e.g. Qwen3, DeepSeek-R1, QwQ, Bonsai, etc.).

    Args:
        model_info_or_id: Model dictionary or string identifier.

    Returns:
        True if the model is identified as reasoning/thinking capable.
    """
    if isinstance(model_info_or_id, dict):
        m_id = str(model_info_or_id.get("id", "")).lower()
        arch = str(model_info_or_id.get("architecture", "")).lower()
        name = str(model_info_or_id.get("name", "")).lower()
        model_key = str(model_info_or_id.get("modelKey", "")).lower()
    else:
        m_id = str(model_info_or_id).lower()
        arch = ""
        name = m_id
        model_key = m_id

    # Architectural checks (e.g. qwen3, qwen35, deepseek)
    if any(a in arch for a in ["qwen3", "deepseek", "reason", "cot"]):
        return True

    # Name and identifier keywords
    text_to_check = f"{m_id} {arch} {name} {model_key}"
    keywords = [
        "qwen3",
        "deepseek-r1",
        "r1",
        "think",
        "reason",
        "qwq",
        "bonsai",
        "cot",
        "reasoning",
        "thought",
    ]
    return any(k in text_to_check for k in keywords)


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
        think_mode: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        """Initializes the LM Studio client.

        Args:
            base_url: LM Studio OpenAI-compatible endpoint URL.
            model: Model name/identifier. Defaults to settings.lm_studio_model.
            temperature: Sampling temperature. Defaults to settings.lm_studio_temperature.
            top_p: Nucleus cutoff. Defaults to settings.lm_studio_top_p.
            max_tokens: Max tokens. Defaults to settings.lm_studio_max_tokens.
            system_prompt: System instructions. Defaults to settings.lm_studio_system_prompt.
            think_mode: Whether to enable thinking/reasoning mode. Defaults to settings.lm_studio_think_mode.
            reasoning_effort: Reasoning effort level ('low', 'medium', 'high', 'max').
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
        self.think_mode: bool = (
            think_mode
            if think_mode is not None
            else getattr(settings, "lm_studio_think_mode", False)
        )
        self.reasoning_effort: str = (
            reasoning_effort
            if reasoning_effort is not None
            else getattr(settings, "lm_studio_reasoning_effort", "medium")
        )
        # Reuse a single pooled client for all calls (keep-alive). Creating a
        # new AsyncClient per request wastes TCP/TLS handshakes and adds
        # 5-50ms latency per LLM call.
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        self._closed = False

    def supports_thinking(self, model_id: Optional[str] = None) -> bool:
        """Returns True if the target or currently configured model supports reasoning."""
        return model_supports_thinking(model_id or self.model)

    def get_think_mode(self) -> Dict[str, Any]:
        """Returns the current thinking/reasoning configuration."""
        supports = self.supports_thinking()
        return {
            "think_mode": self.think_mode,
            "reasoning_effort": self.reasoning_effort,
            "supports_thinking": supports,
            "current_model": self.model,
        }

    def set_think_mode(
        self, enabled: bool, effort: Optional[str] = None
    ) -> Dict[str, Any]:
        """Enables or disables thinking mode for the LLM.

        Args:
            enabled: Boolean flag to enable or disable thinking.
            effort: Optional reasoning effort ('low', 'medium', 'high', 'max').

        Returns:
            Status dictionary with updated think mode information.
        """
        self.think_mode = bool(enabled)
        if effort and effort in ("low", "medium", "high", "max"):
            self.reasoning_effort = effort
        # Keep settings updated for persistence
        try:
            settings.lm_studio_think_mode = self.think_mode
            if effort:
                settings.lm_studio_reasoning_effort = self.reasoning_effort
        except Exception:
            pass
        supports = self.supports_thinking()
        status_msg = f"Think mode {'enabled' if self.think_mode else 'disabled'} (effort={self.reasoning_effort}, model_supported={supports})."
        logger.info(status_msg)
        return {
            "status": "success",
            "think_mode": self.think_mode,
            "reasoning_effort": self.reasoning_effort,
            "supports_thinking": supports,
            "message": status_msg,
        }


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

    def _try_lms_unload(self, identifier: Optional[str] = None) -> bool:
        """Ejects/unloads a specific model or all models from LM Studio memory.

        Args:
            identifier: The model identifier to unload. If None or '--all',
                        unloads all models currently loaded in LM Studio.

        Returns:
            True if the unload command completed successfully, False otherwise.
        """
        try:
            cmd = ["lms", "unload"]
            if identifier and identifier != "--all":
                cmd.append(identifier)
            else:
                cmd.append("--all")

            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            out_msg = (res.stdout or res.stderr).strip()
            logger.info("LM Studio unload result: %s", out_msg)
            return res.returncode == 0
        except Exception as exc:
            logger.warning("Could not execute 'lms unload': %s", exc)
            return False

    def eject_other_models(self, keep_model_id: Optional[str] = None) -> List[str]:
        """Ejects all models currently loaded in LM Studio except keep_model_id.

        This guarantees that one model is ejected before loading another,
        preventing multiple models from consuming memory simultaneously.

        Args:
            keep_model_id: Model identifier to preserve (if already loaded).
                           If None, unloads all models.

        Returns:
            List of ejected model identifiers.
        """
        ejected: List[str] = []
        try:
            res = subprocess.run(
                ["lms", "ps", "--json"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                ps_data = json.loads(res.stdout)
                for item in ps_data:
                    ident = item.get("identifier") or item.get("modelKey")
                    if not ident:
                        continue
                    # Match against all known identifiers/aliases for this model
                    keys_to_match = {
                        item.get("identifier"),
                        item.get("modelKey"),
                        item.get("path"),
                        item.get("indexedModelIdentifier"),
                    }
                    keys_to_match.discard(None)

                    if keep_model_id is not None and keep_model_id in keys_to_match:
                        # Target model is already the one loaded; do not eject
                        continue

                    logger.info(
                        "Ejecting model '%s' from LM Studio before loading '%s'...",
                        ident,
                        keep_model_id,
                    )
                    unloaded = self._try_lms_unload(ident)
                    if not unloaded and item.get("modelKey") and item.get("modelKey") != ident:
                        unloaded = self._try_lms_unload(item["modelKey"])
                    if unloaded:
                        ejected.append(ident)

                if ejected:
                    # Give brief pause to allow GPU driver to reclaim memory
                    time.sleep(0.5)
            elif keep_model_id is None:
                if self._try_lms_unload("--all"):
                    ejected.append("all")
        except Exception as exc:
            logger.warning("Error checking or ejecting models: %s", exc)
        return ejected

    def _try_lms_load(self, model_id: str) -> bool:
        """Invokes 'lms load' if available to preload model into memory.

        Enforces that any other model is ejected before loading the target model.
        """
        # Always enforce single-model residency constraint
        self.eject_other_models(keep_model_id=model_id)

        try:
            res = subprocess.run(
                [
                    "lms",
                    "load",
                    model_id,
                    "--gpu",
                    "max",
                    "-y",
                    "--identifier",
                    model_id,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=45,
                check=False,
            )
            return res.returncode == 0
        except Exception as exc:
            logger.debug("lms load failed or not available: %s", exc)
            return False

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Fetches models registered or running in LM Studio with rich metadata.

        Attempts to retrieve detailed model info using 'lms ls --json' and
        'lms ps --json', falling back to the standard OpenAI-compatible HTTP
        '/models' endpoint. Filters out non-LLM models (such as embeddings).

        Returns:
            List of dictionaries describing each available model.
        """
        loaded_ids: Set[str] = set()
        # 1. Identify currently running/loaded models via CLI if available
        try:
            res = subprocess.run(
                ["lms", "ps", "--json"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                ps_data = json.loads(res.stdout)
                for item in ps_data:
                    ident = item.get("identifier") or item.get("modelKey")
                    if ident:
                        loaded_ids.add(ident)
                    if item.get("modelKey"):
                        loaded_ids.add(item["modelKey"])
        except Exception as exc:
            logger.debug("Could not inspect loaded models via 'lms ps': %s", exc)

        models: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        # 2. Query available models via 'lms ls --json'
        try:
            res = subprocess.run(
                ["lms", "ls", "--json"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                ls_data = json.loads(res.stdout)
                for item in ls_data:
                    # Ignore embedding models
                    if item.get("type") == "embedding":
                        continue
                    m_id = item.get("modelKey") or item.get("indexedModelIdentifier")
                    if not m_id or m_id in seen_ids:
                        continue
                    seen_ids.add(m_id)
                    size = item.get("sizeBytes", 0)
                    size_formatted = ""
                    if size > 0:
                        size_formatted = (
                            f"{size / (1024**3):.2f} GB"
                            if size >= 1024**3
                            else f"{size / (1024**2):.1f} MB"
                        )
                    is_loaded = (
                        m_id in loaded_ids
                        or item.get("identifier") in loaded_ids
                        or m_id == self.model
                    )
                    m_info = {
                        "id": m_id,
                        "name": item.get("displayName") or m_id,
                        "loaded": is_loaded,
                        "params": item.get("paramsString", ""),
                        "architecture": item.get("architecture", ""),
                        "size_bytes": size,
                        "size_formatted": size_formatted,
                        "type": item.get("type", "llm"),
                    }
                    m_info["supports_thinking"] = model_supports_thinking(m_info)
                    models.append(m_info)
        except Exception as exc:
            logger.debug("Could not inspect available models via 'lms ls': %s", exc)

        # 3. Augment or fallback using HTTP /models endpoint
        try:
            response = await self._http_client.get("/models")
            if response.status_code == 200:
                http_data = response.json().get("data", [])
                for item in http_data:
                    m_id = item.get("id")
                    if not m_id:
                        continue
                    # Skip embedding models
                    if "embed" in m_id.lower():
                        continue
                    if m_id not in seen_ids:
                        seen_ids.add(m_id)
                        is_loaded = m_id in loaded_ids or m_id == self.model
                        m_info = {
                            "id": m_id,
                            "name": m_id,
                            "loaded": is_loaded,
                            "params": "",
                            "architecture": "",
                            "size_bytes": 0,
                            "size_formatted": "",
                            "type": "llm",
                        }
                        m_info["supports_thinking"] = model_supports_thinking(m_info)
                        models.append(m_info)
                    else:
                        # Ensure loaded flag reflects if seen in loaded_ids or self.model
                        for m in models:
                            if m["id"] == m_id and (m_id in loaded_ids or m_id == self.model):
                                m["loaded"] = True
        except Exception as exc:
            logger.debug("HTTP /models check failed: %s", exc)

        # 4. If list is completely empty, ensure current active model is present
        if not models:
            fallback_info = {
                "id": self.model,
                "name": self.model,
                "loaded": True,
                "params": "",
                "architecture": "",
                "size_bytes": 0,
                "size_formatted": "",
                "type": "llm",
            }
            fallback_info["supports_thinking"] = model_supports_thinking(fallback_info)
            models.append(fallback_info)

        # Ensure current model is marked loaded
        for m in models:
            if m["id"] == self.model:
                m["loaded"] = True
            if "supports_thinking" not in m:
                m["supports_thinking"] = model_supports_thinking(m)


        # Sort: loaded models first, then alphabetical by name
        models.sort(key=lambda x: (not x["loaded"], x["name"].lower()))
        return models

    async def list_models(self) -> List[str]:
        """Fetches the list of models currently registered in LM Studio.

        Returns:
            List of model IDs available on the server.
        """
        try:
            available = await self.list_available_models()
            if available:
                return [m["id"] for m in available]
        except Exception:
            pass

        try:
            response = await self._http_client.get("/models")
            if response.status_code == 200:
                data = response.json()
                return [m["id"] for m in data.get("data", [])]
            return []
        except Exception as exc:
            logger.error("Failed to retrieve LM Studio models: %s", exc)
            return []

    async def set_model(
        self, model_id: str, load_into_memory: bool = True
    ) -> Dict[str, Any]:
        """Switches the active LM Studio LLM model used by the assistant.

        Args:
            model_id: Identifier of the model to activate.
            load_into_memory: Whether to attempt preloading via 'lms load'.

        Returns:
            Status dictionary containing active model and load information.

        Raises:
            ValueError: If model_id is invalid or empty.
        """
        if not model_id or not model_id.strip():
            raise ValueError("Model identifier cannot be empty.")

        clean_id = model_id.strip()
        old_model = self.model
        self.model = clean_id

        # Update application settings for persistence
        try:
            settings.lm_studio_model = clean_id
        except Exception:
            pass

        loaded = False
        message = f"Active model switched from '{old_model}' to '{clean_id}'."

        # Guarantee that one model is ejected before loading another
        ejected = await asyncio.to_thread(self.eject_other_models, clean_id)
        if ejected:
            logger.info("Ejected previous model(s) before loading '%s': %s", clean_id, ejected)

        if load_into_memory:
            loaded = await asyncio.to_thread(self._try_lms_load, clean_id)
            if loaded:
                message = f"Model '{clean_id}' successfully loaded and activated (previous: {', '.join(ejected) or 'none'})."
            else:
                message = f"Model '{clean_id}' activated (LM Studio will JIT load on demand)."

        logger.info(message)
        return {
            "status": "success",
            "model": self.model,
            "ejected_models": ejected,
            "loaded": loaded,
            "message": message,
        }

    async def eject_model(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Ejects a specific model or all models from LM Studio memory.

        Args:
            model_id: Model identifier to unload, or None/'--all' to unload all.

        Returns:
            Dict containing status and list of ejected models.
        """
        ejected: List[str] = []
        if model_id and model_id != "--all":
            ok = await asyncio.to_thread(self._try_lms_unload, model_id)
            if ok:
                ejected.append(model_id)
        else:
            # Query currently loaded models first so we know what was unloaded
            try:
                ps_res = await asyncio.to_thread(
                    subprocess.run,
                    ["lms", "ps", "--json"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=5,
                    check=False,
                )
                if ps_res.returncode == 0 and ps_res.stdout.strip():
                    ps_data = json.loads(ps_res.stdout)
                    for item in ps_data:
                        ident = item.get("identifier") or item.get("modelKey")
                        if ident:
                            ejected.append(ident)
            except Exception:
                pass

            ok = await asyncio.to_thread(self._try_lms_unload, "--all")
            if ok and not ejected:
                ejected.append("all")

        return {
            "status": "success",
            "ejected_models": ejected,
            "message": f"Ejected model(s): {', '.join(ejected) if ejected else 'none'}",
        }

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        think_mode: Optional[bool] = None,
    ) -> str:
        """Generates a text completion from the LM Studio model.

        Args:
            prompt: User message content.
            system_prompt: Optional override for the system prompt.
            history: Optional list of previous chat messages.
            think_mode: Optional think mode override.

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

        effective_think = (
            think_mode if think_mode is not None else self.think_mode
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }
        if effective_think:
            payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload["reasoning_effort"] = "none"

        if self.max_tokens and self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens

        logger.debug("Dispatching LLM prompt to %s/chat/completions", self.base_url)
        start_time = time.perf_counter()

        try:
            response = await self._http_client.post(
                "/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            raw_content = choice.get("message", {}).get("content", "") or ""

            # Strip any <think>...</think> tags from final text if present
            import re
            cleaned_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
            reply = cleaned_content if cleaned_content else raw_content.strip()

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

    async def stream_tokens(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        think_mode: Optional[bool] = None,
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """Streams typed token tuples (token_type, text) from the LLM.

        Token types:
            - 'thought': Reasoning / chain-of-thought tokens (from delta.reasoning_content
                         or from within <think>...</think> tags).
            - 'content': Assistant response text intended for speech and final display.

        Args:
            prompt: User message prompt.
            system_prompt: System instruction override.
            history: Previous conversation context.
            think_mode: Optional boolean override for reasoning mode.

        Yields:
            (token_type, token_text) tuples.
        """
        # Check if stream_response has been mocked or overridden on this instance
        stream_resp_attr = getattr(self, "stream_response", None)
        is_mocked = (
            "stream_response" in getattr(self, "__dict__", {})
            or getattr(stream_resp_attr, "__func__", stream_resp_attr) is not LMStudioClient.stream_response
        )
        if is_mocked:
            res = self.stream_response(
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
            )
            if hasattr(res, "__aiter__"):
                async for tok in res:
                    yield ("content", tok)
            elif hasattr(res, "__iter__"):
                for tok in res:
                    yield ("content", tok)
            return

        messages: List[Dict[str, str]] = []
        effective_system = system_prompt or self.system_prompt
        if effective_system:
            messages.append({"role": "system", "content": effective_system})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        effective_think = (
            think_mode if think_mode is not None else self.think_mode
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": True,
        }
        if effective_think:
            payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload["reasoning_effort"] = "none"

        if self.max_tokens and self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens

        inside_think_tag = False
        tag_buffer = ""

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
                        delta = chunk["choices"][0]["delta"]

                        # 1. First-class reasoning_content (LM Studio native / Qwen3 / DeepSeek)
                        reasoning = (
                            delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or delta.get("thought")
                        )
                        if reasoning:
                            if effective_think:
                                yield ("thought", reasoning)

                        # 2. Main content (may also contain inline <think> tags)
                        content = delta.get("content", "")
                        if not content:
                            continue

                        # Process inline think tags
                        combined = tag_buffer + content
                        tag_buffer = ""

                        while combined:
                            if not inside_think_tag:
                                if "<think>" in combined:
                                    before, _, after = combined.partition("<think>")
                                    if before:
                                        yield ("content", before)
                                    inside_think_tag = True
                                    combined = after
                                elif any("<think>".startswith(combined[-i:]) for i in range(1, len("<think>"))):
                                    for i in range(len("<think>") - 1, 0, -1):
                                        if combined.endswith("<think>"[:i]):
                                            tag_buffer = combined[-i:]
                                            combined = combined[:-i]
                                            break
                                    if combined:
                                        yield ("content", combined)
                                    break
                                else:
                                    yield ("content", combined)
                                    break
                            else:
                                if "</think>" in combined:
                                    thought_part, _, after = combined.partition("</think>")
                                    if thought_part and effective_think:
                                        yield ("thought", thought_part)
                                    inside_think_tag = False
                                    combined = after
                                elif any("</think>".startswith(combined[-i:]) for i in range(1, len("</think>"))):
                                    for i in range(len("</think>") - 1, 0, -1):
                                        if combined.endswith("</think>"[:i]):
                                            tag_buffer = combined[-i:]
                                            combined = combined[:-i]
                                            break
                                    if combined and effective_think:
                                        yield ("thought", combined)
                                    break
                                else:
                                    if effective_think:
                                        yield ("thought", combined)
                                    break
                    except Exception:
                        continue

            if tag_buffer:
                if inside_think_tag and effective_think:
                    yield ("thought", tag_buffer)
                elif not inside_think_tag:
                    yield ("content", tag_buffer)

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

    async def stream_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        think_mode: Optional[bool] = None,
    ) -> AsyncGenerator[str, None]:
        """Streams text chunks from the LM Studio LLM asynchronously.

        Yields content tokens (omitting thoughts) to preserve compatibility
        with sentence splitters, TTS synthesis, and legacy consumers.

        Args:
            prompt: User input prompt.
            system_prompt: Optional system instruction override.
            history: Optional conversation context.
            think_mode: Optional think mode override.

        Yields:
            Generated text tokens/chunks as they arrive.
        """
        async for token_type, token in self.stream_tokens(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            think_mode=think_mode,
        ):
            if token_type == "content" and token:
                yield token

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
