"""Bounded per-session conversation history manager.

Keeps only the context required for the current request: the last
``max_turns`` user+assistant pairs per session. Small models (Qwen 2.5 0.5B)
degrade and slow down with huge histories, so history is hard-bounded and
isolated per session id -- concurrent browser tabs / CLI / WebSocket clients
never share context.

Why separate from the pipeline (latency rationale):
    History slicing happens on every LLM call. Centralising the bound here
    guarantees the prompt sent to LM Studio stays small (fast TTFT, low VRAM)
    no matter which entrypoint (REST, stream, WebSocket, CLI) initiated it.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Deque, Dict, List, Optional


class ConversationManager:
    """Thread-safe bounded history store keyed by session id.

    Attributes:
        max_turns: Number of user+assistant pairs retained per session.
    """

    def __init__(self, max_turns: int = 3) -> None:
        """Initializes the manager.

        Args:
            max_turns: Pairs of turns to keep (each pair = 2 messages).
        """
        self.max_turns = max(1, int(max_turns))
        self._histories: Dict[str, Deque[Dict[str, str]]] = {}
        self._lock = asyncio.Lock()

    @property
    def limit(self) -> int:
        """Maximum stored messages per session (turns * 2)."""
        return self.max_turns * 2

    async def get_slice(self, session_id: str) -> List[Dict[str, str]]:
        """Returns up to the last ``limit`` messages for a session."""
        async with self._lock:
            dq = self._histories.get(session_id)
            if not dq:
                return []
            return list(dq)[-self.limit:]

    async def append(self, session_id: str, user: str, assistant: str) -> None:
        """Appends one turn, evicting oldest messages beyond the bound."""
        async with self._lock:
            dq = self._histories.setdefault(session_id, deque())
            dq.append({"role": "user", "content": user})
            dq.append({"role": "assistant", "content": assistant})
            while len(dq) > self.limit:
                dq.popleft()

    async def clear(self, session_id: Optional[str] = None) -> None:
        """Clears one session, or all sessions when ``session_id`` is None."""
        async with self._lock:
            if session_id is None:
                self._histories.clear()
            else:
                self._histories.pop(session_id, None)

    def set_max_turns(self, max_turns: int) -> None:
        """Updates the retention bound."""
        self.max_turns = max(1, int(max_turns))
