"""Incremental sentence/phrase buffer for streaming LLM -> TTS handover.

This module owns two responsibilities that were previously buried inside
``app.core.llm`` / ``app.core.pipeline``:

1. :class:`SentenceBuffer` / :func:`split_into_sentence_chunks` -- incremental
   detection of speakable chunks from a raw LLM token stream.
2. :func:`clean_text_for_speech` -- strips markdown / lists / code / URLs so
   the TTS engine only ever receives natural spoken language.

Why a separate module (latency rationale):
    TTS must start as soon as the *first natural sentence* is complete, not
    when the full LLM reply finishes. A shared, well-tested splitter lets the
    pipeline, the REST SSE routes, and the WebSocket path all reuse the exact
    same boundary rules, and keeps abbreviation/decimal edge cases in one
    place instead of drifting across files.
"""

from __future__ import annotations

import logging
import re
from typing import AsyncGenerator, List, Optional, Set

logger = logging.getLogger(__name__)

# Common English abbreviations that must not trigger a sentence split.
COMMON_ABBREVIATIONS: Set[str] = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "vs.", "etc.", "e.g.", "i.e.", "approx.", "dept.", "est.",
    "inc.", "ltd.", "corp.", "co.", "st.", "ave.", "blvd.",
    "no.", "fig.", "vol.", "pp.", "jan.", "feb.", "mar.", "apr.",
    "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec.",
}

# Titles that are almost always followed by a proper noun, never a boundary.
_TITLE_ABBREVS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
})

# Matches "U.S.A.", "U.S.", "e.g.", "a.m.", "p.m." style multi-period tokens.
_MULTI_PERIOD_RE = re.compile(r"(?:[A-Za-z]\.){2,}\.?$")
_SINGLE_LETTER_ABBREV_RE = re.compile(r"[A-Za-z]\.$")

# Matches a decimal/number boundary like "3.14", "$3.50", "1,000.5".
_DECIMAL_TAIL_RE = re.compile(r"\d\.\d$")

# Sentence-ending punctuation followed by whitespace (or end handled by flush).
_SENT_END_RE = re.compile(r'([.!?]+["\')\]]*\s+)')

# Markdown / noise patterns stripped before TTS.
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\w)[*_]([^*_]+)[*_](?!\w)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_TABLE_PIPE_RE = re.compile(r"\|")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def is_abbreviation(text: str) -> bool:
    """Checks whether trailing token in text is an abbreviation.

    Handles single titles (``Dr.``), common abbreviations (``e.g.``),
    single-letter initials (``J.``), and multi-period forms (``U.S.A.``),
    so the splitter does not emit a TTS fragment mid-name/number.

    Args:
        text: Text string to inspect.

    Returns:
        True if the trailing token looks like an abbreviation.
    """
    words = text.strip().split()
    if not words:
        return False
    last_word = words[-1]
    lowered = last_word.lower().rstrip('"\')]:;,')

    if lowered in COMMON_ABBREVIATIONS:
        return True
    if lowered in _TITLE_ABBREVS:
        return True
    # "U.S.A.", "U.S.", "e.g." -- strip trailing punctuation then test.
    stripped = last_word.strip('"\')]:;,!?')
    if _MULTI_PERIOD_RE.search(stripped):
        return True
    # Single capital initial "J." / "T." at end of candidate.
    if _SINGLE_LETTER_ABBREV_RE.match(stripped):
        return True
    # "a.m." / "p.m." lowercase variants.
    if stripped.lower() in ("a.m.", "p.m."):
        return True
    # Lone two-char form "U." handled by legacy rule.
    if len(stripped) == 2 and stripped[0].isalpha() and stripped[1] == ".":
        return True
    return False


def _ends_with_decimal(candidate: str) -> bool:
    """Returns True when candidate ends mid-decimal-number (``3.14``)."""
    tail = candidate.strip()[-6:]
    if _DECIMAL_TAIL_RE.search(candidate.strip()):
        return True
    # Tokenizer may split "$3" + ".50" with a space: "$3. 50" quirk.
    if re.search(r"\d\.\s?\d$", candidate.strip()):
        return True
    # Version-like "v1. 2" guard.
    _ = tail
    return False


def clean_text_for_speech(text: str) -> str:
    """Strips markdown/formatting so TTS receives natural spoken language.

    Removes code blocks, inline code, bold/italic markers, headings, bullet
    and numbered list prefixes, tables pipes, URLs, and HTML tags, then
    collapses whitespace. Empty or punctuation-only input returns "".

    Args:
        text: Raw LLM sentence text.

    Returns:
        Cleaned plain-language string safe for Kokoro synthesis.
    """
    if not text:
        return ""
    cleaned = _CODE_BLOCK_RE.sub(" ", text)
    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = _INLINE_CODE_RE.sub(r"\1", cleaned)
    cleaned = _BOLD_RE.sub(r"\1", cleaned)
    cleaned = _ITALIC_RE.sub(r"\1", cleaned)
    cleaned = _HEADING_RE.sub("", cleaned)
    cleaned = _BULLET_RE.sub("", cleaned)
    cleaned = _NUMBERED_RE.sub("", cleaned)
    cleaned = _TABLE_PIPE_RE.sub(" ", cleaned)
    # Drop markdown table separator rows ("|---|---|").
    cleaned = re.sub(r"^\s*[-|:\s]+\s*$", " ", cleaned, flags=re.MULTILINE)
    # JSON-ish braces are unreadable aloud; soften them.
    cleaned = cleaned.replace("{", " ").replace("}", " ")
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)
    cleaned = cleaned.strip(" \t\n-*:#>")
    cleaned = cleaned.strip()
    # Remove stray empty-list artefacts.
    if cleaned and not any(c.isalnum() for c in cleaned):
        return ""
    return cleaned


class SentenceBuffer:
    """Incremental buffer turning an LLM token stream into speakable chunks.

    Balances latency vs naturalness:

    * ``.`` / ``?`` / ``!`` (+ newline) trigger immediate emission once the
      chunk reaches ``min_chars`` alphanumerics.
    * ``,`` / ``:`` / ``;`` / em-dash trigger a fallback split only when the
      buffer grows long (``max_chars``) so run-on sentences still start
      speaking early.
    * Extremely short fragments (``"Hi."``) stay buffered until more context
      arrives, avoiding wasteful single-word TTS round-trips.

    Attributes:
        min_chars: Minimum characters before a punctuation split is emitted.
        max_chars: Length at which comma/colon fallback splitting activates.
        comma_min_pos: Earliest offset where a comma split is considered.
    """

    def __init__(
        self,
        min_chars: int = 12,
        max_chars: int = 160,
        comma_min_pos: int = 40,
    ) -> None:
        """Initializes the sentence buffer.

        Args:
            min_chars: Minimum characters for a punctuation-triggered chunk.
            max_chars: Buffer length that forces a phrase-level fallback split.
            comma_min_pos: Minimum offset for comma/colon fallback splits.
        """
        self.min_chars = max(1, int(min_chars))
        self.max_chars = max(32, int(max_chars))
        self.comma_min_pos = max(8, int(comma_min_pos))
        self._buffer: str = ""

    def feed(self, token: str) -> List[str]:
        """Feeds one token into the buffer, returning newly completed chunks.

        Args:
            token: Raw LLM token delta string.

        Returns:
            List of completed sentence strings (usually 0 or 1).
        """
        if not token:
            return []
        self._buffer += token
        return self._drain()

    def flush(self) -> Optional[str]:
        """Flushes remaining buffered text at end of stream.

        Returns:
            Remaining text, or None when the buffer holds no speakable text.
        """
        final = self._buffer.strip()
        self._buffer = ""
        if final and any(c.isalnum() for c in final):
            return final
        return None

    @property
    def buffered_text(self) -> str:
        """Returns currently buffered (not yet emitted) text."""
        return self._buffer

    def _drain(self) -> List[str]:
        out: List[str] = []
        while True:
            chunk = self._try_pop()
            if chunk is None:
                break
            out.append(chunk)
        return out

    def _try_pop(self) -> Optional[str]:
        buf = self._buffer
        if not buf or not buf.strip():
            return None

        # 1. Sentence-ending punctuation (. ! ?) followed by whitespace or newline.
        for match in _SENT_END_RE.finditer(buf):
            candidate = buf[:match.start() + len(match.group(1).rstrip())].strip()
            if not candidate or not any(c.isalnum() for c in candidate):
                continue
            if is_abbreviation(candidate):
                continue
            if _ends_with_decimal(candidate):
                continue
            if len(candidate) >= self.min_chars:
                self._buffer = buf[match.end():].lstrip()
                return candidate
            # Too short: keep waiting for more tokens (avoids "Hi." / "Ok."
            # each costing a full Kokoro round-trip).
            continue

        # 2. Hard split on newline (e.g. headings or bullet items without terminal punctuation).
        newline_pos = buf.find("\n")
        if newline_pos != -1:
            chunk = buf[:newline_pos].strip()
            rest = buf[newline_pos + 1:].lstrip()
            if chunk and any(c.isalnum() for c in chunk):
                if is_abbreviation(chunk):
                    # e.g. "Dr.\nSmith" -- keep buffering.
                    return None
                if len(chunk) < self.min_chars and rest:
                    # Tiny fragment with more text coming: merge, wait.
                    self._buffer = chunk + " " + rest
                    return None
                if len(chunk) >= self.min_chars or not rest:
                    self._buffer = rest
                    return chunk
            else:
                self._buffer = rest
                buf = rest
                if not buf or not buf.strip():
                    return None

        # 3. Fallback for long run-on sentences: split at , ; : em-dash.
        if len(buf) > self.max_chars:
            comma_match = re.search(r"([,;:—–]\s+)", buf)
            if comma_match and comma_match.start() >= self.comma_min_pos:
                candidate = buf[:comma_match.start() + 1].strip()
                if candidate and any(c.isalnum() for c in candidate):
                    self._buffer = buf[comma_match.end():].lstrip()
                    return candidate

        return None


async def split_into_sentence_chunks(
    token_stream: AsyncGenerator[str, None],
    min_chunk_length: int = 12,
) -> AsyncGenerator[str, None]:
    """Splits an async token stream into sentence-sized chunks.

    Buffers incoming tokens and yields complete sentences whenever a full
    stop / newline / ``!`` / ``?`` boundary is reached, while avoiding
    premature splits on decimals (``3.14``), abbreviations (``Dr.``,
    ``U.S.A.``, ``e.g.``), or tiny fragments.

    Args:
        token_stream: Async generator yielding raw LLM token deltas.
        min_chunk_length: Minimum characters required to emit a chunk.

    Yields:
        Sentence strings ready for immediate TTS synthesis.
    """
    buf = SentenceBuffer(min_chars=min_chunk_length)
    async for token in token_stream:
        for chunk in buf.feed(token):
            yield chunk
    tail = buf.flush()
    if tail:
        yield tail
