"""Deterministic caption compilation: original text + forced alignment -> SRT.

This module is pure Python. It neither loads a model nor touches CUDA: it turns
the original script text plus the raw output of a forced aligner (a list of
``{"text", "start", "end"}`` records) into a standard SRT document.

Two hard rules keep the result deterministic and verifiable:

1. Caption text is the original text, unchanged. Punctuation and whitespace are
   never invented, dropped, or rewritten; they are only used to decide where a
   caption may be broken.
2. Alignment must match the original content characters exactly (in order).
   Anything else raises :class:`AlignmentMismatchError` instead of guessing.
"""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "AlignedToken",
    "AlignmentError",
    "AlignmentMismatchError",
    "Caption",
    "CaptionError",
    "DEFAULT_BREAK_AFTER",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MAX_DURATION",
    "build_captions",
    "compile_srt",
    "format_timestamp",
    "load_alignment",
    "parse_alignment",
    "render_srt",
]

#: Punctuation that always ends a caption. Sentence terminators plus the
#: semicolon; commas are intentionally left to the character/duration budgets so
#: that short clauses are not fragmented into sub-second captions.
DEFAULT_BREAK_AFTER = "。！？!?…；;"

#: Maximum number of content characters per caption. Whitespace and
#: punctuation do not count against this budget.
DEFAULT_MAX_CHARS = 20

#: Maximum caption duration in seconds.
DEFAULT_MAX_DURATION = 6.0

# Unicode general categories treated as non-spoken text (never aligned).
_SKIPPABLE_CATEGORIES = frozenset("PZ")

# Characters that force a caption break regardless of configuration.
_MANDATORY_BREAKS = "\r\n"


class CaptionError(ValueError):
    """Base class for caption compilation errors."""


class AlignmentError(CaptionError):
    """The forced-alignment data is malformed or unusable."""


class AlignmentMismatchError(AlignmentError):
    """The forced-alignment data does not match the original text."""


@dataclass(frozen=True)
class AlignedToken:
    """One aligned unit of text with a start/end time in seconds."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Caption:
    """A single caption event."""

    start: float
    end: float
    text: str


def _as_seconds(value: object, position: int, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlignmentError(
            f"alignment item {position} field {field!r} must be a number, got {value!r}"
        )
    seconds = float(value)
    if not math.isfinite(seconds):
        raise AlignmentError(
            f"alignment item {position} field {field!r} must be finite, got {value!r}"
        )
    return seconds


def parse_alignment(items: Iterable[object]) -> list[AlignedToken]:
    """Validate and normalize raw alignment records.

    Accepts the JSON shape produced by the forced aligner
    (``{"text": ..., "start": ..., "end": ...}``). Empty/whitespace-only tokens
    are dropped. Raises :class:`AlignmentError` on malformed or overlapping
    records.
    """

    tokens: list[AlignedToken] = []
    previous_end = 0.0
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise AlignmentError(
                f"alignment item {position} must be an object, got {type(item).__name__}"
            )
        for field in ("text", "start", "end"):
            if field not in item:
                raise AlignmentError(f"alignment item {position} is missing field {field!r}")
        raw_text = item["text"]
        if not isinstance(raw_text, str):
            raise AlignmentError(
                f"alignment item {position} field 'text' must be a string, got {raw_text!r}"
            )
        text = raw_text.strip()
        if not text:
            continue
        start = _as_seconds(item["start"], position, "start")
        end = _as_seconds(item["end"], position, "end")
        if start < 0:
            raise AlignmentError(f"alignment item {position} starts before zero: {start}")
        if end < start:
            raise AlignmentError(
                f"alignment item {position} ends before it starts: {start} > {end}"
            )
        if tokens and start < previous_end:
            raise AlignmentError(
                f"alignment item {position} starts at {start}, before the previous "
                f"item ends at {previous_end}"
            )
        tokens.append(AlignedToken(text=text, start=start, end=end))
        previous_end = end
    return tokens


def load_alignment(path: str | Path, *, encoding: str = "utf-8") -> list[AlignedToken]:
    """Load and validate a raw alignment JSON file."""

    try:
        with open(path, encoding=encoding) as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise AlignmentError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise AlignmentError(
            f"{path} must contain a JSON array of alignment items, got {type(data).__name__}"
        )
    return parse_alignment(data)


def _is_skippable(char: str) -> bool:
    """True if a character carries no speech and therefore no alignment entry."""

    return char.isspace() or unicodedata.category(char)[0] in _SKIPPABLE_CATEGORIES


def _assign_timings(
    original_text: str, tokens: Sequence[AlignedToken]
) -> list[tuple[float, float] | None]:
    """Map each original character to a ``(start, end)`` time, or ``None``.

    Aligned tokens are matched against the original text in order. Non-spoken
    characters (whitespace and punctuation) have no timing; any other mismatch
    is an error so that a stale alignment can never silently shift captions.
    """

    timings: list[tuple[float, float] | None] = [None] * len(original_text)
    token_index = 0
    char_index = 0

    for position, char in enumerate(original_text):
        if token_index >= len(tokens):
            if not _is_skippable(char):
                raise AlignmentMismatchError(
                    f"original text character {position} ({char!r}) has no aligned token"
                )
            continue

        token = tokens[token_index]
        if char == token.text[char_index]:
            timings[position] = (token.start, token.end)
            char_index += 1
            if char_index == len(token.text):
                token_index += 1
                char_index = 0
        elif _is_skippable(char):
            continue
        else:
            raise AlignmentMismatchError(
                f"original text character {position} ({char!r}) does not match aligned "
                f"token {token_index} ({token.text!r})"
            )

    if token_index < len(tokens):
        token = tokens[token_index]
        if char_index:
            raise AlignmentMismatchError(
                f"aligned token {token_index} ({token.text!r}) is only partially present "
                "in the original text"
            )
        raise AlignmentMismatchError(
            f"aligned token {token_index} ({token.text!r}) is not present in the original text"
        )

    return timings


def build_captions(
    original_text: str,
    tokens: Sequence[AlignedToken],
    *,
    break_after: str = DEFAULT_BREAK_AFTER,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_duration: float = DEFAULT_MAX_DURATION,
) -> list[Caption]:
    """Split the original text into timed captions.

    A caption is broken after any character in ``break_after``, on a hard line
    break in the original text, or before the next character would exceed
    ``max_chars`` content characters or ``max_duration`` seconds. Whitespace and
    punctuation never count against ``max_chars`` so that punctuation stays
    attached to the caption it belongs to.
    """

    if max_chars < 1:
        raise ValueError(f"max_chars must be at least 1, got {max_chars}")
    if max_duration <= 0:
        raise ValueError(f"max_duration must be positive, got {max_duration}")

    timings = _assign_timings(original_text, tokens)

    captions: list[Caption] = []
    buffer: list[str] = []
    count = 0
    start: float | None = None
    end: float | None = None
    carry = ""

    def flush() -> None:
        nonlocal buffer, count, start, end, carry
        text = "".join(buffer).strip()
        buffer = []
        count = 0
        if not text:
            start = end = None
            return
        if start is None or end is None:
            # Untimed text (e.g. a leading ellipsis) belongs to the next caption.
            carry += text
            start = end = None
            return
        captions.append(Caption(start=start, end=end, text=carry + text))
        carry = ""
        start = end = None

    for char, timing in zip(original_text, timings):
        if char in _MANDATORY_BREAKS:
            flush()
            continue

        weight = 0 if _is_skippable(char) else 1
        if buffer:
            projected_end = timing[1] if timing is not None else end
            over_chars = count + weight > max_chars
            over_time = (
                start is not None
                and projected_end is not None
                and projected_end - start > max_duration
            )
            if over_chars or over_time:
                flush()
                if char.isspace():
                    continue

        buffer.append(char)
        count += weight
        if timing is not None:
            if start is None:
                start = timing[0]
            end = timing[1] if end is None else max(end, timing[1])

        if char in break_after:
            flush()

    flush()
    return captions


def format_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp (``HH:MM:SS,mmm``)."""

    if seconds < 0:
        raise ValueError(f"SRT timestamps cannot be negative, got {seconds}")
    total_ms = int(
        (Decimal(str(seconds)) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def render_srt(captions: Sequence[Caption]) -> str:
    """Render captions as an SRT document (UTF-8 text, ``\\n`` line endings)."""

    blocks = [
        f"{index}\n{format_timestamp(caption.start)} --> "
        f"{format_timestamp(caption.end)}\n{caption.text}"
        for index, caption in enumerate(captions, start=1)
    ]
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def compile_srt(
    original_text: str,
    tokens: Sequence[AlignedToken],
    *,
    break_after: str = DEFAULT_BREAK_AFTER,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_duration: float = DEFAULT_MAX_DURATION,
) -> str:
    """Compile original text and alignment tokens into an SRT document."""

    captions = build_captions(
        original_text,
        tokens,
        break_after=break_after,
        max_chars=max_chars,
        max_duration=max_duration,
    )
    return render_srt(captions)
