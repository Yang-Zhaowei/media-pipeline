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

Breaks are driven by explicit punctuation, hard line breaks in the original
text, and pauses measured from the alignment itself; character and duration
budgets only apply when nothing more natural is available.
"""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

__all__ = [
    "AlignedToken",
    "AlignmentError",
    "AlignmentMismatchError",
    "Caption",
    "CaptionError",
    "DEFAULT_BREAK_AFTER",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MAX_DURATION",
    "DEFAULT_PAUSE_THRESHOLD",
    "DEFAULT_SOFT_BREAK_AFTER",
    "build_captions",
    "compile_srt",
    "format_timestamp",
    "load_alignment",
    "parse_alignment",
    "render_srt",
]

#: Punctuation that always ends a caption. Sentence terminators and the
#: semicolon; commas are handled as candidate breakpoints instead so that short
#: clauses are not fragmented into sub-second captions.
DEFAULT_BREAK_AFTER = "。！？!?…；;"

#: Clause punctuation that is only a *candidate* breakpoint. A candidate is used
#: when a caption would otherwise exceed its character or duration budget, but
#: does not end a caption on its own.
DEFAULT_SOFT_BREAK_AFTER = "，,、：:"

#: An alignment gap of at least this many seconds ends a caption.
DEFAULT_PAUSE_THRESHOLD = 0.3

#: Maximum number of content characters per caption. Whitespace and
#: punctuation do not count against this budget.
DEFAULT_MAX_CHARS = 20

#: Maximum caption duration in seconds.
DEFAULT_MAX_DURATION = 6.0

# Unicode general categories treated as non-spoken text (never aligned).
_SKIPPABLE_CATEGORIES = frozenset("PZ")

# Characters that force a caption break regardless of configuration.
_MANDATORY_BREAKS = "\r\n"

# Punctuation that stays attached to the caption it closes when a break is
# already pending: repeated sentence marks plus closing quotes and brackets.
_TRAILING_PUNCTUATION = frozenset("。！？!?…；;，,、：:.．”’」』）)］]｝}》〉】〕〗〙〛")


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


class _CharTiming(NamedTuple):
    """Timing inherited by one character of the original text."""

    token: int
    start: float
    end: float


def _assign_timings(
    original_text: str, tokens: Sequence[AlignedToken]
) -> list[_CharTiming | None]:
    """Map each original character to its aligned timing, or ``None``.

    Aligned tokens are matched against the original text in order. Non-spoken
    characters (whitespace and punctuation) have no timing; any other mismatch
    is an error so that a stale alignment can never silently shift captions.
    """

    timings: list[_CharTiming | None] = [None] * len(original_text)
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
            timings[position] = _CharTiming(token_index, token.start, token.end)
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


def _content_weight(char: str) -> int:
    """How much a character counts against ``max_chars`` (punctuation is free)."""

    return 0 if _is_skippable(char) else 1


def _buffer_state(
    buffer: Sequence[tuple[str, _CharTiming | None]],
) -> tuple[int, float | None, float | None]:
    """Recount a buffer after it has been split."""

    count = sum(_content_weight(char) for char, _ in buffer)
    start = next((timing.start for _, timing in buffer if timing is not None), None)
    end = next((timing.end for _, timing in reversed(buffer) if timing is not None), None)
    return count, start, end


def build_captions(
    original_text: str,
    tokens: Sequence[AlignedToken],
    *,
    break_after: str = DEFAULT_BREAK_AFTER,
    soft_break_after: str = DEFAULT_SOFT_BREAK_AFTER,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_duration: float = DEFAULT_MAX_DURATION,
    pause_threshold: float = DEFAULT_PAUSE_THRESHOLD,
) -> list[Caption]:
    """Split the original text into timed captions.

    Breaks are decided in this order:

    - a hard line break in the original text always ends the caption;
    - a character in ``break_after`` requests a break, but trailing punctuation
      (closing quotes/brackets and repeated sentence marks) is absorbed into the
      caption it closes before the break takes effect;
    - an alignment gap of at least ``pause_threshold`` seconds ends the caption;
    - otherwise the caption ends before it would exceed ``max_chars`` content
      characters or ``max_duration`` seconds. The most recent natural candidate
      breakpoint (``soft_break_after`` punctuation or any positive alignment
      gap) is preferred, and only when there is no candidate is the caption cut
      at the budget boundary.

    Whitespace and punctuation never count against ``max_chars``. Untimed text
    that is not attached to a caption (for example a leading quote) is carried
    into the following caption.
    """

    if max_chars < 1:
        raise ValueError(f"max_chars must be at least 1, got {max_chars}")
    if max_duration <= 0:
        raise ValueError(f"max_duration must be positive, got {max_duration}")
    if pause_threshold < 0:
        raise ValueError(f"pause_threshold must not be negative, got {pause_threshold}")

    timings = _assign_timings(original_text, tokens)

    # Alignment gaps: the text index of the first character of a token that
    # starts after a pause, mapped to the pause length in seconds.
    pauses: dict[int, float] = {}
    previous_token: int | None = None
    for index, timing in enumerate(timings):
        if timing is None:
            continue
        if previous_token is not None and timing.token != previous_token:
            gap = timing.start - tokens[timing.token - 1].end
            if gap > 0:
                pauses[index] = gap
        previous_token = timing.token

    captions: list[Caption] = []
    buffer: list[tuple[str, _CharTiming | None]] = []
    candidates: list[int] = []
    carry = ""
    break_pending = False
    count, start, end = 0, None, None

    def reset() -> None:
        nonlocal buffer, candidates, count, start, end
        buffer = []
        candidates = []
        count, start, end = 0, None, None

    def finalize() -> None:
        nonlocal carry
        text = "".join(char for char, _ in buffer).strip()
        if text:
            if start is None or end is None:
                # Untimed text (e.g. a leading quote) belongs to the next caption.
                carry += text
            else:
                captions.append(Caption(start=start, end=end, text=carry + text))
                carry = ""
        reset()

    def split(offset: int) -> None:
        nonlocal buffer, candidates, carry, count, start, end
        prefix, buffer = buffer[:offset], buffer[offset:]
        prefix_text = "".join(char for char, _ in prefix).strip()
        prefix_start = next((t.start for _, t in prefix if t is not None), None)
        prefix_end = next((t.end for _, t in reversed(prefix) if t is not None), None)
        if prefix_text and prefix_start is not None and prefix_end is not None:
            captions.append(
                Caption(start=prefix_start, end=prefix_end, text=carry + prefix_text)
            )
            carry = ""
        else:
            carry += prefix_text
        candidates = [position - offset for position in candidates if position > offset]
        count, start, end = _buffer_state(buffer)

    for index, (char, timing) in enumerate(zip(original_text, timings)):
        if char in _MANDATORY_BREAKS:
            finalize()
            break_pending = False
            continue

        # Absorb punctuation that closes the caption a break was requested for.
        if break_pending and buffer:
            if char.isspace() or char in _TRAILING_PUNCTUATION:
                buffer.append((char, timing))
                if timing is not None:
                    if start is None:
                        start = timing.start
                    end = timing.end if end is None else max(end, timing.end)
                continue
            finalize()
            break_pending = False

        # A measurable pause ends the caption; a smaller gap is only a candidate.
        gap = pauses.get(index)
        if gap is not None and buffer and start is not None:
            if gap >= pause_threshold:
                finalize()
                break_pending = False
            else:
                candidates.append(len(buffer))

        weight = _content_weight(char)
        while buffer:
            projected_end = timing.end if timing is not None else end
            over_budget = count + weight > max_chars or (
                start is not None
                and projected_end is not None
                and projected_end - start > max_duration
            )
            if not over_budget:
                break
            break_at = max(
                (position for position in candidates if 0 < position < len(buffer)),
                default=None,
            )
            if break_at is None:
                finalize()
                break
            split(break_at)

        if not buffer and char.isspace():
            continue

        buffer.append((char, timing))
        count += weight
        if timing is not None:
            if start is None:
                start = timing.start
            end = timing.end if end is None else max(end, timing.end)

        if char in soft_break_after:
            candidates.append(len(buffer))
        if char in break_after:
            break_pending = True

    finalize()
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
    soft_break_after: str = DEFAULT_SOFT_BREAK_AFTER,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_duration: float = DEFAULT_MAX_DURATION,
    pause_threshold: float = DEFAULT_PAUSE_THRESHOLD,
) -> str:
    """Compile original text and alignment tokens into an SRT document."""

    captions = build_captions(
        original_text,
        tokens,
        break_after=break_after,
        soft_break_after=soft_break_after,
        max_chars=max_chars,
        max_duration=max_duration,
        pause_threshold=pause_threshold,
    )
    return render_srt(captions)
