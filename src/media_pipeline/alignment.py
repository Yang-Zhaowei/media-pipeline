"""Portable alignment contracts and deterministic validation (v0).

This module holds the *portable* side of the Speech Pipeline's forced-alignment
stage:

- the request/output contracts,
- request and input-WAV validation,
- validation of the raw forced-alignment output before it is written,

It is pure Python. It neither imports ``torch``, ``qwen_asr``, nor any CUDA
binding at import time or at use time. The model/runtime adapter lives in
:mod:`media_pipeline.runtimes.qwen_aligner`, which is imported lazily.

Scope contract (v0):

- One :class:`AlignmentRequest` is one already-segmented utterance. This module
  does not split long text; the caller passes a single utterance.
- The runtime returns one flat list of ``{"text", "start_time", "end_time"}``
  records for that utterance. This module maps those into the shared
  :class:`~media_pipeline.captions.AlignedToken` representation and validates
  them, but never sorts, rounds, truncates, repairs, adds to, or removes from
  the alignment content.
- The output JSON keeps the existing ``{"text", "start", "end"}`` shape that the
  Audio Postprocess and Caption Compiler stages consume, with the model's
  timestamp precision preserved.
- No model, host, or output path is hard-coded here.

Text/alignment *matching* against the original text is the Caption Compiler's
job (:mod:`media_pipeline.captions`); this module only validates the alignment
data itself and never performs matching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .captions import AlignedToken, AlignmentError, parse_alignment
from .postprocess import AudioPostprocessError, read_wav

__all__ = [
    "AlignmentArtifact",
    "AlignmentRequest",
    "AlignmentRequestError",
    "AlignmentRuntimeError",
    "validate_alignment",
    "validate_request",
    "validate_wav",
]

#: A purely numerical tolerance for coarse timestamp-vs-duration comparisons,
#: shared in spirit with Audio Postprocess's own alignment tolerance.
_ALIGNMENTS_TOLERANCE = 1e-6


class AlignmentRequestError(ValueError):
    """A request or its input WAV is malformed and can never be aligned."""


class AlignmentRuntimeError(RuntimeError):
    """Invalid model/runtime input or output, or the runtime failed.

    Raised by the portable validators for malformed alignment output and by the
    runtime adapter for model/runtime failures. Exception chaining is preserved
    so the underlying cause is never lost.
    """


@dataclass(frozen=True)
class AlignmentRequest:
    """One already-segmented utterance to align.

    ``text`` is the full utterance transcript passed to the forced aligner (no
    sentence splitting by this module). ``language`` selects the model's
    language mode. ``wav_path`` is the original WAV to align and is never
    modified or cropped by this module.
    """

    wav_path: Path
    text: str
    language: str


@dataclass(frozen=True)
class AlignmentArtifact:
    """The written alignment JSON, as consumed by Audio Postprocess.

    ``sample_rate`` and ``frames`` are read back from the input WAV so they are
    authoritative (they describe the timeline the alignment was computed on),
    and ``token_count`` is the number of effective alignment tokens.
    """

    alignment_path: Path
    sample_rate: int
    frames: int
    token_count: int

    @property
    def duration(self) -> float:
        """Total input-WAV duration in seconds (derived from ``frames``)."""

        return self.frames / self.sample_rate


def validate_request(request: AlignmentRequest) -> None:
    """Validate a request before it reaches the runtime.

    ``text`` and ``language`` must be non-empty strings. Raises
    :class:`AlignmentRequestError` otherwise.
    """

    if not isinstance(request.text, str) or not request.text.strip():
        raise AlignmentRequestError(
            "AlignmentRequest.text must be a non-empty string"
        )
    if not isinstance(request.language, str) or not request.language.strip():
        raise AlignmentRequestError(
            "AlignmentRequest.language must be a non-empty string"
        )


def validate_wav(wav_path: str | Path) -> tuple[int, int]:
    """Validate the input WAV and return ``(sample_rate, frames)``.

    The input WAV must be a valid, non-empty, mono 16-bit PCM WAV. Reuses the
    tested Audio Postprocess reader so the exact format semantics (including
    junk-chunk ordering, stereo rejection, and bit-depth rejection) stay
    consistent across the pipeline. Raises :class:`AlignmentRequestError` for a
    missing, unreadable, or malformed WAV, or for an empty WAV.
    """

    try:
        _samples, sample_rate, frames = read_wav(wav_path)
    except AudioPostprocessError as exc:
        raise AlignmentRequestError(f"input WAV is invalid: {exc}") from exc

    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise AlignmentRequestError(
            f"input WAV has an invalid sample rate: {sample_rate!r}"
        )
    if not isinstance(frames, int) or frames <= 0:
        raise AlignmentRequestError(f"input WAV is empty: {frames} frame(s)")

    return sample_rate, frames


def validate_alignment(
    items: Iterable[object],
    *,
    sample_rate: int,
    frames: int,
) -> list[AlignedToken]:
    """Validate raw alignment records against the input WAV.

    First enforces the production alignment output contract on *every* raw
    record -- blank or not -- via :func:`_reject_invalid_record_timestamps`, so
    invalid timestamps (non-numeric, non-finite, negative, end-before-start) can
    never be hidden by blank-token dropping. It then delegates to the Caption
    Compiler's :func:`~media_pipeline.captions.parse_alignment` so the same
    normalization, overlap rejection, zero-duration tolerance, and blank-token
    dropping apply here as downstream, and requires at least one effective token.
    Finally it rejects an alignment whose final timestamp exceeds the WAV
    duration beyond :data:`_ALIGNMENTS_TOLERANCE`.

    The timestamps are compared to the WAV duration but never rewritten,
    rounded, sorted, truncated, added to, or removed. Returns the validated
    :class:`AlignedToken` list. Raises :class:`AlignmentError` on any problem.
    """

    # Reject malformed timestamps on *every* raw record before blank-dropping.
    # parse_alignment() drops whitespace-only records before checking their
    # timestamps, so a blank runtime record carrying NaN/Inf/negative/otherwise
    # invalid timestamps would otherwise slip past validation and be written.
    # This boundary check inspects start/end of each record -- blank or not --
    # so invalid data can never be hidden by blank-token dropping. It enforces
    # only per-record validity; ordering/overlap/duration stay with
    # parse_alignment() on the effective tokens, keeping its semantics intact.
    _reject_invalid_record_timestamps(items)

    tokens = parse_alignment(items)
    if not tokens:
        raise AlignmentError("alignment has no effective tokens")

    if not (
        isinstance(sample_rate, int)
        and isinstance(frames, int)
        and sample_rate > 0
        and frames > 0
    ):
        raise AlignmentError(
            f"cannot bound alignment without positive WAV metadata: "
            f"sample_rate={sample_rate!r}, frames={frames!r}"
        )

    duration = frames / sample_rate
    last_end = tokens[-1].end
    if last_end > duration + _ALIGNMENTS_TOLERANCE:
        raise AlignmentError(
            f"alignment ends at {last_end}, past the WAV duration {duration}"
        )

    return tokens


def _reject_invalid_record_timestamps(items: Iterable[object]) -> None:
    """Reject invalid timestamps on every raw record, blank or not.

    ``parse_alignment()`` drops whitespace-only records before checking their
    timestamps, so a blank runtime record carrying NaN/Inf/negative/otherwise
    invalid timestamps would otherwise bypass validation. This helper inspects
    the ``start``/``end`` of every record up front so that invalid data can
    never be hidden by blank-token dropping. It enforces only per-record
    validity (numeric, finite, non-negative, end >= start); ordering,
    overlap, and duration checks stay with :func:`parse_alignment` on the
    effective tokens, so its downstream semantics are unchanged.
    """

    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise AlignmentError(
                f"alignment item {position} must be an object, "
                f"got {type(item).__name__}"
            )
        for field in ("start", "end"):
            if field not in item:
                raise AlignmentError(
                    f"alignment item {position} is missing field {field!r}"
                )
            value = item[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AlignmentError(
                    f"alignment item {position} field {field!r} must be a "
                    f"number, got {value!r}"
                )
            seconds = float(value)
            if not math.isfinite(seconds):
                raise AlignmentError(
                    f"alignment item {position} field {field!r} must be "
                    f"finite, got {value!r}"
                )
        start = float(item["start"])
        end = float(item["end"])
        if start < 0:
            raise AlignmentError(f"alignment item {position} starts before zero: {start}")
        if end < 0:
            raise AlignmentError(f"alignment item {position} field 'end' is negative: {end}")
        if end < start:
            raise AlignmentError(
                f"alignment item {position} ends before it starts: {start} > {end}"
            )
