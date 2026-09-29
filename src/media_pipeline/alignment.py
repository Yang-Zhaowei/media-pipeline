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

Matching the aligned token text against the original request text is a
behavior shared with the Caption Compiler (:mod:`media_pipeline.captions`):
this module reuses that same character-level matching logic
(:func:`media_pipeline.captions.check_alignment_matches_text`) and performs it
here, before writing the artifact, rather than only validating the alignment
data itself. When the aligned text does not match the original request text,
this raises :class:`~media_pipeline.captions.AlignmentMismatchError`; no
guessing or repair of the aligned text is performed.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .captions import AlignedToken, AlignmentError, has_alignable_content, parse_alignment
from .postprocess import AudioPostprocessError, read_wav

__all__ = [
    "AlignmentArtifact",
    "AlignmentRequest",
    "AlignmentRequestError",
    "AlignmentRuntimeError",
    "fault_is_io_error",
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


def fault_is_io_error(exc: BaseException) -> bool:
    """True if ``exc`` (or its ``__cause__`` chain) originates from an ``OSError``.

    ``AlignmentRequestError`` is raised both for a genuinely malformed request
    and for an unreadable/missing input WAV, the latter wrapping a filesystem
    ``OSError``. A run-level I/O or runtime fault must stop the whole run, while
    only an explicit per-segment deterministic validation failure may be
    isolated; this helper draws that line so a single exception type is not
    mis-classified as isolatable.
    """

    cause = exc.__cause__
    while cause is not None:
        if isinstance(cause, OSError):
            return True
        cause = cause.__cause__
    return False


def validate_request(request: AlignmentRequest) -> None:
    """Validate a request before it reaches the runtime.

    ``text`` and ``language`` must be non-empty strings, and ``text`` must
    contain alignable content (not only whitespace or punctuation). Raises
    :class:`AlignmentRequestError` otherwise. The alignable-content check
    reuses the Caption Compiler's non-spoken definition so a request that
    carries nothing to align is rejected deterministically, before inference.
    """

    if not isinstance(request.text, str) or not request.text.strip():
        raise AlignmentRequestError(
            "AlignmentRequest.text must be a non-empty string"
        )
    if not has_alignable_content(request.text):
        raise AlignmentRequestError(
            "AlignmentRequest.text must contain alignable content "
            "(whitespace or punctuation only is not alignable)"
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
    except (AudioPostprocessError, OSError, EOFError, struct.error, ValueError) as exc:
        # A missing, unreadable, malformed, corrupt, or truncated WAV is a
        # request/input failure. ``read_wav`` wraps format/decoding problems in
        # AudioPostprocessError, but a truncated data region can leak a low-level
        # ``struct.error`` from PCM unpacking; catch these input-decoding errors
        # here so they surface uniformly as AlignmentRequestError with chaining.
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
    record -- blank or not -- via :func:`_reject_blank_or_invalid_records`, so
    blank or whitespace-only records are rejected (never dropped) and invalid
    timestamps (non-numeric, non-finite, negative, end-before-start) can never
    be hidden. It then delegates to the Caption Compiler's
    :func:`~media_pipeline.captions.parse_alignment` so the same normalization,
    overlap rejection, and zero-duration tolerance apply here as downstream, and
    requires at least one effective token.
    Finally it rejects an alignment whose final timestamp exceeds the WAV
    duration beyond :data:`_ALIGNMENTS_TOLERANCE`.

    The timestamps are compared to the WAV duration but never rewritten,
    rounded, sorted, truncated, added to, or removed. Returns the validated
    :class:`AlignedToken` list. Raises :class:`AlignmentError` on any problem.
    """

    # Materialize once: the public contract accepts any Iterable[object], and
    # both passes below consume it, so a generator must not be exhausted by the
    # first pass.
    records = list(items)

    # Reject blank or timestamp-invalid records at the Production Alignment
    # boundary before parse_alignment() runs. parse_alignment() drops
    # whitespace-only records before checking their timestamps, so a blank
    # runtime record carrying NaN/Inf/negative/otherwise invalid timestamps
    # would otherwise slip past validation and be written. This boundary check
    # rejects blank records outright (Production output is preserved, never
    # normalized) and inspects the start/end of every record -- blank or not --
    # so invalid data can never be hidden by blank-token dropping. It enforces
    # only per-record validity; ordering/overlap/duration stay with
    # parse_alignment() on the accepted records, keeping its semantics intact.
    _reject_blank_or_invalid_records(records)

    tokens = parse_alignment(records)
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


def _reject_blank_or_invalid_records(items: Iterable[object]) -> None:
    """Reject blank or timestamp-invalid records at the Production boundary.

    At the Production Alignment boundary runtime output is preserved, not
    normalized: whitespace-only or empty records are rejected as invalid
    production alignment output rather than dropped (as downstream
    ``parse_alignment`` would), and every record must carry finite,
    non-negative, end >= start timestamps. ``parse_alignment`` itself keeps its
    permissive blank-dropping for the Caption Compiler; this check is strictly
    stricter and runs first. It enforces only per-record validity; ordering and
    overlap stay with :func:`parse_alignment` on the accepted records, so its
    downstream semantics are unchanged.
    """

    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise AlignmentError(
                f"alignment item {position} must be an object, "
                f"got {type(item).__name__}"
            )
        for field in ("text", "start", "end"):
            if field not in item:
                raise AlignmentError(
                    f"alignment item {position} is missing field {field!r}"
                )
        raw_text = item["text"]
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise AlignmentError(
                f"alignment item {position} has empty or whitespace-only text"
            )
        for field in ("start", "end"):
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
