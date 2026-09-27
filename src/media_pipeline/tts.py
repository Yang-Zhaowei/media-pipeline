"""Portable TTS contracts and deterministic PCM16 processing (v0).

This module holds the *portable* side of the Speech Pipeline's TTS stage:

- the request/output contracts,
- request validation,
- a flat-mono waveform-to-16-bit-PCM conversion,
- WAV I/O helpers (owned by :mod:`media_pipeline.postprocess`).

It is pure Python. It neither imports ``torch``, ``qwen_tts``, ``soundfile``,
nor any CUDA binding at import time or at use time. The model/runtime adapter
lives in :mod:`media_pipeline.runtimes.qwen_tts`, which is imported lazily.

Scope contract (v0):

- One :class:`CustomVoiceRequest` is one already-segmented utterance. This
  module does not split sentences; the caller passes a single utterance.
- The TTS engine returns a single flat mono waveform; the output WAV is mono,
  uncompressed, 16-bit PCM -- the exact format the Audio Postprocess stage
  consumes.
- No model, host, or output path is hard-coded here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

__all__ = [
    "CustomVoiceError",
    "CustomVoiceRequest",
    "TTSRuntimeError",
    "TTSArtifact",
    "validate_request",
    "waveform_to_mono_pcm16",
]


class CustomVoiceError(ValueError):
    """A request is malformed and can never be synthesized."""


class TTSRuntimeError(RuntimeError):
    """Invalid model/runtime input or output, or the runtime failed.

    Raised by the portable converters for malformed waveform output and by the
    runtime adapter for model/runtime failures.
    """


@dataclass(frozen=True)
class CustomVoiceRequest:
    """One already-segmented utterance to synthesize.

    ``text`` is the full utterance (no sentence splitting by this module).
    ``language`` and ``speaker`` select the voice; ``instruct`` is optional
    free-form direction that defaults to the empty string. The verified Chinese
    narration instruction is integration-test input, not a product default.
    """

    text: str
    language: str
    speaker: str
    instruct: str = ""


@dataclass(frozen=True)
class TTSArtifact:
    """A synthesized mono 16-bit PCM WAV, as consumed by Audio Postprocess.

    ``frames`` and ``sample_rate`` are read back from the written file so they
    are authoritative rather than taken from the model's own reporting.
    """

    wav_path: Path
    sample_rate: int
    frames: int

    @property
    def duration(self) -> float:
        """Total duration in seconds (derived from ``frames``)."""

        return self.frames / self.sample_rate


_SIGNED_MIN = -32_768
_SIGNED_MAX = 32_767


def validate_request(request: CustomVoiceRequest) -> None:
    """Validate a request before it reaches the runtime.

    ``text``, ``language``, and ``speaker`` must be non-empty. ``instruct`` may
    be empty. Raises :class:`CustomVoiceError` otherwise.
    """

    if not isinstance(request.text, str) or not request.text.strip():
        raise CustomVoiceError(
            "CustomVoiceRequest.text must be a non-empty string"
        )
    if not isinstance(request.language, str) or not request.language.strip():
        raise CustomVoiceError(
            "CustomVoiceRequest.language must be a non-empty string"
        )
    if not isinstance(request.speaker, str) or not request.speaker.strip():
        raise CustomVoiceError(
            "CustomVoiceRequest.speaker must be a non-empty string"
        )


def waveform_to_mono_pcm16(samples: Sequence[object]) -> list[int]:
    """Convert a flat, finite mono waveform to deterministic int16 PCM.

    v0 expects a single flat mono waveform (as returned by ``wavs[0]``). The
    contract enforced here is flat + finite + mono:

    - nested or channel-shaped input (any element that is itself a sequence) is
      rejected;
    - finite values are mapped to int16 with half-up rounding,
      ``q = floor(f * 32768 + 0.5)``, then clamped into the int16 range;
    - non-finite values (NaN, +/-Inf) raise :class:`TTSRuntimeError` rather than
      silently becoming silence.
    """

    flat = list(samples)
    if not flat:
        raise TTSRuntimeError("waveform has no samples")

    out: list[int] = []
    for value in flat:
        if hasattr(value, "__len__"):
            raise TTSRuntimeError(
                "expected a flat mono waveform, got nested/channel-shaped input"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TTSRuntimeError(f"waveform value is not a number: {value!r}")
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            raise TTSRuntimeError(f"waveform contains a non-finite value: {f!r}")
        q = int(math.floor(f * 32_768 + 0.5))
        if q > _SIGNED_MAX:
            q = _SIGNED_MAX
        elif q < _SIGNED_MIN:
            q = _SIGNED_MIN
        out.append(q)
    return out
