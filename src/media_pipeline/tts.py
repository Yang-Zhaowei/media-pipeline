"""Portable TTS contracts and deterministic PCM16 processing (v0).

This module holds the *portable* side of the Speech Pipeline's TTS stage:

- the request/output contracts,
- request validation,
- a deterministic waveform-to-mono-16-bit-PCM conversion,
- WAV I/O through the existing :mod:`media_pipeline.postprocess` helpers.

It is pure Python. It neither imports ``torch``, ``qwen_tts``, ``soundfile``,
nor any CUDA binding at import time or at use time. The model/runtime adapter
lives in :mod:`media_pipeline.runtimes.qwen_tts`, which is imported lazily.

Scope contract (v0):

- One :class:`CustomVoiceRequest` is one already-segmented utterance. This
  module does not split sentences; the caller passes a single utterance.
- The output WAV is mono, uncompressed, 16-bit PCM -- the exact format the
  Audio Postprocess stage consumes.
- No model, host, or output path is hard-coded here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .postprocess import read_wav, write_wav

__all__ = [
    "DEFAULT_INSTRUCT",
    "CustomVoiceError",
    "CustomVoiceRequest",
    "TTSRuntimeError",
    "TTSArtifact",
    "flatten_to_mono",
    "validate_request",
    "waveform_to_mono_pcm16",
]

#: The only product default for the free-form instruction. The verified Chinese
#: narration instruction is integration-test input, not a default here.
DEFAULT_INSTRUCT = ""


class CustomVoiceError(ValueError):
    """A request is malformed and can never be synthesized."""


class TTSRuntimeError(RuntimeError):
    """The model/runtime failed to load or to synthesize a request.

    Raised by the runtime adapter only; the portable side never raises this.
    """


@dataclass(frozen=True)
class CustomVoiceRequest:
    """One already-segmented utterance to synthesize.

    ``text`` is the full utterance (no sentence splitting by this module).
    ``language`` and ``speaker`` select the voice; ``instruct`` is optional
    free-form direction that defaults to :data:`DEFAULT_INSTRUCT` (``""``).
    """

    text: str
    language: str
    speaker: str
    instruct: str = DEFAULT_INSTRUCT


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


def _is_scalar(value: object) -> bool:
    """True for a single sample (scalar), False for a channel (sequence).

    Uses ``__len__`` rather than type checks so it works for plain Python
    sequences and for NumPy/torch array-likes without depending on either.
    """

    return not hasattr(value, "__len__")


def flatten_to_mono(channel_samples: Sequence[object]) -> list[float]:
    """Mix any returned waveform to a single mono float channel.

    - A flat sequence of scalars is already mono and is returned as floats.
    - A sequence of channels (e.g. a 2-D array) is mixed to mono by averaging
      the channels sample-by-sample, truncated to the shortest channel.

    This never imports NumPy or torch; it only relies on iteration and
    indexing.
    """

    try:
        samples = list(channel_samples)
    except TypeError:  # not iterable (a lone scalar)
        return [float(channel_samples)]

    if not samples:
        return []

    if _is_scalar(samples[0]):
        return [float(value) for value in samples]

    n_channels = len(samples)
    length = min(len(channel) for channel in samples)
    mixed: list[float] = []
    for index in range(length):
        total = 0.0
        for channel in samples:
            total += float(channel[index])
        mixed.append(total / n_channels)
    return mixed


def waveform_to_mono_pcm16(channel_samples: Sequence[object]) -> list[int]:
    """Convert a returned waveform to deterministic mono 16-bit PCM samples.

    Steps, all pure and deterministic:

    1. Mix to mono with :func:`flatten_to_mono`.
    2. Map each float sample (assumed normalized to ``[-1.0, 1.0]``) to int16
       with half-up rounding: ``q = floor(f * 32768 + 0.5)``.
    3. Clamp non-finite floats (``nan``/``inf``) to ``0`` (silence) before
       clamping into ``[_SIGNED_MIN, _SIGNED_MAX]``.

    Clipping is explicit and bounded; no sample ever exceeds int16 range.
    """

    mono = flatten_to_mono(channel_samples)
    out: list[int] = []
    for value in mono:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            value = float(value)
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            f = 0.0
        q = int(math.floor(f * 32_768 + 0.5))
        if q > _SIGNED_MAX:
            q = _SIGNED_MAX
        elif q < _SIGNED_MIN:
            q = _SIGNED_MIN
        out.append(q)
    return out


def validate_request(request: CustomVoiceRequest) -> None:
    """Validate a request before it reaches the runtime.

    ``text``, ``language``, and ``speaker`` must be non-empty. ``instruct`` may
    be empty (its default). Raises :class:`CustomVoiceError` otherwise.
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


def _read_artifact(wav_path: str | Path) -> TTSArtifact:
    """Read back a freshly written WAV into a :class:`TTSArtifact`."""

    samples, frame_rate, frames = read_wav(wav_path)
    return TTSArtifact(
        wav_path=Path(wav_path),
        sample_rate=frame_rate,
        frames=frames,
    )
