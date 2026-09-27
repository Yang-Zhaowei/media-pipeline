"""Audio postprocess for the Speech Pipeline (v0).

This stage sits between forced alignment and the caption compiler:

    raw.wav + alignment.raw.json
        -> trim meaningless leading/trailing audio (bounded by the alignment)
        -> keep a configurable pre/post padding
        -> apply short fades at the two new edges
        -> shift the alignment timestamps to the trimmed timeline
        -> cleaned WAV + adjusted alignment

v0 scope is deliberately narrow. There is **no** VAD, denoise, EQ, compression,
normalization, TTS, or any API/MCP/NLE integration. The "meaningless" audio we
cut is simply everything outside the aligned speech region: the alignment itself
defines where speech begins and ends.

Timing source of truth is the integer frame grid of the WAV, not a float second
value. We therefore:

- take the front boundary at ``floor(requested_start * frame_rate)``;
- take the back boundary at ``ceil(requested_end * frame_rate)``;
- shift every alignment timestamp by ``actual_start_frame / frame_rate`` (the
  quantized offset), which keeps all alignment durations unchanged;
- keep alignment precision intact on write (no millisecond rounding -- that
  belongs to SRT rendering).

Only uncompressed mono 16-bit PCM WAV is supported.

This module is pure Python and runs without CUDA.
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .captions import AlignedToken, parse_alignment

__all__ = [
    "DEFAULT_FADE_IN",
    "DEFAULT_FADE_OUT",
    "DEFAULT_POST_PADDING",
    "DEFAULT_PRE_PADDING",
    "AudioPostprocessError",
    "TrimPlan",
    "compute_trim",
    "postprocess_speech",
    "read_wav",
    "sample_index",
    "trim_and_fade",
    "trim_alignment",
    "trim_audio",
    "write_wav",
]

#: Default seconds of leading silence kept before the first aligned token.
DEFAULT_PRE_PADDING = 0.15

#: Default seconds of trailing silence kept after the last aligned token.
DEFAULT_POST_PADDING = 0.15

#: Default linear fade-in length applied at the new leading edge (seconds).
DEFAULT_FADE_IN = 0.02

#: Default linear fade-out length applied at the new trailing edge (seconds).
DEFAULT_FADE_OUT = 0.02

#: Alignment whose final timestamp exceeds the WAV duration beyond this many
#: seconds is rejected. A purely numerical tolerance for coarse JSON values.
_ALIGNMENTS_TOLERANCE = 1e-6

_SIGNED_MIN = -32_768
_SIGNED_MAX = 32_767


class AudioPostprocessError(ValueError):
    """The audio data, its format, or its pairing with the alignment is unusable."""


@dataclass(frozen=True)
class TrimPlan:
    """The actual integer-frame window kept from the original WAV.

    The frame boundaries are the authoritative timing; the second-valued
    properties are derived from them and should only be used for reporting.
    """

    frame_rate: int
    frames: int
    start_frame: int
    end_frame: int

    @property
    def duration(self) -> float:
        """Total original duration in seconds."""

        return self.frames / self.frame_rate

    @property
    def start(self) -> float:
        """Kept start in seconds (derived from ``start_frame``)."""

        return self.start_frame / self.frame_rate

    @property
    def end(self) -> float:
        """Kept end in seconds (derived from ``end_frame``)."""

        return self.end_frame / self.frame_rate

    @property
    def trim_before(self) -> float:
        """Seconds removed before the first aligned token."""

        return self.start_frame / self.frame_rate

    @property
    def trim_after(self) -> float:
        """Seconds removed after the last aligned token."""

        return (self.frames - self.end_frame) / self.frame_rate

    @property
    def kept_frames(self) -> int:
        """Number of frames retained after trimming."""

        return self.end_frame - self.start_frame


def compute_trim(
    tokens: Sequence[AlignedToken],
    *,
    frame_rate: int,
    frames: int,
    pre_padding: float = DEFAULT_PRE_PADDING,
    post_padding: float = DEFAULT_POST_PADDING,
) -> TrimPlan:
    """Compute the actual frame window kept from the original WAV.

    ``tokens`` is the validated, sorted alignment. The front boundary keeps the
    first token plus ``pre_padding`` seconds; the back boundary keeps the last
    token plus ``post_padding`` seconds. Both boundaries are clamped to the WAV
    duration and then quantized to the integer frame grid:

    - ``start_frame = floor(max(0, first.start - pre_padding) * frame_rate)``
    - ``end_frame = ceil(min(duration, last.end + post_padding) * frame_rate)``

    The alignment is rejected up front if its final timestamp exceeds the WAV
    duration beyond :data:`_ALIGNMENTS_TOLERANCE`, so a stale alignment can never
    silently keep audio past the end of the file.
    """

    if frame_rate <= 0:
        raise ValueError(f"frame_rate must be positive, got {frame_rate}")
    if frames <= 0:
        raise ValueError(f"frames must be positive, got {frames}")
    if not tokens:
        raise AudioPostprocessError("cannot trim: alignment has no tokens")
    if pre_padding < 0:
        raise ValueError(f"pre_padding must not be negative, got {pre_padding}")
    if post_padding < 0:
        raise ValueError(f"post_padding must not be negative, got {post_padding}")

    duration = frames / frame_rate

    last_end = tokens[-1].end
    if last_end > duration + _ALIGNMENTS_TOLERANCE:
        raise AudioPostprocessError(
            f"alignment ends at {last_end}, past the WAV duration {duration}"
        )

    requested_start = max(0.0, tokens[0].start - pre_padding)
    requested_end = min(duration, last_end + post_padding)

    start_frame = math.floor(requested_start * frame_rate)
    end_frame = math.ceil(requested_end * frame_rate)

    start_frame = max(0, min(start_frame, frames))
    end_frame = max(0, min(end_frame, frames))
    if end_frame < start_frame:
        end_frame = start_frame

    return TrimPlan(
        frame_rate=frame_rate,
        frames=frames,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def trim_alignment(
    tokens: Sequence[AlignedToken],
    offset: float,
) -> list[AlignedToken]:
    """Shift every alignment timestamp by ``offset`` seconds.

    ``offset`` must be the *quantized* front trim offset (``start_frame /
    frame_rate``). Durations are preserved by construction; only the absolute
    positions move. Insignificant ``-0.0`` is normalized to ``0.0``.
    """

    if offset < 0:
        raise ValueError(f"trim offset must not be negative, got {offset}")

    shifted: list[AlignedToken] = []
    for token in tokens:
        start = token.start - offset
        end = token.end - offset
        start = 0.0 if start == 0 else start
        end = 0.0 if end == 0 else end
        shifted.append(AlignedToken(text=token.text, start=start, end=end))
    return shifted


def sample_index(seconds: float, frame_rate: int) -> int:
    """Convert a non-negative second value to an integer frame index.

    Rounds half up so that ``0.5`` frames maps to ``1``, keeping the boundary
    decisions deterministic across platforms.
    """

    if seconds < 0:
        raise ValueError(f"seconds must not be negative, got {seconds}")
    return int(math.floor(seconds * frame_rate + 0.5))


def trim_and_fade(
    samples: Sequence[int],
    frame_rate: int,
    *,
    start_frame: int,
    end_frame: int,
    fade_in: float = DEFAULT_FADE_IN,
    fade_out: float = DEFAULT_FADE_OUT,
) -> list[int]:
    """Trim ``samples`` to the frame window and apply short fades.

    The trim slices ``samples[start_frame:end_frame]``. Fades then ramp the
    amplitude of the first ``fade_in`` seconds and the last ``fade_out`` seconds
    of that window up from (near) zero. Fades never exceed half the kept region
    per side, and they never change the sample count or positions -- only the
    amplitudes at the two new edges -- so timing is untouched.
    """

    if frame_rate <= 0:
        raise ValueError(f"frame_rate must be positive, got {frame_rate}")
    if fade_in < 0 or fade_out < 0:
        raise ValueError("fade_in and fade_out must not be negative")

    start_frame = max(0, min(start_frame, len(samples)))
    end_frame = max(start_frame, min(end_frame, len(samples)))
    region = samples[start_frame:end_frame]
    region_frames = len(region)

    def fade_length(seconds: float) -> int:
        if seconds <= 0:
            return 0
        return max(0, min(round(seconds * frame_rate), region_frames // 2))

    fade_in_len = fade_length(fade_in)
    fade_out_len = fade_length(fade_out)

    output: list[int] = []
    for index, sample in enumerate(region):
        factor = 1.0
        if index < fade_in_len:
            factor *= (index + 1) / (fade_in_len + 1)
        tail_index = region_frames - 1 - index
        if tail_index < fade_out_len:
            factor *= (tail_index + 1) / (fade_out_len + 1)
        value = int(round(sample * factor))
        if value < _SIGNED_MIN:
            value = _SIGNED_MIN
        elif value > _SIGNED_MAX:
            value = _SIGNED_MAX
        output.append(value)
    return output


def read_wav(path: str | Path) -> tuple[list[int], int, int]:
    """Read an uncompressed mono 16-bit PCM WAV.

    Returns ``(samples, frame_rate, frames)``. Raises
    :class:`AudioPostprocessError` for any other channel count, sample width, or
    compression code.
    """

    # The stdlib ``wave`` parser handles RIFF/chunk ordering and already fails
    # on formats it cannot decode. We keep the explicit mono + 16-bit restriction
    # for v0; any other channel count, sample width, or compression is rejected.
    try:
        with wave.open(str(path), "rb") as handle:
            nchannels = handle.getnchannels()
            sampwidth = handle.getsampwidth()
            framerate = handle.getframerate()
            nframes = handle.getnframes()
            raw = handle.readframes(nframes)
    except (OSError, wave.Error) as exc:
        raise AudioPostprocessError(f"cannot read WAV {path}: {exc}") from exc

    if nchannels != 1 or sampwidth != 2:
        raise AudioPostprocessError(
            "only uncompressed mono 16-bit PCM WAV is supported; "
            f"got {nchannels} channel(s), {sampwidth * 8}-bit"
        )

    samples = list(struct.unpack(f"<{nframes}h", raw))
    return samples, framerate, nframes


def write_wav(path: str | Path, samples: Sequence[int], frame_rate: int) -> None:
    """Write ``samples`` as an uncompressed mono 16-bit PCM WAV."""

    raw = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(frame_rate)
        handle.writeframes(raw)


def postprocess_speech(
    raw_wav_path: str | Path,
    alignment_path: str | Path,
    output_wav_path: str | Path,
    output_alignment_path: str | Path,
    *,
    pre_padding: float = DEFAULT_PRE_PADDING,
    post_padding: float = DEFAULT_POST_PADDING,
    fade_in: float = DEFAULT_FADE_IN,
    fade_out: float = DEFAULT_FADE_OUT,
    encoding: str = "utf-8",
) -> TrimPlan:
    """Run the full v0 postprocess end to end.

    Reads ``raw_wav_path`` and ``alignment_path``, computes the trim window from
    the current WAV and alignment, writes the cleaned WAV to ``output_wav_path``
    and the adjusted alignment (same ``{"text", "start", "end"}`` shape,
    timestamps shifted and their precision preserved) to ``output_alignment_path``.
    Returns the :class:`TrimPlan` computed.
    """

    import json

    samples, frame_rate, frames = read_wav(raw_wav_path)

    try:
        with open(alignment_path, encoding=encoding) as handle:
            raw_alignment = json.load(handle)
    except json.JSONDecodeError as exc:
        raise AudioPostprocessError(
            f"{alignment_path} is not valid JSON: {exc}"
        ) from exc
    except OSError as exc:
        raise AudioPostprocessError(f"cannot read alignment {alignment_path}: {exc}") from exc

    try:
        tokens = parse_alignment(raw_alignment)
    except Exception as exc:  # captions raises AlignmentError; treat as unusable
        raise AudioPostprocessError(str(exc)) from exc

    # Always compute the plan from this WAV and alignment: a plan built for a
    # different frame rate or length would desync the audio and alignment timelines.
    trim = compute_trim(
        tokens,
        frame_rate=frame_rate,
        frames=frames,
        pre_padding=pre_padding,
        post_padding=post_padding,
    )

    offset = trim.start_frame / trim.frame_rate
    shifted = trim_alignment(tokens, offset)
    faded = trim_and_fade(
        samples,
        frame_rate,
        start_frame=trim.start_frame,
        end_frame=trim.end_frame,
        fade_in=fade_in,
        fade_out=fade_out,
    )

    write_wav(output_wav_path, faded, frame_rate)

    adjusted = [
        {"text": token.text, "start": token.start, "end": token.end}
        for token in shifted
    ]
    with open(output_alignment_path, "w", encoding=encoding, newline="\n") as handle:
        json.dump(adjusted, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return trim


def trim_audio(
    samples: Sequence[int],
    frame_rate: int,
    tokens: Sequence[AlignedToken],
    *,
    pre_padding: float = DEFAULT_PRE_PADDING,
    post_padding: float = DEFAULT_POST_PADDING,
    fade_in: float = DEFAULT_FADE_IN,
    fade_out: float = DEFAULT_FADE_OUT,
) -> tuple[list[int], TrimPlan]:
    """Trim and fade already-loaded samples given a validated alignment.

    Convenience wrapper for callers that already hold the samples (for example a
    unit test): computes the :class:`TrimPlan`, then trims and fades in one step.
    Returns ``(faded_samples, trim_plan)``.
    """

    if frame_rate <= 0:
        raise ValueError(f"frame_rate must be positive, got {frame_rate}")
    if not tokens:
        raise AudioPostprocessError("cannot trim: alignment has no tokens")
    frames = len(samples)
    trim = compute_trim(
        tokens,
        frame_rate=frame_rate,
        frames=frames,
        pre_padding=pre_padding,
        post_padding=post_padding,
    )
    faded = trim_and_fade(
        samples,
        frame_rate,
        start_frame=trim.start_frame,
        end_frame=trim.end_frame,
        fade_in=fade_in,
        fade_out=fade_out,
    )
    return faded, trim
