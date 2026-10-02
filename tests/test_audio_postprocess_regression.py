"""Regression tests for Audio Postprocess against the real smoke fixture.

``tests/fixtures/speech-smoke-001`` is immutable GPU evidence. These tests read
it only: they assert the actual frame counts and the actual quantized trim
offset produced by the postprocess stage, plus that the adjusted alignment keeps
its full precision.

No CUDA is involved: WAV I/O and trimming run on CPU.
"""

from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import pytest

from media_pipeline.postprocess import (
    DEFAULT_FADE_IN,
    DEFAULT_POST_PADDING,
    DEFAULT_PRE_PADDING,
    AudioPostprocessError,
    compute_trim,
    postprocess_speech,
    read_wav,
    trim_and_fade,
    trim_alignment,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "speech-smoke-001"
RAW_WAV = FIXTURE / "raw.wav"
ALIGNMENT = FIXTURE / "alignment.raw.json"


def _frame_rate() -> int:
    with wave.open(str(RAW_WAV), "rb") as wav:
        return wav.getframerate()


def _read_alignment() -> list[dict]:
    return json.loads(ALIGNMENT.read_text(encoding="utf-8"))


def test_fixture_wav_is_mono_16bit_pcm_8s() -> None:
    samples, frame_rate, frames = read_wav(RAW_WAV)
    assert frame_rate == 24000
    assert frames == 192000  # exactly 8.000 s
    assert len(samples) == 192000


def test_quantized_trim_offset_is_integer_frame() -> None:
    alignment = _read_alignment()
    first_start = alignment[0]["start"]
    frame_rate = _frame_rate()
    plan = compute_trim(
        _alignment_tokens(alignment), frame_rate=frame_rate, frames=192000
    )
    # The kept front boundary is an exact sample index, not a float second.
    assert plan.start_frame == int((first_start - DEFAULT_PRE_PADDING) * frame_rate)
    assert isinstance(plan.start_frame, int)


def test_actual_interval_is_1_21s_to_8_00s() -> None:
    frame_rate = _frame_rate()
    plan = compute_trim(
        _alignment_tokens(_read_alignment()),
        frame_rate=frame_rate,
        frames=192000,
    )
    assert plan.start == pytest.approx(1.21)
    assert plan.end == pytest.approx(8.0)
    assert plan.start_frame == 29040
    assert plan.end_frame == 192000


def test_first_alignment_begins_around_0_15s_in_output() -> None:
    alignment = _read_alignment()
    frame_rate = _frame_rate()
    plan = compute_trim(
        _alignment_tokens(alignment), frame_rate=frame_rate, frames=192000
    )
    shifted = trim_alignment(_alignment_tokens(alignment), plan.start_frame / frame_rate)
    assert shifted[0].start == pytest.approx(0.15, abs=1e-6)


def test_trimmed_wav_has_actual_frame_count() -> None:
    frame_rate = _frame_rate()
    samples, _, _ = read_wav(RAW_WAV)
    faded = trim_and_fade(
        samples,
        frame_rate,
        start_frame=29040,
        end_frame=192000,
        fade_in=0.0,
        fade_out=0.0,
    )
    # 192000 - 29040 frames kept, no fade => exact, timing untouched.
    assert len(faded) == 162960


def test_full_pipeline_writes_expected_frames_and_alignment(
    tmp_path: Path,
) -> None:
    frame_rate = _frame_rate()
    out_wav = tmp_path / "cleaned.wav"
    out_alignment = tmp_path / "adjusted.json"

    plan = postprocess_speech(
        RAW_WAV,
        ALIGNMENT,
        out_wav,
        out_alignment,
        fade_in=0.0,
        fade_out=0.0,
    )

    assert plan.start_frame == 29040
    assert plan.end_frame == 192000

    out_samples, out_frame_rate, out_frames = read_wav(out_wav)
    assert out_frame_rate == frame_rate == 24000
    assert out_frames == 162960

    adjusted = json.loads(out_alignment.read_text(encoding="utf-8"))
    # Durations preserved; first speech shifted to ~0.15 s.
    assert adjusted[0]["end"] - adjusted[0]["start"] == pytest.approx(0.32)
    assert adjusted[30]["end"] - adjusted[30]["start"] == pytest.approx(0.24)
    assert adjusted[0]["start"] == pytest.approx(0.15, abs=1e-6)


def test_adjusted_alignment_keeps_full_precision_not_ms_rounded(
    tmp_path: Path,
) -> None:
    out_alignment = tmp_path / "adjusted.json"
    postprocess_speech(RAW_WAV, ALIGNMENT, tmp_path / "c.wav", out_alignment)

    raw_text = out_alignment.read_text(encoding="utf-8")
    adjusted = json.loads(raw_text)
    # The stored floats equal the exact shifted values, not a 2-decimal (10 ms)
    # rounding -- precision is preserved for the caption stage downstream.
    frame_rate = _frame_rate()
    plan = compute_trim(
        _alignment_tokens(_read_alignment()), frame_rate=frame_rate, frames=192000
    )
    shifted = trim_alignment(
        _alignment_tokens(_read_alignment()), plan.start_frame / frame_rate
    )
    for stored, token in zip(adjusted, shifted):
        # Stored values equal the unrounded shifted floats, not ms rounding.
        assert stored["start"] == pytest.approx(token.start, abs=1e-12)
        assert stored["end"] == pytest.approx(token.end, abs=1e-12)
    # At least one stored timestamp carries sub-10 ms precision (proves it was
    # not rounded to milliseconds on write).
    assert any(
        round(value, 2) != value
        for item in adjusted
        for value in (item["start"], item["end"])
    )


def test_no_op_trim_preserves_full_frame_count(tmp_path: Path) -> None:
    frame_rate = _frame_rate()
    out_wav = tmp_path / "noop.wav"
    out_alignment = tmp_path / "noop.json"
    postprocess_speech(
        RAW_WAV,
        ALIGNMENT,
        out_wav,
        out_alignment,
        pre_padding=5.0,
        post_padding=5.0,
        fade_in=0.0,
        fade_out=0.0,
    )
    _, _, frames = read_wav(out_wav)
    assert frames == 192000


# --- helpers ---------------------------------------------------------------


def _alignment_tokens(raw: list[dict]):
    from media_pipeline.captions import parse_alignment

    return parse_alignment(raw)


def alignment_end_start(index: int) -> float:
    raw = _read_alignment()
    return raw[index + 1]["start"] - raw[index]["end"]


# --- WAV format validation (crafted files, not the immutable fixture) -------


def _write_pcm_wav(
    path: Path,
    *,
    n_channels: int = 1,
    bits: int = 16,
    sample_rate: int = 8000,
    frames: int = 8,
    junk_before_fmt: int | None = None,
) -> None:
    """Build a raw RIFF/WAVE file so we control chunk ordering exactly."""

    bytes_per_sample = bits // 8
    fmt = struct.pack(
        "<HHIIHH",
        1,  # WAVE_FORMAT_PCM
        n_channels,
        sample_rate,
        sample_rate * n_channels * bytes_per_sample,  # byte rate
        n_channels * bytes_per_sample,  # block align
        bits,
    )
    data = struct.pack("<%dh" % frames, *([1000] * frames))
    body = b"WAVE"
    if junk_before_fmt:
        # 4096 bytes pushes 'fmt ' past the end of a 4096-byte header scan.
        body += b"JUNK" + struct.pack("<I", junk_before_fmt) + b"\x00" * junk_before_fmt
    body += b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def test_read_wav_accepts_pcm_with_large_junk_chunk_before_fmt(tmp_path: Path) -> None:
    # A 4096-byte JUNK chunk moves 'fmt ' to byte 4116, beyond a fixed 4096-byte
    # header scan. The old fixed-size scan could not find it and rejected this
    # valid PCM file; the stdlib wave parser handles the chunk ordering.
    wav = tmp_path / "junk.wav"
    _write_pcm_wav(wav, junk_before_fmt=4096)
    samples, frame_rate, frames = read_wav(wav)
    assert frame_rate == 8000
    assert frames == 8
    assert len(samples) == 8


def test_read_wav_rejects_stereo(tmp_path: Path) -> None:
    wav = tmp_path / "stereo.wav"
    _write_pcm_wav(wav, n_channels=2)
    with pytest.raises(AudioPostprocessError):
        read_wav(wav)


def test_read_wav_rejects_8bit(tmp_path: Path) -> None:
    wav = tmp_path / "8bit.wav"
    _write_pcm_wav(wav, bits=8)
    with pytest.raises(AudioPostprocessError):
        read_wav(wav)
