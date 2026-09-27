"""Unit tests for the pure audio postprocess stage (no models, no CUDA).

These tests build synthetic ``int16`` sample lists and tiny alignments in
memory; they never read the WAV fixture or run on the GPU.
"""

from __future__ import annotations

import json

import pytest

from media_pipeline.postprocess import (
    DEFAULT_FADE_IN,
    DEFAULT_FADE_OUT,
    DEFAULT_POST_PADDING,
    DEFAULT_PRE_PADDING,
    AudioPostprocessError,
    TrimPlan,
    compute_trim,
    sample_index,
    trim_and_fade,
    trim_alignment,
    trim_audio,
)

def _tokens(*entries: tuple[str, float, float]) -> list:
    from media_pipeline.captions import parse_alignment

    return parse_alignment(
        [{"text": text, "start": start, "end": end} for text, start, end in entries]
    )


def _plan(
    frame_rate: int = 8,
    frames: int = 64,
    *,
    start: float = 1.0,
    end: float = 5.0,
    **kwargs,
) -> TrimPlan:
    return compute_trim(_tokens(("a", start, end)), frame_rate=frame_rate, frames=frames, **kwargs)


# --- frame quantization ----------------------------------------------------


def test_sample_index_rounds_half_up() -> None:
    assert sample_index(0.0, 8) == 0
    assert sample_index(1.0, 8) == 8
    # 0.25 s * 8 = 2.0 frames
    assert sample_index(0.25, 8) == 2
    # half frame rounds up
    assert sample_index(0.3125, 8) == 3


def test_sample_index_rejects_negative() -> None:
    with pytest.raises(ValueError):
        sample_index(-0.1, 8)


def test_compute_trim_quantizes_front_with_floor() -> None:
    # start=1.0, pre_padding=0.25, frame_rate=8 -> requested 0.75 s -> 6 frames
    plan = compute_trim(_tokens(("a", 1.0, 5.0)), frame_rate=8, frames=64, pre_padding=0.25)
    assert plan.start_frame == 6
    assert plan.start == pytest.approx(6 / 8)


def test_compute_trim_quantizes_back_with_ceil() -> None:
    # end=5.0, post_padding=0.25 -> requested 5.25 s -> 42 frames
    plan = compute_trim(_tokens(("a", 1.0, 5.0)), frame_rate=8, frames=64, post_padding=0.25)
    assert plan.end_frame == 42
    assert plan.end == pytest.approx(42 / 8)


# --- defaults on the fixture numbers ---------------------------------------


def test_defaults_keep_150ms_padding() -> None:
    # Numbers mirror the real smoke fixture: first speech 1.36 s, end 7.92 s.
    plan = compute_trim(
        _tokens(("a", 1.36, 7.92)), frame_rate=24000, frames=192000
    )
    assert plan.start_frame == 29040  # floor((1.36 - 0.15) * 24000)
    assert plan.end_frame == 192000  # clamped at the 8.00 s end
    assert plan.trim_before == pytest.approx(1.21)
    assert plan.trim_after == pytest.approx(0.0)


def test_alignment_far_past_duration_is_rejected() -> None:
    # A token whose *own* end exceeds the WAV duration is rejected; padding can
    # only extend toward the (already bounded) duration, never past it.
    with pytest.raises(AudioPostprocessError):
        compute_trim(_tokens(("a", 1.0, 100.0)), frame_rate=8, frames=64)


# --- shifting and duration preservation ------------------------------------


def test_trim_alignment_shifts_by_quantized_offset() -> None:
    plan = compute_trim(_tokens(("a", 1.36, 7.92)), frame_rate=24000, frames=192000)
    offset = plan.start_frame / plan.frame_rate
    shifted = trim_alignment(
        _tokens(("真", 1.36, 1.68), ("往", 5.2, 5.36)), offset
    )
    assert shifted[0].start == pytest.approx(0.15)  # first speech begins ~0.15 s in
    assert shifted[0].end == pytest.approx(0.47)  # 1.68 - 1.21, duration 0.32 preserved
    assert shifted[1].start == pytest.approx(5.2 - offset)


def test_trim_alignment_preserves_durations() -> None:
    tokens = _tokens(
        ("a", 1.36, 1.68),
        ("b", 1.68, 1.84),
        ("c", 5.2, 7.92),
    )
    plan = compute_trim(tokens, frame_rate=24000, frames=192000)
    shifted = trim_alignment(tokens, plan.start_frame / plan.frame_rate)
    for old, new in zip(tokens, shifted):
        assert (new.end - new.start) == pytest.approx(old.end - old.start)


def test_trim_alignment_preserves_alignment_gap() -> None:
    tokens = _tokens(
        ("a", 1.0, 4.88),
        ("b", 5.2, 7.92),
    )
    plan = compute_trim(tokens, frame_rate=24000, frames=192000)
    shifted = trim_alignment(tokens, plan.start_frame / plan.frame_rate)
    gap = shifted[1].start - shifted[0].end
    assert gap == pytest.approx(5.2 - 4.88)


def test_trim_alignment_normalizes_negative_zero() -> None:
    tokens = _tokens(("a", 1.0, 2.0))
    shifted = trim_alignment(tokens, 1.0)
    assert shifted[0].start == 0.0
    assert not str(shifted[0].start).startswith("-")


# --- no-op trimming --------------------------------------------------------


def test_no_op_trim_when_padding_exceeds_edge_audio() -> None:
    plan = compute_trim(
        _tokens(("a", 1.0, 5.0)),
        frame_rate=8,
        frames=64,
        pre_padding=2.0,
        post_padding=4.0,  # 5.0 + 4.0 clamps to the 8.00 s end
    )
    assert plan.start_frame == 0
    assert plan.end_frame == 64
    assert plan.trim_before == 0.0
    assert plan.trim_after == 0.0


def test_no_op_shift_leaves_alignment_unchanged() -> None:
    tokens = _tokens(("a", 1.0, 2.0), ("b", 3.0, 4.0))
    shifted = trim_alignment(tokens, 0.0)
    assert shifted == tokens


# --- padding clamping ------------------------------------------------------


def test_pre_padding_clamped_to_start_of_audio() -> None:
    plan = compute_trim(
        _tokens(("a", 1.0, 5.0)),
        frame_rate=8,
        frames=64,
        pre_padding=10.0,
    )
    assert plan.start_frame == 0
    assert plan.trim_before == 0.0


def test_post_padding_clamped_to_end_of_audio() -> None:
    plan = compute_trim(
        _tokens(("a", 1.0, 5.0)),
        frame_rate=8,
        frames=64,
        post_padding=10.0,
    )
    assert plan.end_frame == 64
    assert plan.trim_after == 0.0


# --- fade in / out, timing untouched ---------------------------------------


def test_zero_fade_leaves_samples_untouched() -> None:
    samples = list(range(1, 101))
    out = trim_and_fade(
        samples,
        8,
        start_frame=10,
        end_frame=90,
        fade_in=0.0,
        fade_out=0.0,
    )
    assert out == samples[10:90]


def test_fade_keeps_frame_count_and_positions() -> None:
    samples = [1000] * 200
    out = trim_and_fade(
        samples,
        24000,
        start_frame=0,
        end_frame=200,
        fade_in=0.001,
        fade_out=0.001,
    )
    assert len(out) == 200
    # Fades are ~24 frames each; middle samples stay exactly at full amplitude.
    assert out[60] == 1000
    assert out[139] == 1000


def test_fade_in_attenuates_leading_samples() -> None:
    samples = [1000] * 200
    out = trim_and_fade(
        samples,
        24000,
        start_frame=0,
        end_frame=200,
        fade_in=0.01,
        fade_out=0.0,
    )
    # First sample is ramped up from near zero; last sample untouched.
    assert 0 < out[0] < 1000
    assert out[-1] == 1000


def test_fade_out_attenuates_trailing_samples() -> None:
    samples = [1000] * 200
    out = trim_and_fade(
        samples,
        24000,
        start_frame=0,
        end_frame=200,
        fade_in=0.0,
        fade_out=0.01,
    )
    assert out[0] == 1000
    assert 0 < out[-1] < 1000


def test_fades_clamped_to_half_of_region() -> None:
    # Region of 10 frames; a 1 s fade at 24000 Hz would be far larger, so it
    # is clamped to half the region on each side.
    samples = [1000] * 10
    out = trim_and_fade(
        samples,
        24000,
        start_frame=0,
        end_frame=10,
        fade_in=1.0,
        fade_out=1.0,
    )
    assert len(out) == 10
    # Every sample is attenuated to some value <= full amplitude.
    assert all(0 <= value <= 1000 for value in out)


# --- invalid inputs --------------------------------------------------------


def test_compute_trim_rejects_empty_alignment() -> None:
    with pytest.raises(AudioPostprocessError):
        compute_trim([], frame_rate=8, frames=64)


def test_compute_trim_rejects_negative_padding() -> None:
    with pytest.raises(ValueError):
        compute_trim(_tokens(("a", 1.0, 5.0)), frame_rate=8, frames=64, pre_padding=-0.1)


def test_trim_and_fade_rejects_negative_fade() -> None:
    with pytest.raises(ValueError):
        trim_and_fade([0] * 10, 8, start_frame=0, end_frame=10, fade_in=-0.1)


def test_trim_alignment_rejects_negative_offset() -> None:
    with pytest.raises(ValueError):
        trim_alignment(_tokens(("a", 1.0, 2.0)), -0.1)


def test_alignment_past_duration_is_rejected() -> None:
    with pytest.raises(AudioPostprocessError):
        compute_trim(_tokens(("a", 1.0, 8.001)), frame_rate=8, frames=64)


def test_alignment_at_duration_is_allowed() -> None:
    plan = compute_trim(_tokens(("a", 1.0, 8.0)), frame_rate=8, frames=64)
    assert plan.end_frame == 64


# --- trim_audio helper -----------------------------------------------------


def test_trim_audio_returns_faded_samples_and_plan() -> None:
    frame_rate = 24000
    samples = [0] * (frame_rate * 8)  # 8 s of silence
    plan = compute_trim(_tokens(("a", 1.36, 7.92)), frame_rate=frame_rate, frames=len(samples))
    faded, trimmed = trim_audio(
        samples, frame_rate, _tokens(("a", 1.36, 7.92))
    )
    assert trimmed.start_frame == plan.start_frame
    assert len(faded) == plan.kept_frames
    assert trimmed.start_frame == 29040


# --- defaults sanity -------------------------------------------------------


def test_default_padding_and_fade_values() -> None:
    assert DEFAULT_PRE_PADDING == 0.15
    assert DEFAULT_POST_PADDING == 0.15
    assert DEFAULT_FADE_IN == 0.02
    assert DEFAULT_FADE_OUT == 0.02
