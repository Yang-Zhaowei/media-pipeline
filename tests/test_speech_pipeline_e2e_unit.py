"""CPU unit test for the Speech Pipeline v0 end-to-end validation harness.

These tests never import ``torch``, ``qwen_tts``, ``qwen_asr`` or ``soundfile``.
They exercise the pure harness (:mod:`tests.e2e_validation`) with lightweight
fake model engines and the *real* portable Audio Postprocess + Caption Compiler
stages, so the full chain wiring and every required end-to-end property is
verified without CUDA and without touching the immutable fixtures.

Two negative tests prove the harness is meaningful: a ``raw.wav`` that changes
after synthesis is detected, and an alignment that does not match the request
text is rejected (never silently compiled).
"""

from __future__ import annotations

import json
import math
import unicodedata
from functools import lru_cache
from pathlib import Path

import pytest

from media_pipeline import CustomVoiceRequest, TTSArtifact
from typing import Any

from media_pipeline.alignment import AlignmentArtifact, AlignmentRequest
from media_pipeline.captions import AlignedToken, AlignmentMismatchError
from media_pipeline.postprocess import read_wav, write_wav

from e2e_validation import (
    RunChainHooks,
    E2EArtifacts,
    E2EAssertionError,
    snapshot_wav,
    run_chain,
    validate_e2e,
)

SAMPLE_RATE = 24_000
TEXT = (
    "真正限制本地人工智能模型使用体验的，"
    "往往并不只是模型本身的参数规模。"
)

_SKIPPABLE_CATEGORIES = frozenset("PZ")


def _is_skippable(char: str) -> bool:
    return char.isspace() or unicodedata.category(char)[0] in _SKIPPABLE_CATEGORIES

# Aligned tokens spell out exactly the spoken characters of TEXT, in order;
# whitespace/punctuation is not aligned (this mirrors the reviewed real model).
_ALIGNABLE = [char for char in TEXT if not _is_skippable(char)]


def _synthetic_waveform(frames: int) -> list[float]:
    """A finite, non-silent, mono float waveform (never non-finite)."""

    out: list[float] = []
    for index in range(frames):
        decay = max(0.05, 1.0 - index / frames)
        out.append(0.3 * math.sin(index / 80.0) * decay)
    return out


class _FakeTTSEngine:
    """Writes a valid mono 16-bit PCM WAV as the fresh TTS artifact."""

    def __call__(self, request: Any, output_wav_path: Any) -> Any:
        write_wav(Path(str(output_wav_path)), _fake_tts_samples(), SAMPLE_RATE)
        _, sample_rate, n_frames = read_wav(Path(str(output_wav_path)))
        return TTSArtifact(
            wav_path=Path(str(output_wav_path)),
            sample_rate=sample_rate,
            frames=n_frames,
        )


class _FakeAlignerEngine:
    """Produces one token per spoken character of the request text.

    Timestamps are strictly increasing with a measurable pause after the token
    that precedes the comma, and the final token ends inside the WAV duration.
    The concatenated token text equals the spoken characters of the request
    text, so it always matches the request text.
    """

    def __call__(self, request: Any, output_alignment_path: Any) -> Any:
        wav_path = Path(str(request.wav_path))
        text = request.text
        _, sample_rate, frames = read_wav(wav_path)
        duration = frames / sample_rate

        spoken = [char for char in text if not _is_skippable(char)]
        tokens: list[AlignedToken] = []
        time = 0.2  # leading silence before first speech
        step = 0.16
        for index, char in enumerate(spoken):
            start = time
            end = time + step
            tokens.append(AlignedToken(text=char, start=start, end=end))
            time = end
            if index == _SPEKEN_INDEX_OF_PAUSE:
                time += 0.5  # measurable pause -> caption break
        # Never extend past the WAV duration.
        if tokens:
            tokens[-1] = AlignedToken(
                text=tokens[-1].text,
                start=min(tokens[-1].start, duration - 0.1),
                end=min(tokens[-1].end, duration - 0.05),
            )

        records = [
            {"text": token.text, "start": token.start, "end": token.end}
            for token in tokens
        ]
        Path(str(output_alignment_path)).write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return AlignmentArtifact(
            alignment_path=Path(str(output_alignment_path)),
            sample_rate=sample_rate,
            frames=frames,
            token_count=len(tokens),
        )


def _waveform_to_samples(waveform: list[float]) -> list[int]:
    from media_pipeline.tts import waveform_to_mono_pcm16

    return waveform_to_mono_pcm16(waveform)


@lru_cache(maxsize=1)
def _fake_tts_samples() -> tuple[int, ...]:
    """Cache the immutable PCM payload; each engine call still writes a new WAV."""

    frames = SAMPLE_RATE * 8  # 8.00 s, a non-empty WAV
    return tuple(_waveform_to_samples(_synthetic_waveform(frames)))


# Spoken-character index at which the injects a measurable pause, just before
# the comma in the smoke text (between "的" and "往").
_SPEKEN_INDEX_OF_PAUSE = 16


# --- positive path ----------------------------------------------------------


def test_successful_chain_preserves_all_e2e_properties(tmp_path: Path) -> None:
    artifacts = run_chain(
        run_dir=tmp_path,
        text=TEXT,
        wav_engine=_FakeTTSEngine(),
        align_engine=_FakeAlignerEngine(),
    )

    # Artifact production and fresh TTS WAV format.
    assert artifacts.raw_wav_path.is_file()
    assert artifacts.alignment_path.is_file()
    assert artifacts.final_wav_path.is_file()
    assert artifacts.adjusted_alignment_path.is_file()
    assert artifacts.srt_path.is_file()
    assert artifacts.raw_wav_sample_rate == SAMPLE_RATE
    assert artifacts.raw_wav_frames == SAMPLE_RATE * 8
    # read_wav only accepts mono 16-bit PCM, so a successful read is the assertion.
    samples, frame_rate, frames = read_wav(artifacts.raw_wav_path)
    assert (frame_rate, frames) == (SAMPLE_RATE, SAMPLE_RATE * 8)
    assert len(samples) == frames

    # Alignment consumed the exact fresh WAV and covered all spoken characters.
    assert artifacts.alignment_sample_rate == artifacts.raw_wav_sample_rate
    assert artifacts.alignment_frames == artifacts.raw_wav_frames
    assert artifacts.token_count == len(_ALIGNABLE)
    assert artifacts.token_count == len(artifacts.original_tokens)

    # Re-reading the raw WAV after the whole run yields the original snapshot.
    assert snapshot_wav(artifacts.raw_wav_path)[1] == artifacts.raw_wav_digest

    # Postprocess frame and rate invariants.
    trim = artifacts.trim
    assert trim is not None
    assert trim.frame_rate == artifacts.raw_wav_sample_rate
    assert trim.frames == artifacts.raw_wav_frames
    assert artifacts.final_wav_frames == trim.kept_frames
    assert artifacts.final_wav_sample_rate == artifacts.raw_wav_sample_rate
    assert trim.kept_frames > 0
    assert trim.trim_before >= 0 and trim.trim_after >= 0

    # Adjusted alignment preserves token text, durations, and inter-token gaps.
    original = artifacts.original_tokens
    adjusted = artifacts.adjusted_tokens
    assert original and adjusted
    assert len(original) == len(adjusted)
    assert [token.text for token in original] == [
        token.text for token in adjusted
    ]

    offset = artifacts.trim.start_frame / artifacts.trim.frame_rate
    for old, new in zip(original, adjusted):
        # trim_alignment shifts every timestamp by the same quantized offset,
        # so durations are preserved and the shift magnitude equals the offset.
        assert (new.end - new.start) == pytest.approx(old.end - old.start)
        assert abs(new.start - old.start) == pytest.approx(offset)
        assert abs(new.end - old.end) == pytest.approx(offset)
        assert (new.start - old.start) == pytest.approx(new.end - old.end)

    for index in range(len(original) - 1):
        gap_old = original[index + 1].start - original[index].end
        gap_new = adjusted[index + 1].start - adjusted[index].end
        assert gap_new == pytest.approx(gap_old)

    # Captions reconstruct the source text and emitted SRT stays within WAV.
    assert artifacts.captions
    joined = "".join(caption.text for caption in artifacts.captions)
    assert "".join(joined.split()) == "".join(TEXT.split())

    assert artifacts.srt_text and artifacts.srt_text.strip()
    # SRT is the deterministic render of the captions and stays within the WAV.
    from e2e_validation import read_srt

    blocks = read_srt(artifacts.srt_text)
    assert len(blocks) == len(artifacts.captions)
    final_duration = artifacts.final_wav_frames / artifacts.final_wav_sample_rate
    for block in blocks:
        assert block["start"] < block["end"]
        assert block["end"] <= final_duration + 1e-6

    # The harness owns the remaining whole-chain properties.
    validate_e2e(artifacts)


# --- negatives: the harness must fail, not silently pass --------------------


def test_detects_raw_wav_changed_after_synthesis(tmp_path: Path) -> None:
    raw_wav_path = tmp_path / "raw.wav"

    def tamper() -> None:
        # Flip one sample: format/frames stay valid, but the bytes change, so
        # postprocess still runs yet the "raw.wav unchanged" invariant fails.
        samples, frame_rate, _ = read_wav(raw_wav_path)
        mutated = list(samples)
        mutated[len(mutated) // 2] += 1
        write_wav(raw_wav_path, mutated, frame_rate)

    with pytest.raises(E2EAssertionError, match="changed after synthesis"):
        run_chain(
            run_dir=tmp_path,
            text=TEXT,
            wav_engine=_FakeTTSEngine(),
            align_engine=_FakeAlignerEngine(),
            hooks=RunChainHooks(after_tts=tamper),
        )


def test_rejects_alignment_that_does_not_match_text(tmp_path: Path) -> None:
    alignment_path = tmp_path / "alignment.raw.json"

    def corrupt() -> None:
        # Valid JSON, valid timestamps, but the token text does not match the
        # request text: the Caption Compiler must reject it, not guess.
        mismatched = [{"text": "无关文本", "start": 0.2, "end": 0.5}]
        alignment_path.write_text(
            json.dumps(mismatched, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    with pytest.raises(AlignmentMismatchError):
        run_chain(
            run_dir=tmp_path,
            text=TEXT,
            wav_engine=_FakeTTSEngine(),
            align_engine=_FakeAlignerEngine(),
            hooks=RunChainHooks(after_align=corrupt),
        )


#--- signed-shift direction: the harness must reject the wrong shift ---------


def test_adjusted_alignment_wrong_direction_shift_is_rejected(tmp_path: Path) -> None:
    # Production Postprocess shifts every timestamp by *minus* the quantized
    # offset (new = old - offset). A shift by the *wrong* sign (old + offset)
    # used to pass under the old magnitude-only check and must now fail.
    artifacts = run_chain(
        run_dir=tmp_path,
        text=TEXT,
        wav_engine=_FakeTTSEngine(),
        align_engine=_FakeAlignerEngine(),
    )
    offset = artifacts.trim.start_frame / artifacts.trim.frame_rate
    # Base the candidate on the *original* tokens and shift by *+*offset (wrong
    # direction). The already-adjusted tokens are original - offset, so the
    # magnitude-only history would have passed this; the signed check must not.
    wrong = [
        AlignedToken(text=t.text, start=t.start + offset, end=t.end + offset)
        for t in artifacts.original_tokens
    ]
    artifacts.adjusted_tokens = wrong
    with pytest.raises(E2EAssertionError, match="signed quantized offset"):
        validate_e2e(artifacts)


def test_adjusted_alignment_correct_direction_shift_passes(tmp_path: Path) -> None:
    # A shift by the documented signed direction (original - offset) must pass.
    artifacts = run_chain(
        run_dir=tmp_path,
        text=TEXT,
        wav_engine=_FakeTTSEngine(),
        align_engine=_FakeAlignerEngine(),
    )
    offset = artifacts.trim.start_frame / artifacts.trim.frame_rate
    correct = [
        AlignedToken(text=t.text, start=t.start - offset, end=t.end - offset)
        for t in artifacts.original_tokens
    ]
    artifacts.adjusted_tokens = correct
    validate_e2e(artifacts)


def test_detects_raw_wav_changed_before_postprocess(tmp_path: Path) -> None:
    # The raw.wav snapshot is captured immediately after synthesis. If the WAV
    # is mutated before Audio Postprocess reads it, the postprocess-stage
    # byte-check must fail the run (proving the baseline was captured early,
    # not after downstream stages).
    raw_wav_path = tmp_path / "raw.wav"

    def tamper() -> None:
        samples, frame_rate, _ = read_wav(raw_wav_path)
        mutated = list(samples)
        mutated[len(mutated) // 2] += 1
        write_wav(raw_wav_path, mutated, frame_rate)

    with pytest.raises(E2EAssertionError, match="changed after synthesis"):
        run_chain(
            run_dir=tmp_path,
            text=TEXT,
            wav_engine=_FakeTTSEngine(),
            align_engine=_FakeAlignerEngine(),
            hooks=RunChainHooks(after_align=tamper),
        )


def test_snapshot_helper_detects_bytes_change(tmp_path: Path) -> None:
    wav = tmp_path / "u.wav"
    write_wav(wav, [0, 1000, -1000, 32767], SAMPLE_RATE)
    snap = snapshot_wav(wav)

    samples, frame_rate, _ = read_wav(wav)
    write_wav(wav, [samples[0] + 5, samples[1], samples[2], samples[3]], frame_rate)

    from e2e_validation import assert_wav_unchanged

    with pytest.raises(E2EAssertionError):
        assert_wav_unchanged(snap, wav)


# --- SRT document validation: the whole document, not just embedded blocks ---


def test_read_srt_rejects_garbage_prefix() -> None:
    # A valid block must not be accepted when garbage precedes it: the validator
    # must prove the *entire* document is the production SRT structure.
    from e2e_validation import CaptionError, read_srt

    with pytest.raises(CaptionError):
        read_srt("garbage\n\n1\n00:00:00,100 --> 00:00:01,000\ncaption\n")


def test_read_srt_rejects_garbage_trailing() -> None:
    # Trailing content after the last block must also be rejected.
    from e2e_validation import CaptionError, read_srt

    with pytest.raises(CaptionError):
        read_srt("1\n00:00:00,100 --> 00:00:01,000\ncaption\n\nGARBAGE\n")


def test_read_srt_keeps_payload_whitespace_verbatim() -> None:
    # The SRT payload is not .strip()ed, so extra leading/trailing whitespace on
    # the caption line is preserved (and therefore rejected by an exact text
    # comparison downstream), instead of being silently normalized away.
    from e2e_validation import read_srt

    trailing = read_srt("1\n00:00:00,100 --> 00:00:01,000\ncaption \n")
    assert trailing[0]["text"] == "caption "
    assert trailing[0]["text"] != "caption"

    leading = read_srt("1\n00:00:00,100 --> 00:00:01,000\n caption\n")
    assert leading[0]["text"] == " caption"
    assert leading[0]["text"] != "caption"


def test_emitted_srt_timestamp_past_final_wav_is_rejected() -> None:
    # A caption.end that is within the final WAV (float check passes) can still
    # round UP (ROUND_HALF_UP) to an emitted millisecond timestamp past the WAV.
    # The timing check must use the *emitted* timestamp, not the float caption.end.
    from media_pipeline.captions import Caption

    from e2e_validation import E2EArtifacts, _srt_timing_consistent_with_final_wav

    final_duration = 1.2346  # frames / rate
    artifacts = E2EArtifacts(
        run_dir=None,
        text="caption",
        final_wav_sample_rate=10_000,
        final_wav_frames=int(round(final_duration * 10_000)),
        captions=[Caption(start=1.0, end=1.2346, text="caption")],
        # caption.end == final_duration (float check passes), but the emitted
        # millisecond end rounds UP to 1.235, which is past the WAV.
        srt_text="1\n00:00:01,000 --> 00:00:01,235\ncaption\n",
    )
    with pytest.raises(E2EAssertionError):
        _srt_timing_consistent_with_final_wav(artifacts)
