"""Unit tests for the portable TTS contracts and PCM16 processing (no GPU).

These tests never import ``torch``, ``qwen_tts``, or ``soundfile``. They cover
the portable side of the TTS stage: the request/output contracts, request
validation, the deterministic waveform-to-mono-16-bit-PCM conversion, and the
WAV round-trip contract the Audio Postprocess stage consumes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from media_pipeline import (
    DEFAULT_INSTRUCT,
    CustomVoiceError,
    CustomVoiceRequest,
    TTSArtifact,
)
from media_pipeline.tts import (
    flatten_to_mono,
    validate_request,
    waveform_to_mono_pcm16,
)
from media_pipeline.postprocess import read_wav, write_wav

_PCM16_MIN = -32_768
_PCM16_MAX = 32_767


# --- portable boundary: importing the package must not load the runtime -----


def _is_imported(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def test_importing_media_pipeline_does_not_pull_torch_or_qwen_or_soundfile() -> None:
    # A fresh interpreter already importing media_pipeline must not have pulled
    # any heavy runtime dependency. The runtime package stays un-imported too.
    assert "torch" not in sys.modules
    assert "qwen_tts" not in sys.modules
    assert "soundfile" not in sys.modules
    assert "media_pipeline.runtimes" not in sys.modules


def test_runtime_import_is_available_only_if_installed() -> None:
    # This documents the environment, not a requirement: the adapter module is
    # import-safe, but torch/qwen_tts are optional (present only on ai-core).
    assert (_is_imported("torch") and _is_imported("qwen_tts")) or (
        not _is_imported("torch") and not _is_imported("qwen_tts")
    )


# --- CustomVoiceRequest semantics ------------------------------------------


def test_request_preserves_text_language_speaker_instruct() -> None:
    request = CustomVoiceRequest(
        text="真正限制本地人工智能模型使用体验的，",
        language="Chinese",
        speaker="Uncle_Fu",
        instruct="自然、清晰、克制的中文知识类视频旁白，语速中等，不要夸张。",
    )
    assert request.text == "真正限制本地人工智能模型使用体验的，"
    assert request.language == "Chinese"
    assert request.speaker == "Uncle_Fu"
    assert request.instruct == "自然、清晰、克制的中文知识类视频旁白，语速中等，不要夸张。"


def test_request_instruct_defaults_to_empty_string() -> None:
    request = CustomVoiceRequest(text="hello", language="English", speaker="Ann")
    assert request.instruct == DEFAULT_INSTRUCT == ""


def test_request_is_frozen() -> None:
    request = CustomVoiceRequest(text="hi", language="English", speaker="Ann")
    with pytest.raises(AttributeError):
        request.speaker = "Bob"  # type: ignore[misc]


# --- validation ------------------------------------------------------------


def test_validate_request_accepts_valid_request() -> None:
    validate_request(
        CustomVoiceRequest(text="你好", language="Chinese", speaker="Uncle_Fu")
    )


def test_validate_request_accepts_empty_instruct() -> None:
    validate_request(
        CustomVoiceRequest(text="你好", language="Chinese", speaker="Uncle_Fu")
    )


def test_validate_request_rejects_empty_text() -> None:
    with pytest.raises(CustomVoiceError):
        validate_request(CustomVoiceRequest(text="   ", language="Chinese", speaker="X"))


def test_validate_request_rejects_empty_language() -> None:
    with pytest.raises(CustomVoiceError):
        validate_request(CustomVoiceRequest(text="hi", language="", speaker="X"))


def test_validate_request_rejects_empty_speaker() -> None:
    with pytest.raises(CustomVoiceError):
        validate_request(CustomVoiceRequest(text="hi", language="English", speaker=""))


# --- PCM16 conversion ------------------------------------------------------


def test_pcm16_conversion_half_up_rounding() -> None:
    # f * 32768 with half-up rounding; 0.5 * 32768 = 16384 exactly.
    assert waveform_to_mono_pcm16([0.5]) == [16384]
    assert waveform_to_mono_pcm16([0.0]) == [0]


def test_pcm16_conversion_clips_positive_and_negative_peaks() -> None:
    assert waveform_to_mono_pcm16([1.0]) == [_PCM16_MAX]
    assert waveform_to_mono_pcm16([-1.0]) == [_PCM16_MIN]
    # Beyond full scale clips explicitly into range.
    assert waveform_to_mono_pcm16([2.5]) == [_PCM16_MAX]
    assert waveform_to_mono_pcm16([-2.5]) == [_PCM16_MIN]


def test_pcm16_conversion_nonfinite_becomes_silence() -> None:
    import math

    assert waveform_to_mono_pcm16([math.nan]) == [0]
    assert waveform_to_mono_pcm16([math.inf]) == [0]
    assert waveform_to_mono_pcm16([-math.inf]) == [0]


def test_pcm16_conversion_stays_in_range_for_many_values() -> None:
    values = [x / 1000.0 for x in range(-1500, 1501)]
    out = waveform_to_mono_pcm16(values)
    assert all(_PCM16_MIN <= sample <= _PCM16_MAX for sample in out)


def test_flatten_to_mono_flat_sequence_is_already_mono() -> None:
    assert flatten_to_mono([0.1, 0.2, 0.3]) == [0.1, 0.2, 0.3]


def test_flatten_to_mono_mixes_channels_by_averaging() -> None:
    # Two channels of 4 samples each -> mono by averaging per sample.
    mixed = flatten_to_mono([[0.0, 0.2, 0.4, 0.6], [1.0, 1.2, 1.4, 1.6]])
    assert mixed == pytest.approx([0.5, 0.7, 0.9, 1.1])


def test_flatten_to_mono_truncates_to_shortest_channel() -> None:
    assert flatten_to_mono([[0.0, 0.2, 0.4], [1.0, 1.2, 1.4, 1.6]]) == pytest.approx(
        [0.5, 0.7, 0.9]
    )


def test_pcm16_treats_integers_as_normalized_floats() -> None:
    # The helper interprets every numeric value as a normalized float (assumed
    # within [-1.0, 1.0]); integers are not treated as already-scaled PCM.
    assert waveform_to_mono_pcm16([0, 1, -1]) == [0, _PCM16_MAX, _PCM16_MIN]


# --- WAV round-trip contract consumed by Audio Postprocess -----------------


def test_write_and_read_wav_are_mono_16bit_pcm(tmp_path) -> None:
    wav_path = tmp_path / "utterance.wav"
    samples = [0, 1000, -1000, 32767, -32768, 0]
    write_wav(wav_path, samples, 16000)

    read_samples, frame_rate, frames = read_wav(wav_path)
    assert frame_rate == 16000
    assert frames == len(samples) == 6
    assert read_samples == samples


def test_tts_artifact_reads_back_authoritative_frame_count(tmp_path) -> None:
    wav_path = tmp_path / "utterance.wav"
    samples = waveform_to_mono_pcm16([0.1 * i for i in range(8)])
    write_wav(wav_path, samples, 24000)

    _, artifact_rate, artifact_frames = read_wav(wav_path)
    artifact = TTSArtifact(wav_path=Path(wav_path), sample_rate=artifact_rate, frames=artifact_frames)
    assert artifact.sample_rate == 24000
    assert artifact.frames == 8
    assert artifact.duration == pytest.approx(8 / 24000)
