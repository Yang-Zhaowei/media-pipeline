"""Unit tests for the portable TTS contracts and PCM16 processing (no GPU).

These tests never import ``torch``, ``qwen_tts``, or ``soundfile``. They cover
the portable side of the TTS stage: the request/output contracts, request
validation, the deterministic waveform-to-mono-16-bit-PCM conversion, and the
WAV round-trip contract the Audio Postprocess stage consumes.
"""

from __future__ import annotations

import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from media_pipeline import (
    CustomVoiceError,
    CustomVoiceRequest,
    TTSRuntimeError,
    TTSArtifact,
)
from media_pipeline.tts import (
    validate_request,
    waveform_to_mono_pcm16,
)
from media_pipeline.postprocess import read_wav, write_wav

_PCM16_MIN = -32_768
_PCM16_MAX = 32_767


# --- portable boundary: importing the package must not load the runtime -----


def test_importing_media_pipeline_does_not_pull_torch_or_qwen_or_soundfile() -> None:
    # A fresh interpreter importing media_pipeline must not pull any heavy
    # runtime dependency, and must keep the runtime package un-imported. This
    # runs in a child interpreter via sys.executable so the result is
    # independent of the parent pytest process -- which may already have
    # imported real torch from the GPU integration tests -- and of pytest
    # collection/execution order. The check inspects the child's own
    # sys.modules, so parent-process state cannot skew it. The portable
    # package lives under src/, which must be on sys.path before import.
    src = Path(__file__).resolve().parent.parent / "src"
    script = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "import media_pipeline\n"
        "loaded = [\n"
        "    m for m in (\n"
        "        'torch', 'qwen_tts', 'qwen_asr',\n"
        "        'soundfile', 'media_pipeline.runtimes',\n"
        "    ) if m in sys.modules\n"
        "]\n"
        "if loaded:\n"
        "    print('unexpectedly loaded: ' + ', '.join(loaded))\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    ) % str(src)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout or result.stderr


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
    assert request.instruct == ""


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


def test_pcm16_conversion_nonfinite_raises() -> None:
    import math

    # NaN and +/-Inf must fail clearly, never silently become silence.
    with pytest.raises(TTSRuntimeError):
        waveform_to_mono_pcm16([math.nan])
    with pytest.raises(TTSRuntimeError):
        waveform_to_mono_pcm16([math.inf])
    with pytest.raises(TTSRuntimeError):
        waveform_to_mono_pcm16([-math.inf])


def test_pcm16_conversion_stays_in_range_for_many_values() -> None:
    values = [x / 1000.0 for x in range(-1500, 1501)]
    out = waveform_to_mono_pcm16(values)
    assert all(_PCM16_MIN <= sample <= _PCM16_MAX for sample in out)


def test_pcm16_rejects_nested_channel_shaped_input() -> None:
    # v0 expects one flat mono waveform; nested/channel-shaped input is rejected.
    with pytest.raises(TTSRuntimeError):
        waveform_to_mono_pcm16([[0.0, 0.2], [0.4, 0.6]])
    with pytest.raises(TTSRuntimeError):
        waveform_to_mono_pcm16([0.1, [0.2], 0.3])


def test_pcm16_treats_integers_as_normalized_floats() -> None:
    # The helper interprets every numeric value as a normalized float (assumed
    # within [-1.0, 1.0]); integers are not treated as already-scaled PCM.
    assert waveform_to_mono_pcm16([0, 1, -1]) == [0, _PCM16_MAX, _PCM16_MIN]


def test_pcm16_accepts_non_python_real_scalars_like_numpy_float32() -> None:
    # The real Qwen CustomVoice waveform is a 1-D float32 ndarray; its samples
    # are NumPy float32 scalars, which are numbers.Real but NOT Python int/float.
    # fractions.Fraction is a numbers.Real proxy so this runs on CPU without
    # depending on NumPy/torch.
    samples = waveform_to_mono_pcm16([Fraction(1, 2), Fraction(-1, 4)])
    assert samples == [16384, -8192]


def test_pcm16_rejects_bool_as_audio_sample() -> None:
    # bool is a numbers.Real but must never be treated as a sample value.
    with pytest.raises(TTSRuntimeError):
        waveform_to_mono_pcm16([True, False])


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
