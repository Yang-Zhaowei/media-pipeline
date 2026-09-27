"""ai-core GPU integration test for the Qwen3-TTS CustomVoice runtime.

This test runs the real model. It is gated so it is skipped everywhere that is
not the GPU integration host:

- ``torch`` and ``qwen_tts`` must be importable, and
- the model path must be provided via the ``MEDIA_PIPELINE_TTS_MODEL``
  environment variable (never hard-coded into the repository).

It reads the immutable speech-smoke-001 fixture text (never modifying it),
synthesizes one utterance, and asserts the output is a valid mono 16-bit PCM
WAV with a positive frame count. This is the real GPU validation; it is
deliberately separated from the ordinary CPU suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from media_pipeline import CustomVoiceRequest
from media_pipeline.postprocess import read_wav

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "speech-smoke-001"
ORIGINAL_TEXT = FIXTURE / "original.txt"

_MODEL_PATH = os.environ.get("MEDIA_PIPELINE_TTS_MODEL")

# Absence of MEDIA_PIPELINE_TTS_MODEL means integration was not requested: skip.
# When it is set, integration is explicitly requested, so missing torch/qwen_tts
# or any model-load / CUDA / synthesis failure must FAIL, not skip.
skip_integration = pytest.mark.skipif(
    not _MODEL_PATH,
    reason=(
        "MEDIA_PIPELINE_TTS_MODEL is not set: GPU integration was not requested. "
        "When set, any torch/qwen_tts/model/synthesis failure fails the test."
    ),
)


@skip_integration
def test_qwen3_custom_voice_synthesizes_mono_pcm16_artifact(tmp_path: Path) -> None:
    from media_pipeline.runtimes.qwen_tts import Qwen3CustomVoiceTTS

    text = ORIGINAL_TEXT.read_text(encoding="utf-8")
    model_path = Path(_MODEL_PATH)  # type: ignore[arg-type]

    engine = Qwen3CustomVoiceTTS(model_path)

    # The runtime configuration is narrow and verified: Chinese + the custom
    # speaker. The verified Chinese narration instruction is integration input.
    artifact = engine.synthesize(
        CustomVoiceRequest(
            text=text,
            language="Chinese",
            speaker="Uncle_Fu",
            instruct="自然、清晰、克制的中文知识类视频旁白，语速中等，不要夸张。",
        ),
        tmp_path / "customvoice.wav",
    )

    assert artifact.sample_rate > 0
    assert artifact.frames > 0

    # The artifact is the exact format Audio Postprocess consumes: mono,
    # uncompressed, 16-bit PCM.
    samples, frame_rate, frames = read_wav(artifact.wav_path)
    assert frame_rate == artifact.sample_rate
    assert frames == artifact.frames
    assert len(samples) == artifact.frames


@skip_integration
def test_engine_loads_model_once_and_synthesizes_twice(tmp_path: Path) -> None:
    from media_pipeline.runtimes.qwen_tts import Qwen3CustomVoiceTTS

    text = ORIGINAL_TEXT.read_text(encoding="utf-8")
    engine = Qwen3CustomVoiceTTS(Path(_MODEL_PATH))  # type: ignore[arg-type]

    first = engine.synthesize(
        CustomVoiceRequest(text=text, language="Chinese", speaker="Uncle_Fu"),
        tmp_path / "a.wav",
    )
    second = engine.synthesize(
        CustomVoiceRequest(text=text, language="Chinese", speaker="Uncle_Fu"),
        tmp_path / "b.wav",
    )

    assert first.frames > 0
    assert second.frames > 0
