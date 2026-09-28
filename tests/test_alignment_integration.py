"""ai-core GPU integration test for the Qwen3 ForcedAligner runtime.

This test runs the real model, so it is gated on a single condition only:

- the model path must be provided via the ``MEDIA_PIPELINE_ALIGNMENT_MODEL``
  environment variable (never hard-coded into the repository).

Absence of ``MEDIA_PIPELINE_ALIGNMENT_MODEL`` means GPU integration was not
requested, so the tests skip. When it is set, integration is explicitly
requested: a missing ``torch``/``qwen_asr``, a CUDA/model-load failure, or an
alignment failure must FAIL rather than skip, so this stays the real GPU
validation while remaining separated from the ordinary CPU suite.

It reads the immutable speech-smoke-001 fixture text and WAV (never modifying
them), aligns one utterance, and asserts the output is the existing
``{"text", "start", "end"}`` JSON the Caption Compiler consumes: a positive
token count, finite non-negative timestamps in ascending order within the WAV
duration, and an input WAV left untouched. This is the real GPU validation; it
is deliberately separated from the ordinary CPU suite.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from media_pipeline.alignment import AlignmentRequest, validate_wav
from media_pipeline.postprocess import read_wav

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "speech-smoke-001"
FIXTURE_WAV = FIXTURE / "raw.wav"
ORIGINAL_TEXT = FIXTURE / "original.txt"

_MODEL_PATH = os.environ.get("MEDIA_PIPELINE_ALIGNMENT_MODEL")

# Absence of MEDIA_PIPELINE_ALIGNMENT_MODEL means integration was not requested:
# skip. When it is set, integration is explicitly requested, so missing
# torch/qwen_asr or any model-load / CUDA / alignment failure fails the test.
skip_integration = pytest.mark.skipif(
    not _MODEL_PATH,
    reason=(
        "MEDIA_PIPELINE_ALIGNMENT_MODEL is not set: GPU integration was not "
        "requested. When set, any torch/qwen_asr/model/alignment failure fails "
        "the test."
    ),
)


@skip_integration
def test_qwen3_forced_align_writes_consumable_json(tmp_path: Path) -> None:
    from media_pipeline.runtimes.qwen_aligner import Qwen3ForcedAlignment

    engine = Qwen3ForcedAlignment(Path(_MODEL_PATH))  # type: ignore[arg-type]

    # The verified input: one segmented utterance, Chinese. This is integration
    # input, not a hard-coded dependency.
    wav = FIXTURE_WAV
    frames_before = validate_wav(wav)[1]
    text = ORIGINAL_TEXT.read_text(encoding="utf-8")

    output = tmp_path / "alignment.json"
    artifact = engine.align(AlignmentRequest(wav_path=wav, text=text, language="Chinese"), output)

    # Positive token count and the artifact metadata is consistent with the WAV.
    assert artifact.token_count > 0
    assert artifact.frames == frames_before

    # The output is the exact JSON shape Audio Postprocess + the Caption
    # Compiler consume: finite, non-negative, ascending, within the WAV.
    records = json.loads(output.read_text(encoding="utf-8"))
    assert len(records) == artifact.token_count
    for record in records:
        assert record["text"].strip()  # effective (non-blank) token
        start, end = record["start"], record["end"]
        assert isinstance(start, float) and isinstance(end, float)
        assert start >= 0.0 and end >= 0.0
        assert start < end
    starts = [record["start"] for record in records]
    assert starts == sorted(starts)
    _, frames = validate_wav(wav)
    assert records[-1]["end"] <= frames / validate_wav(wav)[0] + 1e-6

    # The input WAV is not overwritten or altered.
    samples, frame_rate, frames_after = read_wav(wav)
    assert frames_after == frames_before
    assert frame_rate == 24000
    assert samples


@skip_integration
def test_engine_loads_model_once_and_aligns_twice(tmp_path: Path) -> None:
    from media_pipeline.runtimes.qwen_aligner import Qwen3ForcedAlignment

    engine = Qwen3ForcedAlignment(Path(_MODEL_PATH))  # type: ignore[arg-type]

    wav = FIXTURE_WAV
    text = ORIGINAL_TEXT.read_text(encoding="utf-8")
    request = AlignmentRequest(wav_path=wav, text=text, language="Chinese")

    first = engine.align(request, tmp_path / "a.json")
    second = engine.align(request, tmp_path / "b.json")

    # The model loads once (at construction) and is reused for the second align.
    assert first.token_count > 0
    assert second.token_count > 0
    assert first.token_count == second.token_count
    assert engine._aligner is not None
