"""ai-core end-to-end integration test for the full Speech Pipeline v0.

The real pipeline spans **two** separate GPU virtual environments on ai-core:

- Production TTS lives in the TTS environment;
- Production Alignment lives in the Alignment environment.

A single pytest process cannot import both, so this test runs each model stage
in its own environment via ``subprocess`` and only the portable stages
(Audio Postprocess + Caption Compiler) run in-process. This is the two-process
validation sequence realised as a gated integration test.

Gating mirrors the existing integration tests: this test runs only when all of
the following environment variables are set (never hard-coded here):

- ``MEDIA_PIPELINE_TTS_MODEL`` -- Qwen3-TTS CustomVoice model path;
- ``MEDIA_PIPELINE_ALIGNMENT_MODEL`` -- Qwen3 ForcedAligner model path;
- ``MEDIA_PIPELINE_TTS_PYTHON`` -- interpreter of the TTS environment;
- ``MEDIA_PIPELINE_ALIGNMENT_PYTHON`` -- interpreter of the Alignment environment.

Absence of any of them means GPU integration was not requested, so the test
skips. When all are set, integration is explicitly requested: a missing
interpreter, a ``torch``/``qwen_tts``/``qwen_asr`` import failure, a CUDA/model
failure, or a synthesis/alignment failure must **FAIL**, never skip, so this
stays the real GPU validation while remaining separated from the ordinary CPU
suite. The fresh TTS WAV produced in this run is the exact WAV that Alignment
consumes -- the immutable fixtures are never used for that path.

The chain wiring and every required end-to-end property come from the portable
harness :func:`tests.e2e_validation.run_chain` / ``validate_e2e``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from media_pipeline.postprocess import read_wav, write_wav
from media_pipeline.tts import TTSArtifact

from e2e_validation import run_chain, validate_e2e

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "speech-smoke-001"
ORIGINAL_TEXT = FIXTURE / "original.txt"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"

_TTS_MODEL = os.environ.get("MEDIA_PIPELINE_TTS_MODEL")
_ALIGNMENT_MODEL = os.environ.get("MEDIA_PIPELINE_ALIGNMENT_MODEL")
_TTS_PYTHON = os.environ.get("MEDIA_PIPELINE_TTS_PYTHON")
_ALIGNMENT_PYTHON = os.environ.get("MEDIA_ALIGNMENT_PYTHON") or os.environ.get(
    "MEDIA_PIPELINE_ALIGNMENT_PYTHON"
)

# A fresh Chinese narration utterance, read-only: never modified, never used as
# a substitute for the produced WAV.
TEXT = ORIGINAL_TEXT.read_text(encoding="utf-8")
SPEAKER = "Uncle_Fu"
_INSTRUCT = (
    "自然、清晰、克制的中文知识类视频旁白，语速中等，不要夸张。"
)

_missing = [
    name
    for name, value in (
        ("MEDIA_PIPELINE_TTS_MODEL", _TTS_MODEL),
        ("MEDIA_PIPELINE_ALIGNMENT_MODEL", _ALIGNMENT_MODEL),
        ("MEDIA_PIPELINE_TTS_PYTHON", _TTS_PYTHON),
        ("MEDIA_ALIGNMENT_PYTHON", _ALIGNMENT_PYTHON),
    )
    if not value
]

# Absence of the E2E runtime configuration means integration was not requested:
# skip only the real-runtime E2E. When it is set, every missing dependency,
# model, runtime or stage must FAIL rather than skip.
skip_integration = pytest.mark.skipif(
    _missing,
    reason=(
        "E2E runtime configuration not fully set ("
        + ", ".join(_missing)
        + "): GPU integration was not requested. When all are set, any "
        "torch/qwen_tts/qwen_asr/model/subprocess/synthesis/alignment "
        "failure fails the test."
    ),
)


# --- subprocess-backed stage engines ----------------------------------------


def _write_stage_script(run_dir: Path, name: str, body: str) -> Path:
    script = run_dir / name
    script.write_text(body, encoding="utf-8")
    return script


def _run_stage(python: str, script: Path, *argv: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [python, str(script), *argv],
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"stage failed in its environment:\nstdout: {completed.stdout}\nstderr: {completed.stderr}"
        )


_TTS_STAGE = r"""
import json
import sys
from pathlib import Path

from media_pipeline import CustomVoiceRequest
from media_pipeline.runtimes.qwen_tts import Qwen3CustomVoiceTTS

model_path, out_wav, request = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
engine = Qwen3CustomVoiceTTS(model_path)
engine.synthesize(
    CustomVoiceRequest(
        text=request["text"],
        language=request["language"],
        speaker=request["speaker"],
        instruct=request["instruct"],
    ),
    out_wav,
)
"""

_ALIGN_STAGE = r"""
import json
import sys
from pathlib import Path

from media_pipeline.alignment import AlignmentRequest, validate_wav, load_alignment
from media_pipeline.runtimes.qwen_aligner import Qwen3ForcedAlignment

model_path, out_align, wav, request = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
engine = Qwen3ForcedAlignment(model_path)
engine.align(
    AlignmentRequest(wav_path=wav, text=request["text"], language=request["language"]),
    out_align,
)
artifact = {
    "sample_rate": validate_wav(wav)[0],
    "frames": validate_wav(wav)[1],
    "token_count": len(load_alignment(out_align)),
}
Path(out_align + ".artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
"""


def _subprocess_wav_engine(run_dir: Path, tts_python: str, tts_model: str):
    """A TTS stage engine that synthesizes inside the TTS environment."""

    def engine(request: object, output_wav_path: object) -> object:
        argv = json.dumps(
            {
                "text": request.text,
                "language": request.language,
                "speaker": request.speaker,
                "instruct": request.instruct,
            }
        )
        script = _write_stage_script(run_dir, "e2e_synthesize.py", _TTS_STAGE)
        _run_stage(tts_python, script, tts_model, str(output_wav_path), argv)
        _, sample_rate, frames = read_wav(output_wav_path)
        return TTSArtifact(
            wav_path=Path(output_wav_path),
            sample_rate=sample_rate,
            frames=frames,
        )

    return engine


def _subprocess_align_engine(run_dir: Path, align_python: str, align_model: str):
    """An Alignment stage engine that aligns inside the Alignment environment."""

    def engine(request: object, output_alignment_path: object) -> object:
        argv = json.dumps({"text": request.text, "language": request.language})
        script = _write_stage_script(run_dir, "e2e_align.py", _ALIGN_STAGE)
        _run_stage(
            align_python, script, align_model, str(output_alignment_path), str(request.wav_path), argv
        )
        artifact_meta = json.loads(
            Path(str(output_alignment_path) + ".artifact.json").read_text(encoding="utf-8")
        )
        from media_pipeline.alignment import AlignmentArtifact

        return AlignmentArtifact(
            alignment_path=Path(output_alignment_path),
            sample_rate=artifact_meta["sample_rate"],
            frames=artifact_meta["frames"],
            token_count=artifact_meta["token_count"],
        )

    return engine


@skip_integration
def test_full_pipeline_end_to_end(tmp_path: Path) -> None:
    assert _TTS_PYTHON and _ALIGNMENT_PYTHON and _TTS_MODEL and _ALIGNMENT_MODEL
    tts_engine = _subprocess_wav_engine(tmp_path, _TTS_PYTHON, _TTS_MODEL)
    align_engine = _subprocess_align_engine(tmp_path, _ALIGNMENT_PYTHON, _ALIGNMENT_MODEL)

    artifacts = run_chain(
        run_dir=tmp_path,
        text=TEXT,
        speaker=SPEAKER,
        wav_engine=tts_engine,  # type: ignore[arg-type]
        align_engine=align_engine,  # type: ignore[arg-type]
    )

    # The real run produced every artifact; now assert every required property.
    validate_e2e(artifacts)

    # Audit surface for the ai-core record.
    artifacts.extra["tts_model"] = _TTS_MODEL
    artifacts.extra["alignment_model"] = _ALIGNMENT_MODEL
    artifacts.extra["final_duration"] = artifacts.final_wav_frames / artifacts.final_wav_sample_rate


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
