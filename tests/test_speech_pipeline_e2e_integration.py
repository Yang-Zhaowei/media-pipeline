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
- ``MEDIA_ALIGNMENT_MODEL`` or ``MEDIA_ALIGNMENT_MODEL`` -- Qwen3 ForcedAligner
  model path;
- ``MEDIA_PIPELINE_TTS_PYTHON`` -- interpreter of the TTS environment;
- ``MEDIA_ALIGNMENT_PYTHON`` or ``MEDIA_PIPELINE_ALIGNMENT_PYTHON`` --
  interpreter of the Alignment environment.

A run is gated only when *every* alias for a slot is unset, so a GPU E2E env
configured with either the short ``MEDIA_ALIGNMENT_*`` alias or the long
``MEDIA_ALIGNMENT_*`` alias is never wrongly skipped.

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


def _any_alias(*names: str) -> str | None:
    """Return the first set environment variable in ``names`` (or ``None``)."""

    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


_TTS_MODEL = os.environ.get("MEDIA_PIPELINE_TTS_MODEL")
_TTS_PYTHON = os.environ.get("MEDIA_PIPELINE_TTS_PYTHON")
_ALIGNMENT_MODEL = _any_alias("MEDIA_ALIGNMENT_MODEL", "MEDIA_PIPELINE_ALIGNMENT_MODEL")
_ALIGNMENT_PYTHON = _any_alias(
    "MEDIA_ALIGNMENT_PYTHON", "MEDIA_PIPELINE_ALIGNMENT_PYTHON"
)

# The Alignment model and python slots each accept two documented aliases.
_ALIGNMENT_MODEL_ALIASES = (
    "MEDIA_ALIGNMENT_MODEL",
    "MEDIA_PIPELINE_ALIGNMENT_MODEL",
)
_ALIGNMENT_PYTHON_ALIASES = (
    "MEDIA_ALIGNMENT_PYTHON",
    "MEDIA_PIPELINE_ALIGNMENT_PYTHON",
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
        ("MEDIA_PIPELINE_TTS_PYTHON", _TTS_PYTHON),
        (_ALIGNMENT_MODEL_ALIASES[0], _ALIGNMENT_MODEL),
        (_ALIGNMENT_PYTHON_ALIASES[0], _ALIGNMENT_PYTHON),
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


#--- configuration / import hygiene (CPU, no GPU required) -------------------


def test_alignment_env_alias_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    # Supported Alignment env aliases must not cause a wrong skip: whichever
    # alias is set is honoured, and unset-both yields None (=> gated/skip).
    monkeypatch.delenv("MEDIA_ALIGNMENT_MODEL", raising=False)
    monkeypatch.delenv("MEDIA_PIPELINE_ALIGNMENT_MODEL", raising=False)
    monkeypatch.setenv("MEDIA_PIPELINE_ALIGNMENT_MODEL", "/models/align")
    assert _any_alias("MEDIA_ALIGNMENT_MODEL", "MEDIA_PIPELINE_ALIGNMENT_MODEL") == "/models/align"
    monkeypatch.setenv("MEDIA_ALIGNMENT_MODEL", "/short")
    assert _any_alias(
        "MEDIA_ALIGNMENT_MODEL", "MEDIA_PIPELINE_ALIGNMENT_MODEL"
    ) == "/short"
    monkeypatch.delenv("MEDIA_ALIGNMENT_MODEL", raising=False)
    monkeypatch.delenv("MEDIA_PIPELINE_ALIGNMENT_MODEL", raising=False)
    monkeypatch.delenv("MEDIA_ALIGNMENT_PYTHON", raising=False)
    monkeypatch.delenv("MEDIA_PIPELINE_ALIGNMENT_PYTHON", raising=False)
    assert _any_alias("MEDIA_ALIGNMENT_PYTHON", "MEDIA_PIPELINE_ALIGNMENT_PYTHON") is None


def test_embedded_align_stage_import_target_exists() -> None:
    # The embedded Alignment stage imports ``load_alignment`` from
    # ``media_pipeline.captions`` (its real home), never from
    # ``media_pipeline.alignment`` -- an import from the wrong module raises
    # ImportError and would abort a configured real GPU E2E before Alignment.
    import contextlib
    import io
    import py_compile

    import media_pipeline.alignment as align_mod

    assert not hasattr(align_mod, "load_alignment")
    from media_pipeline.captions import load_alignment  # noqa: F401  (real target)

    # The embedded stage script compiles cleanly under the CPU interpreter.
    script = _REPO_ROOT / "tests" / "_align_stage_compile.py"
    script.write_text(_ALIGN_STAGE, encoding="utf-8")
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            py_compile.compile(str(script), doraise=True)
    finally:
        script.unlink(missing_ok=True)


def test_standalone_driver_starts_from_repo_root_and_gates_on_env() -> None:
    # Documented ``python validation/speech_pipeline_e2e.py`` from the repo root
    # must reach environment-variable configuration validation, not a
    # ModuleNotFoundError for ``e2e_validation`` (i.e. sys.path is wired up
    # before importing the portable harness).
    proc = subprocess.run(
        [sys.executable, "validation/speech_pipeline_e2e.py"],
        cwd=str(_REPO_ROOT),
        env={k: v for k, v in os.environ.items() if not k.startswith("MEDIA_")},
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "ModuleNotFoundError" not in combined
    assert "missing required environment variable" in combined


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
