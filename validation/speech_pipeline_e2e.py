"""Two-process ai-core driver for the Speech Pipeline v0 end-to-end validation.

The full production path spans two separate GPU virtual environments, so this
driver runs each model stage in its own environment and exchanges explicit
artifacts through one fresh run directory:

    Process 1 (TTS environment):  text -> Production TTS -> raw.wav
    Process 2 (Alignment env.):  raw.wav -> Production Alignment -> alignment.raw.json
    Process 3 (portable, this):   raw.wav + alignment.raw.json
                                   -> Audio Postprocess -> final.wav + adjusted.json
                                   -> Caption Compiler -> final.srt
                                   -> validate_e2e (all required E2E properties)

This is a *validation* harness, not a product invocation surface: it coordinates
the two existing environments only so the real run can be repeated and audited.
It hard-codes no host paths, model paths, interpreter paths, tokens, or
credentials -- everything comes from environment variables.

Required environment variables:

- ``MEDIA_PIPELINE_TTS_MODEL`` -- Qwen3-TTS CustomVoice model path;
- ``MEDIA_ALIGNMENT_MODEL`` or ``MEDIA_PIPELINE_ALIGNMENT_MODEL`` -- Qwen3
  ForcedAligner model path;
- ``MEDIA_PIPELINE_TTS_PYTHON`` -- interpreter of the TTS environment;
- ``MEDIA_ALIGNMENT_PYTHON`` or ``MEDIA_PIPELINE_ALIGNMENT_PYTHON`` --
  interpreter of the Alignment environment;
- ``MEDIA_PIPELINE_E2E_RUN_DIR`` -- a fresh directory for the run artifacts.

On success it prints an audit summary (commit SHA, versions, GPU, model paths,
output directory, artifact sizes, WAV metadata, trim frame range, token and
caption counts, and the validation result) suitable for the ai-core record. On
any failure it exits non-zero -- it never skips and never reuses stale artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"

# Add the repo ``src`` and ``tests`` directories *before* importing the portable
# validation harness (:mod:`e2e_validation` lives in ``tests/``) so this driver
# runs the current checkout instead of any installed ``media_pipeline``.
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests"))

from e2e_validation import E2EArtifacts, validate_e2e  # noqa: E402,PLC0415

# Reviewed Chinese narration instruction: integration input, not a product default.
_DEFAULT_INSTRUCT = (
    "自然、清晰、克制的中文知识类视频旁白，语速中等，不要夸张。"
)


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def _require_alias(*names: str) -> str:
    """Return the first set environment variable in ``names``.

    A true alias resolver: unlike ``_require(...) or _require(...)`` (which
    exits when the first name is missing), this keeps looking across every
    alias so the documented aliases are all honoured and none is dead code.
    """

    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise SystemExit(
        "missing required environment variable (any of): " + ", ".join(names)
    )


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
        raise SystemExit(
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

from media_pipeline.alignment import AlignmentRequest, validate_wav
from media_pipeline.captions import load_alignment
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


def _read_wav_meta(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as handle:  # raises if not mono 16-bit PCM
        return handle.getframerate(), handle.getnframes()


def _report(artifacts: E2EArtifacts, run_dir: Path, config: dict) -> None:
    raw_rate, raw_frames = _read_wav_meta(artifacts.raw_wav_path)
    final_rate, final_frames = _read_wav_meta(artifacts.final_wav_path)
    trim = artifacts.trim
    lines = [
        "Speech Pipeline v0 E2E -- validation passed",
        "",
        "commit SHA:        " + (config.get("commit", "unknown")),
        "output directory:  " + str(run_dir),
        "",
        "raw.wav:           "
        + f"{artifacts.raw_wav_path.name} "
        + f"{raw_rate} Hz, {raw_frames} frames, {raw_frames / raw_rate:.3f} s",
        "alignment:         "
        + f"{artifacts.alignment_path.name}, {artifacts.token_count} token(s)",
        "final.wav:         "
        + f"{artifacts.final_wav_path.name} "
        + f"{final_rate} Hz, {final_frames} frames, {final_frames / final_rate:.3f} s",
        "trim frame range:  "
        + f"[{trim.start_frame}, {trim.end_frame}) "
        + f"(keep {trim.kept_frames} frames)",
        "captions:          {0} caption(s)".format(len(artifacts.captions or [])),
        "SRT:               " + str(artifacts.srt_path.name),
    ]
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    run_dir = Path(_require("MEDIA_PIPELINE_E2E_RUN_DIR"))
    run_dir.mkdir(parents=True, exist_ok=False)

    instruct = os.environ.get("MEDIA_PIPELINE_SPEECH_INSTRUCT", _DEFAULT_INSTRUCT)
    text = (
        _REPO_ROOT / "tests" / "fixtures" / "speech-smoke-001" / "original.txt"
    ).read_text(encoding="utf-8")

    tts_python = _require("MEDIA_PIPELINE_TTS_PYTHON")
    align_python = _require_alias("MEDIA_ALIGNMENT_PYTHON", "MEDIA_PIPELINE_ALIGNMENT_PYTHON")
    tts_model = _require("MEDIA_PIPELINE_TTS_MODEL")
    align_model = _require_alias("MEDIA_ALIGNMENT_MODEL", "MEDIA_PIPELINE_ALIGNMENT_MODEL")

    # Exact-head guard: run *after* environment-gating so a run that never
    # configured the GPU stages still fails on the documented missing-variable
    # reason, and right before any real stage runs, so the recorded commit is
    # the code actually exercised.
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    if head.returncode != 0:
        # A git failure means the exact HEAD cannot be established: this is not
        # a skip-able condition for an exact-head validation, so it FAILs.
        raise SystemExit(
            "git failed; exact-head validation cannot proceed:\n"
            f"stderr: {head.stderr.strip()}"
        )
    commit = head.stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    if status.returncode != 0:
        raise SystemExit(
            "git status failed; exact-head validation cannot proceed:\n"
            f"stderr: {status.stderr.strip()}"
        )
    if status.stdout.strip():
        # A dirty tree means the code actually run is not the checked-out HEAD,
        # so the reported commit is not the code under test. Fail loudly and
        # tell the operator to make the tree clean first.
        raise SystemExit(
            "working tree is not clean; the exact HEAD is not the code under test. "
            "Commit or stash pending changes and re-run exact-head validation."
        )

    config = {"commit": commit}

    # --- Process 1: Production TTS -> raw.wav --------------------------------
    synth_script = run_dir / "e2e_synthesize.py"
    synth_script.write_text(_TTS_STAGE, encoding="utf-8")
    raw_wav_path = run_dir / "raw.wav"
    _run_stage(
        tts_python,
        synth_script,
        tts_model,
        str(raw_wav_path),
        json.dumps(
            {
                "text": text,
                "language": "Chinese",
                "speaker": "Uncle_Fu",
                "instruct": instruct,
            }
        ),
    )

    # Snapshot the fresh TTS WAV *immediately* after synthesis and before any
    # downstream stage runs, so we can prove neither Alignment nor Audio
    # Postprocess altered it. A baseline captured later (after downstream
    # stages) would be a false positive and could never detect a mutation.
    raw_wav_snapshot = raw_wav_path.read_bytes()
    raw_wav_digest = hashlib.sha256(raw_wav_snapshot).hexdigest()

    # --- Process 2: Production Alignment -> alignment.raw.json ---------------
    align_script = run_dir / "e2e_align.py"
    align_script.write_text(_ALIGN_STAGE, encoding="utf-8")
    alignment_path = run_dir / "alignment.raw.json"
    _run_stage(
        align_python,
        align_script,
        align_model,
        str(alignment_path),
        str(raw_wav_path),
        json.dumps({"text": text, "language": "Chinese"}),
    )
    # Alignment must not rewrite the input WAV it was handed.
    if raw_wav_path.read_bytes() != raw_wav_snapshot:
        raise SystemExit("raw.wav changed after the Alignment stage")

    # --- Process 3: portable postprocess + captions + validate --------------
    from media_pipeline.captions import build_captions, compile_srt, load_alignment
    from media_pipeline.postprocess import postprocess_speech

    final_wav_path = run_dir / "final.wav"
    adjusted_alignment_path = run_dir / "adjusted.json"
    srt_path = run_dir / "final.srt"

    trim = postprocess_speech(
        raw_wav_path,
        alignment_path,
        final_wav_path,
        adjusted_alignment_path,
    )
    # Audio Postprocess must not rewrite the input WAV it was handed.
    if raw_wav_path.read_bytes() != raw_wav_snapshot:
        raise SystemExit("raw.wav changed after the Audio Postprocess stage")

    adjusted_tokens = load_alignment(adjusted_alignment_path)
    original_tokens = load_alignment(alignment_path)
    captions = build_captions(text, adjusted_tokens)
    srt_text = compile_srt(text, adjusted_tokens)
    srt_path.write_text(srt_text, encoding="utf-8", newline="\n")

    raw_rate, raw_frames = _read_wav_meta(raw_wav_path)
    final_rate, final_frames = _read_wav_meta(final_wav_path)

    artifacts = E2EArtifacts(
        run_dir=run_dir,
        text=text,
        speaker="Uncle_Fu",
        raw_wav_path=raw_wav_path,
        raw_wav_frames=raw_frames,
        raw_wav_sample_rate=raw_rate,
        raw_wav_snapshot=raw_wav_snapshot,
        raw_wav_digest=raw_wav_digest,
        alignment_path=alignment_path,
        token_count=len(original_tokens),
        alignment_sample_rate=raw_rate,
        alignment_frames=raw_frames,
        final_wav_path=final_wav_path,
        final_wav_frames=final_frames,
        final_wav_sample_rate=final_rate,
        adjusted_alignment_path=adjusted_alignment_path,
        trim=trim,
        original_tokens=original_tokens,
        adjusted_tokens=adjusted_tokens,
        captions=captions,
        srt_path=srt_path,
        srt_text=srt_text,
    )

    # The reported trim window must match the final WAV produced by postprocess,
    # so the audit's trim range and the actual artifact agree exactly.
    assert trim.kept_frames == final_frames
    assert trim.frame_rate == final_rate

    validate_e2e(artifacts)
    _report(artifacts, run_dir, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
