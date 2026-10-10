"""Segmented speech rendering orchestration (v0).

Public entry point :func:`render_speech` turns a *pre-segmented* script JSON
into a complete episode mono 16-bit PCM WAV, a matching SRT and a verifiable
total timeline, reusing the four existing production stages:

    text -> TTS -> forced alignment -> audio postprocess -> caption compiler -> WAV + SRT

This module is the orchestration boundary between deterministic portable
processing and the two host-specific GPU runtimes. It follows the segment
assembly contract in :file:`docs/contracts/segmented-speech-v0.md` (C1-C9) and
supports legacy CustomVoice speakers or a reusable Base clone asset. It adds
no HTTP/MCP, long-text splitting, script revision, resume/cache, or loudness mastering.

Design boundaries enforced here:

- All deterministic logic -- request preflight, timeline bookkeeping, WAV data
  frame assembly, final product verification -- is pure Python and runs without
  CUDA.
- The two model stages run in their own subprocesses, each loading its model
  exactly once and processing every segment in one session. Model paths,
  interpreter paths and device are supplied by the caller, never hard-coded.
- The production path never imports :mod:`tests` or :mod:`experiments`. Only
  the portable stages (Audio Postprocess + Caption Compiler) run in the parent
  process; the model adapters are imported lazily inside the subprocesses.
- Per-segment model inference is isolated: each segment's local timeline starts
  at its own audio zero, so a later segment never depends on an earlier one.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable

from .alignment import (
    AlignmentRequest,
    AlignmentRequestError,
    AlignmentRuntimeError,
)
from .captions import (
    Caption,
    CaptionError,
    build_captions,
    load_alignment,
    render_srt,
)
from .postprocess import (
    DEFAULT_FADE_IN,
    DEFAULT_FADE_OUT,
    DEFAULT_POST_PADDING,
    DEFAULT_PRE_PADDING,
    AudioPostprocessError,
    postprocess_speech,
    read_wav,
    write_wav,
)
from .tts import CustomVoiceRequest, TTSRuntimeError

__all__ = [
    "STATUS_COMPLETE",
    "STATUS_FAILED",
    "STATUS_INCOMPLETE",
    "STATUS_RUNNING",
    "RenderError",
    "RenderResult",
    "SegmentResult",
    "load_and_validate_script",
    "render_speech",
]

#: Floating-point tolerance shared with the production stages for coarse
#: second-value comparisons that are not frame-exact.
_TIMING_TOLERANCE = 1e-6

#: Millisecond budget an SRT boundary may deviate from the precise audio edge
#: purely through ``ROUND_HALF_UP`` quantization. Used only for the audio
#: boundary check, never to tolerate accumulated drift.
_SRT_MS_TOLERANCE = 0.5

# Absolute path to this installed package, resolved from this module. The
# embedded model stages load this exact package without adding its parent to
# ``sys.path`` (which could shadow dependencies in the runtime environment).
_PACKAGE_DIR = Path(__file__).resolve().parent

# The public callable signature: ``task_dict -> results_dict`` (one model stage,
# one model load, all segments in a single session). Kept as an optional internal
# seam so the lifecycle can be exercised on CPU with fakes without touching the
# documented production signature.
TaskCallable = Callable[[dict], dict]

# Statuses written into the report and returned to the caller.
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_INCOMPLETE = "incomplete"
STATUS_FAILED = "failed"

# Internal segment result statuses reported by the model subprocesses.
_SEG_OK = "ok"
_SEG_ALIGNMENT_FAILED = "alignment_failed"
_SEG_SYNTHESIS_FAILED = "synthesis_failed"

# Fields allowed in the top-level request and in each segment. Unknown fields are
# rejected so a voice/time parameter typo cannot be silently ignored.
_TOP_LEVEL_KEYS = frozenset({"language", "speaker", "voice", "instruct", "segments"})
_SEGMENT_KEYS = frozenset({"id", "text", "pause_after_ms", "instruct"})


class RenderError(Exception):
    """A preflight or run-level fault raises this.

    Preflight faults (before any model starts) carry no ``run_dir``. Once a run
    has started, ``run_dir`` points at the run directory holding the partial
    artifacts and report so a caller can recover them.
    """

    def __init__(self, message: str, *, run_dir: Path | None = None) -> None:
        super().__init__(message)
        self.run_dir = run_dir


# --- request model ----------------------------------------------------------


@dataclass(frozen=True)
class ValidatedSegment:
    """One caller-approved performance unit, possibly containing many sentences.

    ``instruct=None`` represents an omitted override, never JSON ``null``.
    An explicit empty string clears inherited top-level direction.
    """

    id: str
    text: str
    pause_after_ms: int
    instruct: str | None = None


@dataclass(frozen=True)
class ValidatedScript:
    """A fully preflighted segmented script."""

    language: str
    speaker: str | None
    instruct: str
    max_segment_chars: int
    segments: list[ValidatedSegment]
    clone_asset: str | None = None


@dataclass(frozen=True)
class SegmentResult:
    """The outcome of one input segment after the whole run.

    ``status`` is one of ``ok``, ``alignment_failed``, ``postprocess_failed``
    or ``caption_failed``. ``stage`` names the deterministic stage that failed
    and ``reason`` keeps the original message so the failure stays locatable.
    """

    segment_id: str
    status: str
    stage: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class RenderResult:
    """The object :func:`render_speech` returns after normal processing.

    On full success ``wav_path``/``srt_path``/``timeline_path`` point at the
    published ``final/`` products; otherwise they are ``None`` and the status is
    ``incomplete``. ``segments`` carries the per-segment report.
    """

    status: str
    run_dir: Path
    report_path: Path
    wav_path: Path | None
    srt_path: Path | None
    timeline_path: Path | None
    segments: list[SegmentResult]
    config: dict = field(default_factory=dict)
    completed_stages: list[str] = field(default_factory=list)


# --- request preflight ------------------------------------------------------


def load_and_validate_script(
    path: str | Path,
    *,
    max_segment_chars: int,
    encoding: str = "utf-8",
) -> ValidatedScript:
    """Read and preflight a pre-segmented script JSON.

    Raises :class:`RenderError` (with no ``run_dir``) if anything is wrong so a
    model is never loaded for a bad request. All statically discoverable
    problems are collected and reported together; the input file is never
    modified. ``max_segment_chars`` is the caller input budget measured in
    Unicode code points; a segment over budget is rejected, never truncated.
    """

    if isinstance(max_segment_chars, bool) or not isinstance(max_segment_chars, int):
        raise RenderError("max_segment_chars must be a positive integer (not a bool)")
    if max_segment_chars <= 0:
        raise RenderError("max_segment_chars must be a positive integer")

    try:
        with open(path, encoding=encoding) as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RenderError(f"{path} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise RenderError(f"cannot read script {path}: {exc}") from exc

    validated, errors = _validate_script_data(
        raw, max_segment_chars, script_dir=Path(path).parent
    )
    if errors:
        raise RenderError(
            "script preflight failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    return validated


def _validate_script_data(
    data: object, max_segment_chars: int, *, script_dir: Path | None = None
) -> tuple[ValidatedScript, list[str]]:
    """Validate the parsed request, collecting every static error at once."""

    errors: list[str] = []

    if not isinstance(data, dict):
        return (
            ValidatedScript("", "", "", max_segment_chars, []),
            ["script must be a JSON object"],
        )

    # --- top-level fields ---
    for key in set(data) - _TOP_LEVEL_KEYS:
        errors.append(f"unknown top-level field {key!r}")

    language = data.get("language")
    if not isinstance(language, str) or not language.strip():
        errors.append("language is required and must be a non-empty string")

    if ("speaker" in data) == ("voice" in data):
        errors.append("exactly one of speaker or voice must be present")
    speaker = data.get("speaker")
    if "voice" not in data or "speaker" in data:
        if not isinstance(speaker, str) or not speaker.strip():
            errors.append("speaker is required and must be a non-empty string")

    clone_asset = None
    voice = data.get("voice")
    is_clone = isinstance(voice, dict) and voice.get("type") == "clone"
    if "voice" in data:
        if not isinstance(voice, dict):
            errors.append("voice must be an object with exactly type and asset")
        else:
            for key in set(voice) - {"type", "asset"}:
                errors.append(f"unknown voice field {key!r}")
            if not is_clone:
                errors.append("voice.type must be 'clone'")
            try:
                _resolve_clone_asset(voice.get("asset"), script_dir or Path.cwd())
            except RenderError as exc:
                errors.append(str(exc))
            else:
                clone_asset = voice["asset"]

    instruct = data.get("instruct", "")
    if not isinstance(instruct, str):
        errors.append("instruct must be a string when present (defaults to empty)")
    elif is_clone and instruct != "":
        errors.append("instruct: non-empty instruct is unsupported for cloned voices")

    raw_segments = data.get("segments")  # type: ignore[union-attr]
    segments_list: list[dict] = []
    if not isinstance(raw_segments, list) or not raw_segments:
        errors.append("segments is required and must be a non-empty array")
    else:
        _validate_segments(raw_segments, max_segment_chars, errors)
        if is_clone:
            for index, segment in enumerate(raw_segments):
                if isinstance(segment, dict) and isinstance(segment.get("instruct"), str):
                    if segment["instruct"] != "":
                        errors.append(
                            f"segment[{index}].instruct: non-empty instruct is unsupported for cloned voices"
                        )
        # Build the usable list only from entries that are actually objects: a
        # malformed entry (for example ``null``) is already reported by
        # ``_validate_segments`` and must stay a summarised preflight error,
        # so this comprehension must never raise here (C3). ``segments_list``
        # is only consumed on the no-errors path, where every segment is a
        # validated dict, so the filter does not change successful results.
        segments_list = [
            dict(seg) for seg in raw_segments if isinstance(seg, dict)
        ]

    if errors:
        return ValidatedScript("", "", "", max_segment_chars, []), errors

    return (
        ValidatedScript(
            language=language,  # type: ignore[arg-type]  validated non-empty str above
            speaker=speaker,  # type: ignore[arg-type]
            instruct=instruct,
            max_segment_chars=max_segment_chars,
            segments=[
                ValidatedSegment(
                    id=str(seg["id"]),
                    text=str(seg["text"]),
                    pause_after_ms=int(seg.get("pause_after_ms", 0)),
                    instruct=seg.get("instruct"),
                )
                for seg in segments_list
            ],
            clone_asset=clone_asset,
        ),
        [],
    )


def _resolve_clone_asset(asset: object, script_dir: Path) -> Path:
    """Resolve a portable relative asset inside its script directory tree.

    Existing symlinks are resolved for containment; the asset need not exist
    and is never opened/deserialized during static preflight. The returned host
    path is used only by the private TTS subprocess, never portable provenance.
    """
    if not isinstance(asset, str) or not asset.strip():
        raise RenderError("voice.asset must be a non-empty relative path")
    windows = PureWindowsPath(asset)
    if windows.drive or windows.root or PurePosixPath(asset).is_absolute() or ":" in asset:
        raise RenderError("voice.asset must be a relative path without a drive, root, or URI scheme")
    try:
        if "\x00" in asset:
            raise ValueError("path contains a NUL character")
        root = script_dir.resolve()
        resolved = (root / asset.replace("\\", "/")).resolve()
        resolved.relative_to(root)
    except (ValueError, OSError, RuntimeError) as exc:
        raise RenderError(
            "voice.asset must resolve inside the script directory: " + str(exc)
        ) from exc
    return resolved


def _voice_fields(script: ValidatedScript) -> dict:
    """Portable clone provenance or the unchanged legacy speaker field."""
    if script.clone_asset is not None:
        return {"voice": {"type": "clone", "asset": script.clone_asset}}
    return {"speaker": script.speaker}


def _validate_segments(
    segments_in: list[object], max_segment_chars: int, errors: list[str]
) -> None:
    seen_ids: set[str] = set()
    for index, segment in enumerate(segments_in):
        where = f"segment[{index}]"
        if not isinstance(segment, dict):
            errors.append(f"{where} must be an object, got {type(segment).__name__}")
            continue

        for key in set(segment) - _SEGMENT_KEYS:
            errors.append(f"{where} has unknown field {key!r}")

        segment_id = segment.get("id")
        if not isinstance(segment_id, str) or not segment_id:
            errors.append(f"{where}.id must be a non-empty string")
            segment_id = None

        text = segment.get("text")
        if not isinstance(text, str) or not text:
            errors.append(f"{where}.text must be a non-empty string")
        elif not _has_alignable_text(text):
            errors.append(
                f"{where}.text must contain alignable content "
                "(whitespace or punctuation only is not alignable)"
            )
        elif len(text) > max_segment_chars:
            # Include the segment id when available so the summarised error is
            # locatable (C3). ``segment_id`` is ``None`` only when the id itself
            # was already rejected above.
            id_ref = f" (segment id {segment_id!r})" if segment_id is not None else ""
            errors.append(
                f"{where}.text{id_ref} is {len(text)} code points, over the limit of "
                f"{max_segment_chars}; reject, do not split or truncate"
            )

        pause = segment.get("pause_after_ms", 0)
        if isinstance(pause, bool) or not isinstance(pause, int):
            errors.append(f"{where}.pause_after_ms must be a non-negative integer")
        elif pause < 0:
            errors.append(f"{where}.pause_after_ms must not be negative")

        if "instruct" in segment and not isinstance(segment["instruct"], str):
            errors.append(f"{where}.instruct must be a string when present")

        if segment_id is not None:
            if segment_id in seen_ids:
                errors.append(f"duplicate segment id {segment_id!r}")
            seen_ids.add(segment_id)


def _has_alignable_text(text: str) -> bool:
    """True if ``text`` carries at least one spoken (aligned) character.

    Reuses the Caption Compiler's non-spoken definition so a request of only
    whitespace and/or punctuation is rejected deterministically before any
    model is loaded.
    """

    import unicodedata

    skippable = frozenset("PZ")
    return any(
        not (char.isspace() or unicodedata.category(char)[0] in skippable)
        for char in text
    )


# --- safe intermediate file names -------------------------------------------


def _safe_name(segment_id: str) -> str:
    """Map a segment id to a collision-free, path-safe intermediate name.

    The id is an identifier, never an unprocessed path. The safe name keeps the
    human-readable id body and appends a short digest so distinct ids can never
    collide, and strips anything that could escape the segments directory.
    """

    body = re.sub(r"[^A-Za-z0-9_.-]", "_", segment_id).strip("._") or "_segment"
    digest = hashlib.sha1(segment_id.encode("utf-8")).hexdigest()[:8]
    return f"{body}.{digest}"


# --- report model -----------------------------------------------------------


@dataclass
class _RunReport:
    """The report written to ``report.json`` and re-read by consumers."""

    status: str
    config: dict
    completed_stages: list[str]
    segments: list[dict]
    artifacts: dict


def _write_report(run_dir: Path, report: _RunReport, *, strict: bool = False) -> None:
    """Write ``report.json``.

    By default the write is best-effort: on an unwritable disk the failure is
    swallowed so a report failure never fabricates success and never masks the
    original run-level exception (C6). On the success path ``strict`` is set:
    a report write failure must not be silently ignored, because the completion
    marker is written only after this call returns successfully.
    """

    payload = {
        "status": report.status,
        "run_dir": ".",
        "config": report.config,
        "completed_stages": list(report.completed_stages),
        "segments": list(report.segments),
        "artifacts": dict(report.artifacts),
    }
    try:
        (run_dir / "report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        if strict:
            raise
        # Best-effort: the disk may be unwritable; surface nothing false. The
        # original run-level exception is what the caller sees.
        return


def report_segment(
    report: _RunReport,
    segment_id: str,
    status: str,
    *,
    stage: str | None = None,
    reason: str | None = None,
) -> None:
    """Update one segment's entry in the run report."""

    for entry in report.segments:
        if entry["segment_id"] == segment_id:
            entry["status"] = status
            if stage is not None:
                entry["stage"] = stage
            if reason is not None:
                entry["reason"] = reason
            break


def _to_segment_results(report: _RunReport) -> list[SegmentResult]:
    return [
        SegmentResult(
            segment_id=entry["segment_id"],
            status=entry["status"],
            stage=entry.get("stage"),
            reason=entry.get("reason"),
        )
        for entry in report.segments
    ]


# --- public entry point -----------------------------------------------------


def render_speech(
    script_path: str | Path,
    output_dir: str | Path,
    *,
    tts_python: str,
    alignment_python: str,
    tts_model: str,
    alignment_model: str,
    max_segment_chars: int,
    device: str = "cuda:0",
    # Internal seam (not part of the documented production signature): inject
    # ready model-stage task callables so the lifecycle and subprocess protocol
    # can be exercised on CPU with fakes. When ``None`` the default production
    # subprocess engines are built from the interpreter/model paths above.
    _wav_task: TaskCallable | None = None,
    _align_task: TaskCallable | None = None,
) -> RenderResult:
    """Render a pre-segmented script into a complete episode WAV + SRT + timeline.

    The documented production signature takes the two interpreter paths, the two
    model paths and the caller input budget; ``device`` and the ``_``-prefixed
    task seams are internal. Every deterministic stage runs in this process
    without CUDA; the two model stages each load their model exactly once inside
    a dedicated subprocess.
    """

    run_dir = Path(output_dir)

    # --- preflight (before any model or output directory is touched) --------
    script = load_and_validate_script(
        script_path, max_segment_chars=max_segment_chars
    )

    if run_dir.exists():
        raise RenderError(
            f"output directory {run_dir} already exists; refusing to overwrite "
            f"an existing run or its inputs"
        )

    segments = [
        {
            "id": seg.id,
            "text": seg.text,
            "pause_after_ms": seg.pause_after_ms,
            "effective_instruct": script.instruct if seg.instruct is None else seg.instruct,
            "instruct_source": "top_level" if seg.instruct is None else "segment",
        }
        for seg in script.segments
    ]
    safe_names = {seg.id: _safe_name(seg.id) for seg in script.segments}

    report = _RunReport(
        status=STATUS_RUNNING,
        config={
            "language": script.language,
            **_voice_fields(script),
            "instruct": script.instruct,
            "max_segment_chars": script.max_segment_chars,
            "tts_python": str(tts_python),
            "alignment_python": str(alignment_python),
            "tts_model": str(tts_model),
            "alignment_model": str(alignment_model),
            "device": device,
            "segment_order": [seg.id for seg in script.segments],
        },
        completed_stages=[],
        segments=[
            {
                "segment_id": seg["id"],
                "text": seg["text"],
                "effective_instruct": seg["effective_instruct"],
                "instruct_source": seg["instruct_source"],
                "status": "pending",
            }
            for seg in segments
        ],
        artifacts={},
    )

    # --- create run directory and publish the initial report ----------------
    try:
        run_dir.mkdir(parents=True)
        (run_dir / "request.json").write_text(
            json.dumps(
                {
                    "language": script.language,
                    **_voice_fields(script),
                    "instruct": script.instruct,
                    "segments": [
                        {
                            "id": seg.id,
                            "text": seg.text,
                            "pause_after_ms": seg.pause_after_ms,
                            **({"instruct": seg.instruct} if seg.instruct is not None else {}),
                        }
                        for seg in script.segments
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise RenderError(
            f"cannot create run directory {run_dir}: {exc}", run_dir=run_dir
        ) from exc

    _write_report(run_dir, report)
    report.completed_stages.append("preflight")

    wav_task = _wav_task or _make_tts_task(tts_python, tts_model, device)
    align_task = _align_task or _make_align_task(alignment_python, alignment_model, device)

    try:
        # --- TTS: one load, synthesize every segment, then exit -------------
        _run_tts_stage(
            run_dir, script, safe_names, segments, wav_task, report, device,
            script_dir=Path(script_path).parent.resolve(),
        )
        report.completed_stages.append("tts")

        # --- Alignment: one load, align every segment independently ---------
        failed_alignment = _run_alignment_stage(
            run_dir, script, safe_names, segments, align_task, report, device
        )
        report.completed_stages.append("alignment")

        # --- Postprocess + captions per segment (pure, CPU) -----------------
        cleaned = _run_postprocess_stage(
            run_dir, script, safe_names, segments, failed_alignment, report
        )
        report.completed_stages.append("postprocess")
        report.completed_stages.append("captions")

        # --- Assemble whole episode + verify final products -----------------
        # This performs the full completion transaction: it writes the
        # ``complete`` report successfully and then the ``.complete`` marker
        # last, or returns an ``incomplete`` result, or raises a run-level
        # ``RenderError`` (with ``run_dir``) on any final-verification fault.
        return _assemble_and_publish(run_dir, script, safe_names, cleaned, report)
    except RenderError as exc:  # noqa: PERF203 - teardown then re-raise
        # An explicit run-level fault (TTS synthesis failure, sample-rate
        # mismatch, final-verification failure, ...) must not leave the report
        # as ``running``; record the failure best-effort and keep the partial
        # artifacts recoverable via ``run_dir`` (C6).
        report.status = STATUS_FAILED
        _write_report(run_dir, report)
        if exc.run_dir is None:
            exc.run_dir = run_dir
        raise
    except Exception as exc:  # noqa: BLE001 - run-level, keep partial results
        report.status = STATUS_FAILED
        _write_report(run_dir, report)
        raise RenderError(f"run failed: {exc}", run_dir=run_dir) from exc


# --- model stage: TTS (one subprocess, one model load) ----------------------


def _run_tts_stage(
    run_dir: Path,
    script: ValidatedScript,
    safe_names: dict[str, str],
    segments: list[dict],
    wav_task: TaskCallable,
    report: _RunReport,
    device: str,
    *,
    script_dir: Path,
) -> None:
    """Run the TTS stage: one subprocess loads the model once and synthesizes all segments.

    Any per-segment synthesis failure or model-runtime failure is a run-level
    fault (C5): keep already-generated WAVs and stop before the episode summary.
    """

    task = {
        "run_dir": str(run_dir),
        "language": script.language,
        **_voice_fields(script),
        "instruct": script.instruct,
        "device": device,
        "segments": [
            {
                "id": seg["id"],
                "safe_name": safe_names[seg["id"]],
                "text": seg["text"],
                "instruct": seg["effective_instruct"],
            }
            for seg in segments
        ],
    }
    if script.clone_asset is not None:
        task["script_dir"] = str(script_dir)
    results = wav_task(task)

    runtime_error = results.get("runtime_error")
    if runtime_error:
        raise RenderError(
            f"TTS stage runtime failure: {runtime_error}", run_dir=run_dir
        )

    by_id = {entry["segment_id"]: entry for entry in results.get("segments", [])}
    for seg in segments:
        entry = by_id.get(seg["id"], {"status": "unknown"})
        if entry.get("status") != _SEG_OK:
            raise RenderError(
                f"TTS synthesis failed for segment {seg['id']!r}: "
                f"{entry.get('reason', 'unknown reason')}",
                run_dir=run_dir,
            )

    # Per-segment deterministic verification of every freshly written WAV.
    for seg in segments:
        _verify_segment_wav(run_dir, safe_names[seg["id"]])


def _verify_segment_wav(run_dir: Path, safe_name: str) -> None:
    """A generated segment WAV must be a valid, non-empty, mono 16-bit PCM WAV."""

    wav_path = run_dir / "segments" / (safe_name + ".wav")
    if not wav_path.exists():
        raise RenderError(f"TTS produced no WAV for {safe_name}", run_dir=run_dir)
    try:
        samples, sample_rate, frames = read_wav(wav_path)
    except AudioPostprocessError as exc:
        raise RenderError(
            f"TTS WAV for {safe_name} is not valid mono 16-bit PCM: {exc}",
            run_dir=run_dir,
        ) from exc
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise RenderError(
            f"TTS WAV for {safe_name} has an invalid sample rate: {sample_rate!r}",
            run_dir=run_dir,
        )
    if not isinstance(frames, int) or frames <= 0 or len(samples) != frames:
        raise RenderError(
            f"TTS WAV for {safe_name} is empty or malformed: frames={frames}",
            run_dir=run_dir,
        )


# --- model stage: Alignment (one subprocess, one model load) ----------------


def _run_alignment_stage(
    run_dir: Path,
    script: ValidatedScript,
    safe_names: dict[str, str],
    segments: list[dict],
    align_task: TaskCallable,
    report: _RunReport,
    device: str,
) -> list[str]:
    """Run the Alignment stage: one subprocess loads the model once and aligns all segments.

    A deterministic per-segment alignment failure (invalid timestamps or text
    mapping) is isolated: record ``alignment_failed`` for that segment and keep
    processing the independent ones. A model-runtime failure stops the run.
    Returns the list of segment ids that failed alignment.
    """

    task = {
        "run_dir": str(run_dir),
        "language": script.language,
        "device": device,
        "segments": [
            {"id": seg["id"], "safe_name": safe_names[seg["id"]], "text": seg["text"]}
            for seg in segments
        ],
    }
    results = align_task(task)

    runtime_error = results.get("runtime_error")
    if runtime_error:
        raise RenderError(
            f"Alignment stage runtime failure: {runtime_error}", run_dir=run_dir
        )

    by_id = {entry["segment_id"]: entry for entry in results.get("segments", [])}
    failed: list[str] = []
    for seg in segments:
        entry = by_id.get(seg["id"], {"status": "unknown"})
        status = entry.get("status", "unknown")
        if status == _SEG_OK:
            report_segment(report, seg["id"], "ok")
        elif status == _SEG_ALIGNMENT_FAILED:
            failed.append(seg["id"])
            report_segment(
                report,
                seg["id"],
                "alignment_failed",
                stage="alignment",
                reason=entry.get("reason", "alignment validation failed"),
            )
        else:
            # Unknown/empty result is a run-level fault, not an isolatable
            # segment error, so it must not be silently continued.
            raise RenderError(
                f"Alignment returned an unexpected result for segment {seg['id']!r}: "
                f"{entry!r}",
                run_dir=run_dir,
            )

    return failed


# --- pure stage: postprocess + captions (CPU, one segment at a time) --------


@dataclass(frozen=True)
class _CleanedSegment:
    """A segment that passed every stage and can participate in assembly."""

    segment_id: str
    safe_name: str
    samples: list[int]
    sample_rate: int
    frames: int
    pause_after_ms: int
    local_captions: list[Caption]


def _run_postprocess_stage(
    run_dir: Path,
    script: ValidatedScript,
    safe_names: dict[str, str],
    segments: list[dict],
    failed_alignment: list[str],
    report: _RunReport,
) -> list[_CleanedSegment]:
    """Per-segment Audio Postprocess + Caption Compiler on the successful segments.

    A deterministic postprocess or caption failure records the segment, stage
    and reason and leaves the other independent segments to proceed. Returns the
    ordered list of successfully cleaned segments needed for assembly.
    """

    failed_alignment_set = set(failed_alignment)
    cleaned: list[_CleanedSegment] = []

    for seg in segments:
        segment_id = seg["id"]
        safe_name = safe_names[segment_id]
        if segment_id in failed_alignment_set:
            continue

        raw_wav = run_dir / "segments" / (safe_name + ".wav")
        raw_alignment = run_dir / "segments" / (safe_name + ".alignment.raw.json")
        cleaned_wav = run_dir / "segments" / (safe_name + ".cleaned.wav")
        adjusted_alignment = run_dir / "segments" / (safe_name + ".alignment.adjusted.json")

        try:
            postprocess_speech(
                raw_wav,
                raw_alignment,
                cleaned_wav,
                adjusted_alignment,
                pre_padding=DEFAULT_PRE_PADDING,
                post_padding=DEFAULT_POST_PADDING,
                fade_in=DEFAULT_FADE_IN,
                fade_out=DEFAULT_FADE_OUT,
            )
        except AudioPostprocessError as exc:
            # A deterministic validation failure (malformed WAV, stale or
            # untrimmable alignment) is the only isolatable per-segment fault in
            # this stage; disk I/O (OSError) is a run-level fault and must stop
            # the run (C5), so it is deliberately not caught here.
            report_segment(
                report,
                segment_id,
                "postprocess_failed",
                stage="postprocess",
                reason=str(exc),
            )
            continue

        try:
            tokens = load_alignment(adjusted_alignment)
            local_captions = build_captions(seg["text"], tokens)
        except CaptionError as exc:
            # Only the deterministic caption/alignment semantic validation
            # exceptions (``AlignmentError`` / ``AlignmentMismatchError``) are
            # isolatable per-segment faults. Any other error (for example an
            # I/O failure) is a run-level fault and must stop the run (C5), so
            # it is deliberately not caught here.
            report_segment(
                report,
                segment_id,
                "caption_failed",
                stage="captions",
                reason=str(exc),
            )
            continue

        samples, sample_rate, frames = read_wav(cleaned_wav)
        cleaned.append(
            _CleanedSegment(
                segment_id=segment_id,
                safe_name=safe_name,
                samples=samples,
                sample_rate=sample_rate,
                frames=frames,
                pause_after_ms=seg["pause_after_ms"],
                local_captions=list(local_captions),
            )
        )
        report_segment(report, segment_id, "ok")

    return cleaned


# --- pure stage: assemble whole episode + publish final products ------------


def _assemble_and_publish(
    run_dir: Path,
    script: ValidatedScript,
    safe_names: dict[str, str],
    cleaned: list[_CleanedSegment],
    report: _RunReport,
) -> RenderResult:
    """Concatenate cleaned segment frames, shift captions and publish final products.

    Returns a ``complete`` result when every segment succeeded and the final
    products re-verify; otherwise an ``incomplete`` result with ``None`` final
    paths and no published ``final/`` directory.
    """

    any_failed = any(
        seg.status in ("alignment_failed", "postprocess_failed", "caption_failed")
        for seg in _to_segment_results(report)
    )

    # A failed segment's cropped length is not known, so it can never be
    # conflated with a cleaned duration; and no partial episode is published.
    if not cleaned or any_failed:
        report.status = STATUS_INCOMPLETE
        _write_report(run_dir, report)
        return RenderResult(
            status=STATUS_INCOMPLETE,
            run_dir=run_dir,
            report_path=run_dir / "report.json",
            wav_path=None,
            srt_path=None,
            timeline_path=None,
            segments=_to_segment_results(report),
            config=report.config,
            completed_stages=list(report.completed_stages),
        )

    # --- integer-frame accumulation (C7) ------------------------------------
    common_rate = cleaned[0].sample_rate
    for seg in cleaned:
        if seg.sample_rate != common_rate:
            raise RenderError(
                f"sample rate mismatch during assembly: "
                f"{seg.segment_id} has {seg.sample_rate} Hz != {common_rate} Hz",
                run_dir=run_dir,
            )

    # Local caption text comes from each segment's own original text; never
    # recombined across input segments.
    original_text = {seg.id: seg.text for seg in script.segments}

    total_frames = 0
    all_samples: list[int] = []
    global_captions: list[Caption] = []
    timeline_segments: list[dict] = []
    running = 0  # O_i: cumulative frames before this segment

    for seg in cleaned:
        n_i = seg.frames
        g_i = _half_up_frames(seg.pause_after_ms, common_rate)
        offset_frames = running  # O_i

        for caption in seg.local_captions:
            global_captions.append(
                Caption(
                    start=caption.start + offset_frames / common_rate,
                    end=caption.end + offset_frames / common_rate,
                    text=caption.text,
                )
            )

        all_samples.extend(seg.samples)
        all_samples.extend([0] * g_i)
        running += n_i + g_i

        timeline_segments.append(
            {
                "segment_id": seg.segment_id,
                "start_frame": offset_frames,
                "audio_frames": n_i,
                "pause_after_frames": g_i,
            }
        )

    total_frames = running

    # --- verify the caption timeline before writing anything ----------------
    _verify_caption_timeline(global_captions, total_frames, common_rate, run_dir)

    # --- verify the integer-frame timeline integrity (C7) -------------------
    _verify_timeline_integrity(
        timeline_segments,
        [seg.segment_id for seg in cleaned],
        total_frames,
        run_dir,
    )

    # --- render SRT (only at the very end, existing half-up rules) ----------
    srt_text = render_srt(global_captions)

    # --- generate final products in a staging dir, then re-read & validate --
    staging = run_dir / ".final.staging"
    staging.mkdir(parents=True, exist_ok=True)
    final_wav = staging / "final.wav"
    final_srt = staging / "final.srt"
    final_timeline = staging / "timeline.json"

    write_wav(final_wav, all_samples, common_rate)
    final_srt.write_text(srt_text, encoding="utf-8", newline="\n")
    final_timeline.write_text(
        json.dumps(
            {
                "sample_rate": common_rate,
                "total_frames": total_frames,
                "start_seconds": 0.0,
                "duration_seconds": total_frames / common_rate,
                "segments": timeline_segments,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _verify_final_products(
        staging,
        common_rate,
        total_frames,
        global_captions,
        [seg.segment_id for seg in cleaned],
    )

    # --- publish final/ products, then complete the run transaction (C6) ---
    # Order is a transaction: publish the artifacts, write the ``complete``
    # report successfully, and only then write the ``.complete`` marker last.
    # A failure at any step leaves no ``.complete`` marker, so a consumer never
    # treats a failed publish/report/marker write as success.
    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    _publish(final_wav, final_dir / "final.wav")
    _publish(final_srt, final_dir / "final.srt")
    _publish(final_timeline, final_dir / "timeline.json")

    report.status = STATUS_COMPLETE
    report.completed_stages.append("assemble")
    report.artifacts = {
        "wav_path": "final/final.wav",
        "srt_path": "final/final.srt",
        "timeline_path": "final/timeline.json",
    }

    # The report must be written successfully before the completion marker; on
    # the success path a report write failure is raised, never swallowed.
    _write_report(run_dir, report, strict=True)

    # The completion marker is the very last step: nothing writes after it.
    (final_dir / ".complete").write_text("complete", encoding="utf-8")

    return RenderResult(
        status=STATUS_COMPLETE,
        run_dir=run_dir,
        report_path=run_dir / "report.json",
        wav_path=final_dir / "final.wav",
        srt_path=final_dir / "final.srt",
        timeline_path=final_dir / "timeline.json",
        segments=_to_segment_results(report),
        config=report.config,
        completed_stages=list(report.completed_stages),
    )


def _publish(source: Path, destination: Path) -> None:
    """Copy a staged final product into ``final/``."""

    destination.write_bytes(source.read_bytes())


# --- timeline helpers -------------------------------------------------------


def _half_up_frames(pause_after_ms: int, sample_rate: int) -> int:
    """Number of silence frames inserted after a segment (half-up)."""

    return int((Decimal(str(pause_after_ms)) * sample_rate / Decimal(1000)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    ))


def _verify_caption_timeline(
    captions: list[Caption], total_frames: int, sample_rate: int, run_dir: Path
) -> None:
    """Validate the whole-episode caption timeline before writing.

    Consecutive, non-empty, start before end, non-negative, non-overlapping
    (touching allowed). Millisecond quantization causing a zero-duration caption
    is an explicit failure. Every emitted boundary must fall inside the final
    WAV, allowing the 0.5 ms SRT rounding tolerance against the precise audio
    edge -- but never accumulated drift.
    """

    if not captions:
        raise RenderError("assembly produced no captions", run_dir=run_dir)

    final_duration = total_frames / sample_rate
    previous_end = -1.0
    for index, caption in enumerate(captions):
        if not caption.text:
            raise RenderError(
                f"caption {index + 1} has empty text after assembly", run_dir=run_dir
            )
        if caption.start < 0 or caption.end < 0:
            raise RenderError(
                f"caption {index + 1} has negative timing", run_dir=run_dir
            )
        if not (caption.start < caption.end):
            raise RenderError(
                f"caption {index + 1} has start >= end", run_dir=run_dir
            )
        # Zero duration from millisecond quantization is an explicit failure.
        start_ms = _caption_ms(caption.start)
        end_ms = _caption_ms(caption.end)
        if end_ms - start_ms <= 0:
            raise RenderError(
                f"caption {index + 1} has zero or negative duration after "
                f"millisecond quantization",
                run_dir=run_dir,
            )
        if caption.start < previous_end - _TIMING_TOLERANCE:
            raise RenderError(
                f"caption {index + 1} overlaps the previous caption", run_dir=run_dir
            )
        previous_end = caption.end
        emitted_end_ms = end_ms
        if emitted_end_ms > final_duration * 1000 + _SRT_MS_TOLERANCE + 1e-6:
            raise RenderError(
                f"caption {index + 1} ends at {emitted_end_ms} ms, past the final "
                f"WAV duration {final_duration * 1000} ms",
                run_dir=run_dir,
            )


def _verify_timeline_integrity(
    timeline_segments: object,
    segment_ids: list[str],
    total_frames: int,
    run_dir: Path,
) -> None:
    """Verify the integer-frame timeline accumulation against the source of truth (C7).

    ``segment_ids`` is the ordered list of input segment ids; ``timeline_segments``
    is the ``segments`` array from ``timeline.json``. Each entry must carry the
    four required integer fields, appear in the input order, and have a
    ``start_frame`` equal to the cumulative ``audio_frames + pause_after_frames``
    of the preceding segments. The accumulated total must equal ``total_frames``.
    """

    if not isinstance(timeline_segments, list):
        raise RenderError(
            "timeline is missing its per-segment entries", run_dir=run_dir
        )
    if len(timeline_segments) != len(segment_ids):
        raise RenderError(
            "timeline segment count does not match the input segments",
            run_dir=run_dir,
        )
    running = 0
    for entry, expected_id in zip(timeline_segments, segment_ids):
        for key in ("segment_id", "start_frame", "audio_frames", "pause_after_frames"):
            if key not in entry:
                raise RenderError(
                    f"timeline segment entry missing {key!r}", run_dir=run_dir
                )
        for key in ("start_frame", "audio_frames", "pause_after_frames"):
            # ``bool`` is an ``int`` subclass; reject it so a frame count can
            # never be silently stored as ``True``/``False``.
            if not isinstance(entry[key], int) or isinstance(entry[key], bool):
                raise RenderError(
                    f"timeline {key} must be an integer frame count",
                    run_dir=run_dir,
                )
        if entry["segment_id"] != expected_id:
            raise RenderError(
                "timeline segment order does not match the input segments",
                run_dir=run_dir,
            )
        if entry["start_frame"] != running:
            raise RenderError(
                f"timeline segment {expected_id!r} start_frame "
                f"{entry['start_frame']} does not match the accumulated "
                f"{running} (integer-frame offset)",
                run_dir=run_dir,
            )
        running += entry["audio_frames"] + entry["pause_after_frames"]
    if running != total_frames:
        raise RenderError(
            f"timeline total_frames {total_frames} does not match the "
            f"accumulated {running}",
            run_dir=run_dir,
        )


def _caption_ms(seconds: float) -> int:
    """Render seconds to an SRT millisecond integer (independent of the renderer)."""

    return int((Decimal(str(seconds)) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _verify_final_products(
    final_dir: Path,
    sample_rate: int,
    total_frames: int,
    captions: list[Caption],
    segment_ids: list[str],
) -> None:
    """Re-read the published final WAV/SRT/timeline and verify them against disk."""

    # --- WAV: frame count / rate / format ---
    samples, rate, frames = read_wav(final_dir / "final.wav")
    if rate != sample_rate or frames != total_frames or len(samples) != frames:
        raise RenderError(
            "final WAV re-read mismatch: "
            f"rate={rate}, frames={frames}, expected {sample_rate}/{total_frames}",
            run_dir=final_dir.parent,
        )

    # --- SRT: actual file content must equal the rendered document ---
    srt_text = (final_dir / "final.srt").read_text(encoding="utf-8")
    if srt_text != render_srt(captions):
        raise RenderError(
            "final SRT re-read content does not match the compiled captions",
            run_dir=final_dir.parent,
        )

    # --- timeline: re-read and confirm the derived integers ---
    try:
        timeline = json.loads((final_dir / "timeline.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RenderError(f"cannot re-read timeline: {exc}", run_dir=final_dir.parent) from exc

    if timeline.get("sample_rate") != sample_rate:
        raise RenderError(
            "timeline sample_rate does not match the common WAV rate",
            run_dir=final_dir.parent,
        )
    if timeline.get("total_frames") != total_frames:
        raise RenderError(
            "timeline total_frames does not match the assembled frame count",
            run_dir=final_dir.parent,
        )
    timeline_segments = timeline.get("segments")
    # Re-verify the integer-frame accumulation against the re-read file (C7).
    _verify_timeline_integrity(
        timeline_segments, segment_ids, total_frames, final_dir.parent
    )
    # Second values, if present, must be derived from the integer frames, never
    # from independently rounded millisecond accumulations.
    if "duration_seconds" in timeline:
        expected_duration = total_frames / sample_rate
        if abs(timeline["duration_seconds"] - expected_duration) > _TIMING_TOLERANCE:
            raise RenderError(
                "timeline duration_seconds is not derived from integer frames",
                run_dir=final_dir.parent,
            )
    if "start_seconds" in timeline and timeline["start_seconds"] != 0.0:
        raise RenderError(
            "timeline start_seconds is not zero",
            run_dir=final_dir.parent,
        )


# --- model stage subprocess construction ------------------------------------


def _make_tts_task(tts_python: str, tts_model: str, device: str) -> TaskCallable:
    """Build the default TTS stage task: one subprocess, one model load.

    The subprocess loads the Qwen3-TTS adapter once, synthesizes every segment
    in order, writes each segment WAV and returns a per-segment result manifest.
    A model/CUDA/subprocess failure is reported as ``runtime_error`` with a
    non-zero exit so the parent stops the run.
    """

    stage = _TTS_STAGE_SCRIPT
    model_path = tts_model

    def task(task_dict: dict) -> dict:
        return _run_model_stage(
            interpreter=tts_python,
            stage_script=stage,
            args=(model_path, task_dict),
        )

    return task


def _make_align_task(alignment_python: str, alignment_model: str, device: str) -> TaskCallable:
    """Build the default Alignment stage task: one subprocess, one model load."""

    stage = _ALIGN_STAGE_SCRIPT
    model_path = alignment_model

    def task(task_dict: dict) -> dict:
        return _run_model_stage(
            interpreter=alignment_python,
            stage_script=stage,
            args=(model_path, task_dict),
        )

    return task


def _run_model_stage(
    interpreter: str,
    stage_script: str,
    args: tuple[str, dict],
) -> dict:
    """Write a stage script, run it once, and decode its per-segment results.

    The model stage is launched exactly once per run. The stage writes the
    per-segment result manifest itself; the parent only reads it.

    The child bootstraps this exact ``media_pipeline`` package by file location
    (C4/C8), regardless of whether the interpreter's venv has another copy
    installed. Its environment and dependency search paths remain the worker's.

    The structured manifest is read before deciding on failure (C5): the child
    writes it and then exits non-zero even on a per-segment fault, so the
    segment id, stage and original reason are preserved rather than replaced by
    a generic exit-code/stderr message. The exit code + stderr are only used as
    a fallback when no usable manifest was produced.
    """

    import tempfile

    result_manifest = {"runtime_error": None, "segments": []}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        task_file = tmp_path / "task.json"
        manifest_file = tmp_path / "results.json"
        task_file.write_text(
            json.dumps({"args": list(args), "manifest": str(manifest_file)}),
            encoding="utf-8",
        )
        # Pin the Controller package itself, without exposing its parent (which
        # may contain unrelated top-level modules) on the worker's import path.
        bootstrap = (
            "import importlib.util, pathlib, sys\n"
            f"_package_dir = pathlib.Path({str(_PACKAGE_DIR)!r})\n"
            "_spec = importlib.util.spec_from_file_location(\n"
            "    'media_pipeline', _package_dir / '__init__.py',\n"
            "    submodule_search_locations=[str(_package_dir)],\n"
            ")\n"
            "_package = importlib.util.module_from_spec(_spec)\n"
            "sys.modules['media_pipeline'] = _package\n"
            "_spec.loader.exec_module(_package)\n"
            + stage_script
        )
        completed = subprocess.run(
            [interpreter, "-c", bootstrap, str(task_file)],
            capture_output=True,
            text=True,
        )

        # Read the structured manifest the child wrote, regardless of exit code.
        _load_stage_manifest(manifest_file, result_manifest)

        # Only when no usable structured result was produced do we fall back to
        # the exit code / stderr. A non-zero exit with an empty manifest is a
        # run-level fault, never an isolatable segment failure.
        if (
            completed.returncode != 0
            and not result_manifest["segments"]
            and not result_manifest["runtime_error"]
        ):
            result_manifest["runtime_error"] = (
                f"stage failed (exit {completed.returncode}):\n"
                f"stderr: {completed.stderr}"
            )

    return result_manifest


def _load_stage_manifest(manifest_file: Path, result_manifest: dict) -> None:
    """Populate ``result_manifest`` from the child's structured manifest (C5).

    Prefer any valid structured ``runtime_error`` / ``segments`` the child
    wrote. If the manifest is missing or corrupt, record a parse error so the
    run stops instead of silently continuing on lost information.
    """

    if not manifest_file.exists():
        return
    try:
        loaded = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result_manifest["runtime_error"] = "cannot parse stage result manifest"
        return
    if isinstance(loaded, dict):
        result_manifest["runtime_error"] = loaded.get("runtime_error")
        result_manifest["segments"] = loaded.get("segments", []) or []


# --- embedded model stage scripts -------------------------------------------

# The stage scripts receive ``task_file`` whose JSON is ``{"args": [model_path,
# task_dict], "manifest": <manifest_path>}``. They load the model exactly once,
# process every segment, write the manifest and exit non-zero only on a
# model-runtime fault. They import the portable production code only.

_TTS_STAGE_SCRIPT = r"""
import json
import sys
from pathlib import Path

_manifest_path = None
_task = None


def _dump(manifest):
    Path(_manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


try:
    task_file = Path(sys.argv[1])
    spec = json.loads(task_file.read_text(encoding="utf-8"))
    _task = spec["args"]
    _manifest_path = spec["manifest"]

    model_path, task_dict = _task[0], _task[1]

    from media_pipeline import CustomVoiceRequest
    from media_pipeline.postprocess import read_wav
    from media_pipeline.tts import TTSRuntimeError

    segments_dir = Path(task_dict["run_dir"]) / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    clone = task_dict.get("voice")
    synthesis_errors = (TTSRuntimeError, OSError)
    try:
        if clone is not None:
            from media_pipeline.voice_clone import VoiceAssetError, VoiceCloneError, VoiceCloneRequest
            from media_pipeline.render import _resolve_clone_asset
            from media_pipeline.runtimes.qwen_voice_clone import Qwen3VoiceCloneTTS, load_voice_asset

            synthesis_errors += (VoiceAssetError, VoiceCloneError)
            asset_path = _resolve_clone_asset(clone["asset"], Path(task_dict["script_dir"]))
            asset = load_voice_asset(asset_path)
            engine = Qwen3VoiceCloneTTS(model_path, device=task_dict.get("device", "cuda:0"))
        else:
            from media_pipeline.runtimes.qwen_tts import Qwen3CustomVoiceTTS
            engine = Qwen3CustomVoiceTTS(model_path, device=task_dict.get("device", "cuda:0"))
    except Exception as exc:
        context = "failed to initialize clone TTS" if clone is not None else "failed to load TTS model"
        _dump({"runtime_error": f"{context}: {exc}", "segments": []})
        sys.exit(3)

    results = []
    for seg in task_dict["segments"]:
        out_wav = segments_dir / (seg["safe_name"] + ".wav")
        try:
            if clone is not None:
                if seg["instruct"] != "":
                    raise VoiceCloneError("non-empty instruct is unsupported for cloned voices")
                engine.synthesize(
                    VoiceCloneRequest(text=seg["text"], language=task_dict["language"]),
                    asset, out_wav,
                )
            else:
                engine.synthesize(
                    CustomVoiceRequest(
                        text=seg["text"],
                        language=task_dict["language"],
                        speaker=task_dict["speaker"],
                        instruct=seg["instruct"],
                    ),
                    out_wav,
                )
        except synthesis_errors as exc:
            _dump({"runtime_error": None, "segments": results + [
                {"segment_id": seg["id"], "status": "synthesis_failed", "reason": str(exc)}
            ]})
            sys.exit(2)
        try:
            samples, rate, frames = read_wav(out_wav)
        except Exception as exc:
            _dump({"runtime_error": None, "segments": results + [
                {"segment_id": seg["id"], "status": "synthesis_failed", "reason": f"invalid WAV: {exc}"}
            ]})
            sys.exit(2)
        if rate <= 0 or frames <= 0 or len(samples) != frames:
            _dump({"runtime_error": None, "segments": results + [
                {"segment_id": seg["id"], "status": "synthesis_failed", "reason": "invalid WAV metadata"}
            ]})
            sys.exit(2)
        results.append({"segment_id": seg["id"], "status": "ok"})

    _dump({"runtime_error": None, "segments": results})
    sys.exit(0)
except SystemExit:
    raise
except Exception as exc:  # pragma: no cover - defensive runtime guard
    if _manifest_path is not None:
        _dump({"runtime_error": f"unexpected TTS runtime failure: {exc}", "segments": []})
    sys.exit(3)
"""

_ALIGN_STAGE_SCRIPT = r"""
import json
import sys
from pathlib import Path

_manifest_path = None
_task = None


def _dump(manifest):
    Path(_manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


try:
    task_file = Path(sys.argv[1])
    spec = json.loads(task_file.read_text(encoding="utf-8"))
    _task = spec["args"]
    _manifest_path = spec["manifest"]

    model_path, task_dict = _task[0], _task[1]

    from media_pipeline.alignment import (
        AlignmentRequest,
        AlignmentRequestError,
        AlignmentRuntimeError,
        fault_is_io_error,
    )
    from media_pipeline.captions import (
        AlignmentError,
        AlignmentMismatchError,
    )
    from media_pipeline.runtimes.qwen_aligner import Qwen3ForcedAlignment

    # Per-segment deterministic validation failures (malformed output, text
    # mismatch, input-WAV format error) are isolatable; model-runtime faults and
    # any input I/O fault (an unreadable/missing input WAV wraps OSError) stop
    # the whole run. ``AlignmentRequestError`` can be either, so inspect it.
    _ISOLATABLE = (AlignmentError, AlignmentMismatchError)
    _IO_OR_REQUEST = (AlignmentRequestError,)

    segments_dir = Path(task_dict["run_dir"]) / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    try:
        engine = Qwen3ForcedAlignment(model_path, device=task_dict.get("device", "cuda:0"))
    except Exception as exc:
        _dump({"runtime_error": f"failed to load alignment model: {exc}", "segments": []})
        sys.exit(3)

    results = []
    for seg in task_dict["segments"]:
        raw_wav = segments_dir / (seg["safe_name"] + ".wav")
        out_align = segments_dir / (seg["safe_name"] + ".alignment.raw.json")
        try:
            engine.align(
                AlignmentRequest(
                    wav_path=raw_wav,
                    text=seg["text"],
                    language=task_dict["language"],
                ),
                out_align,
            )
        except _IO_OR_REQUEST as exc:
            # A request error is an isolatable deterministic validation failure
            # only when it is not caused by a filesystem I/O error.
            if fault_is_io_error(exc):
                _dump({"runtime_error": f"alignment input I/O failure on segment {seg['id']!r}: {exc}",
                       "segments": results})
                sys.exit(2)
            results.append({"segment_id": seg["id"], "status": "alignment_failed", "reason": str(exc)})
            continue
        except _ISOLATABLE as exc:
            results.append({"segment_id": seg["id"], "status": "alignment_failed", "reason": str(exc)})
            continue
        except AlignmentRuntimeError as exc:
            _dump({"runtime_error": f"alignment runtime failure on segment {seg['id']!r}: {exc}",
                   "segments": results})
            sys.exit(2)
        except OSError as exc:
            # Writing THIS segment's alignment output raised an I/O fault
            # (the runtime adapter re-raises OSError from the output write). A
            # run-level filesystem failure: preserve the current segment id and
            # the already-completed prior results, do not isolate it as an
            # alignment_failed, and exit non-zero (C5).
            _dump({"runtime_error": f"alignment output I/O failure on segment {seg['id']!r}: {exc}",
                   "segments": results})
            sys.exit(2)
        except Exception as exc:  # noqa: BLE001 - unclassified run-level failure
            # Any other unclassified failure at the segment boundary (a
            # model/runtime bug) is a run-level fault, not an isolatable
            # validation error. Keep the current segment id, stage/reason and
            # the already-completed prior results so no context is lost, and
            # stop the run (C5).
            _dump({"runtime_error": f"alignment runtime failure on segment {seg['id']!r}: {exc}",
                   "segments": results})
            sys.exit(2)

        results.append({"segment_id": seg["id"], "status": "ok"})

    _dump({"runtime_error": None, "segments": results})
    sys.exit(0)
except SystemExit:
    raise
except Exception as exc:  # pragma: no cover - defensive runtime guard
    if _manifest_path is not None:
        _dump({"runtime_error": f"unexpected alignment runtime failure: {exc}", "segments": []})
    sys.exit(3)
"""
