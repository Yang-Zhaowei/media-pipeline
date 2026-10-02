#!/usr/bin/env python3
"""Issue #8 experiment driver: instruction policy vs generation boundaries.

This is a *thin experiment orchestration layer* over the existing production
``render_speech`` entry point. It deliberately adds nothing to the production
path:

- It never modifies ``src/`` and never imports any model/GPU runtime in this
  process. On ai-core the two model stages load their models inside the
  ``render_speech`` subprocesses, exactly as in production.
- Real execution never substitutes mocks, regression fixtures, or stale audio
  for real model generation. The only model-free mode is ``--dry-run``, which
  validates the inputs and prints the run plan without loading any model.
- It does not add a seed, sampling parameters, per-segment ``instruct``, or any
  other production control. The only variables under test are the generation
  boundaries (short vs long) and the top-level instruction policy (empty vs
  non-empty), which the existing ``render_speech`` schema already carries.
- GPU renders are never run concurrently: every case and every repeat runs one
  at a time, sequentially.

Experiment hypothesis is left deliberately open: empty instruction and long
generation units are **not** assumed to be better. The design instead fixes the
common paragraph-boundary ``pause_after_ms`` so the same physical junction is
compared across schemes, and varies only (a) how many original sentences are
generated in one unit and (b) whether the shared instruction is present.

Frozen source of truth: ``original_manuscript.json``. Run ``make_inputs.py`` to
deterministically regenerate the four ``inputs/A..D.json`` scripts; this module
reads them back so the scripts on disk are the single source of truth.

Environment configuration (all explicit, no hard-coded ai-core paths):

- ``EXPERIMENT_TTS_PYTHON`` / ``EXPERIMENT_ALIGNMENT_PYTHON`` -- stage interpreters
- ``EXPERIMENT_TTS_MODEL`` / ``EXPERIMENT_ALIGNMENT_MODEL`` -- model paths
- ``EXPERIMENT_DEVICE`` -- runtime device (default ``cuda:0``)
- ``EXPERIMENT_MAX_SEGMENT_CHARS`` -- input budget (default from manuscript)
- ``EXPERIMENT_REPEATS`` -- fresh renders per case (default ``3``)
- ``EXPERIMENT_CASES`` -- comma list of cases (default ``A,B,C,D``)
- ``EXPERIMENT_RESULT_ROOT`` -- output root (default ``<repo>/issue8-results``)
- ``EXPERIMENT_MANUSCRIPT`` / ``EXPERIMENT_INPUTS_DIR`` -- input locations
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from media_pipeline import (
    RenderError,
    load_and_validate_script,
    render_speech,
)

# --- case identity ----------------------------------------------------------

#: Case code -> (generation boundary type, instruction policy). Kept explicit so
#: the four scripts differ only along the two axes under test.
CASE_KIND = {
    "A": ("short", "instruct"),
    "B": ("short", "empty"),
    "C": ("long", "instruct"),
    "D": ("long", "empty"),
}

_SCORING_DIMENSIONS = [
    "same_sentence_junction",
    "cross_unit_junction",
    "in_segment_sustained_drift",
    "missed_duplicate_misread",
    "subtitle_sync",
]

_SCORE_LABELS = {
    "same_sentence_junction": "相同句接点",
    "cross_unit_junction": "跨生成单元接点",
    "in_segment_sustained_drift": "段内持续漂移",
    "missed_duplicate_misread": "漏读/重复/错读",
    "subtitle_sync": "字幕同步",
}


# --- deterministic plan from the frozen manuscript --------------------------


def _boundary_indices(paragraphs: list[list[str]]) -> list[int]:
    """0-based index of the sentence that *follows* each common boundary.

    A common boundary sits between two paragraphs. For ``[P1, P2, P3]`` the
    boundaries fall after the last sentence of P1 and of P2, so the following
    sentence indices are returned.
    """

    indices: list[int] = []
    cursor = 0
    for paragraph in paragraphs[:-1]:
        cursor += len(paragraph)
        indices.append(cursor)
    return indices


def _build_segments(paragraphs: list[list[str]], kind: str, common_pause: int) -> list[dict]:
    """Build either one-segment-per-sentence (short) or one-per-paragraph (long)."""

    segments: list[dict] = []
    index = 0
    if kind == "short":
        for paragraph_index, paragraph in enumerate(paragraphs):
            for sentence_index, sentence in enumerate(paragraph):
                is_boundary = (
                    sentence_index == len(paragraph) - 1
                    and paragraph_index < len(paragraphs) - 1
                )
                segments.append(
                    {
                        "id": f"sp-{index + 1:02d}",
                        "text": sentence,
                        "pause_after_ms": common_pause if is_boundary else 0,
                    }
                )
                index += 1
    else:  # long
        for paragraph_index, paragraph in enumerate(paragraphs):
            is_last = paragraph_index == len(paragraphs) - 1
            segments.append(
                {
                    "id": f"pg-{paragraph_index + 1}",
                    "text": "".join(paragraph),
                    "pause_after_ms": 0 if is_last else common_pause,
                }
            )
    return segments


def _script(language: str, speaker: str, instruct: str, segments: list[dict]) -> dict:
    return {
        "language": language,
        "speaker": speaker,
        "instruct": instruct,
        "segments": segments,
    }


def build_case_scripts(manuscript: dict) -> dict[str, dict]:
    """Deterministically derive the four render scripts from the frozen manuscript.

    ``concat_text`` is byte-identical across all four and preserves order, by
    construction: the short plan splits every sentence into its own segment, the
    long plan joins the same sentences paragraph by paragraph, and pauses are
    never part of ``text``.
    """

    paragraphs = manuscript["paragraphs"]
    language = manuscript["language"]
    speaker = manuscript["speaker"]
    original_instruct = manuscript["original_instruct"]
    common_pause = manuscript["common_pause_after_ms"]

    short_segments = _build_segments(paragraphs, "short", common_pause)
    long_segments = _build_segments(paragraphs, "long", common_pause)

    return {
        "A": _script(language, speaker, original_instruct, short_segments),
        "B": _script(language, speaker, "", short_segments),
        "C": _script(language, speaker, original_instruct, long_segments),
        "D": _script(language, speaker, "", long_segments),
    }


def concat_text(segments: list[dict]) -> str:
    return "".join(segment["text"] for segment in segments)


def junction_mapping(manuscript: dict, scripts: dict[str, dict]) -> dict:
    """Map original sentence junctions onto each case's generation units."""

    sentences = [sentence for paragraph in manuscript["paragraphs"] for sentence in paragraph]
    boundaries = _boundary_indices(manuscript["paragraphs"])

    short_units: dict[str, dict] = {}
    for segment in scripts["A"]["segments"]:
        number = int(segment["id"].split("-")[1])
        short_units[segment["id"]] = {
            "sentences": [sentences[number - 1]],
            "boundary_pause_ms": segment["pause_after_ms"],
        }

    long_units: dict[str, dict] = {}
    for segment in scripts["C"]["segments"]:
        paragraph_index = int(segment["id"].split("-")[1]) - 1
        start = sum(len(p) for p in manuscript["paragraphs"][:paragraph_index])
        long_units[segment["id"]] = {
            "sentences": sentences[start : start + len(manuscript["paragraphs"][paragraph_index])],
            "boundary_pause_ms": segment["pause_after_ms"],
        }

    return {
        "sentences": sentences,
        "common_boundaries_after_sentence_index": boundaries,
        "short_units": short_units,
        "long_units": long_units,
    }


def plan_is_consistent(scripts: dict[str, dict], max_segment_chars: int) -> list[str]:
    """Assert the four scripts form a valid, comparable experiment set."""

    problems: list[str] = []
    texts = {case: concat_text(scripts[case]["segments"]) for case in scripts}
    reference = texts["A"]
    for case, text in texts.items():
        if text != reference:
            problems.append(f"{case}: concatenated text differs (len {len(text)})")
    for case, segments in scripts.items():
        for segment in segments["segments"]:
            if len(segment["text"]) > max_segment_chars:
                problems.append(
                    f"{case}/{segment['id']}: text len {len(segment['text'])} "
                    f"exceeds max_segment_chars {max_segment_chars}"
                )
    if len(scripts["C"]["segments"]) < 2:
        problems.append("long plan must have at least two generation units")
    return problems


# --- config ------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def load_config() -> dict:
    repo_root = _repo_root()
    default_result_root = str(repo_root / "issue8-results")
    config = {
        "tts_python": _env("EXPERIMENT_TTS_PYTHON"),
        "alignment_python": _env("EXPERIMENT_ALIGNMENT_PYTHON"),
        "tts_model": _env("EXPERIMENT_TTS_MODEL"),
        "alignment_model": _env("EXPERIMENT_ALIGNMENT_MODEL"),
        "device": _env("EXPERIMENT_DEVICE", "cuda:0"),
        "max_segment_chars": None,
        "repeats": 3,
        "cases": ["A", "B", "C", "D"],
        "result_root": _env("EXPERIMENT_RESULT_ROOT", default_result_root),
        "manuscript_path": _env(
            "EXPERIMENT_MANUSCRIPT", str(repo_root / "validation/issue8/original_manuscript.json")
        ),
        "inputs_dir": _env(
            "EXPERIMENT_INPUTS_DIR", str(repo_root / "validation/issue8/inputs")
        ),
    }
    return config


def resolve_config(argv: list[str]) -> tuple[dict, list[str]]:
    """Parse CLI flags and required environment into ``(config, errors)``.

    A non-empty ``errors`` list means the required GPU-stage configuration is
    missing; the caller must not start any model.
    """

    known = {"A", "B", "C", "D"}
    flag_values: dict[str, str] = {}
    value_flags = {"--repeat", "--case", "--result-root", "--max-chars"}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--dry-run":
            pass
        elif token in value_flags:
            if index + 1 >= len(argv):
                raise SystemExit(f"{token} requires a value")
            flag_values[token] = argv[index + 1]
            index += 1
        elif token.startswith("--"):
            raise SystemExit(f"unknown flag: {token}")
        else:
            raise SystemExit(f"unexpected positional argument: {token}")
        index += 1

    config = load_config()
    errors: list[str] = []

    if "--dry-run" in argv:
        config["dry_run"] = True

    if "--repeat" in flag_values:
        config["repeats"] = int(flag_values["--repeat"])

    if "--case" in flag_values:
        selected = []
        for code in flag_values["--case"].split(","):
            code = code.strip().upper()
            if code and code not in known:
                errors.append(f"unknown case code {code!r}; expected one of {sorted(known)}")
            elif code:
                selected.append(code)
        if selected:
            config["cases"] = selected

    if "--result-root" in flag_values:
        config["result_root"] = flag_values["--result-root"]

    if "--max-chars" in flag_values:
        config["max_segment_chars"] = int(flag_values["--max-chars"])

    for required in ("tts_python", "alignment_python", "tts_model", "alignment_model"):
        if not config[required]:
            errors.append(
                f"missing required environment variable: "
                f"EXPERIMENT_{required.upper()}"
            )

    if config["repeats"] < 1:
        errors.append("repeats must be a positive integer")
    if not config["cases"]:
        errors.append("no valid case selected")

    return config, errors


# --- evidence identity -------------------------------------------------------


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def git_commit(repo_root: Path) -> tuple[str, bool]:
    def _run(args: list[str]) -> str:
        completed = subprocess.run(
            args, capture_output=True, text=True, cwd=str(repo_root)
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return commit, status == ""


def _python_version(interpreter: str) -> str:
    try:
        completed = subprocess.run(
            [interpreter, "--version"], capture_output=True, text=True
        )
        return f"{completed.stdout.strip()} {completed.stderr.strip()}".strip() or "unknown"
    except OSError:
        return "unknown"


def _listening_id(case: str, repeat: int) -> str:
    """Anonymous listening label: no run path, script content, or commit is embedded."""

    material = f"{case}-{repeat}-{uuid.uuid4().hex}"
    return "LST-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:6]


# --- dry run (no model) ------------------------------------------------------


def build_case_plan(config: dict) -> tuple[dict[str, dict], str]:
    """Read and preflight the four case scripts. Returns (scripts, error)."""

    manuscript_path = Path(config["manuscript_path"])
    manuscript = json.loads(manuscript_path.read_text(encoding="utf-8"))
    max_chars = config["max_segment_chars"] or manuscript["max_segment_chars"]

    scripts = build_case_scripts(manuscript)

    # Persist the derived scripts so the on-disk inputs are the single source.
    inputs_dir = Path(config["inputs_dir"])
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for code, script in scripts.items():
        (inputs_dir / f"{code}.json").write_text(
            json.dumps(script, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    errors: list[str] = []
    for code in config["cases"]:
        try:
            load_and_validate_script(
                inputs_dir / f"{code}.json", max_segment_chars=max_chars
            )
        except RenderError as exc:
            errors.append(f"{code}: {exc}")
    summary_errors = "\n".join(errors) if errors else ""
    return scripts, summary_errors


def dry_run(config: dict) -> int:
    """Validate every input and print the run plan without loading any model."""

    scripts, preflight_errors = build_case_plan(config)

    print("Issue #8 experiment -- DRY RUN (no model loaded)")
    print("=" * 60)
    if preflight_errors:
        print("PREFLIGHT FAILED -- no model would run:")
        print(preflight_errors)
        return 2

    max_chars = config["max_segment_chars"] or 0
    problems = plan_is_consistent(scripts, max_chars or 10**9)

    reference = concat_text(scripts["A"]["segments"])
    print(f"shared text length : {len(reference)} code points")
    print(f"shared text SHA-256: {hashlib.sha256(reference.encode('utf-8')).hexdigest()}")
    print(f"max_segment_chars  : {max_chars or '(from manuscript)'}")
    print()

    for code in config["cases"]:
        kind, policy = CASE_KIND[code]
        segments = scripts[code]["segments"]
        instruct = scripts[code]["instruct"]
        has_instruct = bool(instruct.strip())
        common_pauses = sorted({s["pause_after_ms"] for s in segments if s["pause_after_ms"]})
        print(f"case {code}: boundary={kind:5s} instruct={'present' if has_instruct else 'empty':7s} units={len(segments)} pauses={common_pauses}")
        for segment in segments:
            print(f"    {segment['id']:6s} pause={segment['pause_after_ms']:4d}  {segment['text']}")
        print()

    if problems:
        print("CONSISTENCY PROBLEMS:")
        for problem in problems:
            print("  - " + problem)
        return 2

    root = Path(config["result_root"])
    print("RUN PLAN")
    print(f"  result root      : {root}")
    for code in config["cases"]:
        for repeat in range(1, config["repeats"] + 1):
            print(f"  {code}/{repeat:02d} -> {root / code}-{repeat:02d}  (fresh, refuse-if-exists)")
    print()
    print("This dry run did not import any model/GPU runtime and created no output directory.")
    return 0


# --- real execution ----------------------------------------------------------


def _fresh_output_dir(root: Path, case: str, repeat: int) -> Path:
    """A never-used output directory name; ``render_speech`` still refuses it if present."""

    return Path(root) / f"{case}-{repeat:02d}"


def collect_success_evidence(result, run_dir: Path, listening_id: str) -> dict:
    """Per-unit frame counts, whole-episode duration and artifact SHA-256."""

    timeline = json.loads((run_dir / "final" / "timeline.json").read_text(encoding="utf-8"))
    per_unit = [
        {
            "unit": entry["segment_id"],
            "start_frame": entry["start_frame"],
            "audio_frames": entry["audio_frames"],
            "pause_after_frames": entry["pause_after_frames"],
        }
        for entry in timeline["segments"]
    ]
    total_frames = timeline["total_frames"]
    duration = total_frames / timeline["sample_rate"]

    artifacts = {}
    for relative in (
        "final/final.wav",
        "final/final.srt",
        "final/timeline.json",
        "report.json",
        "request.json",
    ):
        path = run_dir / relative
        digest = _sha256_file(path)
        if digest is not None:
            artifacts[relative] = digest

    # Original (raw) and cleaned WAVs per generation unit.
    wav_dir = run_dir / "segments"
    for wav in sorted(wav_dir.glob("*.wav")):
        artifacts[f"segments/{wav.name}"] = _sha256_file(wav)

    return {
        "listening_id": listening_id,
        "status": result.status,
        "output_dir": str(run_dir),
        "per_unit_frames": per_unit,
        "total_frames": total_frames,
        "duration_seconds": duration,
        "artifacts": artifacts,
    }


def collect_failure_evidence(run_dir: Path, error: Exception) -> dict:
    """Keep every failure visible: record the attempted dir and any partial artifacts."""

    partial = None
    if run_dir and Path(run_dir).exists():
        partial = sorted(p.name for p in Path(run_dir).iterdir())
    return {
        "output_dir": str(run_dir) if run_dir else None,
        "status": "failed",
        "error": type(error).__name__ + ": " + str(error),
        "partial_artifacts": partial,
    }


def execute(config: dict) -> int:
    """Run each selected case/repeat sequentially and record every outcome."""

    scripts, preflight_errors = build_case_plan(config)
    if preflight_errors:
        print("PREFLIGHT FAILED -- aborting before any model load:\n" + preflight_errors)
        return 2

    repo_root = _repo_root()
    commit, clean = git_commit(repo_root)
    root = Path(config["result_root"])
    root.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_id": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "clean_tree": clean,
        "media_pipeline_src": str(Path(__file__).resolve().parent.parent.parent.parent
                                  / "src" / "media_pipeline" / "__init__.py"),
        "interpreters": {
            "tts_python": config["tts_python"],
            "alignment_python": config["alignment_python"],
            "tts_python_version": _python_version(config["tts_python"]),
            "alignment_python_version": _python_version(config["alignment_python"]),
        },
        "models": {
            "tts_model": config["tts_model"],
            "alignment_model": config["alignment_model"],
            "device": config["device"],
        },
        "config": {
            "max_segment_chars": config["max_segment_chars"]
            or json.loads(Path(config["manuscript_path"]).read_text(encoding="utf-8"))["max_segment_chars"],
            "repeats": config["repeats"],
            "cases": config["cases"],
            "result_root": str(root),
        },
        "runs": [],
        "failures": [],
    }

    listening_ids: dict[tuple[str, int], str] = {}
    code_ok: dict[str, int] = {code: 0 for code in config["cases"]}
    code_fail: dict[str, int] = {code: 0 for code in config["cases"]}

    for code in config["cases"]:
        for repeat in range(1, config["repeats"] + 1):
            run_dir = _fresh_output_dir(root, code, repeat)
            listening_id = _listening_id(code, repeat)
            listening_ids[(code, repeat)] = listening_id

            print(f"[{code}/{repeat}] -> {run_dir}  (instruct: "
                  f"{'present' if CASE_KIND[code][1] == 'instruct' else 'empty'})")
            try:
                result = render_speech(
                    Path(config["inputs_dir"]) / f"{code}.json",
                    run_dir,
                    tts_python=config["tts_python"],
                    alignment_python=config["alignment_python"],
                    tts_model=config["tts_model"],
                    alignment_model=config["alignment_model"],
                    max_segment_chars=summary["config"]["max_segment_chars"],
                    device=config["device"],
                )
                evidence = collect_success_evidence(result, run_dir, listening_id)
                summary["runs"].append(evidence)
                code_ok[code] += 1
                print(f"    status={evidence['status']} "
                      f"duration={evidence['duration_seconds']:.3f}s "
                      f"listening_id={listening_id}")
            except Exception as exc:  # noqa: BLE001 - record, do not hide, do not retry
                # ``run_dir`` is always defined in the loop above.
                evidence = collect_failure_evidence(run_dir, exc)
                evidence["listening_id"] = listening_id
                summary["runs"].append(evidence)
                summary["failures"].append({"case": code, "repeat": repeat, "error": evidence["error"]})
                code_fail[code] += 1
                print(f"    FAILED: {evidence['error']}")

    # --- persist experiment evidence ----------------------------------------

    scheme_mapping = {
        "cases": {
            code: {
                "boundary": CASE_KIND[code][0],
                "instruct_policy": CASE_KIND[code][1],
                "unit_count": len(scripts[code]["segments"]),
            }
            for code in config["cases"]
        },
        "junction_mapping": junction_mapping(
            json.loads(Path(config["manuscript_path"]).read_text(encoding="utf-8")), scripts
        ),
    }
    (root / "scheme_mapping.json").write_text(
        json.dumps(scheme_mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "listening_ids.json").write_text(
        json.dumps(
            {f"{case}-{repeat}": listening_ids[(case, repeat)] for (case, repeat) in listening_ids},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary["summary"] = {
        "total": len(config["cases"]) * config["repeats"],
        "ok": sum(code_ok.values()),
        "failed": sum(code_fail.values()),
        "per_case": {
            code: {"ok": code_ok[code], "failed": code_fail[code]}
            for code in config["cases"]
        },
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    _write_scorecard(root, listening_ids, config["cases"])

    print()
    print("SUMMARY: " + json.dumps(summary["summary"], ensure_ascii=False))
    print("evidence written to: " + str(root))

    return 0 if not summary["failures"] else 1


def _write_scorecard(root: Path, listening_ids: dict[tuple[str, int], str], cases: list[str]) -> None:
    """A human scoring sheet with all score cells left blank (pending).

    Rows are anonymous listening items; columns cover every required dimension.
    No automatic PASS is filled in.
    """

    import csv

    root = Path(root)
    with (root / "scorecard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["listening_id", "case", "repeat", "boundary", "instruct"]
            + list(_SCORING_DIMENSIONS)
        )
        writer.writerow(["", "", "", "", "scale: 0连续自然 1轻微变化可交付 2需重生成 3不可用"] + [""] * len(_SCORING_DIMENSIONS))
        for (case, repeat), listening_id in sorted(listening_ids.items()):
            writer.writerow(
                [listening_id, case, repeat] + list(CASE_KIND[case])
                + [""] * len(_SCORING_DIMENSIONS)
            )

    header = "| listening_id | case | repeat | boundary | instruct |"
    labels = [f"{_SCORE_LABELS[d]} ({d})" for d in _SCORING_DIMENSIONS]
    blank_cell = "|".join("")
    with (root / "scorecard.md").open("w", encoding="utf-8") as handle:
        handle.write("# Issue #8 human listening scorecard\n\n")
        handle.write("Scale: 0 = continuous/natural, 1 = minor change, deliverable, "
                     "2 = needs regeneration, 3 = unusable.\n")
        handle.write("All score cells are intentionally left blank for manual completion. "
                     "No automatic PASS is recorded.\n\n")
        handle.write(header + " |" + " |".join(labels) + " |\n")
        handle.write("|" + "|".join("---" for _ in list(range(len(labels) + 5))) + "|\n")
        for (case, repeat), listening_id in sorted(listening_ids.items()):
            row = [listening_id, case, str(repeat), CASE_KIND[case][0], CASE_KIND[case][1]]
            handle.write("|" + "|".join(row) + "|" + "|".join([""] * len(labels)) + "|\n")


# --- entry point -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    # The experiment prints Chinese narration; reconfigure the console to UTF-8
    # so dry-run plans and summaries render on Windows without a locale change.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    config, errors = resolve_config(argv)
    if errors:
        print("Issue #8 experiment -- configuration error")
        for error in errors:
            print("  - " + error)
        print("\nSet EXPERIMENT_TTS_PYTHON, EXPERIMENT_ALIGNMENT_PYTHON, "
              "EXPERIMENT_TTS_MODEL, EXPERIMENT_ALIGNMENT_MODEL.")
        return 2

    if config.get("dry_run"):
        return dry_run(config)
    return execute(config)


if __name__ == "__main__":
    raise SystemExit(main())
