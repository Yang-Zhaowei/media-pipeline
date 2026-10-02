"""CPU tests for the Issue #8 experiment driver (no models, no CUDA).

These tests never import ``torch``, ``qwen_tts``, ``qwen_asr`` and never run any
GPU runtime. They exercise the *orchestration and credibility boundaries* of
``validation/issue8/run.py``:

- the four derived scripts form a consistent, comparable experiment set;
- ``--dry-run`` validates every input and prints the plan without loading a
  model and without creating any output directory;
- ``execute`` never overwrites an existing run, keeps every failure visible,
  uses a fresh output directory per case/repeat, and returns a non-zero exit
  code when any run fails;
- the driver never runs GPU renders concurrently.

Model generation is replaced by a lightweight fake ``render_speech`` that only
writes the ``final/timeline.json`` the evidence collector reads. The real
``render_speech`` production path is already covered by ``tests/test_render.py``;
this suite only verifies the thin experiment layer on top of it.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

_EIGHTY_EIGHT_DIR = Path(__file__).resolve().parent.parent / "validation" / "issue8"
if str(_EIGHTY_EIGHT_DIR) not in sys.path:
    sys.path.insert(0, str(_EIGHTY_EIGHT_DIR))

import run as experiment  # noqa: E402

from media_pipeline import RenderError  # noqa: E402

SAMPLE_RATE = 24_000


# --- fixtures ---------------------------------------------------------------


@pytest.fixture()
def manuscript(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parent.parent / "validation" / "issue8" / "original_manuscript.json"
    target = tmp_path / "original_manuscript.json"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _config(
    tmp_path: Path,
    *,
    repeats: int = 2,
    cases: list[str] | None = None,
    max_segment_chars: int | None = None,
    manuscript_path: Path,
) -> dict:
    return {
        "tts_python": "python",
        "alignment_python": "python",
        "tts_model": "/m",
        "alignment_model": "/m",
        "device": "cpu",
        "max_segment_chars": max_segment_chars,
        "repeats": repeats,
        "cases": cases or ["A", "B", "C", "D"],
        "result_root": str(tmp_path / "out"),
        "manuscript_path": str(manuscript_path),
        "inputs_dir": str(tmp_path / "inputs"),
    }


class RenderSpeechFake:
    """Stand-in for ``render_speech`` used to test the orchestration only."""

    def __init__(self, *, mode: str = "success", refuse_existing: bool = False) -> None:
        self.mode = mode
        self.refuse_existing = refuse_existing
        self.calls: list[str] = []

    def __call__(self, script_path, output_dir, **kwargs):  # noqa: ARG002
        self.calls.append(str(output_dir))
        target = Path(output_dir)
        if self.refuse_existing and target.exists():
            raise RenderError(f"output directory {target} already exists")
        if self.mode == "fail":
            raise RenderError("simulated model failure")
        segments = json.loads(Path(script_path).read_text(encoding="utf-8"))["segments"]
        (target / "final").mkdir(parents=True, exist_ok=True)
        timeline = {
            "sample_rate": SAMPLE_RATE,
            "total_frames": SAMPLE_RATE * len(segments),
            "duration_seconds": float(len(segments)),
            "segments": [
                {
                    "segment_id": segment["id"],
                    "start_frame": SAMPLE_RATE * index,
                    "audio_frames": SAMPLE_RATE,
                    "pause_after_frames": 0,
                }
                for index, segment in enumerate(segments)
            ],
        }
        (target / "final" / "timeline.json").write_text(
            json.dumps(timeline), encoding="utf-8"
        )

        class _Result:
            status = "complete"
            run_dir = target

        return _Result()


# --- input consistency ------------------------------------------------------


def test_all_four_cases_concatenate_to_identical_text(manuscript: Path) -> None:
    script = experiment.build_case_scripts(json.loads(manuscript.read_text(encoding="utf-8")))
    texts = {code: experiment.concat_text(script[code]["segments"]) for code in script}
    reference = texts["A"]
    for code, text in texts.items():
        assert text == reference, f"{code} concatenated text differs"
    # Order is preserved: every case starts and ends with the same sentences.
    assert reference.startswith("冬天到了")
    assert reference.endswith("打破夜色。")


def test_long_plan_has_at_least_two_units_and_short_has_more(manuscript: Path) -> None:
    script = experiment.build_case_scripts(json.loads(manuscript.read_text(encoding="utf-8")))
    assert len(script["C"]["segments"]) >= 2
    assert len(script["D"]["segments"]) >= 2
    assert len(script["A"]["segments"]) == len(script["B"]["segments"])
    assert len(script["A"]["segments"]) > len(script["C"]["segments"])


def test_common_boundary_pause_fixed_and_internal_short_pause_zero(manuscript: Path) -> None:
    script = experiment.build_case_scripts(json.loads(manuscript.read_text(encoding="utf-8")))
    common_pause = 600

    # Short plan: only the paragraph-boundary sentences (3rd and 6th) carry the
    # common pause; every internal sentence junction pause is 0.
    short_pauses = [seg["pause_after_ms"] for seg in script["A"]["segments"]]
    assert short_pauses == [0, 0, common_pause, 0, 0, common_pause, 0, 0, 0]

    # Long plan: only the paragraph boundary (after pg-1 and pg-2) carries the
    # common pause; the final paragraph tail pause is 0.
    long_pauses = [seg["pause_after_ms"] for seg in script["C"]["segments"]]
    assert long_pauses == [common_pause, common_pause, 0]


def test_instruct_policy_is_the_only_instruction_difference(manuscript: Path) -> None:
    script = experiment.build_case_scripts(json.loads(manuscript.read_text(encoding="utf-8")))
    original = script["A"]["instruct"]
    assert original.strip()
    assert script["B"]["instruct"] == ""
    assert script["C"]["instruct"] == original
    assert script["D"]["instruct"] == ""
    # A and C share the same segments; B and D share the same segments.
    assert [s["id"] for s in script["A"]["segments"]] == [s["id"] for s in script["B"]["segments"]]
    assert [s["id"] for s in script["C"]["segments"]] == [s["id"] for s in script["D"]["segments"]]


def test_junction_mapping_aligns_common_boundaries(manuscript: Path) -> None:
    manuscript_data = json.loads(manuscript.read_text(encoding="utf-8"))
    script = experiment.build_case_scripts(manuscript_data)
    mapping = experiment.junction_mapping(manuscript_data, script)

    # First common boundary sits after sentence index 2 (0-based) in both plans.
    assert mapping["common_boundaries_after_sentence_index"] == [3, 6]
    # Short unit sp-03 ends the first paragraph; long unit pg-1 spans it too.
    short_first_boundary = mapping["short_units"]["sp-03"]
    long_first_paragraph = mapping["long_units"]["pg-1"]
    assert short_first_boundary["boundary_pause_ms"] == long_first_paragraph["boundary_pause_ms"]
    assert short_first_boundary["boundary_pause_ms"] == 600
    # The long unit spans the whole first paragraph; its last sentence is the
    # boundary sentence shared with the short plan's sp-03.
    assert long_first_paragraph["sentences"][-1] == "孩子们在岸边扔下石子，看着冰面泛起涟漪。"


def test_plan_is_consistent_under_max_chars(manuscript: Path) -> None:
    manuscript_data = json.loads(manuscript.read_text(encoding="utf-8"))
    script = experiment.build_case_scripts(manuscript_data)
    assert experiment.plan_is_consistent(script, manuscript_data["max_segment_chars"]) == []


def test_plan_detects_over_max_segments(manuscript: Path) -> None:
    manuscript_data = json.loads(manuscript.read_text(encoding="utf-8"))
    script = experiment.build_case_scripts(manuscript_data)
    problems = experiment.plan_is_consistent(script, 10)
    assert problems, "a 10-char budget must reject the multi-sentence long units"


# --- dry run (no model) -----------------------------------------------------


def test_dry_run_does_not_load_model_or_create_output(
    tmp_path: Path, manuscript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args, **kwargs):  # noqa: ARG005
        raise AssertionError("dry run must not call render_speech")

    monkeypatch.setattr(experiment, "render_speech", boom)
    config = _config(tmp_path, manuscript_path=manuscript, cases=["A", "C"])
    # Model env points at nonsense; a real model load would fail, so proving it
    # is never called is what makes this a model-free check.
    assert experiment.dry_run(config) == 0
    assert not Path(config["result_root"]).exists()
    # The dry run must not import any model/GPU runtime in this process.
    for module in (
        "torch",
        "media_pipeline.runtimes.qwen_tts",
        "media_pipeline.runtimes.qwen_aligner",
    ):
        assert module not in sys.modules, f"dry run imported {module}"
    # The plan names fresh, non-existent output directories.
    assert not (Path(config["result_root"]) / "A-01").exists()


def test_dry_run_reports_preflight_failure(tmp_path: Path, manuscript: Path) -> None:
    config = _config(tmp_path, manuscript_path=manuscript, max_segment_chars=5)
    # A 5-char budget makes the multi-sentence segments fail preflight, so the
    # dry run must abort before any model would run and report the problem.
    assert experiment.dry_run(config) == 2


# --- execute: credibility boundaries ----------------------------------------


def test_execute_success_records_evidence_and_exits_zero(
    tmp_path: Path, manuscript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = RenderSpeechFake()
    monkeypatch.setattr(experiment, "render_speech", fake)
    config = _config(tmp_path, manuscript_path=manuscript, repeats=2, cases=["A", "B", "C"])

    exit_code = experiment.execute(config)
    assert exit_code == 0

    root = Path(config["result_root"])
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["ok"] == 6
    assert summary["summary"]["failed"] == 0

    # Every run recorded its per-unit frame counts and a duration.
    for run in summary["runs"]:
        assert run["status"] == "complete"
        assert run["per_unit_frames"]
        assert run["duration_seconds"] > 0
        assert run["artifacts"]["final/timeline.json"]

    # The anonymous listening identity, scorecard and scheme mapping exist.
    assert (root / "listening_ids.json").is_file()
    assert (root / "scorecard.csv").is_file()
    assert (root / "scorecard.md").is_file()
    scheme = json.loads((root / "scheme_mapping.json").read_text(encoding="utf-8"))
    assert scheme["cases"]["A"]["boundary"] == "short"
    assert scheme["cases"]["C"]["boundary"] == "long"


def test_execute_failure_is_recorded_and_exits_nonzero(
    tmp_path: Path, manuscript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment, "render_speech", RenderSpeechFake(mode="fail"))
    config = _config(tmp_path, manuscript_path=manuscript, repeats=1, cases=["A", "B"])

    exit_code = experiment.execute(config)
    assert exit_code == 1

    summary = json.loads((Path(config["result_root"]) / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["ok"] == 0
    assert summary["summary"]["failed"] == 2
    # Each failure stays visible in the summary with its case and repeat.
    recorded = {(f["case"], f["repeat"]) for f in summary["failures"]}
    assert recorded == {("A", 1), ("B", 1)}
    for run in summary["runs"]:
        assert run["status"] == "failed"
        assert run["error"]


def test_execute_keeps_partial_run_when_mixed_success_and_failure(
    tmp_path: Path, manuscript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HalfFake:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self, script_path, output_dir, **kwargs):  # noqa: ARG002
            self.n += 1
            if self.n == 1:
                raise RenderError("first run fails")
            target = Path(output_dir)
            (target / "final").mkdir(parents=True, exist_ok=True)
            (target / "final" / "timeline.json").write_text('{"sample_rate":24000,"total_frames":24000,"duration_seconds":1.0,"segments":[]}')

            class _Result:
                status = "complete"
                run_dir = target

            return _Result()

    monkeypatch.setattr(experiment, "render_speech", HalfFake())
    config = _config(tmp_path, manuscript_path=manuscript, repeats=1, cases=["A", "B"])
    exit_code = experiment.execute(config)
    assert exit_code == 1
    summary = json.loads((Path(config["result_root"]) / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["ok"] == 1
    assert summary["summary"]["failed"] == 1


def test_execute_refuses_existing_directory(
    tmp_path: Path, manuscript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pre-create the output directory ``render_speech`` would otherwise refuse,
    # then run with a fake that honours the production refuse-if-exists guard.
    config = _config(tmp_path, manuscript_path=manuscript, repeats=1, cases=["A"])
    existing = Path(config["result_root"]) / "A-01"
    existing.mkdir(parents=True)
    (existing / "request.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(experiment, "render_speech", RenderSpeechFake(refuse_existing=True))
    exit_code = experiment.execute(config)
    assert exit_code == 1

    summary = json.loads((Path(config["result_root"]) / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["failed"] == 1
    assert "already exists" in summary["failures"][0]["error"]
    # The pre-existing directory was not overwritten or removed.
    assert (existing / "request.json").read_text(encoding="utf-8") == "{}"


def test_execute_uses_fresh_output_dir_each_repeat(
    tmp_path: Path, manuscript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = RenderSpeechFake()
    monkeypatch.setattr(experiment, "render_speech", fake)
    config = _config(tmp_path, manuscript_path=manuscript, repeats=3, cases=["A"])

    experiment.execute(config)
    assert len(fake.calls) == 3
    assert len(set(fake.calls)) == 3, "each repeat must use a distinct output directory"


def test_execute_writes_blank_scorecard(
    tmp_path: Path, manuscript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment, "render_speech", RenderSpeechFake())
    config = _config(tmp_path, manuscript_path=manuscript, repeats=1, cases=["A"])
    experiment.execute(config)

    with (Path(config["result_root"]) / "scorecard.csv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    assert header[0] == "listening_id"
    # Row 1 carries the scale note; data rows start at index 2.
    assert any("scale" in cell for cell in rows[1])
    for row in rows[2:]:
        score_cells = row[len(header) - len(experiment._SCORING_DIMENSIONS):]
        assert all(cell == "" for cell in score_cells)

    text = (Path(config["result_root"]) / "listening_ids.json").read_text(encoding="utf-8")
    ids = json.loads(text)
    # The scorecard rows use the anonymous listening labels, not run paths.
    assert rows[2][0] in ids.values()


def test_driver_never_runs_concurrently(tmp_path: Path) -> None:
    # The driver must not launch GPU renders in parallel: no threading or
    # multiprocessing anywhere in the module.
    source = Path(str(experiment.__file__)).read_text(encoding="utf-8")
    assert "multiprocessing" not in source
    assert "threading" not in source


# --- dry run integration end to end (CPU, fake model) -----------------------


def test_main_dry_run_flag_and_config_error() -> None:
    # No GPU-stage env set -> resolve_config reports the missing variables.
    config, errors = experiment.resolve_config(["--dry-run", "--repeat", "1"])
    assert config["dry_run"] is True
    assert errors, "missing GPU-stage configuration must be reported"
