"""CPU contract tests for the thin speech CLI; no model runtime or CUDA."""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from media_pipeline import cli, render
from media_pipeline.render import RenderError, RenderResult, SegmentResult
from test_render import FakeAlignerEngine, FakeTTSEngine


_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME = {
    "tts_python": "tts interpreter",
    "alignment_python": "alignment interpreter",
    "tts_model": "tts model",
    "alignment_model": "alignment model",
}


@pytest.fixture(autouse=True)
def _clear_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FakeTTSEngine, "loads", 0)
    monkeypatch.setattr(FakeAlignerEngine, "loads", 0)
    for key in _RUNTIME:
        monkeypatch.delenv("MEDIA_PIPELINE_" + key.upper(), raising=False)
    for key in ("MEDIA_ALIGNMENT_PYTHON", "MEDIA_ALIGNMENT_MODEL"):
        monkeypatch.delenv(key, raising=False)


def _write_script(tmp_path: Path, **overrides: object) -> Path:
    payload = {
        "language": "Chinese",
        "speaker": "Uncle_Fu",
        "segments": [{"id": "unit", "text": "欢迎收听。"}],
        **overrides,
    }
    path = tmp_path / "speech.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_clone_script(tmp_path: Path, **overrides: object) -> Path:
    payload = {
        "language": "Chinese",
        "voice": {"type": "clone", "asset": "voices/voice.pt"},
        "segments": [{"id": "unit", "text": "欢迎收听。"}],
        **overrides,
    }
    path = tmp_path / "speech.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _invoke(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, dict]:
    code = cli.main(argv)
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    return code, payload


def _no_call(*args: object, **kwargs: object) -> None:
    pytest.fail("static preflight must not render or start model stages")


def _set_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _RUNTIME.items():
        monkeypatch.setenv("MEDIA_PIPELINE_" + key.upper(), value)


def _complete(run_dir: Path) -> RenderResult:
    return RenderResult(
        status="complete", run_dir=run_dir, report_path=run_dir / "report.json",
        wav_path=run_dir / "final" / "episode.wav",
        srt_path=run_dir / "final" / "episode.srt",
        timeline_path=run_dir / "final" / "timeline.json",
        segments=[SegmentResult("unit", "ok")],
    )


@pytest.mark.parametrize(
    "argv",
    [
        [], ["speech"], ["speech", "validate"], ["speech", "render", "speech.json"],
        ["unknown"], ["speech", "unknown"],
        ["speech", "validate", "speech.json", "--unknown"],
        ["speech", "validate", "speech.json", "--max-segment-chars", "many"],
        ["speech", "validate", "speech.json", "--max-seg", "10"],
        ["speech", "render", "speech.json", "--out", "run"],
        ["speech", "validate", "speech.json", "--output", "run"],
    ],
)
def test_usage_errors_are_json_and_exit_two(argv, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "load_and_validate_script", _no_call)
    code, payload = _invoke(capsys, argv)
    assert code == 2
    assert payload["command"] is None
    assert payload["status"] == "error"
    assert payload["error"]


@pytest.mark.parametrize("argv", [["--help"], ["speech", "--help"], ["speech", "render", "--help"]])
def test_help_is_human_readable_and_does_not_preflight(argv, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "load_and_validate_script", _no_call)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "usage: media-pipeline" in captured.out


def test_validate_plan_preserves_performance_units_and_directions(tmp_path, capsys, monkeypatch) -> None:
    segments = [
        {"id": "inherit", "text": "第一句。第二句。", "pause_after_ms": 123},
        {"id": "clear", "text": "  保留空白 😀。", "instruct": ""},
        {"id": "local", "text": "最后一段。", "instruct": "轻声。", "pause_after_ms": 9},
    ]
    script = _write_script(tmp_path, instruct="自然、清晰。", segments=segments)
    before = script.read_bytes()
    monkeypatch.setattr(cli, "render_speech", _no_call)
    monkeypatch.setattr(render, "_make_tts_task", _no_call)
    monkeypatch.setattr(render, "_make_align_task", _no_call)
    original_validate = cli.load_and_validate_script
    calls = []

    def validate(path, **kwargs):
        calls.append((path, kwargs))
        return original_validate(path, **kwargs)

    monkeypatch.setattr(cli, "load_and_validate_script", validate)
    code, payload = _invoke(capsys, ["speech", "validate", str(script)])
    assert code == 0
    assert payload["command"] == "validate"
    assert payload["status"] == "valid"
    assert payload["script_path"] == str(script.resolve())
    assert calls == [(str(script), {"max_segment_chars": 200})]
    assert payload["plan"] == {
        "language": "Chinese", "speaker": "Uncle_Fu", "max_segment_chars": 200,
        "segments": [
            {
                "order": index + 1, "id": segment["id"], "text": segment["text"],
                "text_length": len(segment["text"]),
                "pause_after_ms": segment.get("pause_after_ms", 0),
                "effective_instruct": segment.get("instruct", "自然、清晰。"),
                "instruct_source": "segment" if "instruct" in segment else "top_level",
            }
            for index, segment in enumerate(segments)
        ],
    }
    assert script.read_bytes() == before
    assert list(tmp_path.iterdir()) == [script]


def test_validate_default_instruct_and_budget_override(tmp_path, capsys) -> None:
    script = _write_script(tmp_path)
    code, payload = _invoke(capsys, ["speech", "validate", str(script), "--max-segment-chars", "6"])
    assert code == 0
    assert payload["plan"]["max_segment_chars"] == 6
    assert payload["plan"]["segments"][0]["effective_instruct"] == ""
    assert payload["plan"]["segments"][0]["instruct_source"] == "top_level"


def test_validate_clone_plan_is_portable_without_opening_asset_or_model_tasks(tmp_path, capsys, monkeypatch) -> None:
    segments = [
        {"id": "inherit", "text": "第一句。", "pause_after_ms": 100},
        {"id": "clear", "text": "第二句。", "instruct": ""},
    ]
    asset = tmp_path / "voices" / "voice.pt"
    asset.parent.mkdir()
    asset.write_bytes(b"deliberately not a serialized voice asset")
    script = _write_clone_script(tmp_path, segments=segments)
    before = script.read_bytes()
    monkeypatch.setattr(cli, "render_speech", _no_call)
    monkeypatch.setattr(render, "_make_tts_task", _no_call)
    monkeypatch.setattr(render, "_make_align_task", _no_call)
    original_open = builtins.open
    original_path_open = Path.open

    def open_without_asset(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file) == asset:
            pytest.fail("static validation must not open or deserialize the voice asset")
        return original_open(file, *args, **kwargs)

    def path_open_without_asset(path, *args, **kwargs):
        if path == asset:
            pytest.fail("static validation must not open or deserialize the voice asset")
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_without_asset)
    monkeypatch.setattr(Path, "open", path_open_without_asset)
    code, payload = _invoke(capsys, ["speech", "validate", str(script)])
    assert code == 0
    assert payload["command"] == "validate"
    assert payload["status"] == "valid"
    assert payload["plan"] == {
        "language": "Chinese",
        "voice": {"type": "clone", "asset": "voices/voice.pt"},
        "max_segment_chars": 200,
        "segments": [
            {
                "order": index + 1, "id": segment["id"], "text": segment["text"],
                "text_length": len(segment["text"]),
                "pause_after_ms": segment.get("pause_after_ms", 0),
                "effective_instruct": "",
                "instruct_source": "segment" if "instruct" in segment else "top_level",
            }
            for index, segment in enumerate(segments)
        ],
    }
    assert "speaker" not in payload["plan"]
    assert script.read_bytes() == before
    assert set(tmp_path.iterdir()) == {script, asset.parent}


@pytest.mark.parametrize("asset", ["voices/missing.pt", "voices\\missing.pt"])
def test_validate_clone_missing_asset_preserves_authored_relative_form(tmp_path, capsys, asset) -> None:
    script = _write_clone_script(tmp_path, voice={"type": "clone", "asset": asset})
    code, payload = _invoke(capsys, ["speech", "validate", str(script)])
    assert code == 0
    assert payload["plan"]["voice"] == {"type": "clone", "asset": asset}
    assert list(tmp_path.iterdir()) == [script]


@pytest.mark.parametrize("command", ["validate", "render"])
@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"speaker": "Uncle_Fu"}, "speaker"),
        ({"voice": None}, "voice"),
        ({"voice": {"type": "builtin", "asset": "voices/voice.pt"}}, "voice"),
        ({"voice": {"type": "clone", "asset": "voices/voice.pt", "extra": True}}, "voice"),
        ({"voice": {"type": "clone", "asset": ""}}, "voice.asset"),
        ({"voice": {"type": "clone", "asset": "/voice.pt"}}, "voice.asset"),
        ({"voice": {"type": "clone", "asset": "C:\\voices\\voice.pt"}}, "voice.asset"),
        ({"voice": {"type": "clone", "asset": "../voice.pt"}}, "voice.asset"),
        ({"instruct": "更激动一点"}, "non-empty instruct is unsupported for cloned voices"),
        ({"segments": [{"id": "unit", "text": "欢迎收听。", "instruct": "强调这一句"}]},
         "non-empty instruct is unsupported for cloned voices"),
    ],
)
def test_clone_static_failures_keep_json_exit_and_no_artifact_contract(tmp_path, capsys, monkeypatch, command, overrides, reason) -> None:
    script = _write_clone_script(tmp_path, **overrides)
    before = script.read_bytes()
    monkeypatch.setattr(cli, "render_speech", _no_call)
    monkeypatch.setattr(render, "_make_tts_task", _no_call)
    monkeypatch.setattr(render, "_make_align_task", _no_call)
    argv = ["speech", command, str(script)]
    if command == "render":
        argv += ["--output", str(tmp_path / "run")]
    code, payload = _invoke(capsys, argv)
    assert code == 3
    assert payload["command"] == command
    assert payload["status"] == "validation_failed"
    assert reason in payload["error"]
    assert script.read_bytes() == before
    assert list(tmp_path.iterdir()) == [script]


@pytest.mark.parametrize("command", ["validate", "render"])
def test_preflight_preserves_aggregated_api_errors_and_input(tmp_path, capsys, monkeypatch, command) -> None:
    script = _write_script(
        tmp_path, language="", instruct=None, extra="typo",
        segments=[
            {"id": "same", "text": "abcdefghijk", "pause_after_ms": -1},
            {"id": "same", "text": "。", "instruct": None},
            None,
        ],
    )
    before = script.read_bytes()
    with pytest.raises(RenderError) as excinfo:
        render.load_and_validate_script(script, max_segment_chars=10)
    monkeypatch.setattr(cli, "render_speech", _no_call)
    monkeypatch.setattr(render, "_make_tts_task", _no_call)
    monkeypatch.setattr(render, "_make_align_task", _no_call)
    argv = ["speech", command, str(script), "--max-segment-chars", "10"]
    if command == "render":
        argv += ["--output", str(tmp_path / "run")]
    code, payload = _invoke(capsys, argv)
    assert code == 3
    assert payload["command"] == command
    assert payload["status"] == "validation_failed"
    assert payload["error"] == str(excinfo.value)
    for reason in (
        "unknown top-level field 'extra'", "language is required", "instruct must be a string",
        "over the limit of 10", "pause_after_ms must not be negative",
        "must contain alignable content", "duplicate segment id 'same'",
        "segment[1].instruct must be a string", "segment[2] must be an object",
    ):
        assert reason in payload["error"]
    assert script.read_bytes() == before
    assert list(tmp_path.iterdir()) == [script]


@pytest.mark.parametrize("kind", ["invalid_json", "missing", "directory", "invalid_utf8", "zero_limit", "negative_limit"])
def test_invalid_input_is_validation_failed_without_runtime_or_artifacts(tmp_path, capsys, monkeypatch, kind) -> None:
    script = _write_script(tmp_path)
    extra = []
    if kind == "invalid_json":
        script.write_text("{broken", encoding="utf-8")
    elif kind == "missing":
        script = tmp_path / "missing.json"
    elif kind == "directory":
        script = tmp_path
    elif kind == "invalid_utf8":
        script.write_bytes(b"\xff")
    elif kind in {"zero_limit", "negative_limit"}:
        extra = ["--max-segment-chars", "0" if kind == "zero_limit" else "-1"]
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    monkeypatch.setattr(cli, "render_speech", _no_call)
    code, payload = _invoke(
        capsys, ["speech", "render", str(script), "--output", str(tmp_path / "run"), *extra],
    )
    assert code == 3
    assert payload["status"] == "validation_failed"
    assert payload["error"]
    assert not (tmp_path / "run").exists()
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before


def test_missing_runtime_is_usage_error_after_successful_preflight(tmp_path, capsys, monkeypatch) -> None:
    script = _write_script(tmp_path)
    monkeypatch.setattr(cli, "render_speech", _no_call)
    code, payload = _invoke(capsys, ["speech", "render", str(script), "--output", str(tmp_path / "run")])
    assert code == 2
    assert payload["command"] == "render"
    assert payload["status"] == "error"
    for key in _RUNTIME:
        assert "MEDIA_PIPELINE_" + key.upper() in payload["error"]
    assert list(tmp_path.iterdir()) == [script]


@pytest.mark.parametrize("mode", ["env", "aliases", "overrides"])
def test_runtime_resolution_and_exact_render_delegation(tmp_path, capsys, monkeypatch, mode) -> None:
    script = _write_script(tmp_path)
    run_dir = tmp_path / "run"
    _set_runtime(monkeypatch)
    expected = dict(_RUNTIME)
    extra = []
    if mode in {"aliases", "overrides"}:
        for key in ("alignment_python", "alignment_model"):
            monkeypatch.setenv("MEDIA_" + key.upper(), "legacy " + key)
            expected[key] = "legacy " + key
    if mode == "overrides":
        expected = {key: "explicit " + key for key in _RUNTIME}
        for key, value in expected.items():
            extra += ["--" + key.replace("_", "-"), value]
        extra += ["--device", "cpu", "--max-segment-chars", "77"]
    calls = []

    def render_speech(*args, **kwargs):
        calls.append((args, kwargs))
        return _complete(run_dir)

    monkeypatch.setattr(cli, "render_speech", render_speech)
    code, payload = _invoke(capsys, ["speech", "render", str(script), "--output", str(run_dir), *extra])
    assert code == 0
    assert payload["status"] == "complete"
    assert calls == [((str(script), str(run_dir)), {
        **expected, "device": "cpu" if mode == "overrides" else "cuda:0",
        "max_segment_chars": 77 if mode == "overrides" else 200,
    })]
    assert payload["run_dir"] == str(run_dir.resolve())
    assert payload["segments"] == [{"segment_id": "unit", "status": "ok", "stage": None, "reason": None}]
    for key in ("report_path", "wav_path", "srt_path", "timeline_path"):
        assert Path(payload[key]).is_absolute()


def test_clone_render_keeps_existing_python_api_delegation(tmp_path, capsys, monkeypatch) -> None:
    script = _write_clone_script(tmp_path)
    run_dir = tmp_path / "run"
    _set_runtime(monkeypatch)
    calls = []

    def render_speech(*args, **kwargs):
        calls.append((args, kwargs))
        return _complete(run_dir)

    monkeypatch.setattr(cli, "render_speech", render_speech)
    code, payload = _invoke(capsys, ["speech", "render", str(script), "--output", str(run_dir)])
    assert code == 0
    assert payload["status"] == "complete"
    assert calls == [((str(script), str(run_dir)), {
        **_RUNTIME, "device": "cuda:0", "max_segment_chars": 200,
    })]


@pytest.mark.parametrize(
    "reason",
    [
        "voice asset does not exist", "malformed voice asset", "unsafe serialization rejected",
        "incompatible checkpoint", "Base model load failed", "clone synthesis failed",
    ],
)
def test_clone_render_failures_preserve_structured_cli_status_and_reason(tmp_path, capsys, monkeypatch, reason) -> None:
    script = _write_clone_script(tmp_path)
    run_dir = tmp_path / "run"
    _set_runtime(monkeypatch)

    def fail(*args, **kwargs):
        run_dir.mkdir()
        (run_dir / "report.json").write_text(json.dumps({"status": "failed", "reason": reason}), encoding="utf-8")
        raise RenderError(reason, run_dir=run_dir)

    monkeypatch.setattr(cli, "render_speech", fail)
    code, payload = _invoke(capsys, ["speech", "render", str(script), "--output", str(run_dir)])
    assert code == 5
    assert payload["command"] == "render"
    assert payload["status"] == "failed"
    assert payload["error"] == reason
    assert payload["run_dir"] == str(run_dir.resolve())
    assert payload["report_path"] == str((run_dir / "report.json").resolve())
    assert all(payload[key] is None for key in ("wav_path", "srt_path", "timeline_path"))


@pytest.mark.parametrize("value", ["", "   "])
@pytest.mark.parametrize("source", ["env", "override"])
def test_empty_runtime_slots_are_rejected(tmp_path, capsys, monkeypatch, value, source) -> None:
    script = _write_script(tmp_path)
    _set_runtime(monkeypatch)
    extra = []
    if source == "env":
        monkeypatch.setenv("MEDIA_PIPELINE_TTS_PYTHON", value)
    else:
        extra = ["--tts-python", value]
    monkeypatch.setattr(cli, "render_speech", _no_call)
    code, payload = _invoke(capsys, ["speech", "render", str(script), "--output", str(tmp_path / "run"), *extra])
    assert code == 2
    assert payload["status"] == "error"
    assert "MEDIA_PIPELINE_TTS_PYTHON" in payload["error"]
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("outcome", ["complete", "incomplete", "failed"])
def test_cli_runs_existing_deterministic_render_with_fake_model_tasks(tmp_path, capsys, monkeypatch, outcome) -> None:
    segments = [{"id": "first", "text": "第一句。", "pause_after_ms": 150}, {"id": "second", "text": "第二句。", "pause_after_ms": 0, "instruct": ""}]
    script = _write_script(tmp_path, instruct="清晰。", segments=segments)
    before = script.read_bytes()
    run_dir = tmp_path / "run"
    _set_runtime(monkeypatch)
    factory_calls = []

    def tts_factory(*args):
        factory_calls.append(("tts", args))
        return FakeTTSEngine()

    def alignment_factory(*args):
        factory_calls.append(("alignment", args))
        return FakeAlignerEngine(fail_ids={"second"} if outcome == "incomplete" else set(), runtime=outcome == "failed")

    monkeypatch.setattr(render, "_make_tts_task", tts_factory)
    monkeypatch.setattr(render, "_make_align_task", alignment_factory)
    code, payload = _invoke(capsys, ["speech", "render", str(script), "--output", str(run_dir), "--device", "cpu"])
    assert code == {"complete": 0, "incomplete": 4, "failed": 5}[outcome]
    assert payload["command"] == "render"
    assert payload["status"] == outcome
    assert factory_calls == [
        ("tts", (_RUNTIME["tts_python"], _RUNTIME["tts_model"], "cpu")),
        ("alignment", (_RUNTIME["alignment_python"], _RUNTIME["alignment_model"], "cpu")),
    ]
    assert script.read_bytes() == before
    assert payload["run_dir"] == str(run_dir.resolve())
    assert payload["report_path"] == str((run_dir / "report.json").resolve())
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == outcome
    assert json.loads((run_dir / "request.json").read_text(encoding="utf-8")) == json.loads(before)
    assert list((run_dir / "segments").glob("*.wav"))
    assert (run_dir / "final" / ".complete").exists() == (outcome == "complete")
    if outcome == "complete":
        for key in ("wav_path", "srt_path", "timeline_path"):
            assert Path(payload[key]).is_file()
        timeline = json.loads(Path(payload["timeline_path"]).read_text(encoding="utf-8"))
        assert timeline
        assert [segment["status"] for segment in payload["segments"]] == ["ok", "ok"]
    else:
        assert all(payload[key] is None for key in ("wav_path", "srt_path", "timeline_path"))
        assert not (run_dir / "final").exists()
        if outcome == "incomplete":
            assert payload["segments"][1] == {
                "segment_id": "second", "status": "alignment_failed",
                "stage": "alignment", "reason": "alignment text does not match",
            }
        else:
            assert "simulated CUDA/model failure" in payload["error"]


@pytest.mark.parametrize("kind", ["existing_run", "input_file", "input_parent"])
def test_output_refusal_uses_existing_api_and_preserves_inputs(tmp_path, capsys, monkeypatch, kind) -> None:
    script = _write_script(tmp_path)
    before = script.read_bytes()
    _set_runtime(monkeypatch)
    monkeypatch.setattr(render, "_make_tts_task", _no_call)
    monkeypatch.setattr(render, "_make_align_task", _no_call)
    if kind == "existing_run":
        output = tmp_path / "run"
        output.mkdir()
        (output / "keep.txt").write_text("existing artifact", encoding="utf-8")
    else:
        output = script if kind == "input_file" else tmp_path
    code, payload = _invoke(capsys, ["speech", "render", str(script), "--output", str(output)])
    assert code == 2
    assert payload["status"] == "error"
    assert "already exists; refusing to overwrite" in payload["error"]
    assert payload["run_dir"] is None
    assert payload["report_path"] is None
    assert script.read_bytes() == before
    if kind == "existing_run":
        assert (output / "keep.txt").read_text(encoding="utf-8") == "existing artifact"
        assert list(output.iterdir()) == [output / "keep.txt"]


def test_run_error_without_report_keeps_recovery_directory(tmp_path, capsys, monkeypatch) -> None:
    script = _write_script(tmp_path)
    run_dir = tmp_path / "partial"
    _set_runtime(monkeypatch)

    def fail(*args, **kwargs):
        raise RenderError("cannot create request", run_dir=run_dir)

    monkeypatch.setattr(cli, "render_speech", fail)
    code, payload = _invoke(capsys, ["speech", "render", str(script), "--output", str(run_dir)])
    assert code == 5
    assert payload["status"] == "failed"
    assert payload["run_dir"] == str(run_dir.resolve())
    assert payload["report_path"] is None


@pytest.mark.parametrize("failure", [OSError("disk unavailable"), UnicodeError("invalid runtime path")])
def test_render_operating_errors_are_json(tmp_path, capsys, monkeypatch, failure) -> None:
    script = _write_script(tmp_path)
    _set_runtime(monkeypatch)

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(cli, "render_speech", fail)
    code, payload = _invoke(capsys, ["speech", "render", str(script), "--output", str(tmp_path / "run")])
    assert code == 2
    assert payload["status"] == "error"
    assert payload["error"] == str(failure)


def _subprocess_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(_ROOT / "src"), "PYTHONIOENCODING": "ascii"}


@pytest.mark.parametrize("voice_source", ["speaker", "clone"])
def test_module_cli_writes_utf8_json_and_imports_no_model_runtime(tmp_path, voice_source) -> None:
    script = (_write_script if voice_source == "speaker" else _write_clone_script)(tmp_path)
    guard = """
import importlib.abc
import sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'qwen_tts', 'qwen_asr'} or fullname.startswith('media_pipeline.runtimes'):
            raise AssertionError('model import during static validation: ' + fullname)
sys.meta_path.insert(0, NoModels())
from media_pipeline.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    before = script.read_bytes()
    for args in (["-m", "media_pipeline.cli"], ["-c", guard]):
        completed = subprocess.run(
            [sys.executable, *args, "speech", "validate", str(script)],
            env=_subprocess_env(), capture_output=True, check=False,
        )
        assert completed.returncode == 0
        assert completed.stderr == b""
        text = completed.stdout.decode("utf-8")
        assert "欢迎收听。" in text
        assert json.loads(text)["status"] == "valid"
    assert script.read_bytes() == before
    assert list(tmp_path.iterdir()) == [script]


def test_escaped_surrogate_json_still_produces_valid_utf8_output(tmp_path) -> None:
    script = tmp_path / "speech.json"
    # JSON permits escaped surrogates, even though they cannot be emitted as UTF-8 bytes.
    script.write_text(json.dumps({"language": "Chinese", "speaker": "Uncle_Fu", "segments": [{"id": "unit", "text": "A\ud800"}]}), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "media_pipeline.cli", "speech", "validate", str(script)],
        env=_subprocess_env(), capture_output=True, check=False,
    )
    assert completed.returncode == 0
    assert completed.stderr == b""
    assert json.loads(completed.stdout.decode("utf-8"))["plan"]["segments"][0]["text"] == "A\ud800"


def test_standard_console_entrypoint_and_legacy_module_help() -> None:
    metadata = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '[project.scripts]\nmedia-pipeline = "media_pipeline.cli:main"' in metadata
    completed = subprocess.run(
        [sys.executable, "-m", "media_pipeline", "--help"],
        env={**_subprocess_env(), "PYTHONIOENCODING": "utf-8"}, capture_output=True, check=False,
    )
    assert completed.returncode == 0
    assert completed.stderr == b""
    help_text = completed.stdout.decode("utf-8")
    assert "Compile original text and forced alignment into SRT captions." in help_text
    assert "text alignment" in help_text
