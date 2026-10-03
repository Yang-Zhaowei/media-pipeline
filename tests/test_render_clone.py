"""CPU coverage for reusable clone assets in the existing speech renderer.

The genuine embedded TTS stage runs in an isolated Python process with its
runtime adapter replaced. Requests, WAV readback, postprocess, captions and
assembly remain production code; no torch, Qwen runtime or checkpoint is used.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from media_pipeline.postprocess import read_wav
from media_pipeline.render import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    RenderError,
    _TTS_STAGE_SCRIPT,
    _run_model_stage,
    load_and_validate_script,
    render_speech,
)


_VOICE = {"type": "clone", "asset": "voices/owner-v1.pt"}


def _payload(**changes: object) -> dict:
    return {
        "language": "Chinese",
        "voice": dict(_VOICE),
        "segments": [
            {"id": "one", "text": "第一段介绍主题。", "pause_after_ms": 100},
            {"id": "two", "text": "第二段补充背景。", "pause_after_ms": 75},
            {"id": "three", "text": "第三段结束讨论。"},
        ],
        **changes,
    }


def _write_script(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "speech.json"
    path.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _validate(tmp_path: Path, data: dict):
    return load_and_validate_script(_write_script(tmp_path, data), max_segment_chars=100)


def test_legacy_speaker_and_instruction_still_validate(tmp_path: Path) -> None:
    data = _payload(speaker="Uncle_Fu", instruct="自然、清晰。")
    del data["voice"]
    data["segments"][1]["instruct"] = "强调重点。"
    validated = _validate(tmp_path, data)
    assert validated.speaker == "Uncle_Fu"
    assert validated.instruct == "自然、清晰。"
    assert validated.segments[1].instruct == "强调重点。"


@pytest.mark.parametrize("source", ["both", "neither"])
def test_exactly_one_voice_source_is_required(tmp_path: Path, source: str) -> None:
    data = _payload()
    if source == "both":
        data["speaker"] = "Uncle_Fu"
    else:
        del data["voice"]
    with pytest.raises(RenderError, match="exactly one") as caught:
        _validate(tmp_path, data)
    assert caught.value.run_dir is None


@pytest.mark.parametrize(
    "voice",
    [
        None,
        "clone",
        [],
        {},
        {"type": "clone"},
        {"asset": "voices/owner.pt"},
        {"type": "builtin", "asset": "voices/owner.pt"},
        {"type": 1, "asset": "voices/owner.pt"},
        {"type": "clone", "asset": "voices/owner.pt", "speaker": "owner"},
        {"type": "clone", "asset": "voices/owner.pt", "provider": "qwen"},
        {"type": "clone", "asset": None},
        {"type": "clone", "asset": 1},
        {"type": "clone", "asset": ""},
        {"type": "clone", "asset": " \n"},
    ],
)
def test_clone_voice_object_is_narrow_and_strict(tmp_path: Path, voice: object) -> None:
    with pytest.raises(RenderError, match="voice"):
        _validate(tmp_path, _payload(voice=voice))


@pytest.mark.parametrize(
    "asset",
    [
        "/tmp/voice.pt",
        "C:/voices/owner.pt",
        "C:\\voices\\owner.pt",
        "C:voices/owner.pt",
        "\\\\server\\share\\owner.pt",
        "//server/share/owner.pt",
        "\\voices\\owner.pt",
        "../owner.pt",
        "voices/../../owner.pt",
        "..\\owner.pt",
        "voices\\..\\..\\owner.pt",
        "voices/owner\x00.pt",
    ],
)
def test_clone_asset_cannot_be_absolute_or_escape_on_either_platform(
    tmp_path: Path, asset: str
) -> None:
    with pytest.raises(RenderError, match="asset") as caught:
        _validate(tmp_path, _payload(voice={"type": "clone", "asset": asset}))
    assert caught.value.run_dir is None


@pytest.mark.parametrize("asset", ["voices/owner.pt", "./voices/owner.pt", "voices/a/../owner.pt"])
def test_clone_path_validation_does_not_require_an_existing_asset(
    tmp_path: Path, asset: str
) -> None:
    assert not (tmp_path / asset).exists()
    validated = _validate(tmp_path, _payload(voice={"type": "clone", "asset": asset}))
    assert validated.language == "Chinese"
    assert [segment.id for segment in validated.segments] == ["one", "two", "three"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["speech.json"]


def test_clone_path_rejects_symlink_escape(tmp_path: Path) -> None:
    script_dir = tmp_path / "project"
    script_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (script_dir / "voices").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this host cannot create directory symlinks: {exc}")
    with pytest.raises(RenderError, match="asset"):
        _validate(script_dir, _payload())


@pytest.mark.parametrize("where", ["omitted", "top", "local", "both"])
def test_clone_allows_only_omitted_or_empty_instructions(tmp_path: Path, where: str) -> None:
    data = _payload()
    if where in {"top", "both"}:
        data["instruct"] = ""
    if where in {"local", "both"}:
        data["segments"][1]["instruct"] = ""
    validated = _validate(tmp_path, data)
    assert validated.instruct == ""
    assert validated.segments[1].instruct == ("" if where in {"local", "both"} else None)


@pytest.mark.parametrize("where", ["top", "local"])
@pytest.mark.parametrize("instruction", ["更激动一点", " \n"])
def test_clone_nonempty_instruction_is_rejected_explicitly_before_runtime(
    tmp_path: Path, where: str, instruction: str
) -> None:
    data = _payload()
    if where == "top":
        data["instruct"] = instruction
    else:
        data["segments"][1]["instruct"] = instruction
    script = _write_script(tmp_path, data)
    original = script.read_bytes()
    calls = []

    def forbidden(task: dict) -> dict:
        calls.append(task)
        raise AssertionError("static preflight must finish before a runtime stage")

    with pytest.raises(
        RenderError, match="non-empty instruct is unsupported for cloned voices"
    ) as caught:
        render_speech(
            script, tmp_path / "run", tts_python="python", alignment_python="python",
            tts_model="base-checkpoint", alignment_model="align-checkpoint",
            max_segment_chars=100, _wav_task=forbidden, _align_task=forbidden,
        )
    assert caught.value.run_dir is None
    assert calls == []
    assert not (tmp_path / "run").exists()
    assert script.read_bytes() == original


def test_clone_nonempty_top_instruction_is_rejected_even_when_all_segments_clear(
    tmp_path: Path,
) -> None:
    data = _payload(instruct="全局指导")
    for segment in data["segments"]:
        segment["instruct"] = ""
    with pytest.raises(
        RenderError, match="non-empty instruct is unsupported for cloned voices"
    ):
        _validate(tmp_path, data)


def test_static_clone_validation_never_imports_or_deserializes_heavy_runtime(
    tmp_path: Path,
) -> None:
    script = _write_script(tmp_path, _payload())
    probe = r'''
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "qwen_tts"}:
        raise AssertionError("heavy dependency imported by static validation: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from media_pipeline.render import load_and_validate_script
load_and_validate_script(sys.argv[1], max_segment_chars=100)
assert "torch" not in sys.modules
assert "qwen_tts" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", probe, str(script)], capture_output=True, text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr


def _subprocess_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    source = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(source) + os.pathsep + env.get("PYTHONPATH", "")
    return env


_FAKE_CLONE_RUNTIME = r'''
import json
import sys
import types
from pathlib import Path
from media_pipeline.postprocess import write_wav
from media_pipeline.tts import TTSArtifact, TTSRuntimeError
from media_pipeline.voice_clone import VoiceAssetError, VoiceCloneError, VoiceCloneRequest

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
run_dir = Path(spec["args"][1]["run_dir"])
run_dir.mkdir(parents=True, exist_ok=True)
failure = __FAILURE__
asset = object()
def event(value):
    with (run_dir / "clone-events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")

def load_voice_asset(path):
    event({"event": "asset_load", "path": str(path)})
    if failure in {"missing", "malformed", "unsafe"}:
        reasons = {
            "missing": "voice asset does not exist",
            "malformed": "voice asset payload has missing or unknown top-level fields",
            "unsafe": "failed to load voice asset safely: rejected serialization",
        }
        raise VoiceAssetError(reasons[failure])
    return asset

class FakeBaseEngine:
    def __init__(self, model_path, *, device="cuda:0"):
        event({"event": "model_load", "model": str(model_path), "device": device})
        if failure == "model":
            raise TTSRuntimeError("failed to load Qwen3-TTS Base model: simulated load fault")
        self.calls = 0

    def synthesize(self, request, prompt, output_path):
        assert isinstance(request, VoiceCloneRequest)
        assert prompt is asset, "all segments must reuse the loaded asset object"
        self.calls += 1
        event({"event": "synthesize", "text": request.text,
               "language": request.language, "call": self.calls})
        if failure == "incompatible":
            raise VoiceAssetError("voice asset is incompatible with this Base checkpoint")
        if self.calls == 2 and failure == "synthesis":
            raise TTSRuntimeError("failed to synthesize cloned utterance: simulated inference fault")
        if self.calls == 2 and failure == "request":
            raise VoiceCloneError("simulated clone request fault")
        if self.calls == 2 and failure == "output":
            Path(output_path).write_bytes(b"not a WAV")
        else:
            frames = 0 if self.calls == 2 and failure == "empty" else 24000
            write_wav(output_path, [1200] * frames, 24000)
        return TTSArtifact(Path(output_path), 24000, 24000)

module = types.ModuleType("media_pipeline.runtimes.qwen_voice_clone")
module.load_voice_asset = load_voice_asset
module.Qwen3VoiceCloneTTS = FakeBaseEngine
sys.modules[module.__name__] = module

custom = types.ModuleType("media_pipeline.runtimes.qwen_tts")
class ForbiddenCustomEngine:
    def __init__(self, *args, **kwargs):
        raise AssertionError("clone scripts must not construct CustomVoice")
custom.Qwen3CustomVoiceTTS = ForbiddenCustomEngine
sys.modules[custom.__name__] = custom
'''


def _clone_task(failure: str = ""):
    bootstrap = _FAKE_CLONE_RUNTIME.replace("__FAILURE__", repr(failure))

    def run(task: dict) -> dict:
        manifest = _run_model_stage(
            sys.executable, bootstrap + "\n" + _TTS_STAGE_SCRIPT, ("base-checkpoint", task)
        )
        (Path(task["run_dir"]) / "clone-stage-result.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        return manifest

    return run


def _align_task(task: dict) -> dict:
    results = []
    for segment in task["segments"]:
        stem = Path(task["run_dir"]) / "segments" / segment["safe_name"]
        _, rate, frames = read_wav(stem.with_name(stem.name + ".wav"))
        spoken = [
            char for char in segment["text"]
            if not (char.isspace() or unicodedata.category(char)[0] in "PZ")
        ]
        duration = frames / rate
        step = (duration - 0.3) / len(spoken)
        tokens = [
            {"text": char, "start": 0.15 + index * step,
             "end": 0.15 + (index + 1) * step}
            for index, char in enumerate(spoken)
        ]
        stem.with_name(stem.name + ".alignment.raw.json").write_text(
            json.dumps(tokens, ensure_ascii=False), encoding="utf-8"
        )
        results.append({"segment_id": segment["id"], "status": "ok"})
    return {"runtime_error": None, "segments": results}


def _render(tmp_path: Path, *, failure: str = "", asset: str = _VOICE["asset"]):
    script = _write_script(tmp_path, _payload(voice={"type": "clone", "asset": asset}))
    return render_speech(
        script, tmp_path / "run", tts_python=sys.executable,
        alignment_python=sys.executable, tts_model="base-checkpoint",
        alignment_model="align-checkpoint", max_segment_chars=100, device="cpu",
        _wav_task=_clone_task(failure), _align_task=_align_task,
    )


def _events(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "clone-events.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_clone_stage_loads_asset_and_base_once_and_runs_existing_downstream(
    tmp_path: Path,
) -> None:
    result = _render(tmp_path)
    assert result.status == STATUS_COMPLETE
    assert result.completed_stages == [
        "preflight", "tts", "alignment", "postprocess", "captions", "assemble"
    ]
    events = _events(result.run_dir)
    assert [event["event"] for event in events] == [
        "asset_load", "model_load", "synthesize", "synthesize", "synthesize"
    ]
    assert Path(events[0]["path"]) == (tmp_path / _VOICE["asset"]).resolve()
    assert events[1] == {"event": "model_load", "model": "base-checkpoint", "device": "cpu"}
    assert [(event["call"], event["text"], event["language"]) for event in events[2:]] == [
        (index, segment["text"], "Chinese")
        for index, segment in enumerate(_payload()["segments"], start=1)
    ]
    assert (result.run_dir / "final" / ".complete").read_text(encoding="utf-8") == "complete"
    assert result.wav_path is not None and result.wav_path.is_file()
    assert result.srt_path is not None and result.srt_path.is_file()
    assert result.timeline_path is not None and result.timeline_path.is_file()
    samples, rate, frames = read_wav(result.wav_path)
    assert rate == 24000 and frames > 0 and len(samples) == frames
    timeline = json.loads(result.timeline_path.read_text(encoding="utf-8"))
    assert timeline["sample_rate"] == rate
    assert timeline["total_frames"] == frames
    assert [segment["segment_id"] for segment in timeline["segments"]] == ["one", "two", "three"]
    assert all(segment.status == "ok" for segment in result.segments)
    srt = result.srt_path.read_text(encoding="utf-8")
    for segment in _payload()["segments"]:
        assert segment["text"].rstrip("。") in srt


@pytest.mark.parametrize("authored", ["./voices/subdir/../owner-v1.pt", "voices\\owner-v1.pt"])
def test_clone_request_and_report_keep_authored_portable_voice_provenance(
    tmp_path: Path, authored: str,
) -> None:
    result = _render(tmp_path, asset=authored)
    request = json.loads((result.run_dir / "request.json").read_text(encoding="utf-8"))
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    voice = {"type": "clone", "asset": authored}
    assert request["voice"] == voice
    assert report["config"]["voice"] == voice
    assert result.config["voice"] == voice
    assert "speaker" not in request
    assert "speaker" not in report["config"]
    assert str(tmp_path) not in json.dumps(request)
    for value in (request["voice"], report["config"]["voice"]):
        assert set(value) == {"type", "asset"}
    assert "ref_code" not in json.dumps(report)
    assert "ref_spk_embedding" not in json.dumps(report)
    assert "ref_text" not in json.dumps(report)


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("missing", "voice asset does not exist"),
        ("malformed", "missing or unknown top-level fields"),
        ("unsafe", "rejected serialization"),
        ("incompatible", "incompatible with this Base checkpoint"),
        ("model", "simulated load fault"),
        ("synthesis", "simulated inference fault"),
        ("request", "simulated clone request fault"),
        ("output", "invalid WAV"),
        ("empty", "invalid WAV"),
    ],
)
def test_clone_failures_preserve_structured_failed_render_and_original_reason(
    tmp_path: Path, failure: str, reason: str
) -> None:
    with pytest.raises(RenderError, match=reason) as caught:
        _render(tmp_path, failure=failure)
    run_dir = tmp_path / "run"
    assert caught.value.run_dir == run_dir
    assert "Traceback" not in str(caught.value)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == STATUS_FAILED
    assert report["config"]["voice"] == _VOICE
    assert "speaker" not in report["config"]
    assert not (run_dir / "final" / ".complete").exists()
    assert not (run_dir / "final").exists()
    assert report["completed_stages"] == ["preflight"]
    manifest = json.loads((run_dir / "clone-stage-result.json").read_text(encoding="utf-8"))
    if failure in {"missing", "malformed", "unsafe", "model"}:
        assert reason in manifest["runtime_error"]
        assert manifest["segments"] == []
    else:
        assert manifest["runtime_error"] is None
        failed = manifest["segments"][-1]
        assert failed["status"] == "synthesis_failed"
        assert reason in failed["reason"]
        assert failed["segment_id"] == ("one" if failure == "incompatible" else "two")
    events = _events(run_dir)
    assert sum(event["event"] == "asset_load" for event in events) == 1
    assert sum(event["event"] == "model_load" for event in events) <= 1
    if failure in {"synthesis", "request", "output", "empty"}:
        assert "two" in str(caught.value)
        assert [event["call"] for event in events if event["event"] == "synthesize"] == [1, 2]
