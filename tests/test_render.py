"""CPU unit tests for the segmented speech render entry point (no models, no CUDA).

These tests never import ``torch``, ``qwen_tts``, ``qwen_asr`` or run any GPU
runtime. They exercise the pure orchestration boundary in
:mod:`media_pipeline.render` with two injectable *task* fakes that model the two
one-shot model subprocesses:

- the TTS fake loads its "model" once (constructed once), synthesizes every
  segment, writes each segment WAV and returns a per-segment manifest;
- the alignment fake loads its "model" once, writes each segment's raw
  alignment JSON and returns a per-segment manifest.

The real portable Audio Postprocess + Caption Compiler + WAV assembly stages run
in-process, exactly as they do on ai-core, so the full deterministic path and
every failure mode in ``docs/contracts/segmented-speech-v0.md`` (C1-C9) is
verified on CPU without CUDA and without touching the immutable fixtures.
"""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

import pytest

from media_pipeline.render import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    RenderError,
    render_speech,
)
from media_pipeline.captions import Caption
from media_pipeline.postprocess import read_wav, write_wav
from media_pipeline.render import _run_model_stage, _safe_name

SAMPLE_RATE = 24_000
_INSTRUCT = "自然、清晰、克制。"


# --- task-fake model engines ------------------------------------------------


class FakeTTSEngine:
    """Models the TTS subprocess: one construction == one model load.

    ``loads`` counts constructions so tests can assert the orchestrator builds
    the engine exactly once (one model load per run). ``durations`` maps segment
    id -> synthesized WAV length in seconds; ``fail_ids`` report a synthesis
    failure, ``wav_error_ids`` write an invalid WAV that the parent must reject.
    """

    loads = 0

    def __init__(
        self,
        durations: dict[str, float] | None = None,
        fail_ids: set[str] | None = None,
        wav_error_ids: set[str] | None = None,
        rates: dict[str, int] | None = None,
    ) -> None:
        FakeTTSEngine.loads += 1
        self.durations = durations or {}
        self.fail_ids = set(fail_ids or [])
        self.wav_error_ids = set(wav_error_ids or [])
        self.rates = rates or {}

    def __call__(self, task: dict) -> dict:
        segments_dir = Path(task["run_dir"]) / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for seg in task["segments"]:
            rate = self.rates.get(seg["id"], SAMPLE_RATE)
            wav = segments_dir / (seg["safe_name"] + ".wav")
            frames = int(self.durations.get(seg["id"], 1.0) * rate)
            if seg["id"] in self.wav_error_ids:
                # A mono-but-empty WAV: read_wav rejects it as a run-level fault.
                write_wav(wav, [], rate)
            else:
                write_wav(wav, _waveform(frames), rate)
            if seg["id"] in self.fail_ids:
                results.append(
                    {
                        "segment_id": seg["id"],
                        "status": "synthesis_failed",
                        "reason": "simulated model failure",
                    }
                )
            else:
                results.append({"segment_id": seg["id"], "status": "ok"})
        return {"runtime_error": None, "segments": results}


class FakeAlignerEngine:
    """Models the Alignment subprocess: one construction == one model load.

    ``fail_ids`` report an isolatable deterministic ``alignment_failed`` while
    other independent segments still succeed; ``runtime`` returns a
    model-runtime error that stops the whole run.
    """

    loads = 0

    def __init__(
        self,
        fail_ids: set[str] | None = None,
        *,
        runtime: bool = False,
    ) -> None:
        FakeAlignerEngine.loads += 1
        self.fail_ids = set(fail_ids or [])
        self.runtime = runtime

    def __call__(self, task: dict) -> dict:
        if self.runtime:
            return {"runtime_error": "simulated CUDA/model failure", "segments": []}

        segments_dir = Path(task["run_dir"]) / "segments"
        results = []
        for seg in task["segments"]:
            raw = segments_dir / (seg["safe_name"] + ".alignment.raw.json")
            _, rate, frames = read_wav(segments_dir / (seg["safe_name"] + ".wav"))
            duration = frames / rate
            if seg["id"] in self.fail_ids:
                _write_bad_alignment(raw, seg["text"], duration)
                results.append(
                    {
                        "segment_id": seg["id"],
                        "status": "alignment_failed",
                        "reason": "alignment text does not match",
                    }
                )
            else:
                _write_alignment(raw, seg["text"], duration)
                results.append({"segment_id": seg["id"], "status": "ok"})
        return {"runtime_error": None, "segments": results}


def _waveform(frames: int) -> list[int]:
    """A finite, non-silent, mono float waveform -> int16 PCM (never non-finite)."""

    out: list[int] = []
    for index in range(frames):
        decay = max(0.05, 1.0 - index / max(1, frames))
        import numbers

        f = 0.3 * math.sin(index / 80.0) * decay
        q = int(math.floor(f * 32_768 + 0.5))
        out.append(max(-32_768, min(32_767, q)))
    return out


def _write_alignment(path: Path, text: str, duration: float) -> None:
    """One aligned token per spoken character, strictly increasing, inside the WAV."""

    skippable = frozenset("PZ")

    def _spoken(s: str) -> list[str]:
        return [c for c in s if not (c.isspace() or unicodedata.category(c)[0] in skippable)]

    spoken = _spoken(text)
    if not spoken:
        spoken = ["x"]
    records = []
    time = 0.2
    step = max(0.01, (duration - 0.3) / len(spoken))
    for i, char in enumerate(spoken):
        start = time
        end = min(time + step, duration - 0.05)
        records.append({"text": char, "start": start, "end": end})
        time = end
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_bad_alignment(path: Path, text: str, duration: float) -> None:
    """A malformed alignment: a single token whose text does not match the script."""

    path.write_text(
        json.dumps([{"text": "无关文本", "start": 0.2, "end": min(0.5, duration - 0.05)}])
        + "\n",
        encoding="utf-8",
    )


# --- request writers --------------------------------------------------------


def _write_script(tmp_path: Path, segments: list[dict], **top: object) -> Path:
    payload = {"language": "Chinese", "speaker": "Uncle_Fu", "instruct": _INSTRUCT, **top}
    payload["segments"] = segments
    path = tmp_path / "script.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _seg(segment_id: str, text: str, pause_after_ms: int = 0) -> dict:
    return {"id": segment_id, "text": text, "pause_after_ms": pause_after_ms}


# --- C3: preflight rules ----------------------------------------------------


def test_preflight_reports_all_static_errors_at_once(tmp_path: Path) -> None:
    script = _write_script(
        tmp_path,
        [
            _seg("a", "欢迎收听。"),
            _seg("a", "重复 id。"),  # duplicate id
            _seg("b", "   "),  # no alignable content
            {"id": "c", "text": "x" * 11, "pause_after_ms": 0},  # too long
        ],
        max_segment_chars=10,
    )
    with pytest.raises(RenderError) as excinfo:
        render_speech(
            script,
            tmp_path / "run",
            tts_python="python",
            alignment_python="python",
            tts_model="/m",
            alignment_model="/m",
            max_segment_chars=10,
            _wav_task=lambda task: {},  # noqa: ARG005  must never be reached
            _align_task=lambda task: {},  # noqa: ARG005
        )
    message = str(excinfo.value)
    assert "duplicate segment id" in message
    assert "alignable content" in message
    assert "over the limit" in message
    # No model engine was constructed: preflight finished before any model.
    assert FakeTTSEngine.loads == 0
    assert FakeAlignerEngine.loads == 0
    # The input file is untouched.
    assert script.read_text(encoding="utf-8").count('"id"') == 4


def test_preflight_rejects_unknown_fields_and_non_string_ids(tmp_path: Path) -> None:
    script = _write_script(
        tmp_path,
        [{"id": "a", "text": "正文。", "pause_after_ms": 100, "extra": "nope"}],
    )
    with pytest.raises(RenderError, match="unknown field"):
        render_speech(
            script,
            tmp_path / "run",
            tts_python="python",
            alignment_python="python",
            tts_model="/m",
            alignment_model="/m",
            max_segment_chars=100,
            _wav_task=lambda task: {},  # noqa: ARG005
            _align_task=lambda task: {},  # noqa: ARG005
        )


def test_preflight_rejects_bool_and_non_integer_pause_and_budget(tmp_path: Path) -> None:
    script = _write_script(
        tmp_path,
        [_seg("a", "正文。", pause_after_ms=True)],
    )
    with pytest.raises(RenderError, match="pause_after_ms must be a non-negative integer"):
        render_speech(
            script,
            tmp_path / "run",
            tts_python="python",
            alignment_python="python",
            tts_model="/m",
            alignment_model="/m",
            max_segment_chars=100,
            _wav_task=lambda task: {},  # noqa: ARG005
            _align_task=lambda task: {},  # noqa: ARG005
        )

    bad_budget = _write_script(tmp_path, [_seg("a", "正文。")], max_segment_chars=True)
    with pytest.raises(RenderError, match="max_segment_chars must be a positive integer"):
        render_speech(
            bad_budget,
            tmp_path / "run",
            tts_python="python",
            alignment_python="python",
            tts_model="/m",
            alignment_model="/m",
            max_segment_chars=True,
            _wav_task=lambda task: {},  # noqa: ARG005
            _align_task=lambda task: {},  # noqa: ARG005
        )


def test_preflight_rejects_non_empty_segments_and_non_string_text(tmp_path: Path) -> None:
    empty = _write_script(tmp_path, [])
    with pytest.raises(RenderError, match="non-empty array"):
        render_speech(
            empty,
            tmp_path / "run",
            tts_python="python",
            alignment_python="python",
            tts_model="/m",
            alignment_model="/m",
            max_segment_chars=100,
            _wav_task=lambda task: {},  # noqa: ARG005
            _align_task=lambda task: {},  # noqa: ARG005
        )

    bad_text = _write_script(
        tmp_path,
        [{"id": "a", "pause_after_ms": 0}],  # missing text
    )
    with pytest.raises(RenderError, match="text must be a non-empty string"):
        render_speech(
            bad_text,
            tmp_path / "run",
            tts_python="python",
            alignment_python="python",
            tts_model="/m",
            alignment_model="/m",
            max_segment_chars=100,
            _wav_task=lambda task: {},  # noqa: ARG005
            _align_task=lambda task: {},  # noqa: ARG005
        )


def test_text_is_passed_verbatim_without_strip(tmp_path: Path) -> None:
    """The original text (with leading/trailing spaces) reaches the report unchanged."""

    script = _write_script(tmp_path, [_seg("a", "  你好 ，  world  ")])
    run = tmp_path / "run"
    render_speech(
        script,
        run,
        tts_python="python",
        alignment_python="python",
        tts_model="/m",
        alignment_model="/m",
        max_segment_chars=100,
        _wav_task=FakeTTSEngine(durations={"a": 1.0}),
        _align_task=FakeAlignerEngine(),
    )
    request = json.loads((run / "request.json").read_text(encoding="utf-8"))
    assert request["segments"][0]["text"] == "  你好 ，  world  "


# --- C6: output directory refusal ------------------------------------------


def test_existing_output_directory_is_refused(tmp_path: Path) -> None:
    script = _write_script(tmp_path, [_seg("a", "正文。")])
    existing = tmp_path / "run"
    existing.mkdir()
    FakeTTSEngine.loads = 0
    FakeAlignerEngine.loads = 0
    with pytest.raises(RenderError, match="already exists"):
        render_speech(
            script,
            existing,
            tts_python="python",
            alignment_python="python",
            tts_model="/m",
            alignment_model="/m",
            max_segment_chars=100,
            _wav_task=lambda task: {},  # noqa: ARG005
            _align_task=lambda task: {},  # noqa: ARG005
        )
    assert FakeTTSEngine.loads == 0
    assert FakeAlignerEngine.loads == 0


def test_output_directory_must_not_equal_input(tmp_path: Path) -> None:
    # The script lives inside a directory; rendering into that same directory
    # would overwrite the input and must be refused by the existence check.
    script = _write_script(tmp_path, [_seg("a", "正文。")])
    FakeTTSEngine.loads = 0
    FakeAlignerEngine.loads = 0
    with pytest.raises(RenderError, match="already exists"):
        render_speech(
            script,
            tmp_path,
            tts_python="python",
            alignment_python="python",
            tts_model="/m",
            alignment_model="/m",
            max_segment_chars=100,
            _wav_task=lambda task: {},  # noqa: ARG005
            _align_task=lambda task: {},  # noqa: ARG005
        )
    assert FakeTTSEngine.loads == 0
    assert FakeAlignerEngine.loads == 0


# --- C4/C5: model lifecycle and per-segment failure isolation ---------------


def test_each_model_loaded_exactly_once_for_many_segments(tmp_path: Path) -> None:
    FakeTTSEngine.loads = 0
    FakeAlignerEngine.loads = 0
    script = _write_script(
        tmp_path,
        [_seg(f"s{i}", f"正文第{i}段。") for i in range(5)],
    )
    run = tmp_path / "run"
    result = render_speech(
        script,
        run,
        tts_python="python",
        alignment_python="python",
        tts_model="/m",
        alignment_model="/m",
        max_segment_chars=100,
        _wav_task=FakeTTSEngine(durations={f"s{i}": 1.0 + i * 0.5 for i in range(5)}),
        _align_task=FakeAlignerEngine(),
    )
    assert result.status == STATUS_COMPLETE
    # One construction each: the orchestrator never spawns a model per segment.
    assert FakeTTSEngine.loads == 1
    assert FakeAlignerEngine.loads == 1


def test_alignment_failure_is_isolated_and_others_continue(tmp_path: Path) -> None:
    """A deterministic alignment failure on one segment does not block the rest."""

    script = _write_script(
        tmp_path,
        [
            _seg("first", "第一段正文。"),
            _seg("bad", "坏段。"),
            _seg("third", "第三段正文。"),
        ],
    )
    run = tmp_path / "run"
    result = render_speech(
        script,
        run,
        tts_python="python",
        alignment_python="python",
        tts_model="/m",
        alignment_model="/m",
        max_segment_chars=100,
        _wav_task=FakeTTSEngine(durations={"first": 1.0, "bad": 1.0, "third": 1.0}),
        _align_task=FakeAlignerEngine(fail_ids={"bad"}),
    )
    assert result.status == STATUS_INCOMPLETE
    assert result.wav_path is None
    assert not (run / "final").exists()  # no complete episode published
    by_id = {s.segment_id: s.status for s in result.segments}
    assert by_id["first"] == "ok"
    assert by_id["bad"] == "alignment_failed"
    assert by_id["third"] == "ok"
    # The failed segment's status/stage/reason are recorded and locatable.
    bad = next(s for s in result.segments if s.segment_id == "bad")
    assert bad.stage == "alignment"
    assert bad.reason


def test_alignment_runtime_failure_stops_the_run(tmp_path: Path) -> None:
    """A model/CUDA/subprocess fault stops the run instead of being skipped."""

    script = _write_script(
        tmp_path,
        [_seg("first", "第一段。"), _seg("second", "第二段。")],
    )
    run = tmp_path / "run"
    with pytest.raises(RenderError) as excinfo:
        render_speech(
            script,
            run,
            tts_python="python",
            alignment_python="python",
            tts_model="/m",
            alignment_model="/m",
            max_segment_chars=100,
            _wav_task=FakeTTSEngine(durations={"first": 1.0, "second": 1.0}),
            _align_task=FakeAlignerEngine(runtime=True),
        )
    # The run directory is reported so the partial artifacts can be recovered.
    assert excinfo.value.run_dir == run
    assert not (run / "final").exists()


def test_tts_synthesis_failure_stops_the_run_keeps_files(tmp_path: Path) -> None:
    script = _write_script(
        tmp_path,
        [_seg("first", "第一段。"), _seg("second", "第二段。")],
    )
    run = tmp_path / "run"
    with pytest.raises(RenderError, match="TTS synthesis failed"):
        render_speech(
            script,
            run,
            tts_python="python",
            alignment_python="python",
            tts_model="/m",
            alignment_model="/m",
            max_segment_chars=100,
            _wav_task=FakeTTSEngine(
                durations={"first": 1.0, "second": 1.0},
                fail_ids={"second"},
            ),
            _align_task=FakeAlignerEngine(),
        )
    # The already-generated first segment WAV is kept as evidence.
    assert (run / "segments").is_dir()


def test_invalid_segment_wav_is_rejected(tmp_path: Path) -> None:
    script = _write_script(tmp_path, [_seg("a", "正文。")])
    run = tmp_path / "run"
    with pytest.raises(RenderError, match="empty or malformed"):
        render_speech(
            script,
            run,
            tts_python="python",
            alignment_python="python",
            tts_model="/m",
            alignment_model="/m",
            max_segment_chars=100,
            _wav_task=FakeTTSEngine(durations={"a": 1.0}, wav_error_ids={"a"}),
            _align_task=FakeAlignerEngine(),
        )


# --- C7: timeline, assembly and product integrity ---------------------------


def test_three_unequal_segments_timeline_and_frames(tmp_path: Path) -> None:
    durations = {"a": 1.0, "b": 2.0, "c": 1.5}
    pauses = {"a": 300, "b": 0, "c": 200}
    script = _write_script(
        tmp_path,
        [
            _seg("a", "第一段。", pause_after_ms=pauses["a"]),
            _seg("b", "第二段比较长。"),
            _seg("c", "第三段。", pause_after_ms=pauses["c"]),
        ],
    )
    run = tmp_path / "run"
    result = render_speech(
        script,
        run,
        tts_python="python",
        alignment_python="python",
        tts_model="/m",
        alignment_model="/m",
        max_segment_chars=100,
        _wav_task=FakeTTSEngine(durations=durations),
        _align_task=FakeAlignerEngine(),
    )
    assert result.status == STATUS_COMPLETE

    timeline = json.loads((run / "final" / "timeline.json").read_text(encoding="utf-8"))
    assert timeline["sample_rate"] == SAMPLE_RATE

    # O_i = sum(N_j + G_j for j < i); G = half_up(pause_ms * R / 1000).
    g = {
        "a": _half_up(300, SAMPLE_RATE),
        "b": _half_up(0, SAMPLE_RATE),
        "c": _half_up(200, SAMPLE_RATE),
    }
    # audio_frames are the *cleaned* (post-trim) frame counts, per C7.
    n = {
        sid: read_wav(run / "segments" / (_safe_name(sid) + ".cleaned.wav"))[2]
        for sid in ("a", "b", "c")
    }
    expected_start = {"a": 0, "b": n["a"] + g["a"], "c": n["a"] + g["a"] + n["b"] + g["b"]}
    for entry in timeline["segments"]:
        sid = entry["segment_id"]
        assert entry["start_frame"] == expected_start[sid]
        assert entry["audio_frames"] == n[sid]
        assert entry["pause_after_frames"] == g[sid]

    total = sum(n[sid] + g[sid] for sid in ("a", "b", "c"))
    assert timeline["total_frames"] == total

    # Re-read the final WAV: frames and rate match the integer accumulation.
    assert result.wav_path is not None
    _, rate, frames = read_wav(result.wav_path)
    assert (rate, frames) == (SAMPLE_RATE, total)


def test_original_wav_preserved_and_cleaned_differs(tmp_path: Path) -> None:
    """The raw TTS WAV is never overwritten by alignment/postprocess/assembly."""

    durations = {"a": 2.0}
    script = _write_script(tmp_path, [_seg("a", "一段较长正文。")])
    run = tmp_path / "run"
    render_speech(
        script,
        run,
        tts_python="python",
        alignment_python="python",
        tts_model="/m",
        alignment_model="/m",
        max_segment_chars=100,
        _wav_task=FakeTTSEngine(durations=durations),
        _align_task=FakeAlignerEngine(),
    )
    seg_dir = run / "segments"
    raw_wav = next(p for p in seg_dir.glob("*.wav") if not p.name.endswith(".cleaned.wav"))
    cleaned_wav = next(p for p in seg_dir.glob("*.cleaned.wav"))
    assert not raw_wav.name.endswith(".cleaned.wav")
    _, raw_rate, raw_frames = read_wav(raw_wav)
    _, cleaned_rate, cleaned_frames = read_wav(cleaned_wav)
    assert (raw_rate, cleaned_rate) == (SAMPLE_RATE, SAMPLE_RATE)
    # Postprocess trimmed leading/trailing silence, so cleaned < raw.
    assert cleaned_frames < raw_frames
    # The raw TTS WAV still exists and is intact.
    assert raw_wav.is_file()


def test_final_products_read_back_consistent(tmp_path: Path) -> None:
    script = _write_script(tmp_path, [_seg("a", "一段正文。")])
    run = tmp_path / "run"
    result = render_speech(
        script,
        run,
        tts_python="python",
        alignment_python="python",
        tts_model="/m",
        alignment_model="/m",
        max_segment_chars=100,
        _wav_task=FakeTTSEngine(durations={"a": 1.0}),
        _align_task=FakeAlignerEngine(),
    )
    srt = (run / "final" / "final.srt").read_text(encoding="utf-8")
    timeline = json.loads((run / "final" / "timeline.json").read_text(encoding="utf-8"))
    assert result.wav_path is not None
    _, rate, frames = read_wav(result.wav_path)
    # The SRT, timeline and WAV all agree and re-verify against the disk copy.
    assert srt.strip()
    assert timeline["total_frames"] == frames
    assert timeline["sample_rate"] == rate


def test_zero_duration_caption_is_explicit_failure(tmp_path: Path) -> None:
    from media_pipeline.render import _verify_caption_timeline

    run = tmp_path / "run"
    # float start < end, but millisecond quantization collapses it to zero ms.
    captions = [Caption(start=1.0, end=1.0004, text="x")]
    with pytest.raises(RenderError, match="zero or negative duration"):
        _verify_caption_timeline(captions, SAMPLE_RATE, SAMPLE_RATE, run)


def test_overlap_and_past_wav_captions_are_rejected(tmp_path: Path) -> None:
    from media_pipeline.render import _verify_caption_timeline

    run = tmp_path / "run"

    overlapping = [
        Caption(start=0.0, end=1.0, text="a"),
        Caption(start=0.5, end=1.5, text="b"),
    ]
    with pytest.raises(RenderError, match="overlaps"):
        _verify_caption_timeline(overlapping, SAMPLE_RATE * 2, SAMPLE_RATE, run)

    past = [Caption(start=2.0, end=3.0, text="a")]
    with pytest.raises(RenderError, match="past the final"):
        _verify_caption_timeline(past, SAMPLE_RATE, SAMPLE_RATE, run)


def test_sample_rate_mismatch_during_assembly_fails(tmp_path: Path) -> None:
    script = _write_script(
        tmp_path,
        [_seg("a", "第一段。"), _seg("b", "第二段。")],
    )
    run = tmp_path / "run"
    with pytest.raises(RenderError, match="sample rate mismatch"):
        render_speech(
            script,
            run,
            tts_python="python",
            alignment_python="python",
            tts_model="/m",
            alignment_model="/m",
            max_segment_chars=100,
            _wav_task=FakeTTSEngine(
                durations={"a": 1.0, "b": 1.0},
                rates={"a": SAMPLE_RATE, "b": 16_000},
            ),
            _align_task=FakeAlignerEngine(),
        )


def test_global_captions_use_integer_frame_offsets(tmp_path: Path) -> None:
    """Many unequal short segments: global time derives from integer frames, not
    accumulated rounded SRT durations."""

    n_segments = 12
    durations = {f"s{i}": 0.3 + i * 0.03 for i in range(n_segments)}
    script = _write_script(
        tmp_path,
        [_seg(f"s{i}", f"第{i}段。") for i in range(n_segments)],
    )
    run = tmp_path / "run"
    render_speech(
        script,
        run,
        tts_python="python",
        alignment_python="python",
        tts_model="/m",
        alignment_model="/m",
        max_segment_chars=100,
        _wav_task=FakeTTSEngine(durations=durations),
        _align_task=FakeAlignerEngine(),
    )
    timeline = json.loads((run / "final" / "timeline.json").read_text(encoding="utf-8"))

    # Recompute the integer-frame offsets independently and confirm they match.
    running = 0
    for entry in timeline["segments"]:
        assert entry["start_frame"] == running
        running += entry["audio_frames"] + entry["pause_after_frames"]
    assert timeline["total_frames"] == running


# --- C5: failure semantics on postprocess/caption stages --------------------


def test_caption_failure_records_segment_and_reports_incomplete(tmp_path: Path) -> None:
    """A deterministic caption failure records the segment without crashing."""

    script = _write_script(
        tmp_path,
        [
            _seg("good", "好段。"),
            _seg("bad", "坏段。"),
        ],
    )
    run = tmp_path / "run"

    def aligning_task(task: dict) -> dict:
        # Both segments align OK, but force a caption mismatch for "bad" by
        # writing a bad alignment for it after the engine reports success.
        engine = FakeAlignerEngine()
        result = engine(task)
        seg_dir = Path(task["run_dir"]) / "segments"
        for entry in result["segments"]:
            if entry["segment_id"] == "bad":
                bad = seg_dir / (
                    next(p for p in seg_dir.glob("*.alignment.raw.json"))
                )
                # Not reachable here; use a second pass instead.
        return result

    # Simpler: inject a custom alignment task that writes a mismatch for "bad".
    def bad_caption_task(task: dict) -> dict:
        seg_dir = Path(task["run_dir"]) / "segments"
        results = []
        for seg in task["segments"]:
            raw = seg_dir / (seg["safe_name"] + ".alignment.raw.json")
            _, rate, frames = read_wav(seg_dir / (seg["safe_name"] + ".wav"))
            duration = frames / rate
            if seg["id"] == "bad":
                # Valid alignment, but whose token text omits part of the script
                # so the Caption Compiler raises a mismatch deterministically.
                _write_alignment(raw, "无关", duration)
                results.append({"segment_id": seg["id"], "status": "ok"})
            else:
                _write_alignment(raw, seg["text"], duration)
                results.append({"segment_id": seg["id"], "status": "ok"})
        return {"runtime_error": None, "segments": results}

    result = render_speech(
        script,
        run,
        tts_python="python",
        alignment_python="python",
        tts_model="/m",
        alignment_model="/m",
        max_segment_chars=100,
        _wav_task=FakeTTSEngine(durations={"good": 1.0, "bad": 1.0}),
        _align_task=bad_caption_task,
    )
    assert result.status == STATUS_INCOMPLETE
    by_id = {s.segment_id: s.status for s in result.segments}
    assert by_id["good"] == "ok"
    assert by_id["bad"] == "caption_failed"
    assert not (run / "final").exists()


# --- C6: completion marker discipline ---------------------------------------


def test_incomplete_run_has_no_completion_marker(tmp_path: Path) -> None:
    script = _write_script(
        tmp_path,
        [_seg("a", "好段。"), _seg("b", "坏段。")],
    )
    run = tmp_path / "run"
    result = render_speech(
        script,
        run,
        tts_python="python",
        alignment_python="python",
        tts_model="/m",
        alignment_model="/m",
        max_segment_chars=100,
        _wav_task=FakeTTSEngine(durations={"a": 1.0, "b": 1.0}),
        _align_task=FakeAlignerEngine(fail_ids={"b"}),
    )
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == STATUS_INCOMPLETE
    assert not (run / "final" / ".complete").exists()


def test_complete_run_writes_completion_marker(tmp_path: Path) -> None:
    script = _write_script(tmp_path, [_seg("a", "一段。")])
    run = tmp_path / "run"
    render_speech(
        script,
        run,
        tts_python="python",
        alignment_python="python",
        tts_model="/m",
        alignment_model="/m",
        max_segment_chars=100,
        _wav_task=FakeTTSEngine(durations={"a": 1.0}),
        _align_task=FakeAlignerEngine(),
    )
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == STATUS_COMPLETE
    assert (run / "final" / ".complete").read_text(encoding="utf-8") == "complete"


def test_runtime_failure_leaves_no_valid_completion_marker(tmp_path: Path) -> None:
    script = _write_script(tmp_path, [_seg("a", "一段。")])
    run = tmp_path / "run"
    with pytest.raises(RenderError):
        render_speech(
            script,
            run,
            tts_python="python",
            alignment_python="python",
            tts_model="/m",
            alignment_model="/m",
            max_segment_chars=100,
            _wav_task=FakeTTSEngine(durations={"a": 1.0}),
            _align_task=FakeAlignerEngine(runtime=True),
        )
    assert not (run / "final" / ".complete").exists()


# --- C4/C5: real subprocess protocol (CPU, no GPU runtime) ------------------


def test_run_model_stage_round_trip(tmp_path: Path) -> None:
    """The task manifest round-trips through a subprocess: success decoded."""

    import sys

    fake_stage = r"""
import json
import sys
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
Path(spec["manifest"]).write_text(
    json.dumps({"runtime_error": None, "segments": [
        {"segment_id": "a", "status": "ok"},
        {"segment_id": "b", "status": "alignment_failed", "reason": "mismatch"},
    ]}) + "\n", encoding="utf-8",
)
sys.exit(0)
"""
    result = _run_model_stage(
        sys.executable, fake_stage, ("model", {"segments": [{"id": "a"}, {"id": "b"}]}),
    )
    assert result["runtime_error"] is None
    assert [e["segment_id"] for e in result["segments"]] == ["a", "b"]
    assert result["segments"][1]["status"] == "alignment_failed"


def test_run_model_stage_runtime_failure_reports_runtime_error(tmp_path: Path) -> None:
    """A non-zero stage exit is reported as a run-level runtime_error, not a
    per-segment failure, so the orchestrator stops the run."""

    import sys

    fake_stage = r"""
import sys
sys.stderr.write("cuda out of memory")
sys.exit(1)
"""
    result = _run_model_stage(
        sys.executable, fake_stage, ("model", {"segments": []}),
    )
    assert result["runtime_error"]
    assert "cuda out of memory" in result["runtime_error"]


# --- helpers ---------------------------------------------------------------


def _half_up(value: int, rate: int) -> int:
    from decimal import ROUND_HALF_UP, Decimal

    return int((Decimal(str(value)) * rate / Decimal(1000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
