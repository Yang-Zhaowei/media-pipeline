"""CPU unit tests for Production Alignment v0 (no models, no CUDA).

These tests cover the portable contracts/validation in
:mod:`media_pipeline.alignment` and the Qwen3 ForcedAligner runtime adapter's
mapping/validation logic using an injected fake aligner. They never import
``torch`` or ``qwen_asr``: the runtime imports those lazily, so a fake aligner
can stand in for the real model on CPU.

They mirror the structure of the TTS CPU tests: portable-only tests plus tests
that drive :meth:`Qwen3ForcedAlignment.align` with a fake engine, so every
failure path that does not need the real GPU model is exercised here.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from media_pipeline.runtimes.qwen_aligner import Qwen3ForcedAlignment

import pytest

import media_pipeline
from media_pipeline import (
    AlignmentArtifact,
    AlignmentMismatchError,
    AlignmentRequest,
    AlignmentRequestError,
    AlignmentRuntimeError,
)
from media_pipeline.alignment import (
    AlignmentError,
    validate_alignment,
    validate_request,
    validate_wav,
)
from media_pipeline.captions import (
    AlignedToken,
    build_captions,
    compile_srt,
    has_alignable_content,
    load_alignment,
    parse_alignment,
)
from media_pipeline.postprocess import read_wav, write_wav, postprocess_speech

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "speech-smoke-001"
FIXTURE_WAV = FIXTURE / "raw.wav"
FIXTURE_TEXT = FIXTURE / "original.txt"
FIXTURE_ALIGNMENT = FIXTURE / "alignment.raw.json"

EXPECTED_SRT = (
    "1\n"
    "00:00:01,360 --> 00:00:04,880\n"
    "真正限制本地人工智能模型使用体验的，\n"
    "\n"
    "2\n"
    "00:00:05,200 --> 00:00:07,920\n"
    "往往并不只是模型本身的参数规模。\n"
)


# --- test isolation --------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _isolate_runtime_imports() -> Iterator[None]:
    """Prevent these tests from leaking the Qwen3 runtime import into other files.

    Some tests here import :class:`Qwen3ForcedAlignment` (which registers
    ``media_pipeline.runtimes`` in ``sys.modules``) while the TTS import
    independence test asserts that package stays un-imported. Restoring
    ``sys.modules`` when this module finishes keeps the two test files from
    stepping on each other regardless of collection order.
    """

    runtime_keys = (
        lambda: {
            name
            for name in sys.modules
            if name == "media_pipeline.runtimes"
            or name.startswith("media_pipeline.runtimes.")
        }
    )
    before = runtime_keys()
    yield
    # Record which keys these tests added *after* they ran; restore only those,
    # leaving any that pre-existed untouched.
    for name in runtime_keys() - before:
        sys.modules.pop(name, None)


# --- helpers ---------------------------------------------------------------


def _pcm_wav(
    path: Path,
    *,
    n_channels: int = 1,
    bits: int = 16,
    sample_rate: int = 24000,
    frames: int = 24000,
    fill: int = 0,
) -> Path:
    """Write a controlled uncompressed mono/stereo PCM WAV so we control the
    channel count, bit depth, sample rate, and frame count exactly.

    Uses the same proven RIFF/WAVE chunk layout as the Audio Postprocess
    regression test so the file decodes cleanly through stdlib ``wave``.
    """

    bytes_per_sample = bits // 8
    fmt = struct.pack(
        "<HHIIHH",
        1,  # WAVE_FORMAT_PCM
        n_channels,
        sample_rate,
        sample_rate * n_channels * bytes_per_sample,
        n_channels * bytes_per_sample,
        bits,
    )
    data = struct.pack(f"<{frames}h", *([fill] * frames))
    body = b"WAVE"
    body += b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return Path(path)


def _alignment_request(
    wav: Path,
    text: str = "真正限制本地人工智能模型使用体验的，往往并不只是模型本身的参数规模。",
    language: str = "Chinese",
) -> AlignmentRequest:
    return AlignmentRequest(wav_path=wav, text=text, language=language)


class _FakeItem:
    """A minimal stand-in for the upstream ``ForcedAlignItem``."""

    def __init__(self, text: str, start: float, end: float) -> None:
        self.text = text
        self.start_time = start
        self.end_time = end


class _FakeResult:
    """A minimal stand-in for the upstream ``ForcedAlignResult``."""

    def __init__(self, items: list[_FakeItem]) -> None:
        self._items = list(items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> _FakeItem:
        return self._items[idx]


class _FakeAligner:
    """Records calls and returns scripted results; can fail on demand."""

    def __init__(
        self,
        items_by_text,
        *,
        fail_align_exc: Exception | None = None,
    ) -> None:
        self._items_by_text = items_by_text
        self.fail_align_exc = fail_align_exc
        self.align_calls: list[tuple[object, str, str]] = []

    def align(self, audio, text: str, language: str):  # noqa: ANN001 - mirrors arbitrary upstream input
        self.align_calls.append((audio, text, language))
        if self.fail_align_exc is not None:
            raise self.fail_align_exc
        return self._items_by_text(text)


class _RecordingLoader:
    """Stand-in for ``qwen_asr.Qwen3ForcedAligner`` that counts loads."""

    def __init__(self, aligner: _FakeAligner) -> None:
        self._aligner = aligner
        self.load_count = 0

    def from_pretrained(self, path, *, device_map, dtype):  # noqa: ANN001 - mirrors upstream kwargs
        self.load_count += 1
        return self._aligner


def _install_fake(monkeypatch, aligner: _FakeAligner, loader: _RecordingLoader) -> None:
    """Make the runtime's lazy imports resolve to fakes."""

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(
        sys.modules,
        "qwen_asr",
        types.SimpleNamespace(Qwen3ForcedAligner=loader),
    )


def _import_runtime():
    # Imported lazily so merely collecting this test file does not register
    # media_pipeline.runtimes in sys.modules and defeat the TTS import
    # independence test. The annotation is a string (from __future__ import
    # annotations), so this runtime import does not break the type checker.
    from media_pipeline.runtimes.qwen_aligner import Qwen3ForcedAlignment
    return Qwen3ForcedAlignment


def _engine(monkeypatch, aligner: _FakeAligner) -> "Qwen3ForcedAlignment":
    loader = _RecordingLoader(aligner)
    _install_fake(monkeypatch, aligner, loader)
    return _import_runtime()("Qwen/Qwen3-ForcedAligner-0.6B")


# --- portable boundary: importing must not load the runtime ----------------


def test_importing_media_pipeline_does_not_pull_torch_or_qwen_asr() -> None:
    import media_pipeline.alignment  # noqa: F401
    import media_pipeline.runtimes.qwen_aligner  # noqa: F401

    assert "torch" not in sys.modules
    assert "qwen_asr" not in sys.modules


def test_alignment_runtime_import_independence_is_subprocess_stable() -> None:
    # Even if the host has torch/qwen_asr installed and some earlier test
    # imported them, a fresh interpreter importing the portable surface must
    # still not pull them. A subprocess makes the assertion robust, and keeps
    # the in-process runtime tests from leaking import state into other files.
    src = Path(__file__).resolve().parent.parent / "src"
    script = (
        "import sys"
        "; sys.path.insert(0, %r)"
        "; import media_pipeline"
        "; import media_pipeline.alignment"
        "; import media_pipeline.runtimes.qwen_aligner"
        "; bad = [m for m in ('torch', 'qwen_asr') if m in sys.modules]"
        "; raise SystemExit(1 if bad else 0)"
    ) % str(src)
    result = subprocess.run(  # noqa: S603 - controlled script
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# --- AlignmentRequest semantics -------------------------------------------


def test_request_preserves_fields() -> None:
    # No WAV needs to exist for a field-preservation check; the request is a
    # plain dataclass that stores whatever path it is given.
    request = _alignment_request(Path("utterance.wav"), text="你好", language="Chinese")
    assert request.text == "你好"
    assert request.language == "Chinese"
    assert request.wav_path == Path("utterance.wav")


def test_request_is_frozen() -> None:
    request = _alignment_request(Path("x.wav"))
    with pytest.raises(AttributeError):
        request.language = "English"  # type: ignore[misc]


# --- request validation ----------------------------------------------------


def test_validate_request_accepts_valid_request() -> None:
    validate_request(_alignment_request(Path("x.wav")))


def test_validate_request_rejects_empty_text() -> None:
    with pytest.raises(AlignmentRequestError):
        validate_request(_alignment_request(Path("x.wav"), text="   "))


def test_validate_request_rejects_empty_language() -> None:
    with pytest.raises(AlignmentRequestError):
        validate_request(_alignment_request(Path("x.wav"), language=" "))


# --- WAV validation --------------------------------------------------------


def test_validate_wav_accepts_valid_mono_pcm16(tmp_path) -> None:
    wav = _pcm_wav(tmp_path / "a.wav", frames=100)
    sample_rate, frames = validate_wav(wav)
    assert (sample_rate, frames) == (24000, 100)


def test_validate_wav_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(AlignmentRequestError):
        validate_wav(tmp_path / "nope.wav")


def test_validate_wav_rejects_stereo(tmp_path) -> None:
    wav = _pcm_wav(tmp_path / "s.wav", n_channels=2, frames=4)
    with pytest.raises(AlignmentRequestError):
        validate_wav(wav)


def test_validate_wav_rejects_8bit(tmp_path) -> None:
    wav = _pcm_wav(tmp_path / "b8.wav", bits=8, frames=4)
    with pytest.raises(AlignmentRequestError):
        validate_wav(wav)


def test_validate_wav_rejects_empty_wav(tmp_path) -> None:
    wav = _pcm_wav(tmp_path / "empty.wav", frames=0)
    with pytest.raises(AlignmentRequestError):
        validate_wav(wav)


def test_validate_wav_rejects_truncated_wav(tmp_path) -> None:
    wav = _pcm_wav(tmp_path / "trunc.wav", frames=1000)
    path = Path(wav)
    path.write_bytes(path.read_bytes()[:10])  # keep only the RIFF signature
    with pytest.raises(AlignmentRequestError):
        validate_wav(wav)


def test_validate_wav_rejects_garbage(tmp_path) -> None:
    wav = tmp_path / "garbage.wav"
    wav.write_bytes(b"not a wav file at all")
    with pytest.raises(AlignmentRequestError):
        validate_wav(wav)


# --- alignment output validation -------------------------------------------


def _items(*entries: tuple[str, float, float]) -> list[dict]:
    return [{"text": t, "start": s, "end": e} for t, s, e in entries]


def test_validate_alignment_accepts_valid_items() -> None:
    tokens = validate_alignment(
        _items(("你", 0.0, 0.5), ("好", 0.5, 1.0)),
        sample_rate=24000,
        frames=24000,
    )
    assert tokens == [AlignedToken("你", 0.0, 0.5), AlignedToken("好", 0.5, 1.0)]


def test_validate_alignment_requires_at_least_one_token() -> None:
    with pytest.raises(AlignmentError, match="no effective tokens"):
        validate_alignment([], sample_rate=24000, frames=24000)


def test_validate_alignment_rejects_blank_records() -> None:
    # The Production boundary rejects blank/whitespace-only records outright;
    # they are never silently dropped.
    with pytest.raises(AlignmentError, match="empty or whitespace-only text"):
        validate_alignment(_items(("  ", 0.0, 0.5)), sample_rate=24000, frames=24000)
    with pytest.raises(AlignmentError):
        validate_alignment(
            _items(("  ", 0.0, 0.5), ("好", 0.5, 1.0)),
            sample_rate=24000,
            frames=24000,
        )


def test_parse_alignment_drops_blank_tokens_but_keeps_effective() -> None:
    # Caption Compiler downstream semantics are unchanged: blank tokens are
    # dropped, not rejected, by parse_alignment itself. The effective-token
    # requirement lives in validate_alignment, not here.
    tokens = parse_alignment(
        _items(("  ", 0.0, 0.5), ("好", 0.5, 1.0)),
    )
    assert tokens == [AlignedToken("好", 0.5, 1.0)]
    assert parse_alignment(_items(("  ", 0.0, 0.5))) == []


def test_validate_alignment_accepts_generator_input() -> None:
    # The public contract accepts any Iterable[object]; a one-shot generator
    # must be consumed exactly once, not exhausted by an internal pass.
    gen = (item for item in _items(("你", 0.0, 0.5), ("好", 0.5, 1.0)))
    from_generator = validate_alignment(gen, sample_rate=24000, frames=24000)
    from_list = validate_alignment(
        _items(("你", 0.0, 0.5), ("好", 0.5, 1.0)),
        sample_rate=24000,
        frames=24000,
    )
    assert from_generator == from_list
    assert from_generator == [
        AlignedToken("你", 0.0, 0.5),
        AlignedToken("好", 0.5, 1.0),
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validate_alignment_rejects_nonfinite_timestamps(value) -> None:
    with pytest.raises(AlignmentError, match="finite"):
        validate_alignment(_items(("你", 0.0, value)), sample_rate=24000, frames=24000)


@pytest.mark.parametrize(("start", "end"), [(True, 1.0), (0.0, False), (True, True)])
def test_validate_alignment_rejects_bool_timestamps(start, end) -> None:
    with pytest.raises(AlignmentError, match="must be a number"):
        validate_alignment(_items(("你", start, end)), sample_rate=24000, frames=24000)


def test_validate_alignment_rejects_negative_start() -> None:
    with pytest.raises(AlignmentError, match="starts before zero"):
        validate_alignment(_items(("你", -0.1, 0.5)), sample_rate=24000, frames=24000)


def test_validate_alignment_rejects_end_before_start() -> None:
    with pytest.raises(AlignmentError, match="ends before it starts"):
        validate_alignment(_items(("你", 0.5, 0.4)), sample_rate=24000, frames=24000)


def test_validate_alignment_rejects_overlapping_tokens() -> None:
    with pytest.raises(AlignmentError, match="before the previous item ends"):
        validate_alignment(
            _items(("你", 0.0, 1.0), ("好", 0.5, 1.5)),
            sample_rate=24000,
            frames=24000,
        )


def test_validate_alignment_allows_zero_duration_token() -> None:
    tokens = validate_alignment(
        _items(("你", 0.0, 0.5), ("好", 0.5, 0.5), ("吗", 0.5, 1.0)),
        sample_rate=24000,
        frames=24000,
    )
    assert len(tokens) == 3
    assert tokens[1].end == tokens[1].start  # zero duration preserved


def test_validate_alignment_allows_adjacent_touching_tokens() -> None:
    tokens = validate_alignment(
        _items(("你", 0.0, 0.5), ("好", 0.5, 1.0)),
        sample_rate=24000,
        frames=24000,
    )
    assert len(tokens) == 2


def test_validate_alignment_allows_token_at_exact_duration() -> None:
    tokens = validate_alignment(
        _items(("你", 0.0, 8.0)), sample_rate=24000, frames=192000
    )
    assert len(tokens) == 1


def test_validate_alignment_allows_within_tolerance() -> None:
    # 5e-7 past the end is inside the 1e-6 tolerance and must be allowed.
    tokens = validate_alignment(
        _items(("你", 0.0, 8.0 + 5e-7)), sample_rate=24000, frames=192000
    )
    assert len(tokens) == 1


def test_validate_alignment_rejects_past_duration() -> None:
    with pytest.raises(AlignmentError, match="past the WAV duration"):
        validate_alignment(_items(("你", 0.0, 8.001)), sample_rate=24000, frames=192000)


def test_validate_alignment_preserves_precision() -> None:
    tokens = validate_alignment(
        _items(("你", 0.123456789, 0.654321)), sample_rate=24000, frames=24000
    )
    assert tokens[0].start == 0.123456789
    assert tokens[0].end == 0.654321


# --- runtime: fake-model mapping -------------------------------------------


def test_align_maps_runtime_output_and_writes_json(tmp_path, monkeypatch) -> None:
    items = [("真", 1.36, 1.68), ("正", 1.68, 1.84)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "utterance.wav", frames=48000)
    out = tmp_path / "alignment.json"

    artifact = engine.align(_alignment_request(wav, text="真正"), out)

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == [
        {"text": "真", "start": 1.36, "end": 1.68},
        {"text": "正", "start": 1.68, "end": 1.84},
    ]
    assert artifact.alignment_path == out
    assert artifact.sample_rate == 24000
    assert artifact.frames == 48000
    assert artifact.token_count == 2
    assert artifact.duration == pytest.approx(2.0)


def test_align_writes_runtime_output_without_normalization(
    tmp_path,
    monkeypatch,
) -> None:
    # Accepted output is written verbatim: text is not stripped and timestamps
    # are not coerced (note the integer start written as an integer, not a
    # float). token_count matches the number of records written to the artifact.
    items = [("你", 1, 3.0), ("好", 3.0, 4.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=96000)
    out = tmp_path / "a.json"

    artifact = engine.align(_alignment_request(wav, text="你好"), out)

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == [
        {"text": "你", "start": 1, "end": 3.0},
        {"text": "好", "start": 3.0, "end": 4.0},
    ]
    assert artifact.token_count == len(written)


def test_align_passes_original_wav_text_and_language(tmp_path, monkeypatch) -> None:
    items = [("你", 0.0, 0.5), ("好", 0.5, 1.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    engine.align(
        _alignment_request(wav, text="你好", language="Chinese"),
        tmp_path / "a.json",
    )

    audio, text, language = aligner.align_calls[0]
    # Runtime boundary: qwen_asr receives a string filesystem path, not a Path.
    assert isinstance(audio, str)
    assert audio == str(wav)  # the original WAV path, as a string
    assert text == "你好"
    assert language == "Chinese"


def test_align_preserves_timestamp_precision_in_output(tmp_path, monkeypatch) -> None:
    items = [("你", 0.123456789, 0.654321)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    engine.align(_alignment_request(wav, text="你"), tmp_path / "a.json")

    written = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert written[0]["start"] == 0.123456789
    assert written[0]["end"] == 0.654321


def test_align_preserves_return_order_and_does_not_sort(tmp_path, monkeypatch) -> None:
    # Order b before a by text: a lexicographic sort would reorder; we must not.
    items = [("b", 0.0, 1.0), ("a", 1.0, 2.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=48000)
    engine.align(_alignment_request(wav, text="ba"), tmp_path / "a.json")

    written = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert [item["text"] for item in written] == ["b", "a"]


def test_align_unicode_round_trip(tmp_path, monkeypatch) -> None:
    items = [("真", 1.36, 1.68), ("\U0001F600", 1.68, 2.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=48000)
    engine.align(_alignment_request(wav, text="真\U0001F600"), tmp_path / "a.json")

    raw = (tmp_path / "a.json").read_text(encoding="utf-8")
    assert "\U0001F600" in raw  # ensure_ascii=False: written as real characters
    written = json.loads(raw)
    assert written[1]["text"] == "\U0001F600"


def test_align_artifact_token_count_matches_effective_tokens(tmp_path, monkeypatch) -> None:
    items = [("真", 1.36, 1.68), ("正", 1.68, 1.84), ("们", 1.84, 2.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=48000)
    artifact = engine.align(_alignment_request(wav, text="真正们"), tmp_path / "a.json")
    assert artifact.token_count == 3


# --- runtime: model-load-once + reuse --------------------------------------


def test_loads_model_once_and_reuses_engine_for_repeated_aligns(
    tmp_path,
    monkeypatch,
) -> None:
    items = [("你", 0.0, 1.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    engine.align(_alignment_request(wav, text="你"), tmp_path / "a.json")
    engine.align(_alignment_request(wav, text="你"), tmp_path / "b.json")

    # The real _load_aligner ran exactly once; both align calls reused it.
    assert engine._aligner is aligner
    assert len(aligner.align_calls) == 2


def test_engine_reuses_same_fake_instance(tmp_path, monkeypatch) -> None:
    items = [("你", 0.0, 1.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    engine.align(_alignment_request(wav, text="你"), tmp_path / "a.json")
    engine.align(_alignment_request(wav, text="你"), tmp_path / "b.json")
    assert engine._aligner is aligner


# --- runtime: failure classes ----------------------------------------------


def test_align_inference_failure_raises_runtime_with_chaining(
    tmp_path,
    monkeypatch,
) -> None:
    aligner = _FakeAligner(lambda text: [], fail_align_exc=ValueError("inference boom"))
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)

    with pytest.raises(AlignmentRuntimeError) as excinfo:
        engine.align(_alignment_request(wav), tmp_path / "a.json")
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_align_malformed_shape_raises_runtime_with_chaining(
    tmp_path,
    monkeypatch,
) -> None:
    # results[0] is a list of non-item values -> AttributeError on .text.
    aligner = _FakeAligner(lambda text: [["not", "an", "item"]])
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)

    with pytest.raises(AlignmentRuntimeError) as excinfo:
        engine.align(_alignment_request(wav), tmp_path / "a.json")
    assert isinstance(excinfo.value.__cause__, AttributeError)


def test_align_empty_results_raises_runtime_with_chaining(
    tmp_path,
    monkeypatch,
) -> None:
    # Zero result sets is a runtime failure, not a mapping IndexError: the
    # exactly-one result-set count is enforced before mapping. The violated
    # count invariant is represented with a chained ValueError.
    aligner = _FakeAligner(lambda text: [])
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)

    with pytest.raises(AlignmentRuntimeError) as excinfo:
        engine.align(_alignment_request(wav), tmp_path / "a.json")
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_align_empty_content_raises_validation_error(tmp_path, monkeypatch) -> None:
    # A well-formed-but-empty alignment is a validation failure, not runtime.
    aligner = _FakeAligner(lambda text: [_FakeResult([])])
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)

    with pytest.raises(AlignmentError, match="no effective tokens"):
        engine.align(_alignment_request(wav), tmp_path / "a.json")


def test_model_load_failure_raises_runtime_with_chaining(monkeypatch) -> None:
    def boom(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("OOM loading weights")

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(
        sys.modules,
        "qwen_asr",
        types.SimpleNamespace(
            Qwen3ForcedAligner=types.SimpleNamespace(from_pretrained=boom),
        ),
    )

    with pytest.raises(AlignmentRuntimeError) as excinfo:
        _import_runtime()("Qwen/Qwen3-ForcedAligner-0.6B")
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_runtime_error_subclass_of_runtime_error() -> None:
    assert issubclass(AlignmentRuntimeError, RuntimeError)
    assert issubclass(AlignmentRequestError, ValueError)


# --- validation failure never writes output --------------------------------


def test_failed_request_validation_writes_no_file(tmp_path, monkeypatch) -> None:
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem("你", 0.0, 1.0)])])
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    out = tmp_path / "a.json"

    with pytest.raises(AlignmentRequestError):
        engine.align(_alignment_request(wav, text="   "), out)
    assert not out.exists()


def test_failed_wav_validation_writes_no_file(tmp_path, monkeypatch) -> None:
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem("你", 0.0, 1.0)])])
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "stereo.wav", n_channels=2, frames=24000)
    out = tmp_path / "a.json"

    with pytest.raises(AlignmentRequestError):
        engine.align(_alignment_request(wav), out)
    assert not out.exists()


def test_failed_alignment_validation_writes_no_file(tmp_path, monkeypatch) -> None:
    # Alignment ends past the WAV duration -> AlignmentError and no target file.
    items = [("你", 0.0, 100.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    out = tmp_path / "a.json"

    with pytest.raises(AlignmentError):
        engine.align(_alignment_request(wav), out)
    assert not out.exists()


# --- blank runtime records cannot bypass timestamp validation -------------


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("   ", float("nan"), 0.5),       # non-finite in a blank record
        ("  ", float("inf"), 0.5),        # non-finite in a blank record
        ("   ", -1.0, 0.5),               # negative start in a blank record
        ("  ", 1.0, -0.5),                # negative end in a blank record
        ("   ", 1.0, 0.0),                # end before start in a blank record
    ],
)
def test_align_rejects_blank_record_with_invalid_timestamps(
    tmp_path,
    monkeypatch,
    text,
    start,
    end,
) -> None:
    # A whitespace-only runtime record with an invalid timestamp must NOT be
    # silently dropped-and-written: it is rejected at the alignment boundary.
    items = [_FakeItem(text, start, end)]
    aligner = _FakeAligner(lambda t: [_FakeResult(items)])
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    out = tmp_path / "a.json"

    with pytest.raises(AlignmentError):
        engine.align(_alignment_request(wav), out)
    assert not out.exists()  # no new output is produced


def test_align_rejects_blank_record_with_valid_timestamps(
    tmp_path,
    monkeypatch,
) -> None:
    # A whitespace-only runtime record is invalid Production Alignment output
    # and is rejected, even when its timestamps are otherwise valid. Production
    # output is preserved, never normalized away.
    items = [_FakeItem("   ", 0.0, 0.5), _FakeItem("你", 0.5, 1.0)]
    aligner = _FakeAligner(lambda t: [_FakeResult(items)])
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    out = tmp_path / "a.json"

    with pytest.raises(AlignmentError):
        engine.align(_alignment_request(wav), out)
    assert not out.exists()  # no new output is produced


# --- validation failure preserves an existing target (Finding A) -----------


def test_failed_validation_does_not_overwrite_existing_output(
    tmp_path,
    monkeypatch,
) -> None:
    # Sentinel: pre-create the output with known content, then trigger a
    # validation failure before writing. The existing content must be unchanged,
    # proving "failed validation does not overwrite output" (no atomic-write
    # redesign; crash-safe replacement is outside v0).
    out = tmp_path / "a.json"
    original = json.dumps(
        [{"text": "preserved", "start": 0.0, "end": 1.0}]
    ) + "\n"
    out.write_text(original, encoding="utf-8")

    # Alignment ends past the WAV duration -> AlignmentError, nothing written.
    items = [("你", 0.0, 100.0)]
    aligner = _FakeAligner(lambda t: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)

    with pytest.raises(AlignmentError):
        engine.align(_alignment_request(wav), out)

    assert out.read_text(encoding="utf-8") == original


# --- output must not overwrite the input WAV -------------------------------


def test_align_refuses_to_overwrite_input_wav(tmp_path, monkeypatch) -> None:
    items = [("你", 0.0, 1.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=24000, fill=1234)

    with pytest.raises(AlignmentRequestError, match="must not overwrite"):
        engine.align(_alignment_request(wav), wav)

    # The input WAV is untouched and still readable as PCM.
    samples, frame_rate, frames = read_wav(wav)
    assert frame_rate == 24000
    assert frames == 24000
    assert samples[0] == 1234


# --- compatibility with immutable fixture + downstream stages --------------


def test_alignment_output_consumed_by_postprocess_and_captions(
    tmp_path,
    monkeypatch,
) -> None:
    # The fixture alignment is the "model output". Feed it through the engine,
    # then through Audio Postprocess and the Caption Compiler, without touching
    # the immutable fixtures.
    raw = json.loads(FIXTURE_ALIGNMENT.read_text(encoding="utf-8"))
    items = [
        _FakeItem(str(d["text"]), float(d["start"]), float(d["end"])) for d in raw
    ]

    aligner = _FakeAligner(lambda text: [_FakeResult(items)])
    engine = _engine(monkeypatch, aligner)

    out_alignment = tmp_path / "alignment.json"
    engine.align(
        _alignment_request(
            FIXTURE_WAV, text=FIXTURE_TEXT.read_text(encoding="utf-8")
        ),
        out_alignment,
    )

    # Immutable fixtures were only read, never modified.
    assert FIXTURE_WAV.is_file()
    assert FIXTURE_ALIGNMENT.is_file()

    # The engine's JSON carries the same records as the immutable fixture
    # alignment, proving the mapping step faithfully reproduces the reviewed
    # model output (text, start, and end are identical to 3-decimal precision).
    produced = json.loads(out_alignment.read_text(encoding="utf-8"))
    expected = json.loads(FIXTURE_ALIGNMENT.read_text(encoding="utf-8"))
    assert produced == expected

    cleaned_wav = tmp_path / "cleaned.wav"
    adjusted_alignment = tmp_path / "adjusted.json"
    postprocess_speech(
        FIXTURE_WAV, out_alignment, cleaned_wav, adjusted_alignment
    )

    # Audio Postprocess produced an adjusted alignment the Caption Compiler can
    # consume, and the raw fixture alignment itself still compiles to the
    # reviewed SRT (confirming the engine did not silently alter timestamps).
    frame_rate, frames = validate_wav(cleaned_wav)
    text = FIXTURE_TEXT.read_text(encoding="utf-8")
    cleaned_duration = frames / frame_rate

    adjusted_tokens = load_alignment(adjusted_alignment)
    # Audio Postprocess preserves the token count of the alignment (one token
    # per source character here) while shifting timestamps for the cleaned WAV.
    assert len(adjusted_tokens) == len(raw)
    assert adjusted_tokens[0].start >= 0.0
    assert adjusted_tokens[-1].end <= cleaned_duration + 1e-6

    srt = compile_srt(text, adjusted_tokens)
    captions = build_captions(text, adjusted_tokens)
    joined = "".join(caption.text for caption in captions)
    assert "".join(joined.split()) == "".join(text.split())
    assert srt.count("\n\n") == 1  # two captions split on the comma pause

    # The raw fixture alignment stays within the *original* WAV (8.00 s), which
    # is what the engine validated against, so compile it against that duration.
    orig_rate, orig_frames = validate_wav(FIXTURE_WAV)
    raw_tokens = validate_alignment(raw, sample_rate=orig_rate, frames=orig_frames)
    assert compile_srt(text, raw_tokens) == EXPECTED_SRT


def test_fixture_alignment_passes_validate_alignment() -> None:
    raw = json.loads(FIXTURE_ALIGNMENT.read_text(encoding="utf-8"))
    _, frame_rate, frames = read_wav(FIXTURE_WAV)
    tokens = validate_alignment(raw, sample_rate=frame_rate, frames=frames)
    assert len(tokens) == len(raw)
    # The last token ends before the 8.00 s WAV.
    assert tokens[-1].end <= frames / frame_rate


def test_downstream_captions_still_build_from_engine_alignment(
    tmp_path,
    monkeypatch,
) -> None:
    raw = json.loads(FIXTURE_ALIGNMENT.read_text(encoding="utf-8"))
    items = [
        _FakeItem(str(d["text"]), float(d["start"]), float(d["end"])) for d in raw
    ]
    aligner = _FakeAligner(lambda text: [_FakeResult(items)])
    engine = _engine(monkeypatch, aligner)

    out_alignment = tmp_path / "alignment.json"
    engine.align(
        _alignment_request(
            FIXTURE_WAV, text=FIXTURE_TEXT.read_text(encoding="utf-8")
        ),
        out_alignment,
    )
    tokens = load_alignment(out_alignment)
    captions = build_captions(FIXTURE_TEXT.read_text(encoding="utf-8"), tokens)
    assert len(captions) == 2


# =====================================================================
# gap 1: aligned text must match the original request text
# =====================================================================


def test_align_exact_match_writes_artifact(tmp_path, monkeypatch) -> None:
    items = [("你", 0.0, 0.5), ("好", 0.5, 1.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    artifact = engine.align(_alignment_request(wav, text="你好"), tmp_path / "a.json")

    written = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert written == [
        {"text": "\u4f60", "start": 0.0, "end": 0.5},
        {"text": "\u597d", "start": 0.5, "end": 1.0},
    ]
    assert artifact.token_count == 2


def test_align_missing_aligned_token_fails_matching(tmp_path, monkeypatch) -> None:
    # The request text has a spoken char beyond what the tokens cover.
    items = [("你", 0.0, 0.5)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    with pytest.raises(AlignmentMismatchError):
        engine.align(_alignment_request(wav, text="你好"), tmp_path / "a.json")


def test_align_extra_aligned_token_fails_matching(tmp_path, monkeypatch) -> None:
    # An aligned token exists past the end of the request text.
    items = [("你", 0.0, 0.5), ("好", 0.5, 1.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    with pytest.raises(AlignmentMismatchError):
        engine.align(_alignment_request(wav, text="你"), tmp_path / "a.json")


def test_align_incorrect_aligned_token_fails_matching(tmp_path, monkeypatch) -> None:
    # An aligned token's text does not appear in the request text.
    items = [("好", 0.0, 1.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    with pytest.raises(AlignmentMismatchError):
        engine.align(_alignment_request(wav, text="他"), tmp_path / "a.json")


def test_align_matching_failure_writes_no_output_file(tmp_path, monkeypatch) -> None:
    items = [("好", 0.0, 1.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    with pytest.raises(AlignmentMismatchError):
        engine.align(_alignment_request(wav, text="你"), tmp_path / "a.json")
    assert not (tmp_path / "a.json").exists()


def test_align_matching_failure_does_not_overwrite_existing_file(tmp_path, monkeypatch) -> None:
    # A stale alignment must not silently clobber a previously written artifact.
    existing = tmp_path / "a.json"
    existing.write_text('[{"text": "old"}]', encoding="utf-8")

    items = [("好", 0.0, 1.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    with pytest.raises(AlignmentMismatchError):
        engine.align(_alignment_request(wav, text="你"), tmp_path / "a.json")
    assert existing.read_text(encoding="utf-8") == '[{"text": "old"}]'


def test_align_matching_consumes_original_text_punctuation_in_order(tmp_path, monkeypatch) -> None:
    # Whitespace/punctuation in the original text is consumed from the aligned
    # tokens in order and never counts as a mismatch (matches build_captions).
    items = [("你", 0.0, 0.5), ("好", 0.5, 1.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    artifact = engine.align(_alignment_request(wav, text="你,好"), tmp_path / "a.json")
    assert artifact.token_count == 2


# =====================================================================
# gap 2: exactly one result set
# =====================================================================


def test_align_single_result_set_writes_artifact(tmp_path, monkeypatch) -> None:
    items = [("你", 0.0, 0.5), ("好", 0.5, 1.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    artifact = engine.align(_alignment_request(wav, text="你好"), tmp_path / "a.json")
    assert artifact.token_count == 2


def test_align_multiple_result_sets_raises_runtime_error(tmp_path, monkeypatch) -> None:
    items = [("你", 0.0, 0.5), ("好", 0.5, 1.0)]
    aligner = _FakeAligner(
        lambda text: [_FakeResult([_FakeItem(*it) for it in items]), _FakeResult([_FakeItem(*it) for it in items])]
    )
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    with pytest.raises(AlignmentRuntimeError):
        engine.align(_alignment_request(wav, text="你好"), tmp_path / "a.json")


def test_align_multiple_result_sets_raises_runtime_with_chaining(tmp_path, monkeypatch) -> None:
    # A result-count mismatch carries a chained ValueError representing the
    # violated exactly-one result-set invariant.
    items = [("你", 0.0, 0.5)]
    aligner = _FakeAligner(
        lambda text: [_FakeResult([_FakeItem(*it) for it in items]), _FakeResult([_FakeItem(*it) for it in items])]
    )
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    with pytest.raises(AlignmentRuntimeError) as excinfo:
        engine.align(_alignment_request(wav, text="你"), tmp_path / "a.json")
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_align_multiple_result_sets_writes_no_output_file(tmp_path, monkeypatch) -> None:
    items = [("你", 0.0, 0.5)]
    aligner = _FakeAligner(
        lambda text: [_FakeResult([_FakeItem(*it) for it in items]), _FakeResult([_FakeItem(*it) for it in items])]
    )
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    with pytest.raises(AlignmentRuntimeError):
        engine.align(_alignment_request(wav, text="你"), tmp_path / "a.json")
    assert not (tmp_path / "a.json").exists()


# --- runtime boundary: malformed result container / shape ----------------


def test_align_malformed_result_container_raises_runtime_with_chaining(
    tmp_path,
    monkeypatch,
) -> None:
    # A one-entry dict passes the len() == 1 count check, but results[0] raises
    # KeyError. This shape-access failure is translated by the runtime boundary
    # into AlignmentRuntimeError with the original exception chained, so an
    # arbitrary third-party container shape never leaks to the caller.
    items = [("你", 0.0, 0.5)]
    malformed = {"unexpected": [_FakeResult([_FakeItem(*it) for it in items])]}  # type: ignore[dict-item]
    aligner = _FakeAligner(lambda text: malformed)
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)

    with pytest.raises(AlignmentRuntimeError) as excinfo:
        engine.align(_alignment_request(wav, text="你"), tmp_path / "a.json")
    assert isinstance(excinfo.value, AlignmentRuntimeError)
    assert isinstance(excinfo.value.__cause__, KeyError)
    assert not (tmp_path / "a.json").exists()


def test_align_malformed_result_count_raises_runtime_with_chaining(
    tmp_path,
    monkeypatch,
) -> None:
    # A non-length container (len raises TypeError) is also a runtime boundary
    # failure, translated into AlignmentRuntimeError with chaining.
    class _NoLen:
        def __iter__(self):
            return iter([_FakeResult([_FakeItem("你", 0.0, 0.5)])])

    aligner = _FakeAligner(lambda text: _NoLen())
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)

    with pytest.raises(AlignmentRuntimeError) as excinfo:
        engine.align(_alignment_request(wav, text="你"), tmp_path / "a.json")
    assert isinstance(excinfo.value, AlignmentRuntimeError)
    assert isinstance(excinfo.value.__cause__, TypeError)


def test_align_runtime_boundary_does_not_swallow_alignment_error(
    tmp_path,
    monkeypatch,
) -> None:
    # The broad runtime-boundary catch must not swallow semantic Production
    # Alignment validation failures: an alignment that ends past the WAV
    # duration still raises AlignmentError, not AlignmentRuntimeError.
    items = [("你", 0.0, 100.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)

    with pytest.raises(AlignmentError):
        engine.align(_alignment_request(wav, text="你"), tmp_path / "a.json")


def test_align_runtime_boundary_does_not_swallow_mismatch_error(
    tmp_path,
    monkeypatch,
) -> None:
    # Likewise, an alignment-text mismatch must keep AlignmentMismatchError
    # rather than being wrapped as a runtime error by the boundary catch.
    items = [("好", 0.0, 1.0)]
    aligner = _FakeAligner(lambda text: [_FakeResult([_FakeItem(*it) for it in items])])
    engine = _engine(monkeypatch, aligner)
    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)

    with pytest.raises(AlignmentMismatchError):
        engine.align(_alignment_request(wav, text="你"), tmp_path / "a.json")


# =====================================================================
# gap 3: reject requests carrying nothing to align
# =====================================================================


def test_has_alignable_content_classifies_characters() -> None:
    assert has_alignable_content("\u4f60")
    assert has_alignable_content("\u4f60\u597d")
    assert not has_alignable_content("...")
    assert not has_alignable_content("\u3002\uff01\uff1f")  # 。！？
    assert not has_alignable_content("   ")
    assert not has_alignable_content("  \n\t ")
    assert not has_alignable_content("")


def test_validate_request_rejects_punctuation_only_text() -> None:
    with pytest.raises(AlignmentRequestError, match="alignable content"):
        validate_request(_alignment_request(Path("x.wav"), text="..."))


def test_validate_request_rejects_whitespace_and_punctuation_only_text() -> None:
    with pytest.raises(AlignmentRequestError, match="alignable content"):
        validate_request(_alignment_request(Path("x.wav"), text="  ...  "))


def test_validate_request_accepts_text_with_surrounding_whitespace() -> None:
    validate_request(_alignment_request(Path("x.wav"), text="  \u4f60  "))


def test_align_rejects_punctuation_only_without_running_engine(tmp_path, monkeypatch) -> None:
    aligner = _FakeAligner(lambda text: [])
    engine = _engine(monkeypatch, aligner)

    wav = _pcm_wav(tmp_path / "u.wav", frames=24000)
    with pytest.raises(AlignmentRequestError, match="alignable content"):
        engine.align(_alignment_request(wav, text="..."), tmp_path / "a.json")
    assert aligner.align_calls == []
    assert not (tmp_path / "a.json").exists()


# =====================================================================
# gap 4: malformed / corrupt / truncated input WAV
# =====================================================================


def _truncated_pcm_wav(tmp_path, path_name: str = "truncated.wav", frames: int = 24000) -> Path:
    wav = _pcm_wav(tmp_path / path_name, frames=frames)
    raw = wav.read_bytes()
    # Drop the tail of the data region: header still claims all frames, so the
    # PCM unpack fails inside read_wav with a low-level struct error.
    wav.write_bytes(raw[:-100])
    return wav


def test_validate_wav_rejects_truncated_data_region(tmp_path) -> None:
    wav = _truncated_pcm_wav(tmp_path)
    with pytest.raises(AlignmentRequestError) as excinfo:
        validate_wav(wav)
    assert isinstance(excinfo.value.__cause__, struct.error)


def test_align_rejects_truncated_wav_without_running_engine_or_output(tmp_path, monkeypatch) -> None:
    aligner = _FakeAligner(lambda text: [])
    engine = _engine(monkeypatch, aligner)

    wav = _truncated_pcm_wav(tmp_path)
    with pytest.raises(AlignmentRequestError):
        engine.align(_alignment_request(wav, text="你"), tmp_path / "a.json")
    assert aligner.align_calls == []
    assert not (tmp_path / "a.json").exists()


def test_validate_wav_rejects_non_wave_bytes(tmp_path) -> None:
    bad = tmp_path / "not.wav"
    bad.write_bytes(b"this is not a RIFF/WAVE file at all")
    with pytest.raises(AlignmentRequestError):
        validate_wav(bad)


# =====================================================================
# gap 5: missing runtime dependency crosses as AlignmentRuntimeError
# =====================================================================


def test_load_aligner_missing_torch_raises_runtime_error(monkeypatch) -> None:
    # sys.modules=None forces `import torch` to fail even on hosts where torch
    # is installed, so this is deterministic everywhere.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "qwen_asr", None)
    Qwen3ForcedAlignment = _import_runtime()
    with pytest.raises(AlignmentRuntimeError) as excinfo:
        Qwen3ForcedAlignment("/unused/model")
    assert excinfo.value.__cause__ is not None


def test_load_aligner_missing_qwen_asr_raises_runtime_error(monkeypatch) -> None:
    # torch resolves to a stub so import order reaches the qwen_asr failure.
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "qwen_asr", None)
    Qwen3ForcedAlignment = _import_runtime()
    with pytest.raises(AlignmentRuntimeError) as excinfo:
        Qwen3ForcedAlignment("/unused/model")
    assert excinfo.value.__cause__ is not None
