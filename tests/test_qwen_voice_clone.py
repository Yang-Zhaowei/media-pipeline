"""CPU runtime contracts using fake torch/Qwen, never loading real models.

The fake serializer writes JSON solely to test adapter behavior and fresh
process restoration. These tests assert restricted torch.load arguments; they
do not claim to test the real torch restricted unpickler or GPU execution.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from media_pipeline.postprocess import read_wav, write_wav
from media_pipeline.runtimes import qwen_voice_clone as runtime
from media_pipeline.tts import TTSArtifact, TTSRuntimeError
from media_pipeline.voice_clone import (
    VoiceAssetError, VoiceCloneAsset, VoiceCloneError,
    VoiceCloneReference, VoiceCloneRequest,
)


class FakeTensor:
    def __init__(self, data, dtype="float32", *, shape=None, device="cpu"):
        self.data = copy.deepcopy(data)
        self.dtype = dtype
        self.shape = tuple(shape) if shape is not None else self._shape(data)
        self.ndim = len(self.shape)
        self.device = device
        self.operations = []

    @staticmethod
    def _shape(data):
        if isinstance(data, list):
            return (len(data),) + (FakeTensor._shape(data[0]) if data else ())
        return ()

    def numel(self):
        return math.prod(self.shape)

    def is_floating_point(self):
        return self.dtype in {"float16", "float32", "float64", "bfloat16"}

    def detach(self):
        self.operations.append("detach")
        return self

    def cpu(self):
        self.operations.append("cpu")
        result = FakeTensor(self.data, self.dtype, shape=self.shape, device="cpu")
        result.operations = self.operations
        return result

    def clone(self):
        self.operations.append("clone")
        return FakeTensor(self.data, self.dtype, shape=self.shape, device=self.device)


class FakePredicate:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


def _flatten(data):
    if isinstance(data, list):
        for value in data:
            yield from _flatten(value)
    else:
        yield data


def make_torch():
    module = types.ModuleType("torch")
    module.Tensor = FakeTensor
    for dtype in ("uint8", "int8", "int16", "int32", "int64", "float16", "float32", "float64", "bfloat16"):
        setattr(module, dtype, dtype)
    module.isfinite = lambda tensor: FakePredicate(all(math.isfinite(value) for value in _flatten(tensor.data)))
    module.save_calls = []
    module.load_calls = []

    def encode(value):
        if type(value) is FakeTensor:
            return {"fake_tensor": True, "data": value.data, "dtype": value.dtype, "shape": value.shape, "device": value.device}
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [encode(item) for item in value]
        return value

    def decode(value):
        if isinstance(value, dict) and value.get("fake_tensor") is True:
            return FakeTensor(value["data"], value["dtype"], shape=value["shape"], device=value["device"])
        if isinstance(value, dict):
            return {key: decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value

    def save(payload, path):
        module.save_calls.append((payload, Path(path)))
        Path(path).write_text(json.dumps(encode(payload)), encoding="utf-8")

    def load(path, *, weights_only, map_location):
        module.load_calls.append((Path(path), weights_only, map_location))
        assert weights_only is True
        assert map_location == "cpu"
        return decode(json.loads(Path(path).read_text(encoding="utf-8")))

    module.save = save
    module.load = load
    return module


@dataclass
class FakePromptItem:
    ref_code: object
    ref_spk_embedding: object
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str


def make_asset(*, code=None, embedding=None) -> VoiceCloneAsset:
    return VoiceCloneAsset({
        "format_version": 1,
        "model": {"tokenizer_type": "qwen3_tts_tokenizer_12hz", "tts_model_size": "1b7"},
        "items": [{
            "ref_code": code if code is not None else FakeTensor([[1, 2], [3, 4]], "int64", device="cuda:0"),
            "ref_spk_embedding": embedding if embedding is not None else FakeTensor([0.25, -0.5], "bfloat16", device="cuda:0"),
            "x_vector_only_mode": False,
            "icl_mode": True,
            "ref_text": "  The exact transcript.\n",
        }],
    })


@pytest.fixture
def fake_torch(monkeypatch):
    torch = make_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


@pytest.fixture
def fake_runtime(monkeypatch, fake_torch):
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            tts_model_type="base", tokenizer_type="qwen3_tts_tokenizer_12hz", tts_model_size="1b7",
            config=types.SimpleNamespace(
                speaker_encoder_config=types.SimpleNamespace(enc_dim=2),
                talker_config=types.SimpleNamespace(num_code_groups=2),
            ),
        ),
        create_calls=[], generate_calls=[], load_calls=[],
        create_error=None, generate_error=None, load_error=None,
        create_result=None, generate_result=([[0.0, 0.5, -1.0, 1.0]], 24000),
    )

    def from_pretrained(path, **kwargs):
        model.load_calls.append((path, kwargs))
        if model.load_error:
            raise model.load_error
        return model

    def create_voice_clone_prompt(**kwargs):
        model.create_calls.append(kwargs)
        if model.create_error:
            raise model.create_error
        if model.create_result is not None:
            return model.create_result
        item = dict(make_asset().payload["items"][0])
        item["ref_text"] = kwargs["ref_text"]
        return [FakePromptItem(**item)]

    def generate_voice_clone(**kwargs):
        model.generate_calls.append(kwargs)
        if model.generate_error:
            raise model.generate_error
        return model.generate_result

    model.create_voice_clone_prompt = create_voice_clone_prompt
    model.generate_voice_clone = generate_voice_clone
    qwen = types.ModuleType("qwen_tts")
    qwen.Qwen3TTSModel = types.SimpleNamespace(from_pretrained=from_pretrained)
    inference = types.ModuleType("qwen_tts.inference")
    prompt_module = types.ModuleType("qwen_tts.inference.qwen3_tts_model")
    prompt_module.VoiceClonePromptItem = FakePromptItem
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen)
    monkeypatch.setitem(sys.modules, "qwen_tts.inference", inference)
    monkeypatch.setitem(sys.modules, "qwen_tts.inference.qwen3_tts_model", prompt_module)
    return model


def reference(tmp_path, transcript="  The exact transcript.\n") -> VoiceCloneReference:
    path = tmp_path / "private-reference-location.wav"
    write_wav(path, [0, 1000, -1000, 0], 16000)
    return VoiceCloneReference(path, transcript)


def test_serialization_round_trip_preserves_dtype_shape_data_and_copies_to_cpu(fake_torch, tmp_path) -> None:
    asset = make_asset()
    item = asset.payload["items"][0]
    path = tmp_path / "voice.pt"
    runtime.save_voice_asset(asset, path)
    restored = runtime.load_voice_asset(path)
    assert fake_torch.load_calls == [(path, True, "cpu")]
    assert restored.payload["model"] == asset.payload["model"]
    assert restored.payload["items"][0]["ref_text"] == item["ref_text"]
    for field in ("ref_code", "ref_spk_embedding"):
        original = item[field]
        loaded = restored.payload["items"][0][field]
        saved = fake_torch.save_calls[0][0]["items"][0][field]
        assert loaded.dtype == original.dtype
        assert loaded.shape == original.shape
        assert loaded.data == original.data
        assert loaded.device == saved.device == "cpu"
        assert saved is not original
        assert original.operations == ["detach", "cpu", "clone"]
    assert item["ref_code"].device == "cuda:0"
    item["ref_code"].data[0][0] = 999
    assert restored.payload["items"][0]["ref_code"].data[0][0] == 1


def test_asset_restores_in_fresh_process_without_qwen_or_reference(fake_torch, tmp_path) -> None:
    path = tmp_path / "voice.pt"
    runtime.save_voice_asset(make_asset(), path)
    src = Path(__file__).resolve().parents[1] / "src"
    tests = Path(__file__).resolve().parent
    script = f"""
import sys
sys.path[:0] = [{str(src)!r}, {str(tests)!r}]
from test_qwen_voice_clone import make_torch
sys.modules['torch'] = make_torch()
from media_pipeline.runtimes.qwen_voice_clone import load_voice_asset
asset = load_voice_asset({str(path)!r})
item = asset.payload['items'][0]
assert item['ref_code'].data == [[1, 2], [3, 4]]
assert item['ref_spk_embedding'].dtype == 'bfloat16'
assert item['ref_text'] == '  The exact transcript.\\n'
assert 'qwen_tts' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("payload", [None, {}, {"format_version": 2, "model": {}, "items": []}, {
    "format_version": 1, "model": {"tokenizer_type": "x", "tts_model_size": "1b7"}, "items": [],
}])
def test_load_rejects_malformed_or_empty_safe_payload(fake_torch, tmp_path, payload) -> None:
    path = tmp_path / "bad.pt"
    fake_torch.save(payload, path)
    with pytest.raises(VoiceAssetError):
        runtime.load_voice_asset(path)
    assert fake_torch.load_calls == [(path, True, "cpu")]


@pytest.mark.parametrize("field,value", [
    ("ref_code", [1, 2]),
    ("ref_code", FakeTensor([], "int64")),
    ("ref_code", FakeTensor([1.0], "float32")),
    ("ref_code", FakeTensor([1, 2], "int64")),
    ("ref_code", FakeTensor([[[1]]], "int64")),
    ("ref_code", FakeTensor(1, "int64")),
    ("ref_spk_embedding", [0.1, 0.2]),
    ("ref_spk_embedding", FakeTensor([], "float32")),
    ("ref_spk_embedding", FakeTensor([1, 2], "int64")),
    ("ref_spk_embedding", FakeTensor([[0.1, 0.2]], "float32")),
    ("ref_spk_embedding", FakeTensor([math.nan, 0.2], "float32")),
    ("ref_spk_embedding", FakeTensor([math.inf, 0.2], "float32")),
])
def test_save_and_load_reject_invalid_tensor_type_shape_or_values(fake_torch, tmp_path, field, value) -> None:
    asset = make_asset()
    asset.payload["items"][0][field] = value
    path = tmp_path / "invalid.pt"
    with pytest.raises(VoiceAssetError):
        runtime.save_voice_asset(asset, path)
    assert not path.exists()
    fake_torch.save(asset.payload, path)
    with pytest.raises(VoiceAssetError):
        runtime.load_voice_asset(path)


def test_tensor_subclasses_are_not_accepted_as_plain_asset_tensors(fake_torch, tmp_path) -> None:
    class TensorSubclass(FakeTensor):
        pass
    asset = make_asset(code=TensorSubclass([[1, 2]], "int64"))
    with pytest.raises(VoiceAssetError):
        runtime.save_voice_asset(asset, tmp_path / "subclass.pt")


def test_restricted_load_failure_never_retries_with_unsafe_unpickling(fake_torch, monkeypatch, tmp_path) -> None:
    calls = []
    def reject(path, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("unsupported global / malicious payload")
    monkeypatch.setattr(fake_torch, "load", reject)
    with pytest.raises(VoiceAssetError, match="safely"):
        runtime.load_voice_asset(tmp_path / "malicious.pt")
    assert calls == [{"weights_only": True, "map_location": "cpu"}]


def test_persistence_failure_is_translated(fake_torch, monkeypatch, tmp_path) -> None:
    def fail(*args, **kwargs):
        raise OSError("disk failure")
    monkeypatch.setattr(fake_torch, "save", fail)
    with pytest.raises(VoiceAssetError, match="save"):
        runtime.save_voice_asset(make_asset(), tmp_path / "voice.pt")
    with pytest.raises(VoiceAssetError, match="load"):
        runtime.load_voice_asset(tmp_path / "missing.pt")


def test_persistence_requires_torch_only_when_called(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(VoiceAssetError, match="requires torch"):
        runtime.save_voice_asset(make_asset(), tmp_path / "voice.pt")
    with pytest.raises(VoiceAssetError, match="requires torch"):
        runtime.load_voice_asset(tmp_path / "voice.pt")


def test_save_rejects_wrong_asset_wrapper_before_writing(fake_torch, tmp_path) -> None:
    with pytest.raises(VoiceAssetError):
        runtime.save_voice_asset(make_asset().payload, tmp_path / "voice.pt")
    assert fake_torch.save_calls == []


@pytest.mark.parametrize("dependency", ["torch", "qwen_tts"])
def test_missing_runtime_dependency_is_translated(fake_runtime, monkeypatch, tmp_path, dependency) -> None:
    monkeypatch.setitem(sys.modules, dependency, None)
    with pytest.raises(TTSRuntimeError, match="load"):
        runtime.Qwen3VoiceCloneTTS(tmp_path / "model")


def test_fresh_process_translates_missing_torch_without_heavy_packages() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import importlib.abc
import sys
sys.path.insert(0, {str(src)!r})
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise ModuleNotFoundError('torch deliberately unavailable')
        if fullname.split('.')[0] in {{'qwen_tts', 'numpy', 'soundfile'}}:
            raise AssertionError('unexpected heavy dependency: ' + fullname)
sys.meta_path.insert(0, BlockTorch())
from media_pipeline.runtimes.qwen_voice_clone import Qwen3VoiceCloneTTS, save_voice_asset, load_voice_asset
from media_pipeline.voice_clone import VoiceAssetError, VoiceCloneAsset
from media_pipeline.tts import TTSRuntimeError
for operation, expected in [
    (lambda: Qwen3VoiceCloneTTS('unavailable-model'), TTSRuntimeError),
    (lambda: save_voice_asset(VoiceCloneAsset({{}}), 'not-written.pt'), VoiceAssetError),
    (lambda: load_voice_asset('not-read.pt'), VoiceAssetError),
]:
    try:
        operation()
    except expected:
        pass
    else:
        raise AssertionError('missing torch was not translated')
assert not {{'torch', 'qwen_tts', 'numpy', 'soundfile'}}.intersection(sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_existing_custom_voice_keeps_request_semantics_and_loaded_engine(fake_runtime, tmp_path) -> None:
    from media_pipeline.runtimes.qwen_tts import Qwen3CustomVoiceTTS
    from media_pipeline.tts import CustomVoiceRequest

    custom_calls = []
    def generate_custom_voice(**kwargs):
        custom_calls.append(kwargs)
        return [[0.0, 0.5, -1.0, 1.0]], 24000
    fake_runtime.generate_custom_voice = generate_custom_voice
    engine = Qwen3CustomVoiceTTS(tmp_path / "custom-model", device="cpu")
    requests = [
        CustomVoiceRequest("First custom utterance.", "English", "Ryan", "Speak calmly."),
        CustomVoiceRequest("Second custom utterance.", "English", "Ryan", "Speak with energy."),
    ]
    for index, request in enumerate(requests):
        output = tmp_path / f"custom-{index}.wav"
        artifact = engine.synthesize(request, output)
        assert isinstance(artifact, TTSArtifact)
        assert artifact.wav_path == output
        assert artifact.sample_rate == 24000 and artifact.frames == 4
        assert read_wav(output) == ([0, 16384, -32768, 32767], 24000, 4)
        assert custom_calls[index] == {
            "text": request.text, "language": request.language,
            "speaker": request.speaker, "instruct": request.instruct,
        }
    assert fake_runtime.load_calls == [(tmp_path / "custom-model", {"device_map": "cpu", "dtype": "bfloat16"})]
    assert fake_runtime.create_calls == fake_runtime.generate_calls == []


def test_base_loads_once_and_repeated_synthesis_uses_restored_prompt(fake_runtime, fake_torch, tmp_path) -> None:
    engine = runtime.Qwen3VoiceCloneTTS(tmp_path / "model", device="cpu")
    ref = reference(tmp_path)
    asset = engine.create_voice_asset(ref)
    assert fake_runtime.load_calls == [(tmp_path / "model", {"device_map": "cpu", "dtype": "bfloat16"})]
    assert fake_runtime.create_calls == [{
        "ref_audio": str(ref.reference_wav), "ref_text": ref.transcript, "x_vector_only_mode": False,
    }]
    path = tmp_path / "voice.pt"
    runtime.save_voice_asset(asset, path)
    assert str(ref.reference_wav) not in path.read_text(encoding="utf-8")
    assert "private-reference-location" not in path.read_text(encoding="utf-8")
    Path(ref.reference_wav).unlink()
    restored = runtime.load_voice_asset(path)
    for index, text in enumerate(["First utterance.", "Second, different utterance."]):
        output = tmp_path / f"utterance-{index}.wav"
        artifact = engine.synthesize(VoiceCloneRequest(text, "English"), restored, output)
        assert isinstance(artifact, TTSArtifact)
        assert artifact.wav_path == output
        assert artifact.sample_rate == 24000 and artifact.frames == 4
        assert read_wav(output) == ([0, 16384, -32768, 32767], 24000, 4)
        call = fake_runtime.generate_calls[index]
        assert set(call) == {"text", "language", "voice_clone_prompt"}
        assert call["text"] == text and call["language"] == "English"
        assert len(call["voice_clone_prompt"]) == 1
        prompt = call["voice_clone_prompt"][0]
        assert type(prompt) is FakePromptItem
        assert prompt.ref_text == ref.transcript
        assert prompt.ref_code is restored.payload["items"][0]["ref_code"]
        assert prompt.ref_spk_embedding is restored.payload["items"][0]["ref_spk_embedding"]
        assert prompt.x_vector_only_mode is False and prompt.icl_mode is True
    assert len(fake_runtime.load_calls) == len(fake_runtime.create_calls) == 1


@pytest.mark.parametrize("kind", ["wrong_type", "load_failure", "wrong_schema", "unknown_tokenizer"])
def test_model_load_errors_are_translated(fake_runtime, monkeypatch, tmp_path, kind) -> None:
    if kind == "wrong_type":
        fake_runtime.model.tts_model_type = "custom_voice"
    elif kind == "load_failure":
        fake_runtime.load_error = RuntimeError("checkpoint unavailable")
    elif kind == "unknown_tokenizer":
        fake_runtime.model.tokenizer_type = "qwen3_tts_tokenizer_25hz"
    else:
        @dataclass
        class WrongPrompt:
            unexpected: object
        monkeypatch.setattr(sys.modules["qwen_tts.inference.qwen3_tts_model"], "VoiceClonePromptItem", WrongPrompt)
    with pytest.raises(TTSRuntimeError, match="load"):
        runtime.Qwen3VoiceCloneTTS(tmp_path / "model")


@pytest.mark.parametrize("kind", ["missing", "malformed", "empty"])
def test_reference_wav_failure_precedes_prompt_creation(fake_runtime, tmp_path, kind) -> None:
    engine = runtime.Qwen3VoiceCloneTTS(tmp_path / "model")
    path = tmp_path / "reference.wav"
    if kind == "malformed":
        path.write_bytes(b"not a WAV")
    elif kind == "empty":
        write_wav(path, [], 16000)
    with pytest.raises(VoiceCloneError, match="reference WAV"):
        engine.create_voice_asset(VoiceCloneReference(path, "Exact transcript"))
    assert fake_runtime.create_calls == []


def test_prompt_creation_error_is_translated(fake_runtime, tmp_path) -> None:
    engine = runtime.Qwen3VoiceCloneTTS(tmp_path / "model")
    fake_runtime.create_error = RuntimeError("failed encoding")
    with pytest.raises(TTSRuntimeError, match="create"):
        engine.create_voice_asset(reference(tmp_path))


@pytest.mark.parametrize("kind", ["empty", "dict", "wrong_item", "wrong_transcript"])
def test_prompt_creation_rejects_incompatible_result(fake_runtime, tmp_path, kind) -> None:
    engine = runtime.Qwen3VoiceCloneTTS(tmp_path / "model")
    item = FakePromptItem(**make_asset().payload["items"][0])
    fake_runtime.create_result = {"empty": [], "dict": {}, "wrong_item": [object()], "wrong_transcript": [item]}[kind]
    with pytest.raises(VoiceAssetError):
        engine.create_voice_asset(reference(tmp_path, "Different exact transcript"))


@pytest.mark.parametrize("kind", ["model", "embedding_width", "codebook_width", "code_rank", "multiple_items"])
def test_synthesis_rejects_asset_incompatible_with_loaded_checkpoint(fake_runtime, tmp_path, kind) -> None:
    engine = runtime.Qwen3VoiceCloneTTS(tmp_path / "model")
    asset = make_asset()
    if kind == "model":
        asset.payload["model"]["tts_model_size"] = "0b6"
    elif kind == "embedding_width":
        asset.payload["items"][0]["ref_spk_embedding"] = FakeTensor([0.1], "float32")
    elif kind == "codebook_width":
        asset.payload["items"][0]["ref_code"] = FakeTensor([[1, 2, 3]], "int64")
    elif kind == "code_rank":
        asset.payload["items"][0]["ref_code"] = FakeTensor([1, 2], "int64")
    else:
        asset.payload["items"].append(dict(asset.payload["items"][0]))
    with pytest.raises(VoiceAssetError):
        engine.synthesize(VoiceCloneRequest("Hello", "English"), asset, tmp_path / "out.wav")
    assert fake_runtime.generate_calls == []


def test_invalid_request_precedes_generation(fake_runtime, tmp_path) -> None:
    engine = runtime.Qwen3VoiceCloneTTS(tmp_path / "model")
    with pytest.raises(VoiceCloneError):
        engine.synthesize(VoiceCloneRequest(" ", "English"), make_asset(), tmp_path / "out.wav")
    assert fake_runtime.generate_calls == []


def test_generation_error_is_translated(fake_runtime, tmp_path) -> None:
    engine = runtime.Qwen3VoiceCloneTTS(tmp_path / "model")
    fake_runtime.generate_error = RuntimeError("model failed")
    with pytest.raises(TTSRuntimeError, match="synthesize"):
        engine.synthesize(VoiceCloneRequest("Hello", "English"), make_asset(), tmp_path / "out.wav")


@pytest.mark.parametrize("result", [
    ([], 24000), ([[0.0], [0.1]], 24000), (([0.0],), 24000),
    ([[0.0]], 0), ([[0.0]], True), ([[0.0]], 24000.0),
    ([[]], 24000), ([[math.nan]], 24000), ([[math.inf]], 24000),
    ([[[0.0, 0.1]]], 24000), ([[False]], 24000),
])
def test_output_validation_rejects_invalid_model_waveforms_or_rate(fake_runtime, tmp_path, result) -> None:
    engine = runtime.Qwen3VoiceCloneTTS(tmp_path / "model")
    fake_runtime.generate_result = result
    with pytest.raises(TTSRuntimeError):
        engine.synthesize(VoiceCloneRequest("Hello", "English"), make_asset(), tmp_path / "out.wav")
    assert not (tmp_path / "out.wav").exists()


def test_runtime_reuses_pcm_converter_and_authoritative_wav_readback(fake_runtime, monkeypatch, tmp_path) -> None:
    engine = runtime.Qwen3VoiceCloneTTS(tmp_path / "model")
    calls = []
    converter = runtime.waveform_to_mono_pcm16
    reader = runtime.read_wav

    def convert(waveform):
        calls.append(("convert", waveform))
        return converter(waveform)

    def read(path):
        calls.append(("read", path))
        samples, rate, frames = reader(path)
        return samples, rate + 1, frames + 2

    monkeypatch.setattr(runtime, "waveform_to_mono_pcm16", convert)
    monkeypatch.setattr(runtime, "read_wav", read)
    output = tmp_path / "out.wav"
    artifact = engine.synthesize(VoiceCloneRequest("Hello", "English"), make_asset(), output)
    assert calls == [("convert", [0.0, 0.5, -1.0, 1.0]), ("read", output)]
    assert artifact.sample_rate == 24001 and artifact.frames == 6


@pytest.mark.parametrize("stage", ["write_wav", "read_wav"])
def test_wav_io_errors_are_translated(fake_runtime, monkeypatch, tmp_path, stage) -> None:
    engine = runtime.Qwen3VoiceCloneTTS(tmp_path / "model")
    def fail(*args, **kwargs):
        raise OSError("WAV IO failed")
    monkeypatch.setattr(runtime, stage, fail)
    with pytest.raises(TTSRuntimeError, match="PCM16 WAV"):
        engine.synthesize(VoiceCloneRequest("Hello", "English"), make_asset(), tmp_path / "out.wav")
