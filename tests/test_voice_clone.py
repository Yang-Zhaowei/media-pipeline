"""Portable voice-clone contracts; no torch, Qwen, NumPy, or CUDA required."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from media_pipeline.voice_clone import (
    VoiceAssetError,
    VoiceCloneAsset,
    VoiceCloneError,
    VoiceCloneReference,
    VoiceCloneRequest,
    validate_asset_payload,
    validate_reference,
    validate_request,
)


def envelope() -> dict:
    # Portable validation checks the envelope only. Tensor validation belongs
    # to the lazy runtime; opaque sentinels require no heavy dependencies.
    return {
        "format_version": 1,
        "model": {
            "tokenizer_type": "qwen3_tts_tokenizer_12hz",
            "tts_model_size": "1b7",
        },
        "items": [{
            "ref_code": object(),
            "ref_spk_embedding": object(),
            "x_vector_only_mode": False,
            "icl_mode": True,
            "ref_text": "  This is the exact reference transcript.\n",
        }],
    }


def test_portable_and_runtime_imports_do_not_load_heavy_dependencies() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import importlib.abc
import sys
sys.path.insert(0, {str(src)!r})
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'torch', 'qwen_tts', 'numpy', 'soundfile'}}:
            raise AssertionError('heavy import attempted: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
import media_pipeline.voice_clone
import media_pipeline.runtimes.qwen_voice_clone
assert not {{'torch', 'qwen_tts', 'numpy', 'soundfile'}}.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_requests_and_reference_preserve_exact_strings_and_are_frozen() -> None:
    request = VoiceCloneRequest("  Hello!\n", " English ")
    reference = VoiceCloneReference(Path("reference.wav"), "  Hello!\n")
    validate_request(request)
    validate_reference(reference)
    assert request.text == reference.transcript == "  Hello!\n"
    assert request.language == " English "
    with pytest.raises(AttributeError):
        request.text = "changed"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        reference.transcript = "changed"  # type: ignore[misc]
    asset = VoiceCloneAsset(envelope())
    with pytest.raises(AttributeError):
        asset.payload = {}  # type: ignore[misc]


@pytest.mark.parametrize("field", ["text", "language"])
@pytest.mark.parametrize("value", ["", " \n\t", None, 42, False])
def test_request_rejects_missing_blank_or_nonstring_fields(field, value) -> None:
    values = {"text": "Hello", "language": "English", field: value}
    with pytest.raises(VoiceCloneError):
        validate_request(VoiceCloneRequest(**values))


def test_request_rejects_wrong_request_type() -> None:
    with pytest.raises(VoiceCloneError):
        validate_request({"text": "Hello", "language": "English"})


@pytest.mark.parametrize("transcript", ["", " \n\t", None, 42, False])
def test_reference_requires_supplied_nonblank_exact_transcript(transcript) -> None:
    with pytest.raises(VoiceCloneError):
        validate_reference(VoiceCloneReference("reference.wav", transcript))


@pytest.mark.parametrize("path", ["", " \t", None, 42, "https://host/ref.wav", "HTTP://host/ref.wav", "data:audio/wav;base64,AA=="])
def test_reference_rejects_invalid_or_remote_paths(path) -> None:
    with pytest.raises(VoiceCloneError):
        validate_reference(VoiceCloneReference(path, "Exact transcript"))


def test_portable_reference_validation_does_not_require_existing_file(tmp_path) -> None:
    validate_reference(VoiceCloneReference(tmp_path / "not-created.wav", "Exact"))
    with pytest.raises(VoiceCloneError):
        validate_reference("reference.wav")


def test_valid_asset_envelope_accepts_opaque_tensor_values() -> None:
    validate_asset_payload(envelope())


@pytest.mark.parametrize("payload", [None, [], "asset", {}, {"format_version": 1}])
def test_asset_rejects_wrong_or_incomplete_envelope(payload) -> None:
    with pytest.raises(VoiceAssetError):
        validate_asset_payload(payload)


@pytest.mark.parametrize("version", [0, 2, "1", True, None])
def test_asset_rejects_wrong_version_including_bool(version) -> None:
    payload = envelope()
    payload["format_version"] = version
    with pytest.raises(VoiceAssetError):
        validate_asset_payload(payload)


@pytest.mark.parametrize("items", [[], (), {}, None, [None], [{}]])
def test_asset_rejects_empty_or_malformed_items(items) -> None:
    payload = envelope()
    payload["items"] = items
    with pytest.raises(VoiceAssetError):
        validate_asset_payload(payload)


@pytest.mark.parametrize("field", ["ref_code", "ref_spk_embedding", "ref_text", "icl_mode", "x_vector_only_mode"])
def test_asset_rejects_missing_required_item_field(field) -> None:
    payload = envelope()
    del payload["items"][0][field]
    with pytest.raises(VoiceAssetError):
        validate_asset_payload(payload)


@pytest.mark.parametrize("field,value", [
    ("ref_code", None), ("ref_spk_embedding", None),
    ("ref_text", ""), ("ref_text", " \n"), ("ref_text", 4),
    ("icl_mode", False), ("icl_mode", 1),
    ("x_vector_only_mode", True), ("x_vector_only_mode", 0),
])
def test_asset_rejects_invalid_required_item_values(field, value) -> None:
    payload = envelope()
    payload["items"][0][field] = value
    with pytest.raises(VoiceAssetError):
        validate_asset_payload(payload)


@pytest.mark.parametrize("model", [None, {}, {"tokenizer_type": "x"}, {
    "tokenizer_type": "", "tts_model_size": "1b7",
}, {"tokenizer_type": "qwen3_tts_tokenizer_12hz", "tts_model_size": 1}, {
    "tokenizer_type": "qwen3_tts_tokenizer_25hz", "tts_model_size": "1b7",
}])
def test_asset_rejects_malformed_model_metadata(model) -> None:
    payload = envelope()
    payload["model"] = model
    with pytest.raises(VoiceAssetError):
        validate_asset_payload(payload)


@pytest.mark.parametrize("location", ["top", "model", "item"])
def test_asset_rejects_unknown_fields_including_source_host_paths(location) -> None:
    payload = envelope()
    target = payload if location == "top" else payload["model"] if location == "model" else payload["items"][0]
    target["reference_wav"] = "host-private-source.wav"
    with pytest.raises(VoiceAssetError):
        validate_asset_payload(payload)


def test_portable_validation_does_not_mutate_payload() -> None:
    payload = envelope()
    original = copy.copy(payload["items"][0])
    validate_asset_payload(payload)
    assert payload["items"][0] == original
