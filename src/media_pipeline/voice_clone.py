"""Portable request and envelope validation for Qwen3-TTS voice cloning.

This module deliberately imports only the Python standard library. It validates
the stable, CPU-portable asset envelope; tensor classes, dtypes, shapes, and
device-independent serialization are checked by the lazy runtime layer.
"""

from dataclasses import dataclass
from pathlib import Path


class VoiceCloneError(ValueError):
    """A voice-clone request or reference does not meet its input contract."""


class VoiceAssetError(ValueError):
    """A reusable voice asset does not meet the supported envelope contract."""


@dataclass(frozen=True)
class VoiceCloneRequest:
    """One utterance to synthesize using a loaded voice-clone engine."""

    text: str
    language: str


@dataclass(frozen=True)
class VoiceCloneReference:
    """A local reference WAV and its exact transcript used to build an asset."""

    reference_wav: str | Path
    transcript: str


@dataclass(frozen=True)
class VoiceCloneAsset:
    """A reusable prompt envelope, revalidated before saving or synthesis."""

    payload: dict[str, object]


def validate_request(request: VoiceCloneRequest) -> None:
    """Validate required request strings without changing their exact values."""

    if not isinstance(request, VoiceCloneRequest):
        raise VoiceCloneError("request must be a VoiceCloneRequest")
    if not isinstance(request.text, str) or not request.text.strip():
        raise VoiceCloneError("request text must be a nonempty string")
    if not isinstance(request.language, str) or not request.language.strip():
        raise VoiceCloneError("request language must be a nonempty string")


def validate_reference(reference: VoiceCloneReference) -> None:
    """Validate a local path and exact transcript without stripping either.

    File existence and WAV decoding are runtime responsibilities. URL and data
    URI inputs are rejected here; this function never opens or resolves paths.
    """

    if not isinstance(reference, VoiceCloneReference):
        raise VoiceCloneError("reference must be a VoiceCloneReference")
    path = reference.reference_wav
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise VoiceCloneError("reference_wav must be a nonempty local file path")
    if isinstance(path, str):
        lowered = path.strip().lower()
        if lowered.startswith(("http://", "https://", "data:")):
            raise VoiceCloneError("reference_wav must be a local file path")
    if not isinstance(reference.transcript, str) or not reference.transcript.strip():
        raise VoiceCloneError("reference transcript must be a nonempty string")


def validate_asset_payload(payload: object) -> None:
    """Validate the versioned portable envelope and required item fields.

    This checks exact key sets and ordinary Python value types. The runtime
    serializer separately validates tensor objects and their required shapes.
    """

    def fail(message: str) -> None:
        raise VoiceAssetError(message)

    if type(payload) is not dict:
        fail("voice asset payload must be a dict")
    expected_top = {"format_version", "model", "items"}
    if payload.keys() != expected_top:
        fail("voice asset payload has missing or unknown top-level fields")
    if type(payload["format_version"]) is not int or payload["format_version"] != 1:
        fail("unsupported voice asset format_version")

    model = payload["model"]
    if type(model) is not dict or model.keys() != {"tokenizer_type", "tts_model_size"}:
        fail("voice asset model must contain exactly tokenizer_type and tts_model_size")
    for field in ("tokenizer_type", "tts_model_size"):
        if type(model[field]) is not str or not model[field].strip():
            fail(f"voice asset model {field} must be a nonempty string")
    if model["tokenizer_type"] != "qwen3_tts_tokenizer_12hz":
        fail("voice asset requires the supported Qwen3-TTS 12Hz tokenizer")

    items = payload["items"]
    if type(items) is not list or not items:
        fail("voice asset items must be a nonempty list")
    expected_item = {
        "ref_code",
        "ref_spk_embedding",
        "x_vector_only_mode",
        "icl_mode",
        "ref_text",
    }
    for index, item in enumerate(items):
        if type(item) is not dict or item.keys() != expected_item:
            fail(f"voice asset item {index} has missing or unknown fields")
        if item["ref_code"] is None or item["ref_spk_embedding"] is None:
            fail(f"voice asset item {index} is missing required tensor data")
        if type(item["x_vector_only_mode"]) is not bool or type(item["icl_mode"]) is not bool:
            fail(f"voice asset item {index} mode flags must be bool")
        if item["x_vector_only_mode"] or not item["icl_mode"]:
            fail(f"voice asset item {index} must use normal ICL mode")
        if type(item["ref_text"]) is not str or not item["ref_text"].strip():
            fail(f"voice asset item {index} ref_text must be a nonempty string")
