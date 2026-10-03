"""Qwen3-TTS Base normal-ICL cloning and safe local prompt persistence.

The prompt schema follows qwen-tts 0.1.1. Heavy dependencies are imported only
when loading an engine or saving/loading an asset. Persistence needs torch but
does not need qwen_tts, CUDA, a checkpoint, or the original reference WAV.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from ..postprocess import read_wav, write_wav
from ..tts import TTSArtifact, TTSRuntimeError, waveform_to_mono_pcm16
from ..voice_clone import (
    VoiceAssetError,
    VoiceCloneAsset,
    VoiceCloneError,
    VoiceCloneReference,
    VoiceCloneRequest,
    validate_asset_payload,
    validate_reference,
    validate_request,
)

__all__ = ["Qwen3VoiceCloneTTS", "save_voice_asset", "load_voice_asset"]

_ITEM_FIELDS = {
    "ref_code", "ref_spk_embedding", "x_vector_only_mode", "icl_mode", "ref_text"
}


def _asset_torch():
    try:
        import torch
    except ImportError as exc:
        raise VoiceAssetError("voice asset persistence requires torch") from exc
    return torch


def _validate_tensors(payload, torch) -> None:
    """Reject non-tensors, empty/wrong-rank tensors, and invalid tensor types."""
    validate_asset_payload(payload)
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    for item in payload["items"]:
        code, embedding = item["ref_code"], item["ref_spk_embedding"]
        if type(code) is not torch.Tensor or type(embedding) is not torch.Tensor:
            raise VoiceAssetError("prompt ref_code and ref_spk_embedding must be plain tensors")
        if code.ndim != 2 or code.numel() == 0 or code.dtype not in integer_dtypes:
            raise VoiceAssetError("ref_code must be a non-empty rank 2 integer tensor for 12Hz")
        if embedding.ndim != 1 or embedding.numel() == 0 or not embedding.is_floating_point():
            raise VoiceAssetError("ref_spk_embedding must be a non-empty rank 1 floating tensor")
        if not torch.isfinite(embedding).all().item():
            raise VoiceAssetError("ref_spk_embedding must contain finite values")


def _cpu_payload(asset: VoiceCloneAsset, torch) -> dict:
    if not isinstance(asset, VoiceCloneAsset):
        raise VoiceAssetError("expected a VoiceCloneAsset")
    _validate_tensors(asset.payload, torch)
    # Keep tensors as tensors: no dtype conversion, lists, NumPy coercion, or
    # guesses about upstream contents. Only storage/device ownership changes.
    return {
        "format_version": asset.payload["format_version"],
        "model": dict(asset.payload["model"]),
        "items": [
            {
                **item,
                "ref_code": item["ref_code"].detach().cpu().clone(),
                "ref_spk_embedding": item["ref_spk_embedding"].detach().cpu().clone(),
            }
            for item in asset.payload["items"]
        ],
    }


def save_voice_asset(asset: VoiceCloneAsset, path: str | Path) -> None:
    """Save only validated primitives and CPU tensors, without source paths."""
    torch = _asset_torch()
    try:
        payload = _cpu_payload(asset, torch)
        torch.save(payload, Path(path))
    except VoiceAssetError:
        raise
    except Exception as exc:
        raise VoiceAssetError(f"failed to save voice asset: {exc}") from exc


def load_voice_asset(path: str | Path) -> VoiceCloneAsset:
    """Load on CPU with torch's restricted unpickler; never retry unsafely.

    No globals are allowlisted, no custom object is pickled, and unknown schema
    versions/fields are rejected. Tensor dimensions are checked before use.
    """
    torch = _asset_torch()
    try:
        payload = torch.load(Path(path), weights_only=True, map_location="cpu")
        _validate_tensors(payload, torch)
        return VoiceCloneAsset(payload=payload)
    except VoiceAssetError:
        raise
    except Exception as exc:
        raise VoiceAssetError(f"failed to load voice asset safely: {exc}") from exc


class Qwen3VoiceCloneTTS:
    """One loaded Base checkpoint, one reference voice, many utterances.

    This adapter supports normal ICL only. It does not add speaker selection,
    instructions, transcription, or script/CLI integration.
    """

    def __init__(self, model_path: str | Path, *, device: str = "cuda:0") -> None:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

            if {field.name for field in fields(VoiceClonePromptItem)} != _ITEM_FIELDS:
                raise ValueError("incompatible upstream VoiceClonePromptItem schema")
            self._torch = torch
            self._prompt_item_type = VoiceClonePromptItem
            self._model = Qwen3TTSModel.from_pretrained(
                Path(model_path), device_map=device, dtype=torch.bfloat16
            )
            model = self._model.model
            if model.tts_model_type != "base":
                raise ValueError("voice cloning requires a Qwen3-TTS Base checkpoint")
            self._model_metadata = {
                "tokenizer_type": model.tokenizer_type,
                "tts_model_size": model.tts_model_size,
            }
            if any(type(value) is not str or not value for value in self._model_metadata.values()):
                raise ValueError("Base checkpoint is missing prompt compatibility metadata")
            if model.tokenizer_type != "qwen3_tts_tokenizer_12hz":
                raise ValueError("voice cloning supports Qwen3-TTS 12Hz Base checkpoints only")
            self._embedding_size = model.config.speaker_encoder_config.enc_dim
            self._codebooks = model.config.talker_config.num_code_groups
            if any(type(size) is not int or size <= 0 for size in (
                self._embedding_size, self._codebooks
            )):
                raise ValueError("Base checkpoint has invalid prompt dimensions")
        except Exception as exc:
            raise TTSRuntimeError(f"failed to load Qwen3-TTS Base model: {exc}") from exc

    def _validate_for_model(self, asset: VoiceCloneAsset) -> None:
        if not isinstance(asset, VoiceCloneAsset):
            raise VoiceAssetError("expected a VoiceCloneAsset")
        _validate_tensors(asset.payload, self._torch)
        if asset.payload["model"] != self._model_metadata:
            raise VoiceAssetError("voice asset is incompatible with this Base checkpoint")
        if len(asset.payload["items"]) != 1:
            raise VoiceAssetError("single-utterance synthesis requires exactly one prompt item")
        item = asset.payload["items"][0]
        if item["ref_spk_embedding"].shape != (self._embedding_size,):
            raise VoiceAssetError("speaker embedding shape is incompatible with this Base checkpoint")
        code = item["ref_code"]
        if code.shape[1] != self._codebooks:
            raise VoiceAssetError("reference code shape is incompatible with this Base checkpoint")

    def create_voice_asset(self, reference: VoiceCloneReference) -> VoiceCloneAsset:
        """Create a CPU prompt from a local production-format WAV + exact text.

        The caller supplies the transcript; this validates its presence, not
        whether it matches the recording. The original WAV path is never saved.
        """
        validate_reference(reference)
        try:
            _, _, frames = read_wav(reference.reference_wav)
            if frames == 0:
                raise ValueError("reference WAV has no frames")
        except Exception as exc:
            raise VoiceCloneError(f"invalid reference WAV: {exc}") from exc
        try:
            items = self._model.create_voice_clone_prompt(
                ref_audio=str(reference.reference_wav),
                ref_text=reference.transcript,
                x_vector_only_mode=False,
            )
            if type(items) is not list or len(items) != 1:
                raise VoiceAssetError("create_voice_clone_prompt must return one prompt item")
            if type(items[0]) is not self._prompt_item_type:
                raise VoiceAssetError("unexpected upstream prompt item type")
            if items[0].ref_text != reference.transcript:
                raise VoiceAssetError("upstream prompt did not preserve the exact reference transcript")
            asset = VoiceCloneAsset(payload={
                "format_version": 1,
                "model": dict(self._model_metadata),
                "items": [{name: getattr(items[0], name) for name in _ITEM_FIELDS}],
            })
            self._validate_for_model(asset)
            return VoiceCloneAsset(payload=_cpu_payload(asset, self._torch))
        except VoiceAssetError:
            raise
        except Exception as exc:
            raise TTSRuntimeError(f"failed to create voice clone prompt: {exc}") from exc

    def synthesize(
        self, request: VoiceCloneRequest, asset: VoiceCloneAsset,
        output_wav_path: str | Path,
    ) -> TTSArtifact:
        """Generate one utterance using restored upstream prompt items."""
        validate_request(request)
        self._validate_for_model(asset)
        try:
            # The upstream dict path omits ref_text tokenization. Use the list
            # path, exactly as its official persistence demo does for ICL.
            items = [self._prompt_item_type(**item) for item in asset.payload["items"]]
            wavs, sample_rate = self._model.generate_voice_clone(
                text=request.text, language=request.language, voice_clone_prompt=items,
            )
        except Exception as exc:
            raise TTSRuntimeError(f"failed to synthesize cloned utterance: {exc}") from exc
        if type(wavs) is not list or len(wavs) != 1:
            raise TTSRuntimeError("model must return exactly one waveform")
        if type(sample_rate) is not int or sample_rate <= 0:
            raise TTSRuntimeError(f"model returned an invalid sample rate: {sample_rate!r}")
        try:
            samples = waveform_to_mono_pcm16(wavs[0])
            write_wav(output_wav_path, samples, sample_rate)
            _, artifact_rate, artifact_frames = read_wav(output_wav_path)
        except Exception as exc:
            raise TTSRuntimeError(f"failed to produce cloned PCM16 WAV: {exc}") from exc
        return TTSArtifact(Path(output_wav_path), artifact_rate, artifact_frames)
