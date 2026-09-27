"""Qwen3-TTS CustomVoice runtime adapter (v0).

This is the ai-core GPU adapter for the TTS stage. It wraps the verified
``experiments/test_tts_custom.py`` smoke test into a production-shaped engine:

- the model loads once and the engine can synthesize many requests,
- each request is one already-segmented utterance,
- the output is written as mono 16-bit PCM WAV and reported as a
  :class:`~media_pipeline.tts.TTSArtifact`.

The heavy imports (``torch``, ``qwen_tts``) happen lazily inside
:class:`Qwen3CustomVoiceTTS`, so merely importing this module does not require
torch, CUDA, or the model runtime.

v0 is deliberately narrow: Qwen3-TTS CustomVoice only, ``torch.bfloat16``, and
``device="cuda:0"`` (configurable). There is no dtype abstraction, CPU fallback,
FlashAttention configuration, Base/VoiceDesign support, or provider abstraction.
"""

from __future__ import annotations

from pathlib import Path

from ..postprocess import read_wav, write_wav
from ..tts import (
    CustomVoiceRequest,
    TTSRuntimeError,
    TTSArtifact,
    validate_request,
    waveform_to_mono_pcm16,
)

__all__ = ["Qwen3CustomVoiceTTS"]


class Qwen3CustomVoiceTTS:
    """Qwen3-TTS CustomVoice engine: loads once, synthesizes many requests.

    ``model_path`` and ``device`` are call-site arguments; neither a model path
    nor any host path is hard-coded.
    """

    def __init__(self, model_path: str | Path, *, device: str = "cuda:0") -> None:
        import torch  # lazy: only when an engine is actually created
        from qwen_tts import Qwen3TTSModel  # lazy

        self._model_path = Path(model_path)
        self._device = device
        try:
            self._model = Qwen3TTSModel.from_pretrained(
                self._model_path,
                device_map=device,
                dtype=torch.bfloat16,
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            raise TTSRuntimeError(
                f"failed to load Qwen3-TTS CustomVoice model {model_path}: {exc}"
            ) from exc

    def get_supported_speakers(self) -> list[str] | None:
        """Return the speakers the loaded model reports as supported.

        The upstream API may return ``None``; that is reflected in the type.
        """

        return self._model.get_supported_speakers()

    def get_supported_languages(self) -> list[str] | None:
        """Return the languages the loaded model reports as supported.

        The upstream API may return ``None``; that is reflected in the type.
        """

        return self._model.get_supported_languages()

    def synthesize(
        self, request: CustomVoiceRequest, output_wav_path: str | Path
    ) -> TTSArtifact:
        """Synthesize one utterance and write a mono 16-bit PCM WAV.

        Mirrors the verified smoke test: ``generate_custom_voice`` is called
        with the request's ``text`` / ``language`` / ``speaker`` / ``instruct``,
        the first returned waveform is validated and converted to mono PCM16
        deterministically, written as uncompressed mono 16-bit PCM WAV, and read
        back so the reported ``sample_rate``/``frames`` are authoritative.
        """

        validate_request(request)

        import torch  # lazy: keep CPU imports torch-free

        model = self._model
        try:
            wavs, sample_rate = model.generate_custom_voice(
                text=request.text,
                language=request.language,
                speaker=request.speaker,
                instruct=request.instruct,
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            raise TTSRuntimeError(
                f"failed to synthesize utterance: {exc}"
            ) from exc

        # Validate the model/runtime output before writing anything.
        if not wavs:
            raise TTSRuntimeError("model returned no waveform")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise TTSRuntimeError(
                f"model returned an invalid sample rate: {sample_rate!r}"
            )

        wave = wavs[0]
        if not wave:
            raise TTSRuntimeError("model returned an empty waveform")

        samples = waveform_to_mono_pcm16(wave)
        write_wav(output_wav_path, samples, sample_rate)

        # Read back so the artifact's frame rate and frame count are authoritative.
        _, artifact_rate, artifact_frames = read_wav(output_wav_path)
        return TTSArtifact(
            wav_path=Path(output_wav_path),
            sample_rate=artifact_rate,
            frames=artifact_frames,
        )
