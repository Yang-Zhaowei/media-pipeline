"""GPU/model runtime adapters for the Speech Pipeline.

This package contains host-specific adapters that depend on ``torch`` and the
model runtime. It is intentionally **not** exported from the package root:

    import media_pipeline   # never pulls torch/qwen_tts/soundfile

Access an adapter explicitly:

    from media_pipeline.runtimes.qwen_tts import Qwen3CustomVoiceTTS

Every heavy import lives inside the adapter; importing this package on a CPU
host (or one without the model runtime installed) does not import torch.

Runtime adapters:

- :class:`media_pipeline.runtimes.qwen_tts.Qwen3CustomVoiceTTS` (Qwen3-TTS
  CustomVoice),
- :class:`media_pipeline.runtimes.qwen_voice_clone.Qwen3VoiceCloneTTS`
  (Qwen3-TTS Base normal-ICL cloning with safe reusable local assets), and
- :class:`media_pipeline.runtimes.qwen_aligner.Qwen3ForcedAlignment` (Qwen3
  ForcedAligner).

Every heavy import lives inside the adapter; importing this package on a CPU
host (or one without the model runtime installed) does not import torch.
"""
