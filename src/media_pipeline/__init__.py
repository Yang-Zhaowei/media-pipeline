"""Media pipeline: deterministic, portable processing components.

The speech pipeline is ``text -> TTS -> forced alignment -> audio postprocess ->
caption compiler -> WAV + SRT``. It provides the portable caption compiler, the
audio postprocess stage, and the portable TTS contracts (request/output types,
validation, PCM16 conversion). All of these are pure logic that run without
CUDA.

The GPU/model runtime adapters live under :mod:`media_pipeline.runtimes` and are
intentionally **not** exported here: importing this package never pulls in
``torch``, ``qwen_tts``, or ``soundfile``.
"""

from .captions import (
    DEFAULT_BREAK_AFTER,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_DURATION,
    DEFAULT_PAUSE_THRESHOLD,
    DEFAULT_SOFT_BREAK_AFTER,
    AlignedToken,
    AlignmentError,
    AlignmentMismatchError,
    Caption,
    CaptionError,
    build_captions,
    compile_srt,
    format_timestamp,
    load_alignment,
    parse_alignment,
    render_srt,
)
from .postprocess import (
    DEFAULT_FADE_IN,
    DEFAULT_FADE_OUT,
    DEFAULT_POST_PADDING,
    DEFAULT_PRE_PADDING,
    AudioPostprocessError,
    TrimPlan,
    compute_trim,
    postprocess_speech,
    read_wav,
    sample_index,
    trim_and_fade,
    trim_alignment,
    trim_audio,
    write_wav,
)
from .tts import (
    DEFAULT_INSTRUCT,
    CustomVoiceError,
    CustomVoiceRequest,
    TTSRuntimeError,
    TTSArtifact,
    validate_request,
    waveform_to_mono_pcm16,
)

__all__ = [
    "AlignedToken",
    "AlignmentError",
    "AlignmentMismatchError",
    "Caption",
    "CaptionError",
    "CustomVoiceError",
    "CustomVoiceRequest",
    "AudioPostprocessError",
    "TTSRuntimeError",
    "TTSArtifact",
    "DEFAULT_BREAK_AFTER",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MAX_DURATION",
    "DEFAULT_INSTRUCT",
    "DEFAULT_FADE_IN",
    "DEFAULT_FADE_OUT",
    "DEFAULT_POST_PADDING",
    "DEFAULT_PAUSE_THRESHOLD",
    "DEFAULT_PRE_PADDING",
    "DEFAULT_SOFT_BREAK_AFTER",
    "TrimPlan",
    "build_captions",
    "compile_srt",
    "compute_trim",
    "format_timestamp",
    "load_alignment",
    "parse_alignment",
    "postprocess_speech",
    "read_wav",
    "render_srt",
    "sample_index",
    "trim_and_fade",
    "trim_alignment",
    "trim_audio",
    "validate_request",
    "waveform_to_mono_pcm16",
    "write_wav",
]

__version__ = "0.0.0"
