"""Media pipeline: deterministic, portable processing components.

The speech pipeline is ``text -> TTS -> forced alignment -> audio postprocess ->
caption compiler -> WAV + SRT``. It provides the caption compiler and the audio
postprocess stage, both pure logic that run without CUDA.
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

__all__ = [
    "AlignedToken",
    "AlignmentError",
    "AlignmentMismatchError",
    "Caption",
    "CaptionError",
    "AudioPostprocessError",
    "DEFAULT_BREAK_AFTER",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MAX_DURATION",
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
    "write_wav",
]

__version__ = "0.0.0"
