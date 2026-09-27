"""Media pipeline: deterministic, portable processing components.

The speech pipeline is ``text -> TTS -> forced alignment -> audio postprocess ->
caption compiler -> WAV + SRT``. This package currently provides the caption
compiler, which is pure logic and runs without CUDA.
"""

from .captions import (
    DEFAULT_BREAK_AFTER,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_DURATION,
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

__all__ = [
    "AlignedToken",
    "AlignmentError",
    "AlignmentMismatchError",
    "Caption",
    "CaptionError",
    "DEFAULT_BREAK_AFTER",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MAX_DURATION",
    "build_captions",
    "compile_srt",
    "format_timestamp",
    "load_alignment",
    "parse_alignment",
    "render_srt",
]

__version__ = "0.0.0"
