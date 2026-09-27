"""Command line entry point: ``python -m media_pipeline``.

Compiles original text plus a raw forced-alignment JSON file into SRT:

    python -m media_pipeline original.txt alignment.raw.json -o captions.srt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .captions import (
    DEFAULT_BREAK_AFTER,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_DURATION,
    AlignmentError,
    compile_srt,
    load_alignment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m media_pipeline",
        description="Compile original text and forced alignment into SRT captions.",
    )
    parser.add_argument("text", help="path to the original script text file (UTF-8)")
    parser.add_argument("alignment", help="path to the raw forced-alignment JSON file")
    parser.add_argument(
        "-o",
        "--output",
        help="path to the SRT file to write (defaults to stdout)",
    )
    parser.add_argument(
        "--break-after",
        default=DEFAULT_BREAK_AFTER,
        help="characters that always end a caption (default: %(default)r)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="maximum content characters per caption (default: %(default)s)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_MAX_DURATION,
        help="maximum caption duration in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="text and alignment file encoding (default: %(default)s)",
    )
    return parser


def _write_stdout(text: str) -> None:
    """Write text to stdout as UTF-8 regardless of the console code page."""

    stream = getattr(sys.stdout, "buffer", None)
    if stream is not None:
        stream.write(text.encode("utf-8"))
        stream.flush()
    else:  # pragma: no cover - text-only stdout replacement
        sys.stdout.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        original_text = Path(args.text).read_text(encoding=args.encoding)
        tokens = load_alignment(args.alignment, encoding=args.encoding)
        srt = compile_srt(
            original_text,
            tokens,
            break_after=args.break_after,
            max_chars=args.max_chars,
            max_duration=args.max_duration,
        )
    except (AlignmentError, OSError) as exc:
        parser.error(str(exc))
        return 2  # unreachable; parser.error raises SystemExit

    if args.output:
        Path(args.output).write_text(srt, encoding=args.encoding, newline="\n")
    else:
        _write_stdout(srt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
