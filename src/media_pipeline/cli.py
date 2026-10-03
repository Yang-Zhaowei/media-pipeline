"""Thin, JSON-output CLI for the existing production speech entry points."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .render import (
    STATUS_COMPLETE,
    RenderError,
    load_and_validate_script,
    render_speech,
)


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="media-pipeline",
        description="Validate or render caller-authored speech; results are UTF-8 JSON.",
        allow_abbrev=False,
    )
    groups = parser.add_subparsers(dest="group", required=True)
    speech = groups.add_parser("speech", allow_abbrev=False, help="speech tools")
    commands = speech.add_subparsers(dest="command", required=True)
    for command in ("validate", "render"):
        sub = commands.add_parser(command, allow_abbrev=False)
        sub.add_argument("script", help="caller-authored UTF-8 speech JSON")
        sub.add_argument(
            "--max-segment-chars",
            type=int,
            default=200,
            help="maximum Unicode code points per performance unit (default: 200)",
        )
        if command == "render":
            sub.add_argument("--output", required=True, help="new render run directory")
            for name in ("tts-python", "alignment-python", "tts-model", "alignment-model"):
                sub.add_argument(
                    "--" + name,
                    help="override MEDIA_PIPELINE_" + name.upper().replace("-", "_"),
                )
            sub.add_argument(
                "--device", default="cuda:0", help="model device (default: cuda:0)"
            )
    return parser


def _emit(command: str | None, status: str, **fields: object) -> None:
    payload = {"schema_version": 1, "command": command, "status": status, **fields}
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        sys.stdout.write(text)
    else:
        stream.write(text.encode("utf-8", errors="backslashreplace"))
        stream.flush()


def _path(path: str | Path | None) -> str | None:
    return str(Path(path).resolve()) if path is not None else None


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except _UsageError as exc:
        _emit(None, "error", error=str(exc))
        return 2

    # Static preflight always precedes runtime configuration and rendering.
    # The renderer retains its own preflight; its Python API stays untouched.
    try:
        script = load_and_validate_script(
            args.script, max_segment_chars=args.max_segment_chars
        )
    except (RenderError, OSError, UnicodeError) as exc:
        _emit(
            args.command, "validation_failed",
            script_path=_path(args.script), error=str(exc),
        )
        return 3

    if args.command == "validate":
        _emit(
            "validate", "valid", script_path=_path(args.script),
            plan={
                "language": script.language,
                "speaker": script.speaker,
                "max_segment_chars": script.max_segment_chars,
                "segments": [
                    {
                        "order": order,
                        "id": segment.id,
                        "text": segment.text,
                        "text_length": len(segment.text),
                        "pause_after_ms": segment.pause_after_ms,
                        "effective_instruct": (
                            script.instruct if segment.instruct is None else segment.instruct
                        ),
                        "instruct_source": (
                            "top_level" if segment.instruct is None else "segment"
                        ),
                    }
                    for order, segment in enumerate(script.segments, start=1)
                ],
            },
        )
        return 0

    runtime = {}
    missing = []
    for name in ("tts_python", "alignment_python", "tts_model", "alignment_model"):
        env_name = "MEDIA_PIPELINE_" + name.upper()
        # Keep the Alignment aliases and precedence used by the E2E driver.
        legacy = "MEDIA_" + name.upper() if name.startswith("alignment_") else env_name
        value = getattr(args, name)
        if value is None:
            value = os.environ.get(legacy) or os.environ.get(env_name)
        if not value or not value.strip():
            missing.append("--" + name.replace("_", "-") + " / " + env_name)
        runtime[name] = value
    if missing:
        _emit(
            "render", "error",
            error="missing runtime configuration: " + ", ".join(missing),
        )
        return 2

    try:
        result = render_speech(
            args.script, args.output, max_segment_chars=args.max_segment_chars,
            device=args.device, **runtime,
        )
    except RenderError as exc:
        run_dir = exc.run_dir
        report = run_dir / "report.json" if run_dir is not None else None
        _emit(
            "render", "failed" if run_dir is not None else "error",
            error=str(exc), run_dir=_path(run_dir),
            report_path=_path(report) if report is not None and report.is_file() else None,
            wav_path=None, srt_path=None, timeline_path=None,
        )
        return 5 if run_dir is not None else 2
    except (OSError, UnicodeError) as exc:
        _emit("render", "error", error=str(exc))
        return 2

    _emit(
        "render", result.status,
        run_dir=_path(result.run_dir), report_path=_path(result.report_path),
        wav_path=_path(result.wav_path), srt_path=_path(result.srt_path),
        timeline_path=_path(result.timeline_path),
        segments=[asdict(segment) for segment in result.segments],
    )
    return 0 if result.status == STATUS_COMPLETE else 4


if __name__ == "__main__":
    raise SystemExit(main())
