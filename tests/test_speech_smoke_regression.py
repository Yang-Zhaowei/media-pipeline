"""Regression test against the real GPU smoke-test fixture.

``tests/fixtures/speech-smoke-001`` is immutable evidence produced on ai-core by
Qwen3-TTS CustomVoice plus Qwen3 ForcedAligner. These tests only read it; the
expected SRT below is the reviewed output of the caption compiler.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from media_pipeline.__main__ import main
from media_pipeline.captions import (
    DEFAULT_PAUSE_THRESHOLD,
    build_captions,
    compile_srt,
    load_alignment,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "speech-smoke-001"
ORIGINAL_TEXT = FIXTURE / "original.txt"
ALIGNMENT = FIXTURE / "alignment.raw.json"

EXPECTED_SRT = (
    "1\n"
    "00:00:01,360 --> 00:00:04,880\n"
    "真正限制本地人工智能模型使用体验的，\n"
    "\n"
    "2\n"
    "00:00:05,200 --> 00:00:07,920\n"
    "往往并不只是模型本身的参数规模。\n"
)


def _fixture_text() -> str:
    return ORIGINAL_TEXT.read_text(encoding="utf-8")


def test_fixture_is_present() -> None:
    assert ORIGINAL_TEXT.is_file()
    assert ALIGNMENT.is_file()


def test_real_alignment_compiles_to_expected_srt() -> None:
    tokens = load_alignment(ALIGNMENT)
    assert compile_srt(_fixture_text(), tokens) == EXPECTED_SRT


def test_real_captions_preserve_original_text() -> None:
    captions = build_captions(_fixture_text(), load_alignment(ALIGNMENT))
    joined = "".join(caption.text for caption in captions)
    assert "".join(joined.split()) == "".join(_fixture_text().split())


def test_real_captions_stay_within_alignment_bounds() -> None:
    tokens = load_alignment(ALIGNMENT)
    captions = build_captions(_fixture_text(), tokens)
    assert len(captions) == 2
    assert captions[0].start == 1.36
    assert captions[0].end == 4.88
    assert captions[1].start == 5.2
    assert captions[1].end == tokens[-1].end == 7.92


def test_fixture_comma_pause_is_a_measurable_pause() -> None:
    tokens = load_alignment(ALIGNMENT)
    gap = tokens[17].start - tokens[16].end
    assert gap == pytest.approx(0.32)
    assert gap >= DEFAULT_PAUSE_THRESHOLD


def test_comma_pause_splits_without_a_newline() -> None:
    # Removing the source line break must not change the caption split: the
    # 0.32s alignment gap after the comma is enough on its own.
    text_without_newline = _fixture_text().replace("\n", "")
    assert "\n" not in text_without_newline
    assert compile_srt(text_without_newline, load_alignment(ALIGNMENT)) == EXPECTED_SRT


def test_cli_writes_expected_srt(tmp_path: Path) -> None:
    output = tmp_path / "captions.srt"
    exit_code = main([str(ORIGINAL_TEXT), str(ALIGNMENT), "-o", str(output)])
    assert exit_code == 0
    assert output.read_text(encoding="utf-8") == EXPECTED_SRT


def test_cli_prints_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(ORIGINAL_TEXT), str(ALIGNMENT)]) == 0
    assert capsys.readouterr().out == EXPECTED_SRT


def test_cli_reports_malformed_alignment(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad_alignment = tmp_path / "bad.json"
    bad_alignment.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main([str(ORIGINAL_TEXT), str(bad_alignment)])
    assert excinfo.value.code == 2
    assert "JSON array" in capsys.readouterr().err


def test_cli_reports_missing_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "missing.txt"
    with pytest.raises(SystemExit) as excinfo:
        main([str(missing), str(ALIGNMENT)])
    assert excinfo.value.code == 2
    assert capsys.readouterr().err
