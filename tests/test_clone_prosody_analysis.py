"""Synthetic CPU tests; these do not validate a model, GPU, or voice quality."""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from experiments.clone_prosody import analysis
from experiments.clone_prosody.analysis import (
    METRIC_DEFINITIONS,
    analyze_experiment,
    analyze_wav,
    create_blind_package,
)


def _wav(path: Path, samples: list[int], rate: int = 1000, channels: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


def _run(run_id: str, files: list[tuple[str, str, str]], *, condition="S3-default",
         repetition=1, status="complete", listening_wav=None) -> dict:
    return {"id": run_id, "condition": condition, "repetition": repetition,
            "status": status, "outputs": [
                {"unit_id": unit, "text": text, "wav": filename}
                for unit, text, filename in files
            ], "listening_wav": listening_wav}


def _manifest(root: Path, runs: list[dict], status="complete") -> None:
    (root / "manifest.json").write_text(json.dumps(
        {"schema_version": 1, "status": status, "runs": runs}), encoding="utf-8")


def test_exact_constant_metrics_and_text_definition(tmp_path: Path) -> None:
    path = _wav(tmp_path / "constant.wav", [16384, -16384] * 50)
    metric = analyze_wav(path, "你 好!\n")
    assert metric["sample_rate"] == 1000
    assert metric["frames"] == 100
    assert metric["duration_seconds"] == 0.1
    assert metric["text_characters"] == 3
    assert metric["chars_per_second"] == 30
    assert metric["rms_amplitude"] == 0.5
    assert metric["peak_amplitude"] == 0.5
    assert metric["crest_factor"] == 1
    assert metric["activity"]["active_fraction"] == 1


def test_synthetic_sine_rms_peak_crest(tmp_path: Path) -> None:
    samples = [round(8192 * math.sin(2 * math.pi * 100 * index / 8000))
               for index in range(8000)]
    metric = analyze_wav(_wav(tmp_path / "sine.wav", samples, 8000), "a")
    assert metric["duration_seconds"] == 1
    assert metric["rms_amplitude"] == pytest.approx(0.25 / math.sqrt(2), abs=2e-5)
    assert metric["peak_amplitude"] == 0.25
    assert metric["crest_factor"] == pytest.approx(math.sqrt(2), abs=2e-4)


def test_multichannel_metrics_use_all_channel_samples(tmp_path: Path) -> None:
    metric = analyze_wav(_wav(tmp_path / "stereo.wav", [16384, 0] * 100,
                              channels=2), "a")
    assert metric["frames"] == 100
    assert metric["channels"] == 2
    assert metric["rms_amplitude"] == pytest.approx(0.5 / math.sqrt(2))
    assert metric["crest_factor"] == pytest.approx(math.sqrt(2))


def test_zero_audio_has_null_crest_and_no_active_edges(tmp_path: Path) -> None:
    metric = analyze_wav(_wav(tmp_path / "silent.wav", [0] * 105), "a")
    assert metric["rms_amplitude"] == metric["peak_amplitude"] == 0
    assert metric["crest_factor"] is None
    activity = metric["activity"]
    assert activity["total_bins"] == 11
    assert not activity["has_active_bin"]
    assert activity["silent_duration_seconds"] == 0.105
    assert activity["leading_silence_seconds"] == 0.105
    assert activity["trailing_silence_seconds"] == 0.105
    assert activity["internal_silence_seconds"] == 0


def test_activity_threshold_partial_bin_and_silence_definition(tmp_path: Path) -> None:
    # One inactive bin, one active bin, and one shorter inactive bin.
    metric = analyze_wav(_wav(tmp_path / "activity.wav", [327] * 10 + [328] * 10 +
                              [0] * 5), "a")
    activity = metric["activity"]
    assert activity["active_duration_seconds"] == 0.01
    assert activity["silent_duration_seconds"] == 0.015
    assert activity["active_fraction"] == 0.4
    assert activity["leading_silence_seconds"] == 0.01
    assert activity["trailing_silence_seconds"] == 0.005
    assert activity["total_bins"] == 3


def test_boundary_windows_start_at_active_edges(tmp_path: Path) -> None:
    _wav(tmp_path / "a.wav", [0] * 100 + [8192] * 300 + [0] * 50)
    _wav(tmp_path / "b.wav", [0] * 70 + [16384] * 400 + [0] * 30)
    _manifest(tmp_path, [_run("S3-D-1", [
        ("opening", "a", "a.wav"), ("discussion", "b", "b.wav")])])
    report = analyze_experiment(tmp_path)
    boundary = report["runs"][0]["boundaries"][0]
    assert boundary["previous_end_rms"] == 0.25
    assert boundary["next_start_rms"] == 0.5
    assert boundary["start_minus_end_rms"] == 0.25
    assert boundary["absolute_rms_delta"] == 0.25
    assert boundary["raw_concatenation_pause_seconds"] == pytest.approx(0.12)
    assert boundary["window_seconds"] == 0.25
    assert report["analysis_complete"]
    assert not report["incomplete"]
    assert (tmp_path / "analysis.json").is_file()
    assert "Boundary descriptors" in (tmp_path / "analysis.md").read_text(encoding="utf-8")


def test_short_active_region_and_fully_silent_boundary(tmp_path: Path) -> None:
    _wav(tmp_path / "short.wav", [0] * 50 + [8192] * 100 + [0] * 50)
    _wav(tmp_path / "silent.wav", [0] * 200)
    _manifest(tmp_path, [_run("S3-D-1", [
        ("opening", "a", "short.wav"), ("discussion", "b", "silent.wav")])])
    boundary = analyze_experiment(tmp_path)["runs"][0]["boundaries"][0]
    assert boundary["previous_end_rms"] == 0.25
    assert boundary["next_start_rms"] is None
    assert boundary["start_minus_end_rms"] is None
    assert boundary["raw_concatenation_pause_seconds"] is None


def test_differing_rates_measured_without_resampling(tmp_path: Path) -> None:
    _wav(tmp_path / "a.wav", [8192] * 100, 1000)
    _wav(tmp_path / "b.wav", [16384] * 200, 2000)
    _manifest(tmp_path, [_run("S3-D-1", [
        ("opening", "a", "a.wav"), ("discussion", "b", "b.wav")])])
    report = analyze_experiment(tmp_path)
    outputs = report["runs"][0]["outputs"]
    assert [item["metrics"]["sample_rate"] for item in outputs] == [1000, 2000]
    assert [item["metrics"]["duration_seconds"] for item in outputs] == [0.1, 0.1]
    assert report["runs"][0]["boundaries"][0]["sample_rates"] == [1000, 2000]


def test_same_unit_repeat_dispersion_and_pairwise_deltas(tmp_path: Path) -> None:
    runs = []
    for repeat, amplitude in enumerate([4096, 8192, 12288], 1):
        filename = f"repeat-{repeat}.wav"
        _wav(tmp_path / filename, [amplitude] * (repeat * 100))
        runs.append(_run(f"S3-D-{repeat}", [("opening", "ab", filename)],
                         repetition=repeat))
    # A different condition is a separate group even with the same logical unit.
    runs.append(_run("S3-G-1", [("opening", "ab", "repeat-1.wav")],
                     condition="S3-greedy"))
    _manifest(tmp_path, runs)
    report = analyze_experiment(tmp_path)
    assert len(report["repeat_groups"]) == 2
    group = report["repeat_groups"][0]
    rms = group["metrics"]["rms_amplitude"]
    assert rms["n"] == 3
    assert rms["mean"] == 0.25
    assert rms["sample_stddev"] == 0.125
    assert rms["min"] == 0.125
    assert rms["max"] == 0.375
    assert rms["range"] == 0.25
    assert [item["absolute_delta"] for item in rms["pairwise_absolute_deltas"]] == [
        0.125, 0.25, 0.125]
    assert group["metrics"]["duration_seconds"]["mean"] == pytest.approx(0.2)
    assert group["metrics"]["chars_per_second"]["values"][1]["value"] == 10
    assert report["repeat_groups"][1]["metrics"]["peak_amplitude"]["sample_stddev"] is None


def test_rewritten_repeat_content_is_not_aggregated(tmp_path: Path) -> None:
    _wav(tmp_path / "unit.wav", [4096] * 100)
    _manifest(tmp_path, [
        _run("S3-D-1", [("opening", "original", "unit.wav")]),
        _run("S3-D-2", [("opening", "rewritten", "unit.wav")], repetition=2)])
    report = analyze_experiment(tmp_path)
    assert report["status"] == "incomplete"
    assert not report["repeat_groups"]
    assert report["errors"][0]["stage"] == "repeat_validation"


def test_output_text_order_must_match_declared_generation_inputs(tmp_path: Path) -> None:
    _wav(tmp_path / "unit.wav", [4096] * 100)
    run = _run("S3-D-1", [("opening", "rewritten", "unit.wav")])
    run["inputs"] = [{"unit_id": "opening", "text": "original"}]
    _manifest(tmp_path, [run])
    report = analyze_experiment(tmp_path)
    assert report["status"] == "incomplete"
    assert report["errors"][0]["stage"] == "provenance_validation"
    assert report["runs"][0]["analysis_status"] == "failed"
    assert not report["repeat_groups"]


def test_null_crest_values_do_not_corrupt_repeat_statistics(tmp_path: Path) -> None:
    _wav(tmp_path / "silent.wav", [0] * 100)
    _manifest(tmp_path, [_run("S3-G-1", [("opening", "a", "silent.wav")])])
    crest = analyze_experiment(tmp_path)["repeat_groups"][0]["metrics"]["crest_factor"]
    assert crest["n"] == 0
    assert crest["mean"] is None
    assert crest["values"][0]["value"] is None
    assert crest["pairwise_absolute_deltas"] == []


@pytest.mark.parametrize("kind", ["empty", "truncated", "bad_riff", "rate_zero",
                                      "bad_byte_rate", "non_pcm", "partial_frame",
                                      "data_truncated"])
def test_malformed_or_empty_wav_rejected(tmp_path: Path, kind: str) -> None:
    path = _wav(tmp_path / "bad.wav", [] if kind == "empty" else [4096] * 100)
    content = bytearray(path.read_bytes())
    if kind == "truncated":
        content = content[:-3]
    elif kind == "bad_riff":
        content[:4] = b"nope"
    elif kind == "rate_zero":
        struct.pack_into("<I", content, 24, 0)
        struct.pack_into("<I", content, 28, 0)
    elif kind == "bad_byte_rate":
        struct.pack_into("<I", content, 28, 1)
    elif kind == "non_pcm":
        struct.pack_into("<H", content, 20, 3)
    elif kind == "partial_frame":
        # The chunk is structurally present but one byte is not a PCM16 frame.
        content.pop()
        content.append(0)  # RIFF pad byte, not part of data.
        struct.pack_into("<I", content, 40, 199)
    elif kind == "data_truncated":
        content = content[:-2]
        struct.pack_into("<I", content, 4, len(content) - 8)
    path.write_bytes(content)
    with pytest.raises(ValueError):
        analyze_wav(path, "a")


def test_failure_is_locatable_and_failed_analysis_run_excluded(tmp_path: Path) -> None:
    _wav(tmp_path / "good.wav", [4096] * 100)
    failed = _run("S3-D-2", [("opening", "a", "partial.wav")],
                  repetition=2, status="failed")
    failed["error"] = {"stage": "synthesize", "reason": "original failure"}
    _manifest(tmp_path, [
        _run("S3-D-1", [("opening", "a", "good.wav"),
                         ("discussion", "b", "missing.wav")]), failed,
        _run("S3-D-3", [("opening", "a", "good.wav")], repetition=3)], status="failed")
    report = analyze_experiment(tmp_path)
    assert report["incomplete"]
    assert not report["analysis_complete"]
    assert report["runs"][0]["analysis_status"] == "failed"
    assert report["errors"][0]["run_id"] == "S3-D-1"
    assert report["errors"][0]["unit_id"] == "discussion"
    assert report["errors"][1]["reason"] == "original failure"
    assert report["errors"][1]["stage"] == "synthesize"
    assert report["errors"][1]["original_failure"] == failed["error"]
    assert report["errors"][1]["partial_artifacts"] == failed["outputs"]
    assert report["repeat_groups"][0]["metrics"]["rms_amplitude"]["n"] == 1
    assert report["repeat_groups"][0]["metrics"]["rms_amplitude"]["values"][0]["run_id"] == "S3-D-3"
    assert "missing.wav" in (tmp_path / "analysis.md").read_text(encoding="utf-8")


def test_continuous_output_has_no_invented_region_boundaries(tmp_path: Path) -> None:
    _wav(tmp_path / "long.wav", [4096] * 100)
    _manifest(tmp_path, [_run("L1-D-1", [("continuous", "abc", "long.wav")],
                              condition="L1-default")])
    report = analyze_experiment(tmp_path)
    assert report["runs"][0]["boundaries"] == []
    assert "f0" in report["metric_definitions"]
    assert "Not measured" in METRIC_DEFINITIONS["f0"]


def test_success_without_outputs_is_analysis_failure(tmp_path: Path) -> None:
    _manifest(tmp_path, [_run("L1-D-1", [], condition="L1-default")])
    report = analyze_experiment(tmp_path)
    assert report["incomplete"]
    assert report["runs"][0]["analysis_status"] == "failed"


def test_declared_matrix_missing_run_cannot_report_complete(tmp_path: Path) -> None:
    _wav(tmp_path / "unit.wav", [4096] * 100)
    _manifest(tmp_path, [_run("S3-D-1", [("opening", "a", "unit.wav")])])
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["execution_order"] = ["S3-D-1", "S3-D-2"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = analyze_experiment(tmp_path)
    assert report["status"] == "incomplete"
    assert not report["analysis_complete"]
    assert report["errors"][0]["missing_run_ids"] == ["S3-D-2"]


def test_empty_or_duplicate_matrix_cannot_report_complete(tmp_path: Path) -> None:
    _manifest(tmp_path, [])
    assert analyze_experiment(tmp_path)["status"] == "incomplete"
    _wav(tmp_path / "unit.wav", [4096] * 100)
    run = _run("S3-D-1", [("opening", "a", "unit.wav")])
    _manifest(tmp_path, [run, run])
    report = analyze_experiment(tmp_path)
    assert report["status"] == "incomplete"
    assert report["errors"][0]["reason"] == "duplicate recorded run IDs"


def test_audio_hashes_allow_empirical_pcm_equality_without_claiming_determinism(tmp_path: Path) -> None:
    original = _wav(tmp_path / "original.wav", [4096] * 100)
    copied = tmp_path / "metadata.wav"
    content = bytearray(original.read_bytes())
    metadata = b"condition=S3-D-1"
    content.extend(b"LIST" + struct.pack("<I", len(metadata)) + metadata)
    if len(metadata) % 2:
        content.append(0)
    struct.pack_into("<I", content, 4, len(content) - 8)
    copied.write_bytes(content)
    left, right = analyze_wav(original, "a"), analyze_wav(copied, "a")
    assert left["pcm_sha256"] == right["pcm_sha256"]
    assert left["wav_sha256"] != right["wav_sha256"]
    _manifest(tmp_path, [_run("S3-D-1", [("opening", "a", "metadata.wav")],
                              listening_wav="metadata.wav")])
    create_blind_package(tmp_path, seed=1)
    blind = tmp_path / "listening-blind" / "sample-01.wav"
    assert b"condition" not in blind.read_bytes()
    assert analyze_wav(blind, "a")["pcm_sha256"] == right["pcm_sha256"]


@pytest.mark.parametrize("filename", ["../outside.wav", "/absolute.wav", "C:/host.wav",
                                         "folder\\unit.wav"])
def test_analysis_rejects_nonportable_or_escaping_paths(tmp_path: Path, filename: str) -> None:
    _manifest(tmp_path, [_run("S3-D-1", [("opening", "a", filename)])])
    report = analyze_experiment(tmp_path)
    assert not report["analysis_complete"]
    assert report["errors"][0]["stage"] == "analysis"


def test_reports_can_be_regenerated_without_reusing_measurements(tmp_path: Path) -> None:
    wav = _wav(tmp_path / "unit.wav", [4096] * 100)
    _manifest(tmp_path, [_run("S3-D-1", [("opening", "a", "unit.wav")])])
    assert analyze_experiment(tmp_path)["runs"][0]["outputs"][0]["metrics"]["rms_amplitude"] == 0.125
    _wav(wav, [8192] * 100)
    assert analyze_experiment(tmp_path)["runs"][0]["outputs"][0]["metrics"]["rms_amplitude"] == 0.25
    assert not list(tmp_path.glob("*.tmp"))


def test_blind_package_seed_mapping_reviews_and_original_preservation(tmp_path: Path) -> None:
    runs = []
    originals = {}
    for repeat in range(1, 4):
        filename = f"named-{repeat}.wav"
        path = _wav(tmp_path / filename, [repeat * 4096] * 100)
        originals[filename] = path.read_bytes()
        runs.append(_run(f"S3-D-{repeat}", [("opening", "a", filename)],
                         repetition=repeat, listening_wav=filename))
    _manifest(tmp_path, runs)
    mapping = create_blind_package(tmp_path, seed=42)
    assert [sample["run_id"] for sample in mapping["samples"]] == ["S3-D-2", "S3-D-1", "S3-D-3"]
    assert sorted(path.name for path in (tmp_path / "listening-blind").iterdir()) == [
        "sample-01.wav", "sample-02.wav", "sample-03.wav"]
    assert not (tmp_path / "listening-blind" / "mapping.json").exists()
    for item in mapping["samples"]:
        assert (tmp_path / item["sample"]).read_bytes() == originals[item["source_wav"]]
    for filename, content in originals.items():
        assert (tmp_path / filename).read_bytes() == content
    review = (tmp_path / "listening-review-template.md").read_text(encoding="utf-8")
    assert "S3-D" not in review
    for field in ["Voice identity", "Delivery continuity", "Unexpected emotional jump",
                  "Natural paragraph progression", "Pronunciation issues", "Audio artifacts", "Overall notes"]:
        assert field in review
    with pytest.raises(FileExistsError):
        create_blind_package(tmp_path, seed=42)


@pytest.mark.parametrize("stale", ["listening-blind", "listening-mapping.json",
                                     "listening-review-template.md"])
def test_blind_package_refuses_each_stale_artifact(tmp_path: Path, stale: str) -> None:
    (tmp_path / stale).write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_blind_package(tmp_path, seed=1)
    assert (tmp_path / stale).read_text(encoding="utf-8") == "existing"


def test_blind_package_failure_has_no_published_artifacts(tmp_path: Path) -> None:
    _wav(tmp_path / "good.wav", [4096] * 100)
    _manifest(tmp_path, [
        _run("S3-D-1", [("opening", "a", "good.wav")], listening_wav="good.wav"),
        _run("S3-D-2", [], repetition=2, listening_wav="missing.wav")])
    with pytest.raises(ValueError):
        create_blind_package(tmp_path, seed=1)
    assert not (tmp_path / "listening-blind").exists()
    assert not (tmp_path / "listening-mapping.json").exists()
    assert not (tmp_path / "listening-review-template.md").exists()


def test_blind_package_publish_failure_rolls_back_owned_paths(tmp_path: Path, monkeypatch) -> None:
    path = _wav(tmp_path / "good.wav", [4096] * 100)
    original = path.read_bytes()
    _manifest(tmp_path, [_run("S3-D-1", [("opening", "a", "good.wav")],
                              listening_wav="good.wav")])

    def fail_write(*args):
        raise OSError("injected publish failure")

    monkeypatch.setattr(analysis, "_atomic_text", fail_write)
    with pytest.raises(OSError, match="injected publish failure"):
        create_blind_package(tmp_path, seed=1)
    assert path.read_bytes() == original
    assert not (tmp_path / "listening-blind").exists()
    assert not (tmp_path / "listening-mapping.json").exists()
    assert not list(tmp_path.glob(".listening-blind-*"))


def test_cleanup_refuses_directory_outside_owned_root(tmp_path: Path) -> None:
    owned = tmp_path / "root"
    owned.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="refusing cleanup"):
        analysis._remove_created_directory(owned, outside)
    assert outside.is_dir()


def test_incomplete_blind_package_records_missing_conditions(tmp_path: Path) -> None:
    _wav(tmp_path / "good.wav", [4096] * 100)
    _manifest(tmp_path, [
        _run("S3-D-1", [("opening", "a", "good.wav")], listening_wav="good.wav"),
        _run("S3-D-2", [], status="failed", repetition=2)], status="failed")
    mapping = create_blind_package(tmp_path, seed=1)
    assert mapping["incomplete"]
    assert len(mapping["samples"]) == 1
    assert "partial matrix" in (tmp_path / "listening-review-template.md").read_text(encoding="utf-8")


def test_analysis_import_does_not_require_torch_or_qwen() -> None:
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'qwen_tts'}:
        raise AssertionError('heavy import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from experiments.clone_prosody.analysis import analyze_experiment, analyze_wav
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr
