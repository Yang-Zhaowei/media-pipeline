"""CPU-only descriptive audio analysis for the clone prosody experiment.

These measurements are descriptors, not emotion scores or quality verdicts.
Only uncompressed, signed PCM16 RIFF WAV files are accepted.
"""

from __future__ import annotations

import array
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import statistics
import struct
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ACTIVITY_BIN_SECONDS = 0.010
ACTIVITY_RMS_THRESHOLD = 0.01
BOUNDARY_WINDOW_SECONDS = 0.250
REPEAT_METRICS = (
    "duration_seconds", "chars_per_second", "rms_amplitude", "peak_amplitude",
    "crest_factor",
)
METRIC_DEFINITIONS = {
    "amplitude": "Signed PCM16 samples divided by 32768; aggregate all channels.",
    "duration_seconds": "WAV frame count divided by its sample rate; includes silence.",
    "text_characters": "Unicode non-whitespace characters, including punctuation.",
    "chars_per_second": "text_characters divided by complete WAV duration_seconds.",
    "rms_amplitude": "sqrt(mean(sample ** 2)) over every normalized channel sample.",
    "peak_amplitude": "Maximum absolute normalized channel sample.",
    "crest_factor": "peak_amplitude / rms_amplitude; null for zero RMS.",
    "activity": (
        "Non-overlapping round(sample_rate * 0.010) frame bins (at least one frame), "
        "including a shorter final bin. A bin is active when its all-channel RMS "
        "is >= 0.01 full scale. Durations weight bins by actual frame counts. "
        "This threshold detector is not a speech/voicing detector. Leading and "
        "trailing silence refer to bins before/after the first/last active bin; "
        "both equal total duration for an entirely inactive file."
    ),
    "boundaries": (
        "End/start RMS uses up to max(1, round(sample_rate * 0.250)) frames inside the "
        "first-to-last active-bin interval of each adjacent raw unit. No active "
        "bin yields null edge RMS. Raw-concatenation pause is previous trailing "
        "plus next leading inactive-bin duration, excluding any added assembly "
        "gap; null if either file has no active bin. Deltas carry no quality verdict."
    ),
    "repeat_dispersion": (
        "Group identical unit_id within one condition across successful repeats; "
        "report raw values, n, mean, sample standard deviation (null for n < 2), "
        "min, max, range, and every pairwise absolute delta. Two or three repeats "
        "do not establish statistical significance."
    ),
    "hashes": (
        "SHA256 of PCM data bytes and separately of the complete WAV file. Compare "
        "PCM hashes alongside sample rate/channel count for empirical exact replay; "
        "hash equality is not a guarantee that future GPU runs are deterministic. "
        "Exact UTF-8 text hashes guard same-unit repeat aggregation against rewritten content."
    ),
    "f0": (
        "Not measured. No validated pitch dependency is present. A new elementary "
        "autocorrelation tracker would need natural-speech voicing and octave-error "
        "validation beyond synthetic sine tests. Owner-side pitch analysis may be "
        "added separately with a documented method and valid range."
    ),
}


@dataclass(frozen=True)
class _Pcm:
    sample_rate: int
    channels: int
    frames: int
    samples: array.array
    raw: bytes
    wav_sha256: str


def _read_pcm16(path: Path) -> _Pcm:
    """Reject truncated RIFF/chunks, inconsistent PCM headers and empty audio."""
    try:
        content = path.read_bytes()
        if len(content) < 12 or content[:4] != b"RIFF" or content[8:12] != b"WAVE":
            raise ValueError("expected a RIFF WAVE file")
        riff_end = struct.unpack_from("<I", content, 4)[0] + 8
        if riff_end != len(content):
            raise ValueError("RIFF length does not match file length")
        cursor = 12
        fmt = None
        data_size = None
        while cursor < riff_end:
            if cursor + 8 > riff_end:
                raise ValueError("truncated WAV chunk header")
            chunk_id = content[cursor:cursor + 4]
            size = struct.unpack_from("<I", content, cursor + 4)[0]
            start = cursor + 8
            end = start + size
            padded_end = end + (size % 2)
            if padded_end > riff_end:
                raise ValueError("truncated WAV chunk data")
            if chunk_id == b"fmt ":
                if fmt is not None or size < 16:
                    raise ValueError("invalid or duplicate WAV format chunk")
                fmt = struct.unpack_from("<HHIIHH", content, start)
            elif chunk_id == b"data":
                if data_size is not None:
                    raise ValueError("duplicate WAV data chunk")
                data_size = size
            cursor = padded_end
        if fmt is None or data_size is None:
            raise ValueError("missing WAV format or data chunk")
        encoding, channels, rate, byte_rate, block_align, bits = fmt
        if encoding != 1 or bits != 16:
            raise ValueError("only uncompressed signed PCM16 WAV is supported")
        if rate <= 0 or channels <= 0:
            raise ValueError("WAV sample rate and channel count must be positive")
        if block_align != channels * 2 or byte_rate != rate * block_align:
            raise ValueError("inconsistent PCM16 WAV format fields")
        if data_size == 0:
            raise ValueError("empty WAV audio")
        if data_size % block_align:
            raise ValueError("WAV data is not a whole number of frames")
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            raw = wav.readframes(frames)
            if len(raw) != frames * channels * 2 or len(raw) != data_size:
                raise ValueError("truncated WAV audio")
        samples = array.array("h")
        samples.frombytes(raw)
        if sys.byteorder != "little":
            samples.byteswap()
        return _Pcm(rate, channels, frames, samples, raw, hashlib.sha256(content).hexdigest())
    except (OSError, EOFError, wave.Error, struct.error) as exc:
        raise ValueError(f"cannot read PCM16 WAV {path.name}: {exc}") from exc


def _rms(samples: Any) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(value * value for value in samples) / len(samples)) / 32768


def _activity(pcm: _Pcm) -> tuple[dict[str, Any], int | None, int | None]:
    bin_frames = max(1, round(pcm.sample_rate * ACTIVITY_BIN_SECONDS))
    first = None
    last = None
    active_frames = 0
    active_bins = 0
    total_bins = 0
    for start in range(0, pcm.frames, bin_frames):
        end = min(pcm.frames, start + bin_frames)
        total_bins += 1
        samples = pcm.samples[start * pcm.channels:end * pcm.channels]
        if _rms(samples) >= ACTIVITY_RMS_THRESHOLD:
            first = start if first is None else first
            last = end
            active_frames += end - start
            active_bins += 1
    duration = pcm.frames / pcm.sample_rate
    leading = duration if first is None else first / pcm.sample_rate
    trailing = duration if last is None else (pcm.frames - last) / pcm.sample_rate
    silent = (pcm.frames - active_frames) / pcm.sample_rate
    return ({
        "bin_frames": bin_frames,
        "bin_seconds": bin_frames / pcm.sample_rate,
        "threshold_rms": ACTIVITY_RMS_THRESHOLD,
        "total_bins": total_bins,
        "active_bins": active_bins,
        "has_active_bin": first is not None,
        "active_duration_seconds": active_frames / pcm.sample_rate,
        "silent_duration_seconds": silent,
        "active_fraction": active_frames / pcm.frames,
        "leading_silence_seconds": leading,
        "trailing_silence_seconds": trailing,
        "internal_silence_seconds": max(0.0, silent - leading - trailing),
    }, first, last)


def _metrics(pcm: _Pcm, text: str) -> dict[str, Any]:
    rms = _rms(pcm.samples)
    peak = max(abs(value) for value in pcm.samples) / 32768
    characters = sum(not character.isspace() for character in text)
    duration = pcm.frames / pcm.sample_rate
    activity, _, _ = _activity(pcm)
    return {
        "sample_rate": pcm.sample_rate,
        "channels": pcm.channels,
        "frames": pcm.frames,
        "duration_seconds": duration,
        "text_characters": characters,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars_per_second": characters / duration,
        "rms_amplitude": rms,
        "peak_amplitude": peak,
        "crest_factor": peak / rms if rms else None,
        "pcm_sha256": hashlib.sha256(pcm.raw).hexdigest(),
        "wav_sha256": pcm.wav_sha256,
        "activity": activity,
    }


def analyze_wav(path: Path, text: str) -> dict[str, Any]:
    """Measure one WAV without importing any model/runtime dependencies."""
    return _metrics(_read_pcm16(Path(path)), text)


def _relative_file(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("output WAV path must be a relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("output WAV path must stay within the experiment root")
    result = (root / Path(*relative.parts)).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("output WAV path resolves outside the experiment root")
    return result


def _edge_rms(pcm: _Pcm, edge: str) -> float | None:
    _, first, last = _activity(pcm)
    if first is None or last is None:
        return None
    count = max(1, round(pcm.sample_rate * BOUNDARY_WINDOW_SECONDS))
    start, end = (first, min(last, first + count)) if edge == "start" else (
        max(first, last - count), last
    )
    return _rms(pcm.samples[start * pcm.channels:end * pcm.channels])


def _boundary(previous: dict, following: dict, left: _Pcm, right: _Pcm) -> dict:
    end_rms = _edge_rms(left, "end")
    start_rms = _edge_rms(right, "start")
    delta = None if end_rms is None or start_rms is None else start_rms - end_rms
    left_activity = previous["metrics"]["activity"]
    right_activity = following["metrics"]["activity"]
    has_edges = left_activity["has_active_bin"] and right_activity["has_active_bin"]
    return {
        "previous_unit_id": previous["unit_id"],
        "next_unit_id": following["unit_id"],
        "window_seconds": BOUNDARY_WINDOW_SECONDS,
        "previous_end_rms": end_rms,
        "next_start_rms": start_rms,
        "start_minus_end_rms": delta,
        "absolute_rms_delta": abs(delta) if delta is not None else None,
        "previous_trailing_silence_seconds": left_activity["trailing_silence_seconds"],
        "next_leading_silence_seconds": right_activity["leading_silence_seconds"],
        "raw_concatenation_pause_seconds": (
            left_activity["trailing_silence_seconds"] +
            right_activity["leading_silence_seconds"] if has_edges else None
        ),
        "sample_rates": [left.sample_rate, right.sample_rate],
    }


def _dispersion(observations: list[dict], metric: str) -> dict:
    values = [{"run_id": item["run_id"], "repetition": item["repetition"],
               "value": item["metrics"][metric]} for item in observations]
    measured = [item for item in values if item["value"] is not None]
    numbers = [item["value"] for item in measured]
    return {
        "n": len(numbers),
        "mean": statistics.mean(numbers) if numbers else None,
        "sample_stddev": statistics.stdev(numbers) if len(numbers) > 1 else None,
        "min": min(numbers) if numbers else None,
        "max": max(numbers) if numbers else None,
        "range": max(numbers) - min(numbers) if numbers else None,
        "values": values,
        "pairwise_absolute_deltas": [
            {"left_run_id": left["run_id"], "right_run_id": right["run_id"],
             "absolute_delta": abs(left["value"] - right["value"])}
            for left, right in itertools.combinations(measured, 2)
        ],
    }


def _atomic_text(path: Path, content: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _remove_created_directory(root: Path, path: Path) -> None:
    """Delete only an owned direct child, never a symlink or escaped target."""
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if path.is_symlink() or resolved_path.parent != resolved_root:
        raise ValueError("refusing cleanup outside the experiment root")
    shutil.rmtree(path)


def _matrix_errors(manifest: dict) -> list[dict]:
    actual_ids = [run["id"] for run in manifest["runs"]]
    expected_ids = manifest.get("execution_order")
    errors = []
    if not actual_ids:
        errors.append({"run_id": "(experiment)", "stage": "manifest_validation",
                       "reason": "experiment has no recorded runs"})
    if len(set(actual_ids)) != len(actual_ids):
        errors.append({"run_id": "(experiment)", "stage": "manifest_validation",
                       "reason": "duplicate recorded run IDs"})
    if expected_ids is not None:
        if (not isinstance(expected_ids, list) or
                not all(isinstance(run_id, str) for run_id in expected_ids) or
                len(set(expected_ids)) != len(expected_ids)):
            errors.append({"run_id": "(experiment)", "stage": "manifest_validation",
                           "reason": "invalid declared execution_order"})
        elif set(expected_ids) != set(actual_ids):
            errors.append({"run_id": "(experiment)", "stage": "manifest_validation",
                           "reason": "recorded runs do not match declared execution_order",
                           "missing_run_ids": sorted(set(expected_ids) - set(actual_ids)),
                           "unexpected_run_ids": sorted(set(actual_ids) - set(expected_ids))})
    return errors


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _markdown(report: dict) -> str:
    lines = ["# Clone prosody descriptive analysis", "",
             f"Experiment status: **{report['experiment_status']}**. "
             f"Report status: **{report['status']}**. "
             f"Analysis complete: **{report['analysis_complete']}**.", "",
             "Measurements describe timing and amplitude. They do not classify "
             "emotion, identify a root cause, or rank conditions. Natural paragraph "
             "prosody can vary in pitch, intensity and timing. Two or three repeats "
             "provide exploratory evidence, not statistical significance.", "",
             "## Per-output measurements", "",
             "| Run | Unit | Rate | Duration (s) | Chars/s | RMS | Peak | Crest |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in report["runs"]:
        for output in run["outputs"]:
            if "metrics" not in output:
                continue
            metric = output["metrics"]
            lines.append(f"| {run['id']} | {output['unit_id']} | {metric['sample_rate']} | "
                         f"{_number(metric['duration_seconds'])} | "
                         f"{_number(metric['chars_per_second'])} | "
                         f"{_number(metric['rms_amplitude'])} | "
                         f"{_number(metric['peak_amplitude'])} | "
                         f"{_number(metric['crest_factor'])} |")
    lines += ["", "## Boundary descriptors", "",
              "| Run | Boundary | End RMS | Start RMS | Signed delta | Raw pause (s) |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for run in report["runs"]:
        for boundary in run["boundaries"]:
            lines.append(f"| {run['id']} | {boundary['previous_unit_id']} → "
                         f"{boundary['next_unit_id']} | "
                         f"{_number(boundary['previous_end_rms'])} | "
                         f"{_number(boundary['next_start_rms'])} | "
                         f"{_number(boundary['start_minus_end_rms'])} | "
                         f"{_number(boundary['raw_concatenation_pause_seconds'])} |")
    lines += ["", "Continuous outputs have no inferred internal acoustic boundaries. "
              "Different sample rates are measured separately; no resampling is performed.",
              "", "## Same-unit repeat dispersion", "",
              "| Condition | Unit | Metric | n | Mean | Sample SD | Min | Max | Range |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group in report["repeat_groups"]:
        for name, summary in group["metrics"].items():
            lines.append(f"| {group['condition']} | {group['unit_id']} | {name} | "
                         f"{summary['n']} | {_number(summary['mean'])} | "
                         f"{_number(summary['sample_stddev'])} | {_number(summary['min'])} | "
                         f"{_number(summary['max'])} | {_number(summary['range'])} |")
    lines += ["", "Raw repeat values and all pairwise absolute deltas are in analysis.json.",
              "", "## Incompleteness and errors", ""]
    if report["errors"]:
        for error in report["errors"]:
            lines.append(f"- {error['run_id']} / {error.get('unit_id', 'run')} / "
                         f"{error['stage']}: {error['reason']}")
    else:
        lines.append("No analysis errors. See experiment status for synthesis completeness.")
    lines += ["", "## Metric definitions", ""]
    for name, description in METRIC_DEFINITIONS.items():
        lines.append(f"- **{name}**: {description}")
    return "\n".join(lines) + "\n"


def analyze_experiment(root: Path) -> dict:
    """Write derived reports; preserve synthesis failures and analyze other runs.

    Regeneration replaces analysis.json and analysis.md atomically per file.
    A failed output excludes its entire run from repeat comparisons.
    """
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("runs"), list):
        raise ValueError("unsupported experiment manifest")
    report: dict = {
        "schema_version": 1,
        "experiment_status": manifest.get("status", "unknown"),
        "status": "incomplete",
        "analysis_complete": False,
        "incomplete": False,
        "metric_definitions": METRIC_DEFINITIONS,
        "runs": [], "repeat_groups": [], "errors": [],
    }
    report["errors"].extend(_matrix_errors(manifest))
    groups: dict[tuple[str, str], list[dict]] = {}
    for run in manifest["runs"]:
        result = {"id": run["id"], "condition": run["condition"],
                  "repetition": run["repetition"], "synthesis_status": run["status"],
                  "analysis_status": "skipped", "outputs": [], "boundaries": []}
        report["runs"].append(result)
        if run["status"] != "complete":
            failure = run.get("failure", run.get("error", "run did not complete"))
            error = {"run_id": run["id"], "condition": run["condition"],
                     "repetition": run["repetition"], "stage": "synthesis",
                     "reason": failure, "partial_artifacts": run.get("outputs", [])}
            if isinstance(failure, dict):
                error.update({"stage": failure.get("stage", "synthesis"),
                              "reason": failure.get("reason", failure),
                              "original_failure": failure,
                              "partial_artifacts": failure.get("partial_artifacts", run.get("outputs", []))})
            report["errors"].append(error)
            continue
        decoded: list[_Pcm] = []
        outputs = run.get("outputs", [])
        if "inputs" in run and ([
            (item["unit_id"], item["text"]) for item in run["inputs"]
        ] != [(item["unit_id"], item["text"]) for item in outputs]):
            result["analysis_status"] = "failed"
            report["errors"].append({"run_id": run["id"], "stage": "provenance_validation",
                                     "reason": "output texts/order do not match declared generated inputs"})
            continue
        if not outputs:
            report["errors"].append({"run_id": run["id"], "stage": "analysis",
                                     "reason": "successful run has no outputs"})
        for output in outputs:
            measured = {"unit_id": output["unit_id"], "wav": output["wav"]}
            result["outputs"].append(measured)
            try:
                pcm = _read_pcm16(_relative_file(root, output["wav"]))
                measured["metrics"] = _metrics(pcm, output["text"])
                decoded.append(pcm)
            except (ValueError, KeyError, TypeError) as exc:
                measured["error"] = str(exc)
                report["errors"].append({"run_id": run["id"],
                                         "unit_id": output["unit_id"], "stage": "analysis",
                                         "reason": str(exc), "wav": output["wav"]})
        if not outputs or len(decoded) != len(outputs):
            result["analysis_status"] = "failed"
            continue
        result["analysis_status"] = "success"
        if len(outputs) > 1:
            for index in range(len(outputs) - 1):
                result["boundaries"].append(_boundary(
                    result["outputs"][index], result["outputs"][index + 1],
                    decoded[index], decoded[index + 1],
                ))
        for output in result["outputs"]:
            key = (run["condition"], output["unit_id"])
            groups.setdefault(key, []).append({"run_id": run["id"],
                "repetition": run["repetition"], "metrics": output["metrics"]})
    for (condition, unit_id), observations in groups.items():
        if len({item["metrics"]["text_sha256"] for item in observations}) != 1:
            report["errors"].append({"run_id": ", ".join(item["run_id"] for item in observations),
                                     "unit_id": unit_id, "condition": condition,
                                     "stage": "repeat_validation",
                                     "reason": "same-unit repeat texts differ; dispersion omitted"})
            continue
        report["repeat_groups"].append({"condition": condition, "unit_id": unit_id,
            "metrics": {name: _dispersion(observations, name) for name in REPEAT_METRICS}})
    report["analysis_complete"] = not report["errors"]
    report["incomplete"] = manifest.get("status") != "complete" or bool(report["errors"])
    report["status"] = "incomplete" if report["incomplete"] else "complete"
    _atomic_text(root / "analysis.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _atomic_text(root / "analysis.md", _markdown(report))
    return report


def create_blind_package(root: Path, seed: int) -> dict:
    """Create randomized clean PCM copies; refuse any existing package artifacts.

    Mapping lives outside listening-blind. Original WAVs remain unchanged.
    An incomplete matrix may be packaged, but the mapping labels it incomplete.
    """
    root = Path(root)
    destination = root / "listening-blind"
    mapping_path = root / "listening-mapping.json"
    review_path = root / "listening-review-template.md"
    if any(path.exists() for path in (destination, mapping_path, review_path)):
        raise FileExistsError("blinded listening artifacts already exist; use a fresh experiment root")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("runs"), list):
        raise ValueError("unsupported experiment manifest")
    candidates = []
    for run in manifest["runs"]:
        if run["status"] == "complete" and run.get("listening_wav"):
            candidates.append((run, _read_pcm16(_relative_file(root, run["listening_wav"]))))
    if not candidates:
        raise ValueError("no successful full-corpus listening WAVs are available")
    random.Random(seed).shuffle(candidates)
    mapping = {"schema_version": 1, "seed": seed,
               "experiment_status": manifest.get("status", "unknown"),
               "incomplete": manifest.get("status") != "complete" or
                   any(run["status"] != "complete" for run in manifest["runs"]) or
                   bool(_matrix_errors(manifest)),
               "samples": []}
    staging = Path(tempfile.mkdtemp(prefix=".listening-blind-", dir=root))
    published = False
    try:
        rows = []
        for index, (run, pcm) in enumerate(candidates, 1):
            name = f"sample-{index:02d}.wav"
            with wave.open(str(staging / name), "wb") as wav:
                wav.setnchannels(pcm.channels)
                wav.setsampwidth(2)
                wav.setframerate(pcm.sample_rate)
                wav.writeframes(pcm.raw)
            mapping["samples"].append({"sample": f"listening-blind/{name}",
                "run_id": run["id"], "condition": run["condition"],
                "repetition": run["repetition"], "source_wav": run["listening_wav"]})
            rows.append(f"| {name} | | | | | | | |")
        review = "\n".join([
            "# Blind listening review", "",
            f"Package: {'partial matrix' if mapping['incomplete'] else 'complete matrix'}; "
            f"{len(candidates)} recorded full-corpus listening samples.", "",
            "Listen before opening listening-mapping.json or the named synthesis outputs. "
            "Rate each sample independently; do not infer a winning condition. "
            "Natural paragraph progression can include changes in pitch, intensity and timing.", "",
            "| Sample | Voice identity | Delivery continuity | Unexpected emotional jump | "
            "Natural paragraph progression | Pronunciation issues | Audio artifacts | Overall notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |", *rows, "",
            "Record listening device, date, and any repeat-listening observations: ", "",
        ])
        staging.rename(destination)
        published = True
        _atomic_text(mapping_path, json.dumps(mapping, indent=2, ensure_ascii=False) + "\n")
        _atomic_text(review_path, review)
    except Exception:
        if published:
            _remove_created_directory(root, destination)
            mapping_path.unlink(missing_ok=True)
            review_path.unlink(missing_ok=True)
        raise
    finally:
        if staging.exists():
            _remove_created_directory(root, staging)
    return mapping
