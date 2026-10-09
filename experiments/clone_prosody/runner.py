"""Fresh-process matrix orchestration; pure CPU planning and injected lifecycle."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .plan import build_matrix, load_corpus

REPO = Path(__file__).resolve().parents[2]


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def repository_state(expected_head: str) -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, check=True,
                              capture_output=True, text=True).stdout.strip()
    head = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain=v1", "--untracked-files=all"))
    if head != expected_head or dirty:
        raise ValueError(f"requires clean exact HEAD {expected_head}; HEAD={head}, dirty={dirty}")
    return {"head": head, "dirty": dirty, "branch": git("branch", "--show-current")}


def run_condition(spec: dict, directory: Path, *, asset_loader=None, engine_factory=None) -> dict:
    """One run, one asset/engine; injection validates plumbing only, never quality."""
    if asset_loader is None or engine_factory is None:
        from .runtime import ExperimentEngine, load_asset
        asset_loader = asset_loader or load_asset
        engine_factory = engine_factory or ExperimentEngine
    from media_pipeline.postprocess import read_wav, write_wav

    report = {**spec["run"], "schema_version": 1, "status": "running", "started_at": now(),
              "runtime_paths": spec["runtime_paths"], "repository": spec["repository"],
              "corpus_sha256": spec["corpus_sha256"], "outputs": [], "listening_wav": None,
              "lifecycle": {"asset_loads": 0, "model_loads": 0, "generation_invocations": 0}}
    report_path = directory / "report.json"
    stage = "asset_load"
    try:
        write_json(report_path, report)
        paths_config = spec.get("resolved_runtime_paths", spec["runtime_paths"])
        asset = asset_loader(paths_config["voice"])
        report["lifecycle"]["asset_loads"] += 1
        stage = "model_load"
        engine = engine_factory(paths_config["model"],
                                device=spec["device"], seed=report["rng_seed"])
        report["lifecycle"]["model_loads"] += 1
        stage = "runtime_provenance"
        report["runtime"] = engine.provenance(report["generation_controls"])
        write_json(report_path, report)
        paths = [directory / (item["unit_id"] + ".wav") for item in report["inputs"]]
        if report["api_topology"] == "one_list_call":
            groups = [(report["inputs"], paths, True)]
        else:
            groups = [([item], [path], False) for item, path in zip(report["inputs"], paths)]
        for inputs, targets, batch in groups:
            stage = "synthesize:" + ",".join(item["unit_id"] for item in inputs)
            report["lifecycle"]["generation_invocations"] += 1
            write_json(report_path, report)
            engine.generate(asset, inputs, spec["language"], report["generation_controls"], targets, batch=batch)
            for item, path in zip(inputs, targets):
                _, rate, frames = read_wav(path)
                if frames <= 0:
                    raise ValueError(f"empty generated WAV: {path.name}")
                report["outputs"].append({**item, "wav": path.relative_to(directory.parent).as_posix(),
                                          "sample_rate": rate, "frames": frames})
            write_json(report_path, report)
        stage = "listening_assembly"
        samples, rate = [], None
        for path in paths:
            part, part_rate, _ = read_wav(path)
            if rate is not None and rate != part_rate:
                raise ValueError("sample rates differ; no implicit resampling for listening assembly")
            rate = part_rate
            samples.extend(part)
        listening = directory / "listening.wav"
        write_wav(listening, samples, rate)
        report["listening_wav"] = listening.relative_to(directory.parent).as_posix()
        report["status"] = "complete"
        report["finished_at"] = now()
        write_json(report_path, report)
        (directory / ".complete").write_text("complete\n", encoding="utf-8")
    except Exception as exc:
        report["status"] = "failed"
        report["finished_at"] = now()
        report["failure"] = {"stage": stage, "type": type(exc).__name__, "reason": str(exc),
                             "partial_artifacts": sorted(p.name for p in directory.iterdir())}
        # A publication error after writing status complete still invalidates it.
        (directory / ".complete").unlink(missing_ok=True)
        write_json(report_path, report)
    return report


def run_experiment(output: Path, *, model: str, voice: str, device: str,
                   tts_python: str, expected_head: str, conditions=None,
                   _execute=None, _repository=None) -> dict:
    """Continue independent failed conditions; incomplete matrices exit nonzero."""
    from .runtime import sha256_file
    state = _repository or repository_state(expected_head)
    matrix = build_matrix(conditions=conditions)
    model_path, voice_path = Path(model).resolve(), Path(voice).resolve()
    if _execute is None and (not model_path.is_dir() or not voice_path.is_file()):
        raise ValueError("select an existing local Base model directory and voice asset")
    output = output.resolve()
    # Keep generated artifacts outside the checkout so exact-head checks stay clean.
    if output == REPO or REPO in output.parents:
        raise ValueError("experiment output must be outside the repository checkout")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**matrix, "status": "running", "started_at": now(), "repository": state,
                "runtime_paths": {"model": model, "voice": voice, "tts_python": tts_python},
                "resolved_runtime_paths": {"model": str(model_path), "voice": str(voice_path)},
                "device": device, "host": {"platform": platform.platform(), "python": sys.version},
                "runs": [], "failure_policy": "continue other runs; no experiment marker on any failure"}
    write_json(output / "manifest.json", manifest)
    write_json(output / "corpus.json", load_corpus())
    try:
        if _execute is None:
            # Hash checkpoint files once, without loading it; no transcript/tensors
            # are exported. Relative names + hashes identify all local model bytes.
            manifest["input_fingerprints"] = {
                "voice_sha256": sha256_file(voice_path),
                "model_files_sha256": {p.relative_to(model_path).as_posix(): sha256_file(p)
                                       for p in sorted(model_path.rglob("*")) if p.is_file()},
            }
            write_json(output / "manifest.json", manifest)
        by_id = {run["id"]: run for run in matrix["runs"]}
        for run_id in matrix["execution_order"]:
            directory = output / run_id
            directory.mkdir()
            spec = {"run": by_id[run_id], "runtime_paths": manifest["runtime_paths"],
                    "resolved_runtime_paths": manifest["resolved_runtime_paths"],
                    "device": device, "repository": state, "language": load_corpus()["language"],
                    "corpus_sha256": matrix["corpus_sha256"]}
            write_json(directory / "request.json", spec)
            try:
                if _execute is None:
                    env = {**os.environ, "PYTHONPATH": str(REPO / "src") + os.pathsep + str(REPO),
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
                    with (directory / "runtime.log").open("w", encoding="utf-8") as log:
                        completed = subprocess.run(
                            [tts_python, "-m", "experiments.clone_prosody", "_worker", str(directory / "request.json")],
                            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                        )
                    if not (directory / "report.json").is_file():
                        raise RuntimeError(f"worker exited {completed.returncode} without report; original output in {run_id}/runtime.log")
                    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
                    if completed.returncode != 0 and report["status"] == "complete":
                        raise RuntimeError(f"worker exited {completed.returncode} after reporting complete")
                else:
                    report = _execute(spec, directory)
            except Exception as exc:
                (directory / ".complete").unlink(missing_ok=True)
                report = {**by_id[run_id], "status": "failed", "outputs": [], "listening_wav": None,
                          "failure": {"stage": "worker_process", "type": type(exc).__name__, "reason": str(exc),
                                      "partial_artifacts": sorted(p.name for p in directory.iterdir())}}
                write_json(directory / "report.json", report)
            manifest["runs"].append(report)
            write_json(output / "manifest.json", manifest)
        manifest["status"] = "complete" if all(r["status"] == "complete" for r in manifest["runs"]) else "incomplete"
        manifest["finished_at"] = now()
        write_json(output / "manifest.json", manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {"stage": "orchestration", "type": type(exc).__name__, "reason": str(exc)}
        write_json(output / "manifest.json", manifest)
        raise
    return manifest
