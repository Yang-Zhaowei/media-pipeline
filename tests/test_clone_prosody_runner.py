"""CPU plumbing evidence only; fake synthesis cannot validate Qwen acoustics."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.clone_prosody.plan import CONDITIONS, build_matrix, corpus_hash, joined_text, load_corpus
from experiments.clone_prosody.runner import REPO, run_condition, run_experiment
from media_pipeline.postprocess import write_wav


def test_exact_matrix_and_repeats():
    matrix = build_matrix()
    assert [r["id"] for r in matrix["runs"]] == [
        "S3-D-1", "S3-D-2", "S3-D-3", "B3-D-1", "B3-D-2", "B3-D-3",
        "L1-D-1", "L1-D-2", "L1-D-3", "S3-G-1", "S3-G-2", "L1-G-1", "L1-G-2"]
    assert set(matrix["execution_order"]) == {r["id"] for r in matrix["runs"]}
    assert matrix == build_matrix()
    assert all("2" not in r["api_topology"] for r in matrix["runs"])
    greedy = [r for r in matrix["runs"] if r["condition"].endswith("greedy")]
    assert all(r["generation_controls"] == {"do_sample": False, "subtalker_dosample": False} for r in greedy)
    assert {r["rng_seed"] for r in greedy} == {1729, 1730}


def test_unsupported_optional_conditions_omitted():
    matrix = build_matrix(batch_supported=False, greedy_supported=False)
    assert {r["condition"] for r in matrix["runs"]} == {"S3-default", "L1-default"}
    assert {r["condition"] for r in matrix["omitted"]} == {"B3-default", "S3-greedy", "L1-greedy"}


@pytest.mark.parametrize("conditions", [[], ["S3-default"] * 2, ["S2-default"]])
def test_invalid_condition_selection(conditions):
    with pytest.raises(ValueError):
        build_matrix(conditions=conditions)


def test_canonical_corpus_is_verbatim_for_every_condition():
    corpus = load_corpus()
    texts = [u["text"] for u in corpus["units"]]
    continuous, regions = joined_text(corpus)
    assert continuous == " ".join(texts)
    assert [continuous[r["start_codepoint"]:r["end_codepoint"]] for r in regions] == texts
    for run in build_matrix()["runs"]:
        assert [i["text"] for i in run["inputs"]] == ([continuous] if run["id"].startswith("L1") else texts)
    authored = json.dumps(corpus)
    assert all(s not in authored for s in ("/srv/", "C:\\", "ai-core", "voice.pt"))
    assert len(corpus_hash(corpus)) == 64


def spec(condition):
    return {"run": build_matrix(conditions=[condition])["runs"][0],
            "runtime_paths": {"model": "caller-model", "voice": "caller-asset", "tts_python": "caller-python"},
            "repository": {"head": "test-head", "dirty": False}, "device": "cpu",
            "language": "Chinese", "corpus_sha256": corpus_hash(load_corpus())}


class FakeEngine:
    def __init__(self, events, fail=None, rates=None):
        self.events, self.fail, self.rates = events, fail, rates or {}

    def provenance(self, controls):
        return {"runtime_kind": "fake; no GPU/model evidence", "effective_generation_controls": controls}

    def generate(self, asset, inputs, language, controls, paths, *, batch):
        assert asset is ASSET
        self.events.append(("generate", batch, [i["text"] for i in inputs], controls))
        for item, path in zip(inputs, paths):
            if item["unit_id"] == self.fail:
                path.write_bytes(b"partial WAV")
                raise RuntimeError("original synthesis failure")
            write_wav(path, [1200] * 1600, self.rates.get(item["unit_id"], 16000))


ASSET = object()


def execute(specification, directory, events, fail=None, rates=None):
    def load(path):
        events.append(("asset", path))
        return ASSET
    def engine(model, *, device, seed):
        events.append(("model", model, device, seed))
        return FakeEngine(events, fail, rates)
    return run_condition(specification, directory, asset_loader=load, engine_factory=engine)


@pytest.mark.parametrize("condition,calls,batch", [(c, 3 if c.startswith("S3") else 1, c.startswith("B3")) for c in CONDITIONS])
def test_lifecycle_and_files(tmp_path, condition, calls, batch):
    directory = tmp_path / "run"
    directory.mkdir()
    events = []
    report = execute(spec(condition), directory, events)
    assert report["status"] == "complete"
    assert report["lifecycle"] == {"asset_loads": 1, "model_loads": 1, "generation_invocations": calls}
    assert [e[0] for e in events] == ["asset", "model"] + ["generate"] * calls
    assert all(e[1] is batch for e in events[2:])
    assert (directory / ".complete").exists()
    assert (directory / "listening.wav").exists()
    assert len(report["outputs"]) == (1 if condition.startswith("L1") else 3)
    assert all("\\" not in o["wav"] for o in report["outputs"])


def test_partial_synthesis_failure_locatable_no_marker(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    report = execute(spec("S3-default"), directory, [], fail="discussion")
    assert report["status"] == "failed"
    assert report["failure"]["stage"] == "synthesize:discussion"
    assert report["failure"]["reason"] == "original synthesis failure"
    assert "opening.wav" in report["failure"]["partial_artifacts"]
    assert "discussion.wav" in report["failure"]["partial_artifacts"]
    assert not (directory / ".complete").exists()
    assert report["outputs"][0]["unit_id"] == "opening"


@pytest.mark.parametrize("stage", ["asset_load", "model_load", "runtime_provenance"])
def test_initialization_failures_record_original_stage(tmp_path, stage):
    def fail(*args, **kwargs):
        raise ValueError("injected initialization fault")
    directory = tmp_path / "run"
    directory.mkdir()
    engine = FakeEngine([])
    if stage == "runtime_provenance":
        engine.provenance = fail
    report = run_condition(spec("L1-default"), directory,
                           asset_loader=fail if stage == "asset_load" else lambda _: ASSET,
                           engine_factory=fail if stage == "model_load" else lambda *a, **k: engine)
    assert report["failure"]["stage"] == stage
    assert "injected" in report["failure"]["reason"]
    assert not (directory / ".complete").exists()


def test_differing_rates_fail_assembly_without_resampling(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    report = execute(spec("S3-default"), directory, [], rates={"discussion": 24000})
    assert report["failure"]["stage"] == "listening_assembly"
    assert len(report["outputs"]) == 3
    assert not (directory / ".complete").exists()


def matrix_run(root, execute_callback):
    return run_experiment(root, model="caller-model", voice="caller-voice", device="cpu",
                          tts_python="caller-python", expected_head="test-head",
                          _execute=execute_callback, _repository={"head": "test-head", "dirty": False})


def test_matrix_failure_continues_and_records_incompleteness(tmp_path):
    report = matrix_run(tmp_path / "experiment", lambda s, d: execute(s, d, [], fail="discussion" if s["run"]["id"] == "S3-D-2" else None))
    assert report["status"] == "incomplete"
    assert len(report["runs"]) == 13
    assert sum(r["status"] == "failed" for r in report["runs"]) == 1
    assert not (tmp_path / "experiment" / ".complete").exists()


def test_worker_launch_failure_recorded(tmp_path):
    def fail(*args):
        raise OSError("cannot launch interpreter")
    report = matrix_run(tmp_path / "experiment", fail)
    assert report["status"] == "incomplete"
    assert all(r["failure"]["stage"] == "worker_process" for r in report["runs"])


def test_cli_full_fake_matrix_analysis_blinding_and_marker(tmp_path, monkeypatch, capsys):
    from experiments.clone_prosody import __main__ as cli
    root = tmp_path / "experiment"
    monkeypatch.setattr(cli, "run_experiment", lambda output, **kw: matrix_run(output, lambda s, d: execute(s, d, [])))
    assert cli.main(["run", "--output", str(root), "--model", "fake", "--voice", "fake", "--expected-head", "test"]) == 0
    assert (root / ".complete").is_file()
    assert len(list((root / "listening-blind").glob("*.wav"))) == 13
    analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["status"] == "complete"
    assert {g["condition"] for g in analysis["repeat_groups"]} == set(CONDITIONS)
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_cli_incomplete_matrix_has_partial_blind_package_and_no_marker(tmp_path, monkeypatch, capsys):
    from experiments.clone_prosody import __main__ as cli
    root = tmp_path / "experiment"
    monkeypatch.setattr(cli, "run_experiment", lambda output, **kw: matrix_run(output, lambda s, d: execute(s, d, [], fail="discussion")))
    assert cli.main(["run", "--output", str(root), "--model", "fake", "--voice", "fake", "--expected-head", "test"]) == 1
    assert not (root / ".complete").exists()
    mapping = json.loads((root / "listening-mapping.json").read_text(encoding="utf-8"))
    assert mapping["incomplete"]
    assert len(mapping["samples"]) == 5  # all continuous conditions succeeded
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete"


@pytest.mark.parametrize("dirty,head", [(True, "expected"), (False, "different")])
def test_exact_clean_repository_guard(monkeypatch, dirty, head):
    from experiments.clone_prosody import runner
    def git(args, **kwargs):
        value = head if args[1:3] == ["rev-parse", "HEAD"] else (" M authored.py" if dirty else "")
        return type("Result", (), {"stdout": value})()
    monkeypatch.setattr(runner.subprocess, "run", git)
    with pytest.raises(ValueError, match="clean exact HEAD"):
        runner.repository_state("expected")


def test_output_inside_repository_refused_before_creation():
    with pytest.raises(ValueError, match="outside the repository"):
        matrix_run(REPO / "experiments" / "clone_prosody" / "artifacts", lambda *a: None)


def test_reanalysis_failure_invalidates_completion_marker(tmp_path, capsys):
    from experiments.clone_prosody import __main__ as cli
    root = tmp_path / "experiment"
    matrix_run(root, lambda s, d: execute(s, d, []))
    (root / ".complete").write_text("complete", encoding="utf-8")
    (root / "S3-D-1" / "opening.wav").unlink()
    assert cli.main(["analyze", "--output", str(root)]) == 1
    assert not (root / ".complete").exists()
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete"


def test_worker_preflight_original_error_has_structured_report(tmp_path, monkeypatch):
    from experiments.clone_prosody import __main__ as cli
    request = tmp_path / "request.json"
    request.write_text(json.dumps(spec("S3-default")), encoding="utf-8")
    def fail(*args):
        raise ValueError("original exact-head mismatch")
    monkeypatch.setattr(cli, "repository_state", fail)
    assert cli.main(["_worker", str(request)]) == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["failure"]["stage"] == "worker_preflight"
    assert report["failure"]["reason"] == "original exact-head mismatch"
    assert report["id"] == "S3-D-1"


def test_derived_analysis_failure_labels_experiment_and_blind_package_partial(tmp_path, monkeypatch, capsys):
    from experiments.clone_prosody import __main__ as cli
    root = tmp_path / "experiment"
    def run(output, **kwargs):
        manifest = matrix_run(output, lambda s, d: execute(s, d, []))
        (output / "S3-D-1" / "opening.wav").write_bytes(b"invalid raw output")
        return manifest
    monkeypatch.setattr(cli, "run_experiment", run)
    assert cli.main(["run", "--output", str(root), "--model", "fake", "--voice", "fake", "--expected-head", "test"]) == 1
    assert not (root / ".complete").exists()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete" and manifest["synthesis_status"] == "complete"
    assert json.loads((root / "listening-mapping.json").read_text(encoding="utf-8"))["incomplete"]
    assert "partial matrix" in (root / "listening-review-template.md").read_text(encoding="utf-8")


def test_blind_cli_does_not_print_condition_mapping(tmp_path, capsys):
    from experiments.clone_prosody import __main__ as cli
    root = tmp_path / "experiment"
    matrix_run(root, lambda s, d: execute(s, d, []))
    assert cli.main(["blind", "--output", str(root)]) == 0
    printed = capsys.readouterr().out
    assert all(condition not in printed for condition in CONDITIONS)
    assert "S3-D-1" not in printed
    assert json.loads(printed)["samples"] == 13


@pytest.mark.parametrize("stale", ["empty", "partial", "complete"])
def test_existing_output_refused_without_changes(tmp_path, stale):
    root = tmp_path / "experiment"
    root.mkdir()
    if stale != "empty":
        (root / (".complete" if stale == "complete" else "partial.wav")).write_bytes(b"keep")
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(FileExistsError):
        matrix_run(root, lambda *a: pytest.fail("stale root reused"))
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def test_cli_stale_root_never_rewrites_existing_manifest(tmp_path, monkeypatch, capsys):
    from experiments.clone_prosody import __main__ as cli
    root = tmp_path / "existing"
    root.mkdir()
    original = b'{"status":"complete","schema_version":1,"runs":[]}\n'
    (root / "manifest.json").write_bytes(original)
    (root / ".complete").write_text("keep", encoding="utf-8")
    monkeypatch.setattr("experiments.clone_prosody.runner.repository_state", lambda _: {"head": "test", "dirty": False})
    # Fake local inputs keep the actual exclusive-directory check reachable.
    model = tmp_path / "model"
    model.mkdir()
    voice = tmp_path / "voice.pt"
    voice.write_bytes(b"fake")
    code = cli.main(["run", "--output", str(root), "--model", str(model), "--voice", str(voice), "--expected-head", "test"])
    assert code == 1
    assert (root / "manifest.json").read_bytes() == original
    assert (root / ".complete").read_text(encoding="utf-8") == "keep"
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_heavy_import_and_production_isolation():
    code = '''
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'qwen_tts', 'numpy'}:
        raise AssertionError('heavy import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from experiments.clone_prosody.plan import build_matrix
from experiments.clone_prosody.runner import run_condition
from experiments.clone_prosody.analysis import analyze_experiment
assert len(build_matrix()['runs']) == 13
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(REPO / "src") + os.pathsep + str(REPO)})
    assert result.returncode == 0, result.stderr
    for path in (REPO / "src" / "media_pipeline").rglob("*.py"):
        assert "experiments.clone_prosody" not in path.read_text(encoding="utf-8")


def test_module_plan_always_writes_utf8_even_under_ascii_environment():
    result = subprocess.run([sys.executable, "-m", "experiments.clone_prosody", "plan"],
                            cwd=REPO, capture_output=True,
                            env={**os.environ, "PYTHONIOENCODING": "ascii",
                                 "PYTHONPATH": str(REPO / "src") + os.pathsep + str(REPO)})
    assert result.returncode == 0, result.stderr
    matrix = json.loads(result.stdout.decode("utf-8"))
    assert matrix["runs"][0]["inputs"][0]["text"] == load_corpus()["units"][0]["text"]
