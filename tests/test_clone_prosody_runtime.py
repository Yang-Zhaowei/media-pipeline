"""Audited bridge contracts with the existing fake Qwen runtime; no GPU claim."""

import json
from pathlib import Path

import pytest

from experiments.clone_prosody.runtime import ExperimentEngine
from media_pipeline.runtimes.qwen_voice_clone import Qwen3VoiceCloneTTS
from test_qwen_voice_clone import fake_runtime, fake_torch, make_asset  # noqa: F401


@pytest.fixture
def engine(fake_runtime, tmp_path):
    bridge = object.__new__(ExperimentEngine)
    bridge.engine = Qwen3VoiceCloneTTS(tmp_path / "model", device="cpu")
    return bridge


def test_default_uses_validated_production_synthesis_without_kwargs(engine, fake_runtime, tmp_path):
    asset = make_asset()
    inputs = [{"unit_id": "opening", "text": "Exact text."}]
    engine.generate(asset, inputs, "English", {}, [tmp_path / "opening.wav"])
    call = fake_runtime.generate_calls[0]
    assert set(call) == {"text", "language", "voice_clone_prompt"}
    assert call["text"] == "Exact text."
    assert len(fake_runtime.load_calls) == 1


def test_batch_is_one_upstream_list_invocation(engine, fake_runtime, tmp_path):
    inputs = [{"unit_id": name, "text": f"Exact {name}."} for name in ("opening", "discussion", "closing")]
    fake_runtime.generate_result = ([[0.1, -0.1]] * 3, 24000)
    engine.generate(make_asset(), inputs, "English", {}, [tmp_path / f"{i['unit_id']}.wav" for i in inputs], batch=True)
    assert len(fake_runtime.generate_calls) == 1
    call = fake_runtime.generate_calls[0]
    assert call["text"] == [i["text"] for i in inputs]
    assert call["language"] == ["English"] * 3
    assert len(call["voice_clone_prompt"]) == 1  # upstream broadcasts the static prompt
    assert call["voice_clone_prompt"][0].ref_text == make_asset().payload["items"][0]["ref_text"]
    assert len(list(tmp_path.glob("*.wav"))) == 3


def test_greedy_disables_both_samplers_and_preserves_single_text(engine, fake_runtime, tmp_path):
    controls = {"do_sample": False, "subtalker_dosample": False}
    engine.generate(make_asset(), [{"unit_id": "continuous", "text": "One. Two. Three."}],
                    "English", controls, [tmp_path / "continuous.wav"])
    call = fake_runtime.generate_calls[0]
    assert call["text"] == "One. Two. Three."
    assert call["do_sample"] is False and call["subtalker_dosample"] is False
    assert "seed" not in call and "generator" not in call


@pytest.mark.parametrize("result", [([], 24000), ([[0.1]] * 2, 24000), ([[0.1]] * 3, 0), ([[0.1]] * 3, True)])
def test_batch_output_contract_rejects_count_or_rate(engine, fake_runtime, tmp_path, result):
    fake_runtime.generate_result = result
    with pytest.raises(ValueError):
        engine.generate(make_asset(), [{"unit_id": str(i), "text": "Text."} for i in range(3)],
                        "English", {}, [tmp_path / f"{i}.wav" for i in range(3)], batch=True)
    assert not list(tmp_path.glob("*.wav"))


def test_effective_defaults_recorded_not_guessed(engine, fake_runtime):
    engine.versions = {"qwen_tts": "fake"}
    fake_runtime.generate_defaults = {"max_new_tokens": 8192}
    fake_runtime._merge_generate_kwargs = lambda **kw: {"do_sample": True, "subtalker_dosample": True,
                                                      "max_new_tokens": 8192, **kw}
    assert engine.provenance({})["effective_generation_controls"]["max_new_tokens"] == 8192
    fake_runtime._merge_generate_kwargs = lambda **kw: {"do_sample": False, "subtalker_dosample": True}
    with pytest.raises(ValueError, match="samplers enabled"):
        engine.provenance({})


def test_version_guard_precedes_any_heavy_import(monkeypatch):
    monkeypatch.setattr("experiments.clone_prosody.runtime.importlib.metadata.version", lambda _: "0.0.0")
    with pytest.raises(ValueError, match="0.1.1"):
        ExperimentEngine(Path("caller-model"), device="cpu", seed=1729)
