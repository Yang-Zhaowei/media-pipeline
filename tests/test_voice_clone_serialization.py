"""Optional real-torch CPU checks; no checkpoint, qwen_tts, or CUDA needed.

The ordinary dependency-free suite covers the contract with fakes. These tests
also exercise the real restricted unpickler when torch is already installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from media_pipeline.voice_clone import VoiceAssetError, VoiceCloneAsset
from media_pipeline.runtimes.qwen_voice_clone import load_voice_asset, save_voice_asset


def test_real_tensor_roundtrip_in_fresh_cpu_process(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    code = torch.tensor([[0, 100], [200, 300]], dtype=torch.int64).t()
    embedding = torch.tensor([0.125, -0.75], dtype=torch.bfloat16)
    asset = VoiceCloneAsset(payload={
        "format_version": 1,
        "model": {"tokenizer_type": "qwen3_tts_tokenizer_12hz", "tts_model_size": "1b7"},
        "items": [{
            "ref_code": code,
            "ref_spk_embedding": embedding,
            "x_vector_only_mode": False,
            "icl_mode": True,
            "ref_text": "  Exact transcript.\n",
        }],
    })
    path = tmp_path / "voice.pt"
    save_voice_asset(asset, path)
    restored = load_voice_asset(path).payload["items"][0]
    for name, expected in (("ref_code", code), ("ref_spk_embedding", embedding)):
        value = restored[name]
        assert value.dtype == expected.dtype
        assert value.shape == expected.shape
        assert value.device.type == "cpu"
        assert torch.equal(value, expected)
    assert str(tmp_path).encode() not in path.read_bytes()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    script = """
import sys
import torch
from media_pipeline.runtimes.qwen_voice_clone import load_voice_asset
item = load_voice_asset(sys.argv[1]).payload['items'][0]
assert item['ref_text'] == '  Exact transcript.\\n'
assert item['ref_code'].dtype == torch.int64
assert item['ref_spk_embedding'].dtype == torch.bfloat16
assert item['ref_code'].tolist() == [[0, 200], [100, 300]]
assert item['ref_spk_embedding'].tolist() == [0.125, -0.75]
assert all(item[k].device.type == 'cpu' for k in ('ref_code', 'ref_spk_embedding'))
assert 'qwen_tts' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)], env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class _ExecutablePayload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return eval, (f"__import__('pathlib').Path({str(self.marker)!r}).touch()",)


def test_real_restricted_unpickler_does_not_execute_payload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    marker, path = tmp_path / "executed", tmp_path / "unsafe.pt"
    torch.save(_ExecutablePayload(marker), path)
    with pytest.raises(VoiceAssetError, match="safely"):
        load_voice_asset(path)
    assert not marker.exists()


def test_real_safe_load_still_rejects_malformed_payload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "empty.pt"
    torch.save({"format_version": 1, "model": {
        "tokenizer_type": "qwen3_tts_tokenizer_12hz", "tts_model_size": "1b7",
    }, "items": []}, path)
    with pytest.raises(VoiceAssetError):
        load_voice_asset(path)
