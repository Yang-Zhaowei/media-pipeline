"""Experiment-owned access to audited Qwen controls; production API stays unchanged."""

from __future__ import annotations

import hashlib
import importlib.metadata
import random
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_asset(path):
    from media_pipeline.runtimes.qwen_voice_clone import load_voice_asset
    return load_voice_asset(path)


class ExperimentEngine:
    """Reuse the validated engine/asset checks, with a local batch/control bridge.

    The existing public synthesize method cannot pass generation controls or
    list inputs. Access to its pinned internal model is confined to experiments;
    no production seam or new provider is necessary.
    """

    def __init__(self, model, *, device, seed):
        version = importlib.metadata.version("qwen-tts")
        if version != "0.1.1":
            raise ValueError(f"audit requires qwen-tts 0.1.1, found {version}")
        import numpy as np
        import torch
        from media_pipeline.runtimes.qwen_voice_clone import Qwen3VoiceCloneTTS

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # manual_seed seeds all devices; do not enable deterministic algorithms
        # or change cuDNN/attention settings to obtain a false default baseline.
        self.engine = Qwen3VoiceCloneTTS(model, device=device)
        if self.engine._model_metadata["tts_model_size"] != "1b7":
            raise ValueError("experiment requires Qwen3-TTS-12Hz-1.7B-Base")
        self.versions = {
            "qwen_tts": version, "torch": torch.__version__,
            "numpy": np.__version__, "transformers": importlib.metadata.version("transformers"),
            "cuda_runtime": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "dtype": "bfloat16", "non_streaming_mode": "upstream default False",
        }
        if device.startswith("cuda"):
            self.versions["gpu_name"] = torch.cuda.get_device_name(device)
        import qwen_tts.inference.qwen3_tts_model as wrapper
        import qwen_tts.core.models.modeling_qwen3_tts as core
        self.versions["qwen_source_sha256"] = {
            "wrapper": sha256_file(Path(wrapper.__file__)),
            "core": sha256_file(Path(core.__file__)),
        }

    def provenance(self, controls):
        # This exact merger was inspected in 0.1.1. Record its result, rather
        # than treating hard fallback values as the checkpoint's defaults.
        effective = self.engine._model._merge_generate_kwargs(**controls)
        if not controls and (effective["do_sample"] is not True or
                             effective["subtalker_dosample"] is not True):
            raise ValueError("default conditions require both checkpoint samplers enabled")
        return {**self.versions, "effective_generation_controls": effective,
                "checkpoint_generation_defaults": self.engine._model.generate_defaults}

    def generate(self, asset, inputs, language, controls, paths, *, batch=False):
        from media_pipeline.voice_clone import VoiceCloneRequest, validate_request
        from media_pipeline.postprocess import read_wav, write_wav
        from media_pipeline.tts import waveform_to_mono_pcm16

        for item in inputs:
            validate_request(VoiceCloneRequest(item["text"], language))
        if not batch and not controls:
            if len(inputs) != 1:
                raise ValueError("separate call requires exactly one text")
            # S3/L1 default use the unchanged, accepted production operation.
            self.engine.synthesize(VoiceCloneRequest(inputs[0]["text"], language), asset, paths[0])
            return
        self.engine._validate_for_model(asset)
        items = [self.engine._prompt_item_type(**item) for item in asset.payload["items"]]
        # 0.1.1 broadcasts this single reference prompt across all batch items.
        wavs, rate = self.engine._model.generate_voice_clone(
            text=[item["text"] for item in inputs] if batch else inputs[0]["text"],
            language=[language] * len(inputs) if batch else language,
            voice_clone_prompt=items, **controls,
        )
        if type(wavs) is not list or len(wavs) != len(paths):
            raise ValueError("Qwen returned an unexpected waveform count")
        if type(rate) is not int or rate <= 0:
            raise ValueError("Qwen returned an invalid sample rate")
        for wav, path in zip(wavs, paths):
            write_wav(path, waveform_to_mono_pcm16(wav), rate)
            read_wav(path)
