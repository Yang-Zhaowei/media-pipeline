# Qwen3-TTS Base voice-clone runtime v0

[Issue #12](https://github.com/Yang-Zhaowei/media-pipeline/issues/12) adds a
separate `Qwen3VoiceCloneTTS` adapter. Normal ICL uses a local mono PCM16 WAV
and a caller-supplied exact transcript. One loaded Base engine creates a
reusable CPU prompt or synthesizes multiple already-segmented utterances.
v0 supports the 12Hz Base checkpoints; unknown tokenizer families fail rather
than guessing their code shapes. Outputs use the existing waveform converter,
WAV writer/readback, and
`TTSArtifact` contract. CustomVoice, speech scripts, renderer, and speech CLI
are unchanged. `x_vector_only_mode` and automatic transcription are deferred.

## Verified upstream contract

Read-only inspection of ai-core's existing TTS environment found `qwen-tts
0.1.1`. The official upstream source at commit
`022e286b98fbec7e1e916cb940cdf532cd9f488e` agrees:

- [`create_voice_clone_prompt` and `VoiceClonePromptItem`](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/inference/qwen3_tts_model.py)
  return a list of dataclass items containing `ref_code` (integer tensor,
  `(T, Q)` or `(T,)`), `ref_spk_embedding` (floating tensor, `(D,)`),
  `x_vector_only_mode` (bool), `icl_mode` (bool), and `ref_text` (optional str).
  Normal ICL requires text and uses flags `False` / `True`.
- The [official save/load demo](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/cli/demo.py)
  saves dictionaries of item fields under `items`, loads with
  `weights_only=True, map_location="cpu"`, and reconstructs prompt items.
- `generate_voice_clone(text, language, voice_clone_prompt=items)` accepts
  those restored items. Its dictionary-prompt branch omits reference transcript
  tokenization. This adapter always supplies the item list to preserve ICL.

The adapter requires that exact upstream dataclass field set. An incompatible
upstream structure fails explicitly, without guessing missing information.

## Python interface and errors

```python
from media_pipeline.voice_clone import VoiceCloneReference, VoiceCloneRequest
from media_pipeline.runtimes.qwen_voice_clone import (
    Qwen3VoiceCloneTTS, load_voice_asset, save_voice_asset,
)

engine = Qwen3VoiceCloneTTS(model_path, device="cuda:0")
asset = engine.create_voice_asset(VoiceCloneReference(reference_wav, exact_transcript))
save_voice_asset(asset, asset_path)

# In a later process: no original WAV or transcript file is needed.
asset = load_voice_asset(asset_path)
engine = Qwen3VoiceCloneTTS(model_path, device="cuda:0")
first = engine.synthesize(VoiceCloneRequest(first_text, language), asset, first_wav)
second = engine.synthesize(VoiceCloneRequest(second_text, language), asset, second_wav)
```

`VoiceCloneError` means an invalid request/reference (including a missing,
empty, or unsupported reference WAV). `VoiceAssetError` means an invalid,
incompatible, or unreadable/unwritable asset. `TTSRuntimeError` translates heavy
import, checkpoint load, upstream generation, and output/WAV failures.
Whitespace-only required strings fail; valid text/transcript values are
preserved exactly. Transcript correctness is the caller's responsibility.

Importing either module needs no torch, qwen_tts, NumPy, or CUDA. Portable
validators check requests and the payload envelope using the standard library.
Save/load additionally need torch, lazily imported; they need no qwen_tts or
checkpoint. Engine construction lazily imports both torch and qwen_tts.

## Local asset contract

The saved payload is strictly:

```text
{
  "format_version": 1,
  "model": {"tokenizer_type": str, "tts_model_size": str},
  "items": [{
    "ref_code": torch.Tensor,
    "ref_spk_embedding": torch.Tensor,
    "x_vector_only_mode": False,
    "icl_mode": True,
    "ref_text": exact nonempty str
  }]
}
```

No source audio/model/output paths or custom Python objects are saved. Tensor
storage is detached, copied to CPU, and cloned without dtype/value conversion.
Save and load validate exact field sets, version, nonempty items, primitive
types, supported 12Hz tokenizer metadata, integer code tensors of rank 2,
and finite floating embedding tensors
of rank 1. Synthesis additionally requires one item, matching tokenizer/model
size, matching speaker embedding dimension, and the checkpoint's 12Hz codebook
dimension. Empty, malformed, unknown-version, or incompatible data fails.

Load always calls `torch.load(..., weights_only=True, map_location="cpu")`.
There is no unrestricted-pickle fallback, added safe-global allowlist, tensor
coercion, or mode/text defaulting. This project envelope extends the upstream
demo's unversioned payload; demo exports must not be relabeled without
validated checkpoint metadata. Assets are runtime outputs and must not be
committed as regression fixtures.

## Owner merge gate: real ai-core acceptance

No real clone synthesis or human acceptance is recorded by this change.
Read-only inventory found an existing `Qwen3-TTS-12Hz-1.7B-Base` checkpoint;
its config declares `base`, `qwen3_tts_tokenizer_12hz`, `1b7`, speaker embedding
dimension 2048, and 16 codebooks. This is availability evidence, not a load
test. Required checkpoint: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` with its speech
tokenizer. Do not download or alter the runtime to satisfy this gate silently.

On ai-core, check out the exact PR HEAD in a clean tree and use its existing
TTS virtual environment. Set `PYTHONPATH` to that checkout's `src`, and supply
the following caller-owned variables (absolute paths selected by the owner):

- `MEDIA_PIPELINE_VOICE_CLONE_MODEL`: Base checkpoint directory.
- `MEDIA_PIPELINE_VOICE_CLONE_REFERENCE`: mono PCM16 reference WAV.
- `MEDIA_PIPELINE_VOICE_CLONE_TRANSCRIPT`: UTF-8 file with the exact transcript.
- `MEDIA_PIPELINE_VOICE_CLONE_RUN`: a new output directory.
- `MEDIA_PIPELINE_VOICE_CLONE_LANGUAGE`: explicit Qwen language.
- `MEDIA_PIPELINE_VOICE_CLONE_TEXT_1` and `_TEXT_2`: two different utterances.

First, exercise actual CPU serialization safety with already-installed torch:

```bash
python -m pytest tests/test_voice_clone_serialization.py
```

Then create and save the real voice in process 1:

```bash
python - <<'PY'
import os
from pathlib import Path
from media_pipeline.voice_clone import VoiceCloneReference
from media_pipeline.runtimes.qwen_voice_clone import Qwen3VoiceCloneTTS, save_voice_asset

run = Path(os.environ["MEDIA_PIPELINE_VOICE_CLONE_RUN"])
run.mkdir(parents=True, exist_ok=False)
engine = Qwen3VoiceCloneTTS(os.environ["MEDIA_PIPELINE_VOICE_CLONE_MODEL"])
transcript = Path(os.environ["MEDIA_PIPELINE_VOICE_CLONE_TRANSCRIPT"]).read_text(encoding="utf-8")
asset = engine.create_voice_asset(VoiceCloneReference(
    os.environ["MEDIA_PIPELINE_VOICE_CLONE_REFERENCE"], transcript,
))
save_voice_asset(asset, run / "voice.pt")
print("Created", run / "voice.pt", asset.payload["model"])
PY
```

Load that asset in fresh process 2; create exactly one engine and synthesize
both utterances without accessing the reference recording or transcript file:

```bash
python - <<'PY'
import json, os
from pathlib import Path
from media_pipeline.postprocess import read_wav
from media_pipeline.voice_clone import VoiceCloneRequest
from media_pipeline.runtimes.qwen_voice_clone import Qwen3VoiceCloneTTS, load_voice_asset

run = Path(os.environ["MEDIA_PIPELINE_VOICE_CLONE_RUN"])
asset = load_voice_asset(run / "voice.pt")
engine = Qwen3VoiceCloneTTS(os.environ["MEDIA_PIPELINE_VOICE_CLONE_MODEL"])
texts = [os.environ[f"MEDIA_PIPELINE_VOICE_CLONE_TEXT_{i}"] for i in (1, 2)]
assert texts[0] != texts[1]
records = []
for i, text in enumerate(texts, 1):
    artifact = engine.synthesize(VoiceCloneRequest(
        text, os.environ["MEDIA_PIPELINE_VOICE_CLONE_LANGUAGE"],
    ), asset, run / f"utterance-{i}.wav")
    samples, rate, frames = read_wav(artifact.wav_path)
    assert frames == artifact.frames == len(samples) and frames > 0
    assert rate == artifact.sample_rate and rate > 0
    records.append({"wav": artifact.wav_path.name, "sample_rate": rate,
                    "frames": frames, "duration": artifact.duration})
(run / "artifacts.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
print(json.dumps(records, indent=2))
PY
```

Expected layout: `voice.pt`, `utterance-1.wav`, `utterance-2.wav`, and
`artifacts.json` in the chosen run directory. Both WAVs must be production
mono PCM16 and yield positive `TTSArtifact` durations. Record exact HEAD,
runtime versions, commands/results, and human listening against the real
reference. The owner must confirm recognizable, acceptably consistent voice
identity across both different utterances before merging.
