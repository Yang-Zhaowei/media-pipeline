# ai-core Development Runtime

This document records the currently verified GPU integration environment. It is host-specific evidence, not portable project configuration.

## Host

- Host: `ai-core`
- OS: Ubuntu Server 24.04
- GPU: NVIDIA GeForce RTX 4070 12GB
- CUDA runtime verified through PyTorch: 13.0

## Existing environments

### TTS

Environment:
`/srv/ai/apps/media-pipeline/tts/.venv`

Verified:

- Python 3.12.14
- PyTorch 2.14.0+cu130
- qwen-tts
- CUDA 13.0
- NVIDIA GeForce RTX 4070
- Qwen3-TTS-12Hz-1.7B-CustomVoice
- Production TTS v0 GPU integration passed
- repeated synthesis using one loaded model instance
- generated mono 16-bit PCM WAV accepted by project code

PR #3 integration evidence:

2 tests passed in 18.42 s

Manual artifact:

sample rate: 24000 Hz
frames:      165120
duration:    6.88 s

Manual listening check passed.

### Base voice cloning (Issue #12)

Owner-reported real ai-core acceptance was recorded on 2026-10-04 for exact
PR #13 HEAD `8d14639976c7bdcf4d2c1a6ddce3d9b7672cc200`, then squash merged
as `aa27f97b849c2aabae507f3769a59166d44d3ce0`.

Runtime: PyTorch `2.14.0+cu130`, `qwen-tts 0.1.1`, NVIDIA GeForce RTX 4070,
and the existing `Qwen3-TTS-12Hz-1.7B-Base` checkpoint. Real-torch
serialization tests passed (3 tests). The owner successfully created
`voice.pt`, loaded it in a fresh process, generated both utterance WAVs,
and accepted recognizable cloned identity with acceptable consistency.

The saved reference codes are `[443, 16]`, int64 on CPU; the speaker embedding
is `[2048]`, bfloat16 on CPU. Outputs are 24000 Hz: 109440 frames / 4.56 s
and 159360 frames / 6.64 s.

See the [closure evidence](../validation/voice-clone-closure.md) and
[runtime notes / replay commands](voice-clone-runtime.md). These results were
provided by the owner; this documentation update did not rerun inference,
serialization tests, or listening, and made no runtime environment changes.

### Forced Alignment

Environment:

`/srv/ai/apps/media-pipeline/aligner/.venv`

Verified:

- Python 3.12.14
- `qwen-asr`
- PyTorch `2.14.0+cu130`
- CUDA available
- NVIDIA GeForce RTX 4070
- `Qwen3-ForcedAligner-0.6B`

Current status: Production Alignment module v0 implemented; GPU integration
test gated on `MEDIA_PIPELINE_ALIGNMENT_MODEL`, validated on ai-core.

## Model locations

TTS:

`/srv/ai/models/speech/tts/`

ASR / alignment:

`/srv/ai/models/speech/asr/`

Vision:

`/srv/ai/models/vision/`

## Full Speech Pipeline v0 acceptance

PR #5 records a real E2E PASS at commit
`93c650d2ae03e2a0771ff1f7aa9089a7a23e90ee`: fresh Production TTS output passed
Production Alignment, Audio Postprocess, and Caption Compiler on ai-core.
The two model stages used their existing separate environments.

The final WAV is mono PCM16, 24000 Hz, 147840 frames (6.160 s), with two SRT
captions. The real E2E test passed; the E2E unit + integration suites recorded
21 passed in 14.81 s. Human listening and subtitle synchronization acceptance
were confirmed by the repository owner on 2026-09-29.

See [closure evidence](../validation/speech-v0-closure.md). This records the
reported ai-core run; the documentation update did not rerun GPU inference or
re-inventory package versions. Versions above retain their original provenance.

## Runtime boundary

The TTS and forced-alignment integrations currently depend on separate Python environments.

Pure processing code and unit tests must not depend on these host-specific environments.

GPU integration tests must be explicitly separated from ordinary unit tests.

## Deployment rule

The repository is developed on PC_Client.

`ai-core` is a GPU integration target, not the primary editing workspace.

Do not hard-code these absolute paths into portable application logic.
