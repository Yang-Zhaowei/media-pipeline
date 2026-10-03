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

Read-only inventory on 2026-10-04 found `qwen-tts 0.1.1` in the existing TTS
environment and `Qwen3-TTS-12Hz-1.7B-Base` under the existing TTS model root.
The installed prompt API and checkpoint config were inspected without loading
the Base model or running inference. Base config reports a 2048-dimensional
speaker embedding and 16 codebooks. This does not establish real clone or
human acceptance.

The independent runtime, safe CPU asset format, and exact two-process owner
merge-gate commands are in the [voice-clone runtime notes](voice-clone-runtime.md).
No model download or environment changes were performed.

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
