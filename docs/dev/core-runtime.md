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

Current status: verified GPU smoke integration; production module not yet implemented.

## Model locations

TTS:

`/srv/ai/models/speech/tts/`

ASR / alignment:

`/srv/ai/models/speech/asr/`

Vision:

`/srv/ai/models/vision/`

## Current limitation

The TTS and forced-alignment integrations currently depend on separate Python environments.

Pure processing code and unit tests must not depend on these host-specific environments.

GPU integration tests must be explicitly separated from ordinary unit tests.

## Deployment rule

The repository is developed on PC_Client.

`ai-core` is a GPU integration target, not the primary editing workspace.

Do not hard-code these absolute paths into portable application logic.