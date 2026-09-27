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

- Python 3.12
- `qwen-tts`
- CUDA inference
- `Qwen3-TTS-12Hz-1.7B-CustomVoice`

Current smoke test succeeded.

### ASR / Forced Alignment

Environment:

`/srv/ai/apps/media-pipeline/aligner/.venv`

Verified:

- Python 3.12.14
- `qwen-asr`
- PyTorch `2.14.0+cu130`
- CUDA available
- NVIDIA GeForce RTX 4070
- `Qwen3-ForcedAligner-0.6B`

Current forced-alignment smoke test succeeded.

## Model locations

TTS:

`/srv/ai/models/speech/tts/`

ASR / alignment:

`/srv/ai/models/speech/asr/`

Vision:

`/srv/ai/models/vision/`

## Current limitation

The TTS and ASR/alignment integrations currently depend on separate Python environments.

Pure processing code and unit tests must not depend on these host-specific environments.

GPU integration tests must be explicitly separated from ordinary unit tests.

## Deployment rule

The repository is developed on PC_Client.

`ai-core` is a GPU integration target, not the primary editing workspace.

Do not hard-code these absolute paths into portable application logic.