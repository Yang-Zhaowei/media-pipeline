# Voice Clone v0 closure evidence

Status: **complete**, accepted by the repository owner on 2026-10-04
(Asia/Shanghai).

## Provenance

- [PR #13 owner acceptance and ai-core results](https://github.com/Yang-Zhaowei/media-pipeline/pull/13#issuecomment-5971192706).
- Exact GPU-tested PR head: `8d14639976c7bdcf4d2c1a6ddce3d9b7672cc200`.
- Squash merge on main: `aa27f97b849c2aabae507f3769a59166d44d3ce0`.
  Its repository tree is identical to the tested PR head, verified locally.
- The owner provided the real ai-core results to the main Codex chat; this
  record transcribes them. The agent did not rerun inference, tests, or
  listening, and did not independently inspect the audio or assets.

## Real ai-core result

Reported runtime: `torch 2.14.0+cu130`, `qwen-tts 0.1.1`, NVIDIA GeForce RTX
4070. The owner ran `tests/test_voice_clone_serialization.py`: **3 passed**.

The run successfully created `voice.pt`; `ref_code` had shape `[443, 16]`,
dtype `int64`, on CPU, and `ref_spk_embedding` had shape `[2048]`, dtype
`bfloat16`, on CPU. A fresh process loaded `voice.pt` successfully and
synthesized both outputs.

| Artifact | Recorded result |
| --- | --- |
| `utterance-1.wav` | 24000 Hz, 109440 frames, 4.56 s |
| `utterance-2.wav` | 24000 Hz, 159360 frames, 6.64 s |
| Human listening | Recognizable cloned identity; acceptable consistency across both utterances |

The owner explicitly accepted the result and requested merge.

## Local checks and scope

Separate historical agent-local CPU PR-head checks reported: voice-clone
focused tests **128 passed, 3 skipped** (real Torch absent); TTS tests **19
passed, 2 GPU-skipped**; full suite **462 passed, 9 skipped** (6 GPU and 3
real-Torch skips); `git diff --check` passed. These are historical results;
the documentation update did not rerun tests.

This closes the independent 12Hz Base normal-ICL runtime: a saved asset loads
in a fresh process and supports repeated synthesis through the production mono
PCM16 `TTSArtifact` contract. PR #13 left CustomVoice, the renderer, and speech
CLI unchanged. PR #15 later added [accepted segmented clone rendering](clone-render-acceptance.md).
See the [runtime contract and replay commands](../dev/voice-clone-runtime.md).
