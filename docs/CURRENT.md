# Current State

## Production speech pipeline

`text → TTS → forced alignment → audio postprocess → captions → WAV + SRT`

The pipeline accepts caller-authored performance units. Each unit may contain
multiple sentences and is processed as one TTS request. Ordinary narration
defaults to an empty instruction. A segment can inherit the top-level
instruction, explicitly clear it, or provide a local instruction. Caption
segmentation remains independent.

The caller defines the units and directing instructions; automatic segmentation
and text interpretation are outside the product scope. See the [render entry
point](dev/render-entry-point.md) and [performance-unit contract](contracts/performance-unit-instruct.md).

[PR #9](https://github.com/Yang-Zhaowei/media-pipeline/pull/9) is merged. Real
ai-core validation and human listening passed. Historical evidence remains in
the [pipeline closure](validation/speech-v0-closure.md) and
[segmented-render acceptance](validation/segmented-speech-v0-acceptance.md).

## Agent-facing speech CLI

[Issue #10](https://github.com/Yang-Zhaowei/media-pipeline/issues/10) is closed
via [PR #11](https://github.com/Yang-Zhaowei/media-pipeline/pull/11), merged on
2026-10-03. The standard `media-pipeline speech` console entry point provides
static script validation and rendering through the existing Python APIs.
See the [CLI contract](dev/render-entry-point.md#command-line).

Recorded CPU validation passed: 44 focused CLI tests and the full regression
suite (334 passed, 6 GPU integration tests skipped). The owner marked both
merge gates complete in PR #11: a real ai-core CLI render and human acceptance.
This documentation closure does not rerun those CPU or GPU checks.

## Current work

[Issue #12](https://github.com/Yang-Zhaowei/media-pipeline/issues/12) adds an
independent Qwen3-TTS Base normal-ICL voice-clone runtime and safely reusable
local tensor assets. The CustomVoice renderer and speech CLI remain unchanged.
See the [runtime contract and exact owner acceptance commands](dev/voice-clone-runtime.md).
Real ai-core clone generation, fresh-process reuse, and human voice-identity
acceptance remain owner merge gates; checkpoint/API inventory alone is not
GPU validation.

[Issue #8](https://github.com/Yang-Zhaowei/media-pipeline/issues/8) remains open
for production-use validation of long-form narration stability using existing
tools.

See the [podcast pilot workflow](workflows/podcast-pilot.md) and
[project roadmap](ROADMAP.md). Runtime setup is in [core runtime notes](dev/core-runtime.md).

Pure processing and CPU unit tests run without CUDA. Real model integration is
validated separately on ai-core; see [validation instructions](../validation/README.md).
