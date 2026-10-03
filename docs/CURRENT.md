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

## Current work

[Issue #10](https://github.com/Yang-Zhaowei/media-pipeline/issues/10) adds a
standard `media-pipeline speech` console entry point for static script
validation and rendering, using the existing Python entry points. CPU
validation on 2026-10-03 passed: 44 focused CLI tests and the full regression
suite (334 passed, 6 GPU integration tests skipped). See the
[CLI contract](dev/render-entry-point.md#command-line).
An ai-core real CLI render and any required human acceptance remain owner merge
gates. No GPU validation is claimed here.

[Issue #8](https://github.com/Yang-Zhaowei/media-pipeline/issues/8) remains open
for production-use validation of long-form narration stability using existing
tools.

See the [podcast pilot workflow](workflows/podcast-pilot.md) and
[project roadmap](ROADMAP.md). Runtime setup is in [core runtime notes](dev/core-runtime.md).

Pure processing and CPU unit tests run without CUDA. Real model integration is
validated separately on ai-core; see [validation instructions](../validation/README.md).
