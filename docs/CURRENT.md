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

## Reusable Base voice-clone runtime

[Issue #12](https://github.com/Yang-Zhaowei/media-pipeline/issues/12) is closed
via [PR #13](https://github.com/Yang-Zhaowei/media-pipeline/pull/13), squash
merged on 2026-10-04. The independent Qwen3-TTS 12Hz Base normal-ICL runtime
creates reusable local CPU tensor assets and produces production-compatible
mono PCM16 `TTSArtifact` outputs. PR #13 left CustomVoice, the renderer, and the
speech CLI unchanged; segmented selection is the separate Issue #14 work below.

The owner reported real ai-core acceptance at exact PR HEAD
`8d14639976c7bdcf4d2c1a6ddce3d9b7672cc200`: real-torch serialization tests
(3 passed), asset creation, fresh-process loading, two utterance outputs, and
human acceptance of recognizable cloned identity and acceptable consistency.
See the [closure evidence](validation/voice-clone-closure.md) and
[runtime contract / replay commands](dev/voice-clone-runtime.md).
This documentation update does not rerun those checks.

## Current work

[Issue #14](https://github.com/Yang-Zhaowei/media-pipeline/issues/14) connects
accepted reusable clone assets to the existing segmented speech renderer and
CLI. Scripts select exactly one legacy `speaker` or relative clone `voice`
object. Clone instructions must be omitted/empty; CustomVoice semantics and
the downstream pipeline remain compatible. Static validation stays portable;
the TTS subprocess loads one asset and one Base engine for all segments.
See the [script/runtime contract](dev/render-entry-point.md) and
[owner acceptance procedure](validation/clone-render-acceptance.md).
Real ai-core cloned rendering for at least three performance units, human
identity/consistency/transition acceptance, and subtitle synchronization remain
open merge gates. Standalone Issue #12 acceptance does not close these gates.

[Issue #8](https://github.com/Yang-Zhaowei/media-pipeline/issues/8) remains open
for production-use validation of long-form narration stability using existing
tools.

See the [podcast pilot workflow](workflows/podcast-pilot.md) and
[project roadmap](ROADMAP.md). Runtime setup is in [core runtime notes](dev/core-runtime.md).

Pure processing and CPU unit tests run without CUDA. Real model integration is
validated separately on ai-core; see [validation instructions](../validation/README.md).
