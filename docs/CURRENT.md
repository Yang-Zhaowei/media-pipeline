# Current State

## Production speech pipeline

`text → TTS → forced alignment → audio postprocess → captions → WAV + SRT`

The pipeline accepts caller-authored performance units. Each unit may contain
multiple sentences and is processed as one TTS request. Ordinary narration
defaults to an empty instruction. A CustomVoice segment can inherit the top-level
instruction, explicitly clear it, or provide a local instruction. Caption
segmentation remains independent.

The caller defines the units and directing instructions; automatic segmentation
and text interpretation are outside the product scope. See the [render entry
point](dev/render-entry-point.md) and [performance-unit contract](contracts/performance-unit-instruct.md).

[PR #9](https://github.com/Yang-Zhaowei/media-pipeline/pull/9) is merged. Real
ai-core validation and human listening passed. Historical evidence remains in
the [pipeline closure](validation/speech-v0-closure.md) and
[segmented-render acceptance](validation/segmented-speech-v0-acceptance.md).

## Agent-facing speech CLI and release candidate

[Issue #10](https://github.com/Yang-Zhaowei/media-pipeline/issues/10) is closed
via [PR #11](https://github.com/Yang-Zhaowei/media-pipeline/pull/11), merged on
2026-10-03. The `regrain speech` console entry point provides static script
validation and rendering through the existing Python APIs. Distribution and
primary CLI name are `regrain`; Python imports remain `media_pipeline`, and
`media-pipeline` remains a thin CLI compatibility alias. See the
[CLI contract](dev/render-entry-point.md#command-line) and
[naming decision](naming.md).

Recorded CPU validation passed: 44 focused CLI tests and the full regression
suite (334 passed, 6 GPU integration tests skipped). The owner marked both
merge gates complete in PR #11: a real ai-core CLI render and human acceptance.
This documentation closure does not rerun those CPU or GPU checks.

The current work prepares the installable **v0.1.0 release candidate**. Its
wheel and clean-environment CPU packaging checks do not replace the required
owner-side ai-core validation of the installed Controller coordinating the
existing TTS and Alignment environments. Release gates and deployment steps
are in [release preparation](release.md) and [Client/Core deployment](deployment.md).

## Reusable Base voice-clone runtime

[Issue #12](https://github.com/Yang-Zhaowei/media-pipeline/issues/12) is closed
via [PR #13](https://github.com/Yang-Zhaowei/media-pipeline/pull/13), squash
merged on 2026-10-04. The independent Qwen3-TTS 12Hz Base normal-ICL runtime
creates reusable local CPU tensor assets and produces production-compatible
mono PCM16 `TTSArtifact` outputs. PR #13 left CustomVoice, the renderer, and the
speech CLI unchanged; PR #15 adds segmented selection as recorded below.

The owner reported real ai-core acceptance at exact PR HEAD
`8d14639976c7bdcf4d2c1a6ddce3d9b7672cc200`: real-torch serialization tests
(3 passed), asset creation, fresh-process loading, two utterance outputs, and
human acceptance of recognizable cloned identity and acceptable consistency.
See the [closure evidence](validation/voice-clone-closure.md) and
[runtime contract / replay commands](dev/voice-clone-runtime.md).
This documentation update does not rerun those checks.

## Segmented clone rendering

[Issue #14](https://github.com/Yang-Zhaowei/media-pipeline/issues/14) is closed
via [PR #15](https://github.com/Yang-Zhaowei/media-pipeline/pull/15), squash
merged on 2026-10-04 (Asia/Shanghai). Scripts select exactly one legacy
`speaker` or relative clone `voice`; clone instructions must be omitted/empty.
CustomVoice and downstream behavior remain compatible. Static validation is
portable; one TTS subprocess reuses one loaded asset and one Base engine.

The owner accepted a real three-unit ai-core render, all downstream stages,
final artifacts, voice identity/consistency, transitions, and subtitle sync.
Recorded validation: Windows full CPU suite **547 passed, 10 skipped**;
owner ai-core clone tests **55 passed**, CLI tests **75 passed**.
See the [contract](dev/render-entry-point.md) and
[exact-head evidence / replay](validation/clone-render-acceptance.md).
This documentation closure did not rerun those checks.

## Current work

Regrain v0.1.0 is prepared as a release candidate with a dependency-free Controller
wheel, retained `media_pipeline` imports, isolated subprocess workers and reusable
speech-authoring Skill. [Client CPU/installation verification](validation/regrain-v0.1.0-candidate.md)
is separate from the required [owner Core acceptance](deployment.md). No final
release, repository rename or live deployment has occurred.

[Issue #8](https://github.com/Yang-Zhaowei/media-pipeline/issues/8) remains open
for production narration stability, including the accepted clone render's
remaining cross-segment emotion/delivery drift. Evaluate with existing tools.

The owner accepted the [Qwen-only Layer 1 experiment](validation/qwen-clone-prosody-acceptance.md)
at implementation HEAD `506b30dd59702348d6ff11bc7fad435acddf3a76`: Linux tests
(82 experiment-focused; 328 clone/render/CLI; full suite 633 passed, 6 skipped),
S3 GPU smoke, all 13 matrix runs and blind listening of all 13 samples.
Continuous generation showed promising perceived continuity for the roughly
15-second corpus; context reset is the strongest current engineering hypothesis,
with concatenation/topology confounds. Sampling-off alone did not eliminate
delivery changes. Owner PCM comparison passed for both greedy listening repeat
pairs in the fixed environment; multi-minute stability remains open.
[PR #16](https://github.com/Yang-Zhaowei/media-pipeline/pull/16) merged on
2026-10-09T19:04:43Z at
`ff73167436937b3788c297b11d9c4b67235c2f5d`. The owner-reported real GPU and
blind-listening evidence remains tied to implementation HEAD
`506b30dd59702348d6ff11bc7fad435acddf3a76`; the merge commit and current
release candidate were not the tested HEAD. Issue #8 continues as production-use
validation, without an established root cause or universal optimal unit length.
Continuous generation showed promising short-corpus prosody continuity.
Production speech behavior is unchanged. This documentation closure did not
rerun tests, GPU synthesis or listening, and is not a roadmap milestone.

See the [podcast pilot workflow](workflows/podcast-pilot.md) and
[project roadmap](ROADMAP.md). Runtime setup is in [core runtime notes](dev/core-runtime.md).

Pure processing and CPU unit tests run without CUDA. Real model integration is
validated separately on ai-core; see [validation instructions](../validation/README.md).
