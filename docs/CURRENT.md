# Current State

## Production speech pipeline

`text → TTS → forced alignment → audio postprocess → captions → WAV + SRT`

Caller-approved Performance Units produce one TTS request each; subtitle
segmentation is independent. Input and instruction rules are maintained in the
[render contract](dev/render-entry-point.md) and [authoring Skill](../skills/regrain-speech/SKILL.md).

[PR #9](https://github.com/Yang-Zhaowei/media-pipeline/pull/9) is merged. Real
ai-core validation and human listening passed. Historical evidence remains in
the [pipeline closure](validation/speech-v0-closure.md) and
[segmented-render acceptance](validation/segmented-speech-v0-acceptance.md).

## Agent-facing speech CLI and release candidate

[Issue #10](https://github.com/Yang-Zhaowei/media-pipeline/issues/10) is closed
via [PR #11](https://github.com/Yang-Zhaowei/media-pipeline/pull/11), with owner
real ai-core CLI and human acceptance. Current identity and compatibility are
defined in [naming](naming.md); command behavior is in the [CLI contract](dev/render-entry-point.md#command-line).

## Reusable Base voice-clone runtime

[Issue #12](https://github.com/Yang-Zhaowei/media-pipeline/issues/12) is closed
via [PR #13](https://github.com/Yang-Zhaowei/media-pipeline/pull/13), squash
merged on 2026-10-04. The independent Qwen3-TTS 12Hz Base normal-ICL runtime
creates reusable local CPU tensor assets and produces production-compatible
mono PCM16 `TTSArtifact` outputs. PR #13 left CustomVoice, the renderer, and the
speech CLI unchanged; PR #15 adds segmented selection as recorded below.

The owner accepted asset creation, fresh-process reuse and two utterances at
exact PR HEAD `8d14639976c7bdcf4d2c1a6ddce3d9b7672cc200`.
See the [closure evidence](validation/voice-clone-closure.md) and
[runtime contract / replay commands](dev/voice-clone-runtime.md).

## Segmented clone rendering

[Issue #14](https://github.com/Yang-Zhaowei/media-pipeline/issues/14) is closed
via [PR #15](https://github.com/Yang-Zhaowei/media-pipeline/pull/15), squash
merged on 2026-10-04 (Asia/Shanghai). Scripts select exactly one legacy
`speaker` or relative clone `voice`; clone instructions must be omitted/empty.
CustomVoice and downstream behavior remain compatible. Static validation is
portable; one TTS subprocess reuses one loaded asset and one Base engine.

The owner accepted a real three-unit ai-core render, final artifacts, voice
identity/consistency, transitions and subtitle sync. See the [contract](dev/render-entry-point.md) and
[exact-head evidence / replay](validation/clone-render-acceptance.md).

## Current work

PR #17 is in final closure for manual owner review, Ready status and merge.
[Historical CPU/package verification](validation/regrain-v0.1.0-candidate.md)
and [owner GPU / Client Agent acceptance](validation/regrain-v0.1.0-owner-acceptance.md)
are separate evidence. The latter applies only to original source `dbcf0db…`
and its recorded wheel hash: runtime integration and pronunciation/subtitles
passed; CustomVoice consistency was acceptable, Clone consistency was variable,
and unattended Clone production quality was not established. Codex 6.1 Luna
and Pi reported successful Clone authoring/static validation; Codex's initial
failure was subprocess PATH discovery.

The new documentation/Skill closure wheel and future merged release artifact
need their own identities and [exact-artifact release acceptance](release.md).
Application Python and tests are unchanged by closure. The agent has not rerun
GPU inference or inspected raw Core evidence. No final release or repository
rename is recorded; this closure performs no deployment.

[Issue #8](https://github.com/Yang-Zhaowei/media-pipeline/issues/8) remains open
for production narration stability, including the accepted clone render's
remaining cross-segment emotion/delivery drift. Evaluate with existing tools.

The owner accepted the [Qwen-only Layer 1 experiment](validation/qwen-clone-prosody-acceptance.md)
at implementation HEAD `506b30dd59702348d6ff11bc7fad435acddf3a76`: all 13 matrix
runs and owner blind listening passed; detailed counts and provenance stay in that record.
Continuous generation showed promising perceived continuity for the roughly
15-second corpus; context reset is the strongest current engineering hypothesis,
with concatenation/topology confounds. Sampling-off alone did not eliminate
delivery changes. Owner PCM comparison passed for both greedy listening repeat
pairs in the fixed environment; multi-minute stability remains open.
[PR #16](https://github.com/Yang-Zhaowei/media-pipeline/pull/16) merged on
2026-10-09T19:04:43Z at
`ff73167436937b3788c297b11d9c4b67235c2f5d`. The owner-reported real GPU and
blind-listening evidence remains tied to implementation HEAD
`506b30dd59702348d6ff11bc7fad435acddf3a76`; it does not establish PR #17 or
release-wheel acceptance. Issue #8 continues as production-use
validation, without an established root cause or universal optimal unit length.
The new Clone observations lack a controlled old-versus-new comparison, so a
PR #17 regression is not strictly excluded. Closure changes no production
behavior and does not resolve this question.

See the [podcast pilot workflow](workflows/podcast-pilot.md) and
[project roadmap](ROADMAP.md). Runtime setup is in [core runtime notes](dev/core-runtime.md).

Pure processing and CPU unit tests run without CUDA. Real model integration is
validated separately on ai-core; see [validation instructions](../validation/README.md).
