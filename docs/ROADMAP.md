# Roadmap

Status: planning / non-binding.

`docs/CURRENT.md` defines current work.
`AGENTS.md` defines agent scope.

## Project role

media-pipeline provides small, local-first, verifiable media operations
only where real production workflows expose a concrete gap not adequately
served by existing tools.

It is not intended to become a complete NLE, media framework, workflow
engine, or Agent control platform.

It is not a mandatory execution or evidence layer for every media
operation. It reuses complete existing capabilities first, and adapts or
builds only the remaining demonstrated gap, stopping once the gap is
closed.

## Completed milestones

Speech Pipeline v0 closed on 2026-09-29 via PR #5 and human acceptance.
See [recorded evidence](validation/speech-v0-closure.md).

text
→ TTS
→ forced alignment
→ audio postprocess
→ caption compiler
→ WAV + SRT

Met stopping condition:

A supported production utterance runs through the complete production
path on ai-core and produces validated final WAV and SRT artifacts.
Listening and synchronization are checked, and CPU regressions remain green.

No new modality, invocation surface, NLE integration, or orchestration
is required to close this milestone.

The agent-facing speech CLI is complete via
[PR #11](https://github.com/Yang-Zhaowei/media-pipeline/pull/11), merged on
2026-10-03, closing
[Issue #10](https://github.com/Yang-Zhaowei/media-pipeline/issues/10).
It exposes static validation and the existing speech renderer; see the
[CLI contract](dev/render-entry-point.md#command-line).

Reusable Qwen3-TTS Base normal-ICL voice cloning is complete via
[PR #13](https://github.com/Yang-Zhaowei/media-pipeline/pull/13), merged on
2026-10-04, closing
[Issue #12](https://github.com/Yang-Zhaowei/media-pipeline/issues/12).
The owner accepted real asset creation, fresh-process reuse, two utterance
outputs, and human voice identity / consistency. See the
[closure evidence](validation/voice-clone-closure.md).

## Next gate: real workflow evaluation

Speech Pipeline v0 is complete. Stop speculative feature development and
evaluate the owner's [podcast workflow](workflows/podcast-pilot.md).

Use media-pipeline in one real short-form production with:

- one actual finishing tool (FFmpeg or a target NLE, as requested by the owner);
- one actual Agent host;
- real source media;
- narration and captions;
- at least one revision;
- final export.

Try existing capabilities first:

1. NLE-native functionality;
2. official APIs / extensions;
3. maintained integrations / plugins;
4. established tools and standards;
5. media-pipeline custom code only for the remaining demonstrated gap.

The evaluation ends with either:

- no additional development is needed; or
- one reproducible capability gap is identified.

Both outcomes are successful.

## Feature admission rule

A new feature requires:

1. a real production input/problem;
2. a reproducible inadequacy in existing solutions;
3. evidence that media-pipeline is the correct ownership boundary;
4. the smallest implementation that closes the gap;
5. a stopping condition;
6. consideration of maintenance cost and an exit/replacement path.

"Could be useful", "fits the architecture", and "might be needed by an
Agent later" are not sufficient reasons.

Reproducibility means the path runs and its checks are repeatable; model
stages are not required to produce byte-identical output. Determinism is
reserved for input rules, timing, format conversion, and verification.

## Uncommitted questions

These are not roadmap milestones:

- media inspection/extraction;
- ASR;
- Vision / media search;
- timeline interchange;
- NLE integration;
- MCP / HTTP;
- orchestration.

Each must first pass the Feature admission rule with reproducible evidence
before any project code. Vision and ASR remain un-proposed until a real
case re-enters this gate.

## Long-term success

Success does not require the project to grow.

A small, reliable Speech tool that remains useful beside existing NLEs and
Agent integrations is a valid long-term outcome.
