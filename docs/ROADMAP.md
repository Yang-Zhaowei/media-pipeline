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

## Current committed milestone

Close Speech Pipeline v0:

text
→ TTS
→ forced alignment
→ audio postprocess
→ caption compiler
→ WAV + SRT

Stopping condition:

A supported production utterance runs through the complete production
path on ai-core and produces validated final WAV and SRT artifacts.
Listening and synchronization are checked, and CPU regressions remain green.

No new modality, invocation surface, NLE integration, or orchestration
is required to close this milestone.

## Next gate: real workflow evaluation

After Speech Pipeline v0 is complete, stop feature development.

Use media-pipeline in one real short-form production with:

- one actual target NLE;
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

- easier Speech invocation;
- media inspection/extraction;
- ASR;
- Vision / media search;
- timeline interchange;
- NLE integration;
- CLI / MCP / HTTP;
- orchestration.

Each must first pass the Feature admission rule with reproducible evidence
before any project code. Vision and ASR remain un-proposed until a real
case re-enters this gate.

## Long-term success

Success does not require the project to grow.

A small, reliable Speech tool that remains useful beside existing NLEs and
Agent integrations is a valid long-term outcome.