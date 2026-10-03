# Performance-unit instructions

Merged in [PR #9](https://github.com/Yang-Zhaowei/media-pipeline/pull/9).

The caller, including an upper-level Agent, defines coherent performance units
and decides where explicit direction is needed. One segment may contain multiple
sentences and produces one TTS request. Alignment and the Caption Compiler
determine subtitle boundaries independently.

## Instruction rules

For legacy CustomVoice `speaker` scripts, top-level `instruct` is optional and
defaults to `""`. Ordinary narration uses
this empty default, without boilerplate direction.

| Segment input | Effective instruction |
| --- | --- |
| `instruct` omitted | Inherit the top-level value |
| `"instruct": ""` | Clear inherited direction |
| Non-empty string | Use the local direction |

Strings are passed verbatim. Non-string values, including `null`, and unknown
fields are rejected before rendering. Existing scripts retain their behavior.

For `voice.type == "clone"`, top-level and segment instructions must be omitted
or exactly `""`. Every non-empty string (including whitespace) is rejected at
static preflight, even if all segments clear a non-empty top-level instruction.
The error is `non-empty instruct is unsupported for cloned voices`.
See the [voice-source contract](../dev/render-entry-point.md#reusable-cloned-voice).

`request.json` preserves omission versus explicit clearing for replay.
Each report segment records `effective_instruct` and `instruct_source`
(`top_level` or `segment`), including unsuccessful runs. These describe the
request, rather than guaranteeing audible delivery.

## Acceptance and follow-up

Real ai-core generation, full human listening and subtitle synchronization
passed at `031b2cc1d1ed0f17b05da94f98f84fe29c266d43`. The acceptance record is in
[PR #9](https://github.com/Yang-Zhaowei/media-pipeline/pull/9).

[Issue #8](https://github.com/Yang-Zhaowei/media-pipeline/issues/8) remains open
for production-use validation of long-form narration stability.

This contract adds no automatic segmentation, emotion analysis or regeneration.
