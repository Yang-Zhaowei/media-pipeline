---
name: regrain-speech
description: Author and statically validate caller-approved Regrain speech.json scripts, then hand them off safely for rendering. Use when preparing spoken narration for Regrain's speech pipeline.
---

# Regrain speech authoring

Create UTF-8 `speech.json` from an approved manuscript, then run the installed Regrain CLI as the authoritative validator. Do not invent spoken content or production direction beyond the caller's approval.

## Build the script

Prefer coherent, multi-sentence Performance Units for short, connected narration when they fit the approved budget. Do not split mechanically at every sentence: introduce units for meaningful semantic, performance, revision, or structural reasons. This is authoring guidance, not a validation rule; final boundaries need human approval. The default ceiling is **200 Unicode code points per unit**, including whitespace and punctuation, not an optimal length or model limit. Never silently merge, split, truncate, rewrite, or omit approved text; seek approval for needed changes.

Keep spoken words in `text`. Put explicit, caller-approved delivery direction in `instruct`. Keep production notes, edit instructions, and visual direction outside `speech.json`; they are not spoken text or renderer controls. Subtitle segmentation is produced independently by alignment and caption compilation, so Performance Unit boundaries do not prescribe subtitle boundaries.

Use exactly one voice source:

- **CustomVoice:** a non-empty `speaker` string, such as a checkpoint-supported speaker name. Optional top-level `instruct` defaults to `""`.
- **Clone:** `"voice": {"type": "clone", "asset": "relative/path.pt"}`. Do not include `speaker`. The asset path must be non-empty, relative to the script's directory, and resolve within that directory after symlink resolution. No drive, root or URI scheme is allowed. Use only a trusted, owner-approved voice asset; do not put model/checkpoint or interpreter paths in the JSON. Static validation does not load or deserialize the asset; rendering does.

Both forms require non-empty `language` and a non-empty `segments` array. Each segment requires a unique, non-empty `id` and valid spoken `text`. `pause_after_ms` is optional, defaults to `0`, and must be a non-negative integer (not a boolean); it adds postprocessed silence after that unit. Segment `instruct` is optional and must be a string when present. Unknown fields are rejected.

For CustomVoice, omitted segment `instruct` inherits the top-level direction; `"instruct": ""` clears it; a non-empty string overrides it. Ordinary narration can omit both levels. For clone, top-level and segment `instruct` must be omitted or exactly `""`; whitespace is non-empty and rejected. Clone mode does not support instruction text. Do not add experimental sampling or generation controls as production fields.

See [CustomVoice example](examples/custom-voice.json) and [clone example](examples/clone.json).

## Validate and hand off

Run the installed CLI from the script's directory or pass its path:

```console
regrain --version
regrain speech validate speech.json
```

Use `regrain speech validate --help` for options. The default budget is 200; `--max-segment-chars` changes only this caller-side budget. Validation is static: it reads the script and checks its contract without loading models or clone assets, creating render output, or modifying the input.

The CLI prints a JSON result. Exit `0` with `status: "valid"` means preflight passed. Exit `3` with `status: "validation_failed"` includes the validation error; correct the source or request clarification, then rerun validation. Exit `2` indicates command usage or another CLI error. Keep the original error and do not claim validation passed after a failed run.

After validation, obtain human approval of the exact script. Then manually transfer the approved script and its authorized relative voice asset (for clone) to the Core machine, preserving their directory relationship. Core supplies its existing TTS and alignment interpreters and checkpoints through local configuration; keep those paths and private assets out of `speech.json` and source control. Run `regrain speech render` there with a new output directory. Treat only `status: "complete"` as a completed render; `incomplete` or `failed` results do not publish a complete episode. Transfer outputs back manually. Regrain does not transfer files or require Revisual.

## Evidence boundary

PR #16 records owner-reported results from 13 successful real GPU experiment runs and blind listening. Continuous generation showed promising prosody continuity on a short corpus of approximately 15 seconds. This is exploratory evidence: it does not establish a universally optimal unit length or multi-minute narration stability. Issue #8 remains open. Keep units semantically coherent and caller-approved; do not infer a universal duration or expose experiment controls in production scripts.

## Install this Skill on an Agent host

Make this `regrain-speech` Skill available through the host's supported Agent Skills discovery or registration mechanism. Different hosts use different locations and configuration; copy/register this folder using that host's documented process, then confirm the Agent can see the Skill. Do not assume a shared discovery path across hosts.
