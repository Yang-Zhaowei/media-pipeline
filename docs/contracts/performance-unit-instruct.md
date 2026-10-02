# Performance-unit instruction contract — Issue #8

Status: production implementation and CPU validation complete; owner merge
gates below remain pending.
Planning baseline: `main` at `44bdbb6db31db7bead7d9c7c498816e4d96e7862`.
The current PR body is the source of truth for implementation and acceptance
status. This contract defines the behavior reviewed by that PR.

## P1. Ownership and authoring

The caller (including an upper-level Agent) approves coherent performance units
and decides where explicit direction is needed. One speech `segment` is one
such unit and may contain multiple sentences; it remains one TTS request.
media-pipeline does not split, interpret, rewrite, or classify the manuscript.
Alignment and the Caption Compiler independently determine subtitle boundaries.

For ordinary narration, omit top-level `instruct` or use `""`; do not add a
boilerplate style instruction. An instruction is an explicit directing signal
for a requested emotion, emphasis, contrast, or performance change. A shared
top-level instruction remains available for intentional direction of a whole
render, with optional local overrides.

## P2. Backward-compatible resolution

Top-level `instruct` remains optional, defaults to `""`, and must be a string
when present. Each segment additionally accepts optional `instruct`:

| Segment input | Effective TTS instruction | Report source |
| --- | --- | --- |
| Field omitted | Top-level instruction, including its default `""` | `top_level` |
| `"instruct": ""` | `""`, clearing any inherited direction | `segment` |
| `"instruct": "explicit direction"` | The segment string | `segment` |

Resolution depends on field presence, never string truthiness. Strings are
passed verbatim, including surrounding whitespace. JSON `null`, booleans,
numbers, arrays, and objects are invalid instructions at either level.
No input instruction is silently dropped or replaced with a style template.
Old scripts without segment overrides retain their previous TTS instructions.

## P3. Validation and production path

Add only `instruct` to the permitted segment fields (`id`, `text`,
`pause_after_ms`). Keep strict unknown-field rejection and aggregate static
errors before creating a run directory or starting a model stage. Preserve all
existing text, ID, pause, length-budget, model lifecycle, failure/publication,
integer-frame timeline, alignment, and subtitle contracts.

Each segment's resolved instruction must reach the existing
`CustomVoiceRequest.instruct` inside the TTS subprocess. Keep one TTS model load
and one Alignment model load per render, in separate environments. No runtime
generation controls or new public entry point are introduced.

## P4. Provenance

`request.json` retains normalized top-level defaults and includes segment
`instruct` only when explicitly supplied. It therefore preserves inheritance
versus explicit clearing and remains valid input for the same preflight rules.

Each `report.json` segment entry records `effective_instruct` and
`instruct_source` (`segment` or `top_level`). `top_level` includes the empty
default when the top-level field was omitted. The existing
`report.config.instruct` remains the normalized top-level value. Provenance is
present from the initial report and retained in complete, incomplete, and
failed runs; it is not an assertion about the model's audible delivery.

## P5. CPU acceptance

- Cover inheritance, explicit clearing, explicit override, empty defaults,
  verbatim strings, and old-script compatibility.
- Reject invalid instruction types and misspelled/unknown fields at preflight;
  mixed errors must aggregate without invoking either model stage or changing
  the input.
- Verify the actual TTS subprocess script builds the right
  `CustomVoiceRequest` on CPU using a fake runtime, including explicit `""`.
- Verify request replay and report provenance for complete and unsuccessful
  runs, plus a multi-sentence unit producing separate subtitle cues.
- Run focused CPU tests and the full CPU regression; do not modify immutable
  fixtures. Fake runtime checks are CPU evidence, not GPU or listening evidence.

Executed on the client for this implementation:

```text
.venv/Scripts/python.exe -m pytest tests/test_render.py -o addopts=-ra -q
73 passed

.venv/Scripts/python.exe -m pytest -o addopts=-ra -q
306 passed, 6 skipped
```

All six skips are existing gated GPU tests with no runtime configuration. No
GPU inference or human listening acceptance was performed for this PR.

## P6. Merge gates and limits

ai-core GPU validation: pending owner

human listening acceptance: pending owner

Both are merge gates. The owner must validate fresh multi-unit production
output, including inherited direction, explicit empty clearing, an intentional
local change, ordinary empty-instruction narration, and WAV/SRT synchronization.
Record the tested commit, input/configuration, artifacts and runtime results in
the PR body; then record explicit human acceptance there. Historical PR #7
acceptance and experiment-preparation CPU results do not satisfy these gates.

This PR does not claim acoustic continuity or complete Issue #8 from CPU tests.
It must not merge or close the issue while these gates are pending.

## P7. Scope

The experiment branch `origin/codex/issue8-experiments` at `7f108b2` is reference
only; its harness, generated inputs, and tests are not imported into this PR.
No automatic splitting, LLM authoring, emotion analysis, sampling/seed/temperature
controls, regeneration, provider abstraction, HTTP/MCP, VoiceDesign/cloning, or
new subtitle system is included.
