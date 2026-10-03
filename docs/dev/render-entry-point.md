# Speech rendering

`render_speech(...)` turns a caller-authored script into one episode WAV, SRT
and timeline. The caller defines coherent performance units and any explicit
direction. A unit may contain multiple sentences; subtitle boundaries are
determined independently by Alignment and the Caption Compiler.

## Script

The existing `speaker` form remains compatible. For CustomVoice, ordinary
narration omits `instruct` or uses `""`. A segment can inherit the top-level
instruction, explicitly clear it, or provide local direction.

```json
{
  "language": "Chinese",
  "speaker": "Uncle_Fu",
  "instruct": "开场时带有克制的紧迫感。",
  "segments": [
    {"id": "opening", "text": "欢迎收听本期节目。今天我们讨论一个具体问题。"},
    {"id": "plain", "text": "接下来回到普通叙述。", "instruct": ""},
    {"id": "emphasis", "text": "这个结论值得关注。", "instruct": "强调最后一句。"}
  ]
}
```

Here `opening` inherits, `plain` clears, and `emphasis` overrides the direction.
`pause_after_ms` optionally adds a pause after a unit. Instructions must be
strings; unknown fields are rejected. See the
[instruction contract](../contracts/performance-unit-instruct.md).

### Reusable cloned voice

Select exactly one top-level voice source: legacy `speaker`, or this narrow
`voice` object. There is no built-in-speaker `voice` object or provider field.

```json
{
  "language": "Chinese",
  "voice": {"type": "clone", "asset": "voices/zhaowei-v1.pt"},
  "segments": [
    {"id": "opening", "text": "欢迎收听本期节目。今天我们讨论一个具体问题。"},
    {"id": "detail", "text": "接下来说明这个问题的背景。", "instruct": ""}
  ]
}
```

`voice` supports exactly `type: "clone"` and a non-empty relative `asset` path.
Both `speaker` and `voice`, neither source, unknown fields/types, absolute or
drive/URI paths, and paths resolving outside the script directory are rejected.
Paths resolve from the directory containing the input script, including
existing symlinks. Slash or backslash separators work for relative paths;
normalization is used only for internal resolution. Authored spelling is kept
in the plan, `request.json`, and `report.json` config.

Cloned voices allow only omitted or exactly empty top-level and segment
`instruct`. Any non-empty string, including whitespace or a top-level value
cleared by every segment, fails static preflight with
`non-empty instruct is unsupported for cloned voices`. No instruction is
ignored, removed, or emulated. CustomVoice inheritance/clear/override is unchanged.

## Render

```python
from media_pipeline import render_speech

render_speech(
    script_path="script.json",
    output_dir="run_dir",
    tts_python="<TTS Python path>",
    alignment_python="<Alignment Python path>",
    tts_model="<TTS model path>",
    alignment_model="<Alignment model path>",
    max_segment_chars=200,
)
```

Use the verified [ai-core runtime setup](core-runtime.md). The output directory
must be new. `max_segment_chars` is the caller's per-unit limit; over-budget
text is rejected, never truncated. Invalid scripts fail before generation.

The caller supplies a CustomVoice checkpoint for `speaker` scripts and a Base
checkpoint for clone scripts through the same `tts_model` / CLI configuration.
The single TTS subprocess selects `Qwen3CustomVoiceTTS` + `CustomVoiceRequest`
or safely loads the clone asset once, creates `Qwen3VoiceCloneTTS` once, and
reuses that engine and asset for all `VoiceCloneRequest` segments. Model paths
remain runtime configuration and do not belong in `speech.json`.

A complete run publishes `final/final.wav`, `final/final.srt` and
`final/timeline.json`. `request.json` preserves the authored instruction choices;
`report.json` records their effective values and run status.
Legacy provenance records `speaker`; clone provenance records the authored
relative `voice` object without a fabricated speaker, resolved asset path,
tensor payload, reference recording/transcript source, hash, or registry ID.
Alignment, Postprocess, Caption Compiler, and final assembly are shared unchanged.

Missing/malformed/rejected assets, incompatible checkpoints, Base load failure,
and clone synthesis failure use the existing structured failed-run contract,
retain the original reason, and publish no completion marker.

## Validation and next step

[PR #9](https://github.com/Yang-Zhaowei/media-pipeline/pull/9) is merged. Real
ai-core generation, human listening and subtitle synchronization passed.
[PR #11](https://github.com/Yang-Zhaowei/media-pipeline/pull/11) is merged and
[Issue #10](https://github.com/Yang-Zhaowei/media-pipeline/issues/10) is closed.
The owner marked the real ai-core CLI render and human acceptance merge gates
complete in PR #11. Recorded CPU validation and closure status are in
[CURRENT](../CURRENT.md#agent-facing-speech-cli); this documentation update
does not rerun model integration.

[PR #15](https://github.com/Yang-Zhaowei/media-pipeline/pull/15) is merged and
Issue #14 is closed. The owner accepted a real three-unit clone render,
one asset/Base load, complete downstream artifacts, identity/consistency,
transitions, and subtitle sync. See the
[exact-head evidence and replay procedure](../validation/clone-render-acceptance.md).

[Issue #8](https://github.com/Yang-Zhaowei/media-pipeline/issues/8) remains open
for production narration stability, including cross-segment emotion/delivery drift.

The renderer does not automatically split, interpret or rewrite the manuscript.

## Command line

Install the package from a checkout with `python -m pip install -e .` or
`uv sync`. Use an activated virtual environment, or prefix the commands with
`uv run` after `uv sync`. The console command is a thin wrapper around
`load_and_validate_script(...)` and `render_speech(...)`:

```console
media-pipeline speech validate speech.json
media-pipeline speech render speech.json --output new-run-dir
```

Both commands accept `--max-segment-chars` (default `200`), the caller's
preflight budget measured in Unicode code points. It is not a model limit.
`validate` reads UTF-8 JSON and performs static preflight only: it does not load
models, deserialize clone assets, import torch/qwen_tts, create a run directory,
or modify the input. Clone asset existence, safe load, and checkpoint compatibility
are checked at render time. The plan reports language, legacy `speaker` or the
authored clone `voice` object (without `speaker`), the configured budget, and
each performance unit with its one-based
order, id, text, `text_length` (Unicode code points, including whitespace and
punctuation), `pause_after_ms`, `effective_instruct`, and `instruct_source`
(`top_level` or `segment`).

### Output and exit codes

For results and handled errors, stdout contains one pretty-printed UTF-8 JSON
object and stderr is empty. Every object has `schema_version: 1`, `command` (`validate`,
`render`, or `null` for command-line usage errors), and a `status`. Absolute
paths are used for reported artifacts. `--help` prints ordinary human-readable
help and exits `0`.

| Exit | Status | Meaning |
| ---: | --- | --- |
| `0` | `valid` | Static validation passed; includes `script_path` and `plan`. |
| `0` | `complete` | Render finished; includes absolute run, report, WAV, SRT, and timeline paths plus segment results. |
| `2` | `error` | CLI syntax, missing runtime configuration, or a pre-run render error (including an existing output directory). |
| `3` | `validation_failed` | Static preflight failed; `error` preserves the original message, including aggregated script errors. |
| `4` | `incomplete` | Rendering returned incomplete; final WAV, SRT, and timeline paths are `null`, and `segments` carries each result's `segment_id`, `status`, `stage`, and `reason`. |
| `5` | `failed` | A `RenderError` carries `run_dir`; includes a best-effort `report_path` (or `null` if absent) and null final artifact paths. |

Render result paths use the keys `run_dir`, `report_path`, `wav_path`, `srt_path`,
and `timeline_path`. Errors include an `error` string. A failure to create the
run directory can also return `failed`, with no report available.

An `incomplete` result is not a published episode. `failed` identifies a
run-level failure; the report path is included only when `report.json` exists.
The input script is never rewritten, and the CLI does not claim final products
for either status.

### Render runtime options

The `render` command requires `--output <run_dir>` and delegates the new-directory
check to the renderer. Its optional overrides are `--tts-python`,
`--alignment-python`, `--tts-model`, and `--alignment-model`; they correspond to
the `MEDIA_PIPELINE_TTS_PYTHON`, `MEDIA_PIPELINE_ALIGNMENT_PYTHON`,
`MEDIA_PIPELINE_TTS_MODEL`, and `MEDIA_PIPELINE_ALIGNMENT_MODEL` environment
variables. The existing `MEDIA_ALIGNMENT_PYTHON` and `MEDIA_ALIGNMENT_MODEL`
names remain supported as aliases and take precedence over their canonical
`MEDIA_PIPELINE_` variables when no CLI override is supplied. CLI values take
precedence over environment variables. Missing or whitespace-only values are
rejected. Interpreter and model paths are supplied by the caller; the CLI does
not infer or hard-code them. `--device` overrides the render device and defaults
to `cuda:0`.

The Python `render_speech(...)` API above remains available with its existing
signature and semantics. The legacy caption command
`python -m media_pipeline original.txt alignment.raw.json -o captions.srt`
also remains unchanged; this console command is a separate entry point.
