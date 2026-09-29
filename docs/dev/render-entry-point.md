# Render entry point (`render_speech`)

## Status

`render_speech` is the merged pre-segmented render entry point (PR #7). The real
ai-core multi-segment run, load-once instrumentation and recorded human
acceptance are complete; see the [acceptance review](../validation/segmented-speech-v0-acceptance.md)
and [project state](../CURRENT.md). These records are separate from PR #5 evidence.

## Purpose

A single synchronous Python call renders a **pre-segmented** script to one
episode WAV + matching SRT in a `run_dir/`:

```python
from media_pipeline import render_speech, load_and_validate_script, RenderError

render_speech(
    script_path="script.json",        # validated up front, never rewritten
    output_dir="run_dir",             # must not already exist
    tts_python="python",
    alignment_python="python",
    tts_model="/models/qwen3-tts",
    alignment_model="/models/qwen3-forced-aligner",
    max_segment_chars=200,
)
```

It is **not** a full-manuscript interface. Automatic long-script splitting,
full-manuscript authoring, and exact numeric speed/pitch control are out of
scope. Each `segment` already carries its `text`, `id`, and `pause_after_ms`.

## Inputs

- `script_path`: JSON validated by `load_and_validate_script(...)`.
- `max_segment_chars`: positive integer, the hard per-segment cap. Segments over
  the cap are rejected (never truncated).
- `tts_python` / `alignment_python` / `tts_model` / `alignment_model`: the model
  stage interpreter, model path, and stage model path.

Internal seams (injected in tests, real in production):

- `device`: runtime device; the recorded real validation used `cuda:0`.
- `_wav_task` / `_align_task`: the two one-shot stage callables. Production
  parent imports stay portable; model-runtime imports happen inside the stage
  subprocesses. Production code does not depend on `tests/`.

## Lifecycle

One model load per stage, run once as separate subprocesses:

```
TTS stage (subprocess, one model load)   → per-segment raw WAVs + manifest
Alignment stage (subprocess, one load)   → per-segment raw alignment
Audio Postprocess + Caption Compiler     → cleaned WAV + adjusted alignment + captions
Integer-frame assembly + SRT             (in-process, deterministic)
```

No per-segment model lifecycle and never two models resident in one process.

## Failure semantics (contract C5)

- **Preflight** static faults (unknown/duplicate fields, over-max, non-string
  ids, bool pause/budget, etc.) raise `RenderError` **before any model loads**;
  the input file is never rewritten.
- **Run-level faults** raise `RenderError` (with `run_dir`): model/CUDA/
  subprocess faults, TTS synthesis failures, invalid segment WAVs, sample-rate
  mismatch, and the final product verification failure. No catch-all continue.
- **Per-segment deterministic faults** (alignment, postprocess, caption) are
  recorded with `segment_id`/`stage`/`reason`; independent segments continue.
  The result is `incomplete`, with intermediate artifacts retained and all
  final paths `None`. No shortened episode or successful-episode groups publish.
- A failed run never leaves a valid `.complete` marker; a completed run writes
  `.complete` last.

## Timeline & assembly (contract C7)

- Integer-frame accumulation: `O_i = Σ (N_j + G_j)` for `j < i`, with
  `G_i = half_up(pause_after_ms_i · R / 1000)`.
- Global caption time = local caption time + `O_i / R`.
- Only the cleaned WAV **data frames** are concatenated; the header is rewritten for
  the assembled length and sample rate. SRT is rendered only at the end with the
  existing half-up rules.
- Millisecond SRT endpoints allow at most 0.5 ms rounding plus floating-point
  tolerance beyond the audio boundary; offsets accumulate integer frames.

## Output layout (contract C6)

```
run_dir/
  request.json          validated request values, with defaults materialized
  report.json           manifest of segment statuses
  segments/<name>.wav                 per-segment raw WAV
  segments/<name>.alignment.raw.json
  segments/<name>.alignment.adjusted.json
  segments/<name>.cleaned.wav
  .final.staging/        staged WAV/SRT/timeline, re-read before publication
  final/                artifacts copied after staging validation
    final.wav
    final.srt
    timeline.json
    .complete           completion marker, written last
```

## Testing

`tests/test_render.py` exercises the full deterministic path on CPU (no CUDA)
with two injectable fake model engines, covering the C1–C9 checklist: lifecycle
load-once, alignment-failed-continue, runtime-fault-stop, timeline/frame
accuracy, accumulation drift, sample-rate mismatch, zero/overlap/past captions,
original-WAV preservation, output-directory refusal, and marker discipline.
Pure-logic unit tests run without CUDA; the local full suite passes with only
six gated GPU skips (including 54 render tests). The ai-core generation, load-once
instrumentation and human acceptance are recorded in the acceptance review. A
successful report and `final/.complete` are required: an interrupted publication
can leave unfinished files in `final/`.
