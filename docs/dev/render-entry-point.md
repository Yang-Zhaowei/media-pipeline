# Render entry point (`render_speech`)

## Status

**Unmerged work in progress** on branch `feat/segmented-speech-v0`. This entry
point sits on top of the Speech Pipeline v0 components (Production TTS,
Production Alignment, Audio Postprocess, Caption Compiler) and is validated on
CPU by unit tests. **Real ai-core multi-segment validation and human-listening
acceptance are pending** — see [project state](CURRENT.md).

Do not substitute the PR #5 single-utterance validation for the pending
multi-segment validation.

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

It is **not** a full-manuscript interface. Automatic long-script splitting (ASR),
full-manuscript authoring, and exact numeric speed/pitch control are out of
scope. Each `segment` already carries its `text`, `id`, and `pause_after_ms`.

## Inputs

- `script_path`: JSON validated by `load_and_validate_script(...)`.
- `max_segment_chars`: positive integer, the hard per-segment cap. Segments over
  the cap are rejected (never truncated).
- `tts_python` / `alignment_python` / `tts_model` / `alignment_model`: the model
  stage interpreter, model path, and stage model path.

Internal seams (injected in tests, real in production):

- `device`: which model to load per stage (CPU / CUDA).
- `_wav_task` / `_align_task`: the two one-shot stage callables. Production
  code never imports `torch`, the TTS/Alignment runtimes, or `tests/` at
  runtime; those imports happen lazily inside the stage subprocesses.

## Lifecycle

One model load per stage, run once as separate subprocesses:

```
TTS stage (subprocess, one model load)   → per-segment raw WAVs + manifest
Alignment stage (subprocess, one load)   → per-segment raw/adjusted alignment
Audio Postprocess + Caption Compiler     (in-process, deterministic)
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
  recorded with `segment_id`/`stage`/`reason` and skipped; other segments still
  publish. A run with any such fault produces a **partial** run — no complete
  episode is published.
- A failed run never leaves a valid `.complete` marker; a completed run writes
  `.complete` last.

## Timeline & assembly (contract C7)

- Integer-frame accumulation: `O_i = Σ (N_j + G_j)` for `j < i`, with
  `G_i = half_up(pause_after_ms_i · R / 1000)`.
- Global caption time = local caption time + `O_i / R`.
- Only the raw WAV **data frames** are concatenated; the header is rewritten for
  the assembled length and sample rate. SRT is rendered only at the end with the
  existing half-up rules.
- No drift tolerance beyond 0.5 ms + a single float.

## Output layout (contract C6)

```
run_dir/
  request.json          validated input (verbatim)
  report.json           manifest of segment statuses
  segments/<name>.wav                 per-segment raw WAV
  segments/<name>.alignment.raw.json
  segments/<name>.alignment.adjusted.json
  segments/<name>.cleaned.wav
  final/                published only after validation
    final.wav
    final.srt
    timeline.json
    report.json
    .final.staging      staging marker
    .complete           completion marker, written last
```

## Testing

`tests/test_render.py` exercises the full deterministic path on CPU (no CUDA)
with two injectable fake model engines, covering the C1–C9 checklist: lifecycle
load-once, alignment-failed-continue, runtime-fault-stop, timeline/frame
accuracy, accumulation drift, sample-rate mismatch, zero/overlap/past captions,
original-WAV preservation, output-directory refusal, and marker discipline.
Pure-logic unit tests run without CUDA; real GPU validation is pending.
