# Speech Pipeline v0 — two-process end-to-end validation

Real end-to-end validation of the full production Speech Pipeline v0 on ai-core:

```text
→ TTS                 (TTS environment):   text → raw.wav
→ forced alignment    (Alignment env.):    raw.wav → alignment.raw.json
→ audio post-processing (portable):        raw.wav + alignment.raw.json → final.wav + adjusted.json
→ caption compilation (portable):          → final.srt
→ validate_e2e                              all required E2E properties
```

## Why two processes

Production TTS and Production Alignment live in **two separate GPU virtual
environments** on ai-core. A single process cannot import both, so the two
model stages run in their own environments and exchange explicit artifacts
through one fresh run directory. Only the portable stages (Audio Postprocess,
Caption Compiler) plus the validation run in the driver process. This is
inherently a two-process sequence.

## Prerequisites

- ai-core with the TTS and Alignment GPU environments;
- `torch`, `qwen_tts`, and `qwen_asr` importable from each respective
  environment (checked by the run, not hard-coded here);
- CUDA available on the host;
- the production models reachable by their environment-configured paths
  (never hard-coded).

## Configuration (environment variables)

| Variable | Meaning |
| --- | --- |
| `MEDIA_PIPELINE_TTS_MODEL` | Qwen3-TTS CustomVoice model path (production TTS) |
| `MEDIA_ALIGNMENT_MODEL` or `MEDIA_PIPELINE_ALIGNMENT_MODEL` | Qwen3 ForcedAligner model path (production Alignment) |
| `MEDIA_PIPELINE_TTS_PYTHON` | interpreter of the TTS environment |
| `MEDIA_ALIGNMENT_PYTHON` or `MEDIA_PIPELINE_ALIGNMENT_PYTHON` | interpreter of the Alignment environment |
| `MEDIA_PIPELINE_E2E_RUN_DIR` | fresh directory for the run artifacts (created if absent) |
| `MEDIA_PIPELINE_SPEECH_INSTRUCT` | reviewed narration instruction (optional; reviewed default otherwise) |

No host paths, model paths, interpreter paths, tokens, or credentials are
hard-coded. `MEDIA_PIPELINE_E2E_RUN_DIR` **must point to a fresh directory**;
stale artifacts are never reused.

## Run

```bash
# TTS environment
MEDIA_PIPELINE_TTS_MODEL=... \
MEDIA_PIPELINE_TTS_PYTHON=/path/to/tts/.venv/python \
MEDIA_PIPELINE_E2E_RUN_DIR=/tmp/e2e/run1 \
MEDIA_PIPELINE_SPEECH_INSTRUCT="自然、清晰、克制的中文知识类视频旁白，语速中等，不要夸张。" \
  python validation/speech_pipeline_e2e.py

# Alignment environment
MEDIA_ALIGNMENT_MODEL=... \
MEDIA_ALIGNMENT_PYTHON=/path/to/aligner/.venv/python \
MEDIA_PIPELINE_E2E_RUN_DIR=/tmp/e2e/run1 \
  python validation/speech_pipeline_e2e.py
```

Run the two model stages (each in its own environment) pointing at the same
`MEDIA_PIPELINE_E2E_RUN_DIR`; the driver synthesizes, aligns, post-processes,
compiles captions, validates, and prints an audit summary.

## What it does

1. **Process 1 — Production TTS**: runs a small stage script in the TTS
   environment, synthesizing the fresh `raw.wav` from the reviewed input.
2. **Process 2 — Production Alignment**: runs a small stage script in the
   Alignment environment, producing `alignment.raw.json` from `raw.wav`.
3. **Process 3 — portable stages + validation**: in-process, post-processes
   (`postprocess_speech`), compiles captions and SRT (`build_captions`,
   `compile_srt`), builds an `E2EArtifacts` record, and runs
   `validate_e2e` from `tests/e2e_validation.py`.

## Required properties validated (`validate_e2e`)

- The trimmed WAV is unchanged except for the quantized trim window
  (`frame_rate` / `sample_rate` / `channels` / byte content outside the window).
- Every token keeps its duration and relative order; every gap is preserved;
  each timestamp shifts by exactly the quantized trim offset.
- Alignment integrity: no duplicate tokens, no overlap, no gaps, no blank
  records, and no token running past the trimmed WAV.
- Caption and SRT integrity: the SRT text equals the reviewed original text,
  the caption count matches the caption compiler, and every caption timestamp
  is non-empty and inside the final WAV.
- Production postprocess, captions, and alignment validation all succeed on the
  actual produced artifacts (not fixtures).

On success it prints an audit summary (commit SHA, WAV metadata, trim frame
range, token and caption counts, SRT path, validation result). On any failure
it exits non-zero — it never skips and never reuses stale artifacts.

## Verification status

- Local CPU suite: green (unit tests exercise the full chain and the same
  required properties without CUDA). See `tests/test_speech_pipeline_e2e_unit.py`
  and `tests/test_speech_pipeline_e2e_integration.py` (the integration test is
  gated on GPU E2E and skips on PC_Client).
- Real GPU E2E on ai-core: **pending** — this driver run is the final step that
  closes the Speech Pipeline v0 milestone.
