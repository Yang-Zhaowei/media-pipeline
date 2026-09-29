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
environments** on ai-core. Rather than requiring either environment to contain
both runtimes, the two model stages run sequentially as child processes and
exchange artifacts through one fresh run directory. The portable stages
(Audio Postprocess, Caption Compiler) and validation run in the parent driver.
No environment consolidation is needed.

## Prerequisites

- ai-core with the TTS and Alignment GPU environments;
- `torch`, `qwen_tts`, and `qwen_asr` importable from each respective
  environment (checked by the run, not hard-coded here);
- CUDA available on the host;
- the production models reachable by their environment-configured paths
  (never hard-coded).

### Exact-head prerequisite (mandatory before an ai-core run)

The audit records a commit SHA, so the run must reflect exactly that commit.
Before running the driver, confirm the working tree is clean:

```bash
git rev-parse HEAD
git status --porcelain
```

The driver itself checks this too and **fails** (never skips) if `git` errors or
the tree is dirty -- commit or stash pending changes first, so the reported HEAD
is the code under test. This makes the recorded SHA the actual code run; `rev-parse`
alone is not enough, since a dirty tree would change the imported code.

The driver does not reimplement a Git abstraction; it only runs `git rev-parse`
and `git status --porcelain` against the repository root.

## Configuration (environment variables)

| Variable | Meaning |
| --- | --- |
| `MEDIA_PIPELINE_TTS_MODEL` | Qwen3-TTS CustomVoice model path (production TTS) |
| `MEDIA_ALIGNMENT_MODEL` or `MEDIA_PIPELINE_ALIGNMENT_MODEL` | Qwen3 ForcedAligner model path (production Alignment) |
| `MEDIA_PIPELINE_TTS_PYTHON` | interpreter of the TTS environment |
| `MEDIA_ALIGNMENT_PYTHON` or `MEDIA_PIPELINE_ALIGNMENT_PYTHON` | interpreter of the Alignment environment |
| `MEDIA_PIPELINE_E2E_RUN_DIR` | a fresh, non-existent directory for the run artifacts; the driver refuses an existing one |
| `MEDIA_PIPELINE_SPEECH_INSTRUCT` | reviewed narration instruction (optional; reviewed default otherwise) |

No host paths, model paths, interpreter paths, tokens, or credentials are
hard-coded. `MEDIA_PIPELINE_E2E_RUN_DIR` **must point to a fresh directory**;
stale artifacts are never reused.

## Run

A **single** driver invocation runs everything. The driver itself launches both
model stages in their own environments through the fresh run directory, so you
invoke the driver once and pass it both environments' interpreters and models:

```bash
# One invocation. The driver runs the TTS stage in the TTS environment and the
# Alignment stage in the Alignment environment, then post-processes, compiles,
# validates, and prints an audit summary.
# Set these to actual model directories, venv interpreters, and a new output
# directory on your integration host. On Linux, a venv interpreter is normally
# <venv>/bin/python.
MEDIA_PIPELINE_TTS_MODEL="/path/to/tts-model" \
MEDIA_PIPELINE_TTS_PYTHON="/path/to/tts-venv/bin/python" \
MEDIA_ALIGNMENT_MODEL="/path/to/alignment-model" \
MEDIA_ALIGNMENT_PYTHON="/path/to/alignment-venv/bin/python" \
MEDIA_PIPELINE_E2E_RUN_DIR="/path/to/new-run-directory" \
  python validation/speech_pipeline_e2e.py
```

The driver creates `MEDIA_PIPELINE_E2E_RUN_DIR` (`exist_ok=False`), so pass a
fresh, non-existent path on every run: stale artifacts are never reused, and
re-invoking the driver against a directory it already populated fails instead of
overwriting a previous run.

This driver reads the fixed `tests/fixtures/speech-smoke-001/original.txt` and
uses `Chinese` / `Uncle_Fu`. Only the narration instruction is configurable.
It is not an arbitrary-manuscript / speaker entry point; see the
[podcast workflow assessment](../docs/workflows/podcast-pilot.md).

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

- The raw input WAV remains byte-identical; the produced `final.wav` keeps the
  sample rate and mono PCM16 format, with frame count equal to the retained
  integer-frame window (which may be shorter than the input). Short fades are applied at the two
  new edges of that kept window, so the kept window's samples are intentionally
  not byte-identical to the raw input. The adjusted timestamps use the actual
  quantized front trim offset; the production audio processing itself follows
  the existing `postprocess_speech` contract.
- Every token keeps its duration and relative order; legitimate gaps are
  preserved; each timestamp shifts by exactly the signed quantized trim offset.
- Alignment integrity: token text, order and count are preserved (repeated
  words are legitimate), raw records pass production validation, and adjusted
  timing feeds the production caption compiler.
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
- Local CPU review result: **233 passed, 6 skipped** at PR head `93c650d`.
- Real GPU E2E on ai-core: **passed** at the same PR head; E2E unit + integration
  suites: **21 passed in 14.81 s**. This is recorded run evidence, not a GPU
  rerun performed by the documentation update.
- Human listening / subtitle synchronization acceptance: **passed**, confirmed
  by the repository owner on 2026-09-29.
- Speech Pipeline v0: **closed**. See
  [closure evidence](../docs/validation/speech-v0-closure.md).

## Closure criteria (satisfied)

Running this driver successfully on ai-core is **not** by itself the closure of
Speech Pipeline v0. Closure requires **all** of the following, in order:

1. an exact-head real ai-core E2E automated validation PASS from this driver;
2. a full human listening pass of the freshly produced `final.wav`;
3. a human inspection that the freshly produced `final.srt` text and timing are
   correctly synced to the audio;
4. an explicit human acceptance sign-off.

Human listening and SRT sync inspection are **required**, never optional.
