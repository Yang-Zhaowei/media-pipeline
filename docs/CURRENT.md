# Current State
## Milestone
**Speech Pipeline v0 — complete (2026-09-29).**
`text → TTS → forced alignment → audio postprocess → caption compiler → WAV + SRT`

PR #5 is merged. Real ai-core E2E validation and human listening / SRT
synchronization acceptance are complete. See
[closure evidence](validation/speech-v0-closure.md) for the exact tested commit,
artifact metadata, test results, and evidence limits.
## Completed
### Caption Compiler v0
Merged via PR #1.
Implemented:
- deterministic original text + forced alignment → SRT
- punctuation-aware segmentation
- alignment-pause-aware segmentation
- configurable caption length and duration budgets
- natural breakpoint preference before hard cuts
- strict alignment mismatch validation
- CLI and regression tests
The real `speech-smoke-001` fixture produces the reviewed two-caption SRT.
### Audio Postprocess v0
Merged via PR #2.
Implemented:
- alignment-defined leading/trailing audio trimming
- configurable 150 ms pre/post padding
- integer WAV frame boundaries as timing source of truth
- front `floor` / back `ceil` frame quantization
- alignment shifting by the actual quantized trim offset
- alignment duration and precision preservation
- configurable short fade-in/fade-out
- mono 16-bit PCM WAV validation
- real-fixture regression coverage
For `speech-smoke-001`:
```text
input:        24000 Hz / 192000 frames / 8.00 s
kept range:   frame 29040 → 192000
output:       162960 frames / 6.79 s
first speech: 0.15 s
last speech:  6.71 s
```
### Production TTS v0
Merged via PR #3.
Implemented:
- portable `CustomVoiceRequest` and `TTSArtifact` contracts
- deterministic flat-mono float waveform → PCM16 conversion
- strict malformed / non-finite waveform validation
- Qwen3-TTS CustomVoice GPU runtime adapter
- heavy runtime dependencies isolated from portable imports
- one loaded model instance supports repeated synthesis
- generated WAV output directly compatible with Audio Postprocess
- CPU unit tests and separately gated GPU integration tests

PR #3 validation evidence:
```text
PC_Client:
123 passed, 2 skipped

ai-core:
2 GPU integration tests passed in 18.42 s
Python 3.12.14
PyTorch 2.14.0+cu130
CUDA 13.0
NVIDIA GeForce RTX 4070
```

Manual production artifact:
```text
sample rate: 24000 Hz
frames:      165120
duration:    6.88 s
```

The generated Chinese narration was manually listened to and judged acceptable.

Production TTS v0 intentionally operates on one already-segmented utterance
per request; long-text splitting and orchestration remain outside this module.

### Production Alignment v0
Merged via PR #4.
Implemented:
- portable `AlignmentRequest` and `AlignmentArtifact` contracts
- deterministic raw-record timestamp validation rejecting invalid / blank records
- validation applied to every raw record (blank or invalid records are rejected before writing)
- `validate_alignment` enforcing validity and effective-token requirements
- Qwen3 ForcedAligner GPU runtime adapter (promoted from `experiments/`)
- heavy runtime dependencies isolated from portable imports
- one loaded model instance supports repeated alignment
- output written as the existing `{"text", "start", "end"}` JSON that Audio Postprocess and the Caption Compiler consume
- CPU unit tests and separately gated GPU integration tests

Portable validation and mapping logic runs without CUDA; the GPU integration
test is gated on `MEDIA_PIPELINE_ALIGNMENT_MODEL` and skips when it is unset.

Production Alignment v0 intentionally operates on one already-segmented utterance
per request; long-text splitting and orchestration remain outside this module.

### Speech Pipeline E2E Validation v0
Merged via PR #5. Validation code only: portable harness + gated tests + a
driver running the two model stages in separate ai-core environments.

Implemented:
- portable, CUDA-free harness `tests/e2e_validation.py`: `run_chain` / `validate_e2e` checking raw WAV preservation, signed quantized alignment shifts, and token/caption/SRT integrity, with tamper hooks
- CPU unit tests `tests/test_speech_pipeline_e2e_unit.py` (17 tests): end-to-end, invariants, failure injection, and negative cases without CUDA
- gated real-runtime integration test `tests/test_speech_pipeline_e2e_integration.py`: runs the two production model stages in their own environments (fail-not-skip; skips only when GPU E2E config is unset)
- two-process ai-core driver `validation/speech_pipeline_e2e.py` + docs, exchanging artifacts through one fresh run dir and printing an audit summary
- immutable fixtures never substituted for the fresh TTS path; model output checked structurally

Local CPU suite at the reviewed PR head: **233 passed, 6 skipped**. All six
skips are gated GPU tests; they are not GPU validation.

The real E2E remains a two-process sequence (Production TTS and Production
Alignment live in separate ai-core venvs) and is explicitly gated on GPU E2E.

Closure requirements have all been met:

1. Real ai-core E2E automated validation passed at PR head `93c650d`.
2. Full human listening of the fresh `final.wav` passed.
3. Human inspection of the fresh SRT text and synchronization passed.
4. The repository owner explicitly confirmed human acceptance on 2026-09-29.

Human listening and SRT sync inspection are required, never optional.

## Verified on ai-core
- Production Qwen3-TTS CustomVoice integration passes against the real model.
- Qwen3 ForcedAligner produces character-level Chinese timestamps.
- Real speech/alignment regression outputs are stored in `tests/fixtures/speech-smoke-001/`.
- Fresh Production TTS -> Alignment -> Postprocess -> Captions passed on ai-core
  for PR #5, producing a 6.160 s final WAV and two SRT captions. These outputs
  are separate from the immutable regression fixture.

## Work in progress
### Segmented Speech Render Entry Point v0 (unmerged — PR on `feat/segmented-speech-v0`)

Adds a single sync orchestration boundary on top of the v0 components:

```python
from media_pipeline import render_speech, load_and_validate_script, RenderError
render_speech(
    script_path="script.json",           # validated up front, never rewritten
    output_dir="run_dir",                 # must not already exist
    tts_python="python",
    alignment_python="python",
    tts_model="/models/qwen3-tts",
    alignment_model="/models/qwen3-forced-aligner",
    max_segment_chars=200,
)
```

Implemented and **CPU-verified** (unit tests, no CUDA):
- `render_speech(...)` orchestrator + `load_and_validate_script(...)` preflight.
- One model load per stage, run as separate one-shot subprocesses; production
  code never imports `torch`, the TTS/Alignment runtimes, or `tests/` at runtime.
- Deterministic, CUDA-free path: preflight, timeline, integer-frame WAV
  assembly, and final verification. All pure-logic unit tests run without CUDA.
- Failure semantics (per contract C1–C9): preflight & run-level faults raise
  `RenderError`; per-segment alignment/postprocess/caption failures are recorded
  and skipped while other segments still publish; no partial episode is written
  for a failed run, and no catch-all continue.
- Integer-frame timeline `O_i = Σ(N_j + G_j)` with global caption time = local
  + `O_i / R`; concatenates only raw WAV data frames and rewrites the header;
  SRT rendered only at the end with the existing half-up rules; no drift
  tolerance beyond 0.5 ms + float.
- C6 layout: `request.json`, `report.json`, `segments/`, `final/` published
  via `.final.staging` → `.complete`, and `output_dir` refuse-if-exists.
- `tests/test_render.py` — 25 unit tests covering the C1–C9 checklist (lifecycle
  load-once, alignment-failed-continue, runtime-fault-stop, timeline/frame
  accuracy, drift, sample-rate mismatch, zero/overlap/past captions, original
  WAV preservation, output refusal, and marker discipline).

**Not yet executed — pending:**
- Real ai-core multi-segment validation (≥3 unequal segments with pauses, with
  per-stage load-once proof). Do not substitute the PR #5 single-utterance
  evidence.
- Human listening and SRT synchronization acceptance for multi-segment output.

See [contract](contracts/segmented-speech-v0.md) for C1–C9 and the
[render entry-point docs](dev/render-entry-point.md).

## Next

Evaluate one real chapter-based podcast production, with at least one script
revision and a final video export. See the
[workflow assessment](workflows/podcast-pilot.md). No new implementation is
committed by that assessment. This does not authorize the segmented-speech
entry point to proceed past its pending items.

Current production APIs operate on one already-segmented utterance. The E2E
driver fixes the script and speaker. A convenient manuscript/voice input,
long-script segmentation, and whole-episode WAV/SRT assembly are not delivered
by PR #5. First establish what the real pilot needs and what existing tools
already cover; do not introduce a generic orchestration subsystem.

## Not proposed

HTTP API, MCP, ASR, Vision, and NLE control are not roadmap milestones and
are not under development. They become candidates only if a real workflow
exposes a reproducible gap that passes the Feature admission rule. See
`docs/ROADMAP.md`.
