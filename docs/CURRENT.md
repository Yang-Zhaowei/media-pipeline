# Current State
## Milestone
Speech Pipeline v0.
Current task:
**End-to-end validation of the full production Speech Pipeline v0**
`text → TTS → forced alignment → audio postprocess → caption compiler → WAV + SRT`
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
Validation code (no product changes): portable harness + gated tests + a two-process ai-core driver.

Implemented:
- portable, CUDA-free harness `tests/e2e_validation.py`: `run_chain` / `validate_e2e` encoding every required E2E property (WAV unchanged by trim, alignment shift = quantized offset, token/caption/SRT integrity, production postprocess + captions + alignment validation), with tamper hooks
- CPU unit test `tests/test_speech_pipeline_e2e_unit.py` (10 tests): end-to-end + per-property + failure/injection + negative cases with production interfaces, no CUDA
- gated real-runtime integration test `tests/test_speech_pipeline_e2e_integration.py`: runs the two production model stages in their own environments (fail-not-skip; skips only when GPU E2E config is unset)
- two-process ai-core driver `validation/speech_pipeline_e2e.py` + docs, exchanging artifacts through one fresh run dir and printing an audit summary
- immutable fixtures never substituted for the fresh TTS path; model output checked structurally

Local CPU suite: 223 passed, 6 skipped (was 213 passed, 5 skipped).

The real E2E remains a two-process sequence (Production TTS and Production
Alignment live in separate ai-core venvs) and is explicitly gated on GPU E2E.
Real E2E on ai-core is the last step to close the milestone.

## Verified on ai-core
- Production Qwen3-TTS CustomVoice integration passes against the real model.
- Qwen3 ForcedAligner produces character-level Chinese timestamps.
- Real speech/alignment regression outputs are stored in `tests/fixtures/speech-smoke-001/`.

## Next

The end-to-end validation harness for the full production Speech Pipeline v0
is complete on PC_Client (all production stages merged; suite green). Milestone
closure now depends only on the real ai-core validation.

Run the two-process E2E once, on ai-core, using the TTS and Alignment
environments (see `validation/README.md`):

```text
→ TTS          (TTS environment):   text → raw.wav
→ forced alignment (Alignment env.): raw.wav → alignment.raw.json
→ audio post-processing (portable): raw.wav + alignment.raw.json → final.wav + adjusted.json
→ caption compilation (portable):   → final.srt
→ validate_e2e (all required properties)
```

Success on ai-core closes the Speech Pipeline v0 milestone.

## Not proposed

HTTP API, MCP, ASR, Vision, and NLE control are not roadmap milestones and
are not under development. They become candidates only if a real workflow
exposes a reproducible gap that passes the Feature admission rule. See
`docs/ROADMAP.md`.