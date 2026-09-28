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

## Verified on ai-core
- Production Qwen3-TTS CustomVoice integration passes against the real model.
- Qwen3 ForcedAligner produces character-level Chinese timestamps.
- Real speech/alignment regression outputs are stored in `tests/fixtures/speech-smoke-001/`.

## Next

The Production Alignment module v0 is implemented (promoting the previously
verified Qwen3 ForcedAligner experiment into portable project code with GPU/
runtime integration kept separate from pure processing).

The boundary mirrors Production TTS v0: portable alignment contracts /
validation → Qwen ForcedAligner runtime adapter → real ai-core integration.

```text
portable alignment contracts / validation
        ↓
Qwen ForcedAligner runtime adapter
        ↓
real ai-core integration
```

Validate the production stages together end to end:

```text
→ TTS
→ forced alignment
→ audio post-processing
→ caption compilation
→ WAV + SRT
```
Successful end-to-end validation closes the Speech Pipeline v0 milestone.

## Not proposed

HTTP API, MCP, ASR, Vision, and NLE control are not roadmap milestones and
are not under development. They become candidates only if a real workflow
exposes a reproducible gap that passes the Feature admission rule. See
`docs/ROADMAP.md`.