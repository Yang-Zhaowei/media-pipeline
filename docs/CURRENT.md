# Current State
## Milestone
Speech Pipeline v0.
Current task:
**Production TTS module v0**
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
Full pure-CPU test suite:104 passed
```

## Verified on ai-core
- Qwen3-TTS CustomVoice generated valid Chinese WAV.
- Qwen3 ForcedAligner produced character-level Chinese timestamps.
- Real outputs are stored in `tests/fixtures/speech-smoke-001/`.
## Next
Implement **Production TTS module v0** by promoting the already verified TTS experiment into portable project code while keeping GPU/runtime integration separate from pure processing.
Then implement Production **Alignment module v0**.
## Not yet started
- production TTS module
- production alignment module
- HTTP API
- MCP
- ASR workflow
- Vision
- NLE integration