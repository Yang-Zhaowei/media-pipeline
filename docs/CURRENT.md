# Current State
## Milestone
Speech Pipeline v0.
Current task:
**Audio Postprocess v0**
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
Validated with 62 pure-CPU tests.
The real `speech-smoke-001` fixture produces the reviewed two-caption SRT.
## Verified on ai-core
- Qwen3-TTS CustomVoice generated valid Chinese WAV.
- Qwen3 ForcedAligner produced character-level Chinese timestamps.
- Real outputs are stored in `tests/fixtures/speech-smoke-001/`.
## Next
Implement Audio Postprocess v0:
`raw.wav + alignment.raw.json`
→ trim meaningless leading/trailing audio
→ preserve small padding
→ short fade-in/fade-out
→ shift alignment timestamps
→ cleaned WAV + adjusted alignment
## Not yet started
- production TTS module
- production alignment module
- HTTP API
- MCP
- ASR workflow
- Vision
- NLE integration