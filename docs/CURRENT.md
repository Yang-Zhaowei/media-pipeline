# Current State
## Milestone
Speech Pipeline v0.
Current task:
**Caption Compiler v0**
## Verified
On ai-core RTX 4070:
- Qwen3-TTS CustomVoice generated valid Chinese WAV.
- Qwen3 ForcedAligner produced character-level Chinese timestamps.
- Real outputs are stored in `tests/fixtures/speech-smoke-001/`.
## Next
Implement deterministic:
`original.txt + alignment.raw.json → captions.srt`
Then implement Audio Postprocess.
## Not yet started
- production TTS module
- production alignment module
- HTTP API
- MCP
- ASR workflow
- Vision
- NLE integration