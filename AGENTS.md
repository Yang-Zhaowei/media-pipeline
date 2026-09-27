# Media Pipeline
Local-first AI-assisted media processing pipeline.
## Current milestone
Build the Speech Pipeline:
`text → TTS → forced alignment → audio postprocess → caption compiler → WAV + SRT`
Caption Compiler v0 is complete.
Immediate task: **Audio Postprocess v0**.
Do not expand scope unless explicitly requested.
## Repository evidence
- `experiments/`: previously working GPU smoke tests; reference only.
- `tests/fixtures/`: immutable real model outputs used for regression tests.
- `docs/dev/core-runtime.md`: currently verified ai-core GPU environment.
- `docs/legacy/`: historical architecture; not current requirements.
## Rules
- Prefer deterministic processing over LLM decisions.
- Pure logic and unit tests must run without CUDA.
- Keep paths portable across Windows, Linux, and macOS.
- Never hard-code host paths, drive letters, tokens, or credentials.
- Separate pure processing from model/runtime integration.
- Do not modify regression fixtures unless explicitly requested.
- Avoid speculative abstractions and unnecessary dependencies.
## Out of scope for now
HTTP API, MCP, ASR workflow, Vision, NLE automation, deployment automation, job queues, and UI.
## Validation
Before finishing a change:
- run relevant tests;
- report tests actually executed;
- distinguish unit tests from real GPU integration;
- do not claim GPU validation unless it ran on ai-core.