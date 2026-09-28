# Media Pipeline
Local-first AI-assisted media processing pipeline.
## Current milestone
Close Speech Pipeline v0 by validating the existing production components end to end:
`text → TTS → forced alignment → audio postprocess → caption compiler → WAV + SRT`
Completed:
- Caption Compiler v0
- Audio Postprocess v0
- Production TTS v0
- Production Alignment v0
Immediate task: **full production Speech Pipeline v0 E2E validation**.
Do not add new modalities, NLE integration, general orchestration, or other scope unless explicitly requested.
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