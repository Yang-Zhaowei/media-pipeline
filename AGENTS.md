# Media Pipeline
Local-first AI-assisted media processing pipeline.
## Current state
Speech Pipeline v0 is complete: the existing production components passed real
ai-core E2E validation and human listening / subtitle synchronization acceptance.
`text → TTS → forced alignment → audio postprocess → caption compiler → WAV + SRT`
Completed:
- Caption Compiler v0
- Audio Postprocess v0
- Production TTS v0
- Production Alignment v0
PR #9 adds caller-approved performance units and per-segment instructions.
Real ai-core validation and human listening passed. Issue #8 remains open for
production-use validation of long-form narration stability.
Closure evidence: `docs/validation/speech-v0-closure.md` (PR #5).
Next: evaluate one real podcast production using existing tools first; see
`docs/workflows/podcast-pilot.md`. This evaluation does not authorize new features.
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
