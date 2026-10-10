# regrain
Local-first AI-assisted media processing toolkit. The distribution and primary
CLI are `regrain`; the supported Python import namespace stays `media_pipeline`.
The `media-pipeline` CLI is a thin compatibility alias. Existing
`MEDIA_PIPELINE_*` settings remain stable.
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
The agent-facing speech CLI is complete (PR #11; Issue #10 closed).
Its validation/render contract is in `docs/dev/render-entry-point.md`.
Reusable Qwen3-TTS Base voice-clone runtime v0 is complete (PR #13; Issue #12 closed).
Owner-reported real ai-core and human acceptance: `docs/validation/voice-clone-closure.md`.
Segmented clone rendering is complete (PR #15; Issue #14 closed), with real
ai-core and human acceptance: `docs/validation/clone-render-acceptance.md`.
Cross-segment emotion/delivery drift remains tracked in Issue #8.
PR #16 merged on 2026-10-09T19:04:43Z at
`ff73167436937b3788c297b11d9c4b67235c2f5d`. It adds Qwen Base clone prosody
diagnostics; the owner reported 13 successful real GPU runs and blind-listening
evidence. The exact GPU-tested implementation remains
`506b30dd59702348d6ff11bc7fad435acddf3a76`; this historical evidence is not a
claim that the current release candidate was GPU tested.
Closure evidence: `docs/validation/speech-v0-closure.md` (PR #5).
Current objective: prepare the regrain v0.1.0 release candidate. Packaging and
CPU validation are distinct from owner-side ai-core GPU acceptance. See
`docs/release.md` and `docs/deployment.md`; Issue #8 remains open and is not
part of this release-preparation scope. The podcast pilot remains useful for
evaluating existing tools: `docs/workflows/podcast-pilot.md`.
Do not add new modalities, NLE integration, general orchestration, or other scope unless explicitly requested.
## Repository evidence
- `experiments/`: previously working GPU smoke tests; reference only.
- `tests/fixtures/`: immutable real model outputs used for regression tests.
- `docs/dev/core-runtime.md`: currently verified ai-core GPU environment.
- `docs/naming.md`, `docs/deployment.md`, `docs/release.md`: current release decisions and owner procedures.
- `skills/regrain-speech/`: reusable Agent Skill for authoring and validating speech inputs.
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
