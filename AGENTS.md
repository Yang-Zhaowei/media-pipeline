# regrain
Local-first media processing toolkit. Preserve the naming and compatibility
boundary in `docs/naming.md` and the speech contracts in `docs/dev/render-entry-point.md`.
## Current state
`docs/CURRENT.md` owns accepted capabilities, historical evidence and open problems.
PR #17 is in final closure for owner review; its original candidate has owner-reported
GPU/runtime and Client Agent acceptance. Clone consistency and Issue #8 remain open.
Never transfer acceptance to a rebuilt wheel: use `docs/release.md` and its exact-artifact gates.
The owner controls Ready status, merge, release, repository rename and deployment.
Do not add new modalities, NLE integration, general orchestration, or other scope unless explicitly requested.
## Repository evidence
- `experiments/`: previously working GPU smoke tests; reference only.
- `tests/fixtures/`: immutable real model outputs used for regression tests.
- `docs/dev/core-runtime.md`: currently verified ai-core GPU environment.
- `docs/validation/regrain-v0.1.0-owner-acceptance.md`: original candidate owner evidence and limits.
- `docs/deployment.md`, `docs/release.md`: installation/rollback and release procedures.
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
