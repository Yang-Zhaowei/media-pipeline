# Media Pipeline

Local-first AI-assisted media processing pipeline for turning scripts and media assets into reusable, NLE-ready artifacts.

## Status

Early development.

The current milestone is **Speech Pipeline v0**:

```text
text
→ TTS
→ forced alignment
→ audio post-processing
→ caption compilation
→ WAV + SRT
```

Qwen3-TTS and Qwen3 Forced Aligner have already been validated on the GPU integration host. The repository currently contains the resulting real-world fixture for regression development.

Caption Compiler v0 and Audio Postprocess v0 are implemented and covered by deterministic regression tests.

The immediate implementation task is **Production TTS module v0**.

See [`docs/CURRENT.md`](docs/CURRENT.md) for the current project state.

## Design Direction

The project separates:

- model inference from deterministic media processing;
- portable project logic from host-specific GPU runtimes;
- development and unit testing from GPU integration testing.

The repository currently includes deterministic caption compilation and audio post-processing, plus verified GPU experiments for Qwen3-TTS and Qwen3 Forced Aligner.

Pure processing components should run without CUDA and remain portable across Windows, Linux, and macOS.

Longer term, the project may cover:

- TTS
- ASR
- forced alignment
- audio post-processing
- caption generation
- vision-based media analysis
- remote compute APIs
- agent/MCP integration
- NLE-oriented artifacts

These are not all implemented yet.

## Repository Layout

```text
docs/
  CURRENT.md          Current milestone and project state
  dev/                Verified development/runtime environment notes
  legacy/             Historical architecture references

experiments/
  GPU smoke tests that have already been validated

src/
  media_pipeline/      Portable deterministic processing code

tests/
  fixtures/           Real model outputs used as regression evidence
```

## Development Model

Development is performed on the client workstation.

Most logic is developed and tested locally without GPU dependencies. GPU-specific integration is validated separately on `ai-core`.

Real regression fixtures are kept in the repository so deterministic components can be developed without repeatedly invoking AI models.

## Agent Development

Repository-wide instructions for coding agents are in [`AGENTS.md`](AGENTS.md).

Agents should also read [`docs/CURRENT.md`](docs/CURRENT.md) before continuing active development.

## License

No license has been selected yet.