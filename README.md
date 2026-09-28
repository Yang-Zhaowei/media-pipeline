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

Completed production components:

- Caption Compiler v0
- Audio Postprocess v0
- Production TTS v0
- Production Alignment v0

Production TTS has been validated against the real Qwen3-TTS CustomVoice
runtime on the GPU integration host. Production Alignment v0 promotes the
previously verified Qwen3 ForcedAligner experiment into portable project code.

The current implementation task is to validate the production stages together
end to end: `text → TTS → forced alignment → audio post-processing → caption
compilation → WAV + SRT`.

See [`docs/CURRENT.md`](docs/CURRENT.md) for the current project state.

## Design Direction

The project separates:

- model inference from deterministic media processing;
- portable project logic from host-specific GPU runtimes;
- development and unit testing from GPU integration testing.

The repository currently includes portable deterministic processing and model-facing contracts, plus isolated host-specific runtime adapters for Qwen3-TTS and Qwen3 ForcedAligner under `media_pipeline.runtimes`.

```
Production TTS = production code
Production Alignment = production code
Forced Aligner runtime = ai-core GPU adapter, separate from pure processing
```

Pure processing components should run without CUDA and remain portable across Windows, Linux, and macOS.

## Roadmap

### Speech Pipeline v0

The current milestone is to productionize and validate the complete speech pipeline:

```text
text
→ TTS
→ forced alignment
→ audio post-processing
→ caption compilation
→ WAV + SRT
```

Current component status:

- Caption Compiler v0 — complete
- Audio Postprocess v0 — complete
- Production TTS v0 — complete
- Production Alignment v0 — complete

The remaining milestone work is to validate the production stages together as a real end-to-end pipeline and confirm that script input can reliably produce final WAV and SRT artifacts.

The exact orchestration interface is intentionally not fixed yet. HTTP APIs, MCP, job queues, and broader deployment architecture are outside the current milestone.

### Later directions

Potential later work includes:

- ASR workflow
- vision-based media analysis
- remote compute APIs
- agent / MCP integration
- NLE automation
- higher-level pipeline orchestration

## Repository Layout

```text
docs/
  CURRENT.md          Current milestone and project state
  dev/                Verified development/runtime environment notes
  legacy/             Historical architecture references

experiments/
  GPU smoke tests that have already been validated

src/
  media_pipeline/
    portable contracts and deterministic processing
    runtimes/          Model/runtime-specific adapters

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