# Media Pipeline

Local-first AI-assisted media processing pipeline for turning scripts and media assets into reusable, NLE-ready artifacts.

## Status

Early development.

**Speech Pipeline v0 is complete.** The production path is:

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

PR #5 validated fresh Production TTS output through every subsequent production
stage on `ai-core`. The final WAV and SRT passed automated checks and human
listening / synchronization acceptance. See the
[closure evidence](docs/validation/speech-v0-closure.md).

See [`docs/CURRENT.md`](docs/CURRENT.md) for the current project state.

## What is usable today

The Python production interfaces accept one already-segmented utterance with
`text`, `language`, a model-supported `speaker`, and an optional natural-language
`instruct` describing delivery. The stages produce a mono PCM16 WAV and matching
SRT. Only the recorded Chinese / `Uncle_Fu` case has full E2E acceptance; other
voices and scripts need their own listening check.

`render_speech(...)` renders a **pre-segmented** script (a fixed script with
fixed segments/speaker) to a single episode WAV + SRT in one synchronous
call; see [render entry-point docs](docs/dev/render-entry-point.md). It is not
a full-manuscript interface: there is no automatic long-script splitting,
no full-manuscript entry point, and no exact numeric speed / pitch control.
Voice cloning and VoiceDesign are not implemented. `python -m media_pipeline`
only compiles existing text and alignment into SRT.

[`validation/speech_pipeline_e2e.py`](validation/speech_pipeline_e2e.py) is a
repeatable validation driver with a fixed reviewed script and speaker, not an
arbitrary-script product interface. It uses the two existing GPU environments;
see [validation instructions](validation/README.md).

For a chapter-based podcast video, see the
[first-production workflow assessment](docs/workflows/podcast-pilot.md).

`render_speech(...)` is unmerged PR #7. Real ai-core multi-segment generation,
load-once instrumentation, and recorded human acceptance are complete. Final
acceptance at `f34f47a` still requires a narrow input-preflight correction;
see the [acceptance review](docs/validation/segmented-speech-v0-acceptance.md).
Identical voice/instruction settings do not guarantee identical emotion or
prosody across independently generated segments.

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

After closing Speech Pipeline v0, evaluate one real production, including a
revision and final export. Reuse existing authoring, browser rendering, FFmpeg
or NLE capabilities first. Add project code only for a reproducible gap that
passes the [feature admission rule](docs/ROADMAP.md).

HTTP, MCP, ASR, Vision, NLE automation, environment consolidation, and generic
orchestration are not committed next milestones.

## Repository Layout

```text
docs/
  CURRENT.md          Current milestone and project state
  validation/         Recorded milestone acceptance evidence
  workflows/          Real-production assessment and boundaries
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

validation/
  speech_pipeline_e2e.py  Two-environment real-runtime validation driver
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
