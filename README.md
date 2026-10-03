# Media Pipeline

Local-first tools for turning caller-authored scripts into speech and subtitles.

## Speech pipeline

```text
text → TTS → forced alignment → audio postprocess → captions → WAV + SRT
```

The production path uses four components: TTS, forced alignment, audio
postprocessing, and caption compilation. Real ai-core validation and human
listening passed in [PR #9](https://github.com/Yang-Zhaowei/media-pipeline/pull/9).

Install the command from a checkout with `python -m pip install -e .` (or
`uv sync`), then run a static preflight or render a new run directory:

```console
media-pipeline speech validate speech.json
media-pipeline speech render speech.json --output new-run-dir
```

Use an activated virtual environment; with `uv sync`, prefix the commands with
`uv run`.

The CLI prints a UTF-8 JSON result for inspection and automation. See the
[CLI contract](docs/dev/render-entry-point.md#command-line) for its output,
exit codes, and runtime options.

## Authoring speech

The Python interface `render_speech(...)` accepts a script already divided
into caller-approved performance units. A unit may contain multiple sentences
and is sent as one TTS request. The caller chooses unit boundaries and any
directing instructions; the pipeline does not segment or interpret the script.

Ordinary narration uses an empty instruction by default. A segment can inherit
the top-level instruction, explicitly clear it, or provide its own direction.
Subtitle boundaries are determined independently by alignment and caption
compilation.

See the [render entry point](docs/dev/render-entry-point.md) and
[performance-unit contract](docs/contracts/performance-unit-instruct.md).

## Project status

The agent-facing speech CLI is complete in
[PR #11](https://github.com/Yang-Zhaowei/media-pipeline/pull/11);
[Issue #10](https://github.com/Yang-Zhaowei/media-pipeline/issues/10) is closed.

[Issue #8](https://github.com/Yang-Zhaowei/media-pipeline/issues/8) remains open
to validate long-form narration stability in production.
See the [current state](docs/CURRENT.md) and [podcast pilot workflow](docs/workflows/podcast-pilot.md).

Project scope and future proposals are described in the [roadmap](docs/ROADMAP.md).
Runtime setup is documented in [core runtime notes](docs/dev/core-runtime.md).

Pure processing and unit tests run without CUDA. Real model integration is
validated separately on ai-core; see [validation instructions](validation/README.md).

## Development

Repository guidance for contributors and agents is in [AGENTS.md](AGENTS.md).

## License

No license has been selected yet.
