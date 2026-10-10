# regrain

Local-first, agent-friendly tools for turning caller-authored scripts into
speech and subtitles. The v0.1.0 release candidate packages a lightweight
Controller; PyTorch, `qwen-tts`, and `qwen-asr` stay in their existing runtime
environments.

## Install and use

Install the built wheel into a clean Controller environment from the release
artifact. This does not require PyPI or a source checkout:

```console
python -m pip install --no-index --no-deps ./regrain-0.1.0-py3-none-any.whl
regrain --help
regrain --version
regrain speech validate speech.json
regrain speech render speech.json --output new-run-dir
```

For development, install an editable checkout with `python -m pip install -e .`.
The `media-pipeline` command remains as a thin compatibility alias; the Python
import namespace remains `media_pipeline`. Existing `MEDIA_PIPELINE_*` runtime
configuration and its precedence are unchanged. See the [naming decision](docs/naming.md).

The `validate` command performs static validation without loading models or
changing the input. Rendering coordinates separate TTS and Alignment Python
interpreters, then runs deterministic postprocessing, caption compilation and
publication in the Controller. Install the Controller separately from both
GPU runtime environments. The [CLI contract](docs/dev/render-entry-point.md)
documents results, exit codes, runtime settings and child interpreter setup.

## Speech workflow

```text
speech.json → validation → TTS → forced alignment → postprocess → captions
            → WAV + SRT + timeline + report
```

On PC_Client, an Agent prepares `speech.json`, runs `regrain speech validate`,
and hands the validated plan to a human for approval. The approved input and
required voice assets are transferred manually to ai-core for
`regrain speech render`; the resulting artifacts are transferred back manually
for revisual, FFmpeg or NLE finishing. regrain does not depend on revisual.

The reusable [speech-authoring Skill](skills/regrain-speech/SKILL.md) helps
Agents author inputs and run the official validator. To use it on different
Agent hosts, copy the Skill folder to that host's configured Skills location or
load `SKILL.md` directly where supported. Discovery paths vary by host; regrain
does not install it globally.

## Release and deployment

- [Client/Core deployment, verification, upgrade and rollback](docs/deployment.md)
- [v0.1.0 release preparation and owner acceptance gates](docs/release.md)
- [Project and package naming decisions](docs/naming.md)
- [Current state](docs/CURRENT.md) and [roadmap](docs/ROADMAP.md)

The repository is hosted at
[Yang-Zhaowei/media-pipeline](https://github.com/Yang-Zhaowei/media-pipeline)
while the owner prepares a repository rename to `regrain`. Existing PR and
commit identities remain historical references.

## Existing production evidence

Speech Pipeline v0 has passed real ai-core rendering and human listening /
subtitle synchronization acceptance. Production TTS, alignment, postprocess,
caption compilation, clone rendering and prosody diagnostics are documented in
the [current evidence index](docs/CURRENT.md). Issue #8 remains open for
production-use validation of long-form narration stability; the release
candidate does not claim to resolve it.

Pure processing and unit tests run without CUDA. GPU integration is validated
separately on ai-core; see [validation instructions](validation/README.md).

## Development

Contributor and Agent guidance is in [AGENTS.md](AGENTS.md). The distribution is
named `regrain`; the supported Python import namespace remains `media_pipeline`
for compatibility with the accepted APIs.

## License

No license has been selected yet.
