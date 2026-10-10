# regrain

Local-first, agent-friendly tools for turning caller-authored scripts into
speech and subtitles. The v0.1.0 release candidate packages a lightweight
Controller; PyTorch, `qwen-tts`, and `qwen-asr` stay in their existing runtime
environments.

## Install and use

Install a verified local wheel once on Client with `uv tool`, using uv-managed
Python; see [installation and PATH discovery](docs/deployment.md#client-installation-and-authoring).
Then use the shared CLI from any content project:

```console
regrain --help
regrain --version
regrain speech validate speech.json
regrain speech render speech.json --output new-run-dir
```

`validate` checks the script without models or input changes. `render` uses Core's
existing runtimes. The [CLI/input contract](docs/dev/render-entry-point.md) defines
runtime settings, outputs and errors; [naming](docs/naming.md) defines retained
`media_pipeline` imports, the `media-pipeline` alias and configuration compatibility.

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
Agents author inputs and run the official validator. Use one shared Skill copy;
host discovery is configured as described in the deployment guide.

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

The [current state and evidence](docs/CURRENT.md) distinguish accepted speech
capabilities, original candidate owner GPU/Client acceptance, and the remaining
Issue #8 Clone/long-form consistency question. Rebuilt wheels need their own
exact-artifact acceptance. CPU checks run separately from real GPU verification.

## Development

Contributor and Agent guidance is in [AGENTS.md](AGENTS.md). For development,
install an editable checkout with `python -m pip install -e .` or `uv sync`.

## License

No license has been selected yet.
