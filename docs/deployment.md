# Client / Core installation and rollback

The v0.1.0 wheel is a lightweight Controller with no required third-party runtime
dependencies. TTS and alignment still execute in their **existing separate model
environments**. Workers load only the Controller's exact `media_pipeline` package
directory; its surrounding site-packages is not forwarded. Neither worker needs
another regrain installation. Use Python 3.10 or newer for the Controller and
workers; the accepted Core runtime uses Python 3.12.

Keep the development checkout, versioned Controller environments, model environments,
checkpoints, private voice assets and episode directories separate. For example:

```text
/srv/ai/src/regrain/                         development / experiment replay
/srv/ai/apps/regrain/controllers/           versioned Controller environments
  0.1.0-<source-sha>/                       candidate; keep accepted versions
  current -> 0.1.0-<source-sha>/             optional, only after acceptance
/srv/ai/apps/media-pipeline/tts/.venv/       existing accepted TTS environment
/srv/ai/apps/media-pipeline/aligner/.venv/   existing accepted alignment environment
/srv/ai/models/speech/                      existing checkpoints
/srv/ai/episodes/<episode>/                 speech.json, voices/, runs/
```

These are examples, not application defaults. Do not rename the accepted model
environment directories for cosmetic consistency. No SSH, transfer, package update,
checkpoint download or GPU process management is automated by regrain.

## Client installation and authoring

Obtain the reviewed wheel and matching release manifest. Check its SHA256 against
the manifest before installing. In PowerShell, from a directory outside the checkout:

```powershell
Get-FileHash ./regrain-0.1.0-py3-none-any.whl -Algorithm SHA256
py -3.12 -m venv ./regrain-0.1.0-client
./regrain-0.1.0-client/Scripts/python.exe -m pip install --no-index --no-deps ./regrain-0.1.0-py3-none-any.whl
./regrain-0.1.0-client/Scripts/regrain.exe --version
./regrain-0.1.0-client/Scripts/regrain.exe speech validate ./episode/speech.json
```

On Linux/macOS use `python3 -m venv` and the environment's `bin/python` and
`bin/regrain`. No development PYTHONPATH is needed. Use the [speech-authoring Skill](../skills/regrain-speech/SKILL.md)
to turn an approved manuscript into approved Performance Units. The installed wheel
also contains that folder at `media_pipeline/skills/regrain-speech`; find it with:

```console
python -c "from importlib.resources import files; print(files('media_pipeline').joinpath('skills/regrain-speech'))"
```

Run this using the installed Controller's Python. Copy the complete folder to the
Agent host's supported Skill discovery location or register/load its SKILL.md
directly. Host discovery paths differ. Confirm the host can read its examples;
do not assume copying a folder enables it in every host.

Review the validator's JSON plan and obtain human approval of the exact input.
Manually transfer speech.json and authorized clone assets while preserving their
relative directory relationship. Static validation does not check asset existence,
checkpoint compatibility or audible quality. Runtime/model paths belong in Core
local configuration, not in speech.json.

## Owner-only Core candidate installation

The owner performs these steps. They create a new Controller directory and do not
modify accepted runtimes or the current installation. Replace the example variables:

```sh
set -eu
artifact_dir=/path/to/reviewed/artifacts
controller_root=/srv/ai/apps/regrain/controllers
release_id=0.1.0-REPLACE_WITH_SOURCE_SHA
candidate="$controller_root/$release_id"
# Compare this output with the reviewed manifest's wheel.sha256.
sha256sum "$artifact_dir/regrain-0.1.0-py3-none-any.whl"
test ! -e "$candidate"
python3.12 -m venv "$candidate"
"$candidate/bin/python" -m pip install --no-index --no-deps "$artifact_dir/regrain-0.1.0-py3-none-any.whl"
unset PYTHONPATH PYTHONHOME
"$candidate/bin/regrain" --version
"$candidate/bin/regrain" speech validate /path/to/episode/speech.json
```

Keep the manifest alongside the versioned installation for source/hash provenance.
Use a shell where a failed command stops the procedure (`set -e`), or check each
command's result before continuing. Do not reuse an existing candidate directory.

Use the accepted runtime paths and existing checkpoints, for example:

```sh
export MEDIA_PIPELINE_TTS_PYTHON=/srv/ai/apps/media-pipeline/tts/.venv/bin/python
export MEDIA_PIPELINE_ALIGNMENT_PYTHON=/srv/ai/apps/media-pipeline/aligner/.venv/bin/python
export MEDIA_PIPELINE_ALIGNMENT_MODEL=/path/to/existing/Qwen3-ForcedAligner-0.6B
unset MEDIA_ALIGNMENT_PYTHON MEDIA_ALIGNMENT_MODEL
"$candidate/bin/regrain" speech render /path/to/clone-episode/speech.json \
  --tts-model /path/to/existing/Qwen3-TTS-12Hz-1.7B-Base \
  --output /path/to/clone-episode/runs/0.1.0-candidate-clone
"$candidate/bin/regrain" speech render /path/to/custom-voice-episode/speech.json \
  --tts-model /path/to/existing/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --output /path/to/custom-voice-episode/runs/0.1.0-candidate-custom
```

The unset aliases above avoid the established legacy alignment precedence;
explicit CLI paths also take precedence. Preserve the owner's desired local
configuration rather than changing global environment files.

**Required GPU gate:** both installed-CLI renders must run on ai-core with the
accepted separate interpreters and real assets/checkpoints. Record the wheel
SHA256, source commit, runtime/model/asset identities, command, exit code and
report. Check `complete`, every segment, one model load per stage, final WAV/SRT/
timeline, voice identity, transitions, and human listening/subtitle sync. Use
accepted short corpora; this gate does not solve Issue #8 or establish long-form
stability. CPU/injected-worker tests cannot satisfy it.

Only a complete render publishes `final/final.wav`, `final/final.srt` and
`final/timeline.json`; retain `report.json` and the release manifest. Failed or
incomplete runs are diagnostic artifacts. Manually transfer complete media back
to Client for revisual, FFmpeg or the chosen NLE. Those tools consume documented
artifacts and are not Controller dependencies or internal import consumers.
See the [render/artifact contract](dev/render-entry-point.md).

## Promote, upgrade and rollback

Prefer invoking the accepted directory explicitly. An optional Linux current
symlink can be switched after owner acceptance (and when no run is active):

```sh
ln -s "$candidate" "$controller_root/current.next"
mv -Tf "$controller_root/current.next" "$controller_root/current"
```

`current.next` must not already exist; `current` must be absent or a symlink, not
a real directory. Record the previous target before switching. This example uses
GNU/Linux tools; the full environment path is portable and sufficient elsewhere.

For an upgrade, create another version/source directory, install its wheel, and
repeat the static and GPU gates before switching. Keep the previous environment,
manifest and local configuration. Roll back by invoking the previous environment
directly or switching the symlink back with the same two commands and its previous
target. Rollback needs no PyTorch reinstall or model download. Existing outputs
and runtime environments remain usable. Never edit code inside accepted installations.
