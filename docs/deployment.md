# Client / Core installation and rollback

Use Python 3.10+ (accepted Core runtime: 3.12). Install the Controller separately
from the existing TTS and Aligner environments; neither worker needs another
regrain installation. See the [runtime/CLI contract](dev/render-entry-point.md)
for package loading and [release gates](release.md) for exact-artifact acceptance.

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

Use one shared CLI installed from the verified local wheel via
[uv tools](https://docs.astral.sh/uv/concepts/tools/), with
[uv-managed Python](https://docs.astral.sh/uv/guides/install-python/). No global
Python or per-Agent/per-content-project regrain installation is needed. Compare
the hash with the matching manifest first; in PowerShell outside the checkout:

```powershell
Get-FileHash ./regrain-0.1.0-py3-none-any.whl -Algorithm SHA256
uv python install 3.12
uv tool install --managed-python --python 3.12 --no-index ./regrain-0.1.0-py3-none-any.whl
regrain --version
regrain speech validate ./episode/speech.json
```

An Agent subprocess may retain an old PATH after installation. Restart the host
process, or invoke the absolute executable path without reinstalling:

```powershell
$regrainExe = Join-Path (uv tool dir --bin) 'regrain.exe'
& $regrainExe --version
& $regrainExe speech validate ./episode/speech.json
```

Keep one reusable [regrain-speech folder](../skills/regrain-speech/SKILL.md),
including examples, in a compatible Agent Skills discovery location. Codex and Pi
can consume that same copy when configured to discover it; otherwise register its
path with each host. The owner's Client uses `C:\Users\zhaowei\.agents\skills\regrain-speech`.
Other hosts may use different paths. The wheel also ships this folder under
`media_pipeline/skills/regrain-speech`. Confirm host discovery and validator access;
the [reported Client acceptance](validation/regrain-v0.1.0-owner-acceptance.md)
does not prove all hosts' discovery mechanisms.

For a Client update, verify the new wheel/manifest, then use the same local-wheel
install command with `--reinstall` (also for same-version candidate changes).
Retain the previous verified wheel/manifest for rollback by reinstalling that
wheel. Refresh the shared Skill from the selected version and restart hosts as
needed. Do not edit uv's tool environment with pip. No development PYTHONPATH is needed.

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
Stop on command failure and do not reuse an existing candidate directory.

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

**Required GPU gate for the selected artifact:** both installed-CLI renders run
on ai-core with the accepted separate interpreters and real assets/checkpoints.
The [original owner acceptance](validation/regrain-v0.1.0-owner-acceptance.md)
does not transfer to a rebuilt wheel. Record the wheel
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
