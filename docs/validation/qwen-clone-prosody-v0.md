# Qwen3-TTS Base Clone Prosody Stability Experiment v0

Status: Layer 1 infrastructure accepted through owner-reported ai-core execution
and blind listening; [PR #16](https://github.com/Yang-Zhaowei/media-pipeline/pull/16)
merged on 2026-10-09T19:04:43Z at
`ff73167436937b3788c297b11d9c4b67235c2f5d`; [Issue #8](https://github.com/Yang-Zhaowei/media-pipeline/issues/8)
remains open. See the [acceptance evidence and qualified findings](qwen-clone-prosody-acceptance.md).
No definitive cause or remedy is established. Production speech behavior is unchanged.

Starting main: `238901170096cd14f5d6b950b4d84e4663603600`.
Branch: `experiment/issue8-qwen-clone-prosody-v0`.
GPU-tested implementation: `506b30dd59702348d6ff11bc7fad435acddf3a76`.
Later documentation commits are not themselves GPU-tested. The existing
procedure below checks an explicitly selected replay HEAD for a new run;
record that HEAD separately from the accepted implementation and starting main.

## Question and controls

Does audible delivery drift chiefly arise from stochastic sampling (H1),
independent target-generation context resets (H2), API-call topology (H3), or
behavior within one continuous generation (H4)? The
[generation-control audit](qwen-generation-controls.md) pins the supported
`qwen-tts 0.1.1` source and explains the public and forwarded arguments.
Only the existing `Qwen3-TTS-12Hz-1.7B-Base` and accepted reusable voice asset
are used. Sampling controls remain experiment-local.

Batch items have separate target sequences; one Python invocation is not
shared discourse context. Padding, EOS handling and RNG consumption can also
make batch and sequential outputs differ. Continuous target text is the
distinct L1 intervention. A reused clone prompt preserves reference-speaker
conditioning, not the previous target generation's state. The audited public
API provides no previous-text/audio/request or continuation-state mechanism.

Natural paragraph progression can legitimately change pitch, intensity and
timing. These measurements describe audio; they neither classify emotions nor
require identical segment averages. The owner must judge unexpected baseline
changes against natural discourse.

## Fixed corpus and matrix

The [versioned corpus](../../experiments/clone_prosody/corpus.json) preserves
the exact accepted PR #15 opening, discussion and closing wording and order.
S3 and B3 receive its three unchanged strings. L1 joins them with one literal
ASCII space; punctuation is retained, with no headings, spoken markers or
rewriting. Manifest text and hashes make that equivalence inspectable.
Half-open Unicode-codepoint ranges locate logical sections in the joined text;
they are not audio timestamps. L1 has no automatically recovered internal
acoustic boundaries or per-section measurements.

| Condition | IDs | Target generation calls per run | Overrides |
| --- | --- | ---: | --- |
| S3-default | `S3-D-1`, `S3-D-2`, `S3-D-3` | 3 separate calls | None |
| B3-default | `B3-D-1`, `B3-D-2`, `B3-D-3` | 1 true three-item batch call | None |
| L1-default | `L1-D-1`, `L1-D-2`, `L1-D-3` | 1 continuous-text call | None |
| S3-greedy | `S3-G-1`, `S3-G-2` | 3 separate calls | `do_sample=False`, `subtalker_dosample=False` |
| L1-greedy | `L1-G-1`, `L1-G-2` | 1 continuous-text call | Same two switches |

There are exactly 13 runs; no two-unit or cross-model condition is included.
Default means no generation-kwargs override; actual effective configuration
and runtime provenance are recorded. Both sampling switches are disabled in
greedy diagnostics according to the audited source. This is not a promise of
bit-identical replay on a GPU.

Each run uses a fresh TTS process, loads the asset once and the Base engine
once, then performs its condition's calls through that engine. This avoids
model reloads between S3 units and carries no target state between repeats.
After asset loading and before model loading, Python, NumPy and torch
process-global RNGs use `1729`, `1730`, `1731` for
default repetitions 1, 2, 3; greedy repetitions use `1729`, `1730`, paired to
the first two default repeats. This holds seed selection constant when
comparing sampling modes; sampling-off seed sensitivity is measured rather
than assumed absent. No unsupported
Qwen `seed` or `torch.Generator` argument is passed. The schedule is shuffled
with seed `8108` and recorded. Seeds control one source of variation; execution
environment and GPU nondeterminism remain possible influences.

## Outputs, safety and failures

The output root must not exist, including an empty directory. Partial runs
cannot be resumed or silently reused: preserve evidence and choose a fresh
root. A run manifest separates portable authored corpus/conditions from
caller-provided runtime paths. It records exact repository HEAD/clean state,
versions, device, controls, seed, schedule, source/config/voice hashes,
lifecycle counts, inputs, output metadata and status without exposing prompt
tensors or the reference transcript.

Each run directory holds its report, raw PCM16 output(s), a full-content
`listening.wav`, and a completion marker only after success. The root holds
`manifest.json`, `corpus.json`, `analysis.json`, `analysis.md`, blinded copies,
the separate root-level `listening-mapping.json`, and a review template. These are runtime artifacts;
never commit them as fixtures.

Independent conditions continue after a synthesis failure. Reports identify
the run/condition, repetition, stage, original reason and partial artifacts.
The experiment is incomplete and has no root completion marker if any selected
run fails or derived analysis/blinding fails. Partial data are labeled; they
must not be interpreted as a complete matrix. The real runner requires a clean
checkout at the caller's exact expected HEAD and rejects unsupported runtime
or checkpoint metadata.

## Acoustic descriptors

The automatic analyzer uses standard-library PCM16 analysis, with no torch,
Qwen, CUDA, pitch model or new dependency. It reports per-unit measurements
for S3/B3, and full-text measurements for L1:

- sample rate, mono frames and duration (`frames / sample_rate`);
- non-whitespace Unicode text characters per generated second, including punctuation;
- normalized RMS, peak absolute amplitude and crest factor (`peak / RMS`);
- activity/silence and boundary descriptors defined below;
- same-content dispersion across repeats, retaining the individual values.

These are raw generation measurements. They are not directly comparable to
the accepted PR #15 cleaned/assembled audio measurements. Rate includes
punctuation and generated silence; it is not a linguistic syllable rate.
Signed PCM16 samples are divided by `32768`; RMS and peak use all channel
samples. Silent PCM has a null crest ratio. Different sample rates are analyzed
in seconds without resampling; incompatible assembly inputs fail rather than
being silently converted. Malformed or empty WAVs are rejected.

Activity uses nonoverlapping bins of `max(1, round(sample_rate * 0.010))`
frames, including the final partial bin. A bin is active when its all-channel
RMS is at least `0.01`. Active duration and fraction use actual bin frame
counts. Leading/trailing inactive-bin durations are measured before the first
and after the last active bin; both equal the full duration for inactive audio.
This energy threshold is neither a speech nor a voiced/unvoiced detector.

Segmented boundary descriptors compare up to
`max(1, round(sample_rate * 0.250))` frames inside the first-to-last active-bin interval
at the previous unit's end and the next unit's start. They expose both signed
and absolute RMS changes. Raw-concatenation pause duration is the previous
trailing plus next leading inactive duration. There is no inserted listening
gap, trim, fade, normalization or alignment; this deliberately differs from
the accepted production render and does not isolate how its postprocessing
affects perceived transitions. An edge with no activity yields null edge
metrics and a null inferred pause. L1 receives no invented equivalent boundary comparison. No threshold
labels an energy change bad.

For each condition and unchanged logical unit (or L1 full content), repeat
aggregation reports `n`, mean, sample standard deviation (null for `n < 2`),
minimum, maximum, range, individual values and every pairwise absolute delta
for duration, chars/second, RMS, peak and crest. Rate and frame metadata remain
visible. Failed or unanalyzable runs are excluded as whole runs and identified
in the report. Definitions are also recorded in `analysis.json`; there is no
single stability score.

Automatic F0 is omitted: the existing dependency set has no established speech
pitch tracker, and a simple autocorrelation method validated only on sine
waves would not establish reliable natural-speech voicing or octave handling.
An owner may add separate pitch analysis later, recording the method, valid
range, voiced/unvoiced policy and uncertainty. F0 is not an emotion score.

## Listening and interpretation

Listen to `listening-blind/sample-01.wav`, etc., using the generated review
sheet. Keep root-level `listening-mapping.json` closed until every sample has a
review; it necessarily reveals the assignment and is separate from the audio
names. Original named outputs remain intact. Blinding randomizes copies with
the recorded seed and re-encodes plain PCM WAV headers, retaining audio while
removing ancillary metadata. Audible pauses can still reveal segmentation, so this is
not guaranteed condition concealment. Do not rerun blinding into an existing
blind package; stale copies are rejected.

Review voice identity, delivery continuity, unexpected emotional jumps,
natural paragraph progression, pronunciation, artifacts and overall notes.
Then reveal the mapping and inspect acoustic descriptors alongside listening:

- Large same-unit repeat variance in S3-default, reduced in S3-greedy, supports
  sampling as a contributor (H1).
- Similar S3/B3 continuity with better L1 continuity supports continuous target
  context over invocation count as a candidate (H2 over H3).
- A repeatable S3-greedy baseline reset can leave H2 plausible even when H1 is
  reduced. A reproducible S3/B3 difference can make H3 relevant, while padding
  and RNG differences prevent attributing it only to call count.
- Undesirable changes within L1 leave H4 plausible only after considering the
  simpler explanations. This short corpus cannot establish episode-length
  stability, and an L1 result alone does not prove intrinsic model behavior.

Two or three repetitions yield descriptive evidence, not statistical
significance or an objective winner. The report does not announce a root
cause, rank conditions, or solve Issue #8.

## Exact ai-core owner procedure

The procedure below is preserved from the original pre-merge review. For a
post-merge replay of accepted evidence, select GPU-tested implementation
`506b30dd59702348d6ff11bc7fad435acddf3a76` and fetch the persistent PR reference
with `git fetch origin refs/pull/16/head` instead of depending on the old feature
branch. Detach at the selected implementation commit, record it, and continue
the runtime/input checks below. The final PR HEAD and merge commit are different
from the GPU-tested implementation. Experiments require that source checkout;
they are not included in the production Controller wheel.

Use the existing ai-core RTX 4070 12GB TTS environment, local Base checkpoint
and accepted `voice.pt`. Manually check and free GPU capacity before synthesis.
The commands do not inspect/decide GPU availability, alter services, stop
llama, kill processes, install packages, download checkpoints or connect over
SSH. Select runtime paths yourself; all scratch artifacts belong beneath
`/srv/ai/experiments/`.

In the existing media-pipeline checkout, replace the three caller selections
and the expected HEAD with the exact full commit shown by this Draft PR:

```bash
set -euo pipefail
EXPECTED_HEAD='<exact current Draft PR HEAD: 40 hexadecimal characters>'
TTS_PYTHON='<absolute path to existing TTS environment python>'
MODEL='<absolute path to existing Qwen3-TTS-12Hz-1.7B-Base directory>'
VOICE='<absolute path to existing accepted voice.pt>'
export EXPECTED_HEAD TTS_PYTHON MODEL VOICE

test -z "$(git status --porcelain)"
git fetch origin experiment/issue8-qwen-clone-prosody-v0
test "$(git rev-parse FETCH_HEAD)" = "$EXPECTED_HEAD"
git switch --detach "$EXPECTED_HEAD"
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git status --porcelain)"
test -x "$TTS_PYTHON"
test -d "$MODEL"
test -f "$MODEL/config.json"
test -r "$VOICE"
export PYTHONPATH="$PWD/src:$PWD"

# Package/config/asset verification does not load a GPU model or print tensors.
"$TTS_PYTHON" - <<'PY'
import hashlib, importlib.metadata, json, os
from pathlib import Path
from media_pipeline.runtimes.qwen_voice_clone import load_voice_asset

qwen = importlib.metadata.version("qwen-tts")
assert qwen == "0.1.1", f"unsupported qwen-tts: {qwen}"
distribution = importlib.metadata.distribution("qwen-tts")
audited_sources = {
    "qwen_tts/inference/qwen3_tts_model.py": "e1da450732857c1f5fe3e36ebab85db2f6dc6a48caaff6973f463384e30275e4",
    "qwen_tts/core/models/modeling_qwen3_tts.py": "25c42656bcf810f06ef6bc1839bd7083f3c8cfedac3a147c4060b4262b1c96a0",
}
source_hashes = {name: hashlib.sha256(distribution.locate_file(name).read_bytes()).hexdigest()
                 for name in audited_sources}
assert source_hashes == audited_sources, "installed Qwen source differs from the audited release; inspect before running"
model = Path(os.environ["MODEL"])
voice = Path(os.environ["VOICE"])
config = json.loads((model / "config.json").read_text(encoding="utf-8"))
assert config["tts_model_type"] == "base"
assert config["tts_model_size"] == "1b7"
assert config["tokenizer_type"] == "qwen3_tts_tokenizer_12hz"
asset = load_voice_asset(voice)
assert len(asset.payload["items"]) == 1
assert asset.payload["model"] == {
    "tokenizer_type": "qwen3_tts_tokenizer_12hz", "tts_model_size": "1b7"
}
print(json.dumps({
    "qwen_tts": qwen, "torch": importlib.metadata.version("torch"),
    "qwen_source_sha256": source_hashes,
    "model": str(model), "voice": str(voice),
    "voice_sha256": hashlib.sha256(voice.read_bytes()).hexdigest(),
    "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
    "asset_model": asset.payload["model"],
}, indent=2))
PY

# The runner itself creates RUN exclusively; do not mkdir RUN in advance.
mkdir -p /srv/ai/experiments
RUN="/srv/ai/experiments/qwen-clone-prosody-v0-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$RUN"
export RUN
"$TTS_PYTHON" -m experiments.clone_prosody plan
if "$TTS_PYTHON" -m experiments.clone_prosody run \
  --output "$RUN" --model "$MODEL" --voice "$VOICE" \
  --device cuda:0 --tts-python "$TTS_PYTHON" --expected-head "$EXPECTED_HEAD"; then
  printf 'Synthesis and derived reports completed: %s\n' "$RUN"
else
  printf 'Incomplete experiment; inspect manifest and run reports: %s\n' "$RUN" >&2
  if test -f "$RUN/manifest.json"; then
    "$TTS_PYTHON" -m json.tool "$RUN/manifest.json"
  fi
  exit 1
fi

# Successful run includes acoustic analysis and seeded blinded listening copies.
test -f "$RUN/.complete"
test -f "$RUN/analysis.json"
test -f "$RUN/analysis.md"
test -f "$RUN/listening-review-template.md"
printf 'Experiment artifacts: %s\n' "$RUN"
printf '%s\n' \
  '1. Confirm complete manifest, 13 successful runs, exact HEAD/versions/paths.' \
  '2. Review each blind sample: identity, continuity, unexpected emotional jump.' \
  '3. Note natural paragraph progression, pronunciation and audio artifacts.' \
  '4. Finish notes before revealing listening-mapping.json.' \
  '5. Compare same-content repeat dispersion, then S3/B3/L1 and default/greedy.' \
  '6. Report observations and limitations; do not declare an automatic winner.'
```

If synthesis returns nonzero, inspect the manifest and run reports before
anything else; the absent `.complete` is intentional. Keep the failed root and
use a new one for any repeat. `--conditions` can select explicit condition
names for a smaller diagnostic attempt; a subset is not full-matrix evidence.
To inspect a failed root later, set `RUN` to that preserved path and run:

```bash
if test -f "$RUN/manifest.json"; then
  "$TTS_PYTHON" -m json.tool "$RUN/manifest.json"
fi
if test -f "$RUN/analysis.md"; then
  cat "$RUN/analysis.md"
fi
```

Reports preserve the original condition failure and partial artifact paths;
an early preflight failure may leave no output root or derived analysis.

Derived acoustic analysis can be regenerated without the TTS environment:

```bash
PYTHONPATH="$PWD/src:$PWD" python -m experiments.clone_prosody analyze --output "$RUN"
```

For an existing successful audio experiment that has no blind package yet:

```bash
PYTHONPATH="$PWD/src:$PWD" python -m experiments.clone_prosody blind --output "$RUN" --seed 8108
```

No caption alignment, postprocessing, production-schema changes or new model
provider are part of this experiment. CPU/fake tests validate orchestration,
files, provenance, analysis and failure behavior; they provide no model-quality
or GPU evidence. The owner completed the exact-head real run and blind listening
at the implementation HEAD recorded above. The [acceptance document](qwen-clone-prosody-acceptance.md)
records those owner-reported results, the final PASS for both greedy listening
PCM repeat comparisons in the fixed environment, and the remaining production
length question. This documentation closure did not repeat the run or listening.

## CPU validation executed

Windows / Python 3.12.14, at this implementation's working tree:

```text
.venv/Scripts/python.exe -m pytest tests/test_clone_prosody_runner.py tests/test_clone_prosody_runtime.py -o addopts=-ra -q
42 passed in 1.84s

.venv/Scripts/python.exe -m pytest tests/test_clone_prosody_analysis.py -o addopts=-ra -q
40 passed in 0.67s

.venv/Scripts/python.exe -m pytest tests/test_voice_clone.py tests/test_qwen_voice_clone.py tests/test_voice_clone_serialization.py tests/test_render_clone.py tests/test_render.py tests/test_cli.py -o addopts=-ra -q
324 passed, 4 skipped in 13.90s

.venv/Scripts/python.exe -m pytest -o addopts=-ra -q
629 passed, 10 skipped in 16.12s

git diff --check
passed
```

The four focused skips are three real-torch serialization checks (torch absent)
and one Windows symlink privilege check. The full suite additionally skips six
unconfigured GPU integration tests. Experiment tests include matrix/lifecycle,
fake full-path analysis/blinding, stale-root refusal, partial failures and marker
invalidation. These original Windows CPU checks did not include real ai-core
GPU A/B execution or human listening. The subsequent [owner-reported acceptance](qwen-clone-prosody-acceptance.md)
is separate evidence; no definitive root cause is claimed.
