# Issue #14 owner acceptance: cloned segmented speech

Status: **pending**. These commands are an owner procedure, not executed
validation evidence. Keep the PR Draft and the merge gate open until the owner
records the real ai-core run and listening results below.

The gate uses the already accepted reusable `voice.pt`, the existing
`Qwen3-TTS-12Hz-1.7B-Base` checkpoint, and the existing separate Alignment
environment/checkpoint. Do not download a checkpoint, change an environment,
or commit assets as fixtures to perform this acceptance.

## Prepare the exact checkout and caller-owned input

Run in Bash on ai-core. Replace every placeholder with an owner-selected path
or value. Choose new script, evidence, and render directories outside the
checkout. `OWNER_PR_HEAD` must be the full current Draft PR commit, not `main`.

```bash
set -euo pipefail
export CHECKOUT='<exact PR checkout>'
export OWNER_PR_HEAD='<full exact PR HEAD>'
export CLI_PYTHON='<existing parent/CLI Python executable>'
export TTS_PYTHON='<existing TTS Python executable>'
export ALIGNMENT_PYTHON='<existing Alignment Python executable>'
export BASE_MODEL='<existing Qwen3-TTS-12Hz-1.7B-Base directory>'
export ALIGNMENT_MODEL='<existing Forced Alignment checkpoint directory>'
export ACCEPTED_VOICE='<already accepted reusable voice.pt>'
export SCRIPT_PROJECT='<new caller-owned script project directory>'
export EVIDENCE_DIR='<new caller-owned evidence directory>'
export RUN_DIR='<new caller-owned render output directory>'
export DEVICE='cuda:0'

cd "$CHECKOUT"
test "$(git rev-parse HEAD)" = "$OWNER_PR_HEAD"
test -z "$(git status --porcelain)"
test -f "$ACCEPTED_VOICE"
test -d "$BASE_MODEL"
test -d "$ALIGNMENT_MODEL"
test ! -e "$RUN_DIR"
mkdir "$SCRIPT_PROJECT" "$EVIDENCE_DIR"
mkdir "$SCRIPT_PROJECT/voices"
cp "$ACCEPTED_VOICE" "$SCRIPT_PROJECT/voices/zhaowei-v1.pt"
export PYTHONPATH="$CHECKOUT/src${PYTHONPATH:+:$PYTHONPATH}"
git rev-parse HEAD > "$EVIDENCE_DIR/pr-head.txt"
git status --porcelain > "$EVIDENCE_DIR/checkout-status.txt"

cat > "$SCRIPT_PROJECT/speech.json" <<'JSON'
{
  "language": "Chinese",
  "voice": {"type": "clone", "asset": "voices/zhaowei-v1.pt"},
  "segments": [
    {"id": "opening", "text": "欢迎收听今天的节目。我们先从一个具体的问题开始。", "pause_after_ms": 180},
    {"id": "discussion", "text": "面对复杂的信息，我们需要认真核对事实，再作出自己的判断。", "pause_after_ms": 220},
    {"id": "closing", "text": "感谢你的收听。下一期节目，我们继续讨论这个话题。", "instruct": ""}
  ]
}
JSON

"$TTS_PYTHON" - <<'PY' > "$EVIDENCE_DIR/runtime.json"
import importlib.metadata, json, os, torch
print(json.dumps({
    "torch": torch.__version__,
    "qwen_tts": importlib.metadata.version("qwen-tts"),
    "gpu": torch.cuda.get_device_name(torch.device(os.environ["DEVICE"])),
    "device": os.environ["DEVICE"],
}, indent=2))
PY
```

The reference recording/transcript are not inputs to this run. The authored
asset stays relative to `speech.json`; the physical copy avoids a symlink
escaping the script directory. Clone instructions must be omitted or `""` at
both top level and segment level.

## Observe genuine TTS lifecycle calls

This temporary owner-only observer lives in the evidence directory. It wraps
the real production asset loader, Base engine constructor, and synthesis
method, records successful calls, and returns their original results. It does
not use fakes or change production files. The constructor contains one real
`Qwen3TTSModel.from_pretrained` call. Transient object IDs and the process ID
show that every synthesis reused that engine and loaded asset. The JSONL log
is outside the published request/report; it contains no tensor payload or
reference transcript.

```bash
export OWNER_TTS_EVENTS="$EVIDENCE_DIR/tts-events.jsonl"
export OWNER_TTS_OBSERVER="$EVIDENCE_DIR/observe-tts.py"
test ! -e "$OWNER_TTS_EVENTS"
cat > "$OWNER_TTS_OBSERVER" <<'PY'
import json, os, sys
from pathlib import Path
from media_pipeline.runtimes import qwen_voice_clone as runtime

if len(sys.argv) != 4 or sys.argv[1] != "-c":
    raise SystemExit("observer expects the production -c stage and task file")
stage, task_file = sys.argv[2:]

def record(kind, **fields):
    with Path(os.environ["OWNER_TTS_EVENTS"]).open("a", encoding="utf-8") as log:
        log.write(json.dumps({"kind": kind, "pid": os.getpid(), **fields}) + "\n")

original_load = runtime.load_voice_asset
def load_voice_asset(*args, **kwargs):
    asset = original_load(*args, **kwargs)
    record("asset_load", asset_id=id(asset))
    return asset
runtime.load_voice_asset = load_voice_asset

original_init = runtime.Qwen3VoiceCloneTTS.__init__
def initialize(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    record("base_load", engine_id=id(self), model_id=id(self._model))
runtime.Qwen3VoiceCloneTTS.__init__ = initialize

original_synthesize = runtime.Qwen3VoiceCloneTTS.synthesize
def synthesize(self, request, asset, output_wav_path):
    result = original_synthesize(self, request, asset, output_wav_path)
    record("synthesis", engine_id=id(self), model_id=id(self._model),
           asset_id=id(asset), wav=Path(output_wav_path).name)
    return result
runtime.Qwen3VoiceCloneTTS.synthesize = synthesize

sys.argv = ["-c", task_file]
exec(compile(stage, "<string>", "exec"), {"__name__": "__main__"})
PY

cat > "$EVIDENCE_DIR/tts-python-observed" <<'SH'
#!/bin/sh
exec "$TTS_PYTHON" "$OWNER_TTS_OBSERVER" "$@"
SH
chmod +x "$EVIDENCE_DIR/tts-python-observed"
```

The renderer still launches one TTS subprocess and a separate real Alignment
subprocess. The observer forwards the actual embedded stage and task arguments
with the same `sys.argv` that `python -c` supplies. It prints no extra stdout
and does not suppress genuine runtime errors. Do not install this wrapper or
add it to the repository.

## Validate and render

Capture the expanded CLI commands, machine-readable stdout, stderr, and exit
codes. Static validation must pass before the render and leave `RUN_DIR`
absent. No `voice.pt` load or model call occurs during validation.

```bash
exec 3> "$EVIDENCE_DIR/commands.log"
export BASH_XTRACEFD=3
set -x
if "$CLI_PYTHON" -m media_pipeline.cli speech validate "$SCRIPT_PROJECT/speech.json" \
  > "$EVIDENCE_DIR/validate.json" 2> "$EVIDENCE_DIR/validate.stderr"; then
  validate_exit=0
else
  validate_exit=$?
fi
printf '%s\n' "$validate_exit" > "$EVIDENCE_DIR/validate.exit"
test "$validate_exit" -eq 0 || exit "$validate_exit"
test ! -e "$RUN_DIR" || exit 1

if "$CLI_PYTHON" -m media_pipeline.cli speech render "$SCRIPT_PROJECT/speech.json" \
  --output "$RUN_DIR" \
  --tts-python "$EVIDENCE_DIR/tts-python-observed" \
  --alignment-python "$ALIGNMENT_PYTHON" \
  --tts-model "$BASE_MODEL" \
  --alignment-model "$ALIGNMENT_MODEL" \
  --device "$DEVICE" \
  > "$EVIDENCE_DIR/render.json" 2> "$EVIDENCE_DIR/render.stderr"; then
  render_exit=0
else
  render_exit=$?
fi
printf '%s\n' "$render_exit" > "$EVIDENCE_DIR/render.exit"
test "$render_exit" -eq 0 || exit "$render_exit"
set +x
```

Use the existing installed `media-pipeline speech ...` entry point instead of
`python -m media_pipeline.cli` only if it resolves this exact checkout. Keep
the explicitly selected Base model and separate Alignment runtime flags.

## Check artifacts and record listening

```bash
"$CLI_PYTHON" - <<'PY' > "$EVIDENCE_DIR/artifacts.json"
from collections import Counter
import json, os, wave
from pathlib import Path

evidence = Path(os.environ["EVIDENCE_DIR"])
run = Path(os.environ["RUN_DIR"])
validate = json.loads((evidence / "validate.json").read_text(encoding="utf-8"))
render = json.loads((evidence / "render.json").read_text(encoding="utf-8"))
voice = {"type": "clone", "asset": "voices/zhaowei-v1.pt"}
assert validate["status"] == "valid" and validate["plan"]["voice"] == voice
assert "speaker" not in validate["plan"]
assert render["status"] == "complete"
assert len(render["segments"]) == 3 and all(s["status"] == "ok" for s in render["segments"])
assert (evidence / "validate.stderr").read_bytes() == b""
assert (evidence / "render.stderr").read_bytes() == b""
assert (run / "final" / ".complete").is_file()
for name in ("final.wav", "final.srt", "timeline.json"):
    assert (run / "final" / name).is_file()
assert json.loads((run / "final" / "timeline.json").read_text(encoding="utf-8"))
assert (run / "final" / "final.srt").read_text(encoding="utf-8").strip()
request = json.loads((run / "request.json").read_text(encoding="utf-8"))
report = json.loads((run / "report.json").read_text(encoding="utf-8"))
assert request["voice"] == report["config"]["voice"] == voice
assert "speaker" not in request and "speaker" not in report["config"]
assert report["status"] == "complete"
assert report["completed_stages"] == ["preflight", "tts", "alignment", "postprocess", "captions", "assemble"]

events = [json.loads(line) for line in Path(os.environ["OWNER_TTS_EVENTS"]).read_text(encoding="utf-8").splitlines()]
counts = Counter(event["kind"] for event in events)
assert counts == {"asset_load": 1, "base_load": 1, "synthesis": 3}
assert len({event["pid"] for event in events}) == 1
loaded = next(event for event in events if event["kind"] == "asset_load")
base = next(event for event in events if event["kind"] == "base_load")
for event in events:
    if event["kind"] == "synthesis":
        assert event["asset_id"] == loaded["asset_id"]
        assert event["engine_id"] == base["engine_id"] and event["model_id"] == base["model_id"]

wav_records = []
segment_wavs = sorted((run / "segments").glob("*.wav"))
assert len([path for path in segment_wavs if not path.name.endswith(".cleaned.wav")]) == 3
assert len([path for path in segment_wavs if path.name.endswith(".cleaned.wav")]) == 3
for path in segment_wavs + [run / "final" / "final.wav"]:
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
        assert wav.getcomptype() == "NONE" and wav.getnframes() > 0
        rate, frames = wav.getframerate(), wav.getnframes()
        assert rate > 0
        wav_records.append({"wav": str(path.relative_to(run)), "sample_rate": rate,
                            "frames": frames, "duration_seconds": frames / rate,
                            "channels": 1, "sample_width_bytes": 2})
assert len(wav_records) == 7
print(json.dumps({"lifecycle_counts": dict(counts), "wavs": wav_records}, indent=2))
PY
```

Expected render layout includes `request.json`, `report.json`, three raw TTS
WAVs, three `.cleaned.wav` files and their existing downstream artifacts, plus `final/final.wav`,
`final/final.srt`, `final/timeline.json`, and `final/.complete`. The successful
report proves Forced Alignment, Postprocess, Caption Compiler, and assembly
finished; the observer proves one successful asset load, one successful Base
engine/model load, and three genuine syntheses in the same process/session.

Listen to the final WAV against the accepted voice reference and review the
SRT against playback. Record each result explicitly: cloned identity
recognizable; cross-segment identity acceptable; transitions acceptable;
subtitle synchronization accepted. A complete CLI result alone does not close
the human gate.

Add owner evidence to the Draft PR with `pr-head.txt`, `runtime.json`, exact
commands, exit codes and both CLI JSON outputs, `tts-events.jsonl`, final
artifact metadata, and those four listening decisions. Identify the date and
owner-reported provenance. Preserve the selected run/evidence directories for
review; do not claim this procedure has passed until it was actually executed.
