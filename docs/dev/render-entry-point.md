# Speech rendering

`render_speech(...)` turns a caller-authored script into one episode WAV, SRT
and timeline. The caller defines coherent performance units and any explicit
direction. A unit may contain multiple sentences; subtitle boundaries are
determined independently by Alignment and the Caption Compiler.

## Script

Ordinary narration omits `instruct` or uses `""`. A segment can inherit the
top-level instruction, explicitly clear it, or provide local direction.

```json
{
  "language": "Chinese",
  "speaker": "Uncle_Fu",
  "instruct": "开场时带有克制的紧迫感。",
  "segments": [
    {"id": "opening", "text": "欢迎收听本期节目。今天我们讨论一个具体问题。"},
    {"id": "plain", "text": "接下来回到普通叙述。", "instruct": ""},
    {"id": "emphasis", "text": "这个结论值得关注。", "instruct": "强调最后一句。"}
  ]
}
```

Here `opening` inherits, `plain` clears, and `emphasis` overrides the direction.
`pause_after_ms` optionally adds a pause after a unit. Instructions must be
strings; unknown fields are rejected. See the
[instruction contract](../contracts/performance-unit-instruct.md).

## Render

```python
from media_pipeline import render_speech

render_speech(
    script_path="script.json",
    output_dir="run_dir",
    tts_python="<TTS Python path>",
    alignment_python="<Alignment Python path>",
    tts_model="<TTS model path>",
    alignment_model="<Alignment model path>",
    max_segment_chars=200,
)
```

Use the verified [ai-core runtime setup](core-runtime.md). The output directory
must be new. `max_segment_chars` is the caller's per-unit limit; over-budget
text is rejected, never truncated. Invalid scripts fail before generation.

A complete run publishes `final/final.wav`, `final/final.srt` and
`final/timeline.json`. `request.json` preserves the authored instruction choices;
`report.json` records their effective values and run status.

## Validation and next step

[PR #9](https://github.com/Yang-Zhaowei/media-pipeline/pull/9) is merged. Real
ai-core generation, human listening and subtitle synchronization passed.
[Issue #8](https://github.com/Yang-Zhaowei/media-pipeline/issues/8) remains open
for production-use validation of long-form narration stability.

The renderer does not automatically split, interpret or rewrite the manuscript.
