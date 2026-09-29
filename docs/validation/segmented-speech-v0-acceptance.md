# PR #7 feature acceptance review — 2026-09-30

Reviewed implementation: `f34f47aef0d26d0673a2c5334b8185571fae4e8b`.
Original contract: [Segmented Speech v0](../contracts/segmented-speech-v0.md),
unchanged from `992f518`, planned against `main` at `795c4b6`.

**Decision: accepted and merged** (PR #7, 2026-09-30). The narrow C3 preflight
gap below was closed by a repair confined to static preflight, so the real
generation, timeline, load-once and recorded human acceptance evidence need not
be repeated. This document records the local review that preceded the merge.

## Outstanding acceptance gap

At `src/media_pipeline/render.py:263`, `_validate_segments` first records that
a segment is not an object, but the subsequent unconditional `dict(seg)`
conversion raises before the collected errors can be returned.

Reproduced through public `load_and_validate_script` with budget 200:

```json
{"language":"", "speaker":"", "segments":[null]}
```

Actual: `TypeError: 'NoneType' object is not iterable`. Expected: `RenderError`
containing all three static validation errors, before creating a run or loading
models. `render_speech` invokes this same preflight before its run error handler.
The existing passing suite does not cover non-object array elements.

Also reproduced through `render_speech` itself: no stage callable was invoked
and no run directory was created, but the exception was still the raw `TypeError`.

The same C3 boundary also reports an over-budget segment by array index only;
the contract explicitly requires its ID, actual length and limit. Include the
ID when valid, retaining the index for malformed entries.

Public-call reproduction with ID `chapter-07`, text length 11 and budget 10
returned only `segment[0].text is 11 code points, over the limit of 10` (followed
by rejection guidance), confirming the missing ID. No model stage ran.

Required closure: correct the preflight path and add focused CPU cases for
non-object entries mixed with other invalid fields, and for the over-budget
diagnostic. Verify the public render call starts neither model stage and leaves
no run directory. Run the relevant tests and full CPU regression. No splitting,
normalization or expanded input schema is needed.

## Contract and repairs

| Contract | Acceptance assessment |
| --- | --- |
| C1–C2 | Existing four components reused; dedicated synchronous Python entry and two separate runtimes. No material architectural deviation. |
| C3 | Outstanding preflight gap above. Valid script values are preserved; no automatic splitting. |
| C4 | Sequential TTS then Alignment, independent segment alignment; CPU and real load-count evidence available. |
| C5 | Repair commits narrow isolatable faults and preserve I/O/runtime failure context; negative CPU regressions pass. |
| C6 | Staged products are verified before copying to final; successful report precedes the completion marker. Failed/incomplete runs are not complete episodes. |
| C7 | Cleaned PCM frame accumulation, explicit silence, one global SRT render and final artifact readback supported by CPU tests and real timeline. |
| C8 | Current CPU suite executed locally; real ai-core and human evidence recorded separately below. |
| C9 | No new modality, public CLI/service, generic orchestration, splitting, recovery or environment consolidation. |

Reviewed repair history, rather than just the initial implementation:

- `5719a69`: current-checkout subprocess imports, structured error manifests,
  narrower fault handling, failed reports, report/marker ordering and timeline checks.
- `fd3df79`: distinguish wrapped input I/O from isolatable alignment validation.
- `604db36`: preserve segment context on output I/O and unknown runtime faults;
  avoid converting unknown postprocess exceptions into validation failures.
- `8df5262`: reject non-array alignment JSON as an isolatable validation error.
- `f34f47a`: record the completed Alignment stage, with a stage-order regression.

The independent review conversation, “审查分段 Speech PR”, identified missing
real load-once proof and the report metadata omission. Both have subsequent
evidence/fixes. Its recorded acceptance is compared with the actual current
code here; it does not override the newly reproduced C3 gap. Earlier independent
code-review findings were not available as a complete original itemized report
in the retrieved records. The repair diffs and their tests were inspected, but
this review does not claim an independently verified one-to-one closure of every
historical reviewer comment.

## Tests actually executed in this review

On the client at `f34f47a`:

```text
.venv\Scripts\python.exe -m pytest -o addopts=-ra -q
287 passed, 6 skipped in 11.96s
```

This includes 54 render tests. Six existing gated GPU tests were skipped:
three Alignment, two TTS and one prior E2E test. These skips are not GPU evidence.
The separate malformed-script reproduction above failed as described; it was
not a new committed test. No GPU inference was executed by this review.

## Recorded ai-core and human evidence

Sources: [PR #7 acceptance body](https://github.com/Yang-Zhaowei/media-pipeline/pull/7)
and the owner's runtime logs in
[审查分段 Speech PR](https://chatgpt.com/c/6abbd810-85d0-83e8-8d63-9f119ef61441).
Artifact attachments were not exposed by conversation retrieval, so this review
does not claim to have independently downloaded, rehashed or listened to them.

Accepted runtime implementation: `8df5262c23e318c761d83faecc75ec39bdb93c1b`.
Recorded ai-core test results: render suite **53 passed**; full suite
**289 passed, 3 skipped**. Keep these separate from the current local CPU run.

Fresh public `render_speech(...)` used real Qwen3-TTS-12Hz-1.7B-CustomVoice and
Qwen3-ForcedAligner-0.6B in separate environments, on `cuda:0`, with three unequal
Chinese segments and pauses 280/650/420 ms. Status was `complete`, marker present.

| Segment | Start frame | Audio frames | Pause frames |
| --- | ---: | ---: | ---: |
| intro | 0 | 165120 | 6720 |
| middle | 171840 | 270720 | 15600 |
| ending | 458160 | 124800 | 10080 |

Final WAV readback: 24000 Hz, 593040 frames and samples, **24.71 seconds**.
The sums close exactly, including the tail pause.

Recorded SHA-256 values (from the accepted artifact run, not a new inference):

```text
final.wav      87482bda7d2d5daa00cf44b3ff81bbc9824c426e6a246be58cb33791def5b028
final.srt      353c2760b82af6baa64c63b05760b7432f2e2cf76500c9ac61e0c57fbbaea0d3
timeline.json  442f2121f2617a4ca0cb5c4dc44d03f5dbca5c851c920146a34be8de570d460b
report.json    ad031600a9c61434bfd750d4d70f122a28eca4e1e3da558e0fe26f463ef4be5a
```

Separate real instrumentation without production changes recorded:

```text
TTS pid=402450:   1 process, 1 model load, 3 generate calls
ALIGN pid=402621: 1 process, 1 model load, 3 align calls
text lengths:    29, 44, 23
```

This closes the prior load-once evidence gap. The instrumentation run is distinct
from the artifact run; do not attribute the hashes above to that second run.

The owner explicitly said listening was acceptable. The PR acceptance record
also marks intelligibility, junctions/pauses and SRT synchronization PASS.
The owner observed varying emotion/prosody despite identical `instruct`.
Consistent emotional delivery is not promised by C1–C9; record it as a real
quality limitation for longer podcast evaluation, not a new requirement here.

The only change from the GPU-tested implementation to `f34f47a` is completed-stage
metadata plus its CPU test. Its diff does not affect inference, timestamps,
assembly or model lifecycle; a repeat GPU run for that change is unnecessary.

## Documentation and merge follow-up

Merged on 2026-09-30. README, CURRENT and the entry-point documentation now
describe the completed runtime/human validation without unmerged notices. The
original contract and regression fixtures are unchanged.

The C3 preflight repair and its CPU tests were recorded; the merge was captured
and the entry point moved to Completed. Preserve the historical PR #5 closure;
HTML slides, video composition and NLE operations stay outside this feature.
