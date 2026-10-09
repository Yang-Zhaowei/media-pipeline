# Qwen Base clone prosody experiment — owner acceptance

Status: **Layer 1 experimental infrastructure accepted by the owner**.
[PR #16](https://github.com/Yang-Zhaowei/media-pipeline/pull/16) remains Draft;
[Issue #8](https://github.com/Yang-Zhaowei/media-pipeline/issues/8) remains open.
This records evidence and exploratory findings, not a production fix.

## Provenance and evidence boundary

Exact GPU-tested implementation HEAD:
`506b30dd59702348d6ff11bc7fad435acddf3a76`.
Starting main: `238901170096cd14f5d6b950b4d84e4663603600`.
Branch: `experiment/issue8-qwen-clone-prosody-v0`.

The owner supplied the Linux checks, real GPU execution results, blind-listening
ratings, condition mapping and acoustic-summary observations transcribed below.
The owner reports evaluating all 13 full-matrix samples before inspecting the
mapping; ratings and mapping were subsequently supplied for analysis.

The documentation closure on 2026-10-10 did not access ai-core, the experiment
directory, generated WAVs, `analysis.json` or `listening-mapping.json`, and did
not independently listen, recompute acoustics or compare PCM hashes. Its local
checks cover transcription, arithmetic, links, documentation-only scope and
diff hygiene. The new documentation commit is not the GPU-tested HEAD above.
No exact execution date, experiment-directory path or additional runtime version
is inferred from the supplied evidence.

The [experiment definition and replay procedure](qwen-clone-prosody-v0.md) and
[upstream control audit](qwen-generation-controls.md) remain the implementation
references. Original Windows CPU evidence is preserved in the experiment document
and PR description; the Linux evidence below is separate owner-reported work.

## Real Linux acceptance — owner confirmed

Environment: ai-core / Ubuntu Linux, RTX 4070 12GB, `cuda:0`, existing
`Qwen3-TTS-12Hz-1.7B-Base`, `qwen-tts 0.1.1` and accepted reusable `voice.pt`.
All checks below were reported at the exact implementation HEAD above.

| Validation | Owner-reported result |
| --- | --- |
| Clean exact-head checkout | PASS |
| Experiment-focused Linux tests | 82 passed |
| Voice clone/render/CLI regressions | 328 passed |
| Full Linux test suite | 633 passed, 6 skipped |
| Audited Qwen source SHA256 | PASS |
| Base checkpoint metadata | PASS |
| Reusable voice asset compatibility | PASS |

The six skips were unrequested GPU integration tests needing separate runtime
configuration, not failures. These counts do not claim that the agent reran the
Linux suite during documentation closure.

## Real GPU smoke and complete matrix

The owner first ran `S3-default` as a smoke test: **3/3 runs complete**.
Each run reported one asset load, one model load and three generation invocations.
Acoustic analysis and the blind-listening package were generated successfully;
the completion marker was present.

The subsequent full experiment reported **13/13 complete, zero failures**:

| Condition | Successful runs | Generation calls per run |
| --- | ---: | ---: |
| S3-default | 3/3 | 3 |
| B3-default | 3/3 | 1 |
| L1-default | 3/3 | 1 |
| S3-greedy | 2/2 | 3 |
| L1-greedy | 2/2 | 1 |

Every full-matrix run reported one asset load and one model load. The owner
confirmed `manifest.json` complete, `analysis.json` and `analysis.md` present,
the blind-listening package present, and `.complete` present. This records
successful real execution of the experimental infrastructure. The artifact
contents and their hashes were not independently checked in this closure.

## Owner blind-listening results

The following table preserves all supplied scores and explicit drift labels.
Sample identifiers refer to the full-matrix blind package, not the earlier smoke.
Sample-to-repetition assignments were not supplied and are not inferred.

| Sample | Condition | Score / 5 | Drift |
| --- | --- | ---: | --- |
| 01 | L1-greedy | 5 | None |
| 02 | S3-default | 3 | Obvious |
| 03 | L1-default | 5 | None |
| 04 | L1-greedy | 5 | None |
| 05 | L1-default | 5 | Mild |
| 06 | S3-greedy | 4 | Mild |
| 07 | B3-default | 4 | Mild |
| 08 | S3-default | 4 | Obvious |
| 09 | S3-greedy | 4 | Mild |
| 10 | S3-default | 5 | None |
| 11 | L1-default | 5 | None |
| 12 | B3-default | 4 | Obvious |
| 13 | B3-default | 5 | None |

Selected owner observations:

- Sample 02: Opening sounded too quiet relative to the following section.
- Sample 05: Closing sounded slightly hoarse compared with earlier content.
- Sample 08: Opening sounded calm, middle section cautious, closing more cheerful.
- Sample 11: The complete narration sounded consistently calm.
- Sample 12: First and third sections sounded faster than the middle section.

These comments do not replace the owner's drift labels. In particular, sample
05 remains **Mild**, and sample 12 remains **Obvious**.

| Condition | Mean score | None | Mild | Obvious |
| --- | ---: | ---: | ---: | ---: |
| L1-default | 5.00 | 2 | 1 | 0 |
| L1-greedy | 5.00 | 2 | 0 | 0 |
| B3-default | 4.33 | 1 | 1 | 1 |
| S3-greedy | 4.00 | 0 | 2 | 0 |
| S3-default | 4.00 | 1 | 0 | 2 |

Arithmetic checked from the supplied ratings: L1-default `15/3`, L1-greedy
`10/2`, B3-default `13/3` (rounded to 4.33), S3-greedy `8/2`, S3-default
`12/3`. Drift counts sum to 13 observations. These are descriptive results from
one owner and a small corpus, not a statistically validated ranking.

## H1–H4 interpretation

**H1 — Sampling stochasticity.** The owner reports measurable run-to-run
variation in default conditions and matching acoustic summary metrics for
greedy repetitions. Final owner verification confirmed identical `listening.wav`
PCM within each greedy repeat pair in this fixed environment (details below).
Sampling contributes to output variation,
but disabling it alone did not eliminate perceived delivery changes: both
S3-greedy samples were rated 4/5 with Mild drift.

**H2 — Independent generation / context reset.** This is the strongest current
engineering hypothesis. All five L1 entries received 5/5, whereas S3 showed
more variation and more reported emotional discontinuities. One L1-default
entry still had Mild drift despite its 5/5 score. L1 also removes hard
concatenation boundaries and changes generation topology; the observed
improvement cannot be attributed solely to retained model context.

**H3 — Batch topology.** B3's descriptive mean lies between S3 and L1, with one
None, one Mild and one Obvious drift label. There is insufficient evidence that
batch generation reliably solves drift. As established by the upstream audit,
batch items do not share continuous target-text context.

**H4 — Intrinsic long-form instability.** Unresolved. The corpus is only
approximately 15 seconds. Good L1 results cannot establish stability over
multi-minute podcast narration or exclude instability at longer durations.

The hidden greedy pairs received matching subjective evaluations: L1-greedy
5/5 with None drift in both entries; S3-greedy 4/5 with Mild drift in both.
This is encouraging within-session consistency, not a formal listener-reliability
or hearing assessment. The confirmed identical listening PCM must not be counted
as independent evidence of generation diversity.

The real ai-core experiment and owner blind listening provide promising
evidence that longer continuous generation improves perceived narration
continuity for this Qwen Base clone corpus. Sampling variability contributes
to output variation, but disabling sampling does not independently eliminate
perceived delivery changes. The finding is exploratory and does not establish
the optimal production unit length or resolve Issue #8.

## Final owner PCM verification — PASS

In the complete 13-run ai-core experiment directory, the owner computed SHA256
over the WAV PCM frame data of each pair's `listening.wav` and also compared
WAV format metadata. Both greedy repeat comparisons passed:

| Condition | Repeat pair | PCM identical | SHA256 of listening PCM (both repeats) |
| --- | --- | --- | --- |
| S3-greedy | S3-G-1 / S3-G-2 | True — PASS | `0601e9d93b004cf951b0fdf5486c61ddd67f04c911b48c60360b9bb7f675e57c` |
| L1-greedy | L1-G-1 / L1-G-2 | True — PASS | `dbdd94cf7458309590a7e96cd7a9afc8b359b553f194c41bbd3b4e1b42935905` |

Both pairs produced byte-identical listening PCM in this fixed environment.
Their hidden-repeat owner blind scores and drift labels were also consistent
within each pair: S3-greedy 4/5 with Mild drift; L1-greedy 5/5 with None drift.
Repeated identical audio is not independent evidence of generation diversity.
These hashes cover PCM frame data, not complete WAV-file bytes; no per-unit
hashes or additional format values are inferred. This is owner-reported
verification, not an independently recomputed agent result. It does not establish
cross-platform determinism or resolve Issue #8.

## Limits and remaining production validation

Natural paragraph prosody can legitimately vary in pitch, intensity and timing.
Neither the owner's qualitative comments nor descriptive acoustic summaries
are automated emotion measurements. No objective pitch values or statistical
significance are inferred here. Raw zero-gap segmented concatenation and L1
continuous generation also differ from production postprocessing and pauses.

The remaining production-validation question is: what caller-approved continuous
unit length preserves desired continuity in actual multi-minute podcast narration,
while retaining acceptable pronunciation, natural paragraph progression and
revision granularity? This is a question for later production-use validation,
not authorization to change segmentation, sampling APIs, providers or models.
PR #16 remains Draft, unmerged, and Issue #8 remains open.
