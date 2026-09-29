# Speech Pipeline v0 closure evidence

Status: **complete**, accepted by the repository owner on 2026-09-29.

## Provenance

- [PR #5](https://github.com/Yang-Zhaowei/media-pipeline/pull/5): real ai-core
  execution results and artifact hashes reported by the owner.
- Exact GPU-tested PR head: `93c650d2ae03e2a0771ff1f7aa9089a7a23e90ee`.
- Squash merge on main: `4e3422bebf749e7bc41ff68001eff84791819b23`.
  Its repository tree is identical to the tested PR head.
- Local review at that PR head: full pytest suite, **233 passed, 6 skipped**.
  All six skips were gated GPU integration tests with no runtime configuration.
- The owner confirmed human validation passed in the review conversation:
  `检查pr5是否可以merge,人工验证通过`. This is the human listening / SRT
  synchronization acceptance for the fresh artifacts, not an automated result.

This document transcribes existing evidence. The documentation update did not
rerun GPU inference, independently inspect the ai-core artifacts, or claim that
later documentation commits were themselves GPU-tested.

## Real production run

Command on ai-core: `python3 validation/speech_pipeline_e2e.py`.

The driver ran Production TTS and Production Alignment sequentially in their
existing separate GPU environments, then production postprocessing, caption
compilation, and `validate_e2e`. The WAV and alignment were generated fresh;
historical fixture audio/alignment were not substituted.

Input: the read-only `tests/fixtures/speech-smoke-001/original.txt`, Chinese,
speaker `Uncle_Fu`, with the reviewed narration instruction. The two lines form
one supported utterance:

```text
真正限制本地人工智能模型使用体验的，
往往并不只是模型本身的参数规模。
```

Reported ai-core output directory: `/tmp/media-pipeline-e2e-20260929-202619`.
This is a historical host-specific location, not portable configuration or a
guarantee of permanent retention. Preserve the accepted run outside temporary
storage for later replay; no durable artifact upload is evidenced by this PR.

| Artifact / property | Recorded result |
| --- | --- |
| Raw WAV | Mono PCM16, 24000 Hz, 147840 frames, 6.160 s |
| Alignment | 32 tokens |
| Retained frame interval | `[0, 147840)` |
| Final WAV | Mono PCM16, 24000 Hz, 147840 frames, 6.160 s |
| Final SRT | 2 captions |
| Driver validation | PASS |
| Targeted real E2E pytest | 1 passed in 13.43 s |
| Full E2E integration file | 4 passed in 13.10 s (includes CPU checks) |
| E2E unit + integration files | 21 passed in 14.81 s (includes CPU checks) |

The configured real E2E test ran rather than skipping. These reported test
invocations are distinct runs; the artifact metadata above belongs to the
driver run, not necessarily each pytest invocation.

SHA-256:

```text
raw.wav
ff154648e0d3397431bf8f7776d9401f3295160c9e333d99267be48505c1ee2f

final.wav
955888468eeb3cb0e1d0b3d29003a44ae4f130f17f88d7a74d7cffc121f93104
```

The full-range keep is valid. Edge fades can change sample values without
changing frame count, explaining different raw/final hashes. This real run did
not exercise non-zero trimming; deterministic CPU tests cover non-zero trim,
frame quantization and signed alignment shifts.

## Acceptance and scope

Automated validation, full listening, SRT text / synchronization inspection,
and explicit owner acceptance complete the milestone. Human review remains
necessary for new production material; forced alignment does not prove that
TTS pronounced the intended words correctly.

This proves one supported Chinese utterance through the existing production
path. It does not validate arbitrary manuscript length, every voice, episode
assembly, browser visuals, loudness mastering, or final video export.
No production implementation or immutable regression fixture changed in PR #5.
See [the next real-workflow evaluation](../workflows/podcast-pilot.md).
