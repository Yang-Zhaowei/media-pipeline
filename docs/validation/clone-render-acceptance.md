# Segmented clone render acceptance

Status: **complete**. The owner accepted Issue #14 on 2026-10-04
(Asia/Shanghai); [PR #15](https://github.com/Yang-Zhaowei/media-pipeline/pull/15)
is squash merged and Issue #14 is closed.

## Provenance

- Exact accepted PR HEAD: `798f26d7efb60b552483df812d7ca0362732784c`.
- Squash commit: `e8d39f239df36bcb1430c02bb2b439ea687872bf`; its repository
  tree matches the accepted HEAD, verified locally.
- Source: owner-updated PR #15 body and explicit merge authorization.
  This record transcribes owner-reported results. During this documentation
  closure, the agent did not rerun inference, CPU/GPU suites, or listening,
  or independently inspect the media.

## Runtime and validation

Runtime: PyTorch `2.14.0+cu130`, `qwen-tts 0.1.1`, NVIDIA GeForce RTX 4070,
`cuda:0`, existing `Qwen3-TTS-12Hz-1.7B-Base` checkpoint, accepted reusable
`voice.pt`, and the separate Alignment runtime.

At the exact HEAD, owner ai-core checks passed:

- `tests/test_render_clone.py -q`: **55 passed**, including Linux symlink containment.
- `tests/test_cli.py -q`: **75 passed**.
- Three-unit clone `speech validate`: exit **0**, `status=valid`, empty stderr;
  no render directory created. The plan retained
  `voice: {"type":"clone","asset":"voices/zhaowei-v1.pt"}`, no `speaker`,
  and omitted-versus-explicit-empty instruction provenance.
- Real `speech render`: exit **0**, `status=complete`, empty stderr;
  `opening`, `discussion`, and `closing` all `status=ok`.
- Genuine runtime observer: **1 asset load, 1 Base load, 3 syntheses**,
  reusing the same asset, engine/model, and TTS process/session.
- Completed: `preflight → tts → alignment → postprocess → captions → assemble`.
  Published `final/final.wav`, `final/final.srt`, `final/timeline.json`,
  and `final/.complete`.

Separate historical Windows CPU validation: clone **54 passed, 1 skipped**;
CLI **75 passed**; existing render/TTS/clone regressions **214 passed, 5 skipped**;
full suite **547 passed, 10 skipped**; `git diff --check` passed.
Skips covered optional GPU integration, absent real torch, and Windows symlink
permissions; the owner Linux clone suite subsequently covered the symlink case.

## Artifacts and human acceptance

| Output | Sample rate | Frames | Duration |
| --- | ---: | ---: | ---: |
| opening cleaned | 24000 Hz | 97680 | 4.07 s |
| discussion cleaned | 24000 Hz | 124560 | 5.19 s |
| closing cleaned | 24000 Hz | 109200 | 4.55 s |
| final.wav | 24000 Hz | 341040 | 14.21 s |

Human listening: **PASS** for recognizable cloned identity, cross-segment
identity consistency, segment transitions, and subtitle synchronization.
Issue #14's real ai-core and human merge gates are closed.

Audible emotion/delivery drift remains across segments and is tracked in
[Issue #8](https://github.com/Yang-Zhaowei/media-pipeline/issues/8). It does
not block the accepted identity, runtime reuse, or renderer integration.

The owner also recorded descriptive acoustic measurements of the accepted
final WAV (these are not emotion classifications):

| Metric | opening | discussion | closing |
| --- | ---: | ---: | ---: |
| text chars / cleaned second | 5.90 | 5.39 | 5.27 |
| RMS amplitude | 0.123 | 0.101 | 0.083 |
| median F0 | 265 Hz | 254 Hz | 238 Hz |
| F0 IQR | 101 Hz | 57 Hz | 58 Hz |

The decreasing pace, energy, and median pitch, with wider opening pitch
variation, support the owner's delivery-drift observation.

## Replay

The [exact-head procedure and parameterized CLI commands](https://github.com/Yang-Zhaowei/media-pipeline/blob/798f26d7efb60b552483df812d7ca0362732784c/docs/validation/clone-render-acceptance.md)
retain the three-unit script, genuine lifecycle observer, JSON/exit capture,
and artifact checks. Use caller-selected existing interpreters/checkpoints and
accepted asset, with new output/evidence directories; do not download models,
change environments, or commit runtime assets as fixtures.
