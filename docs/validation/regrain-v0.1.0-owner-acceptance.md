# Regrain v0.1.0 — owner acceptance

Recorded for PR #17 closure on 2026-10-11. Execution dates were not supplied.
The results below are **owner-reported / owner-verified**, not independently
rerun by this agent. Historical Client CPU/package checks remain in the
[candidate record](regrain-v0.1.0-candidate.md).

## Original accepted artifact

| Identity | Original GPU-tested candidate |
| --- | --- |
| Source / then PR HEAD | `dbcf0db6b23cbf43b21ee0c43d47f6a5279f5d97` |
| Wheel | `regrain-0.1.0-py3-none-any.whl` |
| Wheel SHA256 | `8779743fe2251ba25e335339aabdbd80153dce65b2346f88f3ae05c1df633ea9` |

The owner used an installed Controller on ai-core / Ubuntu 24.04, NVIDIA RTX
4070 12GB, Python 3.12.14 and PyTorch 2.14.0+cu130. TTS and ForcedAligner stayed
in their existing separate virtual environments; no model dependency migration
or environment merging was reported.

## Real Core rendering and listening

CustomVoice completed real GPU rendering and generated final WAV, SRT, timeline
and report. Pronunciation, spoken-content completeness, subtitle alignment and
synchronization passed. Opening and closing speaker identity was generally
consistent, with slightly more energetic/excited closing delivery and no major
identity drift reported. **Runtime integration and pronunciation/subtitles:
PASS; voice consistency: acceptable with minor delivery variation.**

Base Clone completed three renders of the same short two-unit script:

| Run | Owner listening observations |
| --- | --- |
| 001 | Opening identity uncertain; significant opening/closing emotional-delivery difference; closing matched the accepted reference voice. |
| 002 and 003 | Identity more consistent than 001; opening more expressive/free; closing deeper, with stronger perceived reverberation/spatial coloration. |

All three completed real GPU rendering and passed pronunciation, spoken-content
completeness and subtitle synchronization. **Runtime integration and
pronunciation/subtitles: PASS; repeated-generation identity consistency:
VARIABLE; unattended production-quality consistency: NOT ESTABLISHED.**

Reverberation/spatial coloration is a listening impression, not evidence of a
changed reverb-processing algorithm. These observations fit the broader
[Issue #8](https://github.com/Yang-Zhaowei/media-pipeline/issues/8) question, but
do not strictly exclude a PR #17 regression: no controlled old-versus-new
comparison was performed. Issue #8 remains open.

## Client Agent integration

The owner uses Windows PC_Client with uv-managed Python and one reusable
`uv tool` installation, rather than dependencies in each content project.

| Host | Owner-reported result |
| --- | --- |
| Codex 6.1 Luna | Valid Clone script with two Performance Units and no nonempty instruct. Initial invocation missed the updated subprocess PATH; direct `C:\Users\zhaowei\.local\bin\regrain.exe` succeeded. `--version`: exit 0, `regrain 0.1.0`; `speech validate`: exit 0, `status: valid`. |
| Pi + local model | Valid Clone script with three Performance Units, effectively one sentence per unit, and no nonempty instruct. `regrain --version` and `speech validate`: exit 0; validation `status: valid`. No GPU rendering attempted. |

The Codex failure was executable discovery, not a Regrain CLI failure. Both
reported workflows successfully invoked the authoritative validator. This does
not independently verify every Skill-discovery mechanism, nor establish an
optimal segmentation strategy. The Skill's multi-sentence preference is soft
guidance; both reported inputs were schema-valid.

## Evidence availability and release boundary

The owner retains `custom-report.json`, `clone-report.json`, `runtime.txt`,
`wheel.sha256.txt`, listening records and additional Clone runs 002/003 on Core.
These raw files, audio and Agent execution reports were not available in this
workspace for inspection. No report internals, additional hashes, timings or
model metrics are inferred. Private assets and generated audio are not committed.

This closure changes documentation and a packaged Skill, not application Python,
schema, generation controls, packaging configuration or accepted tests. README
metadata and Skill bytes change the wheel even when application code is identical.
Keep four identities separate:

1. The original source/wheel above: owner GPU-tested evidence.
2. The new closure PR HEAD/wheel: identified by its own generated manifest and
   CPU/package checks; **not GPU-tested by this agent or accepted by inheritance**.
3. The future merged source commit: an owner action, not yet an accepted artifact.
4. The final release wheel/hash: rebuild and test it, then obtain owner acceptance
   of that exact binary before publication, as required by the [release gates](../release.md).

The original wheel and manifest remain historical evidence. Package version
`0.1.0` alone never transfers acceptance between these artifacts.
