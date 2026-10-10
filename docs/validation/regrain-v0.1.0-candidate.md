# Regrain v0.1.0 candidate — Client CPU verification

Date: 2026-10-10. Host: PC_Client / Windows, Python 3.12.14.
Baseline: clean main `ff73167436937b3788c297b11d9c4b67235c2f5d` (merged PR #16).
This record describes release-preparation checks. It does **not** claim real GPU
or listening acceptance of this candidate. Existing evidence stays tied to its
original tested commits. Final candidate artifact/source identities are recorded
in the generated release manifest, not inferred from the package version alone.

## Executed checks

| Command / check | Observed result |
| --- | --- |
| `.venv/Scripts/python.exe -m pytest tests/test_render.py -q` | 67 passed (worker bootstrap change). |
| `.venv/Scripts/python.exe -m pytest tests/test_cli.py tests/test_render.py -o addopts=-ra -q` | 143 passed. |
| `.venv/Scripts/python.exe -m pytest -o addopts=-ra -q` | 630 passed, 13 skipped. |
| `REGRAIN_TEST_WHEEL=<absolute wheel>` then `.venv/Scripts/python.exe -m pytest tests/test_packaging.py -o addopts=-ra -q` | 3 passed. |
| Skill creator `quick_validate.py skills/regrain-speech` | Passed (PyYAML only in disposable build environment). |
| `python -m hatchling build -t wheel -d dist` | Built v0.1.0 wheel; contents and metadata inspected. |
| `python -m pip wheel --no-deps --no-build-isolation . --wheel-dir dist/pre-review` | Standard PEP 517 wheel build passed with pinned Hatchling. |
| `python scripts/prepare_release.py --output dist/dirty-refusal` before commit | Correctly refused dirty source with exit 1. |
| Current documentation local links and `git diff --check` | Passed. |

The full-suite skips were: six unconfigured real GPU integration tests, three
real-torch serialization tests (torch absent), one Windows directory-symlink
privilege test, and three opt-in wheel tests. Those three wheel tests were then
executed separately with an explicit wheel and passed.

The initial wheel-test attempt had two passes and one failure when the preserved
caption module's Unicode help used Windows cp1252 in a pipe. The test now sets
UTF-8 IO encoding, as existing caption-help tests already do, and all three pass.
The legacy caption implementation was not changed. Sandbox-restricted Python/
cache launches were rerun with approved execution; they were not product failures.

## What installed-wheel acceptance proves

An offline installation into a fresh temporary Controller environment **outside
the repository checkout**, with no development PYTHONPATH/PYTHONHOME, provides:

- `regrain` and `media-pipeline` help/version compatibility and `regrain 0.1.0`.
- Package version agreement with distribution metadata; no torch, qwen_tts,
  qwen_asr or soundfile in the Controller environment.
- Static validation of shipped CustomVoice and clone examples; the clone asset
  need not exist during static validation. The legacy caption module remains usable.
- Correct production-only wheel contents: one `media_pipeline` package, runtime
  adapters, Skill/examples and distribution metadata; no tests, experiments,
  checkpoints or private voices.
- A complete two-unit render using two separate pip-free worker environments,
  injected CPU worker scripts and real deterministic downstream processing.
  Both workers have a deliberately unusable stale `media_pipeline`; both still
  import the exact installed Controller package. A conflicting top-level module
  beside the Controller cannot shadow each worker's own dependency. Final WAV,
  SRT, timeline, report and completion marker are checked.

These workers do not import real model frameworks or exercise GPU inference,
model/asset compatibility, voice quality or acoustic synchronization against
actual speech. Those are required owner gates in [deployment](../deployment.md)
and the [release checklist](../release.md).

The final clean-commit build, manifest and reproducibility comparison are emitted
to ignored `dist/` artifact directories. Test the selected final wheel again
with `REGRAIN_TEST_WHEEL` before owner transfer; the task report records its hash
and exact source commit. No immutable fixtures or experiment implementations changed.
