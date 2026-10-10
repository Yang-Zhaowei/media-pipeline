# v0.1.0 release candidate

This is release preparation, not an approved production release. The accepted
predecessor remains available; this candidate needs installed-wheel ai-core E2E
and human acceptance before a final tag or GitHub Release. Issue #8 stays open.
No schema or audio semantics change is intended.

Executed Client checks and their limits are recorded in the
[candidate CPU verification](validation/regrain-v0.1.0-candidate.md).

## Build and identify

From a clean, committed checkout, create a separate build environment (Python
3.10+), install `hatchling==1.27.0`, and run:

```console
python scripts/prepare_release.py --output dist
```

The script refuses an uncommitted tree or an existing artifact name, builds the
wheel with the pinned Hatchling backend and commit timestamp, checks distribution/
package version agreement, and writes `regrain-0.1.0-manifest.json` alongside
`regrain-0.1.0-py3-none-any.whl`. The manifest records exact source commit/tree,
Python/build dependency versions, wheel size and SHA256. It is a sidecar because
embedding a final commit SHA in tracked source would create a self-reference.
Retain wheel and manifest together; `regrain --version` identifies the package
version, while the manifest identifies the source/artifact. The script neither
tags, uploads, deploys nor publishes. The development checkout and private assets
are not included in the wheel.

For an ordinary development wheel, `python -m pip wheel --no-deps . --wheel-dir dist`
also uses the pinned backend, but it does not supply the clean-source manifest.
The Controller has no required third-party runtime dependencies. `pytest` remains
a development extra; GPU requirements remain solely in the existing model environments.
The wheel includes one Python package, its two thin CLI entry points and the
portable authoring Skill. Experiment tools and immutable evidence remain in the
repository and are replayed from their recorded commits.

`uv.lock` retains the existing development dependency versions with only the root
distribution name/version changed. Hatchling is a build dependency, not a runtime
dependency. Build-environment tooling such as PyYAML for Skill linting is not a
product dependency. Future byte-for-byte rebuilds should use the manifest's build
dependency versions as well as its commit timestamp.

## Verification before review

```powershell
.venv/Scripts/python.exe -m pytest -o addopts=-ra -q
$env:REGRAIN_TEST_WHEEL = (Resolve-Path ./dist/regrain-0.1.0-py3-none-any.whl).Path
.venv/Scripts/python.exe -m pytest tests/test_packaging.py -o addopts=-ra -q
git diff --check
```

Set `REGRAIN_TEST_WHEEL` equivalently on other shells. Without it, installed-wheel
tests explicitly skip rather than building/downloading during a unit test run.
These tests install offline into a clean temporary environment outside the
checkout, validate shipped CustomVoice/clone examples, inspect wheel contents and
metadata, and exercise two external CPU workers plus real deterministic output
processing. They check that the Controller's neighboring dependencies cannot
shadow worker dependencies. They are **CPU tests, not GPU acceptance**.

Build a second artifact into a fresh output directory using the same clean commit
and build environment; compare wheel hashes to verify reproducibility locally.
Do not edit immutable fixtures to accommodate branding.

## Owner acceptance and release checklist

- Review and merge the focused PR only when satisfied; this task leaves it unmerged.
- Validate the reviewed wheel hash and manifest. If rebuilding after merge, record
  the resulting merge commit in a new manifest and test that exact artifact.
- Follow [deployment](deployment.md) to install a candidate in a fresh Core
  Controller directory. Preserve existing TTS/Aligner environments, checkpoints,
  private assets and accepted Controller. Verify static validation with no
  development PYTHONPATH, then real CustomVoice and clone GPU E2E plus listening,
  transitions and subtitle sync. Save evidence tied to the exact artifact/source.
- Confirm rollback to the previous Controller and approve promotion/current switch.
- The owner may rename the repository and update current package repository metadata
  as described in [naming](naming.md). Historical PRs and GPU-tested SHAs stay intact.
- Decide the repository license before public release; no license is inferred or
  selected by this task. Review the known naming overlaps before public branding.
- After explicit owner approval, create the final `v0.1.0` tag at the accepted
  source and publish a GitHub Release with wheel, manifest, changelog and acceptance
  evidence. No tag, GitHub Release, PyPI upload or live deployment is authorized here.

If source/packaging changes after acceptance, rebuild and rerun affected CPU checks;
changes to the installed worker path or runtime behavior require owner GPU reacceptance.
An accepted short render does not close the separate long-form stability question.

## Candidate changes

Regrain identity, v0.1.0 packaging and version output; retained `media_pipeline`
imports, caption module entry and a `media-pipeline` console compatibility alias;
exact Controller package loading in isolated workers; installed-wheel regression
coverage; portable speech-authoring Skill; manual deployment and rollback guide.
No production runtime dependencies or model controls added, and no accepted
regression fixtures or historical experiment implementations changed.
