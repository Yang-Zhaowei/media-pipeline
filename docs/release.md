# v0.1.0 release candidate

PR #17 is prepared for manual owner review. The [original candidate](validation/regrain-v0.1.0-owner-acceptance.md)
has owner-reported real Core integration and Client Codex/Pi acceptance, with
variable Clone consistency under Issue #8. The [CPU/package record](validation/regrain-v0.1.0-candidate.md)
remains historical. Neither record approves a new wheel automatically.
The owner controls Ready status, merge, final tag and publication.

## Build and identify

From a clean, committed checkout, create a separate build environment (Python
3.10+), install `hatchling==1.27.0`, and run:

```console
python scripts/prepare_release.py --output dist/REPLACE_WITH_FRESH_ARTIFACT_DIRECTORY
```

The script refuses dirty source and existing artifact names, uses the commit
timestamp, checks package/distribution version agreement, and produces the wheel
and `regrain-0.1.0-manifest.json`. Keep both together: the manifest records source
commit/tree, Python/build versions, size and SHA256; `--version` only reports 0.1.0.
The script publishes nothing. For reproducibility, retain the recorded build
environment versions and compare two builds of the same clean commit locally.

The wheel contains the lightweight Controller, runtime adapters and Skill/examples.
No required runtime dependencies were added; model requirements stay in their
existing environments. Experiments, checkpoints and private assets are not shipped.

## Verification before review

```powershell
.venv/Scripts/python.exe -m pytest -o addopts=-ra -q
$env:REGRAIN_TEST_WHEEL = (Resolve-Path ./dist/REPLACE_WITH_FRESH_ARTIFACT_DIRECTORY/regrain-0.1.0-py3-none-any.whl).Path
.venv/Scripts/python.exe -m pytest tests/test_packaging.py -o addopts=-ra -q
git diff --check
```

Set `REGRAIN_TEST_WHEEL` equivalently on other shells. Without it, installed-wheel
tests explicitly skip rather than building/downloading during a unit test run.
They inspect packaging, validate shipped examples and exercise installed CLI /
isolated CPU workers outside the checkout. Their [evidence limits](validation/regrain-v0.1.0-candidate.md#what-installed-wheel-acceptance-proves)
remain distinct from GPU acceptance. Also run Skill lint, documentation local-link
checks, and review changed-file scope. Do not change immutable fixtures.

## Owner acceptance and release checklist

1. Review the final diff, manually mark PR #17 Ready for Review, and merge when
   satisfied. Closure leaves it Draft and unmerged.
2. Build from the exact intended merged/release source and run the checks above.
   Record its own source commit, final wheel SHA256 and manifest. The original
   GPU-tested candidate, closure HEAD, future merge commit and final wheel are
   separate identities. README metadata and packaged Skill changes alter wheel
   bytes even if application Python is unchanged; retain the old wheel/manifest.
3. Follow [deployment verification](deployment.md#owner-only-core-candidate-installation)
   for the **exact final wheel**: fresh versioned Controller, separate accepted
   runtimes, static validation, real CustomVoice and Clone GPU E2E, artifacts,
   listening/transitions/subtitle sync, and rollback. Record owner acceptance tied
   to that binary before publication. CPU passes or unchanged Python do not replace
   this gate. Explicitly assess/accept the remaining Clone quality limitation;
   unattended consistency and long-form stability are not established.
4. Resolve license and [naming/rename decisions](naming.md) before public release.
   Any source/metadata change requires a newly identified/tested artifact.
5. Only after exact-artifact acceptance and explicit owner approval, create
   `v0.1.0` and publish wheel, manifest, release notes and acceptance evidence.
   Do not silently replace the accepted wheel with a rebuild.

If source changes after acceptance, rebuild and repeat affected CPU and owner GPU
acceptance. Issue #8 remains open. No Ready/merge, repository rename, tag, release,
PyPI publication or deployment is performed by this closure task.
