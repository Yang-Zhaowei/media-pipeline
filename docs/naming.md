# Naming and migration boundary

Decision for v0.1.0:

| Layer | Name | Reason |
| --- | --- | --- |
| Project / intended GitHub repository | `regrain` | Owner-selected identity. |
| Python distribution | `regrain` | A single lightweight wheel. |
| Main command | `regrain` | Stable agent and human entry point. |
| Python import | `media_pipeline` | Preserve accepted APIs, runtime imports and experiment replay. |
| Compatibility command | `media-pipeline` | Existing accepted CLI procedures and local scripts can keep working. |

Both console commands call `media_pipeline.cli:main`; there is no second
implementation or parallel Python package. Help and `--version` identify regrain.
`python -m media_pipeline.cli` still invokes speech tools, while
`python -m media_pipeline` retains its existing caption compiler interface.
`MEDIA_PIPELINE_*` and the existing `MEDIA_ALIGNMENT_*` aliases keep their exact
precedence. This avoids renaming accepted host configuration as part of packaging.
The speech JSON, Python signatures, result formats and error codes stay compatible.

Use a fresh Controller environment when migrating from the old distribution.
Installing both distributions in one environment would make them own the same
`media_pipeline` files and console alias. The new environment is also the rollback
boundary; the old accepted installation is kept intact.

## Checks on 2026-10-10

- Authenticated GitHub `GET /repos/Yang-Zhaowei/regrain` returned 404. The current
  source is still `Yang-Zhaowei/media-pipeline`; a 404 does not reserve the name or
  guarantee visibility into every repository.
- Both the [PyPI JSON endpoint](https://pypi.org/pypi/regrain/json) and
  [simple index endpoint](https://pypi.org/simple/regrain/) returned HTTP 404.
  No published distribution was visible under that exact normalized name at the
  time of checking. Under Python's [name normalization rules](https://packaging.python.org/en/latest/specifications/name-normalization/),
  `ReGrain` normalizes to `regrain`; `re-grain` and `re_grain` normalize to
  `re-grain`, a different name. A local wheel install does not reserve PyPI names.
- The name is already used by an [AI creation website](https://regrain.co/), and
  “Regrain” is a [Boris FX Silhouette node](https://borisfx.com/documentation/silhouette-2025/silhouette-2025/Tutorials-Regrain.html)
  and a [Filmworkz tool](https://filmworkz.com/dvo/regrain-rgb/). The public identity
  is not globally unique. These are observed naming overlaps, not a legal clearance.
- No `regrain` command was found on this Client's PATH before installation.
  That says nothing about other hosts; use the versioned environment's full
  executable path to avoid command collisions.
- No authoritative tokenizer for the execution agent was available locally.
  Token count was not measured and no single-token claim is made. Tokenization
  does not justify changing the accepted internal package.

## Owner repository rename

After review, the owner can rename the GitHub repository in repository Settings
to `regrain`, update local remotes to the new URL, and update the current
`project.urls.Repository` metadata. This task does not perform remote administration.
Preserve historical PR/issue URLs, tested commit SHAs, old environment paths and
recorded evidence; they describe the state actually tested. Recheck name availability
before any future public index publication. No PyPI publication is prepared here.
