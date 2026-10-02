#!/usr/bin/env python3
"""Deterministically regenerate the four Issue #8 render scripts.

Reads the frozen ``original_manuscript.json`` and writes ``inputs/A..D.json``
plus an ``inputs/MANIFEST.json`` that records the character-identity proof and
the sentence-junction -> generation-unit mapping. Re-running is idempotent.

    python validation/issue8/make_inputs.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as experiment  # noqa: E402


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    manuscript_path = repo_root / "validation/issue8/original_manuscript.json"
    inputs_dir = repo_root / "validation/issue8/inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    manuscript = json.loads(manuscript_path.read_text(encoding="utf-8"))
    scripts = experiment.build_case_scripts(manuscript)

    for code in ("A", "B", "C", "D"):
        (inputs_dir / f"{code}.json").write_text(
            json.dumps(scripts[code], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    reference = experiment.concat_text(scripts["A"]["segments"])
    problems = experiment.plan_is_consistent(scripts, manuscript["max_segment_chars"])

    manifest = {
        "source": str(manuscript_path.relative_to(repo_root)),
        "generated_by": "make_inputs.py",
        "shared_concatenated_text": reference,
        "shared_text_sha256": hashlib.sha256(reference.encode("utf-8")).hexdigest(),
        "cases": {
            code: {
                "boundary": experiment.CASE_KIND[code][0],
                "instruct_policy": experiment.CASE_KIND[code][1],
                "unit_count": len(scripts[code]["segments"]),
                "instruct_present": bool(scripts[code]["instruct"].strip()),
            }
            for code in ("A", "B", "C", "D")
        },
        "junction_mapping": experiment.junction_mapping(manuscript, scripts),
        "consistency_problems": problems,
    }
    (inputs_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"shared text: {len(reference)} code points, SHA-256 "
          f"{manifest['shared_text_sha256']}")
    for code in ("A", "B", "C", "D"):
        boundary = experiment.CASE_KIND[code][0]
        instruct = "present" if scripts[code]["instruct"].strip() else "empty"
        print(f"  {code}: boundary={boundary:5s} instruct={instruct:7s} "
              f"units={len(scripts[code]['segments'])}")
    if problems:
        print("CONSISTENCY PROBLEMS:")
        for problem in problems:
            print("  - " + problem)
        return 1
    print("All four concatenated texts are character-identical and order-preserving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
