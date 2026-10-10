"""Build a wheel and source/hash manifest from a clean commit; publish nothing."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_VERSION = "1.27.0"


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def clean_revision() -> str:
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("commit or set aside working-tree changes before preparing a release")
    return git("rev-parse", "HEAD")


def source_version() -> str:
    tree = ast.parse((ROOT / "src/media_pipeline/__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return str(ast.literal_eval(node.value))
    raise ValueError("missing package version")


def prepare(output: Path) -> dict:
    revision = clean_revision()
    if importlib.metadata.version("hatchling") != BUILDER_VERSION:
        raise ValueError(f"use hatchling=={BUILDER_VERSION} in the build environment")
    version = source_version()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    epoch = git("show", "-s", "--format=%ct", revision)
    env = {**os.environ, "SOURCE_DATE_EPOCH": epoch}
    with tempfile.TemporaryDirectory(prefix="regrain-build-", dir=output) as temporary:
        subprocess.run(
            [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", temporary],
            cwd=ROOT, env=env, check=True,
        )
        wheels = list(Path(temporary).glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("expected exactly one wheel")
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            metadata_file = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            metadata = Parser().parsestr(archive.read(metadata_file).decode("utf-8"))
        if metadata["Name"] != "regrain" or metadata["Version"] != version:
            raise ValueError("distribution identity differs from the package source")
        if clean_revision() != revision:
            raise ValueError("source changed while building; rebuild from one clean commit")
        manifest = {
            "schema_version": 1,
            "project": "regrain",
            "version": version,
            "release_status": "candidate; owner ai-core acceptance required",
            "source_commit": revision,
            "source_tree": git("rev-parse", "HEAD^{tree}"),
            "source_date_epoch": int(epoch),
            "requires_python": metadata["Requires-Python"],
            "build_environment": {
                "python": platform.python_version(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("hatchling", "packaging", "pathspec", "pluggy", "trove-classifiers")
                },
            },
            "wheel": {
                "filename": wheel.name,
                "bytes": wheel.stat().st_size,
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            },
        }
        destination = output / wheel.name
        manifest_path = output / f"regrain-{version}-manifest.json"
        if destination.exists() or manifest_path.exists():
            raise ValueError("release files already exist; choose a fresh output directory")
        wheel.replace(destination)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist", help="fresh artifact directory")
    args = parser.parse_args()
    try:
        manifest = prepare(args.output)
    except (ValueError, OSError, subprocess.CalledProcessError, importlib.metadata.PackageNotFoundError) as exc:
        parser.exit(1, f"release preparation failed: {exc}\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
