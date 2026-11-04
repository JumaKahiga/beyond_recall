#!/usr/bin/env python3
"""Write the repository-wide public release manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "RELEASE_MANIFEST.json"
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "generated",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


files = {}
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
        continue
    if path == OUTPUT or path.suffix in {".pyc", ".pyo"}:
        continue
    files[path.relative_to(ROOT).as_posix()] = {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }

manifest = {
    "schema": "beyond_recall_public_release_manifest_v1",
    "files": files,
    "file_count": len(files),
}
OUTPUT.write_text(
    json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(f"RELEASE_MANIFEST_WRITTEN files={len(files)} sha256={sha256(OUTPUT)}")
