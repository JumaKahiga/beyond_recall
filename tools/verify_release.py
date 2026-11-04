#!/usr/bin/env python3
"""Verify the public repository's manifests, model outputs, and release scope."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
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


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def governed_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        result[path.relative_to(root).as_posix()] = path
    return result


def verify_component_manifest(folder: Path) -> None:
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["files"]
    actual = governed_files(folder)
    actual.pop("manifest.json")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            f"Component inventory mismatch in {folder}: missing={missing}, extra={extra}"
        )
    for relative, digest in expected.items():
        if sha256(actual[relative]) != digest:
            raise RuntimeError(f"Component checksum mismatch: {folder / relative}")


def verify_scores() -> None:
    folder = ROOT / "model_outputs"
    manifest = json.loads(
        (folder / "endpoint_scores_manifest.json").read_text(encoding="utf-8")
    )
    bundle = folder / manifest["bundle"]["filename"]
    if bundle.stat().st_size != manifest["bundle"]["bytes"]:
        raise RuntimeError("Endpoint-score size mismatch")
    if sha256(bundle) != manifest["bundle"]["sha256"]:
        raise RuntimeError("Endpoint-score file checksum mismatch")
    with np.load(bundle, allow_pickle=False) as archive:
        if set(archive.files) != set(manifest["arrays"]):
            raise RuntimeError("Endpoint-score array inventory mismatch")
        arrays = {name: archive[name] for name in archive.files}
        for name, specification in manifest["arrays"].items():
            value = arrays[name]
            if list(value.shape) != specification["shape"]:
                raise RuntimeError(f"Endpoint-score shape mismatch: {name}")
            if str(value.dtype) != specification["dtype"]:
                raise RuntimeError(f"Endpoint-score dtype mismatch: {name}")
            if array_sha256(value) != specification["sha256"]:
                raise RuntimeError(f"Endpoint-score array checksum mismatch: {name}")
        rows = manifest["row_order"]
        if len(rows) != manifest["history_count"]:
            raise RuntimeError("Endpoint-score history count mismatch")
        for index, row in enumerate(rows):
            if str(arrays["target"][index]) != row["target"]:
                raise RuntimeError(f"Endpoint-score target mismatch at row {index}")
            if int(arrays["seed"][index]) != row["seed"]:
                raise RuntimeError(f"Endpoint-score seed mismatch at row {index}")
            for key in ("lr_probability", "lr_decision", "svm_decision"):
                if array_sha256(arrays[key][index]) != row[f"{key}_sha256"]:
                    raise RuntimeError(f"Endpoint-score row checksum mismatch: {index}/{key}")


def verify_public_outputs() -> None:
    folder = ROOT / "model_outputs"
    for path in folder.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    for path in folder.glob("*.csv"):
        with path.open(newline="", encoding="utf-8") as stream:
            list(csv.reader(stream))


def verify_scope() -> None:
    files = governed_files(ROOT)
    forbidden_file_terms = (
        "embedding_cache",
        "acquired_ids",
        "flagged_ids",
        "audit_trace",
        "execution_log",
    )
    for relative in files:
        normalized = relative.lower().replace("-", "_")
        if any(term in normalized for term in forbidden_file_terms):
            raise RuntimeError(f"Excluded artifact appears in release: {relative}")

    forbidden_text = ("/" + "Users/",)
    text_suffixes = {
        ".cff",
        ".csv",
        ".ipynb",
        ".json",
        ".md",
        ".py",
        ".txt",
    }
    for relative, path in files.items():
        if path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        for term in forbidden_text:
            if term.lower() in text.lower():
                raise RuntimeError(f"Local path in public file {relative}: {term}")


def verify_release_manifest() -> None:
    path = ROOT / "RELEASE_MANIFEST.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest["files"]
    actual = governed_files(ROOT)
    actual.pop("RELEASE_MANIFEST.json")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"Release inventory mismatch: missing={missing}, extra={extra}")
    for relative, record in expected.items():
        path = actual[relative]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Release manifest mismatch: {relative}")


def main() -> None:
    verify_component_manifest(ROOT / "reproduction" / "primary")
    verify_component_manifest(ROOT / "reproduction" / "budget_sensitivity")
    verify_scores()
    verify_public_outputs()
    verify_scope()
    verify_release_manifest()
    print("PUBLIC_RELEASE_VERIFIED")


if __name__ == "__main__":
    main()
