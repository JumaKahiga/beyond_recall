#!/usr/bin/env python3
"""Build the public endpoint-score bundle from a completed primary run.

The bundle contains only final LR probabilities/decision values and final SVM
decision values. It deliberately excludes embeddings, labels, acquired-window
identities, flagged-window identities, and audit traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


SOURCE_ARCHIVE_SHA256 = (
    "2bee97bc4bc88d3a1645e338d861ec02e821eb1c2158c4428a8285b1ce4534f0"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Unpacked completed primary-results directory containing runs/.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.results_root.expanduser().resolve()
    study = load_json(root / "STUDY_MANIFEST.json")
    pool_size = int(study["pool"]["N"])
    horizon = int(study["constants"]["training_horizon"])
    rows: list[dict] = []
    lr_probabilities: list[np.ndarray] = []
    lr_decisions: list[np.ndarray] = []
    svm_decisions: list[np.ndarray] = []

    for target in study["targets"]:
        target_code = target["code"]
        for seed in target["seeds"]:
            run = root / "runs" / target_code / str(seed)
            checkpoint = load_json(run / "checkpoint.json")
            matches = [
                index
                for index, fit in enumerate(checkpoint["fits"])
                if int(fit["count"]) == horizon
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one horizon LR fit: {target_code}/{seed}")
            index = matches[0]
            snapshot_record = checkpoint["snapshot_files"][index]
            lr_path = run / snapshot_record["path"]
            if sha256(lr_path) != snapshot_record["sha256"]:
                raise RuntimeError(f"LR snapshot checksum mismatch: {target_code}/{seed}")
            lr = load_npz(lr_path)
            if int(lr["count"]) != horizon:
                raise RuntimeError(f"LR horizon mismatch: {target_code}/{seed}")
            probability = np.asarray(lr["probability"], dtype=np.float64)
            decision = np.asarray(lr["decision"], dtype=np.float64)
            if probability.shape != (pool_size,) or decision.shape != (pool_size,):
                raise RuntimeError(f"LR score shape mismatch: {target_code}/{seed}")
            expected_lr_hash = checkpoint["fits"][index]["score_sha256"]
            if array_sha256(probability) != expected_lr_hash:
                raise RuntimeError(f"LR score hash mismatch: {target_code}/{seed}")

            svm_path = run / "svm_3000.npz"
            svm_meta = load_json(run / "svm_3000.json")
            svm = load_npz(svm_path)
            svm_score = np.asarray(svm["scores"], dtype=np.float64)
            if svm_score.shape != (pool_size,):
                raise RuntimeError(f"SVM score shape mismatch: {target_code}/{seed}")
            if array_sha256(svm_score) != svm_meta["score_sha256"]:
                raise RuntimeError(f"SVM score hash mismatch: {target_code}/{seed}")

            lr_probabilities.append(probability)
            lr_decisions.append(decision)
            svm_decisions.append(svm_score)
            rows.append(
                {
                    "target": target_code,
                    "common_name": target["common_name"],
                    "seed": int(seed),
                    "training_count": horizon,
                    "lr_probability_sha256": array_sha256(probability),
                    "lr_decision_sha256": array_sha256(decision),
                    "svm_decision_sha256": array_sha256(svm_score),
                }
            )

    arrays = {
        "target": np.asarray([row["target"] for row in rows]),
        "seed": np.asarray([row["seed"] for row in rows], dtype=np.int64),
        "lr_probability": np.stack(lr_probabilities),
        "lr_decision": np.stack(lr_decisions),
        "svm_decision": np.stack(svm_decisions),
    }
    if len(rows) != 100:
        raise RuntimeError(f"Expected 100 histories, found {len(rows)}")
    if any(not np.isfinite(value).all() for key, value in arrays.items() if "decision" in key or "probability" in key):
        raise RuntimeError("Nonfinite model output")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, output)

    manifest = {
        "schema": "beyond_recall_endpoint_scores_v1",
        "description": (
            "Final 3,000-label LR probabilities/decision values and standardized "
            "RBF-SVM decision values for 100 Sapsucker Woods histories."
        ),
        "source_primary_results_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "bundle": {
            "filename": output.name,
            "sha256": sha256(output),
            "bytes": output.stat().st_size,
        },
        "history_count": len(rows),
        "pool_size": pool_size,
        "row_order": rows,
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": array_sha256(value),
            }
            for key, value in arrays.items()
        },
        "excluded": [
            "embeddings",
            "labels",
            "acquired-window identities",
            "flagged-window identities",
            "audit traces",
            "execution logs",
        ],
    }
    manifest_path = output.with_name("endpoint_scores_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "PUBLIC_MODEL_OUTPUTS_READY",
        f"histories={len(rows)}",
        f"pool={pool_size}",
        f"bytes={output.stat().st_size}",
        f"sha256={manifest['bundle']['sha256']}",
    )


if __name__ == "__main__":
    main()
