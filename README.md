# Beyond Recall

Reproduction code and model outputs for **Beyond Recall: Budget-Constrained Certification of Active Learning with Acoustic Embeddings**.

The study asks whether a classifier that has reached high recall can also be verified within the same finite annotation budget used for training and review.
It compares logistic regression, a standardized RBF support-vector machine, two rank ensembles, and two training-stop signals on fixed Perch embeddings.

## Repository contents

- `reproduction/primary/` — Colab notebook, configuration, and helper modules
  for the prespecified 25% Sapsucker Woods evaluation.
- `reproduction/budget_sensitivity/` — code for the separately specified,
  post hoc 25%/30% evaluation-budget comparison.
- `model_outputs/` — final endpoint score arrays and the scientific summaries
  reported in the manuscript.
- `analysis/` — small scripts that regenerate manuscript figures and compact
  audit/budget tables from the included outputs.
- `tools/build_public_model_outputs.py` — reconstructs the compact public
  endpoint-score bundle from a completed primary run.

The repository intentionally does **not** contain original audio, embeddings,
embedding-cache validation reports, acquired-window identities, flagged-window
identities, audit traces, execution logs, or internal development records.

## Inspect the published scores

The 100 histories are ordered by target and seed as recorded in
`model_outputs/endpoint_scores_manifest.json`.

```python
import numpy as np

with np.load("model_outputs/endpoint_scores.npz", allow_pickle=False) as data:
    targets = data["target"]              # (100,)
    seeds = data["seed"]                  # (100,)
    lr_probability = data["lr_probability"]  # (100, 22052)
    lr_decision = data["lr_decision"]        # (100, 22052)
    svm_decision = data["svm_decision"]      # (100, 22052)
```

All score arrays retain the original `float64` values. Their row-level and
whole-array SHA-256 identities are recorded in the manifest.

## Regenerate manuscript summaries

From the repository root:

```bash
python -m pip install -r requirements-analysis.txt
python analysis/make_figures.py
python analysis/make_budget_table.py
python analysis/make_audit_table.py
```

Generated files are written to `generated/`. Figure generation requires NumPy
and Matplotlib; the table scripts use the Python standard library.

To verify the component manifests, published score arrays, release inventory,
and exclusion of local filesystem paths, run:

```bash
python tools/verify_release.py
```

## Rerun the study

1. Upload `reproduction/primary/` to Google Drive and follow its README and
   notebook. Fresh Perch embedding generation requires a GPU; downstream model
   fitting and analysis use the CPU. The source recordings and annotations are
   obtained from the public Sapsucker Woods v2 Zenodo record.
2. After completing the primary run, upload
   `reproduction/budget_sensitivity/` and follow its README. The sensitivity
   analysis uses a CPU runtime and either the regenerated primary packet or the
   regenerated embedding cache.

The numerical environments used in the study are declared in each
reproduction folder. Exact score hashes can depend on numerical libraries and
hardware; the included endpoint scores preserve the executed results.

## Data and release scope

Sapsucker Woods v2 is distributed separately under CC BY 4.0:
<https://doi.org/10.5281/zenodo.7079380>. This repository contains no copy of
the source audio or annotation tables.

The software in this repository is licensed under the GNU Affero General
Public License v3.0; see `LICENSE`.
