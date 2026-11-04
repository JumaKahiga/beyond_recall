# Model outputs

This directory contains publication-facing model outputs and scientific
summaries. It excludes intermediate research records.

## Endpoint scores

`endpoint_scores.npz` contains five arrays:

| Array | Shape | Meaning |
|---|---:|---|
| `target` | `(100,)` | eBird target code for each history |
| `seed` | `(100,)` | acquisition seed for each history |
| `lr_probability` | `(100, 22052)` | final LR positive-class probabilities |
| `lr_decision` | `(100, 22052)` | final LR log-odds decision values |
| `svm_decision` | `(100, 22052)` | final standardized RBF-SVM decision values |

Each model was fitted after 3,000 purchased labels. Score columns follow the
fixed pool order produced by the primary reproduction code. The complete row
order, dtypes, shapes, and hashes are recorded in
`endpoint_scores_manifest.json`.

## Scientific summaries

- `primary_history_metrics.json` contains 800 slim per-history workflow rows.
- `primary_summary_aggregate.json` and `primary_summary_by_target.json` contain
  the prespecified 25% summaries.
- `primary_paired_results.json` and `primary_independent_statistics.json`
  contain paired comparisons and their independent numerical recheck.
- `primary_trigger_summary.json` records the two stopping-trigger outcomes.
- `primary_decision.json` and `primary_verification.json` record the primary
  decision and verification summary.
- `budget_sensitivity.json` contains the paired 25%/30% sensitivity summaries.
- `budget_sensitivity_verification.json` records the independent recheck of all
  800 sensitivity rows.
- `target_summary.csv` and `audit_mechanism_rows.csv` are compact inputs to the
  manuscript figure and table scripts.

No file in this directory contains embeddings, labels, acquired-window IDs,
flagged-window IDs, or audit traces.

