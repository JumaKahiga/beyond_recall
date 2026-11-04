# Manuscript analysis scripts

These scripts regenerate the manuscript's compact figures and tables from the
published files in `model_outputs/`. They do not fit classifiers, access audio,
or read embeddings.

- `make_figures.py` writes two PDF figures.
- `make_budget_table.py` writes the paired budget comparison as LaTeX.
- `make_audit_table.py` writes the audit-design tables and a JSON summary.

All generated files are written to the repository-level `generated/`
directory, which is ignored by Git.

