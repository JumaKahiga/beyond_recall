# Sapsucker Woods budget sensitivity

Kenneth Mwangi Kabiru — University of Leeds

Open **SSW_Budget_Sensitivity.ipynb**. This separate, post hoc experiment compares a **25% versus 30% label budget**, retaining the existing data and fixed 3,000-label training histories. Acceptance above 80% is an outcome to measure, not an input setting. Read **EXPERIMENT.md** for the complete specification.

## Run in Colab

1. Copy this entire **ssw2** folder to Google Drive and open its notebook in Colab. A **CPU runtime** is sufficient for both routes; the notebook never downloads audio, installs Perch or regenerates embeddings. A GPU runtime can also be used but this analysis uses its CPU.
2. Set **PROJECT_DIRECTORY** to the uploaded folder and **DRIVE_CACHE_PATH** to the existing cache (normally `/content/drive/MyDrive/ssw_cache`). The paths use Colab configuration form fields.
3. Keep **RUN_MODE = saved_histories** for the fast route. The notebook looks for the original completed results ZIP in the cache's `result_exports` folder. If you moved it, set **RESULTS_PATH** to its full Drive path. The expected filename begins `ssw_results_2bee97bc4bc88d3a` and ends `.zip`. An unpacked original `ssw_results` directory is also accepted. Uploaded code in that packet is never executed.
4. Alternatively choose **rebuild_from_embeddings** to refit the original LR acquisition histories and SVM endpoints. This route needs the complete original embedding/label cache but does not require the original results ZIP. It costs CPU time and must reproduce the original score hashes before continuing. Neither route requires another Zenodo download.
5. Leave **OUTPUT_DIRECTORY** as a separate folder (normally `/content/drive/MyDrive/ssw2_outputs`). Run the cells in order: verify inputs, reconcile every 25% history, evaluate 30%, independently check results, display the report and export.

If the complete Drive snapshot is unavailable, a supplied original results packet can provide the same verified embeddings and labels. If neither source is usable the notebook stops with a missing-input message. It does not switch datasets or recreate the expensive embedding stage. The original `ssw` project and cache are unchanged.

The numerical interpreter is installed separately using checksum-pinned micromamba; Colab's system `venv` and `ensurepip` are not required. The installed interpreter is Python 3.9.13 with the original numerical-library versions. Dependency installation requires internet access; audio/model retrieval does not occur.

## What runs independently

The notebook shows acquisition/model code, candidate evaluation, the two-budget execution and analysis in visible cells. Longer audit mathematics, exact checks, input storage and environment setup live in neighbouring `helpers` files. Both routes reconstruct the 25% audit results before the 30% stage. The fast route explicitly reuses existing model scores; the refit route reproduces training from reused embeddings. `reference_25.json` is labelled validation evidence from the completed experiment, not hidden input data.

Outputs and checkpoints are separated by experiment hash and run mode. Completed histories and audit evaluations resume on rerun; a rebuilt acquisition also checkpoints after each LR fit. Do not run two sessions against the same output folder simultaneously. If you intentionally change the code, use a separately versioned project/output location rather than editing a running experiment.

## Interpretation and outputs

LR is the main test of the 80% benchmark. All four fixed rankers, both budgets and every species are reported. The existing allocator is used at each budget; the earlier arithmetic projection that preserved already affordable designs is not substituted for execution. Point estimates and interval lower bounds above 80% are reported separately. This is exploratory evidence conditional on the existing five tasks and pool, alongside the unchanged original 25% result.

The repository-level `model_outputs` directory contains the publication-facing
scores and summaries. The notebook may generate additional histories, labels
and audit traces during a local rerun; those intermediate artifacts are not
part of this repository. Clear notebook outputs before sharing an executed
copy separately.

See **VALIDATION.md** for checks performed during release preparation and their limits. No 30% empirical result is bundled with this code release.
