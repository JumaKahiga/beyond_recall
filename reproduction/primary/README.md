# Sapsucker Woods reproducibility

Kenneth Mwangi Kabiru — University of Leeds

Open **SSW_Reproducibility.ipynb**. The notebook contains the data-construction, model-fitting, acquisition, stopping, embedding and analysis code. Its helpers contain the longer audit mathematics, verification, storage and installation routines.

## Colab

Upload this entire **ssw** folder to Google Drive. Open the notebook in Colab, select a GPU runtime for fresh embedding generation, and set **PROJECT_DIRECTORY** to the folder you uploaded. Opening a notebook from Drive does not automatically mount its neighbouring files: the setup cell mounts Drive and locates the helpers using that one path. Keep the same runtime for the whole run.

Set **DRIVE_CACHE_PATH** to a persistent folder. Complete verified embeddings skip audio download and TensorFlow installation on future runs and can be consumed in a CPU runtime. Set **REGENERATE_EMBEDDINGS** only when you want to generate a separate set from original audio. The output states whether embeddings were regenerated or reused.

Completed recording downloads are also copied and verified under **DRIVE_CACHE_PATH/audio**. An interrupted download resumes from those completed files. To restore an earlier audio backup, set **AUDIO_BACKUP_DIRECTORY** to the folder containing its `BACKUP_MANIFEST.json`. This verifies and preserves the original 33-recording selection, then constructs labels under the current specification. An explicitly requested backup that is missing or incompatible stops the notebook; it does not silently start another download. Leave this field empty for a fresh independent run or automatic reuse of the recording cache.

Source annotation validation and window construction finish in section 10, before Perch installation or model loading. Inspect its construction report before continuing. These stages use the CPU even in a GPU runtime; no runtime switch is required. Existing embedding caches from a different construction policy are not reused.

The original dataset is [Sapsucker Woods v2](https://zenodo.org/records/7079380), under CC BY 4.0. If you already have the original **soundscape_data.zip**, set **ORIGINAL_ARCHIVE_PATH**; the complete size and release checksum are verified before use. An empty path requests the original archive entries from Zenodo. Server outages must be resolved by availability of those same bytes, not by selecting different recordings.

## Reproducibility

`study.json` specifies all seeds, sampling rules, models, audit settings and analysis. `models.json` pins the official Perch exports by identifier and file checksums. `dependencies.json` specifies the numerical environments. The notebook writes its visible scientific code into a separate run directory so it can execute under those pinned Python interpreters without changing the Colab kernel.

Protocol 1.1 excludes the complete five-second window containing each finite, nonnegative zero-duration annotation timestamp from the common pool for every species. At an exact boundary, the timestamp belongs to the window starting there; a timestamp in a discarded tail or at the recording end does not create a window. This rule scans the original annotation table, without a list of specific rows or recordings. Exclusions are deduplicated and combined with unknown-call exclusions. Source rows remain unchanged. The construction output retains all masks and reports overlapping versus additional exclusions, the remaining windows, budget, flagged-set size and target eligibility. Other invalid times still stop construction.

This folder is the primary-study reproduction code. The repository-level
`model_outputs` directory supplies publication-facing endpoint scores and
scientific summaries. Embeddings, acquired-window identities and audit traces
are not part of the public repository. A complete local rerun may create those
intermediate artifacts in its own output directory.

The interpretation is conditional on the selected targets and fixed recording pool. A cache-based classifier run does not independently rerun embedding inference. CPU/GPU regeneration can differ numerically despite matching checkpoint weights.

## Validation

Local synthetic checks and privacy checks accompany release preparation. Live Colab GPU execution and original-audio retrieval require the relevant hardware and source availability. No external experiment is implied by successful synthetic checks.

Keep run data and execution logs outside the source folder. Clear notebook
outputs before sharing an executed copy separately.
