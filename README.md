# Beyond Recall

Reproduction code for **Beyond Recall: Budget-Constrained Certification of Active Learning with Acoustic Embeddings** — the Sapsucker Woods evaluation.

Kenneth Mwangi Kabiru — University of Leeds

Open **SSW_Reproducibility.ipynb**. The notebook contains the data-construction, model-fitting, acquisition, stopping, embedding and analysis code. Its helpers contain the longer audit mathematics, verification, storage and download routines.

## Colab

Upload this entire folder to Google Drive (for example as **ssw**). Open the notebook in Colab, select a GPU runtime for fresh embedding generation, and set **PROJECT_DIRECTORY** to the folder you uploaded. Opening a notebook from Drive does not automatically mount its neighbouring files: the setup cell mounts Drive and locates the helpers using that one path. Keep the same runtime for the whole run.

Set **DRIVE_CACHE_PATH** to a persistent folder. Complete verified embeddings skip audio download and TensorFlow inference on future runs and can be consumed in a CPU runtime. Set **REGENERATE_EMBEDDINGS** only when you want to generate a separate set from original audio. The output states whether embeddings were regenerated or reused.

Completed recording downloads are also copied and verified under **DRIVE_CACHE_PATH/audio**. An interrupted download resumes from those completed files. To restore an earlier audio backup, set **AUDIO_BACKUP_DIRECTORY** to the folder containing its `BACKUP_MANIFEST.json`. This verifies and preserves the original 33-recording selection, then constructs labels under the current specification. An explicitly requested backup that is missing or incompatible stops the notebook; it does not silently start another download. Leave this field empty for a fresh independent run or automatic reuse of the recording cache.

Source annotation validation and window construction finish in section 10, before the Perch model is downloaded or loaded. Inspect its construction report before continuing. These stages use the CPU even in a GPU runtime; no runtime switch is required. Existing embedding caches from a different construction policy are not reused.

## Data and model sources

The original dataset is [Sapsucker Woods v2](https://zenodo.org/records/7079380), under CC BY 4.0. The notebook downloads the annotation files and only the 33 selected recordings from the Zenodo archive; it never downloads the whole 27.8 GB archive. If you already have the original **soundscape_data.zip**, set **ORIGINAL_ARCHIVE_PATH**; the complete size and release checksum are verified before use. Server outages must be resolved by availability of those same bytes, not by selecting different recordings.

Perch v2 embeddings use the official Kaggle exports listed in `models.json` (GPU and CPU variants, pinned by identifier and file checksums). Both the Zenodo files and the Kaggle model archives are public and download without an account.

## Requirements

`requirements.txt` lists the packages needed for a run. Section 9 of the notebook installs them into the active kernel with `pip`; for a local run use `pip install -r requirements.txt` with Python 3.12 or newer.

The thesis results were produced in isolated interpreters with Python 3.9.13, NumPy 1.24.4, SciPy 1.9.1 and scikit-learn 1.1.1. This release runs the same code in a single environment with the versions pinned in `requirements.txt`. The exact-arithmetic audit calculations are unaffected by library versions; floating-point details of the LR and SVM fits can differ between versions, so a rerun can differ from the thesis values at that level. The executed versions are recorded in the run directory's `EXECUTION_MANIFEST.json`.

## Reproducibility

`study.json` specifies all seeds, sampling rules, models, audit settings and analysis. `models.json` pins the official Perch exports by identifier and file checksums. The notebook writes its visible scientific code into a separate run directory and executes each stage as a separate single-threaded process, so the kernel itself never runs the experiment.

Protocol 1.1 excludes the complete five-second window containing each finite, nonnegative zero-duration annotation timestamp from the common pool for every species. At an exact boundary, the timestamp belongs to the window starting there; a timestamp in a discarded tail or at the recording end does not create a window. This rule scans the original annotation table, without a list of specific rows or recordings. Exclusions are deduplicated and combined with unknown-call exclusions. Source rows remain unchanged. The construction output retains all masks and reports overlapping versus additional exclusions, the remaining windows, budget, flagged-set size and target eligibility. Other invalid times still stop construction.

Embeddings, acquired-window identities and audit traces are not part of this repository. The acquired-window identities and audit traces from the thesis run are available from the author on request. A complete rerun creates those intermediate artifacts in its own output directory, and section 13 exports the verified results as a ZIP.

The interpretation is conditional on the selected targets and fixed recording pool. A cache-based classifier run does not independently rerun embedding inference. CPU/GPU regeneration can differ numerically despite matching checkpoint weights.

## Validation

Live Colab GPU execution and original-audio retrieval require the relevant hardware and source availability. Keep run data and execution logs outside the source folder. Clear notebook outputs before sharing an executed copy separately.
