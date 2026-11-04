# Release validation

The experiment is specified and the notebook is ready for the proposed sensitivity run. **The real 30% SSW comparison has not been executed during release preparation.** No new 30% results are included in this folder.

Completed checks:

- All code cells and helper files compile. Notebook scientific-cell hashes match the released project specification. Execution counts and outputs are empty.
- The saved-history route reconstructed all 100 original histories and all 400 fixed-method 25% rows. Purchased-history hashes, complete final LR/SVM score hashes, flagged sets, integer audit designs, probabilities, expected costs and sampled verdicts reconciled with the completed reference packet.
- Independent combinatorial calculations verified every reconstructed baseline audit boundary, adjacent sample/cutoff choice, expected inspection cost and paid trace.
- The baseline stage resumed completed histories and evaluations successfully.
- The original results ZIP and an equivalent local fixture of the complete Drive snapshot were read and verified successfully. Missing inputs, incomplete snapshots and unsafe archive-relative paths were rejected. These checks did not modify the source inputs.
- Synthetic checks passed for equivalent hypergeometric formulas, expected early-failure cost, separate budget arms, paid ledgers, prevention of a second audit, and design independence from hidden tail outcomes.
- Synthetic classifier acquisition resumed after an interrupted checkpoint and reproduced the uninterrupted purchased order and final LR score hash.
- A synthetic 800-row paired analysis exercised the 20,000-replicate bootstrap, readable report, 80% benchmark fields and independent statistical checker. A deliberately changed summary was rejected.
- The scientific export allowlist was tested using synthetic outputs. Session paths and an unrelated state fixture were excluded.
- Every manifest-listed file in the original `ssw` release remains unchanged.

The numerical checks used Python 3.9.13, NumPy 1.24.4, SciPy 1.9.1, scikit-learn 1.1.1 and threadpoolctl 3.6.0 with one numerical thread. The public `tests/test_science.py` contains the synthetic scientific regression checks; its header explains how to run them against the notebook-generated source directory.

Limits: live Colab installation and Google Drive mounting were not exercised in this local validation. The complete 100-history refit route was not run; its acquisition/resume behavior was tested synthetically, and its identity checks block sensitivity evaluation if rebuilt scores differ. Embeddings and original audio were not regenerated. Numerical-library or hardware differences can affect exact score hashes; the saved-history route preserves the original classifiers directly. The executed 25% reconciliation validates the saved-history path and audit computations, not an independent refit of the original SSW models.
