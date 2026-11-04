# Sapsucker Woods: evaluation-budget sensitivity

Kenneth Mwangi Kabiru — University of Leeds

## Question and status

Does increasing the unique-label budget from 25% to 30%, while retaining the same acquired training labels and frozen classifiers, take mean correct acceptance above 80%?

This is a **post hoc sensitivity analysis**, specified after inspecting the completed 25% comparison and discussing its audit affordability. It is not an untouched external replication or a replacement of the original result. The 30% result is unknown at specification time; the mathematical Towhee feasibility projection is already known. The executable settings are in `experiment.json`.

## Fixed population and models

Use the verified original Perch embeddings, common exclusion mask and labels for 22,052 five-second windows from the same 33 recordings. Retain the five selected species and their 20 original seeds each: American Goldfinch, Eastern Towhee, Red-winged Blackbird, Scarlet Tanager and Tufted Titmouse. All 100 histories enter every method average.

Each history retains its original 3,000 paid training labels, including startup, and its 2,205 flagged windows per ranker. LR-led acquisition, model preprocessing, fitted scores, ranking ties and the four fixed rankers remain unchanged: LR, standardized RBF SVM, equal rank blend and 3:1 LR/SVM rank blend. No species, seed, acquisition rule or model is selected from sensitivity outcomes. Adaptive stopping workflows are outside this comparison, which isolates evaluation capacity at the fixed endpoint.

## Budget arms and audit rule

| Quantity | Original arm | Sensitivity arm |
|---|---:|---:|
| Budget fraction | 25% | 30% |
| Maximum unique labels | floor(22,052/4) = 5,513 | floor(3 × 22,052/10) = 6,615 |
| Training labels | 3,000 | 3,000 |
| Flagged windows | 2,205 | 2,205 |
| Recall requirement | 90% | 90% |
| Single-attempt false-acceptance bound | 5% | 5% |

The extra allowance is 1,102 windows, approximately 20% of the original label ceiling. A window incurs one charge at its first label observation; later reuse is free. Expected cost includes training, the flagged census and actually inspected audit windows, including non-acceptance and early audit failure. This is a label-count proxy, not measured listening time.

For each budget arm separately, initialize the ledger with the same purchased training labels. Pay for the remaining flagged census. Using only these paid observations, calculate flagged positives T, known misses b, unseen population d, remaining sample allowance and a = floor(T/9) − b. Use the **existing allocator**: choose the largest feasible hit cutoff at the maximum affordable sample, then the smallest sample supporting that cutoff. Verify the null boundary and adjacent sample/cutoff choices with exact arithmetic at risk 1/20. Preserve the existing zero-positive, known-miss, census and unaffordability terminal rules.

Draw the audit in the original label-independent random order restricted to the unseen tail. Accept only after completing the planned sample within its cutoff; stop at cutoff + 1 hits on failure. The two budget arms are paired counterfactual workflows, each with one attempt. The 30% arm is not a retry after observing the 25% result. No hidden tail outcomes choose the audit design. A preservation policy for already affordable 25% audits is not part of this experiment. Therefore monotonic acceptance for every history is not assumed.

## Reproduction routes and mandatory baseline gate

The default route verifies and reuses purchased IDs and final LR/SVM scores from the completed results packet. It reruns ranking, census accounting, audit design, audit execution and analysis; it does not refit classifiers. The optional route refits the original acquisition histories from the cached embeddings using the pinned Python/NumPy/SciPy/scikit-learn versions. It independently reproduces classifier training conditional on the reused embeddings; neither route regenerates audio or Perch features.

Both routes require exact agreement with the original purchased-history hashes, complete final score hashes and flagged-set hashes. Every one of the 400 fixed-method 25% rows must reconcile, including integer designs, expected probabilities/costs and sampled verdicts. Independent combinatorial calculations check risk, sample/cutoff adjacency, costs and paid audit traces. Only after all 100 histories pass does the notebook permit 30% evaluation. Mismatches stop the analysis; they do not produce replacement histories or silently relax tolerances. Floating-point probability and cost reconciliation uses absolute tolerance 1e-10; IDs, counts and hashes require exact matches.

The supplied `reference_25.json` contains scientific checks from the completed 25% packet, explicitly labelled as reference outputs. It does not supply training labels or model-score arrays and cannot substitute for running either route. Its source archive checksum is retained for provenance. The original Drive cache remains read-only; experiment settings and new outputs live separately. Missing inputs stop with an actionable message, without starting audio retrieval or embedding inference.

## Endpoints, analysis and interpretation

**Primary workflow: fixed LR.** The primary endpoint is its equal-species-weight mean correct-acceptance probability at 30%. Each candidate's acceptance probability is calculated from its frozen audit design and evaluator-only remaining positive count, multiplied by retrospective adequacy. This is different from the observed proportion of audit traces that passed.

Report two distinct benchmark statements: whether the point estimate exceeds 0.80, and whether the lower endpoint of its approximate two-sided 95% interval exceeds 0.80. Report the paired 30%-minus-25% acceptance difference and expected-cost difference. The three other fixed rankers are secondary; do not switch the primary workflow based on results or reuse the original ten-point ensemble-promotion criterion as this analysis's gate.

Use 20,000 percentile bootstrap replicates, PCG64 seed 991250000. Resample the 20 histories independently within each species, preserving pairing across budgets and methods; average the five species equally. Use linear 2.5th and 97.5th percentiles. Retain the resampling indices and independently recompute all summaries. This quantifies acquisition randomization within the selected tasks and pool, not uncertainty over new species or recordings. Secondary intervals are descriptive and do not provide a family-wide superiority guarantee.

Report all per-species acceptance values, recall, audit feasibility and cutoffs; aggregate and per-species expected cost; planned and realized cost; and counts of histories whose acceptance improves, ties or worsens. Audit sample outcomes remain separate from calculated acceptance. A second independent implementation checks the saved statistics. No incomplete execution is exported as a completed comparison.

Exceeding 80% on average does not imply exceeding 80% for every species, 80% power for every classifier at 95% recall, or a solution to adaptive training-versus-evaluation allocation. No budget grid, additional seeds, margin search or model retuning is included. Any later exploration must remain separately labelled.

## Deliverables

The notebook produces 800 rows and traces (100 histories × four fixed rankers × two budgets), saved final history arrays, a 25% reconciliation receipt, exact audit verification, paired summaries and intervals, per-target results, a readable report and a checksum-manifested results ZIP. The export contains scientific inputs necessary to audit these outcomes, source and library-version provenance; it excludes session paths, installation logs, local errors and private development records. Original embeddings remain identified by their hashes and available through the existing cache or original results packet.
