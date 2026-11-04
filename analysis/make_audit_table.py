"""Summarize retained audit designs; no model fitting or new audit sampling.

Run from any directory. Inputs are the slim per-history model-output rows in
``model_outputs``. Empty c and n denote an unaffordable audit, not a
zero-sized sample.
"""
from collections import Counter
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "generated"
TARGETS = {
    "amegfi": "American Goldfinch",
    "eastow": "Eastern Towhee",
    "rewbla": "Red-winged Blackbird",
    "scatan": "Scarlet Tanager",
    "tuftit": "Tufted Titmouse",
}
RANKERS = {"lr": "LR", "svm": "SVM", "equal_blend": "Equal blend", "3to1_blend": "3:1 blend"}


def main():
    OUTPUT.mkdir(exist_ok=True)
    with (ROOT / "model_outputs/audit_mechanism_rows.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 800
    keys = [(r["budget_percent"], r["target"], r["ranker"], r["seed"]) for r in rows]
    assert len(set(keys)) == len(rows)
    summaries = []
    lines = []
    for budget in (25, 30):
        if budget == 30:
            lines.append(r"\clearpage")
        lines.extend([
            r"\begin{table}[!htbp]",
            r"\caption{Audit designs at %d\%% (%s). Feasible counts are out of 20. Cutoff entries show $c\!:\!\text{number of histories}$; $n$ is the planned sample size. $C_0$ is mean pre-audit expenditure in unique labels.}" % (budget, "prespecified" if budget == 25 else "post hoc"),
            r"\label{tab:audit%d}\centering\small" % budget,
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrcrr@{}}\toprule",
            r"Ranker & Feasible & Cutoff distribution & $n$ range & Mean $C_0$\\\midrule",
        ])
        for target, name in TARGETS.items():
            lines.append(r"\multicolumn{5}{@{}l}{\textit{%s}}\\[2pt]" % name)
            for ranker, label in RANKERS.items():
                group = [r for r in rows if int(r["budget_percent"]) == budget and r["target"] == target and r["ranker"] == ranker]
                assert len(group) == 20
                feasible = [r for r in group if r["design_status"] == "AUDIT"]
                assert all(r["design_status"] in {"AUDIT", "AUDIT_UNAFFORDABLE"} for r in group)
                cutoff = Counter(int(r["c"]) for r in feasible)
                sizes = [int(r["n"]) for r in feasible]
                paid = [int(r["pre_attention"]) for r in group]
                mean = Decimal(sum(paid)) / Decimal(len(paid))
                mean_text = format(mean.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP), ",.1f")
                cutoff_text = "; ".join(f"{c}:{n}" for c, n in sorted(cutoff.items())) or "---"
                size_text = f"{min(sizes):,}--{max(sizes):,}" if sizes else "---"
                lines.append(f"{label} & {len(feasible)}/20 & {cutoff_text} & {size_text} & {mean_text}" + r"\\")
                summaries.append({"budget_percent": budget, "target": target, "ranker": ranker,
                                  "histories": 20, "feasible": len(feasible), "cutoff_counts": dict(sorted(cutoff.items())),
                                  "planned_n_min": min(sizes) if sizes else None,
                                  "planned_n_max": max(sizes) if sizes else None,
                                  "pre_audit_mean": float(mean), "pre_audit_min": min(paid), "pre_audit_max": max(paid)})
            lines.append(r"\addlinespace[4pt]")
        lines.extend([r"\bottomrule\end{tabular*}", r"\end{table}"])
    for target in TARGETS:
        for ranker in RANKERS:
            paired = [s for s in summaries if s["target"] == target and s["ranker"] == ranker]
            assert paired[0]["pre_audit_mean"] == paired[1]["pre_audit_mean"]
    (OUTPUT / "audit_mechanism.tex").write_text("\n".join(lines) + "\n")
    (OUTPUT / "audit_mechanism_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print("Wrote two audit-design tables from 800 retained rows.")


if __name__ == "__main__":
    main()
