"""Generate the manuscript budget table from verified, included numeric summaries."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'generated'
OUTPUT.mkdir(exist_ok=True)
data = json.loads((ROOT / 'model_outputs/budget_sensitivity.json').read_text())
names = {'fixed_lr': 'LR', 'fixed_svm': 'SVM', 'fixed_equal_blend': 'Equal blend',
         'fixed_3to1_blend': '3:1 blend'}
lines = [r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}\toprule',
         r'Ranker & $Y_{25}$ (\%) & $Y_{30}$ (\%) & $\Delta Y$ & $\mathbb{E}[C_{25}]$ & $\mathbb{E}[C_{30}]$\\\midrule']
for workflow, label in names.items():
    pair = [next(r for r in data['summary'] if r['workflow'] == workflow and
                 r['budget_percent'] == b) for b in (25, 30)]
    a, b = pair
    lines.append('%s & %.2f & %.2f & +%.2f & %s & %s\\\\' % (
        label, a['mean_acceptance'] * 100, b['mean_acceptance'] * 100,
        (b['mean_acceptance'] - a['mean_acceptance']) * 100,
        format(a['mean_expected_windows'], ',.2f'), format(b['mean_expected_windows'], ',.2f')))
lines.extend([r'\bottomrule', r'\end{tabular*}', ''])
(OUTPUT / 'budget_comparison.tex').write_text('\n'.join(lines))
