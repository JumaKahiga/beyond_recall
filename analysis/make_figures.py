"""Recreate the manuscript's vector figures from the included result summaries.
Requires numpy and matplotlib. No original audio, embeddings, or local paths needed.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'generated'
OUTPUT.mkdir(exist_ok=True)
with (ROOT / 'model_outputs/target_summary.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
TARGETS = ['amegfi', 'eastow', 'rewbla', 'scatan', 'tuftit']
NAMES = ['American Goldfinch', 'Eastern Towhee', 'Red-winged Blackbird',
         'Scarlet Tanager', 'Tufted Titmouse']
METHODS = ['fixed_lr', 'fixed_svm', 'fixed_equal_blend', 'fixed_3to1_blend']
LABELS = ['LR', 'SVM', 'Equal blend', '3:1 blend']
COLORS = ['#0072B2', '#D55E00', '#009E73', '#8B5199']
MARKERS = ['o', 'x', 's', '^']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
    'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 7,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.right': False,
    'axes.spines.top': False, 'axes.linewidth': .6})

def values(method, field):
    return np.array([float(next(r for r in rows if r['target'] == t and
                                r['workflow'] == method)[field]) for t in TARGETS])

def axes_style(ax):
    ax.set_yticks(range(5))
    ax.set_ylim(4.5, -.5)
    ax.grid(axis='x', color='#dddddd', linewidth=.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=.6)

fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.8), sharey=True)
for j, (field, title) in enumerate([
    ('mean_recall_available', '(a) Predictive recall'),
    ('mean_correct_accept_probability', '(b) Certification yield')]):
    ax = axes[j]
    for i, method in enumerate(METHODS):
        yy = np.arange(5)+(i-1.5)*.12
        ax.scatter(values(method, field)*100, yy, s=19, c=COLORS[i],
                   marker=MARKERS[i], linewidths=.8, label=LABELS[i], zorder=3)
    axes_style(ax)
    ax.set_title(title, loc='left', pad=9)
    ax.set_xlabel('Percent')
axes[0].set_xlim(97.8, 100.3)
axes[0].set_xticks([98, 99, 100])
axes[0].set_yticklabels(NAMES)
axes[1].set_xlim(-5, 105)
axes[1].set_xticks([0, 50, 100])
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
           bbox_to_anchor=(.60, -.008), ncol=4, frameon=False,
           handletextpad=.3, columnspacing=.8)
fig.subplots_adjust(left=.31, right=.985, top=.85, bottom=.23, wspace=.22)
fig.savefig(OUTPUT/'target_outcomes.pdf')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.8, 2.8))
for i, method in enumerate(METHODS):
    ax.scatter(values(method, 'mean_overlap_available'), np.arange(5)+(i-1.5)*.12,
               s=24, c=COLORS[i], marker=MARKERS[i], label=LABELS[i], linewidths=.8, zorder=3)
axes_style(ax)
ax.set_yticklabels(NAMES)
ax.set_xlim(1150, 2250)
ax.set_xticks([1200, 1500, 1800, 2100])
ax.set_xlabel('Training–flagged overlap (windows)')
sec = ax.secondary_xaxis('top', functions=(lambda v: 5205-v, lambda v: 5205-v))
sec.set_xlabel('Pre-audit cost (unique labels)', labelpad=6)
sec.set_xticks([4000, 3700, 3400, 3100])
sec.tick_params(length=2, width=.6)
fig.legend(*ax.get_legend_handles_labels(), loc='lower center',
           bbox_to_anchor=(.62, -.006), ncol=4, frameon=False,
           handletextpad=.3, columnspacing=.8)
fig.subplots_adjust(left=.31, right=.98, top=.79, bottom=.25)
fig.savefig(OUTPUT/'label_reuse.pdf')
plt.close(fig)
