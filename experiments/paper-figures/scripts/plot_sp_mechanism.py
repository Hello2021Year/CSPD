"""Instruction and attention-pathway analyses of the GPT-OSS math prompt (Appendix G).

Reported point estimates. The denominators and trace sets differ between panels.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(8.1, 2.8), layout='constrained')
series = [
    (['Soft prompt', 'Verify instruction', 'Check instruction', 'Neutral instruction', 'High reasoning effort'],
     [86.8, 2.1, 1.6, 1.1, 0.0], '(a) Soft prompt vs. written instructions', 'KL gap closed, $G$ (%)',
     ['#2878b5'] + ['#a8b3bd'] * 4),
    (['Full', 'Response only', 'Input only', 'Blocked'],
     [100, 80, 33, 1], '(b) Attention pathways', 'Improvement kept, $K$ (%)',
     ['#2878b5', '#5b9bd0', '#a6c8e4', '#a8b3bd'])]
for ax, (names, values, title, xlabel, cols) in zip(axes, series):
    ax.barh(range(len(names)), values, height=.6, color=cols, zorder=2)
    ax.set_yticks(range(len(names)), names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 113)
    ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9)
    ax.grid(axis='x', alpha=.17, zorder=0)
    for i, value in enumerate(values):
        ax.text(value+1.5, i, f'{value:g}', va='center', fontsize=8, fontweight='bold' if i == 0 else 'normal')
out = Path(__file__).resolve().parents[1] / 'figures'
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'sp_mechanism.pdf')
fig.savefig(out / 'sp_mechanism.png', dpi=180)
