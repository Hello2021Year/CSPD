"""Main-text interpretability figure for the GPT-OSS math soft prompt (Figure 5).

Summarizes the appendix distribution and attention-pathway analyses (experiments/interp).
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
out = Path(__file__).resolve().parents[1] / 'figures'
out.mkdir(parents=True, exist_ok=True)
blue, gray, dark, red = '#2878b5', '#a8b3bd', '#171717', '#d34b49'

fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.7), layout='constrained',
                         gridspec_kw={'width_ratios': [1.15, 1.0, 1.0]})


def hbars(ax, names, vals, colors, xmax, fmt='{:g}'):
    ys = list(range(len(names)))[::-1]
    ax.barh(ys, vals, color=colors, height=.66)
    for y, v in zip(ys, vals):
        ax.text(v + xmax * .02, y, fmt.format(v), va='center', fontsize=8, fontweight='bold' if y == len(names) - 1 else 'normal')
    ax.set_yticks(ys, names, fontsize=8)
    ax.set_xlim(0, xmax)
    ax.grid(axis='x', alpha=.18)
    ax.set_axisbelow(True)


# (a) Soft prompt vs. written instructions: share of the teacher-student KL gap closed.
ax = axes[0]
hbars(ax, ['Soft prompt', 'Verify instruction', 'Check instruction', 'Neutral instruction', 'High reasoning effort'],
      [86.8, 2.1, 1.6, 1.1, 0.0], [blue] + [gray] * 4, 112)
ax.set_xlabel('KL gap closed (%)')
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_title('(a) Soft prompt vs. written instructions', fontsize=9)

# (b) Tokens most raised / lowered by the soft prompt.
ax = axes[1]
up = [('?', 205), ('But', 199), ('Check', 107), ("Let's", 72)]
down = [('.\\n\\n', 95), ('␣So', 71), ('Thus', 24), ('Given', 21)]
names = [t for t, _ in up] + [t for t, _ in down]
vals = [c for _, c in up] + [-c for _, c in down]
ys = list(range(len(names)))[::-1]
ax.barh(ys, vals, color=[blue] * len(up) + [red] * len(down), height=.66)
for y, v in zip(ys, vals):
    ax.text(v + (5 if v > 0 else -5), y, str(abs(v)), va='center', ha='left' if v > 0 else 'right', fontsize=8)
ax.axvline(0, color=dark, lw=.7)
ax.set_yticks(ys, names, fontfamily='DejaVu Sans Mono', fontsize=8)
ax.set(xlim=(-140, 255), xlabel='Positions (lowered | raised)')
ax.set_xticks([-100, 0, 100, 200], ['100', '0', '100', '200'])
ax.set_title('(b) Most changed tokens', fontsize=9)

# (c) Attention pathways: share of the soft prompt's improvement kept.
ax = axes[2]
hbars(ax, ['Full', 'Response only', 'Input only', 'Blocked'], [100, 80, 33, 1],
      [blue, '#5b9bd0', '#a6c8e4', gray], 122)
ax.set_xlabel('Improvement kept (%)')
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_title('(c) Attention pathways', fontsize=9)

fig.savefig(out / 'sp_interp_main.pdf')
fig.savefig(out / 'sp_interp_main.png', dpi=180)
