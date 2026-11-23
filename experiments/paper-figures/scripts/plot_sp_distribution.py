"""Output-distribution analysis of the GPT-OSS math prompt (Appendix G, experiments/interp).

KL reductions on teacher traces, decision-token probabilities and token counts as reported in the paper.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
out = Path(__file__).resolve().parents[1] / 'figures'
out.mkdir(parents=True, exist_ok=True)
blue, gray, dark, red = '#2878b5', '#a8b3bd', '#171717', '#d34b49'

fig, axes = plt.subplots(1, 3, figsize=(8.1, 3.0), layout='constrained',
                         gridspec_kw={'width_ratios': [1.25, 1.0, 1.2]})

# (a) Relative KL reduction.
ax = axes[0]
labels = ['Init.', 'Upd.\n40', 'Upd.\n180', 'Student\ntraces', 'Prompt.\ntraces']
vals = [2, 78, 86.8, 86.5, 86.5]
xs = [0, 1, 2, 3.4, 4.6]
cols = ['#c6dbec', '#79acd3', blue, blue, blue]
ax.bar(xs, vals, width=.72, color=cols)
for x, v in zip(xs, vals):
    ax.text(x, v + 2, f'{v:g}' if v != 86.8 else '86.8', ha='center', fontsize=7.2,
            fontweight='bold' if x == 2 else 'normal')
ax.axvline(2.7, color='#cccccc', lw=.7)
ax.text(1, 106, 'Teacher traces', ha='center', fontsize=7.5, style='italic')
ax.text(4.0, 106, 'Update 180', ha='center', fontsize=7.5, style='italic')
ax.set_xticks(xs, labels, fontsize=7.5)
ax.set(ylim=(0, 115), ylabel='KL gap closed, $G$ (%)')
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_title('(a) Distance to the teacher', fontsize=9)

# (b) Decision-token probability relative to the teacher (analysis channel, teacher traces).
ax = axes[1]
toks = ['So', 'But', '<|end|>']
S = [1.00e-2, 1.05e-3, 1.7e-4]
P = [0.76e-2, 1.80e-3, 1.3e-4]
T = [0.69e-2, 1.92e-3, 1.5e-4]
w = .36
for i in range(3):
    ax.bar(i - w / 2, S[i] / T[i], w, color=gray, label='Student' if i == 0 else None)
    ax.bar(i + w / 2, P[i] / T[i], w, color=blue, label='Student + SP' if i == 0 else None)
ax.axhline(1, color=dark, ls='--', lw=.8, alpha=.6, label='Teacher')
ax.set_xticks(range(3), toks, fontfamily='DejaVu Sans Mono', fontsize=7.5)
ax.set(ylim=(0, 1.75), ylabel='Probability relative to teacher')
ax.legend(frameon=False, fontsize=7, loc='upper right', handlelength=1.2)
ax.set_title('(b) Decision tokens', fontsize=9)

# (c) Tokens most raised / lowered by the prompt (counts among the 2,000 positions with largest / smallest change).
ax = axes[2]
up = [('?', 205), ('But', 199), ('Check', 107), ("Let's", 72), (' But', 60), ('Double', 41)]
down = [('.\\n\\n', 95), (' So', 71), ('Thus', 24), ('Given', 21), (' quickly', 16), ("That's", 15)]
names = [t for t, _ in up] + [t for t, _ in down]
vals = [c for _, c in up] + [-c for _, c in down]
ys = list(range(len(names)))[::-1]
ax.barh(ys, vals, color=[blue] * len(up) + [red] * len(down), height=.7)
for y, v in zip(ys, vals):
    ax.text(v + (4 if v > 0 else -4), y, str(abs(v)), va='center', ha='left' if v > 0 else 'right', fontsize=6.8)
ax.axvline(0, color=dark, lw=.7)
ax.set_yticks(ys, [n.replace(' ', '\u2423') for n in names], fontfamily='DejaVu Sans Mono', fontsize=7)
ax.set(xlim=(-130, 250), xlabel='Positions (lowered | raised)')
ax.set_xticks([-100, 0, 100, 200], ['100', '0', '100', '200'])
ax.set_title('(c) Most changed tokens', fontsize=9)

axes[0].grid(axis='y', alpha=.18); axes[1].grid(axis='y', alpha=.18)
fig.savefig(out / 'sp_distribution.pdf')
fig.savefig(out / 'sp_distribution.png', dpi=180)
