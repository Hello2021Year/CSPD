"""GPT-OSS math recovery during training (Appendix, configs/ablations/gpt-oss-training-curve.yaml).

Earlier untuned AdamW run (constant LR 0.01, 2 epochs), student 77.50%, teacher 90.63%.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'pdf.fonttype': 42})
out = Path(__file__).resolve().parents[1] / 'figures'
out.mkdir(parents=True, exist_ok=True)

upd = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 226]
rec = [-4.46, 10.90, 49.61, 73.71, 81.79, 83.11, 86.26, 86.82, 90.54, 92.62, 87.87, 89.19, 85.68]
blue = '#2f6db3'

fig, ax = plt.subplots(figsize=(4.6, 2.5), layout='constrained')
ax.axhline(100, color='#555555', ls=(0, (4, 3)), lw=.8)
ax.axhline(0, color='#999999', ls=(0, (1, 2)), lw=.8)
ax.text(232, 101.5, 'Teacher', fontsize=7.5, color='#555555', va='bottom', ha='right')
ax.text(232, 1.5, 'Student', fontsize=7.5, color='#888888', va='bottom', ha='right')
# Two epochs over 226 updates: epoch 1 ends at update 113.
ax.axvline(113, color='#bbbbbb', lw=.8, ls='-', zorder=1)
ax.text(56.5, -9, 'Epoch 1', ha='center', fontsize=7.5, color='#777777')
ax.text(169.5, -9, 'Epoch 2', ha='center', fontsize=7.5, color='#777777')
ax.fill_between(upd, rec, 0, where=[r > 0 for r in rec], color=blue, alpha=.08, lw=0, interpolate=True)
ax.plot(upd, rec, color=blue, lw=1.6, marker='o', ms=3.6, mec='white', mew=.6, zorder=3)
i = rec.index(max(rec))
ax.plot(upd[i], rec[i], 'o', ms=6.5, mfc=blue, mec='white', mew=.8, zorder=4)
ax.annotate(f'Peak {rec[i]:.1f}% (update {upd[i]})', (upd[i], rec[i]), xytext=(-12, -40), textcoords='offset points',
            ha='center', fontsize=7.5, color=blue, arrowprops=dict(arrowstyle='-', color=blue, lw=.6))
ax.annotate(f'Final {rec[-1]:.1f}%', (upd[-1], rec[-1]), xytext=(2, -14), textcoords='offset points',
            ha='right', fontsize=7.5, color='#333333')
ax.set(xlim=(-5, 235), ylim=(-12, 112), xlabel='Optimizer update', ylabel='Recovery (%)')
ax.set_xticks(range(0, 221, 40)); ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis='y', alpha=.18)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.savefig(out / 'gptoss_training_curve.pdf')
fig.savefig(out / 'gptoss_training_curve.png', dpi=200)
