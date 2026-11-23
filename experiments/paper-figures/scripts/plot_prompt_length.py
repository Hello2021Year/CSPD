"""GPT-OSS math prompt-length sweep (Appendix, configs/ablations/gpt-oss-prompt-length-*.yaml).

Best recovery over checkpoints, overall macro average, student 76.73%, teacher 90.47%.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'pdf.fonttype': 42})
out = Path(__file__).resolve().parents[1] / 'figures'
out.mkdir(parents=True, exist_ok=True)

K = [2, 8, 32, 256, 512]
rec = [51.23, 74.11, 90.01, 93.10, 87.83]
orange = '#2f6db3'

fig, ax = plt.subplots(figsize=(4.6, 2.4), layout='constrained')
ax.axhline(100, color='#555555', ls=(0, (4, 3)), lw=.8)
ax.text(560, 101.5, 'Teacher', fontsize=7.5, color='#555555', va='bottom', ha='right')
ax.plot(K, rec, color=orange, lw=1.4, alpha=.5, zorder=2)
for k, r in zip(K, rec):
    main = k == 256
    ax.plot(k, r, 'o', ms=7 if main else 5.2, mfc=orange, mec='white', mew=.8, zorder=3)
    ax.annotate(f'{r:.1f}', (k, r), xytext=(0, -14 if main else 8), textcoords='offset points', ha='center',
                fontsize=7.5, color='#333333', fontweight='bold' if main else 'normal')
ax.set_xscale('log', base=2)
ax.set_xticks(K, [str(k) for k in K])
ax.minorticks_off()
ax.set(xlim=(1.6, 650), ylim=(40, 112), xlabel='Prompt vectors', ylabel='Best recovery (%)')
ax.set_yticks([50, 75, 100])
ax.grid(axis='y', alpha=.18)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.savefig(out / 'gptoss_prompt_length.pdf')
fig.savefig(out / 'gptoss_prompt_length.png', dpi=200)
