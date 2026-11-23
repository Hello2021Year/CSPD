"""Vector layouts for Appendix A. Each row is one chat turn."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"pdf.fonttype": 42})
fig, axes = plt.subplots(2, 1, figsize=(5.65, 2.40))
fig.subplots_adjust(left=.003, right=.997, top=.995, bottom=.005, hspace=.07)


def cell(ax, x, y, width, label, fill, color="#243544", dashed=False):
    ax.add_patch(Rectangle((x, y), width, .29, facecolor=fill,
                           edgecolor="#8497a5" if dashed else "white",
                           linewidth=.9 if dashed else .7, zorder=2 if dashed else 1,
                           linestyle="--" if dashed else "-"))
    ax.text(x+width/2, y+.145, label, va="center", ha="center",
            fontsize=9, color=color)

for ax, training in zip(axes, (True, False)):
    ax.set(xlim=(0, 4), ylim=(0, 1.26))
    ax.axis("off")
    ax.text(.02, 1.24, "Training" if training else "Inference",
            va="top", fontsize=10, fontweight="bold", color="#244767")
    for row, role in enumerate(["system", "user", "assistant"]):
        y = .68 - row*.33
        cell(ax, 0, y, 1.00, "<|im_start|>"+role, "#edf0f3")
        if role == "system":
            if training:
                cell(ax, 1.00, y, .90, "Optional context $F$", "#fcf2df", dashed=True)
                cell(ax, 1.90, y, .77, "Soft prompt $P$", "#dceafa")
            else:
                for i, label in enumerate(["$P_1$", "$P_2$", "$\\cdots$", "$P_k$"]):
                    cell(ax, 1.00+i*.4175, y, .4175, label, "#dceafa")
            cell(ax, 2.67, y, .67, "System text $s$", "#f4f5f6")
        elif role == "user":
            cell(ax, 1.00, y, 2.34, "Question $q$", "#f4f5f6")
        elif training:
            cell(ax, 1.00, y, 3.00, "Teacher response $y$", "#edf4ee")
        else:
            cell(ax, 1.00, y, 3.00, "Generate response →", "#fafbfc", dashed=True)
        if role != "assistant":
            cell(ax, 3.34, y, .66, "<|im_end|>", "#edf0f3")

out = Path(__file__).resolve().parents[1]/"figures"
fig.savefig(out/"pfx_template.pdf")
fig.savefig(out/"pfx_template.png", dpi=180)
