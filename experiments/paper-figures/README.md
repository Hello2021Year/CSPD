# Regenerate figures from the manuscript data

Run from the repository root with NumPy and Matplotlib installed.

```bash
for script in experiments/paper-figures/scripts/plot_*.py; do
    python "$script"
done
python experiments/serving/scripts/plot_b200.py --output experiments/paper-figures/figures/b200-paper.pdf
```

The copied manuscript scripts write PDF and PNG files into `experiments/paper-figures/figures/`. They render the published aggregate values. The B200 rendering reads the archived CSV and recreates the metric curves and SP versus backbone deltas.
