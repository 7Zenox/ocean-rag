# code/visualize_ablation.py
import pandas as pd
import matplotlib.pyplot as plt

summary = pd.read_csv("results/ablation_summary_rmse.csv")

methods = summary["Method"].tolist()
rmse = summary["mean"].tolist()
err = summary["std"].tolist()

colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#1f77b4"]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(methods, rmse, yerr=err, capsize=5, color=colors[:len(methods)], edgecolor="black")

ax.set_ylabel("RMSE (m)")
ax.set_title("Ablation Study: Bathymetry Correction Methods")

for b, v in zip(bars, rmse):
    ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.1f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("results/ablation_barplot.png", dpi=150)
print("✓ Saved results/ablation_barplot.png")
