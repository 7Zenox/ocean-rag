# File: code/visualize_results.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/baseline_results.csv')

# Figure 1: RMSE Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['Corrupted\n(Baseline)', 'Baseline 1\n(Heuristic)', 'Baseline 2\n(Interpolation)']
rmse_mean = [
    df['Corrupted_RMSE'].mean(),
    df['Baseline1_RMSE'].mean(),
    df['Baseline2_RMSE'].mean()
]
rmse_std = [
    df['Corrupted_RMSE'].std(),
    df['Baseline1_RMSE'].std(),
    df['Baseline2_RMSE'].std()
]

colors = ['#d62728', '#ff7f0e', '#2ca02c']
bars = ax.bar(methods, rmse_mean, yerr=rmse_std, capsize=10, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

ax.set_ylabel('RMSE (meters)', fontsize=12, fontweight='bold')
ax.set_title('Baseline Method Comparison\nBathymetric Depth Correction Performance', fontsize=14, fontweight='bold')
ax.set_ylim(0, 20)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, rmse_mean)):
    ax.text(bar.get_x() + bar.get_width()/2, val + rmse_std[i] + 0.5, 
            f'{val:.1f}m', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('results/baseline_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved results/baseline_comparison.png")

# Figure 2: Before/After Correction (first tile)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

clean = np.load('benchmark_data/clean_tile_01.npy')
corrupted = np.load('benchmark_data/corrupted_tile_01.npy')

from baselines import BaselineCorrections
b1 = BaselineCorrections.heuristic_spike_removal(corrupted)

titles = ['Corrupted Input', 'Baseline 1\n(Heuristic)', 'Ground Truth']
data = [corrupted, b1, clean]

for ax, img, title in zip(axes, data, titles):
    im = ax.imshow(img, cmap='viridis', vmin=-5000, vmax=500)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Depth (m)')

plt.suptitle('Bathymetry Correction Example: Tile 01', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/correction_example.png', dpi=150, bbox_inches='tight')
print("✓ Saved results/correction_example.png")

# Summary table
summary = pd.DataFrame({
    'Metric': ['Mean RMSE (m)', 'Std Dev (m)', 'Improvement vs Corrupted'],
    'Corrupted': [
        f"{df['Corrupted_RMSE'].mean():.2f}",
        f"{df['Corrupted_RMSE'].std():.2f}",
        "—"
    ],
    'Baseline 1': [
        f"{df['Baseline1_RMSE'].mean():.2f}",
        f"{df['Baseline1_RMSE'].std():.2f}",
        f"{((df['Corrupted_RMSE'].mean() - df['Baseline1_RMSE'].mean()) / df['Corrupted_RMSE'].mean() * 100):.1f}%"
    ],
    'Baseline 2': [
        f"{df['Baseline2_RMSE'].mean():.2f}",
        f"{df['Baseline2_RMSE'].std():.2f}",
        f"{((df['Corrupted_RMSE'].mean() - df['Baseline2_RMSE'].mean()) / df['Corrupted_RMSE'].mean() * 100):.1f}%"
    ]
})

print("\n" + "="*80)
print("RESULTS SUMMARY TABLE")
print("="*80)
print(summary.to_string(index=False))
print("="*80)

summary.to_csv('results/summary_table.csv', index=False)