# File: code/evaluate_all_tiles.py
import numpy as np
import pandas as pd
from metrics import BathymetryMetrics
from baselines import BaselineCorrections

results = []

for tile_id in range(1, 21):
    clean = np.load(f'benchmark_data/clean_tile_{tile_id:02d}.npy')
    corrupted = np.load(f'benchmark_data/corrupted_tile_{tile_id:02d}.npy')
    
    # Baselines
    b1 = BaselineCorrections.heuristic_spike_removal(corrupted)
    b2 = BaselineCorrections.interpolation_based(corrupted)
    
    # Metrics
    m_corrupted = BathymetryMetrics(clean, corrupted).rmse()
    m_b1 = BathymetryMetrics(clean, b1).rmse()
    m_b2 = BathymetryMetrics(clean, b2).rmse()
    
    results.append({
        'Tile': f'tile_{tile_id:02d}',
        'Corrupted_RMSE': m_corrupted,
        'Baseline1_RMSE': m_b1,
        'Baseline2_RMSE': m_b2,
    })

# Save results
df = pd.DataFrame(results)
df.to_csv('results/baseline_results.csv', index=False)

print("\n" + "="*60)
print("BASELINE EVALUATION RESULTS")
print("="*60)
print(df.to_string(index=False))
print("="*60)
print(f"Average Baseline 1 RMSE: {df['Baseline1_RMSE'].mean():.2f}m")
print(f"Average Baseline 2 RMSE: {df['Baseline2_RMSE'].mean():.2f}m")
print(f"Average Corrupted RMSE: {df['Corrupted_RMSE'].mean():.2f}m")
