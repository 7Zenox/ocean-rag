# File: code/test_pipeline.py
import numpy as np
from metrics import BathymetryMetrics
from baselines import BaselineCorrections

print("="*60)
print("FINAL SANITY CHECK")
print("="*60)

all_pass = True

# Check 1: Benchmark exists
try:
    for i in range(1, 21):
        clean = np.load(f'benchmark_data/clean_tile_{i:02d}.npy')
        corrupted = np.load(f'benchmark_data/corrupted_tile_{i:02d}.npy')
    print("✓ All 20 benchmark tiles loaded successfully")
except:
    print("✗ FAIL: Missing benchmark tiles")
    all_pass = False

# Check 2: Metrics work
try:
    m = BathymetryMetrics(clean, corrupted)
    rmse = m.rmse()
    mae = m.mae()
    assert rmse > 0 and mae > 0
    print(f"✓ Metrics working: RMSE={rmse:.2f}m, MAE={mae:.2f}m")
except:
    print("✗ FAIL: Metrics calculation")
    all_pass = False

# Check 3: Baselines work
try:
    b1 = BaselineCorrections.heuristic_spike_removal(corrupted)
    b2 = BaselineCorrections.interpolation_based(corrupted)
    print(f"✓ Baselines working: B1 RMSE={BathymetryMetrics(clean, b1).rmse():.2f}m, "
          f"B2 RMSE={BathymetryMetrics(clean, b2).rmse():.2f}m")
except:
    print("✗ FAIL: Baselines")
    all_pass = False

# Check 4: Results saved
import os
if os.path.exists('results/baseline_results.csv'):
    print("✓ Results CSV exists")
else:
    print("✗ FAIL: Results CSV missing")
    all_pass = False

if os.path.exists('results/baseline_comparison.png'):
    print("✓ Visualization exists")
else:
    print("✗ FAIL: Visualization missing")
    all_pass = False

print("="*60)
if all_pass:
    print("✓✓✓ ALL CHECKS PASSED ✓✓✓")
    print("You're ready to move to Phase 5: Ablation Study & LLM Integration")
else:
    print("✗✗✗ SOME CHECKS FAILED ✗✗✗")
    print("Debug and re-run before proceeding")
print("="*60)
