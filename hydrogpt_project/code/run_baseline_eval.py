# code/run_baseline_eval.py
import numpy as np
from eval_core import evaluate_method_on_all_tiles
from baselines import BaselineCorrections

def baseline1_fn(corrupted, context):
    return BaselineCorrections.heuristic_spike_removal(corrupted)

def baseline2_fn(corrupted, context):
    return BaselineCorrections.interpolation_based(corrupted)

if __name__ == "__main__":
    from metrics import BathymetryMetrics
    import pandas as pd

    df_b1 = evaluate_method_on_all_tiles("Baseline1", baseline1_fn)
    df_b2 = evaluate_method_on_all_tiles("Baseline2", baseline2_fn)

    # Also compute the corrupted baseline
    rows = []
    for tile_id in range(1, 21):
        clean = np.load(f'benchmark_data/clean_tile_{tile_id:02d}.npy')
        corrupted = np.load(f'benchmark_data/corrupted_tile_{tile_id:02d}.npy')
        m = BathymetryMetrics(clean, corrupted)
        rows.append({
            "Tile": f"tile_{tile_id:02d}",
            "Method": "Corrupted",
            "RMSE": m.rmse(),
            "MAE": m.mae()
        })
    df_corr = pd.DataFrame(rows)
    df_corr.to_csv("results/Corrupted_results.csv", index=False)

    print("✓ Saved Baseline1_results.csv, Baseline2_results.csv, Corrupted_results.csv")
