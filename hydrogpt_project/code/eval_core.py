# code/eval_core.py
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any
from metrics import BathymetryMetrics

def evaluate_method_on_all_tiles(
    method_name: str,
    correction_fn: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
    context: Dict[str, Any] = None,
    num_tiles: int = 20
) -> pd.DataFrame:
    """
    Runs a correction method on all tiles and returns a DataFrame with metrics.
    correction_fn(corrupted_tile, context) -> corrected_tile
    """
    rows = []
    context = context or {}

    for tile_id in range(1, num_tiles + 1):
        clean = np.load(f'benchmark_data/clean_tile_{tile_id:02d}.npy')
        corrupted = np.load(f'benchmark_data/corrupted_tile_{tile_id:02d}.npy')

        corrected = correction_fn(corrupted, context)
        m = BathymetryMetrics(clean, corrected)

        rows.append({
            "Tile": f"tile_{tile_id:02d}",
            "Method": method_name,
            "RMSE": m.rmse(),
            "MAE": m.mae()
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"results/{method_name}_results.csv", index=False)
    return df
