# code/run_llm_ablation.py
import numpy as np
from eval_core import evaluate_method_on_all_tiles
from llm_systems import LLMSystems


def baseline_fn(corrupted, context):
    return LLMSystems.system_baseline(corrupted, context)


def lite_fn(corrupted, context):
    return LLMSystems.system_lite_llm(corrupted, context)


def full_fn(corrupted, context):
    return LLMSystems.system_full_llm(corrupted, context)


def real_fn(corrupted, context):
    return LLMSystems.system_real_llm(corrupted, context)


if __name__ == "__main__":
    # We want tile_id available in context; easiest is to wrap evaluate_method
    def eval_with_ids(name, fn):
        rows = []
        for tile_id in range(1, 21):
            clean = np.load(f"benchmark_data/clean_tile_{tile_id:02d}.npy")
            corrupted = np.load(f"benchmark_data/corrupted_tile_{tile_id:02d}.npy")

            ctx = {"tile_id": tile_id}
            corrected = fn(corrupted, ctx)

            from metrics import BathymetryMetrics
            m = BathymetryMetrics(clean, corrected)
            rows.append({
                "Tile": f"tile_{tile_id:02d}",
                "Method": name,
                "RMSE": m.rmse(),
                "MAE": m.mae(),
            })

        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(f"results/{name}_results.csv", index=False)
        return df

    df_base = eval_with_ids("LLM_Baseline", baseline_fn)
    df_lite = eval_with_ids("LLM_Lite", lite_fn)
    df_full = eval_with_ids("LLM_Full", full_fn)
    df_real = eval_with_ids("LLM_Real", real_fn)

    print("✓ Saved LLM_Baseline_results.csv, LLM_Lite_results.csv, LLM_Full_results.csv, LLM_Real_results.csv")
