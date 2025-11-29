# code/test_week2_pipeline.py

import os
import pandas as pd

REQUIRED_FILES = [
    "results/Corrupted_results.csv",
    "results/Baseline1_results.csv",
    "results/Baseline2_results.csv",
    "results/LLM_Baseline_results.csv",
    "results/LLM_Lite_results.csv",
    "results/LLM_Full_results.csv",
    "results/ablation_summary_rmse.csv",
    "results/ablation_barplot.png",
]

def main():
    print("=" * 70)
    print("HYDROGPT – WEEK 2 PIPELINE CHECK")
    print("=" * 70)

    all_ok = True
    for path in REQUIRED_FILES:
        if os.path.exists(path):
            print(f"✓ Found: {path}")
        else:
            print(f"✗ MISSING: {path}")
            all_ok = False

    if all_ok:
        print("\nAll required result files are present.\n")
        try:
            df = pd.read_csv("results/ablation_summary_rmse.csv")
            print("Ablation summary (RMSE by method):")
            print(df.to_string(index=False))

            # Simple sanity checks
            if "LLM_Full" in df["Method"].values and "Baseline1" in df["Method"].values:
                rmse_full = float(df[df["Method"] == "LLM_Full"]["mean"].values[0])
                rmse_baseline = float(df[df["Method"] == "Baseline1"]["mean"].values[0])
                print("\nSanity check:")
                print(f"- Baseline1 mean RMSE: {rmse_baseline:.2f} m")
                print(f"- LLM_Full mean RMSE: {rmse_full:.2f} m")
                if rmse_full < rmse_baseline:
                    print("✓ LLM_Full improves over Baseline1 (as expected).")
                else:
                    print("⚠ LLM_Full does NOT improve over Baseline1. Re-check logic.")
        except Exception as e:
            print("\n⚠ Could not read or interpret ablation_summary_rmse.csv:")
            print(e)
            all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print("✓✓✓ WEEK 2 PIPELINE LOOKS GOOD ✓✓✓")
    else:
        print("✗✗✗ WEEK 2 PIPELINE HAS ISSUES – FIX BEFORE MOVING ON ✗✗✗")
    print("=" * 70)


if __name__ == "__main__":
    main()