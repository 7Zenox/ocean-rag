# code/ablation_summary.py
import pandas as pd

# Include all methods, including the real LLM
methods = [
    "Corrupted",
    "Baseline1",
    "Baseline2",
    "LLM_Baseline",
    "LLM_Lite",
    "LLM_Full",
    "LLM_Real",
]

dfs = []

for m in methods:
    df = pd.read_csv(f"results/{m}_results.csv")
    df["Method"] = m
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
all_df.to_csv("results/all_methods_results.csv", index=False)

summary = all_df.groupby("Method")["RMSE"].agg(["mean", "std"]).reset_index()
summary = summary.sort_values("mean")

print("\nABLATON SUMMARY (RMSE):")
print(summary.to_string(index=False))

summary.to_csv("results/ablation_summary_rmse.csv", index=False)
