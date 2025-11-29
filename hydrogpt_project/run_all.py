import subprocess
import sys
import os

CMD_PREFIX = [sys.executable]  # uses current venv’s python


def run(step_name, cmd):
    print("\n" + "=" * 70)
    print(f"STEP: {step_name}")
    print("=" * 70)
    print("Command:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n✗ Step FAILED: {step_name}")
        sys.exit(res.returncode)
    else:
        print(f"\n✓ Step COMPLETED: {step_name}")


def main():
    # Ensure we are in project root
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    steps = [
        ("Load GEBCO & preview",
         CMD_PREFIX + ["code/load_data.py"]),
        ("Create clean tiles",
         CMD_PREFIX + ["code/create_tiles.py"]),
        ("Corrupt tiles",
         CMD_PREFIX + ["code/corrupt_tiles.py"]),
        ("Run baseline eval (Corrupted, Baseline1, Baseline2)",
         CMD_PREFIX + ["code/run_baseline_eval.py"]),
        ("Run LLM ablation (LLM_Baseline, LLM_Lite, LLM_Full)",
         CMD_PREFIX + ["code/run_llm_ablation.py"]),
        ("Aggregate ablation summary",
         CMD_PREFIX + ["code/ablation_summary.py"]),
        ("Visualize baselines",
         CMD_PREFIX + ["code/visualize_results.py"]),
        ("Visualize ablation",
         CMD_PREFIX + ["code/visualize_ablation.py"]),
        ("Week 2 pipeline check",
         CMD_PREFIX + ["code/test_week2_pipeline.py"]),
    ]

    for name, cmd in steps:
        run(name, cmd)

    print("\n" + "=" * 70)
    print("ALL STEPS COMPLETED SUCCESSFULLY")
    print("Artifacts:")
    print("- benchmark_data/: clean & corrupted tiles")
    print("- results/baseline_results.csv, Corrupted_results.csv, Baseline1/2_results.csv")
    print("- results/all_methods_results.csv, ablation_summary_rmse.csv")
    print("- results/baseline_comparison.png, correction_example.png, ablation_barplot.png")
    print("=" * 70)


if __name__ == "__main__":
    main()