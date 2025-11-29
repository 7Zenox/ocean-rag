# code/run_llm_ablation.py
from eval_core import evaluate_method_on_all_tiles
from llm_systems import LLMSystems

def baseline_fn(corrupted, context):
    return LLMSystems.system_baseline(corrupted, context)

def lite_fn(corrupted, context):
    return LLMSystems.system_lite_llm(corrupted, context)

def full_fn(corrupted, context):
    return LLMSystems.system_full_llm(corrupted, context)

if __name__ == "__main__":
    df_base = evaluate_method_on_all_tiles("LLM_Baseline", baseline_fn)
    df_lite = evaluate_method_on_all_tiles("LLM_Lite", lite_fn)
    df_full = evaluate_method_on_all_tiles("LLM_Full", full_fn)
    print("✓ Saved LLM_Baseline_results.csv, LLM_Lite_results.csv, LLM_Full_results.csv")
