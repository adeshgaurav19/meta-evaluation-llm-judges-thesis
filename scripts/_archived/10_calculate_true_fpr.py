"""
Calculate True False Positive Rate (True FPR) by verifying if the clean 
supporting passages were actually overwritten by the injected poison.

Usage:
    python scripts/10_calculate_true_fpr.py
"""

import json
import pandas as pd
from pathlib import Path

def load_datasets(data_dir: Path) -> dict:
    """Loads all JSON files in the data directory and maps id -> triplet data."""
    triplets = {}
    for path in data_dir.rglob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    triplets[item["id"]] = item
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return triplets

def determine_if_support_poisoned(triplet: dict) -> bool:
    """
    Checks if any of the ground-truth supporting passages were 
    overwritten by the poison injection.
    """
    if not triplet.get("is_poisoned", False):
        return False
        
    original_context = triplet.get("original_context", "")
    passages = original_context.split("\n\n")
    
    ground_truth_str = triplet.get("ground_truth", "{}")
    try:
        gt = json.loads(ground_truth_str)
        titles = gt.get("supporting_titles", [])
    except:
        titles = []
        
    supp_indices = set()
    for title in titles:
        for i, p in enumerate(passages):
            # Match the title within the passage text
            if title.lower() in p.lower():
                supp_indices.add(i)
                
    poisoned_indices = set(triplet.get("poisoned_passage_indices", []))
    
    # If the intersection is not empty, at least one supporting passage was destroyed
    return len(supp_indices.intersection(poisoned_indices)) > 0

def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "v2_fixed_poisonedrag"
    scores_dir = base_dir / "results" / "v2" / "raw_scores"
    
    print("Loading datasets to map supporting passages...")
    triplets = load_datasets(data_dir)
    print(f"Loaded {len(triplets)} unique triplets from dataset.")
    
    # Pre-calculate support poisoned flag
    support_poisoned_map = {}
    for tid, t in triplets.items():
        support_poisoned_map[tid] = determine_if_support_poisoned(t)
        
    # Load judge scores
    conditions = ["baseline", "postfilter", "postfilter_llm"]
    judges = ["gpt-5.4-nano", "gemma-4-26b-a4b-it", "deepseek-chat"]
    file_judges = ["gpt", "gemini", "deepseek"] # Mapped to file names
    results = []
    
    for condition in conditions:
        for file_judge in file_judges:
            if condition == "baseline":
                score_file = scores_dir / f"baseline_{file_judge}.json"
            elif condition == "postfilter":
                score_file = scores_dir / f"postfilter_{file_judge}.json"
            elif condition == "postfilter_llm":
                score_file = scores_dir / f"postfilter_llm_{file_judge}.json"
                
            if not score_file.exists():
                print(f"Warning: {score_file} not found.")
                continue
                
            with open(score_file, "r", encoding="utf-8") as f:
                scores = json.load(f)
                
            for score in scores:
                tid = score.get("triplet_id") or score.get("id")
                if tid not in triplets:
                    continue
                    
                is_poisoned = score.get("is_poisoned", False)
                    
                faithfulness = score.get("faithfulness", score.get("faithfulness_score"))
                context_rel = score.get("context_relevance", score.get("context_relevance_score"))
                answer_rel = score.get("answer_relevance", score.get("answer_relevance_score"))
                
                if faithfulness is None or context_rel is None or answer_rel is None:
                    continue
                    
                is_support_poisoned = support_poisoned_map.get(tid, False)
                
                results.append({
                    "Condition": condition,
                    "Judge": file_judge.upper(),
                    "Injection Type": score.get("injection_type"),
                    "Noise Level": score.get("noise_level"),
                    "Is Poisoned": is_poisoned,
                    "Support Poisoned": is_support_poisoned,
                    "Faithfulness": float(faithfulness),
                    "Context Relevance": float(context_rel),
                    "Answer Relevance": float(answer_rel)
                })
            
    if not results:
        print("No score data found.")
        return
        
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("OVERALL MEAN SCORES COMPARISON (Standard vs True Support Poisoned)")
    print("="*80)
    
    for condition in conditions:
        print(f"\n[[ CONDITION: {condition.upper()} ]]")
        for judge in file_judges:
            judge_df = df[(df["Judge"] == judge.upper()) & (df["Condition"] == condition)]
            if judge_df.empty: continue
            
            clean_df = judge_df[judge_df["Is Poisoned"] == False]
            clean_f = clean_df["Faithfulness"].mean() if not clean_df.empty else float('nan')
            clean_c = clean_df["Context Relevance"].mean() if not clean_df.empty else float('nan')
            clean_a = clean_df["Answer Relevance"].mean() if not clean_df.empty else float('nan')
            
            poisoned_df = judge_df[judge_df["Is Poisoned"] == True]
            std_f = poisoned_df["Faithfulness"].mean() if not poisoned_df.empty else float('nan')
            std_c = poisoned_df["Context Relevance"].mean() if not poisoned_df.empty else float('nan')
            std_a = poisoned_df["Answer Relevance"].mean() if not poisoned_df.empty else float('nan')
            
            true_df = poisoned_df[poisoned_df["Support Poisoned"] == True]
            true_f = true_df["Faithfulness"].mean() if not true_df.empty else float('nan')
            true_c = true_df["Context Relevance"].mean() if not true_df.empty else float('nan')
            true_a = true_df["Answer Relevance"].mean() if not true_df.empty else float('nan')
            
            surv_df = poisoned_df[poisoned_df["Support Poisoned"] == False]
            surv_f = surv_df["Faithfulness"].mean() if not surv_df.empty else float('nan')
            surv_c = surv_df["Context Relevance"].mean() if not surv_df.empty else float('nan')
            surv_a = surv_df["Answer Relevance"].mean() if not surv_df.empty else float('nan')
            
            print(f"\n  {judge.upper()}:")
            print(f"    Clean (Unpoisoned Triplets, N={len(clean_df)}):")
            print(f"      Faithfulness: {clean_f:.3f} | Context Rel: {clean_c:.3f} | Answer Rel: {clean_a:.3f}")
            print(f"    Standard (All 'Poisoned' Triplets, N={len(poisoned_df)}):")
            print(f"      Faithfulness: {std_f:.3f} | Context Rel: {std_c:.3f} | Answer Rel: {std_a:.3f}")
            print(f"    True (Supporting Passages Killed, N={len(true_df)}):")
            print(f"      Faithfulness: {true_f:.3f} | Context Rel: {true_c:.3f} | Answer Rel: {true_a:.3f}")
            print(f"    Survived (Clean Support Remained, N={len(surv_df)}):")
            print(f"      Faithfulness: {surv_f:.3f} | Context Rel: {surv_c:.3f} | Answer Rel: {surv_a:.3f}")
        
    print("\n" + "="*80)
    print("TRUE MEAN FAITHFULNESS BY NOISE LEVEL (Validating the clean context theory)")
    print("="*80)
    noise_levels = sorted(df["Noise Level"].dropna().unique())
    for condition in conditions:
        print(f"\n[[ CONDITION: {condition.upper()} ]]")
        for judge in file_judges:
            print(f"\n--- {judge.upper()} ---")
            judge_df = df[(df["Judge"] == judge.upper()) & (df["Condition"] == condition)]
            for nl in noise_levels:
                nl_df = judge_df[(judge_df["Noise Level"] == nl) & (judge_df["Is Poisoned"] == True)]
                true_df = nl_df[nl_df["Support Poisoned"] == True]
                surv_df = nl_df[nl_df["Support Poisoned"] == False]
                
                std_f = nl_df["Faithfulness"].mean() if not nl_df.empty else float('nan')
                true_f = true_df["Faithfulness"].mean() if not true_df.empty else float('nan')
                surv_f = surv_df["Faithfulness"].mean() if not surv_df.empty else float('nan')
                
                print(f"  Noise {nl}: Std Mean Faith = {std_f:.3f} | True Mean Faith = {true_f:.3f} (N={len(true_df)}) | Surv Mean Faith = {surv_f:.3f} (N={len(surv_df)})")

if __name__ == "__main__":
    main()