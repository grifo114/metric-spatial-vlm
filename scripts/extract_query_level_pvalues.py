#!/usr/bin/env python3
"""
extract_query_level_pvalues.py

Computes exact-binomial McNemar p-values at the query level, under two
aggregation rules:
  - STRICT:  query is correct iff both runs are correct.
  - LENIENT: query is correct iff at least one run is correct.

For each (model, SCI condition) cell, the test compares the aggregated
per-query decisions against the model's own baseline aggregated the same
way (strict baseline vs strict SCI, lenient baseline vs lenient SCI).

Output: a printed table that maps directly to Table VI / Section V-A
robustness claims.

Usage (from repo root):
    python scripts/extract_query_level_pvalues.py
"""

import sys
import re
from pathlib import Path
import pandas as pd
from scipy.stats import binomtest

DEFAULT_RESULTS_DIR = Path("results/benchmark_v1")

CONDITIONS_ORDER = ["ctx_l1", "ctx_l2", "ctx_l3", "ctx_l2_ref_only"]
CONDITION_LABELS = {
    "original":         "Baseline",
    "ctx_l1":           "L1",
    "ctx_l2":           "L2",
    "ctx_l3":           "L3",
    "ctx_l2_ref_only":  "L2-Ref",
}

MODEL_DISPLAY = {
    "gpt-4.1":                                "GPT-4.1",
    "claude-sonnet-4-5":                      "Claude Sonnet 4.5",
    "google/gemini-2.5-flash":                "Gemini 2.5 Flash",
    "qwen/qwen3-vl-235b-a22b-instruct":       "Qwen-235B",
    "qwen/qwen3-vl-32b-instruct":             "Qwen-32B",
    "qwen/qwen3-vl-8b-instruct":              "Qwen-8B",
}

MODELS_ORDER = [
    "gpt-4.1",
    "google/gemini-2.5-flash",
    "claude-sonnet-4-5",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen/qwen3-vl-32b-instruct",
    "qwen/qwen3-vl-8b-instruct",
]


def infer_condition(row):
    mode  = row.get("prompt_mode")
    level = row.get("descriptor_level")
    scope = row.get("context_scope")
    if mode == "original":
        return "original"
    if mode == "context":
        try:
            lvl = int(level)
        except (TypeError, ValueError):
            return None
        if scope == "reference_only" and lvl == 2:
            return "ctx_l2_ref_only"
        if lvl == 1: return "ctx_l1"
        if lvl == 2: return "ctx_l2"
        if lvl == 3: return "ctx_l3"
    return None


def model_from_filename(name):
    n = name.lower()
    if "gpt41" in n or "gpt-4.1" in n: return "gpt-4.1"
    if "claude" in n:                  return "claude-sonnet-4-5"
    if "gemini" in n:                  return "google/gemini-2.5-flash"
    if "qwen3vl235b" in n or "235b" in n: return "qwen/qwen3-vl-235b-a22b-instruct"
    if "qwen3vl32b" in n or "32b" in n:   return "qwen/qwen3-vl-32b-instruct"
    if "qwen3vl8b" in n or "8b" in n:     return "qwen/qwen3-vl-8b-instruct"
    return None


def run_from_filename(name):
    if "_run1" in name: return 1
    if "_run2" in name: return 2
    return 0


def mcnemar_exact(b, c):
    """
    Exact-binomial McNemar test on discordant pairs.
    b = baseline-correct, SCI-incorrect.
    c = baseline-incorrect, SCI-correct.
    Two-sided exact binomial test.
    Returns p-value and the discordant counts.
    """
    n = b + c
    if n == 0:
        return 1.0, b, c
    result = binomtest(c, n, p=0.5, alternative="two-sided")
    return result.pvalue, b, c


def aggregate_per_query(sub, rule):
    """
    sub: dataframe with rows (query_id, run, grounding_correct).
    rule: 'strict' or 'lenient'
    Returns dict[query_id] -> bool.
    """
    out = {}
    for qid, grp in sub.groupby("query_id"):
        vals = grp["grounding_correct"].astype(bool).tolist()
        if len(vals) < 2:
            # If only one run available, treat that single value as the answer
            out[qid] = bool(vals[0]) if vals else False
            continue
        if rule == "strict":
            out[qid] = all(vals)
        elif rule == "lenient":
            out[qid] = any(vals)
        else:
            raise ValueError(rule)
    return out


def main(results_dir):
    csvs = sorted(results_dir.glob("e2e_grounding_test_official_raw*.csv"))
    dfs = []
    for f in csvs:
        try:
            d = pd.read_csv(f)
            d["__source_file__"] = f.name
            d["__run__"]         = run_from_filename(f.name)
            dfs.append(d)
        except Exception as e:
            print(f"  [warn] {f.name}: {e}", file=sys.stderr)
    df = pd.concat(dfs, ignore_index=True)

    if "model" not in df.columns:
        df["model"] = None
    df["model"] = df.apply(
        lambda r: r["model"] if pd.notna(r["model"])
                  else model_from_filename(r["__source_file__"]),
        axis=1
    )
    df = df[df["model"].notna()].copy()
    df = df[df["operator"] == "distance"].copy()
    df["__cond__"] = df.apply(infer_condition, axis=1)
    df = df[df["__cond__"].notna()].copy()
    df = df[df["__run__"].isin([1, 2])].copy()
    df["grounding_correct"] = df["grounding_correct"].astype(bool)

    # Print as one block per model, with strict and lenient side by side.
    print("=" * 96)
    print("Query-level McNemar p-values vs baseline of same model")
    print("Each cell: SCI condition compared to baseline under given rule.")
    print("=" * 96)
    print(f"  {'Model':22s}  {'Cond':8s}  "
          f"{'strict (b/c)':>20s}  {'p_strict':>10s}  "
          f"{'lenient (b/c)':>20s}  {'p_lenient':>10s}")
    print("  " + "-" * 92)

    for model in MODELS_ORDER:
        sub_m = df[df["model"] == model]
        if len(sub_m) == 0:
            continue

        base = sub_m[sub_m["__cond__"] == "original"]
        base_strict  = aggregate_per_query(base, "strict")
        base_lenient = aggregate_per_query(base, "lenient")

        # Baseline accuracy under each rule (for reference)
        b_strict  = sum(base_strict.values())
        b_lenient = sum(base_lenient.values())
        n_qs = len(base_strict)
        print(f"  {MODEL_DISPLAY[model]:22s}  Baseline  "
              f"{b_strict}/{n_qs} correct (strict)            "
              f"{b_lenient}/{n_qs} correct (lenient)")

        for cond in CONDITIONS_ORDER:
            sub_c = sub_m[sub_m["__cond__"] == cond]
            sci_strict  = aggregate_per_query(sub_c, "strict")
            sci_lenient = aggregate_per_query(sub_c, "lenient")

            common = sorted(set(base_strict).intersection(sci_strict))
            if len(common) == 0:
                continue

            b_s = sum(1 for q in common
                      if base_strict[q] and not sci_strict[q])
            c_s = sum(1 for q in common
                      if not base_strict[q] and sci_strict[q])
            p_s, _, _ = mcnemar_exact(b_s, c_s)

            b_l = sum(1 for q in common
                      if base_lenient[q] and not sci_lenient[q])
            c_l = sum(1 for q in common
                      if not base_lenient[q] and sci_lenient[q])
            p_l, _, _ = mcnemar_exact(b_l, c_l)

            p_s_str = f"{p_s:.3f}" if p_s >= 0.001 else "<0.001"
            p_l_str = f"{p_l:.3f}" if p_l >= 0.001 else "<0.001"

            print(f"  {'':22s}  {CONDITION_LABELS[cond]:8s}  "
                  f"{f'{b_s}/{c_s}':>20s}  {p_s_str:>10s}  "
                  f"{f'{b_l}/{c_l}':>20s}  {p_l_str:>10s}")
        print()


if __name__ == "__main__":
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    main(results_dir)