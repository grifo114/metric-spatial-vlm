from pathlib import Path
from math import comb
import pandas as pd


RESULTS_DIR = Path("results/benchmark_v1")


FILES = {
    "GPT-4.1": {
        "Baseline": [
            "e2e_grounding_test_official_raw_distance_baseline_en_gpt41_run1.csv",
            "e2e_grounding_test_official_raw_distance_baseline_en_gpt41_run2.csv",
        ],
        "L1": [
            "e2e_grounding_test_official_raw_distance_ctx_l1_en_gpt41_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l1_en_gpt41_run2.csv",
        ],
        "L2": [
            "e2e_grounding_test_official_raw_distance_ctx_l2_en_gpt41_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l2_en_gpt41_run2.csv",
        ],
        "L3": [
            "e2e_grounding_test_official_raw_distance_ctx_l3_en_gpt41_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l3_en_gpt41_run2.csv",
        ],
        "L2-Ref": [
            "e2e_grounding_test_official_raw_distance_ctx_l2_ref_only_en_gpt41_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l2_ref_only_en_gpt41_run2.csv",
        ],
    },

    "Gemini 2.5 Flash": {
        "Baseline": [
            "e2e_grounding_test_official_raw_gemini_flash_baseline_run1.csv",
            "e2e_grounding_test_official_raw_gemini_flash_baseline_run2.csv",
        ],
        "L1": [
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l1_run1.csv",
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l1_run2.csv",
        ],
        "L2": [
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l2_run1.csv",
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l2_run2.csv",
        ],
        "L3": [
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l3_run1.csv",
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l3_run2.csv",
        ],
        "L2-Ref": [
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l2ref_run1.csv",
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l2ref_run2.csv",
        ],
    },

        "Claude Sonnet 4.5": {
        "Baseline": [
            "e2e_grounding_test_official_raw_distance_baseline_en_claude_sonnet45_t1_run1.csv",
            "e2e_grounding_test_official_raw_distance_baseline_en_claude_sonnet45_t1_run2.csv",
        ],
        "L1": [
            "e2e_grounding_test_official_raw_distance_ctx_l1_en_claude_sonnet45_t1_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l1_en_claude_sonnet45_t1_run2.csv",
        ],
        "L2": [
            "e2e_grounding_test_official_raw_distance_ctx_l2_en_claude_sonnet45_t1_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l2_en_claude_sonnet45_t1_run2.csv",
        ],
        "L3": [
            "e2e_grounding_test_official_raw_distance_ctx_l3_en_claude_sonnet45_t1_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l3_en_claude_sonnet45_t1_run2.csv",
        ],
        "L2-Ref": [
            "e2e_grounding_test_official_raw_distance_ctx_l2_ref_only_en_claude_sonnet45_t1_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l2_ref_only_en_claude_sonnet45_t1_run2.csv",
        ],
    },
    "Qwen-235B": {
        "Baseline": [
            "e2e_grounding_test_official_raw_distance_baseline_en_qwen3vl235b_run1.csv",
            "e2e_grounding_test_official_raw_distance_baseline_en_qwen3vl235b_run2.csv",
        ],
        "L1": [
            "e2e_grounding_test_official_raw_distance_ctx_l1_en_qwen3vl235b_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l1_en_qwen3vl235b_run2.csv",
        ],
        "L2": [
            "e2e_grounding_test_official_raw_distance_ctx_l2_en_qwen3vl235b_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l2_en_qwen3vl235b_run2.csv",
        ],
        "L3": [
            "e2e_grounding_test_official_raw_distance_ctx_l3_en_qwen3vl235b_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l3_en_qwen3vl235b_run2.csv",
        ],
        "L2-Ref": [
            "e2e_grounding_test_official_raw_distance_ctx_l2_ref_only_en_qwen3vl235b_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l2_ref_only_en_qwen3vl235b_run2.csv",
        ],
    },

    "Qwen-32B": {
        "Baseline": [
            "e2e_grounding_test_official_raw_distance_baseline_en_qwen3vl32b_run1.csv",
            "e2e_grounding_test_official_raw_distance_baseline_en_qwen3vl32b_run2.csv",
        ],
        "L1": [
            "e2e_grounding_test_official_raw_distance_ctx_l1_en_qwen3vl32b_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l1_en_qwen3vl32b_run2.csv",
        ],
        "L2": [
            "e2e_grounding_test_official_raw_distance_ctx_l2_en_qwen3vl32b_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l2_en_qwen3vl32b_run2.csv",
        ],
        "L3": [
            "e2e_grounding_test_official_raw_distance_ctx_l3_en_qwen3vl32b_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l3_en_qwen3vl32b_run2.csv",
        ],
        "L2-Ref": [
            "e2e_grounding_test_official_raw_distance_ctx_l2_ref_only_en_qwen3vl32b_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l2_ref_only_en_qwen3vl32b_run2.csv",
        ],
    },

    "Qwen-8B": {
        "Baseline": [
            "e2e_grounding_test_official_raw_distance_baseline_en_qwen3vl8b_run1.csv",
            "e2e_grounding_test_official_raw_distance_baseline_en_qwen3vl8b_run2.csv",
        ],
        "L1": [
            "e2e_grounding_test_official_raw_distance_ctx_l1_en_qwen3vl8b_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l1_en_qwen3vl8b_run2.csv",
        ],
        "L2": [
            "e2e_grounding_test_official_raw_distance_ctx_l2_en_qwen3vl8b_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l2_en_qwen3vl8b_run2.csv",
        ],
        "L3": [
            "e2e_grounding_test_official_raw_distance_ctx_l3_en_qwen3vl8b_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l3_en_qwen3vl8b_run2.csv",
        ],
        "L2-Ref": [
            "e2e_grounding_test_official_raw_distance_ctx_l2_ref_only_en_qwen3vl8b_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l2_ref_only_en_qwen3vl8b_run2.csv",
        ],
    },

}


CONDITIONS = ["L1", "L2", "L3", "L2-Ref"]


def normalize_bool(s):
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )


def exact_mcnemar_p(b, c):
    """
    Exact two-sided McNemar using binomial test.
    b = baseline wrong, SCI correct
    c = baseline correct, SCI wrong
    """
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)
    p = 2 * sum(comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, p)


def load_condition(files):
    rows = []

    for run_idx, fname in enumerate(files, start=1):
        path = RESULTS_DIR / fname
        df = pd.read_csv(path)

        df = df[["query_id", "grounding_correct"]].copy()
        df["grounding_correct"] = normalize_bool(df["grounding_correct"])
        df["run"] = run_idx

        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def mcnemar_from_binary_pairs(base, cond):
    merged = base.merge(
        cond,
        on=["query_id", "run"],
        suffixes=("_base", "_cond"),
    )

    b = int(((merged["grounding_correct_base"] == False) &
             (merged["grounding_correct_cond"] == True)).sum())

    c = int(((merged["grounding_correct_base"] == True) &
             (merged["grounding_correct_cond"] == False)).sum())

    return b, c, exact_mcnemar_p(b, c), len(merged)


def aggregate_query_level(df, rule):
    """
    Converts two runs per query into one binary decision per query.

    strict: correct only if both runs are correct
    lenient: correct if at least one run is correct
    """
    grouped = df.groupby("query_id")["grounding_correct"]

    if rule == "strict":
        out = grouped.all()
    elif rule == "lenient":
        out = grouped.any()
    else:
        raise ValueError(rule)

    out = out.reset_index()
    out["run"] = 1
    return out


def main():
    for model, model_files in FILES.items():
        print("\n" + model)
        print("-" * len(model))

        baseline = load_condition(model_files["Baseline"])

        for cond_name in CONDITIONS:
            cond = load_condition(model_files[cond_name])

            # Original query-run level, n=90
            b90, c90, p90, n90 = mcnemar_from_binary_pairs(baseline, cond)

            # Query-level strict, n=45
            base_strict = aggregate_query_level(baseline, "strict")
            cond_strict = aggregate_query_level(cond, "strict")
            bs, cs, ps, ns = mcnemar_from_binary_pairs(base_strict, cond_strict)

            # Query-level lenient, n=45
            base_lenient = aggregate_query_level(baseline, "lenient")
            cond_lenient = aggregate_query_level(cond, "lenient")
            bl, cl, pl, nl = mcnemar_from_binary_pairs(base_lenient, cond_lenient)

            print(
                f"{cond_name:<6} | "
                f"query-run n={n90:>2} b={b90:>2} c={c90:>2} p={p90:.3f} | "
                f"strict n={ns:>2} b={bs:>2} c={cs:>2} p={ps:.3f} | "
                f"lenient n={nl:>2} b={bl:>2} c={cl:>2} p={pl:.3f}"
            )


if __name__ == "__main__":
    main()
