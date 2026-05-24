from pathlib import Path
import pandas as pd
from scipy.stats import binomtest

# ajuste esta pasta
BASE_DIR = Path("results/benchmark_v1")

files = {
    "Gemini 2.5 Flash": {
        "Baseline": [
            BASE_DIR / "e2e_grounding_test_official_raw_gemini_flash_baseline.csv",
            BASE_DIR / "e2e_grounding_test_official_raw_gemini_flash_baseline_run2.csv",
        ],
        "L1": [
            BASE_DIR / "e2e_grounding_test_official_raw_gemini_flash_ctx_l1.csv",
            BASE_DIR / "e2e_grounding_test_official_raw_gemini_flash_ctx_l1_run2.csv",
        ],
        "L2": [
            BASE_DIR / "e2e_grounding_test_official_raw_gemini_flash_ctx_l2.csv",
            BASE_DIR / "e2e_grounding_test_official_raw_gemini_flash_ctx_l2ref_run2.csv",
        ],
        "L3": [
            BASE_DIR / "e2e_grounding_test_official_raw_gemini_flash_ctx_l3.csv",
            BASE_DIR / "e2e_grounding_test_official_raw_gemini_flash_ctx_l3_run2.csv",
        ],
        "L2-Ref": [
            BASE_DIR / "e2e_grounding_test_official_raw_gemini_flash_ctx_l2ref.csv",
            BASE_DIR / "e2e_grounding_test_official_raw_gemini_flash_ctx_l2ref_run2.csv",
        ],
    },

    "Claude Sonnet 4.5": {
        "Baseline": [
            BASE_DIR / "e2e_grounding_test_official_raw_distance_baseline_en_claude_sonnet45_run1.csv",
            BASE_DIR / "e2e_grounding_test_official_raw_distance_baseline_en_claude_sonnet45_run2.csv",
        ],
        "L1": [
            BASE_DIR / "e2e_grounding_test_official_raw_distance_ctx_l1_en_claude_sonnet45_run1.csv",
            BASE_DIR / "e2e_grounding_test_official_raw_distance_ctx_l1_en_claude_sonnet45_run2.csv",
        ],
        "L2": [
            BASE_DIR / "e2e_grounding_test_official_raw_distance_ctx_l2_en_claude_sonnet45_run1.csv",
            BASE_DIR / "e2e_grounding_test_official_raw_distance_ctx_l2_en_claude_sonnet45_run2.csv",
        ],
        "L3": [
            BASE_DIR / "e2e_grounding_test_official_raw_distance_ctx_l3_en_claude_sonnet45_run1.csv",
            BASE_DIR / "e2e_grounding_test_official_raw_distance_ctx_l3_en_claude_sonnet45_run2.csv",
        ],
        "L2-Ref": [
            BASE_DIR / "e2e_grounding_test_official_raw_distance_ctx_l2_ref_only_en_claude_sonnet45_run1.csv",
            BASE_DIR / "e2e_grounding_test_official_raw_distance_ctx_l2_ref_only_en_claude_sonnet45_run2.csv",
        ],
    },
}


def load_condition(paths):
    dfs = []

    for run_idx, path in enumerate(paths, start=1):
        df = pd.read_csv(path)

        df = df[["query_id", "grounding_correct"]].copy()
        df["run_id"] = f"run{run_idx}"
        df["pair_id"] = df["run_id"] + "__" + df["query_id"].astype(str)

        df["grounding_correct"] = df["grounding_correct"].astype(bool)

        dfs.append(df[["pair_id", "grounding_correct"]])

    return pd.concat(dfs, ignore_index=True)


def mcnemar_exact(baseline_df, condition_df):
    merged = baseline_df.merge(
        condition_df,
        on="pair_id",
        suffixes=("_baseline", "_condition"),
        how="inner",
    )

    if len(merged) != 90:
        print(f"Warning: expected n=90, got n={len(merged)}")

    baseline = merged["grounding_correct_baseline"]
    condition = merged["grounding_correct_condition"]

    # b: baseline errado, SCI certo
    b = ((baseline == False) & (condition == True)).sum()

    # c: baseline certo, SCI errado
    c = ((baseline == True) & (condition == False)).sum()

    if b + c == 0:
        p = 1.0
    else:
        p = binomtest(
            k=min(b, c),
            n=b + c,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return b, c, p


def format_p(p):
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.3f}"


for model, model_files in files.items():
    print("\n" + model)
    print("-" * len(model))

    baseline_df = load_condition(model_files["Baseline"])

    for condition in ["L1", "L2", "L3", "L2-Ref"]:
        condition_df = load_condition(model_files[condition])
        b, c, p = mcnemar_exact(baseline_df, condition_df)

        print(
            f"{condition:6s} | "
            f"b={b:2d} | c={c:2d} | "
            f"p={format_p(p)}"
        )