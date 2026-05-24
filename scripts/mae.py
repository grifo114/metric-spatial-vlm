from pathlib import Path
import pandas as pd

files = {
    "Gemini 2.5 Flash": {
        "Baseline": [
            "e2e_grounding_test_official_raw_gemini_flash_baseline.csv",
            "e2e_grounding_test_official_raw_gemini_flash_baseline_run2.csv",
        ],
        "L1": [
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l1.csv",
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l1_run2.csv",
        ],
        "L2": [
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l2.csv",
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l2_run2.csv",
        ],
        "L3": [
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l3.csv",
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l3_run2.csv",
        ],
        "L2-Ref": [
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l2ref.csv",
            "e2e_grounding_test_official_raw_gemini_flash_ctx_l2ref_run2.csv",
        ],
    },
    "Claude Sonnet 4.5": {
        "Baseline": [
            "e2e_grounding_test_official_raw_distance_baseline_en_claude_sonnet45_run1.csv",
            "e2e_grounding_test_official_raw_distance_baseline_en_claude_sonnet45_run2.csv",
        ],
        "L1": [
            "e2e_grounding_test_official_raw_distance_ctx_l1_en_claude_sonnet45_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l1_en_claude_sonnet45_run2.csv",
        ],
        "L2": [
            "e2e_grounding_test_official_raw_distance_ctx_l2_en_claude_sonnet45_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l2_en_claude_sonnet45_run2.csv",
        ],
        "L3": [
            "e2e_grounding_test_official_raw_distance_ctx_l3_en_claude_sonnet45_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l3_en_claude_sonnet45_run2.csv",
        ],
        "L2-Ref": [
            "e2e_grounding_test_official_raw_distance_ctx_l2_ref_only_en_claude_sonnet45_run1.csv",
            "e2e_grounding_test_official_raw_distance_ctx_l2_ref_only_en_claude_sonnet45_run2.csv",
        ],
    },
}

base_dir = Path("results/benchmark_v1")

rows = []

for model, conditions in files.items():
    baseline_mae = None

    for condition, condition_files in conditions.items():
        dfs = []
        for fname in condition_files:
            path = base_dir / fname
            dfs.append(pd.read_csv(path))

        df = pd.concat(dfs, ignore_index=True)

        mae_surf = df["e_total_surface"].mean()
        mae_cent = df["e_total_centroid"].mean()

        if condition == "Baseline":
            baseline_mae = mae_surf
            delta = None
        else:
            delta = 100 * (mae_surf - baseline_mae) / baseline_mae

        rows.append({
            "Model": model,
            "Condition": condition,
            "MAE_surf": mae_surf,
            "MAE_cent": mae_cent,
            "Delta_surf": delta,
            "n": len(df),
        })

out = pd.DataFrame(rows)

for model in out["Model"].unique():
    print(model)
    sub = out[out["Model"] == model]
    for _, r in sub.iterrows():
        if r["Condition"] == "Baseline":
            delta_txt = "---"
        else:
            delta_txt = f"${r['Delta_surf']:+.1f}\\%$"

        print(
            f"  & {r['Condition']:<8} "
            f"& {r['MAE_surf']:.3f} m "
            f"& {r['MAE_cent']:.3f} m "
            f"& {delta_txt} \\\\"
        )
    print()