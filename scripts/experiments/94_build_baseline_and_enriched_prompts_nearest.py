import os
import pandas as pd

QUERIES_PATH = "benchmark/queries_test_official_stage1_distance_nearest_final.csv"
DESCRIPTORS_PATH = "results/experiments/spatial_descriptors_nearest_test_official.csv"

OUTPUT_PATH = "results/experiments/prompts_nearest_baseline_enriched_test_official.csv"


def build_query_text(group: pd.DataFrame) -> str:
    target = group.iloc[0]["target_category_x"]
    ref_obj = group.iloc[0]["reference_object_x"]
    ref_label = ref_obj.split("__")[-1].split("_")[0]
    return f"Which {target} is closest to the {ref_label}?"


def build_baseline_prompt(query_text: str, group: pd.DataFrame) -> str:
    lines = [
        query_text,
        "",
        "Candidates:",
    ]

    for _, row in group.iterrows():
        lines.append(str(row["candidate_object"]))

    lines.extend([
        "",
        "Answer with the object_id only."
    ])

    return "\n".join(lines)


def build_enriched_prompt(query_text: str, group: pd.DataFrame) -> str:
    lines = [
        query_text,
        "",
        "Candidates:",
    ]

    for _, row in group.iterrows():
        lines.append(f'{row["candidate_object"]}: {row["descriptor"]}')

    lines.extend([
        "",
        "Answer with the object_id only."
    ])

    return "\n".join(lines)


def main():
    queries = pd.read_csv(QUERIES_PATH)
    descriptors = pd.read_csv(DESCRIPTORS_PATH)

    queries = queries[queries["operator"] == "nearest"].copy()
    queries["query_id"] = [f"nearest_{i:04d}" for i in range(len(queries))]

    keep_cols = [
        "query_id",
        "scene_id",
        "reference_object",
        "reference_label",
        "target_category",
        "answer_object",
        "answer_distance_m",
        "second_best_object",
        "second_best_distance_m",
        "margin_to_second_m",
    ]

    meta = queries[keep_cols].copy()

    merged = descriptors.merge(meta, on="query_id", how="left")

    rows = []

    for qid, group in merged.groupby("query_id"):
        group = group.sort_values("rank").reset_index(drop=True)

        query_text = build_query_text(group)
        prompt_baseline = build_baseline_prompt(query_text, group)
        prompt_enriched = build_enriched_prompt(query_text, group)

        rows.append({
            "query_id": qid,
            "scene_id": group.iloc[0]["scene_id_x"] if "scene_id_x" in group.columns else group.iloc[0]["scene_id"],
            "reference_object": group.iloc[0]["reference_object_x"] if "reference_object_x" in group.columns else group.iloc[0]["reference_object"],
            "reference_label": group.iloc[0]["reference_label"],
            "target_category": group.iloc[0]["target_category_x"] if "target_category_x" in group.columns else group.iloc[0]["target_category"],
            "answer_object": group.iloc[0]["answer_object"],
            "answer_distance_m": group.iloc[0]["answer_distance_m"],
            "second_best_object": group.iloc[0]["second_best_object"],
            "second_best_distance_m": group.iloc[0]["second_best_distance_m"],
            "margin_to_second_m": group.iloc[0]["margin_to_second_m"],
            "n_candidates": len(group),
            "prompt_baseline": prompt_baseline,
            "prompt_enriched": prompt_enriched,
        })

    df_out = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved prompts to: {OUTPUT_PATH}")
    print(f"Total prompts: {len(df_out)}")


if __name__ == "__main__":
    main()