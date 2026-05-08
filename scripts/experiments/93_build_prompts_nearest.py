import pandas as pd
import os

INPUT_QUERIES = "benchmark/queries_test_official_stage1_distance_nearest_final.csv"
INPUT_DESCRIPTORS = "results/experiments/spatial_descriptors_nearest_test_official.csv"

OUTPUT = "results/experiments/prompts_nearest_test_official.csv"


def build_prompt(query_text, group):
    lines = []
    lines.append(query_text)
    lines.append("")
    lines.append("Candidates:")

    for _, row in group.iterrows():
        obj = row["candidate_object"]
        desc = row["descriptor"]
        lines.append(f"{obj}: {desc}")

    lines.append("")
    lines.append("Answer with the object_id only.")

    return "\n".join(lines)


def main():
    queries = pd.read_csv(INPUT_QUERIES)
    descriptors = pd.read_csv(INPUT_DESCRIPTORS)

    queries = queries[queries["operator"] == "nearest"].copy()
    queries["query_id"] = [f"nearest_{i:04d}" for i in range(len(queries))]

    merged = descriptors.merge(
        queries[["query_id", "natural_query"]],
        on="query_id",
        how="left"
    )

    prompts = []

    for qid, group in merged.groupby("query_id"):
        target = group.iloc[0]["target_category"]
        ref = group.iloc[0]["reference_object"]

        # extrair só o label do objeto de referência
        ref_label = ref.split("__")[-1].split("_")[0]

        query_text = f"Which {target} is closest to the {ref_label}?"

        prompt = build_prompt(query_text, group)

        prompts.append({
            "query_id": qid,
            "prompt_enriched": prompt
        })

    df_out = pd.DataFrame(prompts)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df_out.to_csv(OUTPUT, index=False)

    print(f"Saved prompts to: {OUTPUT}")
    print(f"Total prompts: {len(df_out)}")


if __name__ == "__main__":
    main()