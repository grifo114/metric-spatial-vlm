import pandas as pd
import time
from openai import OpenAI

# =========================
# CONFIG
# =========================
INPUT_PATH = "results/experiments/prompts_nearest_baseline_enriched_test_official.csv"
OUTPUT_PATH = "results/experiments/vlm_pilot_results.csv"

MODEL = "gpt-4.1"
N_SAMPLES = 39  # piloto pequeno

client = OpenAI()

# =========================
# UTILS
# =========================
import re

def extract_answer(text):
    if text is None:
        return None

    # padrão completo
    match = re.search(r"scene\d+_\d+__\w+_\d+", text)
    if match:
        return match.group(0)

    # tenta recuperar formato parcial (ex: só número)
    match_partial = re.search(r"chair_\d+", text)
    if match_partial:
        return match_partial.group(0)

    return None


def run_model(prompt):
    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )
    return response.output[0].content[0].text


# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv(INPUT_PATH)

    df = df.head(N_SAMPLES)

    results = []

    for _, row in df.iterrows():

        print(f"\nQuery: {row['query_id']}")

        # BASELINE
        print("Running baseline...")
        out_base = run_model(row["prompt_baseline"])
        pred_base = extract_answer(out_base)

        # ENRICHED
        print("Running enriched...")
        out_enriched = run_model(row["prompt_enriched"])
        pred_enriched = extract_answer(out_enriched)

        results.append({
            "query_id": row["query_id"],
            "gt": row["answer_object"],
            "pred_baseline": pred_base,
            "pred_enriched": pred_enriched,
            "correct_baseline": pred_base == row["answer_object"],
            "correct_enriched": pred_enriched == row["answer_object"],
        })

        time.sleep(1)

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved results:")
    print(df_out)


if __name__ == "__main__":
    main()