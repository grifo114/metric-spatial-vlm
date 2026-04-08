import json
from pathlib import Path


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = PROJECT_ROOT / "results" / "retrieval"
GEMMA_OUTPUT_DIR = PROJECT_ROOT / "results" / "gemma_outputs"

SCENE_ID = "scene0114_00"

INPUT_JSON = RETRIEVAL_DIR / f"{SCENE_ID}_retrieval.json"
OUTPUT_JSON = GEMMA_OUTPUT_DIR / f"{SCENE_ID}_gemma_selection.json"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(retrieval_result: dict) -> str:
    raw_query = retrieval_result["raw_query"]
    scene_id = retrieval_result["scene_id"]

    object_a_candidates = retrieval_result["object_a_candidates"]
    object_b_candidates = retrieval_result["object_b_candidates"]

    lines = []
    lines.append("You are given a scene and a natural language question.")
    lines.append("Your task is to choose the correct object ids referred to in the question.")
    lines.append("")
    lines.append(f"Scene: {scene_id}")
    lines.append(f"Question: {raw_query}")
    lines.append("")
    lines.append("Candidates for object A:")
    for c in object_a_candidates:
        lines.append(
            f'- object_id={c["object_id"]}, label={c["label"]}, description="{c["description"]}"'
        )

    lines.append("")
    lines.append("Candidates for object B:")
    for c in object_b_candidates:
        lines.append(
            f'- object_id={c["object_id"]}, label={c["label"]}, description="{c["description"]}"'
        )

    lines.append("")
    lines.append("Return only valid JSON in this format:")
    lines.append('{')
    lines.append('  "object_a": <integer>,')
    lines.append('  "object_b": <integer>')
    lines.append('}')

    return "\n".join(lines)


def parse_model_response(text: str):
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by model: {e}")

    if "object_a" not in parsed or "object_b" not in parsed:
        raise ValueError("JSON must contain 'object_a' and 'object_b'.")

    if not isinstance(parsed["object_a"], int) or not isinstance(parsed["object_b"], int):
        raise ValueError("'object_a' and 'object_b' must be integers.")

    return parsed


def validate_selection(selection: dict, retrieval_result: dict):
    valid_a = {c["object_id"] for c in retrieval_result["object_a_candidates"]}
    valid_b = {c["object_id"] for c in retrieval_result["object_b_candidates"]}

    if selection["object_a"] not in valid_a:
        raise ValueError(f'object_a={selection["object_a"]} is not a valid candidate.')

    if selection["object_b"] not in valid_b:
        raise ValueError(f'object_b={selection["object_b"]} is not a valid candidate.')


# ============================================================
# PIPELINE
# ============================================================

def main():
    GEMMA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    retrieval_result = load_json(INPUT_JSON)

    prompt = build_prompt(retrieval_result)

    print("\n[INFO] Prompt to send to Gemma:\n")
    print("=" * 80)
    print(prompt)
    print("=" * 80)

    print("\n[INFO] Paste Gemma JSON response below.")
    print('[INFO] Example: {"object_a": 3, "object_b": 18}\n')

    response_text = input("Gemma response: ").strip()

    selection = parse_model_response(response_text)
    validate_selection(selection, retrieval_result)

    output = {
        "scene_id": retrieval_result["scene_id"],
        "raw_query": retrieval_result["raw_query"],
        "object_a": selection["object_a"],
        "object_b": selection["object_b"]
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n[INFO] Valid selection saved:")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[INFO] Output saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()