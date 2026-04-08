import json
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = PROJECT_ROOT / "results" / "retrieval"
GEMMA_OUTPUT_DIR = PROJECT_ROOT / "results" / "gemma_outputs"

SCENE_ID = "scene0114_00"

INPUT_JSON = RETRIEVAL_DIR / f"{SCENE_ID}_retrieval.json"
OUTPUT_JSON = GEMMA_OUTPUT_DIR / f"{SCENE_ID}_gemma_selection.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(retrieval_result: dict) -> str:
    raw_query = retrieval_result["raw_query"]
    scene_id = retrieval_result["scene_id"]

    object_a_text = retrieval_result["object_a_text"]
    object_b_text = retrieval_result["object_b_text"]

    object_a_candidates = retrieval_result["object_a_candidates"]
    object_b_candidates = retrieval_result["object_b_candidates"]

    lines = []
    lines.append("You are given a scene and a natural language question.")
    lines.append("Your task is to choose exactly one object for object_a and one object for object_b.")
    lines.append("")
    lines.append("IMPORTANT RULES:")
    lines.append(f'- object_a MUST be selected only from the candidates for "{object_a_text}"')
    lines.append(f'- object_b MUST be selected only from the candidates for "{object_b_text}"')
    lines.append("- Do not swap them.")
    lines.append("- Return only valid JSON.")
    lines.append("")
    lines.append(f"Scene: {scene_id}")
    lines.append(f"Question: {raw_query}")
    lines.append("")
    lines.append(f'Candidates for object_a ("{object_a_text}"):')
    for c in object_a_candidates:
        lines.append(
            f'- object_id={c["object_id"]}, label={c["label"]}, description="{c["description"]}"'
        )

    lines.append("")
    lines.append(f'Candidates for object_b ("{object_b_text}"):')
    for c in object_b_candidates:
        lines.append(
            f'- object_id={c["object_id"]}, label={c["label"]}, description="{c["description"]}"'
        )

    lines.append("")
    lines.append("Return exactly this JSON format:")
    lines.append("{")
    lines.append('  "object_a": <integer from object_a candidates>,')
    lines.append('  "object_b": <integer from object_b candidates>')
    lines.append("}")

    return "\n".join(lines)


def parse_model_response(text: str):
    parsed = json.loads(text)

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


def call_ollama(prompt: str):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    if "response" not in data:
        raise ValueError("Ollama response does not contain 'response' field.")

    return data["response"]


def main():
    GEMMA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    retrieval_result = load_json(INPUT_JSON)
    prompt = build_prompt(retrieval_result)

    print("\n[INFO] Sending prompt to Ollama...\n")
    raw_model_output = call_ollama(prompt)

    print("[INFO] Raw model output:")
    print(raw_model_output)

    selection = parse_model_response(raw_model_output)
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