import json
from pathlib import Path


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_DB_DIR = PROJECT_ROOT / "results" / "scene_db"
RETRIEVAL_DIR = PROJECT_ROOT / "results" / "retrieval"

SCENE_ID = "scene0114_00"

OBJECT_DB_PATH = SCENE_DB_DIR / f"{SCENE_ID}_objects.json"
DESCRIPTION_DB_PATH = SCENE_DB_DIR / f"{SCENE_ID}_descriptions.json"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_descriptions_by_object_id(description_db):
    desc_index = {}
    for item in description_db["descriptions"]:
        desc_index[item["object_id"]] = item
    return desc_index


def retrieve_candidates_by_label(object_db, description_index, label_text: str):
    candidates = []

    for obj in object_db["objects"]:
        if obj["label"] == label_text:
            desc = description_index.get(obj["object_id"], {})
            candidates.append({
                "object_id": obj["object_id"],
                "label": obj["label"],
                "n_points": obj["n_points"],
                "centroid": obj["centroid"],
                "bbox_min": obj["bbox_min"],
                "bbox_max": obj["bbox_max"],
                "description": desc.get("description", "")
            })

    return candidates


def run_retrieval(scene_id: str, parsed_query: dict):
    object_db = load_json(OBJECT_DB_PATH)
    description_db = load_json(DESCRIPTION_DB_PATH)

    description_index = index_descriptions_by_object_id(description_db)

    object_a_text = parsed_query.get("object_a_text")
    object_b_text = parsed_query.get("object_b_text")

    a_candidates = []
    b_candidates = []

    if object_a_text is not None:
        a_candidates = retrieve_candidates_by_label(object_db, description_index, object_a_text)

    if object_b_text is not None:
        b_candidates = retrieve_candidates_by_label(object_db, description_index, object_b_text)

    result = {
        "scene_id": scene_id,
        "raw_query": parsed_query.get("raw_query"),
        "normalized_query": parsed_query.get("normalized_query"),
        "intent": parsed_query.get("intent"),
        "object_a_text": object_a_text,
        "object_b_text": object_b_text,
        "object_a_candidates": a_candidates,
        "object_b_candidates": b_candidates,
        "n_object_a_candidates": len(a_candidates),
        "n_object_b_candidates": len(b_candidates)
    }

    return result


# ============================================================
# DEMO
# ============================================================

def main():
    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Candidate retrieval for scene query\n")
    print(f"[INFO] Scene: {SCENE_ID}\n")

    raw_query = input("Enter a query: ").strip()

    # Import local do parser para reaproveitar a lógica já criada
    import importlib.util
    parser_path = PROJECT_ROOT / "scripts" / "13_parse_natural_query.py"
    spec = importlib.util.spec_from_file_location("query_parser", parser_path)
    query_parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(query_parser)

    parsed_query = query_parser.parse_query(raw_query)

    if parsed_query["intent"] != "distance":
        print("[WARN] Query intent is not 'distance'. Retrieval not executed.")
        print(json.dumps(parsed_query, indent=2, ensure_ascii=False))
        return

    result = run_retrieval(SCENE_ID, parsed_query)

    output_path = RETRIEVAL_DIR / f"{SCENE_ID}_retrieval.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n[INFO] Retrieval result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n[INFO] Saved retrieval output to: {output_path}")


if __name__ == "__main__":
    main()