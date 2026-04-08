import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_DB_DIR = PROJECT_ROOT / "results" / "scene_db"
GEMMA_OUTPUT_DIR = PROJECT_ROOT / "results" / "gemma_outputs"
QUERY_RESULTS_DIR = PROJECT_ROOT / "results" / "query_results"

SCENE_ID = "scene0114_00"

OBJECT_DB_PATH = SCENE_DB_DIR / f"{SCENE_ID}_objects.json"
SELECTION_PATH = GEMMA_OUTPUT_DIR / f"{SCENE_ID}_gemma_selection.json"
OUTPUT_PATH = QUERY_RESULTS_DIR / f"{SCENE_ID}_query_result.json"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_objects_by_id(object_db):
    return {obj["object_id"]: obj for obj in object_db["objects"]}


def compute_distance(c1, c2):
    c1 = np.array(c1)
    c2 = np.array(c2)
    return float(np.linalg.norm(c1 - c2))


def build_answer(label_a: str, label_b: str, distance_m: float):
    return f"The {label_a} is {distance_m:.2f} m from the {label_b}."


# ============================================================
# PIPELINE
# ============================================================

def main():
    QUERY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading object database: {OBJECT_DB_PATH}")
    print(f"[INFO] Loading selection file: {SELECTION_PATH}")

    object_db = load_json(OBJECT_DB_PATH)
    selection = load_json(SELECTION_PATH)

    object_index = index_objects_by_id(object_db)

    object_a_id = selection["object_a"]
    object_b_id = selection["object_b"]

    if object_a_id not in object_index:
        raise ValueError(f"object_a={object_a_id} not found in object database.")
    if object_b_id not in object_index:
        raise ValueError(f"object_b={object_b_id} not found in object database.")

    obj_a = object_index[object_a_id]
    obj_b = object_index[object_b_id]

    distance_m = compute_distance(obj_a["centroid"], obj_b["centroid"])
    answer = build_answer(obj_a["label"], obj_b["label"], distance_m)

    result = {
        "scene_id": selection["scene_id"],
        "raw_query": selection["raw_query"],
        "object_a": {
            "object_id": obj_a["object_id"],
            "label": obj_a["label"],
            "centroid": obj_a["centroid"],
            "bbox_min": obj_a["bbox_min"],
            "bbox_max": obj_a["bbox_max"],
        },
        "object_b": {
            "object_id": obj_b["object_id"],
            "label": obj_b["label"],
            "centroid": obj_b["centroid"],
            "bbox_min": obj_b["bbox_min"],
            "bbox_max": obj_b["bbox_max"],
        },
        "distance_m": round(distance_m, 4),
        "answer": answer,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n[INFO] Query result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[INFO] Saved query result to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()