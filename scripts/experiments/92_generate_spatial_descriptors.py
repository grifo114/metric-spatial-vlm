import os
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
QUERIES_PATH = "benchmark/queries_test_official_stage1_distance_nearest_final.csv"
OBJECTS_PATH = "benchmark/objects_manifest_test_official_stage1.csv"

OUTPUT_PATH = "results/experiments/spatial_descriptors_nearest_test_official.csv"

# =========================
# UTILS
# =========================
def euclidean_xy(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def build_descriptor(rank, total, ref_label, distance):
    return f"distance to the {ref_label}: {distance:.2f} meters"
    
def build_descriptor(rank, total, ref_label, distance):
    if distance < 0.5:
        return f"very close to the {ref_label}"
    elif distance < 1.5:
        return f"close to the {ref_label}"
    else:
        return f"far from the {ref_label}"    


# =========================
# MAIN
# =========================
def main():
    print("Loading data...")

    queries = pd.read_csv(QUERIES_PATH)
    objects = pd.read_csv(OBJECTS_PATH)

    # filtrar apenas objetos válidos
    objects = objects[objects["is_valid_object"] == True]

    # filtrar apenas nearest
    queries = queries[queries["operator"] == "nearest"].copy()
    queries["query_id"] = [f"nearest_{i:04d}" for i in range(len(queries))]

    results = []

    print(f"Total nearest queries: {len(queries)}")

    for _, q in queries.iterrows():

        scene_id = q["scene_id"]
        ref_obj_id = q["reference_object"]
        target_category = q["target_category"]  

        # objetos da cena
        scene_objects = objects[objects["scene_id"] == scene_id]

        # objeto de referência
        ref_obj = scene_objects[scene_objects["object_id"] == ref_obj_id]

        if ref_obj.empty:
            continue

        ref_obj = ref_obj.iloc[0]
        ref_xy = (ref_obj["centroid_xy_x"], ref_obj["centroid_xy_y"])
        ref_label = ref_obj["label_norm"]

        # candidatos da categoria alvo
        candidates = scene_objects[scene_objects["label_norm"] == target_category]

        if len(candidates) < 2:
            continue

        # calcular distâncias
        distances = []
        for _, obj in candidates.iterrows():
            obj_xy = (obj["centroid_xy_x"], obj["centroid_xy_y"])
            dist = euclidean_xy(ref_xy, obj_xy)

            distances.append({
                "object_id": obj["object_id"],
                "label": obj["label_norm"],
                "distance": dist
            })

        # ordenar
        distances = sorted(distances, key=lambda x: x["distance"])

        total = len(distances)

        # gerar descritores
        for rank, item in enumerate(distances):
            descriptor = build_descriptor(rank, total, ref_label, item["distance"])

            results.append({
                "query_id": q["query_id"],
                "scene_id": scene_id,
                "reference_object": ref_obj_id,
                "target_category": target_category,
                "candidate_object": item["object_id"],
                "distance": item["distance"],
                "rank": rank,
                "descriptor": descriptor
            })

    # salvar
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved descriptors to: {OUTPUT_PATH}")
    print(f"Total rows: {len(df_out)}")


if __name__ == "__main__":
    main()