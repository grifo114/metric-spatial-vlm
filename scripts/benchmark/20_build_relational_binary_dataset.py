from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geometry.geometry_ops import is_between_xy, is_aligned_xy

MANIFEST = ROOT / "benchmark" / "objects_manifest_dev.csv"
POS_PATH = ROOT / "benchmark" / "queries_dev_relational_binary.csv"
OUT_PATH = ROOT / "benchmark" / "queries_dev_relational_binary_labeled.csv"

FURNITURE = {"chair", "sofa", "bed"}
ALL_ALLOWED = {"chair", "sofa", "bed", "table", "desk", "door", "cabinet", "monitor"}

def centroid_xyz(row: pd.Series) -> np.ndarray:
    return row[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)

def parse_between_row(q: pd.Series):
    return q["object_x"], q["object_a"], q["object_b"]

def parse_aligned_row(q: pd.Series):
    return q["object_a"], q["object_b"], q["object_c"]

def main():
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest[manifest["is_valid_object"] == True].copy()

    pos = pd.read_csv(POS_PATH)

    manifest_by_scene = {scene_id: g.copy() for scene_id, g in manifest.groupby("scene_id")}
    manifest_by_id = {row["object_id"]: row for _, row in manifest.iterrows()}

    rows = []

    for _, q in pos.iterrows():
        op = q["operator"]
        scene_id = q["scene_id"]
        scene_df = manifest_by_scene[scene_id]

        # sempre manter o positivo
        pos_row = q.copy()
        pos_row["binary_label"] = 1
        pos_row["pairing_group"] = q["query_id"] if "query_id" in q else q["structured_query"]
        rows.append(pos_row.to_dict())

        if op == "between":
            object_x, object_a, object_b = parse_between_row(q)

            row_a = manifest_by_id[object_a]
            row_b = manifest_by_id[object_b]
            ca = centroid_xyz(pd.Series(row_a))
            cb = centroid_xyz(pd.Series(row_b))

            # candidatos negativos: furniture da mesma cena, exceto o positivo
            candidates = scene_df[
                (scene_df["label_norm"].isin(FURNITURE)) &
                (scene_df["object_id"] != object_x)
            ].copy()

            negative_found = False
            for _, cand in candidates.iterrows():
                cx = centroid_xyz(cand)
                if not is_between_xy(cx, ca, cb, tau_between=float(q.get("tau_between_m", 0.35))):
                    neg = q.copy()
                    neg["object_x"] = cand["object_id"]
                    neg["label_x"] = cand["label_norm"]
                    neg["structured_query"] = f'between("{cand["object_id"]}", "{object_a}", "{object_b}")'
                    neg["natural_query"] = f'{cand["object_id"]} está entre {object_a} e {object_b}?'
                    neg["binary_label"] = 0
                    neg["pairing_group"] = pos_row["pairing_group"]
                    rows.append(neg.to_dict())
                    negative_found = True
                    break

            if not negative_found:
                print(f"[WARN] No negative found for between in {scene_id}: {q['structured_query']}")

        elif op == "aligned":
            object_a, object_b, object_c = parse_aligned_row(q)

            row_a = manifest_by_id[object_a]
            row_b = manifest_by_id[object_b]

            ca = centroid_xyz(pd.Series(row_a))
            cb = centroid_xyz(pd.Series(row_b))

            # tentar trocar object_c por outro objeto da mesma cena que quebre alinhamento
            candidates = scene_df[
                (scene_df["label_norm"].isin(ALL_ALLOWED)) &
                (~scene_df["object_id"].isin([object_a, object_b, object_c]))
            ].copy()

            negative_found = False
            for _, cand in candidates.iterrows():
                cc = centroid_xyz(cand)
                if not is_aligned_xy(ca, cb, cc, tau_align=float(q.get("tau_align_m", 0.25))):
                    neg = q.copy()
                    neg["object_c"] = cand["object_id"]
                    neg["label_c"] = cand["label_norm"]
                    neg["structured_query"] = f'aligned("{object_a}", "{object_b}", "{cand["object_id"]}")'
                    neg["natural_query"] = f'{object_a}, {object_b} e {cand["object_id"]} estão alinhados?'
                    neg["binary_label"] = 0
                    neg["pairing_group"] = pos_row["pairing_group"]
                    rows.append(neg.to_dict())
                    negative_found = True
                    break

            if not negative_found:
                print(f"[WARN] No negative found for aligned in {scene_id}: {q['structured_query']}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print()
    print("Counts by operator and label:")
    print(out_df.groupby(["operator", "binary_label"]).size())
    print()
    print(out_df[["scene_id", "operator", "binary_label", "structured_query"]].head(30).to_string(index=False))

if __name__ == "__main__":
    main()