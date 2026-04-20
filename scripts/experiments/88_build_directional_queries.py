#!/usr/bin/env python3
"""
88_build_directional_queries.py

Constrói o dataset binário para o operador direcional `above` nas cenas
do test_official_stage1.

Critério positivo: aabb_min_z(A) >= aabb_max_z(B) + MIN_GAP_M
  → A está claramente acima de B (sem sobreposição vertical)

Dataset binário:
  Positivo: "A está acima de B?"  → label = 1
  Negativo: "B está acima de A?"  → label = 0  (par invertido)

Uso:
    python scripts/88_build_directional_queries.py
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT         = Path(__file__).resolve().parents[1]
BENCHMARK    = ROOT / "benchmark"
MANIFEST_CSV = BENCHMARK / "objects_manifest_test_official_stage1.csv"
SCENES_CSV   = BENCHMARK / "scenes_test_official_stage1.csv"
OUT_CSV      = BENCHMARK / "queries_test_official_stage1_above_binary.csv"

MIN_GAP_M    = 0.10   # folga mínima entre aabb_min_z(A) e aabb_max_z(B)
MAX_PER_SCENE = 5     # pares positivos por cena

def main() -> None:
    manifest = pd.read_csv(MANIFEST_CSV)
    manifest = manifest[manifest["is_valid_object"] == True].copy()

    scenes = pd.read_csv(SCENES_CSV)["scene_id"].tolist()

    records = []
    pair_id = 0

    for scene_id in scenes:
        sdf = manifest[manifest["scene_id"] == scene_id].copy()
        if len(sdf) < 2:
            continue

        rows  = sdf.reset_index(drop=True)
        pairs = []

        for i in range(len(rows)):
            for j in range(len(rows)):
                if i == j:
                    continue
                A = rows.iloc[i]
                B = rows.iloc[j]
                gap = A["aabb_min_z"] - B["aabb_max_z"]
                if gap >= MIN_GAP_M:
                    pairs.append((gap, A, B))

        # Ordena por gap decrescente, pega os top-k mais claros
        pairs.sort(reverse=True, key=lambda x: x[0])
        selected = pairs[:MAX_PER_SCENE]

        for gap, A, B in selected:
            oid_a = A["object_id"]
            oid_b = B["object_id"]
            lab_a = A["label_norm"]
            lab_b = B["label_norm"]

            sq_pos = f'above("{oid_a}", "{oid_b}")'
            nq_pos = f"{oid_a} está acima de {oid_b}?"
            sq_neg = f'above("{oid_b}", "{oid_a}")'
            nq_neg = f"{oid_b} está acima de {oid_a}?"

            group = f"above_{scene_id}_{pair_id:04d}"

            records.append({
                "binary_query_id":  f"{group}__pos",
                "scene_id":         scene_id,
                "operator":         "above",
                "structured_query": sq_pos,
                "natural_query":    nq_pos,
                "object_a":         oid_a, "label_a": lab_a,
                "object_b":         oid_b, "label_b": lab_b,
                "gap_z_m":          round(gap, 4),
                "aabb_min_z_a":     round(float(A["aabb_min_z"]), 4),
                "aabb_max_z_b":     round(float(B["aabb_max_z"]), 4),
                "binary_label":     1,
                "pairing_group":    group,
            })
            records.append({
                "binary_query_id":  f"{group}__neg",
                "scene_id":         scene_id,
                "operator":         "above",
                "structured_query": sq_neg,
                "natural_query":    nq_neg,
                "object_a":         oid_b, "label_a": lab_b,
                "object_b":         oid_a, "label_b": lab_a,
                "gap_z_m":          round(-gap, 4),
                "aabb_min_z_a":     round(float(B["aabb_min_z"]), 4),
                "aabb_max_z_b":     round(float(A["aabb_max_z"]), 4),
                "binary_label":     0,
                "pairing_group":    group,
            })
            pair_id += 1

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)

    n_pos = (df["binary_label"] == 1).sum()
    n_neg = (df["binary_label"] == 0).sum()
    print(f"Dataset above/below gerado: {len(df)} exemplos "
          f"({n_pos} positivos + {n_neg} negativos)")
    print(f"Cenas cobertas: {df['scene_id'].nunique()}")
    print(f"Gap Z mínimo:   {df[df['binary_label']==1]['gap_z_m'].min():.3f} m")
    print(f"Gap Z máximo:   {df[df['binary_label']==1]['gap_z_m'].max():.3f} m")
    print(f"\nCategorias mais frequentes (objeto acima):")
    top = df[df["binary_label"]==1]["label_a"].value_counts().head(8)
    for cat, n in top.items():
        print(f"  {cat}: {n}")
    print(f"\nSalvo em: {OUT_CSV}")


if __name__ == "__main__":
    main()