from __future__ import annotations

from pathlib import Path
import argparse
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geometry.geometry_ops import load_points_npz, surface_distance, is_between_xy, is_aligned_xy

ALIAS_DIR = ROOT / "benchmark" / "demo_alias_maps"

def centroid_xyz(row: pd.Series) -> np.ndarray:
    return row[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)

def nearest_by_surface(ref_points: np.ndarray, candidates: dict[str, np.ndarray]):
    best_alias = None
    best_dist = float("inf")

    for alias, pts in candidates.items():
        d = float(surface_distance(ref_points, pts))
        if d < best_dist:
            best_dist = d
            best_alias = alias

    return best_alias, best_dist

def normalize_text(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace("distância", "distancia")
        .replace("próximo", "proximo")
        .replace("estão", "estao")
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True)
    args = parser.parse_args()

    alias_path = ALIAS_DIR / f"{args.scene_id}_alias_map.csv"
    if not alias_path.exists():
        raise RuntimeError(f"Alias map não encontrado: {alias_path}")

    df = pd.read_csv(alias_path)
    by_alias = {row["alias"]: row for _, row in df.iterrows()}

    print(f"Cena: {args.scene_id}")
    print("Aliases disponíveis:")
    print(df[["alias", "label_norm"]].to_string(index=False))
    print()
    print("Exemplos:")
    print("  qual a distancia entre chair1 e table1?")
    print("  qual chair esta mais proximo de cabinet1?")
    print("  chair2 esta entre desk1 e table1?")
    print("  cabinet1, chair2 e table1 estao alinhados?")
    print()
    print("Digite 'sair' para encerrar.")
    print()

    while True:
        q = input("Consulta> ").strip()
        if not q:
            continue
        if q.lower() in {"sair", "exit", "quit"}:
            break

        text = normalize_text(q)

        # distance
        m = re.match(r"qual a distancia entre ([a-z0-9_]+) e ([a-z0-9_]+)\?", text)
        if m:
            a1, a2 = m.group(1), m.group(2)
            if a1 not in by_alias or a2 not in by_alias:
                print("Alias não encontrado.\n")
                continue

            row1 = by_alias[a1]
            row2 = by_alias[a2]
            pts1 = load_points_npz(ROOT / row1["points_path"])
            pts2 = load_points_npz(ROOT / row2["points_path"])
            d = surface_distance(pts1, pts2)

            print(f"Resposta: a distância entre {a1} e {a2} é {d:.4f} m.\n")
            continue

        # nearest
        m = re.match(r"qual ([a-z_]+) esta mais proximo de ([a-z0-9_]+)\?", text)
        if m:
            category, ref_alias = m.group(1), m.group(2)
            if ref_alias not in by_alias:
                print("Alias de referência não encontrado.\n")
                continue

            ref_row = by_alias[ref_alias]
            ref_points = load_points_npz(ROOT / ref_row["points_path"])

            sub = df[df["label_norm"] == category].copy()
            if len(sub) == 0:
                print("Categoria sem candidatos nesta cena.\n")
                continue

            candidates = {}
            for _, row in sub.iterrows():
                if row["alias"] == ref_alias:
                    continue
                candidates[row["alias"]] = load_points_npz(ROOT / row["points_path"])

            if len(candidates) == 0:
                print("Nenhum candidato válido encontrado.\n")
                continue

            best_alias, best_dist = nearest_by_surface(ref_points, candidates)
            print(f"Resposta: o objeto da categoria {category} mais próximo de {ref_alias} é {best_alias}, a {best_dist:.4f} m.\n")
            continue

        # between
        m = re.match(r"([a-z0-9_]+) esta entre ([a-z0-9_]+) e ([a-z0-9_]+)\?", text)
        if m:
            x_alias, a_alias, b_alias = m.group(1), m.group(2), m.group(3)
            if x_alias not in by_alias or a_alias not in by_alias or b_alias not in by_alias:
                print("Um ou mais aliases não foram encontrados.\n")
                continue

            cx = centroid_xyz(by_alias[x_alias])
            ca = centroid_xyz(by_alias[a_alias])
            cb = centroid_xyz(by_alias[b_alias])

            pred = is_between_xy(cx, ca, cb, tau_between=0.30)
            resp = "sim" if pred else "não"
            print(f"Resposta: {resp}, {x_alias} {'está' if pred else 'não está'} entre {a_alias} e {b_alias}.\n")
            continue

        # aligned
        m = re.match(r"([a-z0-9_]+), ([a-z0-9_]+) e ([a-z0-9_]+) estao alinhados\?", text)
        if m:
            a1, a2, a3 = m.group(1), m.group(2), m.group(3)
            if a1 not in by_alias or a2 not in by_alias or a3 not in by_alias:
                print("Um ou mais aliases não foram encontrados.\n")
                continue

            c1 = centroid_xyz(by_alias[a1])
            c2 = centroid_xyz(by_alias[a2])
            c3 = centroid_xyz(by_alias[a3])

            pred = is_aligned_xy(c1, c2, c3, tau_align=0.25)
            resp = "sim" if pred else "não"
            print(f"Resposta: {resp}, {a1}, {a2} e {a3} {'estão' if pred else 'não estão'} alinhados.\n")
            continue

        print("Consulta fora do padrão esperado.\n")

if __name__ == "__main__":
    main()