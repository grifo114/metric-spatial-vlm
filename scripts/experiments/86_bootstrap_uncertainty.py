#!/usr/bin/env python3
"""
86_bootstrap_uncertainty.py

Bootstrap de incerteza sobre as distâncias de superfície no test_official_stage1.

Para o operador distance:
  - Reamostagem com reposição de ambas as nuvens de pontos (B=1000 iterações)
  - IC 95% pelo método do percentil
  - Output: estimativa pontual + [ci_lower, ci_upper] por query

Para o operador nearest:
  - Reamostagem da nuvem do objeto de referência (B=500 iterações)
  - Trees dos candidatos pré-construídas fora do loop (eficiência)
  - Output: estabilidade da seleção (% de iterações em que o vencedor GT ganha)

Uso:
    python scripts/86_bootstrap_uncertainty.py
    python scripts/86_bootstrap_uncertainty.py --B-dist 500 --B-near 200  # rápido
    python scripts/86_bootstrap_uncertainty.py --dry-run 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
BENCHMARK   = ROOT / "benchmark"
POINTS_DIR  = ROOT / "artifacts" / "object_points_test_official_stage1"
RESULTS_DIR = ROOT / "results" / "benchmark_v1"

GT_CSV      = BENCHMARK / "ground_truth_distance_nearest_test_official_stage1.csv"
QUERIES_CSV = BENCHMARK / "queries_test_official_stage1_distance_nearest_final.csv"
MANIFEST_CSV = BENCHMARK / "objects_manifest_test_official_stage1.csv"

OUT_DIST = RESULTS_DIR / "bootstrap_distance_uncertainty.csv"
OUT_NEAR = RESULTS_DIR / "bootstrap_nearest_stability.csv"

# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------
SUBSAMPLE   = 2000    # cap de pontos por objeto antes do bootstrap
ALPHA       = 0.05    # nível de significância → IC 95%
SEED        = 42


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def load_points(obj_id: str, scene_id: str) -> np.ndarray | None:
    path = POINTS_DIR / scene_id / f"{obj_id}.npz"
    if not path.exists():
        return None
    return np.load(path)["points"].astype(np.float32)


def presample(pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pré-subsample sem reposição até n pontos (se nuvem maior que n)."""
    if len(pts) > n:
        idx = rng.choice(len(pts), size=n, replace=False)
        return pts[idx]
    return pts


# ---------------------------------------------------------------------------
# Bootstrap — distance
# ---------------------------------------------------------------------------

def bootstrap_distance(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    B: int,
    subsample: int,
    rng: np.random.Generator,
) -> dict:
    """
    Bootstrap do IC para d_surface entre duas nuvens.

    Estratégia: pré-subsamplea ambas as nuvens para `subsample` pontos;
    em cada iteração reamosteia pts_a com reposição e reconstrói a tree
    de pts_b (também reamostrado), capturando variância de ambas as nuvens.
    """
    base_a = presample(pts_a, subsample, rng)
    base_b = presample(pts_b, subsample, rng)
    n_a, n_b = len(base_a), len(base_b)

    # Estimativa pontual (sem bootstrap)
    tree_b = cKDTree(base_b)
    dists_pt, _ = tree_b.query(base_a, k=1, workers=-1)
    d_point = float(dists_pt.min())

    # Bootstrap
    boot_dists = np.empty(B, dtype=np.float32)
    for i in range(B):
        sa = base_a[rng.integers(0, n_a, n_a)]
        sb = base_b[rng.integers(0, n_b, n_b)]
        tree = cKDTree(sb)
        dd, _ = tree.query(sa, k=1, workers=-1)
        boot_dists[i] = dd.min()

    ci_lo = float(np.percentile(boot_dists, 100 * ALPHA / 2))
    ci_hi = float(np.percentile(boot_dists, 100 * (1 - ALPHA / 2)))

    return {
        "d_surface_point":  d_point,
        "d_boot_mean":      float(boot_dists.mean()),
        "d_boot_std":       float(boot_dists.std()),
        "ci_lower_95":      ci_lo,
        "ci_upper_95":      ci_hi,
        "ci_width_95":      ci_hi - ci_lo,
        "ci_relative_pct":  (ci_hi - ci_lo) / d_point * 100 if d_point > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Bootstrap — nearest stability
# ---------------------------------------------------------------------------

def bootstrap_nearest(
    ref_pts: np.ndarray,
    candidates: list[tuple[str, np.ndarray]],
    gt_winner: str,
    B: int,
    subsample: int,
    rng: np.random.Generator,
) -> dict:
    """
    Bootstrap de estabilidade para a seleção do vizinho mais próximo.

    Estratégia eficiente: pré-constrói as trees de todos os candidatos
    (uma vez, fora do loop); em cada iteração reamosteia apenas a nuvem
    de referência.
    """
    base_ref = presample(ref_pts, subsample, rng)
    n_ref    = len(base_ref)

    # Pré-subsamplea candidatos e constrói trees uma única vez
    cand_trees: list[tuple[str, cKDTree]] = []
    for cid, cpts in candidates:
        base_c = presample(cpts, subsample, rng)
        cand_trees.append((cid, cKDTree(base_c)))

    # Estimativa pontual do vencedor
    best_id, best_d = None, float("inf")
    for cid, tree in cand_trees:
        dd, _ = tree.query(base_ref, k=1, workers=-1)
        d = float(dd.min())
        if d < best_d:
            best_d, best_id = d, cid

    # Bootstrap
    winner_counts: dict[str, int] = {cid: 0 for cid, _ in cand_trees}
    for _ in range(B):
        sr      = base_ref[rng.integers(0, n_ref, n_ref)]
        bi, bd  = None, float("inf")
        for cid, tree in cand_trees:
            dd, _ = tree.query(sr, k=1, workers=-1)
            d     = float(dd.min())
            if d < bd:
                bd, bi = d, cid
        if bi in winner_counts:
            winner_counts[bi] += 1

    gt_stability    = winner_counts.get(gt_winner, 0) / B
    boot_winner     = max(winner_counts, key=winner_counts.get)
    boot_winner_stab = winner_counts[boot_winner] / B

    return {
        "point_winner":       best_id,
        "gt_winner":          gt_winner,
        "gt_correct_point":   best_id == gt_winner,
        "gt_stability":       float(gt_stability),
        "boot_winner":        boot_winner,
        "boot_winner_stability": float(boot_winner_stab),
        "n_candidates":       len(candidates),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B-dist", type=int, default=1000,
                        help="Iterações bootstrap para distance (default 1000)")
    parser.add_argument("--B-near", type=int, default=500,
                        help="Iterações bootstrap para nearest (default 500)")
    parser.add_argument("--subsample", type=int, default=SUBSAMPLE,
                        help=f"Cap de pontos por objeto (default {SUBSAMPLE})")
    parser.add_argument("--dry-run", type=int, default=0,
                        help="Processar apenas N queries de cada operador")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Carrega dados ---
    gt_df      = pd.read_csv(GT_CSV)
    queries_df = pd.read_csv(QUERIES_CSV)
    manifest_df = pd.read_csv(MANIFEST_CSV)

    queries_df = queries_df[queries_df["review_keep"] == "yes"].copy()

    extra = ["scene_id","operator","structured_query",
             "reference_object","target_category","answer_object"]
    for c in extra:
        if c not in queries_df.columns:
            queries_df[c] = None

    merged = gt_df.merge(
        queries_df[extra].drop_duplicates("structured_query"),
        on=["scene_id","operator","structured_query"], how="left"
    )

    dist_df = merged[merged["operator"] == "distance"].copy()
    near_df = merged[merged["operator"] == "nearest"].copy()

    if args.dry_run > 0:
        dist_df = dist_df.head(args.dry_run)
        near_df = near_df.head(args.dry_run)

    # ======================================================================
    # DISTANCE
    # ======================================================================
    print(f"\n=== DISTANCE bootstrap (B={args.B_dist}, subsample={args.subsample}) ===")
    dist_results = []
    t0 = time.time()

    for i, row in dist_df.iterrows():
        qid      = str(row["query_id"])
        scene_id = str(row["scene_id"])
        obj_a    = str(row["gt_object_a"])
        obj_b    = str(row["gt_object_b"])
        gt_dist  = float(row["gt_distance_m"])

        print(f"  [{len(dist_results)+1}/{len(dist_df)}] {qid}", end=" ", flush=True)

        pts_a = load_points(obj_a, scene_id)
        pts_b = load_points(obj_b, scene_id)

        if pts_a is None or pts_b is None:
            print("SKIP (nuvem não encontrada)")
            dist_results.append({
                "query_id": qid, "scene_id": scene_id,
                "gt_distance_m": gt_dist,
                "obj_a": obj_a, "obj_b": obj_b,
                "n_pts_a": None, "n_pts_b": None,
                "d_surface_point": None,
                "d_boot_mean": None, "d_boot_std": None,
                "ci_lower_95": None, "ci_upper_95": None,
                "ci_width_95": None, "ci_relative_pct": None,
            })
            continue

        res = bootstrap_distance(pts_a, pts_b, args.B_dist, args.subsample, rng)
        print(f"d={res['d_surface_point']:.3f} "
              f"IC=[{res['ci_lower_95']:.3f},{res['ci_upper_95']:.3f}] "
              f"w={res['ci_width_95']:.3f} ({res['ci_relative_pct']:.1f}%)")

        dist_results.append({
            "query_id":      qid,
            "scene_id":      scene_id,
            "gt_distance_m": gt_dist,
            "obj_a": obj_a, "obj_b": obj_b,
            "n_pts_a": len(pts_a), "n_pts_b": len(pts_b),
            **res,
        })

    dist_out = pd.DataFrame(dist_results)
    dist_out.to_csv(OUT_DIST, index=False)
    print(f"Tempo distance: {time.time()-t0:.1f}s")
    print(f"Salvo: {OUT_DIST}")

    # Resumo
    valid_d = dist_out.dropna(subset=["ci_width_95"])
    if len(valid_d) > 0:
        print(f"\n  CI width 95% — média:   {valid_d['ci_width_95'].mean():.4f} m")
        print(f"  CI width 95% — mediana: {valid_d['ci_width_95'].median():.4f} m")
        print(f"  CI width 95% — máximo:  {valid_d['ci_width_95'].max():.4f} m")
        print(f"  CI relativo  — média:   {valid_d['ci_relative_pct'].mean():.1f}%")

    # ======================================================================
    # NEAREST
    # ======================================================================
    print(f"\n=== NEAREST bootstrap (B={args.B_near}, subsample={args.subsample}) ===")
    near_results = []
    t0 = time.time()

    for i, row in near_df.iterrows():
        qid       = str(row["query_id"])
        scene_id  = str(row["scene_id"])
        ref_obj   = str(row["gt_object_a"])
        target_cat = str(row.get("target_category") or "")
        gt_answer = str(row.get("gt_answer_object") or "")

        print(f"  [{len(near_results)+1}/{len(near_df)}] {qid}", end=" ", flush=True)

        ref_pts = load_points(ref_obj, scene_id)
        if ref_pts is None:
            print("SKIP (ref não encontrado)")
            near_results.append({
                "query_id": qid, "scene_id": scene_id,
                "gt_winner": gt_answer, "n_candidates": None,
                "gt_stability": None, "boot_winner": None,
                "boot_winner_stability": None, "gt_correct_point": None,
            })
            continue

        # Carrega candidatos
        sdf = manifest_df[
            (manifest_df["scene_id"] == scene_id) &
            (manifest_df["is_valid_object"] == True) &
            (manifest_df["label_norm"] == target_cat) &
            (manifest_df["object_id"] != ref_obj)
        ]
        candidates = []
        for _, crow in sdf.iterrows():
            cpts = load_points(crow["object_id"], scene_id)
            if cpts is not None:
                candidates.append((crow["object_id"], cpts))

        if not candidates:
            print("SKIP (sem candidatos)")
            continue

        res = bootstrap_nearest(
            ref_pts, candidates, gt_answer, args.B_near, args.subsample, rng
        )
        print(f"GT_stab={res['gt_stability']:.1%} "
              f"boot_winner_stab={res['boot_winner_stability']:.1%} "
              f"n_cand={res['n_candidates']}")

        near_results.append({
            "query_id":  qid,
            "scene_id":  scene_id,
            "ref_obj":   ref_obj,
            "target_cat": target_cat,
            **res,
        })

    near_out = pd.DataFrame(near_results)
    near_out.to_csv(OUT_NEAR, index=False)
    print(f"Tempo nearest: {time.time()-t0:.1f}s")
    print(f"Salvo: {OUT_NEAR}")

    # Resumo
    valid_n = near_out.dropna(subset=["gt_stability"])
    if len(valid_n) > 0:
        print(f"\n  GT stability — média:   {valid_n['gt_stability'].mean():.1%}")
        print(f"  GT stability — mediana: {valid_n['gt_stability'].median():.1%}")
        print(f"  GT stability — mínimo:  {valid_n['gt_stability'].min():.1%}")
        s100 = (valid_n['gt_stability'] >= 0.95).sum()
        print(f"  Queries com stab ≥ 95%: {s100}/{len(valid_n)}")

    print("\nScript 86 concluído.")


if __name__ == "__main__":
    main()