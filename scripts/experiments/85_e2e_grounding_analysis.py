#!/usr/bin/env python3
"""
84_e2e_grounding_analysis.py

Analisa os resultados do experimento E2E grounding (script 83) e produz:

  - Tabela de decomposição: E_geometric (GT grounding) vs E_total (grounding real)
  - Taxa de grounding correto por operador
  - Figura de barras empilhadas mostrando a contribuição de cada componente
  - Tabela LaTeX para a dissertação

Lê:
  results/benchmark_v1/e2e_grounding_test_official_raw.csv
  results/benchmark_v1/test_stage1_surface_vs_centroid_per_query.csv

Grava:
  results/benchmark_v1/e2e_grounding_summary.csv
  artifacts/figures/fig_e2e_grounding_decomposition.pdf
  artifacts/figures/fig_e2e_grounding_decomposition.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "benchmark_v1"
FIGS_DIR    = ROOT / "artifacts" / "figures"

E2E_CSV = RESULTS_DIR / "e2e_grounding_test_official_fixed.csv" 
GEOMETRIC_CSV = RESULTS_DIR / "test_stage1_surface_vs_centroid_per_query.csv"
SUMMARY_CSV  = RESULTS_DIR / "e2e_grounding_summary.csv"
FIG_PDF      = FIGS_DIR / "fig_e2e_grounding_decomposition.pdf"
FIG_PNG      = FIGS_DIR / "fig_e2e_grounding_decomposition.png"

# Paleta acessível (consistente com outros gráficos da dissertação)
C_GEOMETRIC = "#009E73"   # verde  — E_geometric (grounding perfeito)
C_GROUNDING = "#E69F00"   # laranja — contribuição do erro de grounding
C_TOTAL     = "#D55E00"   # vermelho — E_total (grounding real)


# ---------------------------------------------------------------------------
# Carga e validação
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not E2E_CSV.exists():
        raise FileNotFoundError(
            f"Não encontrado: {E2E_CSV}\n"
            "Execute o script 84 primeiro."
        )
    # Lê com engine python para tolerar número variável de colunas
    e2e = pd.read_csv(E2E_CSV, engine="python", on_bad_lines="warn")
    e2e = e2e[e2e["vlm_response"] != "ERROR"].copy()

    geo = pd.read_csv(GEOMETRIC_CSV) if GEOMETRIC_CSV.exists() else None
    return e2e, geo


# ---------------------------------------------------------------------------
# Análise — distance
# ---------------------------------------------------------------------------

def analyze_distance(
    e2e: pd.DataFrame,
    geo: pd.DataFrame | None,
) -> dict:
    dist_e2e = e2e[e2e["operator"] == "distance"].copy()
    dist_e2e["gt_value"]       = pd.to_numeric(dist_e2e["gt_value"],       errors="coerce")
    dist_e2e["e_total_surface"]= pd.to_numeric(dist_e2e["e_total_surface"],errors="coerce")

    n_total   = len(dist_e2e)
    n_correct = int(dist_e2e["grounding_correct"].sum())

    # E_total (grounding real, superfície)
    valid_total = dist_e2e.dropna(subset=["gt_value", "e_total_surface"])
    mae_total   = float((valid_total["e_total_surface"] - valid_total["gt_value"]).abs().mean()) \
                  if len(valid_total) > 0 else float("nan")
    medae_total = float((valid_total["e_total_surface"] - valid_total["gt_value"]).abs().median()) \
                  if len(valid_total) > 0 else float("nan")

    # E_geometric (grounding perfeito = GT objects) — vem do CSV existente
    if geo is not None:
        geo_dist = geo[geo["operator"] == "distance"].copy()
        geo_dist["gt_distance_m"]         = pd.to_numeric(geo_dist["gt_distance_m"],         errors="coerce")
        geo_dist["surface_pred_distance_m"] = pd.to_numeric(geo_dist.get("surface_pred_distance_m",
                                                              pd.Series(dtype=float)), errors="coerce")
        gv = geo_dist.dropna(subset=["gt_distance_m","surface_pred_distance_m"])
        mae_geometric   = float((gv["surface_pred_distance_m"] - gv["gt_distance_m"]).abs().mean()) \
                          if len(gv) > 0 else 0.0
        medae_geometric = float((gv["surface_pred_distance_m"] - gv["gt_distance_m"]).abs().median()) \
                          if len(gv) > 0 else 0.0
    else:
        # Valores do paper (test set oficial)
        mae_geometric   = 0.0
        medae_geometric = 0.0

    return {
        "distance_n_total":    n_total,
        "distance_n_grounding_correct": n_correct,
        "distance_grounding_accuracy": n_correct / n_total if n_total > 0 else float("nan"),
        "distance_MAE_geometric":  mae_geometric,
        "distance_MAE_total":      mae_total,
        "distance_MAE_grounding_contribution": mae_total - mae_geometric,
        "distance_MedAE_geometric": medae_geometric,
        "distance_MedAE_total":     medae_total,
    }


# ---------------------------------------------------------------------------
# Análise — nearest
# ---------------------------------------------------------------------------

def analyze_nearest(
    e2e: pd.DataFrame,
    geo: pd.DataFrame | None,
) -> dict:
    near_e2e = e2e[e2e["operator"] == "nearest"].copy()

    n_total   = len(near_e2e)
    n_correct = int(near_e2e["grounding_correct"].sum())

    # Top-1 E_total: nearest_vlm_answer == gt_value
    if "nearest_vlm_answer" in near_e2e.columns:
        valid = near_e2e.dropna(subset=["nearest_vlm_answer"])
        top1_total = float((valid["nearest_vlm_answer"] == valid["gt_value"]).mean()) \
                     if len(valid) > 0 else float("nan")
    else:
        # Fallback: e_total_surface já tem 0/1 para nearest
        top1_total = float(near_e2e["e_total_surface"].mean()) \
                     if "e_total_surface" in near_e2e else float("nan")

    # E_geometric: Top-1 com GT grounding (valor do paper = 1.0)
    top1_geometric = 1.0

    return {
        "nearest_n_total":    n_total,
        "nearest_n_grounding_correct": n_correct,
        "nearest_grounding_accuracy": n_correct / n_total if n_total > 0 else float("nan"),
        "nearest_Top1_geometric": top1_geometric,
        "nearest_Top1_total":     top1_total,
        "nearest_Top1_grounding_cost": top1_geometric - top1_total,
    }


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

def make_figure(dist_res: dict, near_res: dict) -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.40)

    # --- (a) Distance: MAE empilhado ---
    ax = axes[0]
    mae_geo   = dist_res["distance_MAE_geometric"]   # 0.0 (definitório)
    mae_total = dist_res["distance_MAE_total"]
    mae_grd   = max(mae_total - mae_geo, 0.0)        # contribuição do grounding

    labels   = ["E_geometric\n(grounding GT)", "E_total\n(grounding real)"]
    base_bar = [mae_geo,   mae_geo]
    grd_bar  = [0.0,       mae_grd]

    ax.bar(labels, base_bar, color=C_GEOMETRIC, width=0.45, label="E_geométrico",
           edgecolor="black", linewidth=0.7, zorder=3)
    ax.bar(labels, grd_bar,  color=C_GROUNDING, width=0.45, label="ΔE_grounding",
           bottom=base_bar, edgecolor="black", linewidth=0.7, zorder=3)

    ax.set_ylabel("MAE (m)", fontsize=11)
    ax.set_title("(a)", fontsize=11, pad=8)
    ax.set_ylim(0, max(mae_total, 0.1) * 1.45)
    ax.grid(axis="y", linestyle="--", alpha=0.40, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper left")

    for xi, (b, g) in enumerate(zip(base_bar, grd_bar)):
        total = b + g
        if total > 0:
            ax.text(xi, total + 0.01, f"{total:.3f} m",
                    ha="center", va="bottom", fontsize=9)

    # Anotação taxa de grounding
    ga = dist_res["distance_grounding_accuracy"]
    ax.text(0.98, 0.96, f"Grounding correto: {ga:.0%}",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=8, color="#333333")

    # --- (b) Nearest: Top-1 ---
    ax = axes[1]
    top1_geo   = near_res["nearest_Top1_geometric"]
    top1_total = near_res["nearest_Top1_total"]
    grd_cost   = near_res["nearest_Top1_grounding_cost"]

    labels2   = ["Top-1 geometric\n(grounding GT)", "Top-1 total\n(grounding real)"]
    ax.bar(labels2, [top1_geo,   top1_total],
           color=[C_GEOMETRIC, C_TOTAL], width=0.45,
           edgecolor="black", linewidth=0.7, zorder=3)

    ax.set_ylabel("Acurácia Top-1", fontsize=11)
    ax.set_title("(b)", fontsize=11, pad=8)
    ax.set_ylim(0, 1.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(axis="y", linestyle="--", alpha=0.40, zorder=0)
    ax.set_axisbelow(True)

    for xi, val in enumerate([top1_geo, top1_total]):
        ax.text(xi, val + 0.02, f"{val:.1%}",
                ha="center", va="bottom", fontsize=9)

    ga2 = near_res["nearest_grounding_accuracy"]
    ax.text(0.98, 0.96, f"Grounding correto: {ga2:.0%}",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=8, color="#333333")

    plt.savefig(FIG_PDF, bbox_inches="tight", dpi=150)
    plt.savefig(FIG_PNG, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Figura: {FIG_PDF}")


# ---------------------------------------------------------------------------
# Tabela LaTeX
# ---------------------------------------------------------------------------

def print_latex_table(dist_res: dict, near_res: dict) -> None:
    da  = dist_res["distance_grounding_accuracy"]
    na  = near_res["nearest_grounding_accuracy"]
    mae_geo   = dist_res["distance_MAE_geometric"]
    mae_tot   = dist_res["distance_MAE_total"]
    mae_delta = dist_res["distance_MAE_grounding_contribution"]
    t1_geo    = near_res["nearest_Top1_geometric"]
    t1_tot    = near_res["nearest_Top1_total"]
    t1_delta  = near_res["nearest_Top1_grounding_cost"]

    def f(v): return f"{v:.4f}" if not np.isnan(v) else "---"

    print("\n" + "="*70)
    print("TABELA LATEX — decomposição E_total = E_grounding + E_geometric")
    print("="*70)
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Decomposição do erro de ponta a ponta no conjunto de teste "
          r"oficial. $E_{\text{geométrico}}$ corresponde ao erro com grounding "
          r"perfeito (objetos de referência); $E_{\text{total}}$ inclui os erros "
          r"de identificação introduzidos pelo módulo de grounding.}")
    print(r"\label{tab:e2e_grounding_decomposition}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Operador & Grounding correto & $E_{\text{geométrico}}$ & "
          r"$E_{\text{total}}$ & $\Delta E_{\text{grounding}}$ \\")
    print(r"\midrule")
    print(f"distance (MAE, m) & {da:.0%} & {f(mae_geo)} & "
          f"{f(mae_tot)} & {f(mae_delta)} \\\\")
    print(f"nearest (Top-1)   & {na:.0%} & {f(t1_geo)} & "
          f"{f(t1_tot)} & $-${f(t1_delta)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("="*70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Carregando resultados E2E...")
    e2e, geo = load_data()

    dist_e2e = e2e[e2e["operator"] == "distance"]
    near_e2e = e2e[e2e["operator"] == "nearest"]
    print(f"distance: {len(dist_e2e)} queries  |  nearest: {len(near_e2e)} queries")

    print("\nCalculando métricas...")
    dist_res = analyze_distance(e2e, geo)
    near_res = analyze_nearest(e2e, geo)

    print("\n--- DISTANCE ---")
    print(f"  Grounding correto:         {dist_res['distance_grounding_accuracy']:.0%} "
          f"({dist_res['distance_n_grounding_correct']}/{dist_res['distance_n_total']})")
    print(f"  MAE_geometric (GT grnd):   {dist_res['distance_MAE_geometric']:.4f} m")
    print(f"  MAE_total (real grnd):     {dist_res['distance_MAE_total']:.4f} m")
    print(f"  ΔE_grounding contribution: {dist_res['distance_MAE_grounding_contribution']:.4f} m")

    print("\n--- NEAREST ---")
    print(f"  Grounding correto:         {near_res['nearest_grounding_accuracy']:.0%} "
          f"({near_res['nearest_n_grounding_correct']}/{near_res['nearest_n_total']})")
    print(f"  Top-1_geometric (GT grnd): {near_res['nearest_Top1_geometric']:.4f}")
    print(f"  Top-1_total (real grnd):   {near_res['nearest_Top1_total']:.4f}")
    print(f"  ΔTop-1 custo grounding:    -{near_res['nearest_Top1_grounding_cost']:.4f}")

    summary = {**dist_res, **near_res}
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)
    print(f"\nSumário: {SUMMARY_CSV}")

    print("\nGerando figura...")
    make_figure(dist_res, near_res)

    print_latex_table(dist_res, near_res)
    print("Script 84 concluído.")


if __name__ == "__main__":
    main()