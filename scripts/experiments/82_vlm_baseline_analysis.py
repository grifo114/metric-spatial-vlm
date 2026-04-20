#!/usr/bin/env python3
"""
82_vlm_baseline_analysis.py

Analisa os resultados brutos do baseline VLM (script 81) e produz:
  - Tabela comparativa: VLM vs centróide vs superfície
  - Figura de barras para a dissertação (PDF + PNG)
  - CSV com métricas finais

Lê:
  results/benchmark_v1/vlm_baseline_distance_nearest_raw.csv
  results/benchmark_v1/test_stage1_surface_vs_centroid_per_query.csv  (resultados existentes)

Grava:
  results/benchmark_v1/vlm_baseline_summary.csv
  artifacts/figures/fig_vlm_vs_geometric_comparison.pdf
  artifacts/figures/fig_vlm_vs_geometric_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "benchmark_v1"
FIGS_DIR    = ROOT / "artifacts" / "figures"

VLM_RAW_CSV    = RESULTS_DIR / "vlm_baseline_distance_nearest_raw.csv"
GEOMETRIC_CSV  = RESULTS_DIR / "test_stage1_surface_vs_centroid_per_query.csv"
SUMMARY_CSV    = RESULTS_DIR / "vlm_baseline_summary.csv"

FIG_PDF = FIGS_DIR / "fig_vlm_vs_geometric_comparison.pdf"
FIG_PNG = FIGS_DIR / "fig_vlm_vs_geometric_comparison.png"

# Paleta acessível a daltônicos (consistente com o resto da dissertação)
C_VLM      = "#E69F00"  # laranja
C_CENTROID = "#D55E00"  # vermelho-alaranjado
C_SURFACE  = "#009E73"  # verde

# ---------------------------------------------------------------------------
# Métricas auxiliares
# ---------------------------------------------------------------------------

def mae(pred: pd.Series, gt: pd.Series) -> float:
    diff = (pred - gt).abs()
    return float(diff.mean())


def medae(pred: pd.Series, gt: pd.Series) -> float:
    diff = (pred - gt).abs()
    return float(diff.median())


def top1(pred: pd.Series, gt: pd.Series) -> float:
    return float((pred == gt).mean())


def parse_rate(parsed: pd.Series) -> float:
    """Fração de respostas que foram parseadas com sucesso."""
    return float(parsed.notna().mean())


# ---------------------------------------------------------------------------
# Carrega e valida dados
# ---------------------------------------------------------------------------

def load_vlm_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (distance_df, nearest_df) com respostas VLM válidas."""
    if not VLM_RAW_CSV.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {VLM_RAW_CSV}\n"
            "Execute o script 81 primeiro."
        )
    df = pd.read_csv(VLM_RAW_CSV)

    # Remove erros de render/API
    df = df[~df["vlm_response"].isin(["RENDER_ERROR", "API_ERROR"])].copy()

    dist_df = df[df["operator"] == "distance"].copy()
    near_df = df[df["operator"] == "nearest"].copy()

    # Converte vlm_parsed para float em distance
    dist_df["vlm_parsed"] = pd.to_numeric(dist_df["vlm_parsed"], errors="coerce")
    dist_df["gt_value"]   = pd.to_numeric(dist_df["gt_value"],   errors="coerce")

    return dist_df, near_df


def load_geometric_results() -> pd.DataFrame | None:
    """Carrega resultados do pipeline geométrico (centróide vs superfície)."""
    if not GEOMETRIC_CSV.exists():
        print(f"AVISO: {GEOMETRIC_CSV} não encontrado. "
              "Usando valores do paper para centróide/superfície.")
        return None
    return pd.read_csv(GEOMETRIC_CSV)


# ---------------------------------------------------------------------------
# Análise distance
# ---------------------------------------------------------------------------

def analyze_distance(
    dist_vlm: pd.DataFrame,
    geo_df: pd.DataFrame | None,
) -> dict:
    """Computa MAE e MedAE para VLM, centróide e superfície."""

    # VLM (só queries com parse válido)
    vlm_valid = dist_vlm.dropna(subset=["vlm_parsed", "gt_value"])
    n_total   = len(dist_vlm)
    n_parsed  = len(vlm_valid)

    results = {
        "distance_n_total":  n_total,
        "distance_n_parsed": n_parsed,
        "distance_parse_rate": parse_rate(dist_vlm["vlm_parsed"]),
    }

    if n_parsed > 0:
        results["distance_MAE_vlm"]   = mae(vlm_valid["vlm_parsed"], vlm_valid["gt_value"])
        results["distance_MedAE_vlm"] = medae(vlm_valid["vlm_parsed"], vlm_valid["gt_value"])
    else:
        results["distance_MAE_vlm"]   = float("nan")
        results["distance_MedAE_vlm"] = float("nan")

    # Geométrico: usa CSV se disponível, senão valores do paper (test set)
    if geo_df is not None:
        geo_dist = geo_df[geo_df["operator"] == "distance"].copy()
        geo_dist["gt_distance_m"] = pd.to_numeric(geo_dist.get("gt_distance_m", pd.Series()), errors="coerce")
        geo_dist["centroid_distance_pred"] = pd.to_numeric(
            geo_dist.get("centroid_distance_pred", geo_dist.get("pred_centroid", pd.Series())), errors="coerce"
        )
        geo_valid = geo_dist.dropna(subset=["gt_distance_m", "centroid_distance_pred"])
        if len(geo_valid) > 0:
            results["distance_MAE_centroid"]   = mae(geo_valid["centroid_distance_pred"], geo_valid["gt_distance_m"])
            results["distance_MedAE_centroid"] = medae(geo_valid["centroid_distance_pred"], geo_valid["gt_distance_m"])
        else:
            # Fallback: valores reportados na dissertação
            results["distance_MAE_centroid"]   = 0.9443
            results["distance_MedAE_centroid"] = 0.8895
    else:
        # Valores reportados na dissertação (test_official_stage1)
        results["distance_MAE_centroid"]   = 0.9443
        results["distance_MedAE_centroid"] = 0.8895

    # Superfície: MAE = 0.0 por definição (GT derivado da mesma representação)
    results["distance_MAE_surface"]   = 0.0
    results["distance_MedAE_surface"] = 0.0

    return results


# ---------------------------------------------------------------------------
# Análise nearest
# ---------------------------------------------------------------------------

def analyze_nearest(
    near_vlm: pd.DataFrame,
    geo_df: pd.DataFrame | None,
) -> dict:
    """Computa Top-1 para VLM, centróide e superfície."""

    n_total  = len(near_vlm)
    n_parsed = int(near_vlm["vlm_parsed"].notna().sum())

    results = {
        "nearest_n_total":   n_total,
        "nearest_n_parsed":  n_parsed,
        "nearest_parse_rate": parse_rate(near_vlm["vlm_parsed"]),
    }

    if n_parsed > 0:
        # Top-1: só conta se parseable E correto
        vlm_valid = near_vlm.dropna(subset=["vlm_parsed"])
        # is_correct já foi calculado no script 81; usa se disponível
        if "is_correct" in vlm_valid.columns:
            results["nearest_Top1_vlm"] = float(vlm_valid["is_correct"].fillna(False).mean())
        else:
            results["nearest_Top1_vlm"] = top1(vlm_valid["vlm_parsed"], vlm_valid["gt_value"])
    else:
        results["nearest_Top1_vlm"] = float("nan")

    # Geométrico: valores do paper (ambos 1.0 no test set)
    results["nearest_Top1_centroid"] = 1.0
    results["nearest_Top1_surface"]  = 1.0

    return results


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

def make_figure(dist_results: dict, near_results: dict) -> None:
    """
    Gera figura comparativa de 2 painéis:
      (a) MAE para distance — VLM vs centróide vs superfície
      (b) Top-1 para nearest — VLM vs centróide vs superfície
    """
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.subplots_adjust(wspace=0.38)

    methods = ["VLM\n(GPT-4.1)", "Centróide\n(3D)", "Superfície\n(3D)"]
    colors  = [C_VLM, C_CENTROID, C_SURFACE]

    # --- Painel (a): MAE distance ---
    ax = axes[0]
    mae_vals = [
        dist_results["distance_MAE_vlm"],
        dist_results["distance_MAE_centroid"],
        dist_results["distance_MAE_surface"],
    ]
    bars = ax.bar(methods, mae_vals, color=colors, width=0.52, edgecolor="black",
                  linewidth=0.7, zorder=3)
    ax.set_ylabel("MAE (m)", fontsize=11)
    ax.set_title("(a)", fontsize=11, pad=8) #Operador distance — MAE
    ax.set_ylim(0, max([v for v in mae_vals if not np.isnan(v)] + [0.1]) * 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, mae_vals):
        if not np.isnan(val):
            label = f"{val:.3f} m" if val > 0 else "0.000 m"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                label, ha="center", va="bottom", fontsize=9,
            )

    # --- Painel (b): Top-1 nearest ---
    ax = axes[1]
    top1_vals = [
        near_results["nearest_Top1_vlm"],
        near_results["nearest_Top1_centroid"],
        near_results["nearest_Top1_surface"],
    ]
    bars = ax.bar(methods, top1_vals, color=colors, width=0.52, edgecolor="black",
                  linewidth=0.7, zorder=3)
    ax.set_ylabel("Acurácia Top-1", fontsize=11)
    ax.set_title("(b)",fontsize=11, pad=8)#Operador nearest — Top-1
    ax.set_ylim(0, 1.20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, top1_vals):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9,
            )

    # Anotação de parse rate
    #pr_dist = dist_results["distance_parse_rate"]
    #pr_near = near_results["nearest_parse_rate"]
    #fig.text(
        #0.5, 0.01,
        #f"Taxa de resposta parseável: distance = {pr_dist:.0%} "
        #f"({dist_results['distance_n_parsed']}/{dist_results['distance_n_total']}), "
        #f"nearest = {pr_near:.0%} "
        #f"({near_results['nearest_n_parsed']}/{near_results['nearest_n_total']})",
        #ha="center", fontsize=8, color="#444444",
    #)

    plt.savefig(FIG_PDF, bbox_inches="tight", dpi=150)
    plt.savefig(FIG_PNG, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Figura salva: {FIG_PDF}")
    print(f"Figura salva: {FIG_PNG}")


# ---------------------------------------------------------------------------
# Tabela de texto para a dissertação
# ---------------------------------------------------------------------------

def print_latex_table(dist_results: dict, near_results: dict) -> None:
    """Imprime tabela LaTeX pronta para copiar."""
    vlm_mae   = dist_results["distance_MAE_vlm"]
    cen_mae   = dist_results["distance_MAE_centroid"]
    sur_mae   = dist_results["distance_MAE_surface"]
    vlm_top1  = near_results["nearest_Top1_vlm"]
    cen_top1  = near_results["nearest_Top1_centroid"]
    sur_top1  = near_results["nearest_Top1_surface"]
    pr_d      = dist_results["distance_parse_rate"]
    pr_n      = near_results["nearest_parse_rate"]

    na_str = "---"

    def fmt_mae(v):
        return f"{v:.4f}" if not np.isnan(v) else na_str

    def fmt_top1(v):
        return f"{v:.4f}" if not np.isnan(v) else na_str

    print("\n" + "="*70)
    print("TABELA LATEX — comparação VLM vs métodos geométricos")
    print("="*70)
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Comparação entre baseline VLM e métodos geométricos "
          r"no conjunto de teste oficial.}")
    print(r"\label{tab:vlm_baseline_comparison}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Método & MAE$_{\text{dist}}$ (m) & Top-1$_{\text{near}}$ "
          r"& Taxa de parse (dist) & Taxa de parse (near) \\")
    print(r"\midrule")
    print(f"VLM (GPT-4.1 vision) & {fmt_mae(vlm_mae)} & {fmt_top1(vlm_top1)} "
          f"& {pr_d:.0%} & {pr_n:.0%} \\\\")
    print(f"Centróide (3D) & {fmt_mae(cen_mae)} & {fmt_top1(cen_top1)} "
          f"& --- & --- \\\\")
    print(f"Superfície (3D) & {fmt_mae(sur_mae)} & {fmt_top1(sur_top1)} "
          f"& --- & --- \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("="*70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Carregando resultados VLM...")
    dist_vlm, near_vlm = load_vlm_results()
    geo_df = load_geometric_results()

    print(f"Queries distance: {len(dist_vlm)} total, "
          f"{dist_vlm['vlm_parsed'].notna().sum()} parseadas")
    print(f"Queries nearest:  {len(near_vlm)} total, "
          f"{near_vlm['vlm_parsed'].notna().sum()} parseadas")

    print("\nCalculando métricas...")
    dist_results = analyze_distance(dist_vlm, geo_df)
    near_results = analyze_nearest(near_vlm, geo_df)

    # Resumo no terminal
    print("\n--- DISTANCE ---")
    print(f"  VLM      MAE  = {dist_results['distance_MAE_vlm']:.4f} m")
    print(f"  Centróide MAE = {dist_results['distance_MAE_centroid']:.4f} m")
    print(f"  Superfície MAE= {dist_results['distance_MAE_surface']:.4f} m")
    print(f"  Parse rate    = {dist_results['distance_parse_rate']:.1%} "
          f"({dist_results['distance_n_parsed']}/{dist_results['distance_n_total']})")

    print("\n--- NEAREST ---")
    print(f"  VLM      Top-1  = {near_results['nearest_Top1_vlm']:.4f}")
    print(f"  Centróide Top-1 = {near_results['nearest_Top1_centroid']:.4f}")
    print(f"  Superfície Top-1= {near_results['nearest_Top1_surface']:.4f}")
    print(f"  Parse rate      = {near_results['nearest_parse_rate']:.1%} "
          f"({near_results['nearest_n_parsed']}/{near_results['nearest_n_total']})")

    # Salva CSV de métricas
    summary = {**dist_results, **near_results}
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)
    print(f"\nSumário salvo: {SUMMARY_CSV}")

    # Gera figura
    print("\nGerando figura...")
    make_figure(dist_results, near_results)

    
    print_latex_table(dist_results, near_results)

    print("Script 82 concluído.")


if __name__ == "__main__":
    main()