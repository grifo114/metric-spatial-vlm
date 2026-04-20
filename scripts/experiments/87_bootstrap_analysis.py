#!/usr/bin/env python3
"""
87_bootstrap_analysis.py

Analisa os resultados do bootstrap (script 86) e produz:

  Figuras:
    fig_bootstrap_distance_ci.pdf     — estimativas com barras de IC 95%
    fig_bootstrap_nearest_stability.pdf — histograma de estabilidade

  CSV:
    bootstrap_summary.csv             — métricas agregadas

  LaTeX:
    Tabela resumo e texto para dissertação (impresso no terminal)

Uso:
    python scripts/87_bootstrap_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "benchmark_v1"
FIGS_DIR    = ROOT / "artifacts" / "figures"

DIST_CSV    = RESULTS_DIR / "bootstrap_distance_uncertainty.csv"
NEAR_CSV    = RESULTS_DIR / "bootstrap_nearest_stability.csv"
SUMMARY_CSV = RESULTS_DIR / "bootstrap_summary.csv"

FIG_DIST_PDF = FIGS_DIR / "fig_bootstrap_distance_ci.pdf"
FIG_DIST_PNG = FIGS_DIR / "fig_bootstrap_distance_ci.png"
FIG_NEAR_PDF = FIGS_DIR / "fig_bootstrap_nearest_stability.pdf"
FIG_NEAR_PNG = FIGS_DIR / "fig_bootstrap_nearest_stability.png"

# Paleta colorblind-friendly
C_DIST  = "#009E73"   # verde
C_CI    = "#56B4E9"   # azul claro (barra de erro)
C_NEAR  = "#E69F00"   # laranja
C_HIGH  = "#009E73"   # verde (alta estabilidade)
C_LOW   = "#D55E00"   # vermelho (baixa estabilidade)


# ---------------------------------------------------------------------------
# Figura 1 — Distance: estimativas com IC 95%
# ---------------------------------------------------------------------------

def fig_distance_ci(dist_df: pd.DataFrame) -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    valid = dist_df.dropna(subset=["d_surface_point", "ci_lower_95"]).copy()
    valid = valid.sort_values("d_surface_point").reset_index(drop=True)

    n = len(valid)
    x = np.arange(n)

    d_pt    = valid["d_surface_point"].values
    ci_lo   = valid["ci_lower_95"].values
    ci_hi   = valid["ci_upper_95"].values
    err_lo  = np.maximum(d_pt - ci_lo, 0)   # mínimo bootstrap = estimativa pontual
    err_hi  = ci_hi - d_pt

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.errorbar(
        x, d_pt,
        yerr=[err_lo, err_hi],
        fmt="o", color=C_DIST, ecolor=C_CI,
        elinewidth=1.4, capsize=3, capthick=1.2,
        markersize=4, zorder=4, label="$d_{\\mathrm{surface}}$ + IC 95% (barra superior)"
    )

    ax.set_xlabel("Consulta (ordenada por distância crescente)", fontsize=10)
    ax.set_ylabel("Distância de superfície (m)", fontsize=10)
    #ax.set_title(
        #f"Estimativas de $d_{{\\mathrm{{surface}}}}$ com intervalos de confiança 95% "
        #f"via bootstrap ({n} queries de distance, $B=1000$)",
        #fontsize=10, pad=8
    #)
    ax.set_xticks([])
    ax.grid(axis="y", linestyle="--", alpha=0.40, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    # Anotações de largura média
    mean_w  = float(valid["ci_width_95"].mean())
    mean_rp = float(valid["ci_relative_pct"].mean())
    ax.text(
        0.99, 0.97,
        f"Largura média IC: {mean_w:.3f} m  ({mean_rp:.1f}% relativo)",
        ha="right", va="top", transform=ax.transAxes,
        fontsize=8.5, color="#333333"
    )

    plt.tight_layout()
    plt.savefig(FIG_DIST_PDF, bbox_inches="tight", dpi=150)
    plt.savefig(FIG_DIST_PNG, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Figura salva: {FIG_DIST_PDF}")


# ---------------------------------------------------------------------------
# Figura 2 — Nearest: estabilidade bootstrap
# ---------------------------------------------------------------------------

def fig_nearest_stability(near_df: pd.DataFrame) -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    valid = near_df.dropna(subset=["gt_stability"]).copy()
    valid = valid.sort_values("gt_stability", ascending=False).reset_index(drop=True)

    n    = len(valid)
    x    = np.arange(n)
    stab = valid["gt_stability"].values
    cols = [C_HIGH if s >= 0.95 else C_LOW for s in stab]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.subplots_adjust(wspace=0.35)

    # Painel (a): bar chart por query
    ax = axes[0]
    bars = ax.bar(x, stab * 100, color=cols, edgecolor="black",
                  linewidth=0.5, zorder=3)
    ax.axhline(95, color="gray", linestyle="--", linewidth=1.0, zorder=2,
               label="Limiar 95%")
    ax.set_xlabel("Consulta nearest (ordenada por estabilidade)", fontsize=10)
    ax.set_ylabel("Estabilidade do vencedor GT (%)", fontsize=10)
    ax.set_title(f"(a)", #Estabilidade bootstrap por query ($N={n}$, $B=500$)",
                 fontsize=10, pad=6)
    ax.set_xticks([])
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.40, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    high = (stab >= 0.95).sum()
    ax.text(0.99, 0.97,
            f"Stab ≥ 95%: {high}/{n} ({high/n:.0%})",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=8.5, color="#333333")

    # Painel (b): histograma
    ax2 = axes[1]
    bins = np.linspace(0, 1, 11)
    ax2.hist(stab, bins=bins, color=C_NEAR, edgecolor="black",
             linewidth=0.6, zorder=3)
    ax2.axvline(0.95, color="gray", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Estabilidade do vencedor GT", fontsize=10)
    ax2.set_ylabel("Frequência", fontsize=10)
    ax2.set_title("(b)", #Distribuição de estabilidade",
                   fontsize=10, pad=6)
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.grid(axis="y", linestyle="--", alpha=0.40, zorder=0)
    ax2.set_axisbelow(True)

    plt.savefig(FIG_NEAR_PDF, bbox_inches="tight", dpi=150)
    plt.savefig(FIG_NEAR_PNG, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Figura salva: {FIG_NEAR_PDF}")


# ---------------------------------------------------------------------------
# Tabela LaTeX
# ---------------------------------------------------------------------------

def print_latex_table(dist_df: pd.DataFrame, near_df: pd.DataFrame) -> None:
    vd   = dist_df.dropna(subset=["ci_width_95"])
    vn   = near_df.dropna(subset=["gt_stability"])

    def f3(v): return f"{v:.3f}" if not np.isnan(v) else "---"
    def f1p(v): return f"{v:.1f}" if not np.isnan(v) else "---"
    def fp(v):  return f"{v:.0%}" if not np.isnan(v) else "---"

    mean_w  = float(vd["ci_width_95"].mean())
    med_w   = float(vd["ci_width_95"].median())
    max_w   = float(vd["ci_width_95"].max())
    mean_rp = float(vd["ci_relative_pct"].mean())

    stab_m  = float(vn["gt_stability"].mean())
    stab_md = float(vn["gt_stability"].median())
    stab_mn = float(vn["gt_stability"].min())
    n_high  = int((vn["gt_stability"] >= 0.95).sum())
    n_total = len(vn)

    print("\n" + "="*70)
    print("TABELA LATEX — bootstrap de incerteza")
    print("="*70)
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Resultados do bootstrap de incerteza ($B=1\,000$ para "
          r"\textit{distance}; $B=500$ para \textit{nearest}; subamostras de "
          r"2\,000 pontos por objeto). Para \textit{distance}: estatísticas da "
          r"largura do intervalo de confiança de 95\% por query. Para "
          r"\textit{nearest}: estabilidade da seleção do vencedor correto ao "
          r"longo das iterações bootstrap.}")
    print(r"\label{tab:bootstrap-uncertainty}")
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(r"\textbf{Operador} & \textbf{Métrica} & "
          r"\textbf{Média} & \textbf{Mediana} & \textbf{Máximo} \\")
    print(r"\midrule")
    print(f"\\textit{{distance}} & Largura IC 95\\% (m) & "
          f"{f3(mean_w)} & {f3(med_w)} & {f3(max_w)} \\\\")
    print(f"\\textit{{distance}} & Largura IC relativa (\\%) & "
          f"{f1p(mean_rp)} & --- & --- \\\\")
    print(r"\midrule")
    print(f"\\textit{{nearest}} & Estabilidade GT (\\%) & "
          f"{fp(stab_m)} & {fp(stab_md)} & --- \\\\")
    print(f"\\textit{{nearest}} & Queries com estab.~$\\geq 95$\\% & "
          f"\\multicolumn{{3}}{{c}}{{{n_high}/{n_total} "
          f"({n_high/n_total:.0%})}} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("="*70 + "\n")


# ---------------------------------------------------------------------------
# Texto para dissertação
# ---------------------------------------------------------------------------

def print_dissertation_text(dist_df: pd.DataFrame, near_df: pd.DataFrame) -> None:
    vd = dist_df.dropna(subset=["ci_width_95"])
    vn = near_df.dropna(subset=["gt_stability"])

    mean_w  = vd["ci_width_95"].mean()
    med_w   = vd["ci_width_95"].median()
    max_w   = vd["ci_width_95"].max()
    mean_rp = vd["ci_relative_pct"].mean()
    stab_m  = vn["gt_stability"].mean()
    stab_md = vn["gt_stability"].median()
    stab_mn = vn["gt_stability"].min()
    n_high  = int((vn["gt_stability"] >= 0.95).sum())
    n_near  = len(vn)
    n_dist  = len(vd)

    print("="*70)
    print("TEXTO PARA DISSERTAÇÃO (rascunho para cap6/cap7)")
    print("="*70)
    print(f"""
Para avaliar a estabilidade das estimativas de $d_{{\\text{{surface}}}}$
em relação ao conteúdo finito e potencialmente ruidoso das nuvens de
pontos, foi conduzido um experimento de bootstrap com $B = 1{'{'}000{'}'}$
iterações e subamostras de até $2{'{'}000{'}'}$ pontos por objeto.
Em cada iteração, ambas as nuvens envolvidas na consulta foram
reamostradas com reposição, e a distância mínima superfície-a-superfície
foi recomputada. O intervalo de confiança de 95\\% foi construído
pelo método do percentil.

Nos {n_dist} pares de objetos do bloco de \\textit{{distance}}, a largura
média do IC 95\\% foi de {mean_w:.3f}\\,m (mediana {med_w:.3f}\\,m,
máximo {max_w:.3f}\\,m), correspondendo a {mean_rp:.1f}\\% da distância
pontual em termos relativos. Esse resultado indica que as estimativas de
$d_{{\\text{{surface}}}}$ são estáveis: a variabilidade introduzida pelo
conteúdo finito das nuvens de pontos é pequena em relação às distâncias
medidas no benchmark.

Para o operador \\textit{{nearest}}, foi avaliada a estabilidade da
seleção do vencedor correto (GT) ao longo de $B = 500$ iterações,
em cada uma das quais a nuvem do objeto de referência foi reamostrada
com reposição enquanto as trees dos candidatos foram mantidas fixas.
A estabilidade média foi de {stab_m:.1%} (mediana {stab_md:.1%},
mínimo {stab_mn:.1%}), e {n_high} das {n_near} consultas
({n_high/n_near:.0%}) apresentaram estabilidade superior a 95\\%.
Esses valores indicam que a seleção do vizinho mais próximo é robusta
à variação amostral das nuvens de pontos: na grande maioria das
consultas, o mesmo objeto seria selecionado independentemente de quais
pontos específicos compõem a nuvem em qualquer iteração particular.
""")
    print("="*70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not DIST_CSV.exists() or not NEAR_CSV.exists():
        raise FileNotFoundError(
            "Arquivos de bootstrap não encontrados.\n"
            "Execute o script 86 primeiro:\n"
            "  python scripts/86_bootstrap_uncertainty.py"
        )

    dist_df = pd.read_csv(DIST_CSV)
    near_df = pd.read_csv(NEAR_CSV)

    vd = dist_df.dropna(subset=["ci_width_95"])
    vn = near_df.dropna(subset=["gt_stability"])

    print(f"Distance: {len(vd)}/{len(dist_df)} queries válidas")
    print(f"Nearest:  {len(vn)}/{len(near_df)} queries válidas")

    print("\n--- DISTANCE ---")
    print(f"  Largura IC 95% média:    {vd['ci_width_95'].mean():.4f} m")
    print(f"  Largura IC 95% mediana:  {vd['ci_width_95'].median():.4f} m")
    print(f"  Largura IC 95% máxima:   {vd['ci_width_95'].max():.4f} m")
    print(f"  IC relativo médio:       {vd['ci_relative_pct'].mean():.1f}%")

    print("\n--- NEAREST ---")
    print(f"  Estabilidade GT média:   {vn['gt_stability'].mean():.1%}")
    print(f"  Estabilidade GT mediana: {vn['gt_stability'].median():.1%}")
    print(f"  Estabilidade GT mínima:  {vn['gt_stability'].min():.1%}")
    n_high = int((vn["gt_stability"] >= 0.95).sum())
    print(f"  Queries stab ≥ 95%:      {n_high}/{len(vn)} ({n_high/len(vn):.0%})")

    # Salva sumário
    summary = {
        "dist_n": len(vd),
        "dist_ci_width_mean": vd["ci_width_95"].mean(),
        "dist_ci_width_median": vd["ci_width_95"].median(),
        "dist_ci_width_max": vd["ci_width_95"].max(),
        "dist_ci_relative_mean_pct": vd["ci_relative_pct"].mean(),
        "near_n": len(vn),
        "near_gt_stability_mean": vn["gt_stability"].mean(),
        "near_gt_stability_median": vn["gt_stability"].median(),
        "near_gt_stability_min": vn["gt_stability"].min(),
        "near_n_above_95pct": n_high,
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)
    print(f"\nSumário: {SUMMARY_CSV}")

    print("\nGerando figuras...")
    fig_distance_ci(dist_df)
    fig_nearest_stability(near_df)

    print_latex_table(dist_df, near_df)
    print_dissertation_text(dist_df, near_df)
    print("Script 87 concluído.")


if __name__ == "__main__":
    main()