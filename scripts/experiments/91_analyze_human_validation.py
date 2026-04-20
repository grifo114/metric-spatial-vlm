#!/usr/bin/env python3
"""
91_analyze_human_validation.py

Analisa os resultados da anotação humana do bloco relacional e produz:

  - Cohen's kappa entre cada par de anotadores
  - Concordância de cada anotador com o critério geométrico (GT)
  - Concordância da maioria de votos com o GT
  - Casos de discordância (anotadores vs GT) para inspeção
  - Figura e tabela LaTeX para a dissertação

Lê:
  results/benchmark_v1/human_annotation_filled.csv
  (preenchido pelos anotadores com colunas annotator_1, annotator_2, annotator_3)

Uso:
    python scripts/91_analyze_human_validation.py
    python scripts/91_analyze_human_validation.py --annotators 2
"""

from __future__ import annotations

import argparse
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

FILLED_CSV  = RESULTS_DIR / "human_annotation_filled.csv"
SUMMARY_CSV = RESULTS_DIR / "human_validation_summary.csv"
DISAGR_CSV  = RESULTS_DIR / "human_validation_disagreements.csv"
FIG_PDF     = FIGS_DIR / "fig_human_validation.pdf"
FIG_PNG     = FIGS_DIR / "fig_human_validation.png"

C_GEO  = "#009E73"
C_HUM  = "#E69F00"
C_MAJ  = "#56B4E9"


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------

def cohen_kappa(y1: list[int], y2: list[int]) -> float:
    """Cohen's kappa para duas listas de rótulos binários."""
    n    = len(y1)
    if n == 0:
        return float("nan")
    p_obs = sum(a == b for a, b in zip(y1, y2)) / n
    p1_a  = sum(y1) / n
    p1_b  = sum(y2) / n
    p_exp = p1_a * p1_b + (1 - p1_a) * (1 - p1_b)
    if p_exp >= 1.0:
        return 1.0
    return (p_obs - p_exp) / (1 - p_exp)


def accuracy(y_pred: list[int], y_true: list[int]) -> float:
    if not y_pred:
        return float("nan")
    return sum(p == t for p, t in zip(y_pred, y_true)) / len(y_pred)


def majority_vote(rows: pd.DataFrame, ann_cols: list[str]) -> list[int]:
    votes = []
    for _, row in rows.iterrows():
        vals = [int(row[c]) for c in ann_cols if pd.notna(row[c]) and row[c] != ""]
        if not vals:
            votes.append(-1)
        else:
            votes.append(1 if sum(vals) > len(vals) / 2 else 0)
    return votes


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

def make_figure(summary: dict, ann_cols: list[str]) -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.subplots_adjust(wspace=0.40)

    # Painel (a): concordância com GT por anotador + maioria
    # Painel (a)
    ax = axes[0]
    ax.set_title("(a)", fontsize=11, fontweight="normal", loc="center", pad=8)
    labels  = [f"Anotador {i+1}" for i in range(len(ann_cols))] + ["Maioria"]
    acc_vals = [summary[f"acc_ann_{i+1}_vs_gt"] for i in range(len(ann_cols))]
    acc_vals.append(summary["acc_majority_vs_gt"])
    colors  = [C_HUM] * len(ann_cols) + [C_MAJ]

    bars = ax.bar(labels, acc_vals, color=colors, edgecolor="black",
                  linewidth=0.7, zorder=3)
    ax.set_ylim(0, 1.20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_ylabel("Concordância com critério geométrico", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.40, zorder=0)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, acc_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=9)

    # Painel (b): kappa entre pares
    # Painel (b)
    ax2 = axes[1]
    ax2.set_title("(b)", fontsize=11, fontweight="normal", loc="center", pad=8)
    pairs  = []
    kappas = []
    for i in range(len(ann_cols)):
        for j in range(i+1, len(ann_cols)):
            pairs.append(f"A{i+1}–A{j+1}")
            kappas.append(summary.get(f"kappa_ann{i+1}_ann{j+1}", float("nan")))

    ax2.bar(pairs, kappas, color=C_HUM, edgecolor="black",
            linewidth=0.7, zorder=3)
    ax2.set_ylim(-0.1, 1.20)
    ax2.set_ylabel("Cohen's κ", fontsize=10)
    ax2.axhline(0.60, color="gray", linestyle="--", linewidth=1.0,
                label="κ = 0,60 (concordância substancial)")
    ax2.axhline(0.80, color="black", linestyle=":", linewidth=1.0,
                label="κ = 0,80 (concordância quase perfeita)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.40, zorder=0)
    ax2.set_axisbelow(True)
    for xi, val in enumerate(kappas):
        if not np.isnan(val):
            ax2.text(xi, val + 0.02, f"{val:.2f}",
                     ha="center", va="bottom", fontsize=9)

    plt.savefig(FIG_PDF, bbox_inches="tight", dpi=150)
    plt.savefig(FIG_PNG, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Figura: {FIG_PDF}")


# ---------------------------------------------------------------------------
# Tabela LaTeX
# ---------------------------------------------------------------------------

def print_latex(summary: dict, ann_cols: list[str]) -> None:
    def f(v): return f"{v:.2f}" if not np.isnan(v) else "---"
    def fp(v): return f"{v:.1%}" if not np.isnan(v) else "---"

    print("\n" + "="*70)
    print("TABELA LATEX — validação humana")
    print("="*70)
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Resultados da validação humana do bloco relacional. "
          r"A concordância com o critério geométrico indica a proporção de "
          r"exemplos em que o julgamento humano coincide com o rótulo definido "
          r"pelo operador geométrico. O $\kappa$ de Cohen mede a concordância "
          r"entre pares de anotadores, descontando o acaso.}")
    print(r"\label{tab:human-validation}")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"\textbf{Comparação} & \textbf{Concordância} & \textbf{$\kappa$} \\")
    print(r"\midrule")
    for i in range(len(ann_cols)):
        acc = summary.get(f"acc_ann_{i+1}_vs_gt", float("nan"))
        print(f"Anotador {i+1} vs critério geométrico & {fp(acc)} & --- \\\\")
    print(r"\midrule")
    print(f"Maioria vs critério geométrico & "
          f"{fp(summary.get('acc_majority_vs_gt', float('nan')))} & --- \\\\")
    print(r"\midrule")
    for i in range(len(ann_cols)):
        for j in range(i+1, len(ann_cols)):
            k = summary.get(f"kappa_ann{i+1}_ann{j+1}", float("nan"))
            print(f"Anotador {i+1} vs Anotador {j+1} & --- & {f(k)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("="*70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotators", type=int, default=3,
                        help="Número de anotadores (default=3)")
    parser.add_argument("--filled", type=str, default=str(FILLED_CSV),
                        help="CSV preenchido pelos anotadores")
    args = parser.parse_args()

    filled_path = Path(args.filled)
    if not filled_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {filled_path}\n"
            "Execute o script 90 primeiro e preencha o CSV template."
        )

    # Detecta separador automaticamente
    sep = ";" if ";" in open(filled_path).readline() else ","
    df = pd.read_csv(filled_path, sep=sep)

    ann_cols = [f"annotator_{i+1}" for i in range(args.annotators)]
    for col in ann_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no CSV.")

    # Filtra apenas linhas com todas as anotações preenchidas
    df_valid = df.copy()
    for col in ann_cols:
        df_valid = df_valid[df_valid[col].notna() & (df_valid[col] != "")]
    df_valid = df_valid.copy()
    for col in ann_cols:
        df_valid[col] = df_valid[col].astype(int)

    print(f"Exemplos com anotação completa: {len(df_valid)}/{len(df)}")
    print(f"  between: {(df_valid['operator']=='between').sum()}")
    print(f"  aligned: {(df_valid['operator']=='aligned').sum()}")

    gt = df_valid["geometric_label"].tolist()
    summary = {"n_total": len(df_valid)}

    # Concordância de cada anotador com GT
    for i, col in enumerate(ann_cols):
        preds = df_valid[col].tolist()
        acc   = accuracy(preds, gt)
        summary[f"acc_ann_{i+1}_vs_gt"] = acc
        print(f"  Anotador {i+1} vs GT: {acc:.1%}")

    # Kappa entre pares de anotadores
    for i in range(len(ann_cols)):
        for j in range(i+1, len(ann_cols)):
            y1 = df_valid[ann_cols[i]].tolist()
            y2 = df_valid[ann_cols[j]].tolist()
            k  = cohen_kappa(y1, y2)
            summary[f"kappa_ann{i+1}_ann{j+1}"] = k
            print(f"  κ Anotador {i+1} vs {j+1}: {k:.3f}")

    # Maioria vs GT
    maj = majority_vote(df_valid, ann_cols)
    maj_valid = [(p, t) for p, t in zip(maj, gt) if p >= 0]
    acc_maj   = accuracy([p for p,_ in maj_valid], [t for _,t in maj_valid])
    summary["acc_majority_vs_gt"] = acc_maj
    print(f"  Maioria vs GT:         {acc_maj:.1%}")

    # Casos de discordância (maioria ≠ GT)
    df_valid["majority_vote"] = maj
    disagr = df_valid[df_valid["majority_vote"] != df_valid["geometric_label"]].copy()
    disagr.to_csv(DISAGR_CSV, index=False)
    print(f"\n  Casos de discordância (maioria ≠ GT): {len(disagr)}")
    if len(disagr) > 0:
        print(disagr[["ann_num","operator","natural_query",
                       "geometric_label","majority_vote"]].to_string(index=False))

    # Salva sumário
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)
    print(f"\nSumário: {SUMMARY_CSV}")

    # Análise por operador
    print("\n--- Por operador ---")
    for op in ["between", "aligned"]:
        sub = df_valid[df_valid["operator"] == op]
        if len(sub) == 0:
            continue
        maj_sub = majority_vote(sub, ann_cols)
        acc_sub = accuracy(maj_sub, sub["geometric_label"].tolist())
        print(f"  {op} (N={len(sub)}): maioria vs GT = {acc_sub:.1%}")

    make_figure(summary, ann_cols)
    print_latex(summary, ann_cols)
    print("Script 91 concluído.")


if __name__ == "__main__":
    main()