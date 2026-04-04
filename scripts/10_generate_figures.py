"""
10_generate_figures.py

Generates all dissertation figures as PDF vector files.
Output: figures/fig1_mae_comparacao.pdf ... fig7_egrounding_egeometrico.pdf

Style: academic, white background, colorblind-friendly palette,
       12pt axis labels, Portuguese labels.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "pdf.fonttype":      42,
})

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

C = {
    "baseline_a": "#E69F00",
    "baseline_b": "#CC79A7",
    "proposto":   "#0072B2",
    "gpt":        "#009E73",
    "llava13":    "#56B4E9",
    "llava7":     "#F0E442",
    "short":      "#0072B2",
    "medium":     "#E69F00",
    "long":       "#009E73",
}

def load_all():
    return dict(
        pred_a  = pd.read_csv("results/predictions_baseline_a.csv"),
        pred_b  = pd.read_csv("results/predictions_baseline_b.csv"),
        pred_r  = pd.read_csv("results/predictions_proposed_rgbd.csv"),
        e2e_gpt = pd.read_csv("results/e2e_gpt41.csv"),
        e2e_l13 = pd.read_csv("results/e2e_llava13b.csv"),
        e2e_l7  = pd.read_csv("results/e2e_llava7b.csv"),
        gr_gpt  = pd.read_csv("results/grounding_gpt41.csv"),
        gr_l13  = pd.read_csv("results/grounding_llava13b.csv"),
        gr_l7   = pd.read_csv("results/grounding_llava7b.csv"),
    )

def add_value_labels(ax, bars, fmt="{:.3f}", offset=0.005):
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h): continue
        ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                fmt.format(h), ha="center", va="bottom",
                fontsize=10, fontweight="bold")

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def valid_mae(df):
    v = df[df["status"] == "ok"].copy()
    v["ae"] = (v["pred_distance_m"] - v["gt_distance_m"]).abs()
    return v["ae"]

# ── Fig 1 — MAE por método ────────────────────────────────────────
def fig1(d):
    methods = ["Baseline A\n(Pixel 2D)", "Baseline B\n(DPT Monocular)", "Proposto\n(RGBD)"]
    colors  = [C["baseline_a"], C["baseline_b"], C["proposto"]]
    maes    = [valid_mae(d["pred_a"]).mean(),
               valid_mae(d["pred_b"]).mean(),
               valid_mae(d["pred_r"]).mean()]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(methods, maes, color=colors, width=0.5,
                  edgecolor="white", linewidth=0.8)
    add_value_labels(ax, bars, fmt="{:.3f} m", offset=0.01)
    ax.set_ylabel("Erro Absoluto Médio (m)")
    ax.set_title("Comparação do Erro Absoluto Médio por Método")
    ax.set_ylim(0, max(maes) * 1.3)

    red = (maes[0] - maes[2]) / maes[0] * 100
    ax.annotate(f"Redução de\n{red:.1f}%",
                xy=(2, maes[2] + 0.03), xytext=(1.5, maes[0] * 0.6),
                fontsize=11, color="gray", style="italic",
                arrowprops=dict(arrowstyle="->", color="gray"))
    save(fig, "fig1_mae_comparacao.pdf")

# ── Fig 2 — MAE por faixa ─────────────────────────────────────────
def fig2(d):
    ranges  = ["short", "medium", "long"]
    labels  = ["Curta\n(< 1,5 m)", "Média\n(1,5–3,0 m)", "Longa\n(≥ 3,0 m)"]
    methods = {
        "Baseline A": (d["pred_a"], C["baseline_a"]),
        "Baseline B": (d["pred_b"], C["baseline_b"]),
        "Proposto":   (d["pred_r"], C["proposto"]),
    }
    x = np.arange(len(ranges))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, (df, color)) in enumerate(methods.items()):
        ae = valid_mae(df)
        df2 = df[df["status"] == "ok"].copy()
        df2["ae"] = (df2["pred_distance_m"] - df2["gt_distance_m"]).abs()
        maes = [df2[df2["range"] == r]["ae"].mean() for r in ranges]
        ax.bar(x + i*w, maes, w, label=label, color=color,
               edgecolor="white", linewidth=0.8)
    ax.set_xticks(x + w)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Erro Absoluto Médio (m)")
    ax.set_title("MAE por Faixa de Distância e Método")
    ax.legend()
    ax.set_ylim(0, 1.6)
    save(fig, "fig2_mae_por_faixa.pdf")

# ── Fig 3 — Boxplot ───────────────────────────────────────────────
def fig3(d):
    data   = [valid_mae(d["pred_a"]).values,
              valid_mae(d["pred_b"]).values,
              valid_mae(d["pred_r"]).values]
    labels = ["Baseline A", "Baseline B", "Proposto\n(RGBD)"]
    colors = [C["baseline_a"], C["baseline_b"], C["proposto"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylabel("Erro Absoluto (m)")
    ax.set_title("Distribuição dos Erros Absolutos por Método")
    save(fig, "fig3_boxplot_erros.pdf")

# ── Fig 4 — Scatter predito vs GT ────────────────────────────────
def fig4(d):
    v = d["pred_r"][d["pred_r"]["status"] == "ok"].copy()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    rl = {"short": "Curta", "medium": "Média", "long": "Longa"}
    rc = {"short": C["short"], "medium": C["medium"], "long": C["long"]}
    for rng, color in rc.items():
        sub = v[v["range"] == rng]
        ax.scatter(sub["gt_distance_m"], sub["pred_distance_m"],
                   c=color, label=rl[rng], alpha=0.7, s=40, edgecolors="none")
    lim = max(v["gt_distance_m"].max(), v["pred_distance_m"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Ideal")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Distância Real (m)")
    ax.set_ylabel("Distância Predita (m)")
    ax.set_title("Predito vs Real — Método Proposto (RGBD)")
    ax.legend(title="Faixa"); ax.set_aspect("equal")
    save(fig, "fig4_scatter_predito_gt.pdf")

# ── Fig 5 — Radar modelos de linguagem ───────────────────────────
def fig5(d):
    def metrics(gr, e2e):
        vg  = gr[gr["status"] == "ok"]
        acc = vg["id_correct"].mean() if len(vg) > 0 else 0
        comp = 1 - (gr["status"] != "ok").mean()
        ok  = e2e[e2e["status"] == "ok"].copy()
        ok["ae"] = (ok["pred_distance_m"] - ok["gt_distance_m"]).abs()
        cov = len(ok) / 162
        mae = ok["ae"].mean() if len(ok) > 0 else 1.0
        return acc, comp, cov, mae

    m = [metrics(d["gr_gpt"], d["e2e_gpt"]),
         metrics(d["gr_l13"], d["e2e_l13"]),
         metrics(d["gr_l7"],  d["e2e_l7"])]
    max_mae = max(x[3] for x in m)

    def norm(vals):
        a, c, cv, mae = vals
        return [a, c, cv, 1 - mae/max_mae]

    data  = [norm(x) for x in m]
    cats  = ["Acurácia\nde ID", "Conformidade\nde Formato",
             "Cobertura\ne2e", "Precisão\nMétrica"]
    N     = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    models = ["GPT-4.1", "LLaVA 13B", "LLaVA 7B"]
    colors = [C["gpt"], C["llava13"], C["llava7"]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for vals, color, label in zip(data, colors, models):
        v = vals + vals[:1]
        ax.plot(angles, v, color=color, linewidth=2, label=label)
        ax.fill(angles, v, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0,25", "0,50", "0,75", "1,00"], fontsize=9)
    ax.set_title("Comparação dos Modelos de Linguagem\n(maior = melhor)",
                 pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    save(fig, "fig5_radar_modelos_linguagem.pdf")

# ── Fig 6 — Status breakdown ──────────────────────────────────────
def fig6(d):
    methods = {
        "Baseline A":      d["pred_a"],
        "Baseline B":      d["pred_b"],
        "Proposto\n(RGBD)":d["pred_r"],
    }
    status_colors = {
        "ok":              "#0072B2",
        "no_valid_frame":  "#E69F00",
        "centroid_failed": "#CC79A7",
        "scale_error":     "#D55E00",
        "metric_error":    "#999999",
        "fusion_failed":   "#56B4E9",
    }
    status_labels = {
        "ok":              "Sucesso",
        "no_valid_frame":  "Sem frame válido",
        "centroid_failed": "Centroide falhou",
        "scale_error":     "Erro de escala",
        "metric_error":    "Erro métrico",
        "fusion_failed":   "Fusão falhou",
    }
    all_statuses = set()
    for df in methods.values():
        all_statuses |= set(df["status"].unique())

    x      = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for status in sorted(all_statuses):
        counts = [(df["status"] == status).sum() / 162 * 100
                  for df in methods.values()]
        # Skip statuses that never reach 1% in any method (invisible in chart)
        if max(counts) < 1.0:
            continue
        ax.bar(x, counts, bottom=bottom,
               color=status_colors.get(status, "gray"),
               label=status_labels.get(status, status),
               edgecolor="white", linewidth=0.5)
        bottom += np.array(counts)

    ax.set_xticks(x)
    ax.set_xticklabels(list(methods.keys()))
    ax.set_ylabel("Proporção de pares (%)")
    ax.set_title("Distribuição de Status por Método")
    ax.set_ylim(0, 115)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
    save(fig, "fig6_status_breakdown.pdf")

# ── Fig 7 — Decomposição do erro ──────────────────────────────────
def fig7(d):
    models  = ["GPT-4.1", "LLaVA 13B", "LLaVA 7B"]
    e2e_dfs = [d["e2e_gpt"], d["e2e_l13"], d["e2e_l7"]]
    colors  = [C["gpt"], C["llava13"], C["llava7"]]

    mae_ok  = []
    mae_err = []
    for e2e in e2e_dfs:
        ok = e2e[e2e["status"] == "ok"].copy()
        ok["ae"] = (ok["pred_distance_m"] - ok["gt_distance_m"]).abs()
        mae_ok.append(ok["ae"].mean() if len(ok) > 0 else np.nan)

        err = e2e[e2e["status"] == "grounding_error"].copy()
        err["ae"] = (err["pred_distance_m"] - err["gt_distance_m"]).abs()
        mae_err.append(err["ae"].mean() if len(err) > 0 else np.nan)

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars1 = ax.bar(x - w/2, mae_ok,  w, color=colors,
                   edgecolor="white", linewidth=0.8,
                   label="Grounding correto ($E_{geométrico}$)")
    bars2 = ax.bar(x + w/2, mae_err, w, color="#D55E00",
                   edgecolor="white", linewidth=0.8, alpha=0.8,
                   label="Grounding incorreto ($E_{grounding}$ domina)")

    add_value_labels(ax, bars1, fmt="{:.3f} m", offset=0.01)
    add_value_labels(ax, bars2, fmt="{:.3f} m", offset=0.01)

    geo_ref = np.nanmean(mae_ok)
    ax.axhline(geo_ref, color="gray", linestyle="--", linewidth=1,
               label=f"Nível geométrico médio ({geo_ref:.3f} m)")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Erro Absoluto Médio (m)")
    ax.set_title(r"Decomposição do Erro: $E_{geométrico}$ vs $E_{grounding}$")
    ax.legend(loc="upper left", fontsize=10)
    ymax = max(v for v in mae_err if not np.isnan(v)) * 1.3
    ax.set_ylim(0, ymax)
    save(fig, "fig7_decomposicao_erro.pdf")

# ── Main ──────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    d = load_all()
    print("Generating figures...")
    fig1(d); fig2(d); fig3(d); fig4(d)
    fig5(d); fig6(d); fig7(d)
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print("\nLaTeX usage:")
    print(r"  \includegraphics[width=0.9\textwidth]{figures/fig1_mae_comparacao.pdf}")

if __name__ == "__main__":
    main()
