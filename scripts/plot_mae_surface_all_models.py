import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

conditions = ["Baseline", "L1", "L2", "L3", "L2-Ref"]
x = np.arange(len(conditions))

mae_data = {
    "GPT-4.1": [0.884, 0.501, 0.604, 0.525, 0.505],
    "Qwen3-VL-235B-A22B": [1.006, 0.592, 0.553, 0.604, 0.613],
    "Qwen3-VL-32B": [0.869, 0.629, 0.671, 0.664, 0.585],
    "Qwen3-VL-8B": [1.093, 0.981, 1.188, 1.217, 1.373],
    "Gemini 2.5 Flash": [1.298, 0.833, 0.725, 0.920, 0.800],
    "Claude Sonnet 4.5": [1.359, 1.266, 1.573, 1.382, 1.488],
}

styles = {
    "GPT-4.1": ("o", "-", "#1f77b4"),
    "Qwen3-VL-235B-A22B": ("D", "--", "#ff7f0e"),
    "Qwen3-VL-32B": ("s", "-.", "#2ca02c"),
    "Qwen3-VL-8B": ("^", ":", "#d62728"),
    "Gemini 2.5 Flash": ("v", "-", "#7f3fbf"),
    "Claude Sonnet 4.5": ("h", "--", "#8b4513"),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 18,
    "axes.labelsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "axes.linewidth": 1.8,
})

# Mais largo e com altura suficiente para a legenda
fig, ax = plt.subplots(figsize=(14.5, 7.6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for model, values in mae_data.items():
    marker, linestyle, color = styles[model]

    ax.plot(
        x,
        values,
        label=model,
        marker=marker,
        linestyle=linestyle,
        color=color,
        linewidth=2.6,
        markersize=11,
        markeredgewidth=1.2,
    )

ax.set_xticks(x)
ax.set_xticklabels(conditions)

ax.set_xlabel("Spatial Context Condition", labelpad=18)
ax.set_ylabel("End-to-end surface MAE (m)", labelpad=20)

ax.set_ylim(0.4, 1.7)
ax.set_yticks(np.arange(0.4, 1.8, 0.2))

ax.grid(
    True,
    axis="y",
    linestyle=":",
    linewidth=1.1,
    alpha=0.65,
)

ax.grid(False, axis="x")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.8)
ax.spines["bottom"].set_linewidth(1.8)

ax.tick_params(
    axis="both",
    which="major",
    width=1.8,
    length=9,
    direction="out",
    pad=10,
)

# Legenda acima do gráfico, fora da área dos dados
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=2,
    frameon=False,
    handlelength=2.5,
    columnspacing=2.2,
    handletextpad=0.8,
    borderaxespad=0.0,
)

# Deixa espaço em cima para a legenda e nas laterais para o eixo y
fig.subplots_adjust(
    left=0.10,
    right=0.985,
    bottom=0.17,
    top=0.72,
)

png_path = os.path.join(OUT_DIR, "mae_surface_all_models_style_fixed.png")
pdf_path = os.path.join(OUT_DIR, "mae_surface_all_models_style_fixed.pdf")

plt.savefig(png_path, dpi=600)
plt.savefig(pdf_path)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")