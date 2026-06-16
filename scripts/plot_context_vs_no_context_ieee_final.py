import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

models = [
    "GPT-4.1",
    "Qwen-235B",
    "Qwen-32B",
    "Qwen-8B",
    "Gemini 2.5\nFlash",
    "Claude\nSonnet 4.5",
]

conditions = ["Baseline", "L1", "L2", "L3", "L2-Ref"]

# ===== Table 1: Grounding accuracy (%) =====
acc = {
    "Baseline": [31.1, 26.7, 32.2, 30.0, 36.7, 24.4],
    "L1":       [50.0, 43.3, 41.1, 34.4, 52.2, 26.7],
    "L2":       [43.3, 48.9, 37.8, 28.9, 53.3, 23.3],
    "L3":       [48.9, 46.7, 44.4, 28.9, 48.9, 23.3],
    "L2-Ref":   [44.4, 47.8, 36.7, 26.7, 53.3, 27.8],
}

# ===== Table 2: End-to-end surface MAE (m) =====
mae = {
    "Baseline": [0.884, 1.006, 0.869, 1.093, 1.298, 1.359],
    "L1":       [0.501, 0.592, 0.629, 0.981, 0.833, 1.266],
    "L2":       [0.604, 0.553, 0.671, 1.188, 0.725, 1.573],
    "L3":       [0.525, 0.604, 0.664, 1.217, 0.920, 1.382],
    "L2-Ref":   [0.505, 0.613, 0.585, 1.373, 0.800, 1.488],
}

colors = {
    "Baseline": "#1f77b4",
    "L1": "#ff7f0e",
    "L2": "#2ca02c",
    "L3": "#9467bd",
    "L2-Ref": "#d62728",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
fig.patch.set_facecolor("white")

bar_w = 0.14
x = np.arange(len(models))
offsets = np.linspace(-2, 2, len(conditions)) * bar_w

# =========================
# (a) Grounding accuracy
# =========================
ax = axes[0]
for i, cond in enumerate(conditions):
    vals = acc[cond]
    bars = ax.bar(x + offsets[i], vals, width=bar_w, label=cond, color=colors[cond])
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.8,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

ax.set_title("Grounding accuracy", fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontweight="bold")
ax.set_xlabel("Model", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 60)
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.text(-0.10, 1.05, "(a)", transform=ax.transAxes, fontsize=13, fontweight="bold")

# =========================
# (b) End-to-end surface MAE
# =========================
ax = axes[1]
for i, cond in enumerate(conditions):
    vals = mae[cond]
    bars = ax.bar(x + offsets[i], vals, width=bar_w, label=cond, color=colors[cond])
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )

ax.set_title("End-to-end surface MAE", fontweight="bold")
ax.set_ylabel(r"MAE$_{\mathrm{surf}}$ (m)", fontweight="bold")
ax.set_xlabel("Model", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.7)
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.text(-0.10, 1.05, "(b)", transform=ax.transAxes, fontsize=13, fontweight="bold")

# legenda global
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=5,
    frameon=True,
    bbox_to_anchor=(0.5, -0.02),
)

fig.text(0.5, -0.08, "Pooled across two runs ($n = 90$ per condition)", ha="center", fontsize=11)

plt.tight_layout(rect=[0, 0.08, 1, 1])

pdf_path = os.path.join(OUT_DIR, "grounding_accuracy_and_end_to_end_mae_comparison.pdf")
png_path = os.path.join(OUT_DIR, "grounding_accuracy_and_end_to_end_mae_comparison.png")

plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=600, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")