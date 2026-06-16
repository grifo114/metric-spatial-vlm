import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results/benchmark_v1"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

CONDITIONS = {
    "baseline": "Baseline",
    "ctx_l1": "L1",
    "ctx_l2": "L2",
    "ctx_l3": "L3",
    "ctx_l2_ref_only": "L2-Ref",
}

CONDITION_ORDER = [
    "baseline",
    "ctx_l1",
    "ctx_l2",
    "ctx_l3",
    "ctx_l2_ref_only",
]

MODEL_ORDER = [
    "GPT-4.1",
    "Qwen3-VL-235B-A22B",
    "Qwen3-VL-32B",
    "Qwen3-VL-8B",
]

STYLES = {
    "GPT-4.1": ("o", "-"),
    "Qwen3-VL-235B-A22B": ("D", "--"),
    "Qwen3-VL-32B": ("s", "-."),
    "Qwen3-VL-8B": ("^", ":"),
}


def infer_model(filename: str) -> str:
    if "qwen3vl8b" in filename:
        return "Qwen3-VL-8B"
    if "qwen3vl32b" in filename:
        return "Qwen3-VL-32B"
    if "qwen3vl235b" in filename:
        return "Qwen3-VL-235B-A22B"
    return "GPT-4.1"


def infer_condition(filename: str) -> str | None:
    if "ctx_l2_ref_only" in filename:
        return "ctx_l2_ref_only"
    if "ctx_l1" in filename:
        return "ctx_l1"
    if "ctx_l2" in filename:
        return "ctx_l2"
    if "ctx_l3" in filename:
        return "ctx_l3"
    if "baseline" in filename:
        return "baseline"
    return None


files = sorted(glob.glob(f"{RESULTS_DIR}/e2e_grounding_test_official_raw_distance_*.csv"))

rows = []

for f in files:
    name = os.path.basename(f)
    condition = infer_condition(name)

    if condition is None:
        continue

    model = infer_model(name)
    df = pd.read_csv(f)

    rows.append({
        "model": model,
        "condition": condition,
        "file": name,
        "n": len(df),
        "correct": int(df["grounding_correct"].sum()),
    })

runs = pd.DataFrame(rows)

pooled_rows = []
for (model, condition), group in runs.groupby(["model", "condition"]):
    n = group["n"].sum()
    correct = group["correct"].sum()
    acc = 100 * correct / n

    pooled_rows.append({
        "model": model,
        "condition": condition,
        "condition_label": CONDITIONS[condition],
        "n": n,
        "correct": correct,
        "accuracy": acc,
    })

summary = pd.DataFrame(pooled_rows)

summary_path = os.path.join(OUT_DIR, "context_vs_no_context_summary.csv")
summary.to_csv(summary_path, index=False)

print("\nSummary used in figure:")
print(summary.sort_values(["model", "condition"]))
print(f"\nSaved audit table: {summary_path}")

x = np.arange(len(CONDITION_ORDER))
xlabels = [CONDITIONS[c] for c in CONDITION_ORDER]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 6.5,
})

fig, ax = plt.subplots(figsize=(3.5, 2.45))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for model in MODEL_ORDER:
    y = []
    for cond in CONDITION_ORDER:
        row = summary[
            (summary["model"] == model) &
            (summary["condition"] == cond)
        ]

        if row.empty:
            y.append(np.nan)
        else:
            y.append(row.iloc[0]["accuracy"])

    marker, linestyle = STYLES[model]

    ax.plot(
        x,
        y,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.05,
        markersize=3.8,
        label=model,
    )

ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_ylabel("Grounding accuracy (%)")
ax.set_xlabel("Spatial Context Condition")

ax.set_ylim(0, 65)
ax.set_yticks(np.arange(0, 70, 10))

ax.grid(True, axis="y", linestyle=":", linewidth=0.45, alpha=0.65)
ax.grid(False, axis="x")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False,
    fontsize=7,
    handlelength=2.0,
    columnspacing=1.2,
)

plt.tight_layout(rect=[0, 0, 1, 0.88], pad=0.4)

pdf_path = os.path.join(OUT_DIR, "context_vs_no_context_ieee_final.pdf")
png_path = os.path.join(OUT_DIR, "context_vs_no_context_ieee_final.png")

plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=600, bbox_inches="tight")

print(f"Saved figure: {pdf_path}")
print(f"Saved figure: {png_path}")