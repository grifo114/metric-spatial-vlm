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
    "Gemini 2.5 Flash",
    "Claude Sonnet 4.5",
]

STYLES = {
    "GPT-4.1": ("o", "-", "#1f77b4"),
    "Qwen3-VL-235B-A22B": ("D", "--", "#ff7f0e"),
    "Qwen3-VL-32B": ("s", "-.", "#2ca02c"),
    "Qwen3-VL-8B": ("^", ":", "#d62728"),
    "Gemini 2.5 Flash": ("v", "-", "#7f3fbf"),
    "Claude Sonnet 4.5": ("h", "--", "#8b4513"),
}


def infer_model(filename: str) -> str:
    name = filename.lower()

    if "claude_sonnet45" in name or "claude-sonnet" in name:
        return "Claude Sonnet 4.5"

    if "gemini_flash" in name or "gemini" in name:
        return "Gemini 2.5 Flash"

    if "qwen3vl8b" in name or "qwen_8b" in name:
        return "Qwen3-VL-8B"

    if "qwen3vl32b" in name or "qwen_32b" in name:
        return "Qwen3-VL-32B"

    if "qwen3vl235b" in name or "qwen_235b" in name:
        return "Qwen3-VL-235B-A22B"

    return "GPT-4.1"


def infer_condition(filename: str) -> str | None:
    name = filename.lower()

    # Precisa vir antes de ctx_l2
    if "ctx_l2_ref_only" in name or "ctx_l2ref" in name:
        return "ctx_l2_ref_only"

    if "ctx_l1" in name:
        return "ctx_l1"

    if "ctx_l2" in name:
        return "ctx_l2"

    if "ctx_l3" in name:
        return "ctx_l3"

    if "baseline" in name:
        return "baseline"

    return None


files = sorted(glob.glob(f"{RESULTS_DIR}/*.csv"))

rows = []

for f in files:
    filename = os.path.basename(f)
    condition = infer_condition(filename)

    if condition is None:
        continue

    model = infer_model(filename)
    df = pd.read_csv(f)

    if "grounding_correct" not in df.columns:
        print(f"Skipping file without grounding_correct: {filename}")
        continue

    # Converte corretamente caso a coluna tenha vindo como string
    correct_col = df["grounding_correct"].astype(str).str.lower().map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    })

    # Se já veio como booleano, preserva
    if correct_col.isna().all():
        correct_col = df["grounding_correct"].astype(bool)

    valid_n = correct_col.notna().sum()
    correct = int(correct_col.sum())

    rows.append({
        "model": model,
        "condition": condition,
        "condition_label": CONDITIONS[condition],
        "file": filename,
        "n": len(df),
        "valid_n": valid_n,
        "correct": correct,
        "accuracy": 100 * correct / valid_n,
    })

runs = pd.DataFrame(rows)

if runs.empty:
    raise RuntimeError("No valid CSV files found. Check RESULTS_DIR and file names.")

pooled_rows = []

for (model, condition), group in runs.groupby(["model", "condition"]):
    valid_n = group["valid_n"].sum()
    correct = group["correct"].sum()
    accuracy = 100 * correct / valid_n

    pooled_rows.append({
        "model": model,
        "condition": condition,
        "condition_label": CONDITIONS[condition],
        "valid_n": valid_n,
        "correct": correct,
        "accuracy": accuracy,
        "num_files": len(group),
    })

summary = pd.DataFrame(pooled_rows)

summary_path = os.path.join(OUT_DIR, "accuracy_summary_all_models.csv")
summary.to_csv(summary_path, index=False)

print("\nSummary used in figure:")
print(summary.sort_values(["model", "condition"]))

print(f"\nSaved audit table: {summary_path}")

x = np.arange(len(CONDITION_ORDER))
xlabels = [CONDITIONS[c] for c in CONDITION_ORDER]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

fig, ax = plt.subplots(figsize=(12, 7))
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

    marker, linestyle, color = STYLES[model]

    ax.plot(
        x,
        y,
        marker=marker,
        linestyle=linestyle,
        color=color,
        linewidth=2.6,
        markersize=10,
        label=model,
    )

ax.set_xticks(x)
ax.set_xticklabels(xlabels)

ax.set_xlabel("Spatial Context Condition")
ax.set_ylabel("Grounding accuracy (%)")

ax.set_ylim(0, 65)
ax.set_yticks(np.arange(0, 70, 10))

ax.grid(True, axis="y", linestyle=":", linewidth=1.2, alpha=0.7)
ax.grid(False, axis="x")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)

ax.tick_params(
    axis="both",
    which="major",
    width=2,
    length=10,
    direction="out",
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.30),
    ncol=2,
    frameon=False,
    handlelength=2.8,
    columnspacing=2.0,
)

plt.tight_layout(rect=[0, 0, 1, 0.86])

pdf_path = os.path.join(OUT_DIR, "accuracy_all_models_lineplot.pdf")
png_path = os.path.join(OUT_DIR, "accuracy_all_models_lineplot.png")

plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=600, bbox_inches="tight")

print(f"Saved figure: {pdf_path}")
print(f"Saved figure: {png_path}")