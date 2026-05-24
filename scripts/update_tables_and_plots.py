from pathlib import Path
from math import comb, sqrt
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/benchmark_v1")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

MODEL_ORDER = [
    "GPT-4.1",
    "Qwen-235B",
    "Qwen-32B",
    "Qwen-8B",
    "Gemini 2.5 Flash",
    "Claude Sonnet 4.5",
]

CONDITION_ORDER = ["Baseline", "L1", "L2", "L3", "L2-Ref"]

STYLE_MAP = {
    "GPT-4.1": ("o", "-"),
    "Qwen-235B": ("D", "--"),
    "Qwen-32B": ("s", "-."),
    "Qwen-8B": ("^", ":"),
    "Gemini 2.5 Flash": ("v", "-"),
    "Claude Sonnet 4.5": ("P", "--"),
}

def is_result_file(path: str) -> bool:
    name = os.path.basename(path).lower()
    if not name.startswith("e2e_grounding_test_official_raw"):
        return False
    if not name.endswith(".csv"):
        return False
    if not any(k in name for k in ["baseline", "ctx_l1", "ctx_l2", "ctx_l3", "ctx_l2ref", "ctx_l2_ref_only"]):
        return False
    return True

def infer_model(name: str) -> str:
    lname = name.lower()
    if "claude_sonnet45" in lname:
        return "Claude Sonnet 4.5"
    if "gemini_flash" in lname:
        return "Gemini 2.5 Flash"
    if "qwen3vl235b" in lname:
        return "Qwen-235B"
    if "qwen3vl32b" in lname:
        return "Qwen-32B"
    if "qwen3vl8b" in lname:
        return "Qwen-8B"
    return "GPT-4.1"

def infer_condition(name: str) -> str | None:
    lname = name.lower()
    if "ctx_l2_ref_only" in lname or "ctx_l2ref" in lname:
        return "L2-Ref"
    if "ctx_l1" in lname:
        return "L1"
    if "ctx_l2" in lname:
        return "L2"
    if "ctx_l3" in lname:
        return "L3"
    if "baseline" in lname:
        return "Baseline"
    return None

def infer_run_id(name: str) -> int:
    lname = name.lower()
    if "_run2" in lname:
        return 2
    return 1

def to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

def load_all_results() -> pd.DataFrame:
    files = sorted(glob.glob(str(RESULTS_DIR / "*.csv")))
    files = [f for f in files if is_result_file(f)]

    rows = []

    for f in files:
        name = os.path.basename(f)
        model = infer_model(name)
        condition = infer_condition(name)
        run_id = infer_run_id(name)

        if condition is None:
            continue

        df = pd.read_csv(f)

        required_cols = {"query_id", "grounding_correct", "e_total_surface", "e_total_centroid"}
        if not required_cols.issubset(df.columns):
            continue

        tmp = df[["query_id", "grounding_correct", "e_total_surface", "e_total_centroid"]].copy()
        tmp["grounding_correct"] = to_bool_series(tmp["grounding_correct"])
        tmp["model"] = model
        tmp["condition"] = condition
        tmp["run_id"] = run_id
        tmp["source_file"] = name

        rows.append(tmp)

    if not rows:
        raise RuntimeError("Nenhum CSV de resultado foi encontrado em results/benchmark_v1/")

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["grounding_correct", "e_total_surface", "e_total_centroid"])
    return out

def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + (z ** 2) / n
    center = (phat + (z ** 2) / (2 * n)) / denom
    margin = (z / denom) * sqrt((phat * (1 - phat) / n) + ((z ** 2) / (4 * n ** 2)))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi

def exact_binom_two_sided(k: int, n: int, p: float = 0.5):
    if n == 0:
        return 1.0
    tail = 0.0
    for i in range(0, min(k, n - k) + 1):
        tail += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return min(1.0, 2 * tail)

def mcnemar_exact_p(b: int, c: int):
    return exact_binom_two_sided(min(b, c), b + c, 0.5)

def format_p(p: float | None) -> str:
    if p is None:
        return "---"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"

def build_accuracy_summary(df: pd.DataFrame):
    rows = []
    pair_rows = []

    for model in MODEL_ORDER:
        sub_model = df[df["model"] == model].copy()

        # pooled accuracy per condition
        for condition in CONDITION_ORDER:
            sub = sub_model[sub_model["condition"] == condition]
            n = len(sub)
            correct = int(sub["grounding_correct"].sum())
            acc = 100.0 * correct / n if n else np.nan
            lo, hi = wilson_ci(correct, n)

            rows.append({
                "model": model,
                "condition": condition,
                "n": n,
                "correct": correct,
                "accuracy": acc,
                "ci_low": 100 * lo,
                "ci_high": 100 * hi,
                "mcnemar_p": None,
            })

        # McNemar vs baseline
        baseline = sub_model[sub_model["condition"] == "Baseline"][["query_id", "run_id", "grounding_correct"]].copy()
        baseline = baseline.rename(columns={"grounding_correct": "baseline_correct"})

        for condition in CONDITION_ORDER:
            if condition == "Baseline":
                continue

            cond = sub_model[sub_model["condition"] == condition][["query_id", "run_id", "grounding_correct"]].copy()
            cond = cond.rename(columns={"grounding_correct": "cond_correct"})

            merged = baseline.merge(cond, on=["query_id", "run_id"], how="inner")
            b = int(((~merged["baseline_correct"]) & (merged["cond_correct"])).sum())
            c = int(((merged["baseline_correct"]) & (~merged["cond_correct"])).sum())
            p = mcnemar_exact_p(b, c)

            pair_rows.append({
                "model": model,
                "condition": condition,
                "b": b,
                "c": c,
                "p": p,
                "n_pairs": len(merged),
            })

    summary = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows)

    for i, row in summary.iterrows():
        if row["condition"] == "Baseline":
            continue
        hit = pair_df[
            (pair_df["model"] == row["model"]) &
            (pair_df["condition"] == row["condition"])
        ]
        if not hit.empty:
            summary.at[i, "mcnemar_p"] = float(hit.iloc[0]["p"])

    return summary, pair_df

def build_mae_summary(df: pd.DataFrame):
    rows = []
    for model in MODEL_ORDER:
        sub_model = df[df["model"] == model].copy()
        baseline_mae = None

        for condition in CONDITION_ORDER:
            sub = sub_model[sub_model["condition"] == condition]
            mae_surf = float(sub["e_total_surface"].mean())
            mae_cent = float(sub["e_total_centroid"].mean())

            if condition == "Baseline":
                baseline_mae = mae_surf
                delta = None
            else:
                delta = 100.0 * (mae_surf - baseline_mae) / baseline_mae

            rows.append({
                "model": model,
                "condition": condition,
                "mae_surf": mae_surf,
                "mae_cent": mae_cent,
                "delta_surf": delta,
                "n": len(sub),
            })

    return pd.DataFrame(rows)

def print_latex_accuracy_table(summary: pd.DataFrame):
    print("\n" + "=" * 80)
    print("TABLE 1 - GROUNDING ACCURACY (LATEX)")
    print("=" * 80)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Grounding accuracy by model and prompt condition. Values are")
    print(r"pooled across two runs ($n{=}90$). McNemar $p$-values compare each SCI")
    print(r"condition against the baseline of the same model.}")
    print(r"\label{tab:grounding}")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(r"Model & Condition & Acc.\ (\%) & 95\% CI & McNemar $p$ \\")
    print(r"\midrule")

    for m_idx, model in enumerate(MODEL_ORDER):
        sub = summary[summary["model"] == model].copy()
        sub["condition"] = pd.Categorical(sub["condition"], CONDITION_ORDER, ordered=True)
        sub = sub.sort_values("condition")

        print(rf"\multirow{{5}}{{*}}{{{model}}}")
        for j, (_, row) in enumerate(sub.iterrows()):
            cond = row["condition"]
            acc = row["accuracy"]
            ci = f"[{row['ci_low']:.1f}, {row['ci_high']:.1f}]"
            ptxt = format_p(row["mcnemar_p"])
            prefix = "  & "
            suffix = r" \\"
            print(f"{prefix}{cond:<8} & {acc:.1f} & {ci} & {ptxt} {suffix}")
        if m_idx != len(MODEL_ORDER) - 1:
            print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

def print_latex_mae_table(summary: pd.DataFrame):
    print("\n" + "=" * 80)
    print("TABLE 2 - END-TO-END MAE (LATEX)")
    print("=" * 80)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{End-to-end MAE on all queries. Values are pooled across two runs")
    print(r"($n{=}90$). $\Delta_{\mathrm{surf}}$ is the relative change in")
    print(r"$\mathrm{MAE}_{\mathrm{surf}}$ compared with the baseline of the same")
    print(r"model.}")
    print(r"\label{tab:etotal}")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(r"Model & Condition & $\mathrm{MAE}_{\mathrm{surf}}$ &")
    print(r"$\mathrm{MAE}_{\mathrm{cent}}$ & $\Delta_{\mathrm{surf}}$ \\")
    print(r"\midrule")

    for m_idx, model in enumerate(MODEL_ORDER):
        sub = summary[summary["model"] == model].copy()
        sub["condition"] = pd.Categorical(sub["condition"], CONDITION_ORDER, ordered=True)
        sub = sub.sort_values("condition")

        print(rf"\multirow{{5}}{{*}}{{{model}}}")
        for _, row in sub.iterrows():
            cond = row["condition"]
            mae_s = row["mae_surf"]
            mae_c = row["mae_cent"]
            if row["delta_surf"] is None or pd.isna(row["delta_surf"]):
                dtext = "---"
            else:
                dtext = f"${row['delta_surf']:+.1f}\\%$"
            print(f"  & {cond:<8} & {mae_s:.3f} m & {mae_c:.3f} m & {dtext} \\\\")
        if m_idx != len(MODEL_ORDER) - 1:
            print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

def plot_accuracy(summary: pd.DataFrame):
    x = np.arange(len(CONDITION_ORDER))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 18,
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 21,
        "axes.linewidth": 1.8,
    })

    fig, ax = plt.subplots(figsize=(14.5, 7.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    color_map = {
        "GPT-4.1": "#1f77b4",
        "Qwen-235B": "#ff7f0e",
        "Qwen-32B": "#2ca02c",
        "Qwen-8B": "#d62728",
        "Gemini 2.5 Flash": "#7f3fbf",
        "Claude Sonnet 4.5": "#8b4513",
    }

    marker_map = {
        "GPT-4.1": "o",
        "Qwen-235B": "D",
        "Qwen-32B": "s",
        "Qwen-8B": "^",
        "Gemini 2.5 Flash": "v",
        "Claude Sonnet 4.5": "h",
    }

    linestyle_map = {
        "GPT-4.1": "-",
        "Qwen-235B": "--",
        "Qwen-32B": "-.",
        "Qwen-8B": ":",
        "Gemini 2.5 Flash": "-",
        "Claude Sonnet 4.5": "--",
    }

    label_map = {
        "GPT-4.1": "GPT-4.1",
        "Qwen-235B": "Qwen3-VL-235B-A22B",
        "Qwen-32B": "Qwen3-VL-32B",
        "Qwen-8B": "Qwen3-VL-8B",
        "Gemini 2.5 Flash": "Gemini 2.5 Flash",
        "Claude Sonnet 4.5": "Claude Sonnet 4.5",
    }

    for model in MODEL_ORDER:
        sub = summary[summary["model"] == model].copy()
        sub["condition"] = pd.Categorical(
            sub["condition"],
            CONDITION_ORDER,
            ordered=True
        )
        sub = sub.sort_values("condition")

        y = sub["accuracy"].to_numpy()

        ax.plot(
            x,
            y,
            label=label_map[model],
            marker=marker_map[model],
            linestyle=linestyle_map[model],
            color=color_map[model],
            linewidth=2.6,
            markersize=11,
            markeredgewidth=1.2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_ORDER)

    ax.set_xlabel("Spatial Context Condition", labelpad=18)
    ax.set_ylabel("Grounding accuracy (%)", labelpad=20)

    ax.set_ylim(0, 65)
    ax.set_yticks(np.arange(0, 70, 10))

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

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.17,
        top=0.72,
    )

    plt.savefig(
        OUT_DIR / "grounding_accuracy_all_models.png",
        dpi=600
    )
    plt.savefig(
        OUT_DIR / "grounding_accuracy_all_models.pdf"
    )
    plt.close()

def plot_mae(summary: pd.DataFrame):
    x = np.arange(len(CONDITION_ORDER))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 18,
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 21,
        "axes.linewidth": 1.8,
    })

    fig, ax = plt.subplots(figsize=(14.5, 7.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    color_map = {
        "GPT-4.1": "#1f77b4",
        "Qwen-235B": "#ff7f0e",
        "Qwen-32B": "#2ca02c",
        "Qwen-8B": "#d62728",
        "Gemini 2.5 Flash": "#7f3fbf",
        "Claude Sonnet 4.5": "#8b4513",
    }

    marker_map = {
        "GPT-4.1": "o",
        "Qwen-235B": "D",
        "Qwen-32B": "s",
        "Qwen-8B": "^",
        "Gemini 2.5 Flash": "v",
        "Claude Sonnet 4.5": "h",
    }

    linestyle_map = {
        "GPT-4.1": "-",
        "Qwen-235B": "--",
        "Qwen-32B": "-.",
        "Qwen-8B": ":",
        "Gemini 2.5 Flash": "-",
        "Claude Sonnet 4.5": "--",
    }

    label_map = {
        "GPT-4.1": "GPT-4.1",
        "Qwen-235B": "Qwen3-VL-235B-A22B",
        "Qwen-32B": "Qwen3-VL-32B",
        "Qwen-8B": "Qwen3-VL-8B",
        "Gemini 2.5 Flash": "Gemini 2.5 Flash",
        "Claude Sonnet 4.5": "Claude Sonnet 4.5",
    }

    for model in MODEL_ORDER:
        sub = summary[summary["model"] == model].copy()
        sub["condition"] = pd.Categorical(
            sub["condition"],
            CONDITION_ORDER,
            ordered=True
        )
        sub = sub.sort_values("condition")

        y = sub["mae_surf"].to_numpy()

        ax.plot(
            x,
            y,
            label=label_map[model],
            marker=marker_map[model],
            linestyle=linestyle_map[model],
            color=color_map[model],
            linewidth=2.6,
            markersize=11,
            markeredgewidth=1.2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_ORDER)

    ax.set_xlabel("Spatial Context Condition", labelpad=18)
    ax.set_ylabel("End-to-end surface MAE (m)", labelpad=20)

    ax.set_ylim(0.4, 1.9)
    ax.set_yticks(np.arange(0.4, 2.0, 0.2))

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

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.17,
        top=0.72,
    )

    plt.savefig(
        OUT_DIR / "end_to_end_mae_all_models.png",
        dpi=600
    )
    plt.savefig(
        OUT_DIR / "end_to_end_mae_all_models.pdf"
    )
    plt.close()

def main():
    df = load_all_results()

    acc_summary, pair_df = build_accuracy_summary(df)
    mae_summary = build_mae_summary(df)

    acc_summary.to_csv(OUT_DIR / "grounding_accuracy_summary_all_models.csv", index=False)
    pair_df.to_csv(OUT_DIR / "mcnemar_pairs_all_models.csv", index=False)
    mae_summary.to_csv(OUT_DIR / "mae_summary_all_models.csv", index=False)

    print_latex_accuracy_table(acc_summary)
    print_latex_mae_table(mae_summary)

    plot_accuracy(acc_summary)
    plot_mae(mae_summary)

    print("\nArquivos salvos em figures/:")
    print(" - grounding_accuracy_all_models.png")
    print(" - grounding_accuracy_all_models.pdf")
    print(" - end_to_end_mae_all_models.png")
    print(" - end_to_end_mae_all_models.pdf")
    print(" - grounding_accuracy_summary_all_models.csv")
    print(" - mae_summary_all_models.csv")
    print(" - mcnemar_pairs_all_models.csv")

if __name__ == "__main__":
    main()