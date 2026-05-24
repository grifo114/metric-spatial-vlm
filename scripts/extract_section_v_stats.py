#!/usr/bin/env python3
"""
extract_section_v_stats.py

Computes:

(1) Geometric residual under correct grounding
(2) Run-to-run stability

Run from repo root:
    python scripts/extract_section_v_stats.py
"""

import sys
from pathlib import Path
import pandas as pd


DEFAULT_RESULTS_DIR = Path("results/benchmark_v1")

CONDITIONS_ORDER = ["original", "ctx_l1", "ctx_l2", "ctx_l3", "ctx_l2_ref_only"]

CONDITION_LABELS = {
    "original": "Baseline",
    "ctx_l1": "L1",
    "ctx_l2": "L2",
    "ctx_l3": "L3",
    "ctx_l2_ref_only": "L2-Ref",
}

MODEL_ORDER = [
    "GPT-4.1",
    "Gemini 2.5 Flash",
    "Claude Sonnet 4.5",
    "Qwen-235B",
    "Qwen-32B",
    "Qwen-8B",
]

MODEL_DISPLAY = {
    "gpt-4.1": "GPT-4.1",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "qwen/qwen3-vl-235b-a22b-instruct": "Qwen-235B",
    "qwen/qwen3-vl-32b-instruct": "Qwen-32B",
    "qwen/qwen3-vl-8b-instruct": "Qwen-8B",
}


def infer_model_from_filename(filename: str):
    name = filename.lower()

    if "gpt41" in name or "gpt_41" in name or "gpt-4.1" in name or "gpt4" in name:
        return "GPT-4.1"

    if "claude_sonnet45" in name or "claude-sonnet" in name:
        return "Claude Sonnet 4.5"

    if "gemini_flash" in name or "gemini" in name:
        return "Gemini 2.5 Flash"

    if "qwen3vl235b" in name or "qwen_235b" in name:
        return "Qwen-235B"

    if "qwen3vl32b" in name or "qwen_32b" in name:
        return "Qwen-32B"

    if "qwen3vl8b" in name or "qwen_8b" in name:
        return "Qwen-8B"

    return None


def infer_condition_from_filename(filename: str):
    name = filename.lower()

    if "ctx_l2_ref_only" in name or "ctx_l2ref" in name:
        return "ctx_l2_ref_only"

    if "ctx_l1" in name:
        return "ctx_l1"

    if "ctx_l2" in name:
        return "ctx_l2"

    if "ctx_l3" in name:
        return "ctx_l3"

    if "baseline" in name or "original" in name:
        return "original"

    return None


def infer_run_from_filename(filename: str):
    name = filename.lower()

    if "run1" in name:
        return 1

    if "run2" in name:
        return 2

    return None


def infer_condition_from_row(row):
    mode = row.get("prompt_mode")
    level = row.get("descriptor_level")
    scope = row.get("context_scope")

    if mode == "original":
        return "original"

    if mode == "context":
        try:
            lvl = int(float(level))
        except (TypeError, ValueError):
            return None

        if scope == "reference_only" and lvl == 2:
            return "ctx_l2_ref_only"

        if lvl == 1:
            return "ctx_l1"

        if lvl == 2:
            return "ctx_l2"

        if lvl == 3:
            return "ctx_l3"

    return None


def display_model(model_id: str) -> str:
    if pd.isna(model_id):
        return "UNKNOWN"

    model_id = str(model_id)

    if model_id in MODEL_ORDER:
        return model_id

    return MODEL_DISPLAY.get(model_id, model_id)


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )


def main(results_dir: Path):
    csv_files = sorted(results_dir.glob("e2e_grounding_test_official_raw*.csv"))

    if not csv_files:
        sys.exit(f"No CSVs found in {results_dir}/")

    dfs = []

    for f in csv_files:
        name = f.name.lower()

        # Ignore old or broken files
        if name == "e2e_grounding_test_official_raw.csv":
            continue

        if "qwen2.5-vl-72b" in name:
            continue

        run_id = infer_run_from_filename(f.name)
        if run_id is None:
            continue

        model_from_name = infer_model_from_filename(f.name)
        cond_from_name = infer_condition_from_filename(f.name)

        if model_from_name is None:
            print(f"  [warn] skipping file with unknown model: {f.name}", file=sys.stderr)
            continue

        if cond_from_name is None:
            print(f"  [warn] skipping file with unknown condition: {f.name}", file=sys.stderr)
            continue

        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  [warn] failed to read {f.name}: {e}", file=sys.stderr)
            continue

        required = [
            "query_id",
            "operator",
            "grounding_correct",
            "gt_value",
            "e_total_surface",
            "e_total_centroid",
        ]

        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  [warn] skipping {f.name}, missing columns: {missing}", file=sys.stderr)
            continue

        df = df.copy()
        df["__source_file__"] = f.name
        df["__run__"] = run_id

        # Use model column when available, otherwise infer from filename.
        if "model" not in df.columns:
            df["model"] = model_from_name
        else:
            df["model"] = df["model"].fillna(model_from_name)
            df.loc[df["model"].astype(str).str.strip() == "", "model"] = model_from_name

        # Normalize display names immediately
        df["model"] = df["model"].map(display_model)

        # Use row condition when available, otherwise infer from filename.
        df["__cond__"] = df.apply(infer_condition_from_row, axis=1)
        df["__cond__"] = df["__cond__"].fillna(cond_from_name)

        # Normalize booleans
        df["grounding_correct"] = normalize_bool(df["grounding_correct"])

        dfs.append(df)

    if not dfs:
        sys.exit("No valid CSVs loaded.")

    df_all = pd.concat(dfs, ignore_index=True)

    # Restrict to valid distance rows
    df_all = df_all[df_all["operator"] == "distance"].copy()
    df_all = df_all[df_all["__cond__"].isin(CONDITIONS_ORDER)].copy()
    df_all = df_all[df_all["__run__"].isin([1, 2])].copy()
    df_all = df_all[df_all["model"].isin(MODEL_ORDER)].copy()

    print("=" * 78)
    print("LOADED DATA SUMMARY")
    print("=" * 78)

    counts = (
        df_all.groupby(["model", "__cond__", "__run__"])
        .size()
        .reset_index(name="rows")
    )

    # Sort cleanly
    counts["model"] = pd.Categorical(counts["model"], MODEL_ORDER, ordered=True)
    counts["__cond__"] = pd.Categorical(counts["__cond__"], CONDITIONS_ORDER, ordered=True)
    counts = counts.sort_values(["model", "__cond__", "__run__"])

    for _, r in counts.iterrows():
        print(
            f"  {str(r['model']):22s} | "
            f"{CONDITION_LABELS.get(r['__cond__'], r['__cond__']):8s} | "
            f"run{int(r['__run__'])} | n={int(r['rows'])}"
        )

    print()

    print("=" * 78)
    print("(1) TABLE IV: Geometric residual under correct grounding")
    print("    Pooled across the 5 prompt conditions and 2 runs.")
    print("=" * 78)
    print(f"  {'Model':22s} {'Correct pairs':>15s} {'MAE_surf':>12s} {'MAE_cent':>12s}")
    print("  " + "-" * 65)

    for model in MODEL_ORDER:
        sub = df_all[df_all["model"] == model].copy()

        correct = sub[sub["grounding_correct"] == True].copy()
        n_correct = len(correct)

        if n_correct == 0:
            mae_surf = float("nan")
            mae_cent = float("nan")
        else:
            correct["__err_surf__"] = (
                correct["e_total_surface"].astype(float)
                - correct["gt_value"].astype(float)
            ).abs()

            correct["__err_cent__"] = (
                correct["e_total_centroid"].astype(float)
                - correct["gt_value"].astype(float)
            ).abs()

            mae_surf = correct["__err_surf__"].mean()
            mae_cent = correct["__err_cent__"].mean()

        print(
            f"  {model:22s} {n_correct:>15d} "
            f"{mae_surf:>10.3f} m {mae_cent:>10.3f} m"
        )

    print()

    print("=" * 78)
    print("(2) RUN-TO-RUN STABILITY: per-query agreement between Run 1 and Run 2")
    print("    For each (model x condition) cell.")
    print("=" * 78)
    print(f"  {'Model':22s} {'Condition':10s} {'Agreement':>10s}  {'(matches/45)':>13s}")
    print("  " + "-" * 65)

    for model in MODEL_ORDER:
        sub_m = df_all[df_all["model"] == model].copy()
        per_model_agreements = []

        for cond_key in CONDITIONS_ORDER:
            sub_c = sub_m[sub_m["__cond__"] == cond_key]

            r1 = sub_c[sub_c["__run__"] == 1].set_index("query_id")["grounding_correct"]
            r2 = sub_c[sub_c["__run__"] == 2].set_index("query_id")["grounding_correct"]

            common = r1.index.intersection(r2.index)

            if len(common) == 0:
                print(
                    f"  {model:22s} "
                    f"{CONDITION_LABELS[cond_key]:10s} "
                    f"{'(missing)':>10s}"
                )
                continue

            agree_mask = r1.loc[common] == r2.loc[common]
            n_match = int(agree_mask.sum())
            n_total = int(len(common))
            pct = 100.0 * n_match / n_total

            per_model_agreements.append(pct)

            print(
                f"  {model:22s} "
                f"{CONDITION_LABELS[cond_key]:10s} "
                f"{pct:>9.1f}%  {n_match:>5d}/{n_total:<5d}"
            )

        if per_model_agreements:
            median = sorted(per_model_agreements)[len(per_model_agreements) // 2]
            print(f"  {'':22s} {'median':10s} {median:>9.1f}%")
            print("  " + "-" * 65)


if __name__ == "__main__":
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    main(results_dir)