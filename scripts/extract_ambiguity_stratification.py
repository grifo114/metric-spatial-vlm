#!/usr/bin/env python3
"""
extract_ambiguity_stratification.py

Cross-tabulates grounding accuracy by (model x SCI condition x ambiguity
stratum). Tests the hypothesis that SCI helps most on pairs where the
category alone does not resolve the reference.

Strata follow Table I of the paper:
  - Both unambiguous: n_a = 1 and n_b = 1
  - One ambiguous:    n_a = 1 and n_b >= 2  (or vice versa)
  - Both ambiguous:   n_a >= 2 and n_b >= 2

Usage (from repo root):
    python scripts/extract_ambiguity_stratification.py
"""

import sys
import re
from pathlib import Path
import pandas as pd

DEFAULT_RESULTS_DIR = Path("results/benchmark_v1")
MANIFEST_CSV        = Path("benchmark/objects_manifest_test_official_stage1.csv")

CONDITIONS_ORDER = ["original", "ctx_l1", "ctx_l2", "ctx_l3", "ctx_l2_ref_only"]
CONDITION_LABELS = {
    "original":         "Baseline",
    "ctx_l1":           "L1",
    "ctx_l2":           "L2",
    "ctx_l3":           "L3",
    "ctx_l2_ref_only":  "L2-Ref",
}

MODEL_DISPLAY = {
    "gpt-4.1":                                "GPT-4.1",
    "claude-sonnet-4-5":                      "Claude Sonnet 4.5",
    "google/gemini-2.5-flash":                "Gemini 2.5 Flash",
    "qwen/qwen3-vl-235b-a22b-instruct":       "Qwen-235B",
    "qwen/qwen3-vl-32b-instruct":             "Qwen-32B",
    "qwen/qwen3-vl-8b-instruct":              "Qwen-8B",
}

STRATA_ORDER = ["Both unambiguous", "One ambiguous", "Both ambiguous"]


def infer_condition(row):
    mode  = row.get("prompt_mode")
    level = row.get("descriptor_level")
    scope = row.get("context_scope")
    if mode == "original":
        return "original"
    if mode == "context":
        try:
            lvl = int(level)
        except (TypeError, ValueError):
            return None
        if scope == "reference_only" and lvl == 2:
            return "ctx_l2_ref_only"
        if lvl == 1: return "ctx_l1"
        if lvl == 2: return "ctx_l2"
        if lvl == 3: return "ctx_l3"
    return None


def model_from_filename(name):
    n = name.lower()
    if "gpt41" in n or "gpt-4.1" in n or "gpt_41" in n:
        return "gpt-4.1"
    if "claude" in n:
        return "claude-sonnet-4-5"
    if "gemini" in n:
        return "google/gemini-2.5-flash"
    if "qwen3vl235b" in n or "235b" in n:
        return "qwen/qwen3-vl-235b-a22b-instruct"
    if "qwen3vl32b" in n or "32b" in n:
        return "qwen/qwen3-vl-32b-instruct"
    if "qwen3vl8b" in n or "8b" in n:
        return "qwen/qwen3-vl-8b-instruct"
    if "qwen2.5-vl-72b" in n or "qwen72b" in n:
        return "qwen/qwen2.5-vl-72b-instruct"
    return None


def extract_category(object_id):
    if not isinstance(object_id, str):
        return None
    m = re.match(r"scene\d+_\d+__(.+)_\d+$", object_id)
    return m.group(1) if m else None


def stratum_for(n_a, n_b):
    if n_a == 1 and n_b == 1:
        return "Both unambiguous"
    if n_a >= 2 and n_b >= 2:
        return "Both ambiguous"
    return "One ambiguous"


def display_model(m):
    return MODEL_DISPLAY.get(m, m)


def main(results_dir):
    manifest = pd.read_csv(MANIFEST_CSV)
    manifest = manifest[manifest["is_valid_object"] == True].copy()
    scene_cat_counts = (manifest
                        .groupby(["scene_id", "label_norm"])
                        .size()
                        .reset_index(name="n_instances"))

    csvs = sorted(results_dir.glob("e2e_grounding_test_official_raw*.csv"))
    if not csvs:
        sys.exit(f"No CSVs found in {results_dir}/")

    dfs = []
    for f in csvs:
        try:
            d = pd.read_csv(f)
            d["__source_file__"] = f.name
            dfs.append(d)
        except Exception as e:
            print(f"  [warn] failed to read {f.name}: {e}", file=sys.stderr)

    df = pd.concat(dfs, ignore_index=True)

    if "model" not in df.columns:
        df["model"] = None
    df["model"] = df.apply(
        lambda r: r["model"] if pd.notna(r["model"])
                  else model_from_filename(r["__source_file__"]),
        axis=1
    )
    pre = len(df)
    df = df[df["model"].notna()].copy()
    if len(df) < pre:
        print(f"  [warn] dropped {pre - len(df)} rows with unknown model.",
              file=sys.stderr)

    df = df[df["operator"] == "distance"].copy()
    df["__cond__"] = df.apply(infer_condition, axis=1)
    df = df[df["__cond__"].notna()].copy()

    df["category_a"] = df["gt_object_a"].map(extract_category)
    df["category_b"] = df["gt_object_b"].map(extract_category)

    df = df.merge(
        scene_cat_counts.rename(columns={"label_norm":"category_a",
                                        "n_instances":"n_a"}),
        on=["scene_id", "category_a"], how="left"
    )
    df = df.merge(
        scene_cat_counts.rename(columns={"label_norm":"category_b",
                                        "n_instances":"n_b"}),
        on=["scene_id", "category_b"], how="left"
    )

    missing = df["n_a"].isna() | df["n_b"].isna()
    if missing.sum() > 0:
        print(f"  [warn] {missing.sum()} rows could not be stratified; dropping.",
              file=sys.stderr)
        df = df[~missing].copy()

    df["stratum"] = [stratum_for(int(a), int(b))
                     for a, b in zip(df["n_a"], df["n_b"])]

    unique = (df[["query_id", "stratum"]]
              .drop_duplicates(subset=["query_id"])
              .groupby("stratum").size())
    print("=" * 78)
    print("Stratum coverage (unique distance pairs):")
    for s in STRATA_ORDER:
        print(f"  {s:20s}  {unique.get(s, 0)}")
    print(f"  Total: {unique.sum()}")
    print()

    print("=" * 78)
    print("Grounding accuracy (%) by ambiguity stratum")
    print("Cell: acc% (n). Pooled across two runs per condition.")
    print("=" * 78)
    print()

    for model_id, sub_m in df.groupby("model"):
        print(f"--- {display_model(model_id)} ---")
        print(f"  {'Stratum':20s}  " +
              " ".join(f"{CONDITION_LABELS[c]:>11s}" for c in CONDITIONS_ORDER))
        for stratum in STRATA_ORDER:
            cells = []
            for cond in CONDITIONS_ORDER:
                sub = sub_m[(sub_m["__cond__"] == cond) &
                            (sub_m["stratum"] == stratum)]
                if len(sub) == 0:
                    cells.append("--")
                    continue
                acc = 100.0 * sub["grounding_correct"].mean()
                cells.append(f"{acc:>5.1f} ({len(sub)})")
            print(f"  {stratum:20s}  " + " ".join(f"{c:>11s}" for c in cells))
        print()

    print("=" * 78)
    print("POOLED across the 5 SCI-responsive models (excluding Qwen-8B)")
    print("=" * 78)
    pooled = df[df["model"] != "qwen/qwen3-vl-8b-instruct"].copy()
    pooled = pooled[pooled["model"] != "qwen/qwen2.5-vl-72b-instruct"].copy()
    print(f"  {'Stratum':20s}  " +
          " ".join(f"{CONDITION_LABELS[c]:>11s}" for c in CONDITIONS_ORDER))
    for stratum in STRATA_ORDER:
        cells = []
        for cond in CONDITIONS_ORDER:
            sub = pooled[(pooled["__cond__"] == cond) &
                         (pooled["stratum"] == stratum)]
            acc = 100.0 * sub["grounding_correct"].mean() if len(sub) else 0
            cells.append(f"{acc:>5.1f} ({len(sub)})")
        print(f"  {stratum:20s}  " + " ".join(f"{c:>11s}" for c in cells))
    print()

    print("=" * 78)
    print("POOLED across all 6 models (Qwen-8B included)")
    print("=" * 78)
    all6 = df[df["model"] != "qwen/qwen2.5-vl-72b-instruct"].copy()
    print(f"  {'Stratum':20s}  " +
          " ".join(f"{CONDITION_LABELS[c]:>11s}" for c in CONDITIONS_ORDER))
    for stratum in STRATA_ORDER:
        cells = []
        for cond in CONDITIONS_ORDER:
            sub = all6[(all6["__cond__"] == cond) & (all6["stratum"] == stratum)]
            acc = 100.0 * sub["grounding_correct"].mean() if len(sub) else 0
            cells.append(f"{acc:>5.1f} ({len(sub)})")
        print(f"  {stratum:20s}  " + " ".join(f"{c:>11s}" for c in cells))


if __name__ == "__main__":
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    main(results_dir)