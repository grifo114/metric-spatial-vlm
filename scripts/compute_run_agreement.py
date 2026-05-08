"""
compute_run_agreement.py
========================
Computes per-cell run-to-run agreement on grounding decisions.

A run-to-run "agreement" for a query = (run1 picked the same pair as run2).
Pair equality is symmetric: (a,b) == (b,a).

Output: a table with min/median/max agreement per model, and a flat CSV with
all cell-level values, ready to drop into the paper.
"""
from pathlib import Path
import re
import pandas as pd

RESULTS_DIR  = Path("results/benchmark_v1")
RESULTS_GLOB = "e2e_grounding_test_official_raw_*.csv"


def parse_filename(name: str) -> dict:
    """
    Extract (condition, model, run) from filenames like:
      e2e_grounding_test_official_raw_distance_baseline_en.csv               -> GPT-4.1, run 1
      e2e_grounding_test_official_raw_distance_ctx_l2_en_qwen3vl235b_run2.csv
    """
    base = name.replace("e2e_grounding_test_official_raw_", "").replace(".csv", "")
    # Run number
    m_run = re.search(r"_run(\d+)$", base)
    run = int(m_run.group(1)) if m_run else 1
    base = re.sub(r"_run\d+$", "", base)
    # Model
    m_model = re.search(r"_(qwen3vl\d+b|qwen3vl235b)$", base)
    if m_model:
        model = m_model.group(1)
        base = base[: m_model.start()]
    else:
        model = "gpt-4.1"
    # Condition: drop "distance_" prefix and "_en" suffix
    cond = base.replace("distance_", "").replace("_en", "")
    return {"condition": cond, "model": model, "run": run}


def main():
    files = sorted(RESULTS_DIR.glob(RESULTS_GLOB))
    if not files:
        raise SystemExit(f"No CSVs in {RESULTS_DIR}/{RESULTS_GLOB}")

    rows = []
    for f in files:
        meta = parse_filename(f.name)
        df = pd.read_csv(f)
        df = df[df["operator"] == "distance"]
        for k, v in meta.items():
            df[k] = v
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    print(f"Loaded {len(all_df)} rows from {len(files)} files")

    # Need run 1 and run 2 for each (model, condition, query_id)
    pivot_cols = ["model", "condition", "query_id"]
    grp = all_df.groupby(pivot_cols + ["run"])

    cell_results = []
    for (model, cond), sub in all_df.groupby(["model", "condition"]):
        runs = sub["run"].unique()
        if set(runs) != {1, 2}:
            print(f"  skip {model}/{cond}: runs={runs}")
            continue
        r1 = sub[sub["run"] == 1].set_index("query_id")[["grounded_a", "grounded_b"]]
        r2 = sub[sub["run"] == 2].set_index("query_id")[["grounded_a", "grounded_b"]]
        common = r1.index.intersection(r2.index)
        if len(common) == 0:
            continue
        r1 = r1.loc[common]
        r2 = r2.loc[common]
        # Symmetric pair equality: (a,b)==(b,a) counts as match
        eq = (
            ((r1["grounded_a"] == r2["grounded_a"]) & (r1["grounded_b"] == r2["grounded_b"])) |
            ((r1["grounded_a"] == r2["grounded_b"]) & (r1["grounded_b"] == r2["grounded_a"]))
        )
        agreement = eq.mean()
        cell_results.append({
            "model": model, "condition": cond,
            "n": len(common), "agreement": agreement,
            "n_disagree": int((~eq).sum()),
        })

    out = pd.DataFrame(cell_results).sort_values(["model", "condition"])
    print("\nPer-cell agreement:")
    print(out.to_string(index=False))

    print("\nPer-model summary:")
    summary = out.groupby("model")["agreement"].agg(["min", "median", "max", "mean"])
    summary = (summary * 100).round(1)
    summary.columns = [f"{c}_pct" for c in summary.columns]
    print(summary.to_string())

    out.to_csv("run_agreement_cells.csv", index=False)
    summary.to_csv("run_agreement_summary.csv")
    print("\nWrote run_agreement_cells.csv and run_agreement_summary.csv")

    # Also: which conditions are most/least stable?
    print("\nMost stable cells (top 5):")
    print(out.nlargest(5, "agreement")[["model","condition","n","agreement"]]
          .to_string(index=False))
    print("\nLeast stable cells (bottom 5):")
    print(out.nsmallest(5, "agreement")[["model","condition","n","agreement"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()