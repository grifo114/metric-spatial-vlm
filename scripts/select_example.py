"""
select_example.py
=================
Scans benchmark prediction CSVs and ranks misgrounded distance queries
as candidates for the conceptual figure of the paper.

Uses the real schema:
  - operator, scene_id, query_id
  - gt_object_a, gt_object_b
  - grounded_a, grounded_b
  - grounding_correct
  - gt_value           (= d_surf*, ground-truth surface distance)
  - e_total_surface    (= E_grounding, since E_geom = 0 in this pipeline)

A good illustrative case is one where:
  - grounding_correct == False (misgrounded — there's an E_grounding to show)
  - e_total_surface in [0.5, 1.5] m (visible but not absurd)
  - operator == 'distance'

Output:
  - candidates.csv (all misgrounded distance rows ranked)
  - prints top-10 with the most relevant columns
"""

from pathlib import Path
import pandas as pd

RESULTS_DIR  = Path("results/benchmark_v1")
RESULTS_GLOB = "e2e_grounding_test_official_raw_*.csv"


def main():
    files = sorted(RESULTS_DIR.glob(RESULTS_GLOB))
    if not files:
        raise SystemExit(f"No CSVs in {RESULTS_DIR}/{RESULTS_GLOB}")
    print(f"Loading {len(files)} CSVs...")

    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d["__source"] = f.name
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    print(f"  total rows: {len(df)}")

    # Filter: distance operator only, and only misgrounded cases
    df = df[df["operator"] == "distance"]
    print(f"  distance rows: {len(df)}")

    # grounding_correct may be bool, "True"/"False", or 0/1 — normalize
    if df["grounding_correct"].dtype == object:
        df["grounding_correct"] = df["grounding_correct"].astype(str).str.lower().isin(
            ("true", "1", "yes"))

    miss = df[~df["grounding_correct"]].copy()
    print(f"  misgrounded distance rows: {len(miss)} "
          f"({len(miss)/max(len(df),1)*100:.1f}%)")

    # E_grounding is exactly e_total_surface here (since E_geom = 0)
    miss["E_grounding"] = miss["e_total_surface"].astype(float)

    # Score: prefer the "tellable" range, with a bell around 0.8 m
    sweet = miss.query("0.4 <= E_grounding <= 1.5").copy()
    sweet["score"] = -((sweet["E_grounding"] - 0.8) ** 2)
    sweet = sweet.sort_values("score", ascending=False)

    show_cols = ["__source", "scene_id", "query_id",
                 "descriptor_level", "language",
                 "gt_object_a", "gt_object_b",
                 "grounded_a", "grounded_b",
                 "gt_value", "e_total_surface", "E_grounding", "score"]
    show_cols = [c for c in show_cols if c in sweet.columns]

    print("\nTop 10 illustrative candidates:")
    print(sweet[show_cols].head(10).to_string(index=False))

    sweet.to_csv("candidates.csv", index=False)
    print(f"\nFull ranked list ({len(sweet)} rows) → candidates.csv")
    print("\nNext step: pick a row, then plug its values into make_panel_b.py CONFIG.")


if __name__ == "__main__":
    main()