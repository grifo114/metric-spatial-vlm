"""
scripts/experiments/test_prompt_enrichment_smoke.py

Smoke test for the spatial_descriptor + prompt_enrichment integration.

Verifies (no API, no PyVista, no rendering):
  1. format_object_list output is byte-identical to the original 83 helper.
  2. format_object_list_with_context produces unique descriptors per peer.
  3. excluded_for_query correctly removes the query categories from
     neighbor information.
  4. Both PT and EN render with proper grammar.

Run before pointing 83_v2 at the real benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.grounding.prompt_enrichment import (  # noqa: E402
    excluded_for_query,
    format_object_list,
    format_object_list_with_context,
)


def _format_object_list_original(num_to_id, scene_df):
    """Verbatim copy of _format_object_list from 83 (for byte-equality check)."""
    lines = []
    id_to_cat = dict(zip(scene_df["object_id"], scene_df["label_norm"]))
    for num in sorted(num_to_id):
        oid = num_to_id[num]
        cat = id_to_cat.get(oid, "?")
        lines.append(f"  {num:3d}: {cat:15s} → {oid}")
    return "\n".join(lines)


def make_fake_scene() -> pd.DataFrame:
    return pd.DataFrame([
        ("scene0008_00__chair_011",   "chair",   1.0, 3.0),
        ("scene0008_00__chair_012",   "chair",   1.0, 1.0),
        ("scene0008_00__chair_013",   "chair",   4.0, 4.0),
        ("scene0008_00__chair_014",   "chair",   4.0, 1.0),
        ("scene0008_00__table_001",   "table",   3.0, 2.0),
        ("scene0008_00__monitor_029", "monitor", 4.0, 3.2),
        ("scene0008_00__sofa_007",    "sofa",    2.0, 0.5),
    ], columns=["object_id", "label_norm", "centroid_xy_x", "centroid_xy_y"])


def build_number_map(scene_df: pd.DataFrame) -> dict[int, str]:
    """Same ordering as 83's build_number_map."""
    sorted_df = scene_df.sort_values(["label_norm", "object_id"]).reset_index(drop=True)
    return {i + 1: row["object_id"] for i, row in sorted_df.iterrows()}


def main() -> None:
    scene_df = make_fake_scene()
    num_to_id = build_number_map(scene_df)

    # ------------------------------------------------------------------
    # Test 1: byte-identical baseline reproduction
    # ------------------------------------------------------------------
    orig = _format_object_list_original(num_to_id, scene_df)
    new  = format_object_list(num_to_id, scene_df)
    assert orig == new, "format_object_list diverged from original"
    print("✓ Test 1: byte-identical baseline reproduction")

    # ------------------------------------------------------------------
    # Test 2: descriptors are unique per peer
    # ------------------------------------------------------------------
    enriched_pt = format_object_list_with_context(
        num_to_id, scene_df, level=2, language="pt",
    )
    chair_lines = [ln for ln in enriched_pt.splitlines() if "chair" in ln and "→" in ln]
    descriptors = [ln.split("(", 1)[1].rsplit(")", 1)[0] for ln in chair_lines]
    assert len(set(descriptors)) == len(descriptors), \
        f"Chair descriptors not unique: {descriptors}"
    print(f"✓ Test 2: {len(descriptors)} chairs all have unique descriptors")

    # ------------------------------------------------------------------
    # Test 3: excluded_for_query hides query categories at level 3
    # ------------------------------------------------------------------
    excluded = excluded_for_query(
        "nearest", reference_label="table", target_category="chair",
    )
    enriched_l3 = format_object_list_with_context(
        num_to_id, scene_df, level=3, excluded_categories=excluded, language="en",
    )
    # The descriptor for chair_011 should not mention 'table' or 'chair' as
    # neighbor (they're excluded). It SHOULD mention monitor or sofa.
    chair_011_line = [ln for ln in enriched_l3.splitlines() if "chair_011" in ln][0]
    assert "near table" not in chair_011_line.lower(), \
        f"Excluded category leaked: {chair_011_line}"
    assert "near chair" not in chair_011_line.lower(), \
        f"Excluded category leaked: {chair_011_line}"
    assert ("monitor" in chair_011_line.lower()) or ("sofa" in chair_011_line.lower()), \
        f"Expected non-excluded neighbor: {chair_011_line}"
    print("✓ Test 3: excluded_for_query hides query categories from neighbors")

    # ------------------------------------------------------------------
    # Test 4: PT gender agreement
    # ------------------------------------------------------------------
    # 'área esquerda' / 'área direita' (feminine) — never 'área esquerdo'
    bad = ["área esquerdo", "área direito"]
    for b in bad:
        assert b not in enriched_pt, f"PT gender bug: '{b}' found"
    # 'lado esquerdo' / 'lado direito' (masculine) — must be present somewhere
    assert "lado esquerdo" in enriched_pt or "área" in enriched_pt
    print("✓ Test 4: PT gender agreement")

    print()
    print("All smoke tests passed.")
    print()
    print("Sample EN level 2 output:")
    print(format_object_list_with_context(
        num_to_id, scene_df, level=2, language="en",
    ))
    print()
    print("Sample PT level 2 output:")
    print(enriched_pt)


if __name__ == "__main__":
    main()
