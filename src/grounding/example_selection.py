"""
src/grounding/example_selection.py

Deterministic, non-leaking example selection for prompt templates.

The original 83 prompt template ended with a static example
("scene0008_00__monitor_029, scene0008_00__table_032") hard-coded into
the string. For queries in scene0008_00 about (monitor, table), this
literally encodes a plausible answer in the prompt. For queries elsewhere,
it cites IDs that don't exist in the candidate list, which is also bad.

This module picks example IDs that:
  1. Exist in the actual scene (so the format example matches reality).
  2. Are NOT the ground truth (no answer leak).
  3. Are NOT in the same category as the query terms (no category bias).
  4. Are deterministic per query_id (reproducible across runs).

For 'distance(label_a, label_b)' returns (id_a, id_b) — two IDs whose
categories are neither label_a nor label_b, when possible.

For 'nearest(target_category, reference_label)' returns (id_ref, None) —
one ID whose category is neither target_category nor reference_label,
when possible.

Falls back gracefully if the scene doesn't have enough non-query objects.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import pandas as pd


def _seed(s: str) -> int:
    """Stable 64-bit seed from a string."""
    return int(hashlib.md5(s.encode()).hexdigest()[:16], 16)


def _pick_two_distinct(ids: list[str], seed_int: int) -> tuple[str, str]:
    """Pick two distinct ids deterministically given a seed."""
    n = len(ids)
    i = seed_int % n
    # Use a different stride for j so it's unlikely to collide with i
    j = (seed_int // max(n, 1) + 1) % n
    if j == i:
        j = (j + 1) % n
    return ids[i], ids[j]


def pick_example_ids(
    scene_df: pd.DataFrame,
    operator: str,
    *,
    gt_object_a: Optional[str] = None,
    gt_object_b: Optional[str] = None,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    reference_label: Optional[str] = None,
    target_category: Optional[str] = None,
    seed_key: str = "",
) -> tuple[Optional[str], Optional[str]]:
    """Pick example object_ids for the prompt template.

    Returns
    -------
    (id_a, id_b)
        For operator='distance': two distinct ids, neither GT nor in
            {label_a, label_b} when possible.
        For operator='nearest': (id_ref, None) — one id, neither GT nor in
            {reference_label, target_category} when possible.
        Returns (None, None) only for degenerate scenes too small to draw from.
    """
    valid = scene_df[scene_df["is_valid_object"] == True].copy()
    if valid.empty:
        return None, None

    seed_int = _seed(seed_key)

    if operator == "distance":
        gt_set = {x for x in (gt_object_a, gt_object_b) if x}
        cat_excl = {x for x in (label_a, label_b) if x}

        # Tier 1: not GT and not in query categories
        primary = valid[
            (~valid["object_id"].isin(gt_set)) &
            (~valid["label_norm"].isin(cat_excl))
        ]
        if len(primary) >= 2:
            ids = sorted(primary["object_id"].tolist())
            return _pick_two_distinct(ids, seed_int)

        # Tier 2: just not GT (allow query categories — better than nothing)
        fallback = valid[~valid["object_id"].isin(gt_set)]
        if len(fallback) >= 2:
            ids = sorted(fallback["object_id"].tolist())
            return _pick_two_distinct(ids, seed_int)

        # Tier 3: degenerate
        return None, None

    if operator == "nearest":
        gt_set = {x for x in (gt_object_a,) if x}
        cat_excl = {x for x in (reference_label, target_category) if x}

        primary = valid[
            (~valid["object_id"].isin(gt_set)) &
            (~valid["label_norm"].isin(cat_excl))
        ]
        if len(primary) >= 1:
            ids = sorted(primary["object_id"].tolist())
            return ids[seed_int % len(ids)], None

        fallback = valid[~valid["object_id"].isin(gt_set)]
        if len(fallback) >= 1:
            ids = sorted(fallback["object_id"].tolist())
            return ids[seed_int % len(ids)], None

        return None, None

    return None, None