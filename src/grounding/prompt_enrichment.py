"""
src/grounding/prompt_enrichment.py

Drop-in replacement for the `_format_object_list` function in script 83.

Three modes:

  - 'original'        : exact byte-compatible reproduction of the existing
                        format (so the baseline run is preserved).
  - 'context_all'     : every object gets a scene-relative descriptor.
  - 'context_ref_only': only objects whose category matches `scope_category`
                        get a descriptor; all others print plain (like baseline).
                        Used to test whether selective context injection
                        avoids the category-confusion failure mode where
                        rich descriptors on target candidates pull the VLM
                        away from the reference.

Original line:
      1: chair           → scene0008_00__chair_011

context_all line (en, level 2):
      1: chair           (left side of the scene; 1 of 4 chair (left to right)) → scene0008_00__chair_011

context_ref_only with scope_category='table':
   table lines:
     11: table           (right side of the scene; 5 of 5 table (left to right)) → scene0008_00__table_007
   non-table lines stay plain:
      1: chair           → scene0008_00__chair_011
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from .spatial_descriptor import DescriptorLevel, Language, describe_scene


_LINE_FMT_PLAIN   = "  {num:3d}: {cat:15s} → {oid}"
_LINE_FMT_CONTEXT = "  {num:3d}: {cat:15s} ({desc}) → {oid}"


def format_object_list(
    num_to_id: dict[int, str],
    scene_objects: pd.DataFrame,
) -> str:
    """Byte-identical replacement for the existing _format_object_list."""
    id_to_cat = dict(zip(scene_objects["object_id"], scene_objects["label_norm"]))
    lines = []
    for num in sorted(num_to_id):
        oid = num_to_id[num]
        cat = id_to_cat.get(oid, "?")
        lines.append(_LINE_FMT_PLAIN.format(num=num, cat=cat, oid=oid))
    return "\n".join(lines)


def format_object_list_with_context(
    num_to_id: dict[int, str],
    scene_objects: pd.DataFrame,
    *,
    level: DescriptorLevel = 2,
    excluded_categories: Iterable[str] = (),
    language: Language = "en",
    scope_category: Optional[str] = None,
) -> str:
    """Same numbered list, with a scene-relative descriptor per object.

    Parameters
    ----------
    scope_category
        If None, every object gets a descriptor (context_all).
        If a category label (e.g. 'table'), only objects whose label_norm
        matches it get a descriptor; everything else prints plain.

        Pass the query's reference_label here to inject context selectively
        on the reference candidates only — this isolates the desambiguation
        signal from the rest of the scene and tests whether descriptors on
        the target category were creating the category-confusion failure.

    Note: descriptors are still computed using the full scene context
    (so e.g. peer ordering for tables is computed across all 5 tables in
    the scene), regardless of scope. Only the *display* is filtered.
    """
    descriptors = describe_scene(
        scene_objects, level=level,
        excluded_categories=excluded_categories,
        language=language,
    )
    id_to_cat = dict(zip(scene_objects["object_id"], scene_objects["label_norm"]))
    lines = []
    for num in sorted(num_to_id):
        oid = num_to_id[num]
        cat = id_to_cat.get(oid, "?")
        desc = descriptors.get(oid, "")

        # Apply scope filter: only inject context for the scoped category.
        in_scope = (scope_category is None) or (cat == scope_category)

        if desc and in_scope:
            lines.append(_LINE_FMT_CONTEXT.format(
                num=num, cat=cat, desc=desc, oid=oid,
            ))
        else:
            lines.append(_LINE_FMT_PLAIN.format(num=num, cat=cat, oid=oid))
    return "\n".join(lines)


def excluded_for_query(
    operator: str,
    *,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    reference_label: Optional[str] = None,
    target_category: Optional[str] = None,
) -> frozenset[str]:
    """Categories whose neighbors should be hidden at level 3, so the
    descriptor cannot leak the answer through neighbor identities.

    For distance(a, b)        → exclude {a, b}
    For nearest(target, ref)  → exclude {target, ref}
    """
    if operator == "distance":
        return frozenset(c for c in (label_a, label_b) if c)
    if operator == "nearest":
        return frozenset(c for c in (reference_label, target_category) if c)
    return frozenset()