"""
src/grounding/spatial_descriptor.py

Generate scene-relative spatial descriptors for objects in a 3D scene.
Bilingual: English ('en') or Portuguese ('pt').

Design constraint: descriptors must be SCENE-relative, not QUERY-relative.
Encoding distances or ordering relative to a query reference object turns
the experiment into circular evaluation (the descriptor leaks the answer).

Convention: top-down render with up = (0, 1, 0).
- Image vertical axis  ↔ scene Y  (larger Y → upper in view)
- Image horizontal axis ↔ scene X (larger X → right in view)

Category labels (e.g. 'chair', 'table') are kept verbatim from the manifest
in either language — they are object IDs, not natural-language words.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

DescriptorLevel = int  # 1, 2, or 3
Language = Literal["en", "pt"]


# ----------------------------------------------------------------------
# Localization tables
# ----------------------------------------------------------------------

_LOC = {
    "en": {
        # EN has no grammatical gender — masc/fem variants collapse:
        "left_m":         "left",
        "right_m":        "right",
        "left_f":         "left",
        "right_f":        "right",
        "center_h_m":     "center",
        "center_h_f":     "center",
        "upper":          "upper",
        "lower":          "lower",
        "middle_v":       "middle",
        "scene_center":   "center of the scene",
        "side_only":      "{v} part of the scene",
        "horiz_only":     "{h_m} side of the scene",
        "quadrant":       "{v}-{h_f} area of the scene",
        # Pluralize the category when n>1: "1 of 4 chairs", not "1 of 4 chair"
        "order":          "{rank} of {n} {label_plural} (left to right)",
        "order_v":        "{rank} of {n} {label_plural} (top to bottom)",
        "near":           "near {neighbors}",
        "join":           ", ",
        "sep":            "; ",
    },
    "pt": {
        # Horizontal: masculine/feminine forms because the head noun changes:
        #   "lado esquerdo" (masc.) vs "área esquerda" (fem.)
        "left_m":         "esquerdo",
        "right_m":        "direito",
        "left_f":         "esquerda",
        "right_f":        "direita",
        "center_h_m":     "central",
        "center_h_f":     "central",
        # Vertical adjectives are invariant in PT (superior/inferior/meio):
        "upper":          "superior",
        "lower":          "inferior",
        "middle_v":       "do meio",
        "scene_center":   "centro da cena",
        "side_only":      "parte {v} da cena",          # parte superior/inferior/do meio
        "horiz_only":     "lado {h_m} da cena",         # lado esquerdo/direito/central
        "quadrant":       "área {v}-{h_f} da cena",     # área inferior-esquerda
        "order":          "{rank} de {n} {label} (esq→dir)",
        "order_v":        "{rank} de {n} {label} (cima→baixo)",
        "near":           "próximo a {neighbors}",
        "join":           ", ",
        "sep":            "; ",
    },
}


def _t(language: Language, key: str, **kwargs) -> str:
    s = _LOC[language][key]
    return s.format(**kwargs) if kwargs else s


# ----------------------------------------------------------------------
# Scene frame
# ----------------------------------------------------------------------

@dataclass
class SceneFrame:
    """Bounding rectangle of valid objects in XY (top-down projection)."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def x_extent(self) -> float:
        return max(self.x_max - self.x_min, 1e-6)

    @property
    def y_extent(self) -> float:
        return max(self.y_max - self.y_min, 1e-6)


def compute_scene_frame(scene_objects: pd.DataFrame) -> SceneFrame:
    return SceneFrame(
        x_min=float(scene_objects["centroid_xy_x"].min()),
        x_max=float(scene_objects["centroid_xy_x"].max()),
        y_min=float(scene_objects["centroid_xy_y"].min()),
        y_max=float(scene_objects["centroid_xy_y"].max()),
    )


# ----------------------------------------------------------------------
# Level 1 — quadrant in scene (3x3 grid)
# ----------------------------------------------------------------------

def _quadrant_label(x: float, y: float, frame: SceneFrame, language: Language) -> str:
    x_lo = frame.x_min + frame.x_extent / 3.0
    x_hi = frame.x_min + 2.0 * frame.x_extent / 3.0
    y_lo = frame.y_min + frame.y_extent / 3.0
    y_hi = frame.y_min + 2.0 * frame.y_extent / 3.0

    h_kind = "left" if x < x_lo else "right" if x > x_hi else "center_h"
    v_key = "upper" if y > y_hi else "lower" if y < y_lo else "middle_v"

    h_m = _t(language, f"{h_kind}_m")
    h_f = _t(language, f"{h_kind}_f")
    v   = _t(language, v_key)

    if h_kind == "center_h" and v_key == "middle_v":
        return _t(language, "scene_center")
    if h_kind == "center_h":
        return _t(language, "side_only", v=v)
    if v_key == "middle_v":
        return _t(language, "horiz_only", h_m=h_m)
    return _t(language, "quadrant", v=v, h_f=h_f)


def descriptor_level1(
    obj_row: pd.Series,
    frame: SceneFrame,
    language: Language = "en",
) -> str:
    return _quadrant_label(
        float(obj_row["centroid_xy_x"]),
        float(obj_row["centroid_xy_y"]),
        frame,
        language,
    )


# ----------------------------------------------------------------------
# Level 2 — Level 1 + ordering among same-category peers
# ----------------------------------------------------------------------

def _pluralize_en(label: str) -> str:
    """Naive English pluralizer for ScanNet category labels.

    Handles the 8 categories in this benchmark plus a generic fallback:
      chair → chairs, table → tables, door → doors, monitor → monitors,
      cabinet → cabinets, desk → desks, sofa → sofas, bed → beds,
      bookshelf → bookshelves, etc.
    """
    if label.endswith(("s", "x", "z", "ch", "sh")):
        return label + "es"
    if label.endswith("y") and len(label) > 1 and label[-2] not in "aeiou":
        return label[:-1] + "ies"
    if label.endswith("f"):
        return label[:-1] + "ves"
    if label.endswith("fe"):
        return label[:-2] + "ves"
    return label + "s"


def _peer_ordering(
    obj_row: pd.Series,
    scene_objects: pd.DataFrame,
    language: Language,
) -> str | None:
    label = obj_row["label_norm"]
    peers = scene_objects[scene_objects["label_norm"] == label]
    if len(peers) <= 1:
        return None

    x_spread = peers["centroid_xy_x"].max() - peers["centroid_xy_x"].min()
    y_spread = peers["centroid_xy_y"].max() - peers["centroid_xy_y"].min()

    if x_spread >= y_spread:
        sorted_peers = peers.sort_values("centroid_xy_x", kind="stable")
        key = "order"
    else:
        sorted_peers = peers.sort_values("centroid_xy_y", ascending=False, kind="stable")
        key = "order_v"

    rank = list(sorted_peers["object_id"]).index(obj_row["object_id"]) + 1
    n = len(sorted_peers)

    # PT keeps the singular label (Portuguese pluralization is more complex
    # and the category names are kept in English in the manifest anyway, so
    # "1 de 4 chair" reads as a label reference, not a grammatical sentence).
    # EN pluralizes for grammatical correctness.
    if language == "en":
        label_plural = _pluralize_en(label)
        return _t(language, key, rank=rank, n=n, label_plural=label_plural)
    return _t(language, key, rank=rank, n=n, label=label)


def descriptor_level2(
    obj_row: pd.Series,
    scene_objects: pd.DataFrame,
    frame: SceneFrame,
    language: Language = "en",
) -> str:
    base = descriptor_level1(obj_row, frame, language)
    ordering = _peer_ordering(obj_row, scene_objects, language)
    if ordering:
        return f"{base}{_t(language, 'sep')}{ordering}"
    return base


# ----------------------------------------------------------------------
# Level 3 — Level 2 + nearest other-category neighbors (categories only)
# ----------------------------------------------------------------------

def _nearest_other_categories(
    obj_row: pd.Series,
    scene_objects: pd.DataFrame,
    k: int,
    excluded_categories: Iterable[str],
) -> list[str]:
    own_cat = obj_row["label_norm"]
    excl = set(excluded_categories) | {own_cat}
    others = scene_objects[~scene_objects["label_norm"].isin(excl)].copy()
    if others.empty:
        return []

    dx = others["centroid_xy_x"] - obj_row["centroid_xy_x"]
    dy = others["centroid_xy_y"] - obj_row["centroid_xy_y"]
    others["_d"] = np.sqrt(dx * dx + dy * dy)
    others = others.sort_values("_d", kind="stable")

    seen: set[str] = set()
    out: list[str] = []
    for lab in others["label_norm"]:
        if lab not in seen:
            out.append(lab)
            seen.add(lab)
        if len(out) >= k:
            break
    return out


def descriptor_level3(
    obj_row: pd.Series,
    scene_objects: pd.DataFrame,
    frame: SceneFrame,
    excluded_categories: Iterable[str] = (),
    k_neighbors: int = 2,
    language: Language = "en",
) -> str:
    base = descriptor_level2(obj_row, scene_objects, frame, language)
    neighbors = _nearest_other_categories(
        obj_row, scene_objects, k=k_neighbors,
        excluded_categories=excluded_categories,
    )
    if neighbors:
        nlist = _t(language, "join").join(neighbors)
        return f"{base}{_t(language, 'sep')}{_t(language, 'near', neighbors=nlist)}"
    return base


# ----------------------------------------------------------------------
# Top-level API
# ----------------------------------------------------------------------

def describe(
    obj_row: pd.Series,
    scene_objects: pd.DataFrame,
    *,
    frame: SceneFrame | None = None,
    level: DescriptorLevel = 2,
    excluded_categories: Iterable[str] = (),
    language: Language = "en",
) -> str:
    """Generate a scene-relative descriptor for a single object."""
    if frame is None:
        frame = compute_scene_frame(scene_objects)

    if level == 1:
        return descriptor_level1(obj_row, frame, language)
    if level == 2:
        return descriptor_level2(obj_row, scene_objects, frame, language)
    if level == 3:
        return descriptor_level3(
            obj_row, scene_objects, frame,
            excluded_categories=excluded_categories,
            language=language,
        )
    raise ValueError(f"Unknown descriptor level: {level!r}")


def describe_scene(
    scene_objects: pd.DataFrame,
    *,
    level: DescriptorLevel = 2,
    excluded_categories: Iterable[str] = (),
    language: Language = "en",
) -> dict[str, str]:
    """Return {object_id: descriptor_text} for every valid object in the scene."""
    if scene_objects.empty:
        return {}
    frame = compute_scene_frame(scene_objects)
    out: dict[str, str] = {}
    for _, row in scene_objects.iterrows():
        out[row["object_id"]] = describe(
            row, scene_objects,
            frame=frame, level=level,
            excluded_categories=excluded_categories,
            language=language,
        )
    return out