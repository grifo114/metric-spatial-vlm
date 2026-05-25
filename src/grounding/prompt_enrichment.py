from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _label(row: pd.Series) -> str:
    for col in ["label_norm", "category", "label", "class_name", "object_label"]:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return "object"


def _object_id(row: pd.Series) -> str:
    for col in ["object_id", "id", "instance_id"]:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    raise KeyError("Could not find object id column.")


def _xy(row: pd.Series) -> tuple[float, float]:
    candidates = [
        ("centroid_x", "centroid_y"),
        ("center_x", "center_y"),
        ("cx", "cy"),
        ("x", "y"),
    ]

    for x_col, y_col in candidates:
        if x_col in row and y_col in row:
            return float(row[x_col]), float(row[y_col])

    if "centroid" in row and pd.notna(row["centroid"]):
        value = row["centroid"]

        if isinstance(value, str):
            value = (
                value.replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
                .split(",")
            )

        if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
            return float(value[0]), float(value[1])

    raise KeyError(
        "Could not infer object centroid. Expected centroid_x/centroid_y, "
        "center_x/center_y, cx/cy, x/y, or centroid."
    )


def _xyz(row: pd.Series) -> np.ndarray:
    x, y = _xy(row)

    for z_col in ["centroid_z", "center_z", "cz", "z"]:
        if z_col in row:
            return np.array([x, y, float(row[z_col])], dtype=float)

    return np.array([x, y, 0.0], dtype=float)


def _scene_position(row: pd.Series, scene_df: pd.DataFrame, language: str = "en") -> str:
    xs = []
    ys = []

    for _, r in scene_df.iterrows():
        try:
            x, y = _xy(r)
            xs.append(x)
            ys.append(y)
        except Exception:
            continue

    x, y = _xy(row)

    if not xs or not ys:
        return "scene" if language == "en" else "cena"

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    if math.isclose(xmin, xmax):
        x_bin = 1
    else:
        x_bin = int(np.clip(np.floor(3 * (x - xmin) / (xmax - xmin + 1e-12)), 0, 2))

    if math.isclose(ymin, ymax):
        y_bin = 1
    else:
        y_bin = int(np.clip(np.floor(3 * (y - ymin) / (ymax - ymin + 1e-12)), 0, 2))

    if language == "pt":
        x_words = ["esquerda", "centro", "direita"]
        y_words = ["inferior", "central", "superior"]
        xw = x_words[x_bin]
        yw = y_words[y_bin]

        if x_bin == 1 and y_bin == 1:
            return "centro da cena"
        if y_bin == 1:
            return f"lado {xw} da cena"
        if x_bin == 1:
            return f"parte {yw} da cena"
        return f"área {yw}-{xw} da cena"

    x_words = ["left", "center", "right"]
    y_words = ["lower", "center", "upper"]
    xw = x_words[x_bin]
    yw = y_words[y_bin]

    if x_bin == 1 and y_bin == 1:
        return "center of the scene"
    if y_bin == 1:
        return f"{xw} side of the scene"
    if x_bin == 1:
        return f"{yw} part of the scene"
    return f"{yw}-{xw} area of the scene"


def _peer_order(row: pd.Series, scene_df: pd.DataFrame, language: str = "en") -> str | None:
    label = _label(row)
    oid = _object_id(row)

    peers = []
    for _, r in scene_df.iterrows():
        if _label(r) != label:
            continue
        try:
            x, _ = _xy(r)
            peers.append((_object_id(r), x))
        except Exception:
            continue

    if len(peers) <= 1:
        return None

    peers = sorted(peers, key=lambda t: t[1])
    ids = [p[0] for p in peers]

    if oid not in ids:
        return None

    idx = ids.index(oid) + 1
    total = len(ids)

    if language == "pt":
        return f"{idx} de {total} {label}s da esquerda para a direita"

    return f"{idx} of {total} {label}s left to right"


def _nearest_neighbor_categories(
    row: pd.Series,
    scene_df: pd.DataFrame,
    excluded_categories: Iterable[str] | None = None,
    k: int = 2,
    language: str = "en",
) -> str | None:
    excluded = {str(c).strip().lower() for c in (excluded_categories or []) if str(c).strip()}
    oid = _object_id(row)
    p = _xyz(row)

    neighbors = []

    for _, r in scene_df.iterrows():
        rid = _object_id(r)
        if rid == oid:
            continue

        cat = _label(r)
        if cat.lower() in excluded:
            continue

        try:
            q = _xyz(r)
        except Exception:
            continue

        d = float(np.linalg.norm(p - q))
        neighbors.append((d, cat))

    if not neighbors:
        return None

    neighbors = sorted(neighbors, key=lambda t: t[0])
    cats = []

    for _, cat in neighbors:
        if cat not in cats:
            cats.append(cat)
        if len(cats) >= k:
            break

    if not cats:
        return None

    if language == "pt":
        return "próximo de " + ", ".join(cats)

    return "near " + ", ".join(cats)


def excluded_for_query(
    row_or_operator=None,
    scene_df: pd.DataFrame | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
    **kwargs,
) -> set[str]:
    """
    Returns query categories to be excluded from L3 neighbor descriptors.

    Supports two call patterns:

    1. excluded_for_query(row)
       Used by experiment scripts when query information is stored in a row.

    2. excluded_for_query(operator, label_a=..., label_b=...)
       Used by the Streamlit demo.
    """
    out = set()

    if label_a:
        out.add(str(label_a).strip())

    if label_b:
        out.add(str(label_b).strip())

    for key in ["category_a", "category_b", "expected_a", "expected_b"]:
        value = kwargs.get(key)
        if value:
            out.add(str(value).strip())

    if isinstance(row_or_operator, pd.Series):
        cols = [
            "label_a",
            "label_b",
            "category_a",
            "category_b",
            "expected_a",
            "expected_b",
            "object_a_category",
            "object_b_category",
        ]

        for col in cols:
            if col in row_or_operator and pd.notna(row_or_operator[col]):
                value = str(row_or_operator[col]).strip()
                if value:
                    out.add(value)

    return {x for x in out if x}


def format_object_list(num_to_id: dict[int, str], scene_df: pd.DataFrame) -> str:
    lines = []

    for num in sorted(num_to_id):
        oid = num_to_id[num]
        match = scene_df[scene_df["object_id"].astype(str) == str(oid)]

        if match.empty:
            continue

        row = match.iloc[0]
        label = _label(row)

        lines.append(f"{num}: {label:<15} -> {oid}")

    return "\n".join(lines)


def format_object_list_with_context(
    num_to_id: dict[int, str],
    scene_df: pd.DataFrame,
    level: int = 1,
    excluded_categories: Iterable[str] | None = None,
    language: str = "en",
    scope_category: str | None = None,
) -> str:
    """
    Formats the object list with SCI descriptors.

    level = 1: scene-relative position.
    level = 2: L1 + within-category peer ordering.
    level = 3: L2 + nearest semantic-neighborhood categories.

    If scope_category is provided, descriptors are added only to that
    category. Other objects remain listed without descriptors.
    """
    lines = []
    scope = str(scope_category).strip().lower() if scope_category else None

    for num in sorted(num_to_id):
        oid = num_to_id[num]
        match = scene_df[scene_df["object_id"].astype(str) == str(oid)]

        if match.empty:
            continue

        row = match.iloc[0]
        label = _label(row)

        use_descriptor = scope is None or label.lower() == scope

        if not use_descriptor:
            lines.append(f"{num}: {label:<15} -> {oid}")
            continue

        descriptors = []

        if level >= 1:
            descriptors.append(_scene_position(row, scene_df, language=language))

        if level >= 2:
            order = _peer_order(row, scene_df, language=language)
            if order:
                descriptors.append(order)

        if level >= 3:
            neigh = _nearest_neighbor_categories(
                row,
                scene_df,
                excluded_categories=excluded_categories,
                language=language,
            )
            if neigh:
                descriptors.append(neigh)

        if descriptors:
            desc = "; ".join(descriptors)
            lines.append(f"{num}: {label:<15} ({desc}) -> {oid}")
        else:
            lines.append(f"{num}: {label:<15} -> {oid}")

    return "\n".join(lines)