from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_points_npz(points_path: str | Path) -> np.ndarray:
    data = np.load(points_path)
    pts = data["points"].astype(np.float32)
    return pts


def centroid_from_points(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0)


def centroid_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    ca = centroid_from_points(points_a)
    cb = centroid_from_points(points_b)
    return float(np.linalg.norm(ca - cb))


def surface_distance(points_a: np.ndarray, points_b: np.ndarray, chunk_size: int = 2048) -> float:
    """
    Distância mínima entre superfícies por busca exaustiva em blocos.
    Não é a implementação mais rápida do mundo, mas é determinística
    e suficiente para o piloto.
    """
    if len(points_a) == 0 or len(points_b) == 0:
        raise ValueError("Empty point cloud received.")

    min_dist = np.inf

    # iterar sobre a menor nuvem como fonte, para reduzir custo
    if len(points_a) > len(points_b):
        points_a, points_b = points_b, points_a

    for i in range(0, len(points_a), chunk_size):
        chunk = points_a[i:i + chunk_size]  # [m, 3]
        diff = chunk[:, None, :] - points_b[None, :, :]  # [m, n, 3]
        d2 = np.sum(diff * diff, axis=2)  # [m, n]
        local_min = np.min(d2)
        if local_min < min_dist:
            min_dist = local_min

    return float(np.sqrt(min_dist))


def project_xy(point_xyz: np.ndarray) -> np.ndarray:
    return np.asarray(point_xyz[:2], dtype=np.float32)


def point_to_segment_distance_xy(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Retorna:
    - distância perpendicular do ponto x ao segmento ab no plano XY
    - parâmetro t da projeção
    """
    ab = b - a
    ab_norm_sq = float(np.dot(ab, ab))

    if ab_norm_sq <= 1e-12:
        return float(np.linalg.norm(x - a)), 0.0

    t = float(np.dot(x - a, ab) / ab_norm_sq)
    t_clamped = max(0.0, min(1.0, t))
    proj = a + t_clamped * ab
    d = float(np.linalg.norm(x - proj))
    return d, t


def is_between_xy(
    centroid_x: np.ndarray,
    centroid_a: np.ndarray,
    centroid_b: np.ndarray,
    tau_between: float = 0.35,
) -> bool:
    x = project_xy(centroid_x)
    a = project_xy(centroid_a)
    b = project_xy(centroid_b)

    d_perp, t = point_to_segment_distance_xy(x, a, b)
    return (0.0 <= t <= 1.0) and (d_perp <= tau_between)


def point_to_line_distance_xy(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    ab_norm = float(np.linalg.norm(ab))

    if ab_norm <= 1e-12:
        return float(np.linalg.norm(x - a))

    # área do paralelogramo / comprimento da base
    ax = x - a
    cross = abs(ab[0] * ax[1] - ab[1] * ax[0])
    return float(cross / ab_norm)


def is_aligned_xy(
    centroid_a: np.ndarray,
    centroid_b: np.ndarray,
    centroid_c: np.ndarray,
    tau_align: float = 0.25,
) -> bool:
    a = project_xy(centroid_a)
    b = project_xy(centroid_b)
    c = project_xy(centroid_c)

    d = point_to_line_distance_xy(c, a, b)
    return d <= tau_align


def nearest_object_by_surface(
    ref_points: np.ndarray,
    candidate_dict: Dict[str, np.ndarray],
) -> Tuple[str, float]:
    """
    candidate_dict: {object_id: points}
    """
    best_id = None
    best_dist = np.inf

    for object_id, pts in candidate_dict.items():
        d = surface_distance(ref_points, pts)
        if d < best_dist:
            best_dist = d
            best_id = object_id

    if best_id is None:
        raise RuntimeError("No candidates provided.")

    return best_id, float(best_dist)