"""
geometric_engine.py
Motor geométrico explícito: distance, nearest, between, aligned.
Opera sobre objetos já identificados (centróide ou superfície).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Representação de objeto detectado
# ---------------------------------------------------------------------------

@dataclass
class DetectedObject:
    obj_id:   str           # ex: "chair_0"
    category: str           # ex: "chair"
    cx: float               # centróide x (metros)
    cy: float               # centróide y (metros)
    cz: float               # centróide z (metros)
    dx: float = 0.6         # dimensão x (metros)
    dy: float = 0.6         # dimensão y (metros)
    dz: float = 0.8         # dimensão z (metros)
    angle: float = 0.0      # ângulo de rotação (rad)
    points: Optional[np.ndarray] = None  # nuvem de pontos (N,3), se disponível

    @property
    def centroid(self) -> np.ndarray:
        return np.array([self.cx, self.cy, self.cz])

    @property
    def centroid_xy(self) -> np.ndarray:
        return np.array([self.cx, self.cy])

    def to_dict(self) -> dict:
        return {
            "obj_id":   self.obj_id,
            "category": self.category,
            "cx": self.cx, "cy": self.cy, "cz": self.cz,
            "dx": self.dx, "dy": self.dy, "dz": self.dz,
            "angle":    self.angle,
        }


# ---------------------------------------------------------------------------
# Utilitários de distância
# ---------------------------------------------------------------------------

def distance_centroid(a: DetectedObject, b: DetectedObject) -> float:
    """Distância euclidiana entre centróides (metros)."""
    return float(np.linalg.norm(a.centroid - b.centroid))


def distance_surface(a: DetectedObject, b: DetectedObject) -> float:
    """
    Distância mínima entre superfícies.
    Se os pontos não estiverem disponíveis, usa distância entre centróides
    menos metade das dimensões (aproximação conservadora).
    """
    if a.points is not None and b.points is not None and \
       len(a.points) > 0 and len(b.points) > 0:
        tree = cKDTree(b.points)
        d, _  = tree.query(a.points, k=1, workers=-1)
        return float(d.min())

    # Fallback: distância centróide menos metade das extensões
    c_dist = float(np.linalg.norm(a.centroid - b.centroid))
    margin = (min(a.dx, a.dy) + min(b.dx, b.dy)) / 4.0
    return max(0.0, c_dist - margin)


def best_distance(a: DetectedObject, b: DetectedObject) -> float:
    """Usa superfície se disponível, caso contrário centróide."""
    if a.points is not None and b.points is not None:
        return distance_surface(a, b)
    return distance_centroid(a, b)


# ---------------------------------------------------------------------------
# Operadores espaciais
# ---------------------------------------------------------------------------

def op_distance(obj_a: DetectedObject, obj_b: DetectedObject) -> dict:
    """
    distance(A, B): distância entre dois objetos.
    Retorna distância em metros.
    """
    d_surf    = distance_surface(obj_a, obj_b)
    d_centroid = distance_centroid(obj_a, obj_b)
    return {
        "operator":          "distance",
        "object_a":          obj_a.obj_id,
        "object_b":          obj_b.obj_id,
        "distance_surface_m": round(d_surf, 4),
        "distance_centroid_m": round(d_centroid, 4),
        "answer":            f"{d_surf:.2f} m",
        "answer_pt":         f"{d_surf:.2f} m",
    }


def op_nearest(
    ref: DetectedObject,
    candidates: list[DetectedObject],
) -> dict:
    """
    nearest(ref, candidates): objeto mais próximo do ref.
    Retorna o candidato mais próximo e a distância.
    """
    if not candidates:
        return {
            "operator": "nearest",
            "ref":      ref.obj_id,
            "answer":   None,
            "error":    "Nenhum candidato disponível / No candidates available",
        }

    dists = [(c, best_distance(ref, c)) for c in candidates]
    dists.sort(key=lambda x: x[1])
    winner, d_win = dists[0]

    ranking = [
        {"obj_id": c.obj_id, "category": c.category,
         "distance_m": round(d, 4)}
        for c, d in dists
    ]
    return {
        "operator":   "nearest",
        "ref":        ref.obj_id,
        "winner":     winner.obj_id,
        "distance_m": round(d_win, 4),
        "ranking":    ranking,
        "answer":     winner.obj_id,
        "answer_pt":  f"{winner.obj_id} ({d_win:.2f} m)",
    }


def op_between(
    obj_x: DetectedObject,
    obj_a: DetectedObject,
    obj_b: DetectedObject,
    tau: float = 0.30,
) -> dict:
    """
    between(X, A, B): X está entre A e B no plano XY?
    Usa a definição calibrada do benchmark (τ = 0.30).
    """
    cx = obj_x.centroid_xy
    ca = obj_a.centroid_xy
    cb = obj_b.centroid_xy

    seg     = cb - ca
    seg_len = float(np.linalg.norm(seg))

    if seg_len < 1e-6:
        result = False
        t_val  = 0.0
        d_val  = float(np.linalg.norm(cx - ca))
    else:
        t_val = float(np.dot(cx - ca, seg) / seg_len**2)
        proj  = ca + t_val * seg
        d_val = float(np.linalg.norm(cx - proj))
        result = (0 <= t_val <= 1) and (d_val <= tau * seg_len)

    return {
        "operator":     "between",
        "object_x":     obj_x.obj_id,
        "object_a":     obj_a.obj_id,
        "object_b":     obj_b.obj_id,
        "tau":          tau,
        "t_projection": round(t_val, 4),
        "lateral_dist": round(d_val, 4),
        "result":       result,
        "answer":       "Sim" if result else "Não",
        "answer_en":    "Yes" if result else "No",
    }


def op_aligned(
    obj_a: DetectedObject,
    obj_b: DetectedObject,
    obj_c: DetectedObject,
    tau: float = 0.25,
) -> dict:
    """
    aligned(A, B, C): A, B e C estão aproximadamente alinhados no plano XY?
    B é o ponto intermediário. Usa a definição calibrada do benchmark (τ = 0.25).
    """
    ca = obj_a.centroid_xy
    cb = obj_b.centroid_xy
    cc = obj_c.centroid_xy

    seg     = cc - ca
    seg_len = float(np.linalg.norm(seg))

    if seg_len < 1e-6:
        result = False
        d_val  = float(np.linalg.norm(cb - ca))
        t_val  = 0.0
    else:
        t_val  = float(np.dot(cb - ca, seg) / seg_len**2)
        proj   = ca + t_val * seg
        d_val  = float(np.linalg.norm(cb - proj))
        result = d_val <= tau * seg_len

    return {
        "operator":     "aligned",
        "object_a":     obj_a.obj_id,
        "object_b":     obj_b.obj_id,
        "object_c":     obj_c.obj_id,
        "tau":          tau,
        "t_projection": round(t_val, 4),
        "lateral_dist": round(d_val, 4),
        "result":       result,
        "answer":       "Sim" if result else "Não",
        "answer_en":    "Yes" if result else "No",
    }


# ---------------------------------------------------------------------------
# Dispatcher principal
# ---------------------------------------------------------------------------

def resolve_object(
    name: str,
    objects: dict[str, DetectedObject],
) -> Optional[DetectedObject]:
    """
    Resolve nome da query para objeto detectado.
    Tenta: match exato → match por categoria → match por índice.
    """
    name_lower = name.lower().strip()

    # Match exato por obj_id
    if name in objects:
        return objects[name]

    # Match case-insensitive
    for k, v in objects.items():
        if k.lower() == name_lower:
            return v

    # Match por categoria (retorna o primeiro)
    from query_parser import normalize_category
    cat = normalize_category(name_lower)
    if cat:
        for v in objects.values():
            if v.category == cat:
                return v

    # Match parcial (ex: "chair" encontra "chair_0")
    for k, v in objects.items():
        if name_lower in k.lower() or k.lower() in name_lower:
            return v

    return None


def execute(
    parsed: dict,
    objects: dict[str, DetectedObject],
) -> dict:
    """
    Executa o operador espacial sobre os objetos detectados.

    Args:
        parsed:  resultado do query_parser (parsed.to_dict())
        objects: dicionário {obj_id: DetectedObject}

    Returns:
        dict com resultado da operação
    """
    op       = parsed["operator"]
    entities = parsed["entities"]

    if op == "distance":
        if len(entities) < 2:
            return {"error": "distance requer dois objetos"}
        a = resolve_object(entities[0], objects)
        b = resolve_object(entities[1], objects)
        if a is None:
            return {"error": f"Objeto não encontrado: '{entities[0]}'"}
        if b is None:
            return {"error": f"Objeto não encontrado: '{entities[1]}'"}
        return op_distance(a, b)

    elif op == "nearest":
        if len(entities) < 1:
            return {"error": "nearest requer objeto de referência"}
        ref = resolve_object(entities[0], objects)
        if ref is None:
            return {"error": f"Objeto de referência não encontrado: '{entities[0]}'"}

        # Filtra candidatos pela categoria alvo
        cat = parsed.get("category")
        if cat:
            from query_parser import normalize_category
            cat_norm = normalize_category(cat) or cat
            cands = [v for k, v in objects.items()
                     if v.category == cat_norm and v.obj_id != ref.obj_id]
        else:
            cands = [v for v in objects.values() if v.obj_id != ref.obj_id]

        if not cands:
            return {"error": f"Nenhum candidato da categoria '{cat}' encontrado"}
        return op_nearest(ref, cands)

    elif op == "between":
        if len(entities) < 3:
            return {"error": "between requer três objetos"}
        x = resolve_object(entities[0], objects)
        a = resolve_object(entities[1], objects)
        b = resolve_object(entities[2], objects)
        for name, obj in [(entities[0], x), (entities[1], a), (entities[2], b)]:
            if obj is None:
                return {"error": f"Objeto não encontrado: '{name}'"}
        return op_between(x, a, b)

    elif op == "aligned":
        if len(entities) < 3:
            return {"error": "aligned requer três objetos"}
        a = resolve_object(entities[0], objects)
        b = resolve_object(entities[1], objects)
        c = resolve_object(entities[2], objects)
        for name, obj in [(entities[0], a), (entities[1], b), (entities[2], c)]:
            if obj is None:
                return {"error": f"Objeto não encontrado: '{name}'"}
        return op_aligned(a, b, c)

    return {"error": f"Operador desconhecido: '{op}'"}
