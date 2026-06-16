"""
spatiallm_runner.py
Wrapper para o SpatialLM: roda a inferência sobre um PLY e
converte o layout de saída em DetectedObjects.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from geometric_engine import DetectedObject


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

# Caminho do repositório SpatialLM (ajuste se necessário)
SPATIALLM_DIR = Path(os.environ.get("SPATIALLM_DIR", "/SpatialLM"))
INFERENCE_PY  = SPATIALLM_DIR / "inference.py"
MODEL_PATH    = os.environ.get(
    "SPATIALLM_MODEL",
    "manycore-research/SpatialLM1.1-Llama-1B"
)

# Categorias suportadas pelo SpatialLM 1.1
SUPPORTED_CATEGORIES = [
    "chair", "dining_chair", "bar_chair", "stool",
    "sofa", "bed",
    "dining_table", "side_table", "coffee_table", "dressing_table",
    "desk",
    "wardrobe", "nightstand", "tv_cabinet",
]

# Mapeamento de categoria SpatialLM → label normalizado do sistema
CATEGORY_NORM = {
    "chair": "chair", "dining_chair": "chair",
    "bar_chair": "chair", "stool": "chair",
    "sofa": "sofa",
    "bed": "bed",
    "dining_table": "table", "side_table": "table",
    "coffee_table": "table", "dressing_table": "table",
    "desk": "desk",
    "wardrobe": "cabinet", "nightstand": "cabinet",
    "tv_cabinet": "cabinet",
}

# Padrão de parsing do layout
BBOX_PATTERN = re.compile(
    r"bbox_(\d+)=Bbox\((\w+),\s*([\d\.\-]+),\s*([\d\.\-]+),\s*([\d\.\-]+)"
    r"(?:,\s*([\d\.\-]+),\s*([\d\.\-]+),\s*([\d\.\-]+),\s*([\d\.\-]+))?\)"
)


# ---------------------------------------------------------------------------
# Parser de layout
# ---------------------------------------------------------------------------

def parse_layout(layout_text: str) -> list[dict]:
    """
    Parseia o texto do layout gerado pelo SpatialLM.
    Formato: bbox_N=Bbox(categoria, cx, cy, cz, angle, dx, dy, dz)
    Retorna lista de dicts com os campos de cada objeto.
    """
    objects = []
    # Conta instâncias por categoria para gerar IDs únicos
    cat_counts: dict[str, int] = {}

    for m in BBOX_PATTERN.finditer(layout_text):
        cat_raw = m.group(2).lower()
        cx, cy, cz = float(m.group(3)), float(m.group(4)), float(m.group(5))

        # Dimensões (opcionais no formato 1.0)
        angle = float(m.group(6)) if m.group(6) else 0.0
        dx    = float(m.group(7)) if m.group(7) else 0.6
        dy    = float(m.group(8)) if m.group(8) else 0.6
        dz    = float(m.group(9)) if m.group(9) else 0.8

        # Normaliza categoria
        cat_norm = CATEGORY_NORM.get(cat_raw, cat_raw)
        idx      = cat_counts.get(cat_norm, 0)
        cat_counts[cat_norm] = idx + 1
        obj_id = f"{cat_norm}_{idx}"

        objects.append({
            "obj_id":   obj_id,
            "category": cat_norm,
            "category_raw": cat_raw,
            "cx": cx, "cy": cy, "cz": cz,
            "dx": dx, "dy": dy, "dz": dz,
            "angle": angle,
        })

    return objects


def layout_to_detected_objects(
    objects: list[dict],
) -> dict[str, DetectedObject]:
    """Converte lista de dicts em dicionário de DetectedObjects."""
    return {
        o["obj_id"]: DetectedObject(
            obj_id   = o["obj_id"],
            category = o["category"],
            cx=o["cx"], cy=o["cy"], cz=o["cz"],
            dx=o["dx"], dy=o["dy"], dz=o["dz"],
            angle    = o["angle"],
        )
        for o in objects
    }


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------

def run_spatiallm(
    ply_path: str,
    categories: Optional[list[str]] = None,
    inference_dtype: str = "bfloat16",
    seed: int = 42,
    timeout: int = 600,
) -> tuple[list[dict], str]:
    """
    Roda o SpatialLM sobre um PLY e retorna a lista de objetos detectados.

    Args:
        ply_path:        caminho para o arquivo PLY
        categories:      lista de categorias a detectar (None = todas suportadas)
        inference_dtype: dtype para inferência (bfloat16 recomendado)
        seed:            semente aleatória
        timeout:         timeout em segundos

    Returns:
        (lista de objetos, texto do layout bruto)

    Raises:
        RuntimeError: se o SpatialLM falhar
        FileNotFoundError: se o PLY ou o script não existir
    """
    if not Path(ply_path).exists():
        raise FileNotFoundError(f"PLY não encontrado: {ply_path}")
    if not INFERENCE_PY.exists():
        raise FileNotFoundError(
            f"SpatialLM não encontrado em: {SPATIALLM_DIR}\n"
            f"Defina a variável de ambiente SPATIALLM_DIR."
        )

    cats = categories or SUPPORTED_CATEGORIES

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        output_path = tf.name

    try:
        cmd = [
            "python", str(INFERENCE_PY),
            "--point_cloud",     ply_path,
            "--output",          output_path,
            "--model_path",      MODEL_PATH,
            "--detect_type",     "object",
            "--category",        *cats,
            "--inference_dtype", inference_dtype,
            "--seed",            str(seed),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            # Filtra warnings irrelevantes do stderr
            errs = [
                l for l in result.stderr.split("\n")
                if l and not l.startswith(("W0", "E0", "2026", "WARNING", "I tf"))
            ]
            raise RuntimeError(
                f"SpatialLM falhou (código {result.returncode}):\n"
                + "\n".join(errs[-5:])
            )

        layout_text = Path(output_path).read_text() if Path(output_path).exists() else ""
        objects     = parse_layout(layout_text)
        return objects, layout_text

    finally:
        try:
            os.unlink(output_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Modo fallback: parseia layout já existente (sem rodar inferência)
# ---------------------------------------------------------------------------

def load_layout_file(
    layout_path: str,
) -> tuple[list[dict], str]:
    """
    Carrega um layout já existente (gerado anteriormente pelo SpatialLM).
    Útil para desenvolvimento e testes sem GPU.
    """
    if not Path(layout_path).exists():
        raise FileNotFoundError(f"Layout não encontrado: {layout_path}")
    text    = Path(layout_path).read_text()
    objects = parse_layout(text)
    return objects, text


# ---------------------------------------------------------------------------
# Teste rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Testa o parser com um layout de exemplo
    sample_layout = """
bbox_0=Bbox(chair,7.197,2.634,0.724,-3.1416,0.765,0.796,0.890)
bbox_1=Bbox(chair,5.397,2.684,0.724,-3.1416,0.765,0.796,0.890)
bbox_2=Bbox(desk,5.347,5.709,1.174,-1.5708,1.125,0.484,1.875)
bbox_3=Bbox(wardrobe,5.547,6.684,1.249,-3.1416,0.640,0.562,1.984)
bbox_4=Bbox(dining_table,3.222,7.234,0.674,-3.1416,0.859,0.859,0.859)
"""
    objects, _ = load_layout_file.__wrapped__(sample_layout) \
        if hasattr(load_layout_file, "__wrapped__") \
        else (parse_layout(sample_layout), sample_layout)
    print(f"Objetos detectados: {len(objects)}")
    for o in objects:
        print(f"  {o['obj_id']:15s} ({o['category']:12s}) "
              f"@ ({o['cx']:.2f}, {o['cy']:.2f}, {o['cz']:.2f})")
