"""
main.py
Servidor FastAPI para o sistema de QA espacial.

Endpoints:
    GET  /              → frontend HTML
    POST /upload        → recebe PLY, roda SpatialLM, retorna objetos
    POST /query         → recebe query NL, executa operador, retorna resposta
    GET  /scene/objects → lista objetos da cena atual (sessão)
    GET  /health        → status do servidor

Uso:
    pip install fastapi uvicorn python-multipart scipy numpy
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from geometric_engine import DetectedObject, execute
from spatiallm_runner import layout_to_detected_objects
from query_parser import parse_query_or_error
from spatiallm_runner import run_spatiallm, load_layout_file, SUPPORTED_CATEGORIES

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Spatial QA System",
    description="Consultas espaciais sobre cenas 3D via linguagem natural",
    version="1.0.0",
)

# Serve arquivos estáticos (frontend)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Estado da sessão (simples — single user para demo)
# ---------------------------------------------------------------------------

class SessionState:
    def __init__(self):
        self.ply_path:    Optional[str] = None
        self.layout_text: Optional[str] = None
        self.objects:     dict[str, DetectedObject] = {}
        self.objects_raw: list[dict] = []
        self.scene_name:  str = ""

STATE = SessionState()

# ---------------------------------------------------------------------------
# Modelos de request/response
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str

class HealthResponse(BaseModel):
    status: str
    spatiallm_available: bool
    objects_loaded: int
    scene: str

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve o frontend HTML."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Spatial QA System — API running. Frontend not found."})


@app.get("/health", response_model=HealthResponse)
async def health():
    from spatiallm_runner import INFERENCE_PY
    return HealthResponse(
        status="ok",
        spatiallm_available=INFERENCE_PY.exists(),
        objects_loaded=len(STATE.objects),
        scene=STATE.scene_name,
    )


@app.post("/upload")
async def upload_scene(
    file: UploadFile = File(...),
    layout_file: Optional[UploadFile] = File(None),
):
    """
    Recebe um arquivo PLY (cena 3D) e opcionalmente um layout .txt
    já gerado pelo SpatialLM.

    Se o layout for fornecido, pula a inferência (mais rápido, sem GPU).
    Se não for fornecido, roda o SpatialLM (requer GPU).

    Returns:
        JSON com lista de objetos detectados.
    """
    if not file.filename.endswith(".ply"):
        raise HTTPException(400, "Apenas arquivos .ply são aceitos")

    # Salva o PLY em arquivo temporário
    tmp_dir  = tempfile.mkdtemp(prefix="spatial_qa_")
    ply_path = os.path.join(tmp_dir, file.filename)

    try:
        with open(ply_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Usa layout pré-existente se fornecido
        if layout_file and layout_file.filename.endswith(".txt"):
            layout_path = os.path.join(tmp_dir, "layout.txt")
            with open(layout_path, "wb") as f:
                shutil.copyfileobj(layout_file.file, f)
            objects_raw, layout_text = load_layout_file(layout_path)
            method = "layout_preloaded"
        else:
            # Roda SpatialLM
            objects_raw, layout_text = run_spatiallm(
                ply_path=ply_path,
                categories=SUPPORTED_CATEGORIES,
                inference_dtype="bfloat16",
                seed=42,
            )
            method = "spatiallm_inference"

        # Atualiza estado da sessão
        STATE.ply_path    = ply_path
        STATE.layout_text = layout_text
        STATE.objects_raw = objects_raw
        STATE.objects     = layout_to_detected_objects(objects_raw)
        STATE.scene_name  = Path(file.filename).stem

        return JSONResponse({
            "ok":         True,
            "method":     method,
            "scene":      STATE.scene_name,
            "n_objects":  len(objects_raw),
            "objects":    objects_raw,
            "categories": list({o["category"] for o in objects_raw}),
        })

    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Erro interno: {e}")


@app.get("/scene/objects")
async def get_objects():
    """Retorna os objetos da cena carregada na sessão."""
    if not STATE.objects:
        raise HTTPException(404, "Nenhuma cena carregada. Faça upload de um PLY primeiro.")
    return JSONResponse({
        "scene":   STATE.scene_name,
        "objects": STATE.objects_raw,
    })


@app.post("/query")
async def query(req: QueryRequest):
    """
    Recebe uma query em linguagem natural (PT-BR ou EN),
    parseia o operador e executa o motor geométrico.

    Returns:
        JSON com resultado da operação espacial.
    """
    if not STATE.objects:
        raise HTTPException(
            404,
            "Nenhuma cena carregada. Faça upload de um arquivo PLY primeiro."
        )

    q = req.query.strip()
    if not q:
        raise HTTPException(400, "Query vazia")

    # Parse da query
    parse_result = parse_query_or_error(q)
    if not parse_result["ok"]:
        return JSONResponse({
            "ok":    False,
            "query": q,
            "error": parse_result["error"],
        })

    parsed = parse_result["parsed"]

    # Executa o operador geométrico
    try:
        result = execute(parsed, STATE.objects)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({
            "ok":    False,
            "query": q,
            "error": f"Erro na execução do operador: {e}",
        })

    if "error" in result:
        # Lista objetos disponíveis para ajudar o usuário
        available = [f"{o['obj_id']} ({o['category']})"
                     for o in STATE.objects_raw]
        return JSONResponse({
            "ok":       False,
            "query":    q,
            "parsed":   parsed,
            "error":    result["error"],
            "hint":     f"Objetos disponíveis: {', '.join(available)}",
        })

    return JSONResponse({
        "ok":     True,
        "query":  q,
        "parsed": parsed,
        "result": result,
    })


@app.get("/scene/layout")
async def get_layout():
    """Retorna o layout bruto gerado pelo SpatialLM."""
    if not STATE.layout_text:
        raise HTTPException(404, "Nenhum layout carregado")
    return JSONResponse({"layout": STATE.layout_text})


# ---------------------------------------------------------------------------
# Inicialização
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
