#!/bin/bash
# start.sh — instala e inicia o Spatial QA System no RunPod
# Uso: bash start.sh

set -e

echo "========================================"
echo " Spatial QA System — Setup & Start"
echo "========================================"

# 1. Instala dependências Python
echo "[1/3] Instalando dependências..."
pip install fastapi uvicorn[standard] python-multipart numpy scipy -q

# 2. Configura variáveis de ambiente
export SPATIALLM_DIR="${SPATIALLM_DIR:-/SpatialLM}"
export SPATIALLM_MODEL="${SPATIALLM_MODEL:-manycore-research/SpatialLM1.1-Llama-1B}"
export PORT="${PORT:-8000}"

echo "[2/3] Configuração:"
echo "  SPATIALLM_DIR   = $SPATIALLM_DIR"
echo "  SPATIALLM_MODEL = $SPATIALLM_MODEL"
echo "  PORT            = $PORT"

# 3. Verifica SpatialLM
if [ -f "$SPATIALLM_DIR/inference.py" ]; then
    echo "  SpatialLM       = ✓ encontrado"
else
    echo "  SpatialLM       = ✗ NÃO encontrado em $SPATIALLM_DIR"
    echo "  O servidor vai rodar mas a inferência não estará disponível."
    echo "  Forneça o layout .txt pré-gerado ao fazer upload da cena."
fi

# 4. Inicia o servidor
echo ""
echo "[3/3] Iniciando servidor na porta $PORT..."
echo "  Acesse: http://localhost:$PORT"
echo "  No RunPod: use a aba 'Connect' → 'HTTP Service' → porta $PORT"
echo ""

cd "$(dirname "$0")"
python -m uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
