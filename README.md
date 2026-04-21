# Raciocínio Espacial sobre Cenas 3D via Representação Geométrica Explícita

**Dissertação de Mestrado — PPGM/UFBA 2026**
Autor: Jefferson Lopes

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ScanNet](https://img.shields.io/badge/Dataset-ScanNet-orange.svg)](http://www.scan-net.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Visão Geral

Investigação do papel da representação geométrica explícita na execução de consultas espaciais sobre cenas 3D estruturadas. A formulação central separa grounding e geometria como camadas independentes, formalizando o erro total como:

**E_total = E_geométrico + Delta_E_grounding**

## Operadores Espaciais

| Operador | Descrição | Saída |
|---|---|---|
| distance(A, B) | Distância mínima entre superfícies | metros |
| nearest(ref, cat) | Objeto mais próximo de uma referência | objeto |
| between(X, A, B) | X está entre A e B no plano XY? | Sim/Não |
| aligned(A, B, C) | A, B e C estão alinhados? | Sim/Não |

## Resultados — Test Official Stage 1

| Operador | Superfície | Centróide |
|---|---|---|
| distance (MAE) | **0,000 m** | 0,944 m |
| nearest (Top-1) | **1,000** | 1,000 |
| between (F1) | **1,000** | — |
| aligned (F1) | **1,000** | — |

## Estrutura
metric-spatial-vlm/
├── benchmark/          # Queries e ground truth oficiais (dev + test)
├── configs/            # Limiares oficiais (tau_between=0.30, tau_align=0.25)
├── src/                # Motor geométrico
├── scripts/
│   ├── benchmark/      # Construção do benchmark (scripts 00-71)
│   ├── experiments/    # Experimentos (scripts 81-91)
│   └── demo/           # Demonstração visual (scripts 72-80, 92-106)
└── results/benchmark_v1/  # Resultados oficiais
## Instalação

```bash
git clone https://github.com/grifo114/metric-spatial-vlm
cd metric-spatial-vlm
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

O benchmark usa cenas do [ScanNet](http://www.scan-net.org/). Os dados geométricos não são distribuídos aqui por licença. Os resultados em results/benchmark_v1/ são auto-contidos.

## Citação

```bibtex
@mastersthesis{lopes2026spatial,
  author  = {Lopes, Jefferson},
  title   = {Raciocínio Espacial sobre Cenas 3D via Representação Geométrica Explícita},
  school  = {Universidade Federal da Bahia},
  year    = {2026},
  program = {Programa de Pós-Graduação em Mecatrônica}
}
```

## Licença

MIT License
