# Raciocínio Espacial sobre Cenas 3D via Representação Geométrica Explícita

Dissertação de Mestrado — PPGM/UFBA 2026  
Autor: Jefferson Lopes  
Orientador: [nome do orientador]

## Resumo

Este repositório contém o código, benchmark e resultados da dissertação que investiga o papel da representação geométrica explícita na execução de consultas espaciais sobre cenas 3D estruturadas. O trabalho propõe uma arquitetura modular que separa a identificação das entidades da cena (grounding) da execução de operadores geométricos explícitos.

## Operadores Espaciais

| Operador | Descrição | Métrica |
|---|---|---|
| `distance(A, B)` | Distância entre superfícies de dois objetos | MAE (metros) |
| `nearest(ref, cat)` | Objeto mais próximo de uma referência | Top-1 accuracy |
| `between(X, A, B)` | X está entre A e B no plano XY? | F1 binário |
| `aligned(A, B, C)` | A, B e C estão alinhados no plano XY? | F1 binário |

## Resultados Principais (Test Official Stage1)

| Operador | Superfície | Centróide |
|---|---|---|
| distance (MAE) | **0.000 m** | 0.944 m |
| nearest (Top-1) | **1.000** | 1.000 |
| between (F1) | **1.000** | — |
| aligned (F1) | **1.000** | — |

## Estrutura do Repositório
metric-spatial-vlm/
├── benchmark/          # Queries e ground truth oficiais (dev + test)
├── configs/            # Configurações do benchmark
│   ├── benchmark_config.yaml
│   └── label_map.yaml
├── src/                # Motor geométrico e utilitários
│   ├── geometry/       # Operadores espaciais
│   ├── dataset/        # Carregamento de dados
│   ├── evaluation/     # Métricas
│   └── queries/        # Geração de queries
├── scripts/
│   ├── benchmark/      # Construção do benchmark (scripts 00-71)
│   ├── experiments/    # Experimentos (scripts 81-91)
│   └── demo/           # Demonstração visual (scripts 72-80, 92-106)
├── results/benchmark_v1/  # Resultados oficiais
├── figures/            # Figuras geradas
└── notebooks/          # Análises e demos interativos
## Instalação

```bash
git clone https://github.com/grifo114/metric-spatial-vlm
cd metric-spatial-vlm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reprodução dos Resultados

```bash
# 1. Benchmark métrico (distance + nearest)
python scripts/benchmark/61_run_benchmark_distance_nearest_test_official_stage1.py

# 2. Benchmark relacional (between + aligned)
python scripts/benchmark/70_run_benchmark_relational_binary_test_official_stage1.py

# 3. Baseline VLM
python scripts/experiments/81_vlm_baseline_distance_nearest.py

# 4. Experimento E2E GPT-4.1
python scripts/experiments/84_e2e_grounding_test_official.py
```

## Dataset

O benchmark usa cenas do [ScanNet](http://www.scan-net.org/). Os dados geométricos não são distribuídos neste repositório por questões de licença. Siga as instruções em [ScanNet](http://www.scan-net.org/) para obter acesso.

Os arquivos de benchmark (queries, ground truth, manifestos) estão em `benchmark/` e não dependem dos dados brutos do ScanNet para reprodução das métricas.

## Limiares Oficiais

Calibrados no conjunto de desenvolvimento (dev official):

- `τ_between = 0.30`
- `τ_aligned = 0.25`

Definidos em `configs/benchmark_config.yaml`.

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

MIT License — veja `LICENSE` para detalhes.
