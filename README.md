# Raciocínio Espacial sobre Cenas 3D via Representação Geométrica Explícita

**PPGM/UFBA 2026**  
Autor: Jefferson Lopes B SIlva · Orientador: Prof. Dr. Luciano Rebouças de Oliveira

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ScanNet](https://img.shields.io/badge/Dataset-ScanNet-orange.svg)](http://www.scan-net.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Visão Geral

Este repositório contém o código, benchmark e resultados da dissertação que investiga o papel da **representação geométrica explícita** na execução de consultas espaciais sobre cenas 3D estruturadas.

A formulação central separa duas camadas:

```
Query em linguagem natural
        ↓
  [ Grounding ]       ← identifica as entidades da cena
        ↓
  [ Motor Geométrico ] ← executa o operador espacial
        ↓
  Resposta métrica
```

O erro total é decomposto formalmente como:

$$E_{\text{total}} = E_{\text{geométrico}} + \Delta E_{\text{grounding}}$$

---

## Operadores Espaciais

| Operador | Descrição | Entrada | Saída |
|---|---|---|---|
| `distance(A, B)` | Distância mínima entre superfícies | 2 objetos | metros |
| `nearest(ref, cat)` | Objeto mais próximo de uma referência | 1 ref + categoria | objeto |
| `between(X, A, B)` | X está entre A e B no plano XY? | 3 objetos | Sim/Não |
| `aligned(A, B, C)` | A, B e C estão alinhados? | 3 objetos | Sim/Não |

---

## Resultados Principais

### Benchmark Oficial — Test Official Stage 1 (20 cenas, 84 queries métricas + 138 binárias)

| Operador | Representação | Métrica | Resultado |
|---|---|---|---|
| `distance` | Superfície | MAE (m) | **0,000** |
| `distance` | Centróide | MAE (m) | 0,944 |
| `nearest` | Superfície | Top-1 | **1,000** |
| `nearest` | Centróide | Top-1 | 1,000 |
| `between` | Geométrico | F1 | **1,000** |
| `aligned` | Geométrico | F1 | **1,000** |

### Experimento E2E — GPT-4.1 como módulo de grounding

| Operador | Grounding correto | E_geométrico | E_total |
|---|---|---|---|
| `distance` | 33,3% (15/45) | 0,000 m | 0,786 m |
| `nearest` | 38,5% (15/39) | 1,000 | 0,692 |

### Hierarquia empírica — operador `nearest`

```
VLM puro (35,9%) → E2E GPT-4.1 (69,2%) → Geometria perfeita (100,0%)
```

### Experimento E2E — SpatialLM como módulo de grounding

| Operador | N avaliável | Grounding correto |
|---|---|---|
| `distance` | 21/45 | 14,3% |
| `nearest` | 12/39 | 8,3% |

> ScanNet avaliado em subconjunto sem `monitor`/`door` (ausentes no vocabulário do SpatialLM 1.1).

---

## Estrutura do Repositório

```
metric-spatial-vlm/
│
├── benchmark/                          # Queries e ground truth oficiais
│   ├── scenes_dev_official.csv         # 20 cenas — desenvolvimento
│   ├── scenes_test_official_stage1.csv # 20 cenas — teste
│   ├── objects_manifest_*.csv          # Manifestos de objetos válidos
│   ├── ground_truth_*.csv              # Ground truth dos operadores
│   ├── queries_*_final.csv             # Queries oficiais revisadas
│   └── demo_alias_maps/                # Aliases para demonstração
│
├── configs/
│   ├── benchmark_config.yaml           # Limiares oficiais (τ_between, τ_align)
│   └── label_map.yaml                  # Mapeamento de categorias
│
├── src/
│   ├── geometry/                       # Operadores geométricos
│   ├── dataset/                        # Carregamento de dados
│   ├── evaluation/                     # Métricas de avaliação
│   └── queries/                        # Geração de queries
│
├── scripts/
│   ├── benchmark/                      # Construção do benchmark (scripts 00–71)
│   ├── experiments/                    # Experimentos (scripts 81–91)
│   └── demo/                           # Demonstração visual (scripts 72–80, 92–106)
│
├── results/
│   └── benchmark_v1/                   # Resultados oficiais (CSVs)
│
├── figures/                            # Figuras geradas
├── notebooks/                          # Análises interativas
└── requirements.txt
```

---

## Instalação

```bash
git clone https://github.com/grifo114/metric-spatial-vlm
cd metric-spatial-vlm
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

**Dependências principais:** `numpy`, `scipy`, `pandas`, `open3d`, `matplotlib`

---

## Dataset

O benchmark usa cenas do [ScanNet](http://www.scan-net.org/). Os dados geométricos (PLYs, nuvens de pontos) não são distribuídos neste repositório por questões de licença — solicite acesso em [scan-net.org](http://www.scan-net.org/).

Os arquivos de benchmark em `benchmark/` (queries, ground truth, manifestos) **não dependem dos dados brutos** para reprodução das métricas — os resultados em `results/benchmark_v1/` são auto-contidos.

---

## Reprodução dos Resultados

```bash
# Benchmark métrico — conjunto de teste oficial
python scripts/benchmark/61_run_benchmark_distance_nearest_test_official_stage1.py
python scripts/benchmark/62_run_baseline_centroid_distance_nearest_test_official_stage1.py

# Benchmark relacional — conjunto de teste oficial
python scripts/benchmark/70_run_benchmark_relational_binary_test_official_stage1.py

# Baseline VLM (requer chave OpenAI)
python scripts/experiments/81_vlm_baseline_distance_nearest.py

# Experimento E2E GPT-4.1 (requer chave OpenAI)
python scripts/experiments/84_e2e_grounding_test_official.py

# Bootstrap de incerteza
python scripts/experiments/86_bootstrap_uncertainty.py
```

---

## Limiares Oficiais

Calibrados no conjunto de desenvolvimento (`dev_official`) por análise de sensibilidade:

| Parâmetro | Valor | Operador |
|---|---|---|
| `τ_between` | **0,30** | `between` |
| `τ_aligned` | **0,25** | `aligned` |

Definidos em `configs/benchmark_config.yaml`.

---

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

---

## Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.