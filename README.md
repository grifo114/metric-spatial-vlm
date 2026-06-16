# Raciocínio Espacial sobre Cenas 3D via Representação Geométrica Explícita

Código, benchmark e resultados do trabalho de qualificação de mestrado
(PPGM/UFBA) que investiga o papel da representação geométrica explícita na
execução de consultas espaciais sobre cenas 3D estruturadas. O trabalho propõe
uma arquitetura modular que separa a identificação das entidades da cena
(*grounding*) da execução de operadores geométricos explícitos, formalizada pela
decomposição do erro total:

```
E_total = E_grounding + E_geométrico
```

> **Branches deste repositório**
> - **`main`** — estado correspondente ao trabalho de qualificação (benchmark
>   completo, operadores, experimentos VLM, SCI e sistema de QA). É a branch de
>   referência citada na dissertação.
> - **`article-release-clean`** — recorte isolado do artigo SIBGRAPI, restrito à
>   Spatial Context Injection (SCI). Mantida separada por escopo.

## Operadores Espaciais

| Operador | Descrição | Métrica |
|---|---|---|
| `distance(A, B)` | Distância entre superfícies de dois objetos | MAE (metros) |
| `nearest(ref, cat)` | Objeto mais próximo de uma referência | Top-1 |
| `between(X, A, B)` | X está entre A e B no plano XY? | F1 binário |
| `aligned(A, B, C)` | A, B e C estão alinhados no plano XY? | F1 binário |

## Resultados Principais (teste oficial — stage 1)

| Operador | Superfície | Centróide |
|---|---|---|
| distance (MAE) | **0.000 m** | 0.944 m |
| nearest (Top-1) | **1.000** | 1.000 |
| between (F1) | **1.000** | n/a |
| aligned (F1) | **1.000** | n/a |

`n/a` indica que a comparação centróide/superfície não se aplica aos operadores
relacionais. O MAE de superfície de 0.000 m é **definicional**: a referência é
derivada da mesma representação de superfície usada pelo operador.

A injeção de contexto espacial (**SCI**) foi avaliada sobre `distance` em uma
matriz de seis modelos de visão e linguagem (GPT-4.1, Claude Sonnet, Gemini 2.5
Flash, Qwen3-VL 8B/32B/235B). O ganho de *grounding* é dependente da capacidade
do modelo, expressivo nos modelos maiores e ausente na menor escala avaliada.

## Estrutura do Repositório

```
metric-spatial-vlm/
├── benchmark/             # Queries e ground truth oficiais (dev + test)
├── configs/               # Configurações do benchmark e label maps
├── src/                   # Motor geométrico e utilitários
│   ├── geometry/          # Operadores espaciais
│   ├── dataset/           # Carregamento de dados
│   ├── evaluation/        # Métricas
│   └── queries/           # Geração de queries
├── scripts/
│   ├── benchmark/         # Construção do benchmark
│   ├── experiments/       # Baseline VLM, E2E, SCI
│   └── (análises)         # Estratificação, p-valores, sensibilidade
├── results/               # Resultados oficiais (CSV)
├── figures/               # Figuras geradas
├── spatial_qa_system/     # Protótipo de QA espacial (FastAPI + visualização)
└── notebooks/             # Análises interativas
```

## Mapeamento Dissertação → Código

> Preencher os números reais das tabelas/figuras da versão final.

| Resultado na dissertação | Script |
|---|---|
| Tab. — distance/nearest (teste oficial) | `scripts/benchmark/61_run_benchmark_distance_nearest_test_official_stage1.py` |
| Tab. — between/aligned (teste oficial) | `scripts/benchmark/70_run_benchmark_relational_binary_test_official_stage1.py` |
| Fig. — sensibilidade de limiares (dev) | `results/dev_official_*_threshold_sensitivity.csv` |
| Tab. — baseline VLM | `scripts/experiments/81_vlm_baseline_distance_nearest.py` |
| Sec./Cap. — E2E GPT-4.1 e SpatialLM | `scripts/experiments/84_e2e_grounding_test_official.py` |
| SCI — matriz multi-VLM | `scripts/plot_context_vs_no_context_ieee_final.py` |
| Estratificação de ambiguidade | `scripts/extract_ambiguity_stratification.py` |
| p-valores (McNemar) por query | `scripts/extract_query_level_pvalues.py` |
| Validação da distância de superfície | `scripts/validate_surface_distance_against_triangle_oracle.py` |

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
# Benchmark métrico (distance + nearest)
python scripts/benchmark/61_run_benchmark_distance_nearest_test_official_stage1.py

# Benchmark relacional (between + aligned)
python scripts/benchmark/70_run_benchmark_relational_binary_test_official_stage1.py

# Baseline VLM
python scripts/experiments/81_vlm_baseline_distance_nearest.py

# Experimento E2E (grounding automático)
python scripts/experiments/84_e2e_grounding_test_official.py
```

As chaves de API dos modelos de visão e linguagem são lidas de variáveis de
ambiente (ex.: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`). Defina-as
em um arquivo `.env` local (não versionado).

## Limiares Oficiais

Calibrados no conjunto de desenvolvimento como o menor valor em que acurácia,
revocação e F1 atingem o platô em 1.0 sem degradar a precisão:

- `τ_between = 0.30`
- `τ_aligned = 0.25`

Definidos em `configs/benchmark_config.yaml`.

## Dataset

O benchmark usa cenas do [ScanNet](http://www.scan-net.org/). Os dados
geométricos brutos não são redistribuídos neste repositório por questões de
licença; siga as instruções do ScanNet para obter acesso. Os arquivos de
benchmark (queries, ground truth, manifestos) em `benchmark/` permitem reproduzir
as métricas sem os dados brutos.

## Citação

```bibtex
@mastersthesis{lopes2026raciocinio,
  author  = {Jefferson Lopes},
  title   = {Raciocínio Espacial sobre Cenas 3D via Representação Geométrica Explícita},
  school  = {Programa de Pós-Graduação em Mecatrônica (PPGM), Universidade Federal da Bahia (UFBA)},
  year    = {2026},
  type    = {Qualificação de Mestrado}
}
```

## Licença

MIT License — veja `LICENSE`.
