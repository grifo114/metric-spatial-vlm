[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ScanNet](https://img.shields.io/badge/Dataset-ScanNet-orange.svg)](http://www.scan-net.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# Spatial Context Injection for Metric Distance Queries in 3D Indoor Scenes

This repository contains the code, benchmark metadata, prompts, figures,
and processed outputs used in the paper:

**Spatial Context Injection for Metric Distance Queries in 3D Indoor Scenes**

The paper evaluates metric distance queries over reconstructed 3D indoor
scenes by separating two sources of error:

1. instance grounding error;
2. geometric distance-computation error.

The proposed Spatial Context Injection (SCI) strategy enriches candidate
object lists with scene-relative textual descriptors while keeping the
visual input fixed. The selected object pair is then passed to a deterministic
surface-based geometric engine, which computes the minimum surface-to-surface
distance between the two reconstructed object meshes.

## Demo

A short system demo is available on YouTube:

[Watch the demo](COLE_AQUI_O_LINK_DO_YOUTUBE)

The demo illustrates the main pipeline: top-down scene rendering, numbered
candidate objects, textual object list construction, VLM-based object
selection, and surface-based distance computation.

## Repository scope

This repository is intended for reproducibility and audit of the paper
experiments. It does not contain raw ScanNet data.

Raw ScanNet meshes must be obtained through the official ScanNet access
procedure and used according to the original dataset terms.

## Benchmark

The benchmark contains:

- 20 ScanNet scenes;
- 45 reviewed distance queries;
- 8 object categories;
- explicit ground-truth instance pairs for each query;
- 6 VLMs;
- 5 prompt conditions;
- 2 runs per condition;
- 2,700 multimodal calls.

The evaluated models are:

- GPT-4.1;
- Claude Sonnet 4.5;
- Gemini 2.5 Flash;
- Qwen3-VL-235B-A22B-Instruct;
- Qwen3-VL-32B-Instruct;
- Qwen3-VL-8B-Instruct.

## Prompt conditions

- **Baseline**: no spatial descriptors.
- **L1**: object position in a 3 x 3 scene grid.
- **L2**: L1 plus peer ordering within the same category.
- **L3**: L2 plus nearby object categories.
- **L2-Ref**: L2 descriptors only for queried categories.

Example of an L2 object-list entry:

```text
1: chair (right side of the scene; 5 of 5 chairs left to right)
   -> scene0087_00__chair_004
```
## Repository structure

```text
benchmark/      Benchmark metadata and reviewed query files.
configs/        Model and benchmark configuration files.
docs/           Benchmark card, data access, and reproducibility notes.
figures/        Figures and summary plots used in the paper.
prompts/        Prompt templates for each condition.
results/        Processed outputs and summary result tables.
scripts/        Evaluation, analysis, plotting, and statistical scripts.
paper/          Paper source files, if included.
```

## Main scripts   

# Main VLM grounding experiment
python scripts/83_e2e_grounding_test_official_v2.py

# Claude-specific runner through the Anthropic API
python scripts/83b_claude_e2e_grounding.py

# End-to-end MAE computation
python scripts/mae.py

# McNemar tests at query-run level
python scripts/mcnemar.py

# Query-level McNemar aggregation checks
python scripts/mcnemar_query_level.py

# Section V statistics
python scripts/extract_section_v_stats.py

# Plot generation
python scripts/plot_accuracy_all_models.py
python scripts/plot_mae_surface_all_models.py

## Data access
Raw ScanNet data are not redistributed. Users must obtain ScanNet through
the official access procedure and follow the original dataset terms.

See docs/DATA_ACCESS.md.

## Reproducibility
The repository provides scripts for rebuilding the benchmark artifacts from
a local authorized ScanNet download. The released files include query
definitions, prompt construction code, SCI descriptor generation, evaluation
scripts, statistical tests, and plotting scripts.

See docs/REPRODUCIBILITY.md, if available.

## Citation