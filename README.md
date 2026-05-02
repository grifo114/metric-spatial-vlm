# Spatial Context Injection for Metric Distance Queries in 3D Indoor Scenes

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ScanNet](https://img.shields.io/badge/Dataset-ScanNet-orange.svg)](http://www.scan-net.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the benchmark files, prompts, scripts, and result-processing code used in the paper:

**Spatial Context Injection for Metric Distance Queries in 3D Indoor Scenes**

The project evaluates metric distance queries over reconstructed 3D indoor scenes by separating two error sources:

- instance grounding error;
- geometric distance-computation error.

The method combines a surface-based distance engine with Spatial Context Injection (SCI), a textual prompt-enrichment strategy that adds scene-relative descriptors to candidate objects.

## Repository status

This repository is intended for reproducibility and audit of the benchmark and paper results.

## Main components

- `benchmark/`: scene IDs, query files, object categories, and ground-truth instance pairs.
- `prompts/`: prompt templates for baseline and SCI variants.
- `scripts/`: metric computation, run agreement, table generation, and figure generation.
- `results/`: raw and processed model outputs.
- `docs/`: benchmark documentation, data access instructions, and audit trail.
- `paper/`: paper source files or compiled PDF.

## Dataset

The benchmark uses ScanNet scenes. Raw ScanNet data are not redistributed in this repository. Users must obtain ScanNet through the official access procedure.

See `docs/DATA_ACCESS.md` for details.

## Evaluation summary

The benchmark contains:

- 20 ScanNet scenes;
- 45 reviewed distance queries;
- 8 object categories;
- 4 VLMs;
- 5 prompt conditions;
- 2 runs per condition;
- 1,800 multimodal calls.

## Prompt conditions

- Baseline: no spatial descriptors.
- L1: scene-grid location.
- L2: L1 plus peer ordering.
- L3: L2 plus nearby object categories.
- L2-Ref: L2 descriptors only for queried categories.

## Reproducibility

See `docs/REPRODUCIBILITY.md`.

## License

Code is released under the license specified in `LICENSE`. Dataset access follows the original ScanNet terms.
