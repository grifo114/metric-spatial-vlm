
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ScanNet](https://img.shields.io/badge/Dataset-ScanNet-orange.svg)](http://www.scan-net.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/mit)

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
visual input fixed.

## Repository scope

This repository is intended for reproducibility and audit of the paper
experiments. It does not contain raw ScanNet data.

## Benchmark

The benchmark contains:

- 20 ScanNet scenes;
- 45 reviewed distance queries;
- 8 object categories;
- explicit ground-truth instance pairs for each query;
- 4 VLMs;
- 5 prompt conditions;
- 2 runs per condition;
- 1,800 multimodal calls.

## Prompt conditions

- Baseline: no spatial descriptors.
- L1: object position in a 3 x 3 scene grid.
- L2: L1 plus peer ordering within the same category.
- L3: L2 plus nearby object categories.
- L2-Ref: L2 descriptors only for queried categories.

## Repository structure

```text
benchmark/      Benchmark metadata and reviewed query files.
configs/        Model and benchmark configuration files.
docs/           Benchmark card, data access, and reproducibility notes.
figures/        Figures used in the paper.
prompts/        Prompt templates for each condition.
results/        Raw and processed outputs used in the paper.
scripts/        Evaluation and analysis scripts.
```
## Data access

Raw ScanNet data are not redistributed. Users must obtain ScanNet through
the official access procedure and follow the original dataset terms.

See docs/DATA_ACCESS.md.

Reproducibility

See docs/REPRODUCIBILITY.md.

Citation

Citation information will be added after submission or acceptance.


## License

The code in this repository is released under the MIT License.

The benchmark metadata, prompts, configuration files, and result-processing
scripts are provided for research and reproducibility purposes.

Raw ScanNet data are not redistributed in this repository. Access to
ScanNet is subject to the original ScanNet license and terms of use.

Model outputs included in this repository are provided only to support
auditability and reproduction of the paper results.
