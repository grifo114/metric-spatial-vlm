# Data Access

This repository does not redistribute raw ScanNet data.

The benchmark uses selected ScanNet scenes and derived metadata for
reproducibility and audit of the paper:

**Spatial Context Injection for Metric Distance Queries in 3D Indoor Scenes**

## ScanNet

Raw ScanNet data must be obtained through the official ScanNet access
procedure. Users are responsible for complying with the original ScanNet
license and terms of use.

Official dataset page:

https://www.scan-net.org/

## Files included in this repository

This repository includes benchmark metadata, such as:

- selected scene identifiers;
- reviewed distance queries;
- object category metadata;
- reference instance pairs;
- model outputs used to reproduce the reported tables.

## Files not included

The repository does not include:

- raw RGB-D scans;
- ScanNet `.sens` files;
- raw reconstructed meshes;
- raw point clouds;
- full dataset archives.

## Expected local data layout

After obtaining ScanNet, users may organize local data as:

```text
data/
└── scannet/
    ├── scans/
    └── scans_test/
