# Benchmark Card

## Benchmark name

Metric Distance Queries on 3D Indoor Scenes

## Purpose

This benchmark evaluates metric distance queries over reconstructed 3D
indoor scenes. It is designed to separate instance-grounding errors from
geometric distance-computation errors.

## Dataset source

The benchmark uses selected scenes from ScanNet.

Raw ScanNet data are not redistributed in this repository.

## Scene selection

Number of scenes: 20

Room categories:

- bedrooms: 5
- office or study rooms: 6
- living rooms: 3
- mixed indoor scenes: 6

## Object vocabulary

The retained object categories are:

- chair
- door
- monitor
- table
- cabinet
- desk
- sofa
- bed

## Query format

Each query has the form:

```text
distance(category_a, category_b)ø
