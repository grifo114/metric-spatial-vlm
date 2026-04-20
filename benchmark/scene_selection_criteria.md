# Scene Selection Criteria

## Objective

Define a reproducible protocol for selecting ScanNet scenes for the benchmark on spatial querying over structured 3D scenes.

## Primary dataset

The main benchmark uses ScanNet as the primary dataset.

## Target benchmark size

- 100 scenes total
- 20 development scenes
- 80 test scenes

## Room-type stratification

Whenever possible, the selected scenes should be distributed across the following room groups:

- bedroom
- living room
- office/study
- mixed/other

Target distribution:

- 25 scenes per group

Development split target:

- 5 scenes per group

Test split target:

- 20 scenes per group

## Scene-level inclusion criteria

A scene is considered a valid candidate if it satisfies all the following conditions:

1. It contains at least 8 valid objects after filtering.
2. It contains at least 3 benchmark categories.
3. It contains at least 1 surface-like object among:
   - table
   - desk
4. It contains at least 1 furniture object among:
   - chair
   - sofa
   - bed
5. It contains at least 1 structural or reference object among:
   - door
   - cabinet
   - monitor
6. The scene point cloud is not severely corrupted or incomplete.

## Object-level validity criteria

An object is considered valid for benchmarking if:

1. Its normalized label belongs to the benchmark category set.
2. It has enough points:
   - at least 150 points for most categories
   - at least 80 points for monitor
3. Its 3D bounding box is not degenerate.
4. Its geometric extent is plausible.
5. Its point cloud is not severely fragmented.

## Scene-level exclusion criteria

A scene should be excluded if any of the following holds:

- too few valid objects after filtering
- extremely poor object segmentation
- missing or unusable structural geometry
- excessive fragmentation in most target objects
- strong lack of category diversity

## Notes on manual review

Automatic filtering is used as the first stage only.

After automatic scoring, scenes should be manually reviewed to:

- confirm room-type assignment
- remove clearly problematic scans
- avoid over-representation of nearly identical scenes
- ensure geometric diversity

## Reproducibility

The following artifacts must be saved:

- full scene inventory table
- final list of development scenes
- final list of test scenes
- version of the filtering script
- date of benchmark freeze
