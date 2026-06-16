#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import trimesh
import numpy as np
import fcl


def mesh_to_fcl(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    model = fcl.BVHModel()
    model.beginModel(len(vertices), len(faces))
    model.addSubModel(vertices, faces)
    model.endModel()

    return fcl.CollisionObject(model)


def triangle_mesh_distance(mesh_a, mesh_b):
    obj_a = mesh_to_fcl(mesh_a)
    obj_b = mesh_to_fcl(mesh_b)

    req = fcl.DistanceRequest(enable_nearest_points=True)
    res = fcl.DistanceResult()

    return float(fcl.distance(obj_a, obj_b, req, res))


def main():
    queries = pd.read_csv("benchmark/ground_truth_distance_nearest_test_official_stage1.csv")
    queries = queries[queries["operator"] == "distance"].copy()

    rows = []

    for _, row in queries.iterrows():
        query_id = row["query_id"]
        object_a = row["gt_object_a"]
        object_b = row["gt_object_b"]

        mesh_a_path = Path("artifacts/object_meshes") / f"{object_a}.ply"
        mesh_b_path = Path("artifacts/object_meshes") / f"{object_b}.ply"

        mesh_a = trimesh.load(mesh_a_path, process=False)
        mesh_b = trimesh.load(mesh_b_path, process=False)

        d_tri = triangle_mesh_distance(mesh_a, mesh_b)
        d_surf = float(row["gt_distance_m"])

        rows.append({
            "query_id": query_id,
            "object_a": object_a,
            "object_b": object_b,
            "d_surf": d_surf,
            "d_tri": d_tri,
            "abs_error": abs(d_surf - d_tri),
        })

    out = pd.DataFrame(rows)

    print("n:", len(out))
    print("mean:", out["abs_error"].mean())
    print("median:", out["abs_error"].median())
    print("p95:", out["abs_error"].quantile(0.95))
    print("max:", out["abs_error"].max())

    out.to_csv("results/triangle_oracle/surface_vs_triangle_oracle.csv", index=False)


if __name__ == "__main__":
    main()