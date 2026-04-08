import json
from pathlib import Path

import numpy as np
import open3d as o3d


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENE_ID = "scene0114_00"

SCENE_DIR = PROJECT_ROOT / "data" / "scannet" / "scans" / SCENE_ID
SCENE_PLY = SCENE_DIR / f"{SCENE_ID}_vh_clean_2.ply"
SEGS_JSON = SCENE_DIR / f"{SCENE_ID}_vh_clean_2.0.010000.segs.json"
AGG_JSON = SCENE_DIR / f"{SCENE_ID}.aggregation.json"

QUERY_RESULT_PATH = PROJECT_ROOT / "results" / "query_results" / f"{SCENE_ID}_query_result.json"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scene_mesh_and_vertices(ply_path: Path):
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    if mesh.is_empty():
        raise FileNotFoundError(f"Could not load mesh: {ply_path}")
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    return mesh, vertices


def load_segmentation(segs_path: Path):
    with open(segs_path, "r", encoding="utf-8") as f:
        segs_data = json.load(f)
    return np.array(segs_data["segIndices"])


def load_aggregation(agg_path: Path):
    with open(agg_path, "r", encoding="utf-8") as f:
        agg_data = json.load(f)
    return agg_data["segGroups"]


def get_object_group(seg_groups, object_id: int):
    for group in seg_groups:
        if group["objectId"] == object_id:
            return group
    raise ValueError(f"objectId {object_id} not found in aggregation.")


def extract_object_vertices(vertices, seg_indices, segments):
    mask = np.isin(seg_indices, list(segments))
    return vertices[mask]


def make_colored_point_cloud(points: np.ndarray, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.tile(np.array(color), (len(points), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_centroid_marker(center, radius=0.05, color=(1.0, 1.0, 0.0)):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


def make_connection_line(p1, p2, color=(1.0, 0.0, 0.0)):
    line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([p1, p2]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    line.colors = o3d.utility.Vector3dVector([color])
    return line


# ============================================================
# PIPELINE
# ============================================================

def main():
    print(f"[INFO] Loading query result: {QUERY_RESULT_PATH}")

    query_result = load_json(QUERY_RESULT_PATH)

    object_a_id = query_result["object_a"]["object_id"]
    object_b_id = query_result["object_b"]["object_id"]

    label_a = query_result["object_a"]["label"]
    label_b = query_result["object_b"]["label"]

    centroid_a = np.array(query_result["object_a"]["centroid"])
    centroid_b = np.array(query_result["object_b"]["centroid"])

    distance_m = query_result["distance_m"]
    answer = query_result["answer"]

    print(f"[INFO] Query: {query_result['raw_query']}")
    print(f"[INFO] Object A: {object_a_id} ({label_a})")
    print(f"[INFO] Object B: {object_b_id} ({label_b})")
    print(f"[INFO] Distance: {distance_m:.4f} m")
    print(f"[INFO] Answer: {answer}")

    # carregar cena e segmentação
    mesh, vertices = load_scene_mesh_and_vertices(SCENE_PLY)
    seg_indices = load_segmentation(SEGS_JSON)
    seg_groups = load_aggregation(AGG_JSON)

    # encontrar grupos dos objetos
    group_a = get_object_group(seg_groups, object_a_id)
    group_b = get_object_group(seg_groups, object_b_id)

    pts_a = extract_object_vertices(vertices, seg_indices, set(group_a["segments"]))
    pts_b = extract_object_vertices(vertices, seg_indices, set(group_b["segments"]))

    if len(pts_a) == 0 or len(pts_b) == 0:
        raise ValueError("One of the selected objects has no vertices.")

    # point clouds coloridas
    pcd_a = make_colored_point_cloud(pts_a, color=(0.0, 1.0, 0.0))  # verde
    pcd_b = make_colored_point_cloud(pts_b, color=(0.0, 0.0, 1.0))  # azul

    # caixas
    bbox_a = pcd_a.get_axis_aligned_bounding_box()
    bbox_a.color = (0.0, 1.0, 0.0)

    bbox_b = pcd_b.get_axis_aligned_bounding_box()
    bbox_b.color = (0.0, 0.0, 1.0)

    # linha e marcadores
    line = make_connection_line(centroid_a, centroid_b, color=(1.0, 0.0, 0.0))
    marker_a = make_centroid_marker(centroid_a, radius=0.05, color=(1.0, 1.0, 0.0))
    marker_b = make_centroid_marker(centroid_b, radius=0.05, color=(1.0, 1.0, 0.0))
    marker_mid = make_centroid_marker((centroid_a + centroid_b) / 2.0, radius=0.04, color=(1.0, 0.0, 0.0))

    # visualizador
    window_title = f"{label_a} ↔ {label_b} | d = {distance_m:.2f} m"
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title)

    vis.add_geometry(mesh)
    vis.add_geometry(bbox_a)
    vis.add_geometry(bbox_b)
    vis.add_geometry(line)
    vis.add_geometry(marker_a)
    vis.add_geometry(marker_b)
    vis.add_geometry(marker_mid)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([1.0, 1.0, 1.0])
    render_option.point_size = 2.0
    try:
        render_option.line_width = 10.0
    except Exception:
        pass

    vis.poll_events()
    vis.update_renderer()

    print("\n[INFO] Interactive demo window opened.")
    print("[INFO] Adjust the camera manually and take a screenshot if needed.\n")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()