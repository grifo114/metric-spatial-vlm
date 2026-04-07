import json
from pathlib import Path

import numpy as np
import open3d as o3d


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = PROJECT_ROOT / "data" / "scannet" / "scans" / "scene0114_00"

SCENE_PLY = SCENE_DIR / "scene0114_00_vh_clean_2.ply"
SEGS_JSON = SCENE_DIR / "scene0114_00_vh_clean_2.0.010000.segs.json"
AGG_JSON = SCENE_DIR / "scene0114_00.aggregation.json"

# Escolha manual dos objetos
OBJECT_ID_A = 2   # chair
OBJECT_ID_B = 13  # desk


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def load_scene_mesh_and_vertices(ply_path: Path):
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    if mesh.is_empty():
        raise FileNotFoundError(f"Não foi possível carregar a malha: {ply_path}")
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
    raise ValueError(f"objectId {object_id} não encontrado na agregação.")


def get_object_vertices(vertices, seg_indices, seg_groups, object_id: int):
    group = get_object_group(seg_groups, object_id)
    target_segments = set(group["segments"])
    mask = np.isin(seg_indices, list(target_segments))
    obj_vertices = vertices[mask]

    if len(obj_vertices) == 0:
        raise ValueError(f"Objeto {object_id} não possui vértices associados.")

    return obj_vertices, group["label"]


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
# DEMO
# ============================================================

def main():
    print("\n[INFO] Iniciando demo manual da scene0114_00\n")

    # Carregar cena
    mesh, vertices = load_scene_mesh_and_vertices(SCENE_PLY)
    seg_indices = load_segmentation(SEGS_JSON)
    seg_groups = load_aggregation(AGG_JSON)

    # Extrair objetos
    obj_a_pts, label_a = get_object_vertices(vertices, seg_indices, seg_groups, OBJECT_ID_A)
    obj_b_pts, label_b = get_object_vertices(vertices, seg_indices, seg_groups, OBJECT_ID_B)

    # Calcular centroides
    c_a = obj_a_pts.mean(axis=0)
    c_b = obj_b_pts.mean(axis=0)

    # Distância
    dist = np.linalg.norm(c_a - c_b)

    # Imprimir resumo
    print(f"[INFO] Objeto A: id={OBJECT_ID_A}, label={label_a}, n_points={len(obj_a_pts)}")
    print(f"[INFO] Objeto B: id={OBJECT_ID_B}, label={label_b}, n_points={len(obj_b_pts)}")
    print(f"[INFO] Distância entre centroides: {dist:.3f} m\n")

    # Objetos destacados
    pcd_a = make_colored_point_cloud(obj_a_pts, color=(0.0, 1.0, 0.0))  # verde
    pcd_b = make_colored_point_cloud(obj_b_pts, color=(0.0, 0.0, 1.0))  # azul

    bbox_a = pcd_a.get_axis_aligned_bounding_box()
    bbox_a.color = (0.0, 1.0, 0.0)

    bbox_b = pcd_b.get_axis_aligned_bounding_box()
    bbox_b.color = (0.0, 0.0, 1.0)

    line = make_connection_line(c_a, c_b, color=(1.0, 0.0, 0.0))  # vermelho

    marker_a = make_centroid_marker(c_a, radius=0.05, color=(1.0, 1.0, 0.0))
    marker_b = make_centroid_marker(c_b, radius=0.05, color=(1.0, 1.0, 0.0))
    marker_mid = make_centroid_marker((c_a + c_b) / 2.0, radius=0.04, color=(1.0, 0.0, 0.0))

    # Visualização
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"scene0114_00 | {label_a} ↔ {label_b} | d = {dist:.2f} m")

    vis.add_geometry(mesh)       # cena com cor do .ply
    vis.add_geometry(bbox_a)
    vis.add_geometry(bbox_b)
    vis.add_geometry(line)
    vis.add_geometry(marker_a)
    vis.add_geometry(marker_b)
    vis.add_geometry(marker_mid)

    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.line_width = 10.0
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # fundo branco

    vis.poll_events()
    vis.update_renderer()

    print("[INFO] Janela aberta. Ajuste a câmera manualmente.")
    print("[Lembrete pra mim] FAZER DEPOIS automatizar exportação.\n")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()