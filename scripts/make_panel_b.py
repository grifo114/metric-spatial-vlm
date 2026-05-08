"""make_panel_b.py — panel (b) v3 (cropped + recolored)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

CONFIG = {
    "scene_id":       "scene0012_01",
    "gt_object_a":    "scene0012_01__door_032",
    "gt_object_b":    "scene0012_01__table_008",
    "grounded_a":     "scene0012_01__door_029",
    "grounded_b":     "scene0012_01__table_008",
    "gt_value":       0.586,
    "predicted_dist": 1.420,
    "manifest":       "benchmark/objects_manifest_test_official_stage1.csv",
    "render":         "artifacts/e2e_grounding_renders/scene0012_01_numbered.jpg",
    "out":            "panel_b_overlay.pdf",
    "crop_margin_m":  1.0,
}
GT_COLOR = "#0d47a1"
PR_COLOR = "#e65100"


def compute_pipeline_bounds(scene_df):
    cx = float(scene_df["centroid_x"].mean())
    cy = float(scene_df["centroid_y"].mean())
    x_ext = float(scene_df["aabb_max_x"].max() - scene_df["aabb_min_x"].min())
    y_ext = float(scene_df["aabb_max_y"].max() - scene_df["aabb_min_y"].min())
    extent = max(x_ext, y_ext, 1.0) * 1.25
    return (cx - extent/2, cy - extent/2, cx + extent/2, cy + extent/2)

def world_to_image(x, y, bounds, image_size):
    x_min, y_min, x_max, y_max = bounds
    W, H = image_size
    return ((x - x_min) / (x_max - x_min) * W,
            (y_max - y) / (y_max - y_min) * H)

def bbox_world_to_image(row, bounds, image_size):
    x0, y0 = float(row["aabb_min_x"]), float(row["aabb_min_y"])
    x1, y1 = float(row["aabb_max_x"]), float(row["aabb_max_y"])
    p00 = world_to_image(x0, y0, bounds, image_size)
    p11 = world_to_image(x1, y1, bounds, image_size)
    px0, px1 = sorted([p00[0], p11[0]])
    py0, py1 = sorted([p00[1], p11[1]])
    return px0, py0, px1 - px0, py1 - py0

def centroid_image(row, bounds, image_size):
    return world_to_image(float(row["centroid_x"]), float(row["centroid_y"]),
                          bounds, image_size)


def main():
    cfg = CONFIG
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","DejaVu Serif"],
                         "mathtext.fontset":"stix","pdf.fonttype":42,"svg.fonttype":"none"})

    mdf = pd.read_csv(cfg["manifest"])
    sdf = mdf[(mdf["scene_id"]==cfg["scene_id"]) & (mdf["is_valid_object"]==True)].copy()
    by_id = {r["object_id"]: r for _, r in sdf.iterrows()}
    needed = [cfg["gt_object_a"], cfg["gt_object_b"], cfg["grounded_a"], cfg["grounded_b"]]
    bounds = compute_pipeline_bounds(sdf)
    print(f"Pipeline bounds: x[{bounds[0]:.3f}, {bounds[2]:.3f}], y[{bounds[1]:.3f}, {bounds[3]:.3f}]")

    img = np.asarray(Image.open(cfg["render"]).convert("RGB"))
    H_img, W_img = img.shape[:2]
    image_size = (W_img, H_img)
    print(f"Render: {W_img}x{H_img}")

    gt_a_b = bbox_world_to_image(by_id[cfg["gt_object_a"]], bounds, image_size)
    gt_b_b = bbox_world_to_image(by_id[cfg["gt_object_b"]], bounds, image_size)
    pr_a_b = bbox_world_to_image(by_id[cfg["grounded_a"]],  bounds, image_size)
    pr_b_b = bbox_world_to_image(by_id[cfg["grounded_b"]],  bounds, image_size)
    gt_a_c = centroid_image(by_id[cfg["gt_object_a"]], bounds, image_size)
    gt_b_c = centroid_image(by_id[cfg["gt_object_b"]], bounds, image_size)
    pr_a_c = centroid_image(by_id[cfg["grounded_a"]],  bounds, image_size)
    pr_b_c = centroid_image(by_id[cfg["grounded_b"]],  bounds, image_size)

    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=300)
    ax.imshow(img)

    margin = float(cfg["crop_margin_m"])
    xs, ys = [], []
    for oid in needed:
        r = by_id[oid]
        xs += [float(r["aabb_min_x"]), float(r["aabb_max_x"])]
        ys += [float(r["aabb_min_y"]), float(r["aabb_max_y"])]
    px_lo, py_hi = world_to_image(min(xs)-margin, min(ys)-margin, bounds, image_size)
    px_hi, py_lo = world_to_image(max(xs)+margin, max(ys)+margin, bounds, image_size)
    px_lo = max(0, min(W_img, px_lo)); px_hi = max(0, min(W_img, px_hi))
    py_lo = max(0, min(H_img, py_lo)); py_hi = max(0, min(H_img, py_hi))
    ax.set_xlim(px_lo, px_hi); ax.set_ylim(py_hi, py_lo)
    print(f"Cropped to pixel box: x[{px_lo:.0f}, {px_hi:.0f}], y[{py_lo:.0f}, {py_hi:.0f}]")
    ax.set_aspect("equal"); ax.axis("off")

    for x, y, w, h in (gt_a_b, gt_b_b):
        ax.add_patch(Rectangle((x,y), w, h, fill=False, edgecolor=GT_COLOR, linewidth=2.5))
    for x, y, w, h in (pr_a_b, pr_b_b):
        ax.add_patch(Rectangle((x,y), w, h, fill=False, edgecolor=PR_COLOR, linewidth=2.5,
                               linestyle=(0,(5,3))))

    ax.plot([gt_a_c[0], gt_b_c[0]], [gt_a_c[1], gt_b_c[1]], color=GT_COLOR, linewidth=2)
    mx, my = (gt_a_c[0]+gt_b_c[0])/2, (gt_a_c[1]+gt_b_c[1])/2
    ax.text(mx, my-16, r"$d^{\,*}_{\mathrm{surf}} = " + f"{cfg['gt_value']:.3f}" + r"$ m",
            color=GT_COLOR, fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GT_COLOR, lw=0.7))

    ax.plot([pr_a_c[0], pr_b_c[0]], [pr_a_c[1], pr_b_c[1]], color=PR_COLOR, linewidth=2,
            linestyle=(0,(5,3)))
    mx, my = (pr_a_c[0]+pr_b_c[0])/2, (pr_a_c[1]+pr_b_c[1])/2
    ax.text(mx, my+16, r"$d_{\mathrm{surf}} = " + f"{cfg['predicted_dist']:.3f}" + r"$ m",
            color=PR_COLOR, fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PR_COLOR, lw=0.7))

    ax.text(px_hi-12, py_lo+18, r"GT $(o_a^*,\, o_b^*)$",
            color=GT_COLOR, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GT_COLOR, lw=0.6))
    ax.text(px_hi-12, py_lo+56, r"prediction $(\hat{o}_a,\, \hat{o}_b)$",
            color=PR_COLOR, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PR_COLOR, lw=0.6))

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(cfg["out"], pad_inches=0)
    fig.savefig(cfg["out"].replace(".pdf",".png"), pad_inches=0, dpi=300)
    print(f"Wrote {cfg['out']} (and .png)")
    print(f"E_grounding = {abs(cfg['gt_value']-cfg['predicted_dist']):.3f} m")

if __name__ == "__main__":
    main()
