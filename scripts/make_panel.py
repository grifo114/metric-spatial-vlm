"""
make_panel_b.py
================
Renders panel (b) of the paper's conceptual figure:
GT objects (green) vs VLM-predicted objects (orange) overlaid on the
top-down render the model actually saw, with both surface distances
annotated.

Concept:
  Panel (a) = the existing top-down marked render (what the VLM sees).
  Panel (b) = the same image with two pairs of bounding boxes plus two
              distance lines, telling the reader at a glance:
              "the model picked the wrong instance — this is E_grounding".

Pipeline:
  1. Load the existing 1024x1024 render of the chosen scene as background.
  2. For each object (GT and predicted), convert its world-XY bounding box
     into image pixel coordinates using the same orthographic projection
     that produced the render.
  3. Draw rectangles, distance segments, and a tiny legend.
  4. Save as PDF (LaTeX-ready) and PNG.

How to use:
  - Pick ONE illustrative case (see select_example.py).
  - Fill in the CONFIG block below.
  - Run: python make_panel_b.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np

# ============================================================
# CONFIG — fill in for your chosen example
# ============================================================
CONFIG = {
    # Path to the existing top-down marked render (panel a)
    "render_path":  "artifacts/directional_renders/scene0142_00/topdown_marked.png",

    # Pixel size of the render
    "image_size":   (1024, 1024),

    # World XY bounds used by your rendering pipeline (in meters).
    # These are the values your render script feeds into the orthographic
    # projection — open script ~78 (or whichever produces the marked render)
    # and copy the (x_min, y_min, x_max, y_max) it uses for THIS scene.
    "xy_bounds":    (-2.5, -3.0, 4.0, 3.5),

    # Ground-truth pair (drawn green, solid).
    # bbox_xy is the axis-aligned XY bounding box of the object's mesh,
    # in world meters. You can compute it from the mesh:
    #     mesh.vertices[:, [0,1]].min(0), .max(0)
    # or read it from objects_manifest if it's already there.
    "gt_objects": [
        {"name": "chair_4", "bbox_xy": (1.2, 0.4, 1.8, 1.0)},
        {"name": "table_2", "bbox_xy": (2.6, 0.2, 3.6, 1.4)},
    ],

    # VLM-predicted pair (drawn orange, dashed).
    "pr_objects": [
        {"name": "chair_7", "bbox_xy": (-1.5, 1.2, -0.9, 1.8)},
        {"name": "table_2", "bbox_xy": (2.6, 0.2, 3.6, 1.4)},
    ],

    # Surface distances (meters) — read these from your benchmark CSVs.
    "d_surf_gt":   0.946,   # ground-truth surface distance
    "d_surf_pred": 0.160,   # VLM-predicted surface distance

    # Output
    "out_path":    "panel_b_overlay.pdf",
}

# Colors (colorblind-safe Okabe-Ito-ish)
GT_COLOR = "#2e7d32"   # green
PR_COLOR = "#e65100"   # orange


# ============================================================
# Coordinate transform: world XY (m) → image pixels
# ============================================================
def world_to_image(x, y, xy_bounds, image_size):
    """Top-down orthographic projection used by the render pipeline."""
    x_min, y_min, x_max, y_max = xy_bounds
    W, H = image_size
    px = (x - x_min) / (x_max - x_min) * W
    py = (y_max - y) / (y_max - y_min) * H   # y flipped: image y grows down
    return px, py


def bbox_world_to_image(bbox, xy_bounds, image_size):
    """Convert world-XY bbox to (x, y, w, h) for matplotlib Rectangle."""
    x0, y0, x1, y1 = bbox
    p00 = world_to_image(x0, y0, xy_bounds, image_size)
    p11 = world_to_image(x1, y1, xy_bounds, image_size)
    px0, px1 = sorted([p00[0], p11[0]])
    py0, py1 = sorted([p00[1], p11[1]])
    return px0, py0, px1 - px0, py1 - py0


def bbox_center(bbox):
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


# ============================================================
# Render
# ============================================================
def main():
    cfg = CONFIG

    plt.rcParams.update({
        "font.family":     "serif",
        "font.serif":      ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype":    42,
        "svg.fonttype":    "none",
    })

    img = np.asarray(Image.open(cfg["render_path"]).convert("RGB"))
    W, H = cfg["image_size"]

    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=300)
    ax.imshow(img, extent=(0, W, H, 0))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect("equal"); ax.axis("off")

    # GT bounding boxes (solid green)
    for obj in cfg["gt_objects"]:
        x, y, w, h = bbox_world_to_image(
            obj["bbox_xy"], cfg["xy_bounds"], cfg["image_size"])
        ax.add_patch(Rectangle((x, y), w, h, fill=False,
                               edgecolor=GT_COLOR, linewidth=2.5))

    # Predicted bounding boxes (dashed orange)
    for obj in cfg["pr_objects"]:
        x, y, w, h = bbox_world_to_image(
            obj["bbox_xy"], cfg["xy_bounds"], cfg["image_size"])
        ax.add_patch(Rectangle((x, y), w, h, fill=False,
                               edgecolor=PR_COLOR, linewidth=2.5,
                               linestyle=(0, (5, 3))))

    # GT distance line
    p0 = world_to_image(*bbox_center(cfg["gt_objects"][0]["bbox_xy"]),
                        cfg["xy_bounds"], cfg["image_size"])
    p1 = world_to_image(*bbox_center(cfg["gt_objects"][1]["bbox_xy"]),
                        cfg["xy_bounds"], cfg["image_size"])
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
            color=GT_COLOR, linewidth=2)
    mx, my = (p0[0]+p1[0])/2, (p0[1]+p1[1])/2
    ax.text(mx, my - 14,
            r"$d_{\mathrm{surf}}^{\,*} = " + f"{cfg['d_surf_gt']:.3f}" + r"$ m",
            color=GT_COLOR, fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="white", ec=GT_COLOR, lw=0.6))

    # Predicted distance line
    p0 = world_to_image(*bbox_center(cfg["pr_objects"][0]["bbox_xy"]),
                        cfg["xy_bounds"], cfg["image_size"])
    p1 = world_to_image(*bbox_center(cfg["pr_objects"][1]["bbox_xy"]),
                        cfg["xy_bounds"], cfg["image_size"])
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
            color=PR_COLOR, linewidth=2, linestyle=(0, (5, 3)))
    mx, my = (p0[0]+p1[0])/2, (p0[1]+p1[1])/2
    ax.text(mx, my + 14,
            r"$d_{\mathrm{surf}} = " + f"{cfg['d_surf_pred']:.3f}" + r"$ m",
            color=PR_COLOR, fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="white", ec=PR_COLOR, lw=0.6))

    # Legend
    ax.text(W - 12, 18,
            r"green: ground truth $(o_a^*, o_b^*)$",
            color=GT_COLOR, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=GT_COLOR, lw=0.6))
    ax.text(W - 12, 56,
            r"orange: prediction $(\hat{o}_a, \hat{o}_b)$",
            color=PR_COLOR, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=PR_COLOR, lw=0.6))

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(cfg["out_path"], pad_inches=0)
    fig.savefig(cfg["out_path"].replace(".pdf", ".png"),
                pad_inches=0, dpi=300)
    print(f"Wrote {cfg['out_path']} (and .png companion)")


if __name__ == "__main__":
    main()