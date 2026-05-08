"""
Generates panel (c) of the conceptual figure: Error Decomposition.

Companion to panels (a) and (b), which show the rendered scene and
the system's prediction overlay (generated separately from the
ScanNet pipeline).

Outputs both PDF (LaTeX-ready) and SVG (editable in Inkscape).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Use mathtext (no system LaTeX needed); Times-like serif
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",   # Times-like math
    "pdf.fonttype": 42,           # TrueType, editable in Inkscape
    "svg.fonttype": "none",       # Keep text as text, not paths
})

# Figure: ~3.6 inch wide, ~4.6 inch tall (matches typical (a)(b) panel height)
fig, ax = plt.subplots(figsize=(3.6, 4.6), dpi=300)
ax.set_xlim(0, 360)
ax.set_ylim(0, 460)
ax.invert_yaxis()           # SVG-like coords: y grows downward
ax.set_aspect("equal")      # preserve coordinate proportions
ax.axis("off")
# Make the axes fill the figure exactly — no margins
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Outer container
outer = FancyBboxPatch(
    (8, 8), 344, 444,
    boxstyle="round,pad=0,rounding_size=12",
    linewidth=1.5, edgecolor="#2a2a2a", facecolor="white",
)
ax.add_patch(outer)

# --- Title -----------------------------------------------------------
ax.text(180, 38, "Error Decomposition",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color="#1a1a1a")
ax.plot([30, 330], [52, 52], color="#bbb", linewidth=0.8)

# --- Main equation ---------------------------------------------------
ax.text(180, 100,
        r"$E_{\mathrm{total}} \;=\; E_{\mathrm{grounding}} \;+\; E_{\mathrm{geom}}$",
        ha="center", va="center", fontsize=15, color="#1a1a1a")

ax.text(180, 128, "illustrative query: distance(door, table)",
        ha="center", va="center", fontsize=9, style="italic", color="#666")

# --- E_geom box (navy/blue, "exact") ---------------------------------
geom_box = FancyBboxPatch(
    (28, 150), 304, 62,
    boxstyle="round,pad=0,rounding_size=8",
    linewidth=1.2, edgecolor="#1976d2", facecolor="#e3f2fd",
)
ax.add_patch(geom_box)
ax.text(46, 178, r"$E_{\mathrm{geom}} \;=\; 0.000$ m",
        ha="left", va="center", fontsize=12, color="#0d47a1")
ax.text(46, 200, "surface engine is exact",
        ha="left", va="center", fontsize=9, style="italic", color="#1565c0")

# --- E_grounding box (orange, "dominant") ----------------------------
grnd_box = FancyBboxPatch(
    (28, 225), 304, 62,
    boxstyle="round,pad=0,rounding_size=8",
    linewidth=1.5, edgecolor="#fb8c00", facecolor="#ffe0b2",
)
ax.add_patch(grnd_box)
ax.text(46, 253, r"$E_{\mathrm{grounding}} \;=\; 0.834$ m",
        ha="left", va="center", fontsize=12, color="#bf360c")
ax.text(46, 275, "VLM picked the wrong instance",
        ha="left", va="center", fontsize=9, style="italic", color="#d84315")

# Divider
ax.plot([30, 330], [315, 315], color="#bbb", linewidth=0.8)

# --- Centroid baseline ------------------------------------------------
ax.text(180, 340, "Centroid baseline (model-independent)",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="#444")
ax.text(180, 380,
        r"$d_{\mathrm{cent}} - d_{\mathrm{surf}} \;\approx\; 1.0$ m",
        ha="center", va="center", fontsize=14, color="#444")
ax.text(180, 412, "constant bias, removed only by surface distance",
        ha="center", va="center", fontsize=9, style="italic", color="#666")

fig.savefig("panel_c_decomposition.pdf", pad_inches=0)
fig.savefig("panel_c_decomposition.svg", pad_inches=0)
fig.savefig("panel_c_decomposition.png", pad_inches=0, dpi=300)
print("Wrote panel_c_decomposition.{pdf,svg,png}")