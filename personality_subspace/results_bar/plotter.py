#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Representational PCA diagram:
- Simulate high-D data
- PCA -> top-3 components
- 3D scatter (PC1, PC2, PC3)
- Mini pipeline header: "Big data → PCA (k=3) → 3D embedding"

Only requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import os

# -----------------------------
# Config (tweak as you like)
# -----------------------------
N_PER_CLASS = 400     # points per class
D = 50                # original dimensionality
SEED = 42
TITLE = "PCA: Top-3 Components (3D Embedding)"
OUTDIR = "figs"
OUTNAME = "pca_3d_embedding"

# -----------------------------
# Synthetic high-D data
# -----------------------------
rng = np.random.default_rng(SEED)

# Two Gaussian clouds with slight mean offset to make structure visible
mu0 = np.zeros(D)
mu1 = np.zeros(D)
mu1[:6] = 0.8  # small shift in first few dims

cov = np.eye(D)
X0 = rng.multivariate_normal(mu0, cov, size=N_PER_CLASS)
X1 = rng.multivariate_normal(mu1, cov, size=N_PER_CLASS)
X = np.vstack([X0, X1])
y = np.array([0]*N_PER_CLASS + [1]*N_PER_CLASS)

# Center data (standard PCA without scaling—good enough for a schematic)
Xc = X - X.mean(axis=0, keepdims=True)

# -----------------------------
# PCA via SVD (no sklearn)
# -----------------------------
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
# Columns of V (rows of Vt) are principal directions in feature space
# Project to first 3 PCs
k = 3
Vk = Vt[:k, :]                 # shape (3, D)
Z = Xc @ Vk.T                  # shape (N, 3), coordinates along PC1..PC3

# Explained variance ratio
eigvals = (S**2) / (X.shape[0] - 1)
evr = eigvals / eigvals.sum()
evr3 = evr[:3]

# -----------------------------
# Figure
# -----------------------------
plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

fig = plt.figure(figsize=(9.5, 6.5))

# ---- (A) Pipeline header (axes in figure coordinates) ----
ax_header = fig.add_axes([0.08, 0.83, 0.84, 0.12])  # [left, bottom, width, height]
ax_header.axis("off")

def box(ax, xy, text, box_color="#f0f0f0"):
    x, y = xy
    rect = FancyBboxPatch((x, y), 0.24, 0.55,
                          boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=1.0, edgecolor="#777777", facecolor=box_color)
    ax.add_patch(rect)
    ax.text(x+0.12, y+0.28, text, ha="center", va="center", fontsize=11)

# Three boxes
box(ax_header, (0.02, 0.15), "Big data\n(High-D)", box_color="#eaeaea")
box(ax_header, (0.38, 0.15), "PCA\n(k = 3)", box_color="#eaeaea")
box(ax_header, (0.74, 0.15), "3D embedding\n(PC1, PC2, PC3)", box_color="#eaeaea")

# Arrows
ax_header.annotate("", xy=(0.36, 0.42), xytext=(0.26, 0.42),
                   arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555"))
ax_header.annotate("", xy=(0.72, 0.42), xytext=(0.62, 0.42),
                   arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555"))

# ---- (B) 3D scatter of top-3 PCs ----
ax3d = fig.add_axes([0.06, 0.08, 0.62, 0.68], projection="3d")

# Separate classes just for visual separation (you can drop y if you want one color)
Z0 = Z[y == 0]
Z1 = Z[y == 1]

# Scatter (default Matplotlib colors for a neutral palette)
ax3d.scatter(Z0[:, 0], Z0[:, 1], Z0[:, 2], s=10, alpha=0.75, depthshade=False, label="Cluster A")
ax3d.scatter(Z1[:, 0], Z1[:, 1], Z1[:, 2], s=10, alpha=0.75, depthshade=False, label="Cluster B")

# Axes labels w/ variance explained
ax3d.set_xlabel(f"PC1 ({evr3[0]*100:.1f}%)", labelpad=6)
ax3d.set_ylabel(f"PC2 ({evr3[1]*100:.1f}%)", labelpad=6)
ax3d.set_zlabel(f"PC3 ({evr3[2]*100:.1f}%)", labelpad=6)

# Tidy 3D look
ax3d.set_title(TITLE, pad=10, fontsize=12)
ax3d.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True)

# Light grid feel
for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
    for t in axis.get_ticklines():
        t.set_markersize(4)

# ---- (C) Side panel with interpretation text ----
ax_side = fig.add_axes([0.71, 0.12, 0.24, 0.62])
ax_side.axis("off")

text = (
    r"\textbf{Representation (schematic)}" "\n"
    r"$X \in \mathbb{R}^{N \times D}$ centered $\rightarrow$ SVD/PCA" "\n"
    r"$Z = X W_{1:3},\;\; W_{1:3} \in \mathbb{R}^{D \times 3}$" "\n"
    r"\hspace{0.35cm} (columns are PC1–PC3)" "\n\n"
    r"\textbf{What you see:}" "\n"
    r"$\bullet$ The dataset compressed to 3D (PC1–PC3)" "\n"
    r"$\bullet$ Axes labeled by variance explained" "\n"
    r"$\bullet$ Two visible clusters for illustration"
)
# Render as plain text (no MathJax here; it's just a figure). Keep it readable.
ax_side.text(0, 1, text, va="top", fontsize=10, linespacing=1.35)

# -----------------------------
# Save outputs
# -----------------------------
os.makedirs(OUTDIR, exist_ok=True)
png_path = os.path.join(OUTDIR, f"{OUTNAME}.png")
pdf_path = os.path.join(OUTDIR, f"{OUTNAME}.pdf")
fig.savefig(png_path, dpi=300)
fig.savefig(pdf_path)
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

plt.show()
