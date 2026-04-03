#!/usr/bin/env python3
"""
Layer-wise stacked bar plots for OCEAN traits.

Input JSON schema (per trait -> per layer):
{
  "openness": {
    "7":  {"delta_l2": ..., "first_kl": ..., "flip": ..., "combined": ...},
    "8":  {...},
    ...
  },
  "conscientiousness": { ... },
  "extraversion": { ... },
  "agreeableness": { ... },
  "neuroticism": { ... }
}

Outputs:
- layer_select_figure/ocean_layers_stacked_all.png (1x5 panels)
- layer_select_figure/<trait>_layers_stacked.png (one per trait)
(PDF versions too)

Usage:
    python layer_stacked_plot.py steering_metrics.json
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =========================
# ====== CONFIG AREA ======
# =========================
# Figure / style
DPI = 300
FONT_SIZE = 14
TITLE_SIZE = 18
LABEL_SIZE = 14
TICK_SIZE = 12

# Colors (customize)
COLOR_DELTA_L2 = "#264653"   # stack part 1
COLOR_FIRST_KL = "#2a9d8f"   # stack part 2
COLOR_FLIP     = "#fb8b24"   # stack part 3
COLOR_COMBINED_LINE = "#fe4a49"  # optional overlay line

# Bar look
BAR_WIDTH = 0.7
EDGE_COLOR = "white"
EDGE_WIDTH = 0.6

# Combined handling
ENFORCE_SUM_TO_COMBINED = False  # rescale components to match 'combined'
SHOW_COMBINED_LINE = True       # overlay combined as a thin line

# Layout
BIG_FIG_SIZE = (18, 3.8)   # 1x5 panels (width, height) in inches
IND_FIG_SIZE = (7.0, 4.5)  # per-trait figure size

# Y-axis padding (add headroom for neat look)
Y_TOP_PAD_FACTOR = 1.08  # 8% headroom

# Output dir
OUT_DIR = Path("layer_select_figure")
# =========================


OCEAN_ORDER = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
OCEAN_TITLES = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
}


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def extract_trait_arrays(trait_dict):
    """
    Convert {layer_str: {delta_l2, first_kl, flip, combined}} into sorted arrays.
    Returns: layers(list[int]), delta_l2(np.array), first_kl(np.array), flip(np.array), combined(np.array)
    """
    # Sort layers numerically (keys are strings)
    layer_keys = sorted(trait_dict.keys(), key=lambda s: int(s))
    layers = [int(k) for k in layer_keys]

    delta_l2   = np.array([trait_dict[k]["delta_l2"]  for k in layer_keys], dtype=float)
    first_kl   = np.array([trait_dict[k]["first_kl"]  for k in layer_keys], dtype=float)
    flip       = np.array([trait_dict[k]["flip"]      for k in layer_keys], dtype=float)
    combined   = np.array([trait_dict[k]["combined"]  for k in layer_keys], dtype=float)

    return layers, delta_l2, first_kl, flip, combined


def rescale_to_match_combined(a, b, c, combined, eps=1e-12):
    """
    Rescale components a,b,c so that (a+b+c) == combined elementwise.
    Keeps proportions among (a,b,c) per layer. If a+b+c is ~0, splits combined equally.
    """
    stacked = a + b + c
    out_a = np.empty_like(a)
    out_b = np.empty_like(b)
    out_c = np.empty_like(c)

    zero_mask = stacked < eps
    nonzero_mask = ~zero_mask

    # Nonzero: scale proportionally
    scale = np.zeros_like(stacked)
    scale[nonzero_mask] = (combined[nonzero_mask] / stacked[nonzero_mask])

    out_a[nonzero_mask] = a[nonzero_mask] * scale[nonzero_mask]
    out_b[nonzero_mask] = b[nonzero_mask] * scale[nonzero_mask]
    out_c[nonzero_mask] = c[nonzero_mask] * scale[nonzero_mask]

    # Zero: split equally
    out_a[zero_mask] = combined[zero_mask] / 3.0
    out_b[zero_mask] = combined[zero_mask] / 3.0
    out_c[zero_mask] = combined[zero_mask] / 3.0

    return out_a, out_b, out_c


def plot_trait_stacked(ax, trait_name, layers, a, b, c, combined):
    """
    Draw stacked bars for a,b,c across layers on ax.
    Optionally overlay combined line.
    """
    x = np.arange(len(layers))

    # Stack: delta_l2 (a), first_kl (b), flip (c)
    p1 = ax.bar(
        x, a, width=BAR_WIDTH, color=COLOR_DELTA_L2, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, label="ΔL2"
    )
    p2 = ax.bar(
        x, b, width=BAR_WIDTH, bottom=a, color=COLOR_FIRST_KL, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, label="First KL"
    )
    p3 = ax.bar(
        x, c, width=BAR_WIDTH, bottom=a + b, color=COLOR_FLIP, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, label="Flip"
    )

    if SHOW_COMBINED_LINE:
        ax.plot(x, combined, color=COLOR_COMBINED_LINE, linewidth=1.8, marker="o", markersize=3, label="Combined")

    # Cosmetics
    ax.set_title(OCEAN_TITLES.get(trait_name, trait_name.capitalize()), fontsize=TITLE_SIZE, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=TICK_SIZE)
    ax.set_xlabel("Layer", fontsize=LABEL_SIZE)

    # Set y-limit with padding
    y_max = max((a + b + c).max(), combined.max() if combined.size else 0.0)
    ax.set_ylim(0, y_max * Y_TOP_PAD_FACTOR if y_max > 0 else 1.0)

    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.grid(axis="y", alpha=0.25, linewidth=1)

    return (p1, p2, p3)


def save_individual(trait_key, layers, a, b, c, combined):
    fig, ax = plt.subplots(figsize=IND_FIG_SIZE, dpi=DPI)
    handles = plot_trait_stacked(ax, trait_key, layers, a, b, c, combined)
    # Only add legend once per individual figure
    leg_items = [("ΔL2", COLOR_DELTA_L2), ("First KL", COLOR_FIRST_KL), ("Flip", COLOR_FLIP)]
    if SHOW_COMBINED_LINE:
        ax.legend(loc="upper right", fontsize=FONT_SIZE - 2)
    ax.set_ylabel("Score", fontsize=LABEL_SIZE)
    fig.tight_layout()
    out_png = OUT_DIR / f"{trait_key}_layers_stacked.png"
    out_pdf = OUT_DIR / f"{trait_key}_layers_stacked.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Stacked layer plots for OCEAN traits from JSON.")
    parser.add_argument("json_path", type=str, help="Path to steering_metrics JSON file")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    assert json_path.exists(), f"File not found: {json_path}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_json(json_path)

    # Global font sizing
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": FONT_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "savefig.bbox": "tight",
    })

    # ---- Big 1x5 figure ----
    fig, axes = plt.subplots(1, 5, figsize=BIG_FIG_SIZE, dpi=DPI, sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    # Build a legend only once (from the first panel)
    legend_handles = None

    for idx, trait_key in enumerate(OCEAN_ORDER):
        if trait_key not in data:
            # If missing, skip gracefully (draw empty panel)
            axes[idx].set_title(OCEAN_TITLES.get(trait_key, trait_key.capitalize()), fontsize=TITLE_SIZE, pad=8)
            axes[idx].text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONT_SIZE)
            axes[idx].axis("off")
            continue

        layers, delta_l2, first_kl, flip, combined = extract_trait_arrays(data[trait_key])

        if ENFORCE_SUM_TO_COMBINED:
            delta_l2, first_kl, flip = rescale_to_match_combined(delta_l2, first_kl, flip, combined)

        handles = plot_trait_stacked(axes[idx], trait_key, layers, delta_l2, first_kl, flip, combined)

        # Capture legend handles from first real trait
        if legend_handles is None:
            legend_handles = handles

        # Save individual figure for this trait
        save_individual(trait_key, layers, delta_l2, first_kl, flip, combined)

        # Only the left-most panel gets a y-axis label (cleaner look)
        if idx == 0:
            axes[idx].set_ylabel("Score", fontsize=LABEL_SIZE)

    # Common legend for stacks + optional combined line
    # Build legend entries manually to keep order nice
    legend_elems = []
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elems.append(Patch(facecolor=COLOR_DELTA_L2, edgecolor=EDGE_COLOR, label="ΔL2"))
    legend_elems.append(Patch(facecolor=COLOR_FIRST_KL, edgecolor=EDGE_COLOR, label="First KL"))
    legend_elems.append(Patch(facecolor=COLOR_FLIP, edgecolor=EDGE_COLOR, label="Flip"))
    if SHOW_COMBINED_LINE:
        legend_elems.append(Line2D([0], [0], color=COLOR_COMBINED_LINE, lw=2, marker="o", markersize=4, label="Combined"))

    fig.legend(
        handles=legend_elems,
        loc="upper center",
        ncol=4,
        fontsize=FONT_SIZE,
        frameon=True,
        bbox_to_anchor=(0.5, 1.12)
    )

    fig.tight_layout()
    out_big_png = OUT_DIR / "ocean_layers_stacked_all.png"
    out_big_pdf = OUT_DIR / "ocean_layers_stacked_all.pdf"
    fig.savefig(out_big_png, bbox_inches="tight")
    fig.savefig(out_big_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved big figure: {out_big_png}")
    print(f"Saved per-trait figures into: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
