#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Layer Selection — Side-by-side per trait (O C E A N), layers 7..25.
Dark, bold, paper-ready with huge fonts.

Input JSONL lines should include:
{
  "trait": "openness",
  "selection": {
    "mode": "hybrid",
    "verified_layer": 11,
    "dynamic_layer": 15,
    "dynamic_delta_l2": 355.22
  }
}

Usage:
  python plot_layer_summary_row.py \
    --input runs/hybrid_logs.jsonl \
    --out figs/hybrid_layers_row_llama3_8b \
    --title "Hybrid Layer Selection (Llama-3-8B, α=4)" \
    --layer-min 7 --layer-max 25 \
    --dpi 300
"""

import json
import argparse
import os
from collections import defaultdict, Counter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

TRAIT_ORDER = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
TRAIT_LABEL = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
}
TRAIT_SHORT = {"openness":"O","conscientiousness":"C","extraversion":"E","agreeableness":"A","neuroticism":"N"}

def load_jsonl(paths):
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rows.append(json.loads(s))
                except Exception:
                    pass
    return rows

def collect_by_trait(rows, layer_min, layer_max):
    data = {t: {"verified": None, "dyn_layers": []} for t in TRAIT_ORDER}
    for r in rows:
        t = str(r.get("trait","")).strip().lower()
        if t not in data:
            continue
        sel = r.get("selection", {}) or {}
        vL = sel.get("verified_layer", None)
        dL = sel.get("dynamic_layer", None)

        if vL is not None and data[t]["verified"] is None:
            data[t]["verified"] = int(vL)

        if dL is not None:
            dL = int(dL)
            if layer_min <= dL <= layer_max:
                data[t]["dyn_layers"].append(dL)
    return data

def plot_row(data, out_prefix, title, layer_min, layer_max, dpi=300):
    # ---- Big, bold, dark theme ----
    mpl.rcParams.update({
        "font.size": 80,        # base
        "axes.titlesize": 90,   # figure title
        "axes.labelsize": 80,
        "xtick.labelsize": 80,
        "ytick.labelsize": 80,
        "legend.fontsize": 80,
        "axes.edgecolor": "black",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })

    # One row, five columns
    fig, axes = plt.subplots(1, 5, figsize=(95, 20), dpi=dpi, constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Colors
    bar_color = "#1f4e79"     # dark steel blue
    bar_edge  = "#0c2233"
    line_color = "#d62728"    # bold red for verified line
    star_color = "#ff7f0e"    # orange for mean

    bins = list(range(layer_min, layer_max+1))
    for i, trait in enumerate(TRAIT_ORDER):
        ax = axes[i]
        bundle = data[trait]
        verified = bundle["verified"]
        dyn = bundle["dyn_layers"]
        counts = Counter(dyn)
        heights = [counts.get(x, 0) for x in bins]

        # If empty, keep the frame but show text
        if sum(heights) == 0:
            ax.set_title(f"{TRAIT_LABEL[trait]}", pad=24, fontweight="bold")
            ax.set_xlim(layer_min-0.5, layer_max+0.5)
            ax.set_ylim(0, 1)
            ax.set_xticks(range(layer_min, layer_max+1, 2))
            ax.set_xlabel("Layers")
            if i == 0:
                ax.set_ylabel("Frequency")
            ax.text(0.5, 0.5, "No dynamic picks in range",
                    ha="center", va="center", transform=ax.transAxes, fontsize=42, color="#444")
            continue

        # Bars
        ax.bar(bins, heights, width=0.8, color=bar_color, edgecolor=bar_edge, linewidth=2)

        # Verified layer (dashed)
        if verified is not None and (layer_min <= verified <= layer_max):
            ax.axvline(verified, color=line_color, linestyle=(0, (10, 10)), linewidth=10, alpha=0.999,
                       label="Verified (offline best)")

        # Star for mean dynamic layer
        mean_dyn = np.mean(dyn) if len(dyn) else None
        if mean_dyn is not None and (layer_min <= mean_dyn <= layer_max):
            ymax = max(heights) if heights else 1
            ax.plot([mean_dyn], [ymax*1.05], marker="*", markersize=70,
                    color=star_color, markeredgecolor="black", markeredgewidth=5,
                    label="Mean dynamic")

        # Cosmetics
        ax.set_title(f"{TRAIT_LABEL[trait]}", pad=24, fontweight="bold")
        ax.set_xlim(layer_min-0.5, layer_max+0.5)
        # Nice y-limit with headroom for star
        ymax = max(heights) if heights else 1
        ax.set_ylim(0, ymax * 1.2)
        ax.set_xticks(range(layer_min, layer_max+1, 2))
        ax.set_xlabel("Layer", fontsize = 90)
        if i == 0:
            ax.set_ylabel("Frequency", fontsize = 90)

        # Bold axes
        for spine in ax.spines.values():
            spine.set_linewidth(3)

        ax.grid(axis="y", linestyle=":", linewidth=4, alpha=0.4)

    # ---------- SINGLE, FIGURE-LEVEL LEGEND (BOTTOM) ----------
    # ---------- SINGLE, FIGURE-LEVEL LEGEND (BOTTOM) ----------
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_handles = [
        Patch(facecolor=bar_color, edgecolor=bar_edge, label="Dynamic picks (count)"),
        Line2D([0], [0], color=line_color, lw=10, ls=(0, (10, 10)), label="Verified (offline best)"),
        Line2D([0], [0], marker="*", markersize=70, markeredgecolor="black",
               markeredgewidth=5, linestyle="None", color=star_color, label="Mean dynamic"),
    ]
    legend_labels = ["Dynamic picks (count)", "Verified (offline best)", "Mean dynamic"]

    fig.legend(
        handles=legend_handles,          # <-- IMPORTANT: pass as handles=
        labels=legend_labels,            # optional, but keeps ordering explicit
        loc="lower center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=True,
        fontsize = 80
    )
    # ----------------------------------------------------------

    # ----------------------------------------------------------

    # Big shared title
    fig.suptitle(title, y=1.04, fontweight="bold")

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    png = out_prefix + ".png"
    svg = out_prefix + ".svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, dpi=dpi, bbox_inches="tight")
    print(f"[OK] Saved: {png}\n[OK] Saved: {svg}")


def main():
    ap = argparse.ArgumentParser(description="Hybrid layer selection summary (row of 5 traits), layers 7..25.")
    ap.add_argument("--input", type=str, nargs="+", required=True, help="JSONL file(s) with hybrid selection logs.")
    ap.add_argument("--out", type=str, required=True, help="Output prefix (no extension).")
    ap.add_argument("--title", type=str, default="", help="Figure title.")
    ap.add_argument("--layer-min", type=int, default=7)
    ap.add_argument("--layer-max", type=int, default=25)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    data = collect_by_trait(rows, args.layer_min, args.layer_max)
    plot_row(data, args.out, args.title, args.layer_min, args.layer_max, dpi=args.dpi)

if __name__ == "__main__":
    main()
