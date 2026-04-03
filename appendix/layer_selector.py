#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid layer figure (clean): one row of 5 small-multiples.
- Reads JSONL logs (quick_steer outputs) to count dynamic layers per trait.
- Plots per-trait histogram of dynamic layers (bars), overlaying verified/best layer as a big diamond.
- Focuses on layer range [Lmin, Lmax] (default 7..25).
- Big fonts, dark option, saves PNG+SVG.

Usage:
  python plot_hybrid_layers_clean.py --input runs/hybrid.jsonl \
      --out figs_hybrid/hybrid_clean --lmin 7 --lmax 25 --dark
"""

import os, json, argparse, numpy as np
import matplotlib.pyplot as plt

TRAITS = ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]
T_COL = {
    "openness":"#1f77b4",
    "conscientiousness":"#ff7f0e",
    "extraversion":"#2ca02c",
    "agreeableness":"#d62728",
    "neuroticism":"#9467bd",
}

TITLE = 68
AX = 58
TICK = 46
LEG = 40

def load_jsonl(paths):
    rows=[]
    for p in paths:
        with open(p,"r",encoding="utf-8") as f:
            for line in f:
                s=line.strip()
                if not s: continue
                try: rows.append(json.loads(s))
                except: pass
    return rows

def collect_counts(rows, lmin, lmax):
    # per trait: dynamic layer counts + one verified layer (if present)
    out={t: {"counts":{i:0 for i in range(lmin,lmax+1)}, "verified":None} for t in TRAITS}
    for r in rows:
        t=str(r.get("trait","")).lower()
        if t not in out: continue
        sel=r.get("selection",{}) or {}
        if sel.get("verified_layer") is not None and out[t]["verified"] is None:
            v=int(sel["verified_layer"])
            if lmin<=v<=lmax: out[t]["verified"]=v
        d=sel.get("dynamic_layer",None)
        if d is not None:
            d=int(d)
            if lmin<=d<=lmax:
                out[t]["counts"][d]+=1
    return out

def bold_axes(ax, dark=False):
    for s in ["left","right","bottom","top"]:
        ax.spines[s].set_linewidth(3.5)
        ax.spines[s].set_color("white" if dark else "black")
    ax.tick_params(axis="both", labelsize=TICK, width=3.5, length=10, colors=("white" if dark else "black"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lmin", type=int, default=7)
    ap.add_argument("--lmax", type=int, default=25)
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    args=ap.parse_args()

    rows=load_jsonl(args.input)
    dat=collect_counts(rows, args.lmin, args.lmax)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    bg = "#0e0f13" if args.dark else "white"
    fg = "white" if args.dark else "black"

    # one row of 5 subplots
    fig, axes = plt.subplots(1, 5, figsize=(28, 7), dpi=args.dpi, sharey=True)
    fig.patch.set_facecolor(bg)
    for ax in axes: ax.set_facecolor(bg)

    for i, t in enumerate(TRAITS):
        ax=axes[i]
        bold_axes(ax, dark=args.dark)
        xs=sorted(dat[t]["counts"].keys())
        ys=[dat[t]["counts"][x] for x in xs]
        ax.bar(xs, ys, width=0.8, color=T_COL[t], alpha=0.9, edgecolor="white" if args.dark else "black", linewidth=1.5)
        v=dat[t]["verified"]
        if v is not None:
            ax.scatter([v],[max(ys)*1.05 if ys else 0.5], s=1200, marker="D",
                       color="#ffc107", edgecolor="black", linewidth=2.5, zorder=5, label="Verified")
            ax.text(v, (max(ys)*1.12 if ys else 0.7), "verified", ha="center", va="bottom",
                    fontsize=LEG, color=fg, weight="bold")

        ax.set_xlim(args.lmin-0.5, args.lmax+0.5)
        ax.set_xticks(xs[::max(1,(args.lmax-args.lmin)//9 or 1)])
        ax.set_xlabel("Layer", fontsize=AX, color=fg, weight="bold")
        if i==0:
            ax.set_ylabel("Dynamic count", fontsize=AX, color=fg, weight="bold")
        ax.set_title(t.title(), fontsize=TITLE, color=fg, pad=14, weight="bold")
        ax.grid(axis="y", alpha=0.2, color="white" if args.dark else "black")

    plt.tight_layout()
    fig.savefig(args.out + ".png", bbox_inches="tight")
    fig.savefig(args.out + ".svg", bbox_inches="tight")
    print("[OK] Saved:", args.out + ".png")

if __name__=="__main__":
    main()
