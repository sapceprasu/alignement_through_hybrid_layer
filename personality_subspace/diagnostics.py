
from typing import Dict, List, Tuple, Iterable
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from .main import load_steerer
from .layer_selector import select_layers_for_prompt, SteerConfigPatch, delta_logits_norms_for_prompt

# ---------------------------
# Low-level helpers
# ---------------------------
EPS = 1e-9

def _format_for_chat(steerer, text: str, system: str = None) -> str:
    return steerer._format_prompt(text, use_chat_template=True, system=system)

def _ensure_frac_rms_if_needed(steerer, formatted_text: str):
    if getattr(steerer, "alpha_mode", "abs") == "frac":
        steerer._measure_layer_rms(formatted_text)

@torch.no_grad()
def _next_token_probs(steerer, formatted_text: str) -> torch.Tensor:
    """
    Deterministic next-token distribution at the current last token.
    Matches your evaluation path (no sampling; just logits -> softmax).
    """
    enc = steerer.tok(formatted_text, return_tensors="pt").to(steerer.device)
    out = steerer.model(**enc)
    logits = out.logits[:, -1, :].float()
    return torch.softmax(logits, dim=-1).squeeze(0)  # [V]

def entropy(p: torch.Tensor) -> float:
    p = p.clamp_min(EPS)
    return float(-(p * p.log()).sum().item())

def kl_div(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.clamp_min(EPS); q = q.clamp_min(EPS)
    return float((p * (p.log() - q.log())).sum().item())

def js_div(p: torch.Tensor, q: torch.Tensor) -> float:
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

def l2_dist(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(torch.norm(p - q, p=2).item())

def topk_cover_indices(p: torch.Tensor, target_mass: float = 0.995) -> torch.Tensor:
    """
    Return indices of smallest top-K set whose cumulative prob >= target_mass.
    """
    vals, idx = torch.sort(p, descending=True)
    csum = torch.cumsum(vals, dim=0)
    k = int((csum >= min(0.9999, target_mass)).nonzero(as_tuple=False)[0].item()) + 1
    top_idx = idx[:k]
    return torch.sort(top_idx).values  # return sorted indices

def topk_kl(p: torch.Tensor, q: torch.Tensor, target_mass: float = 0.995) -> float:
    idx = topk_cover_indices(p, target_mass=target_mass)
    p_s = p[idx].clamp_min(EPS); q_s = q[idx].clamp_min(EPS)
    # renormalize within the subset (optional; here we do not renorm, we compute partial KL)
    return float((p_s * (p_s.log() - q_s.log())).sum().item())

def argmax_flip(p0: torch.Tensor, p1: torch.Tensor) -> int:
    return int(torch.argmax(p0).item() != torch.argmax(p1).item())

# ---------------------------
# Data containers
# ---------------------------
@dataclass
class SelectionInfo:
    layers: List[int]
    weights: List[float]
    norms: Dict[int, float]  # per-layer Δlogits norms averaged over probes

@dataclass
class DiagnosticRow:
    prompt: str
    trait: str
    alpha: float
    system: str
    entropy_base: float
    kl: float
    js: float
    topk_kl: float
    delta_l2: float
    flip: int
    sel_layers: List[int]
    sel_weights: List[float]
    # optional metadata
    verified_layers: List[int]

# ---------------------------
# Core diagnostic for one (prompt, alpha)
# ---------------------------
@torch.no_grad()
def diagnose_single(
    steerer,
    prompt_text: str,
    trait: str,
    alpha: float,
    system_line: str = None,
    topk_mass: float = 0.995,
    k_runtime: int = 2,
    prior_boost: float = 0.15,
    temperature: float = 0.50,
    max_layers: int = 2,
    min_weight: float = 0.25,
) -> Tuple[DiagnosticRow, SelectionInfo]:
    """
    1) Format prompt
    2) Get p0 (baseline next-token distribution)
    3) Hybrid-select layers for THIS prompt (does not change generation yet)
    4) Under those layers/weights, register hooks with alpha and get p1
    5) Compute diagnostics (H, KL, JS, Top-K KL, ΔL2, flip)
    """
    formatted = _format_for_chat(steerer, prompt_text, system=system_line)
    _ensure_frac_rms_if_needed(steerer, formatted)

    # Baseline p0
    p0 = _next_token_probs(steerer, formatted)

    # Hybrid selection (this *internally* computes per-layer Δlogits norms)
    layers, weights, norms = select_layers_for_prompt(
        steerer, prompt_text, trait, intensity=abs(alpha),
        system=system_line, k_runtime=k_runtime, prior_boost=prior_boost,
        temperature=temperature, max_layers=max_layers, min_weight=min_weight
    )

    # Verified (static) layers for reference
    vmap = getattr(steerer, "_trait_layers", {})
    verified = vmap.get(trait.lower(), [])
    if isinstance(verified, int): verified = [int(verified)]
    verified = [int(x) for x in verified] if isinstance(verified, list) else []

    # Steered p1 under the selected mixture
    with SteerConfigPatch(steerer, layers, weights):
        steerer._register(trait, alpha)
        try:
            p1 = _next_token_probs(steerer, formatted)
        finally:
            steerer._clear()

    row = DiagnosticRow(
        prompt=prompt_text, trait=trait, alpha=float(alpha), system=system_line or "",
        entropy_base=entropy(p0),
        kl=kl_div(p0, p1),
        js=js_div(p0, p1),
        topk_kl=topk_kl(p0, p1, target_mass=topk_mass),
        delta_l2=l2_dist(p0, p1),
        flip=argmax_flip(p0, p1),
        sel_layers=list(layers),
        sel_weights=[float(w) for w in weights],
        verified_layers=verified
    )
    sel = SelectionInfo(layers=list(layers), weights=list(weights), norms={int(k): float(v) for k, v in norms.items()})
    return row, sel

# ---------------------------
# Alpha sweep for a single prompt
# ---------------------------
def sweep_alphas(
    steerer,
    prompt_text: str,
    trait: str,
    alphas: Iterable[float],
    system_line: str = None,
    out_dir: Path = None,
    tag: str = "diag",
    **kwargs
) -> Dict:
    """
    Run diagnose_single for each alpha. Save CSV, JSONL, and plots if out_dir provided.
    Returns a dict with 'rows' and 'selections' (per-alpha).
    """
    rows: List[DiagnosticRow] = []
    sels: Dict[float, SelectionInfo] = {}

    for a in alphas:
        row, sel = diagnose_single(steerer, prompt_text, trait, a, system_line=system_line, **kwargs)
        rows.append(row); sels[float(a)] = sel

    # Save
    result = {
        "rows": [asdict(r) for r in rows],
        "selections": {str(a): {"layers": s.layers, "weights": s.weights, "norms": s.norms} for a, s in sels.items()}
    }
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        # CSV
        import csv
        csv_path = out_dir / f"{tag}__{trait}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(result["rows"][0].keys()))
            w.writeheader()
            for r in result["rows"]:
                w.writerow(r)
        # JSONL
        jsonl_path = out_dir / f"{tag}__{trait}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in result["rows"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # Selections JSON
        sel_path = out_dir / f"{tag}__{trait}.selections.json"
        with open(sel_path, "w", encoding="utf-8") as f:
            json.dump(result["selections"], f, ensure_ascii=False, indent=2)

        # Plots
        _plot_curves(rows, out_dir / f"{tag}__{trait}__curves.png")
        _plot_selection(rows, sels, out_dir / f"{tag}__{trait}__selection.png")
        _plot_norms_heatmap(sels, out_dir / f"{tag}__{trait}__norms.png")

    return result

# ---------------------------
# Plotting
# ---------------------------
def _plot_curves(rows: List[DiagnosticRow], png_path: Path):
    rows = sorted(rows, key=lambda r: r["alpha"] if isinstance(r, dict) else r.alpha)
    alphas = [r["alpha"] if isinstance(r, dict) else r.alpha for r in rows]
    kl = [r["kl"] if isinstance(r, dict) else r.kl for r in rows]
    js = [r["js"] if isinstance(r, dict) else r.js for r in rows]
    tkk = [r["topk_kl"] if isinstance(r, dict) else r.topk_kl for r in rows]
    d2 = [r["delta_l2"] if isinstance(r, dict) else r.delta_l2 for r in rows]
    flips = [r["flip"] if isinstance(r, dict) else r.flip for r in rows]
    H = [r["entropy_base"] if isinstance(r, dict) else r.entropy_base for r in rows]

    plt.figure(figsize=(9,6))
    plt.plot(alphas, kl, marker='o', label="KL(p0||p1)")
    plt.plot(alphas, js, marker='o', label="JS")
    plt.plot(alphas, tkk, marker='o', label="Top-K KL (99.5%)")
    plt.plot(alphas, d2, marker='o', label="ΔL2")
    plt.scatter(alphas, flips, label="Flip(0/1)", s=60)
    plt.twinx()
    plt.plot(alphas, H, marker='x', linestyle='--', label="Entropy(p0)")
    plt.title("Diagnostics vs α")
    plt.xlabel("α")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

def _plot_selection(rows: List[DiagnosticRow], sels: Dict[float, SelectionInfo], png_path: Path):
    rows = sorted(rows, key=lambda r: r["alpha"] if isinstance(r, dict) else r.alpha)
    alphas = [r["alpha"] if isinstance(r, dict) else r.alpha for r in rows]
    main_layers = [sels[float(a)].layers[0] if sels[float(a)].layers else None for a in alphas]
    sec_layers  = [sels[float(a)].layers[1] if len(sels[float(a)].layers) > 1 else None for a in alphas]
    w1 = [sels[float(a)].weights[0] if sels[float(a)].weights else 0.0 for a in alphas]
    w2 = [sels[float(a)].weights[1] if len(sels[float(a)].weights) > 1 else 0.0 for a in alphas]

    plt.figure(figsize=(9,6))
    plt.plot(alphas, main_layers, marker='o', label="Primary layer")
    plt.plot(alphas, sec_layers, marker='o', label="Secondary layer")
    plt.title("Selected layers vs α (Hybrid)")
    plt.xlabel("α"); plt.ylabel("Layer index")
    plt.legend(); plt.tight_layout()
    plt.savefig(png_path, dpi=160); plt.close()

    # Also save weights as a companion chart
    png_w = png_path.with_name(png_path.stem + "__weights.png")
    plt.figure(figsize=(9,6))
    plt.plot(alphas, w1, marker='o', label="w(primary)")
    plt.plot(alphas, w2, marker='o', label="w(secondary)")
    plt.title("Selected layer weights vs α")
    plt.xlabel("α"); plt.ylabel("Weight")
    plt.legend(); plt.tight_layout()
    plt.savefig(png_w, dpi=160); plt.close()

def _plot_norms_heatmap(sels: Dict[float, SelectionInfo], png_path: Path):
    # union of layers present in norms
    all_layers = sorted({L for s in sels.values() for L in s.norms.keys()})
    alphas = sorted(sels.keys())
    if not all_layers or not alphas:
        return
    M = np.zeros((len(all_layers), len(alphas)), dtype=float)
    for j, a in enumerate(alphas):
        for i, L in enumerate(all_layers):
            M[i, j] = sels[a].norms.get(L, 0.0)
    plt.figure(figsize=(10, max(3, 0.3*len(all_layers))))
    plt.imshow(M, aspect='auto')
    plt.colorbar(label="Δlogits L2 norm")
    plt.yticks(range(len(all_layers)), all_layers)
    plt.xticks(range(len(alphas)), [str(a) for a in alphas], rotation=45)
    plt.xlabel("α"); plt.ylabel("Layer")
    plt.title("Per-layer Δlogits norms (averaged over probes)")
    plt.tight_layout(); plt.savefig(png_path, dpi=160); plt.close()
