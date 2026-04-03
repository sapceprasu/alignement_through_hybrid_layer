
from typing import Dict, List, Tuple
import os, json, pickle
import numpy as np
import torch
from dataclasses import dataclass
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from .config import Config
from .layer_search import _p0p1_for_layer_batched, _format_chat

# -----------------------------
# Representation metrics (activations only)
# -----------------------------

def _delta_mean_norm(Xh: np.ndarray, Xl: np.ndarray) -> float:
    if Xh.size == 0 or Xl.size == 0:
        return 0.0
    mh = Xh.mean(axis=0)
    ml = Xl.mean(axis=0)
    d  = mh - ml
    return float(np.linalg.norm(d))


def _linear_probe_acc(Xh: np.ndarray, Xl: np.ndarray, seed: int = 42, cap_per_class: int = 2000) -> float:
    if Xh.size == 0 or Xl.size == 0:
        return 0.0
    # Optional cap for speed/stability
    def _cap(X):
        if cap_per_class is None or X.shape[0] <= cap_per_class:
            return X
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], cap_per_class, replace=False)
        return X[idx]

    Xh_ = _cap(Xh)
    Xl_ = _cap(Xl)
    X   = np.vstack([Xh_, Xl_])
    y   = np.array([1]*len(Xh_) + [0]*len(Xl_))

    # Stratified split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    # Simple linear probe
    clf = LogisticRegression(solver="liblinear", max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    return float(accuracy_score(yte, yhat))


def compute_representation_metrics(cfg: Config, activs: Dict[int, Dict[str, np.ndarray]]) -> Dict:
    """
    Returns dict:
      {
        trait: {
          layer: {"delta_norm": float, "probe_acc": float}
        }
      }
    and writes a CSV-like JSON summary.
    """
    results: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)

    for trait in cfg.trait_mapping.values():
        hi_key, lo_key = f"{trait}_high", f"{trait}_low"
        for L in cfg.layer_range:
            if L not in activs:
                continue
            Xh = activs[L].get(hi_key, None)
            Xl = activs[L].get(lo_key, None)
            if Xh is None or Xl is None:
                continue
            dn = _delta_mean_norm(Xh, Xl)
            acc = _linear_probe_acc(Xh, Xl, seed=cfg.seed)
            results[trait][int(L)] = {"delta_norm": dn, "probe_acc": acc}

    # Save
    out_path = os.path.join(cfg.results_dir, "representation_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also flatten to CSV-like rows
    flat = []
    for trait, layer_dict in results.items():
        for L, vals in layer_dict.items():
            flat.append({"trait": trait, "layer": L, **vals})
    with open(os.path.join(cfg.results_dir, "representation_metrics.csv"), "w") as f:
        f.write("trait,layer,delta_norm,probe_acc\n")
        for r in flat:
            f.write(f"{r['trait']},{r['layer']},{r['delta_norm']:.6f},{r['probe_acc']:.6f}\n")

    return results


# -----------------------------
# Steering metrics (reuses your helpers)
# -----------------------------
def compute_steering_metrics(cfg: Config, steerer) -> Dict:
    """
    Compute ΔL2, first-token KL, and flip-rate per trait×layer using batched first-token calls.
    Reuses layer_search._p0p1_for_layer_batched; no dependency on single-text helpers.
    Writes JSON + CSV.
    """
    # Align steerer runtime with intended evaluation settings
    steerer.steer_mode = getattr(cfg.layer_search, "eval_steer_mode", "weighted")
    steerer.injection_point = getattr(cfg.layer_search, "eval_injection_point", "post")

    # Sanity: device alignment
    assert next(steerer.model.parameters()).device.type == steerer.device.type, "model/device mismatch"
    tmp = steerer.tok("hi", return_tensors="pt").to(steerer.device)
    assert tmp.input_ids.device.type == steerer.device.type, "encodings not on device"

    # Format probes once
    probe_texts = [_format_chat(steerer, p) for p in cfg.layer_search.probe_prompts]
    alpha = float(cfg.layer_search.alpha_probe)
    w = cfg.layer_search.metric_weights

    out: Dict[str, Dict[int, Dict[str, float]]] = {}

    for trait in cfg.trait_mapping.values():
        out[trait] = {}
        for L in cfg.layer_range:
            pairs = _p0p1_for_layer_batched(steerer, probe_texts, trait, L, alpha)
            if not pairs:
                out[trait][int(L)] = {"delta_l2": 0.0, "first_kl": 0.0, "flip": 0.0, "combined": 0.0}
                continue

            # Δlogits L2
            d_l2 = float(np.mean([torch.norm(p1 - p0, p=2).item() for p0, p1 in pairs]))

            # First-token KL(p0 || p1)
            eps = 1e-9
            fkl = float(np.mean([
                torch.sum(
                    p0.clamp_min(eps) * (torch.log(p0.clamp_min(eps)) - torch.log(p1.clamp_min(eps)))
                ).item() for p0, p1 in pairs
            ]))

            # Flip-rate (argmax change)
            flip = float(np.mean([
                int(torch.argmax(p0).item() != torch.argmax(p1).item())
                for p0, p1 in pairs
            ]))

            combined = float(np.nan_to_num(
                w["delta_l2"] * d_l2 + w["first_kl"] * fkl + w["flip"] * flip,
                nan=0.0, posinf=0.0, neginf=0.0
            ))

            out[trait][int(L)] = {
                "delta_l2": d_l2,
                "first_kl": fkl,
                "flip": flip,
                "combined": combined,
            }

    # Persist
    with open(os.path.join(cfg.results_dir, "steering_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(cfg.results_dir, "steering_metrics.csv"), "w") as f:
        f.write("trait,layer,delta_l2,first_kl,flip,combined\n")
        for trait, layer_dict in out.items():
            for L, vals in layer_dict.items():
                f.write(f"{trait},{L},{vals['delta_l2']:.6f},{vals['first_kl']:.6f},{vals['flip']:.6f},{vals['combined']:.6f}\n")

    return out


# -----------------------------
# Normalization & combination
# -----------------------------

def _normalize_per_trait(layer_dict: Dict[int, float]) -> Dict[int, float]:
    values = np.array([v for v in layer_dict.values()], dtype=np.float64)
    if values.size == 0:
        return {k: 0.0 for k in layer_dict}
    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin + 1e-12:
        return {k: 0.0 for k in layer_dict}
    return {k: (float(layer_dict[k]) - vmin) / (vmax - vmin) for k in layer_dict}


def combine_representation_and_steering(rep: Dict, steer: Dict) -> Dict:
    """
    Creates a combined normalized score per trait × layer and selects best layers.
    Returns dict:
      {
        trait: {
          "scores": {layer: {"rep": r, "steer": s, "combined": c}},
          "best_layer": int
        }
      }
    Also writes combined JSON and a compact table.
    """
    combined: Dict[str, Dict] = {}

    for trait in rep.keys():
        # Pull per-layer metrics; guard if missing
        rep_layers = rep.get(trait, {})
        steer_layers = steer.get(trait, {})

        rep_score_raw = {L: rep_layers[L]["probe_acc"] for L in rep_layers}
        steer_score_raw = {L: steer_layers[L]["combined"] for L in steer_layers}

        # Normalize within trait
        rep_norm   = _normalize_per_trait(rep_score_raw)
        steer_norm = _normalize_per_trait(steer_score_raw)

        # Intersection of available layers
        layers = sorted(set(rep_norm.keys()) & set(steer_norm.keys()))
        scores = {}
        best_L, best_val = None, -1.0
        for L in layers:
            r = rep_norm[L]
            s = steer_norm[L]
            c = 0.5*r + 0.5*s
            scores[int(L)] = {"rep": r, "steer": s, "combined": c}
            if c > best_val:
                best_val, best_L = c, int(L)

        combined[trait] = {"scores": scores, "best_layer": best_L}

    return combined


# -----------------------------
# Visualization helpers
# -----------------------------

def _plot_line_per_trait(cfg: Config, data: Dict[str, Dict[int, float]], title: str, fname: str, ylabel: str):
    plt.figure(figsize=(8, 4.5))
    for trait, layer_dict in data.items():
        if not layer_dict:
            continue
        xs = sorted(layer_dict.keys())
        ys = [layer_dict[x] for x in xs]
        plt.plot(xs, ys, marker="o", label=trait)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, fname), dpi=200)
    plt.close()


def _plot_heatmap(cfg: Config, traits: List[str], layers: List[int], matrix: np.ndarray, title: str, fname: str, cbar_label: str):
    plt.figure(figsize=(10, 4.8))
    im = plt.imshow(matrix, aspect="auto", origin="lower")
    plt.colorbar(im, fraction=0.046, pad=0.04, label=cbar_label)
    plt.yticks(range(len(traits)), traits)
    plt.xticks(range(len(layers)), layers, rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, fname), dpi=200)
    plt.close()


def visualize_representation(cfg: Config, rep: Dict):
    # Heatmaps per metric
    traits = list(cfg.trait_mapping.values())
    layers = list(cfg.layer_range)

    # probe_acc heatmap
    M_acc = np.full((len(traits), len(layers)), np.nan)
    M_dn  = np.full((len(traits), len(layers)), np.nan)
    for i, t in enumerate(traits):
        for j, L in enumerate(layers):
            v = rep.get(t, {}).get(int(L), {})
            if "probe_acc" in v:
                M_acc[i, j] = v["probe_acc"]
            if "delta_norm" in v:
                M_dn[i, j] = v["delta_norm"]

    _plot_heatmap(cfg, traits, layers, np.nan_to_num(M_acc, nan=0.0),
                  title="Representation: Linear Probe Accuracy per Layer",
                  fname="rep_probe_acc_heatmap.png", cbar_label="Accuracy")

    _plot_heatmap(cfg, traits, layers, np.nan_to_num(M_dn, nan=0.0),
                  title="Representation: |μ_high − μ_low| per Layer",
                  fname="rep_delta_norm_heatmap.png", cbar_label="L2 Norm")


def visualize_steering(cfg: Config, steer: Dict):
    traits = list(cfg.trait_mapping.values())
    layers = list(cfg.layer_range)

    for metric, label in [("delta_l2", "Δlogits L2"), ("first_kl", "First-token KL"), ("flip", "Flip-rate"), ("combined", "Combined")]:
        series = {}
        for t in traits:
            series[t] = {int(L): steer.get(t, {}).get(int(L), {}).get(metric, np.nan) for L in layers}
        _plot_line_per_trait(cfg, series, title=f"Steering metric: {label}", fname=f"steer_{metric}_lines.png", ylabel=label)


def visualize_combined(cfg: Config, combo: Dict):
    traits = list(cfg.trait_mapping.values())
    layers = list(cfg.layer_range)
    M = np.full((len(traits), len(layers)), np.nan)
    for i, t in enumerate(traits):
        score_map = combo.get(t, {}).get("scores", {})
        for j, L in enumerate(layers):
            M[i, j] = score_map.get(int(L), {}).get("combined", np.nan)
    _plot_heatmap(cfg, traits, layers, np.nan_to_num(M, nan=0.0),
                  title="Combined (normalized) representation + steering score",
                  fname="combined_scores_heatmap.png", cbar_label="Score (0-1)")


# -----------------------------
# Orchestration
# -----------------------------

def run_layer_justification(cfg: Config, steerer) -> Dict:
    """
    Loads activations, computes representation & steering metrics, writes plots and summaries,
    and returns a dict with best layers per trait.
    """
    # Load activations created by the extraction pipeline
    act_path = os.path.join(cfg.results_dir, "multi_layer_activations.pkl")
    # breakpoint()
    if not os.path.exists(act_path):
        raise FileNotFoundError(f"Activations not found: {act_path}. Run extraction first.")
    with open(act_path, "rb") as f:
        activs = pickle.load(f)

    # Representation
    rep = compute_representation_metrics(cfg, activs)
    visualize_representation(cfg, rep)

    # Steering (reuses helpers)
    steer = compute_steering_metrics(cfg, steerer)
    visualize_steering(cfg, steer)

    # Combine
    combo = combine_representation_and_steering(rep, steer)

    # Save combo
    with open(os.path.join(cfg.results_dir, "combined_layer_scores.json"), "w") as f:
        json.dump(combo, f, indent=2)

    # Best layers table
    best = {t: v.get("best_layer", None) for t, v in combo.items()}
    with open(os.path.join(cfg.results_dir, "best_layers_by_trait.json"), "w") as f:
        json.dump(best, f, indent=2)

    # Visualize combined
    visualize_combined(cfg, combo)

    # Emit a compact markdown summary
    md_path = os.path.join(cfg.results_dir, "layer_justification_summary.md")
    with open(md_path, "w") as f:
        f.write("# Layer Justification Summary\n\n")
        f.write("Based on 40k activations: representation separability (probe acc, Δ mean norm)\n"
                "and steering sensitivity (ΔL2, KL, flip), combined per trait.\n\n")
        f.write("## Best layers per trait\n\n")
        for t in cfg.trait_mapping.values():
            f.write(f"- **{t}**: {best.get(t)}\n")
        f.write("\nArtifacts:\n\n")
        f.write("- representation_metrics.json / .csv\n")
        f.write("- steering_metrics.json / .csv\n")
        f.write("- combined_layer_scores.json\n")
        f.write("- best_layers_by_trait.json\n")
        f.write("- plots: rep_* png, steer_* png, combined_scores_heatmap.png\n")

    return best
