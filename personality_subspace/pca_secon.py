# -*- coding: utf-8 -*-

import os
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

# --- CONFIGURATION ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 16,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "Openness": "#3366CC", "Conscientiousness": "#DC3912", 
    "Extraversion": "#FF9900", "Agreeableness": "#109618", 
    "Neuroticism": "#990099", "Default": "#7F8C8D"
}

# =============================================================================
# PART 1: DATA EXPORT (THE JSON DUMP)
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def compute_and_save_json(artifacts, activations, var_stats, out_dir):
    print("[DATA] Computing metrics for JSON export...")
    
    # 1. ORTHOGONALITY DATA
    directions = artifacts["trait_directions"]
    traits = sorted(directions.keys())
    n = len(traits)
    ortho_matrix = np.zeros((n, n))
    for i, t1 in enumerate(traits):
        for j, t2 in enumerate(traits):
            v1, v2 = np.array(directions[t1]).flatten(), np.array(directions[t2]).flatten()
            ortho_matrix[i, j] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
    # 2. SIGNAL STRENGTH DATA
    subspace = artifacts["subspace"]
    signal_strengths = {}
    for t, vec in directions.items():
        vec = np.array(vec).flatten()
        proj = subspace @ (subspace.T @ vec)
        signal_strengths[t] = np.linalg.norm(proj)

    # 3. ACTIVATION METRICS (If available)
    cluster_scores = {}
    emergence_curve = {}
    activation_layers = []
    
    if activations:
        activation_layers = sorted(activations.keys())
        
        # Silhouette Scores (Cluster Quality)
        # Use middle layer if available, else first
        if len(activation_layers) > 0:
            mid_idx = len(activation_layers)//2
            mid_layer = activation_layers[mid_idx]
            for t in traits:
                if f"{t}_high" in activations[mid_layer]:
                    high = activations[mid_layer][f"{t}_high"]
                    low = activations[mid_layer][f"{t}_low"]
                    # Stack
                    if len(high) > 0 and len(low) > 0:
                        X = np.vstack([high, low])
                        y = np.array([0]*len(high) + [1]*len(low))
                        # Subsample for speed
                        if len(X) > 1000:
                            idx = np.random.choice(len(X), 1000, replace=False)
                            X, y = X[idx], y[idx]
                        try:
                            cluster_scores[t] = silhouette_score(X, y)
                        except:
                            cluster_scores[t] = 0.0
            
            # Emergence (Distance over layers)
            for t in traits:
                curve = []
                for L in activation_layers:
                    if f"{t}_high" in activations[L]:
                        h = np.mean(activations[L][f"{t}_high"], axis=0)
                        l = np.mean(activations[L][f"{t}_low"], axis=0)
                        curve.append(np.linalg.norm(h - l))
                    else:
                        curve.append(0.0)
                emergence_curve[t] = curve

    # --- BUILD THE MASTER DICT ---
    master_data = {
        "metadata": {
            "model": artifacts["config"].get("model_name", "unknown"),
            "layer_range": artifacts["config"]["layer_range"],
            "activation_layers": activation_layers # Saved separately to track mismatch
        },
        "layer_weights": artifacts["layer_weights"],
        "variance_stats": var_stats,
        "metrics": {
            "orthogonality_matrix": ortho_matrix,
            "traits_ordered": traits,
            "signal_strength": signal_strengths,
            "cluster_quality_silhouette": cluster_scores,
            "emergence_curves": emergence_curve
        }
    }
    
    # Save
    json_path = os.path.join(out_dir, "experiment_data.json")
    with open(json_path, "w") as f:
        json.dump(master_data, f, cls=NumpyEncoder, indent=2)
    print(f"[DATA] Saved ALL raw data to -> {json_path}")
    return master_data

# =============================================================================
# PART 2: VISUALIZATION GENERATORS
# =============================================================================

def plot_layer_profile(data, out_dir):
    weights = np.array(data["layer_weights"])
    cfg_layers = np.array(data["metadata"]["layer_range"])
    
    # --- CRITICAL FIX FOR CRASH ---
    # If lengths don't match, we cannot use cfg_layers as X-axis.
    # We fallback to simple indices 0..N
    x_axis = cfg_layers
    x_label = "Layer Number"
    
    if len(weights) != len(cfg_layers):
        print(f"[WARN] Length Mismatch! Config Layers: {len(cfg_layers)}, Weights: {len(weights)}")
        print(f"       -> Switching to sequential index for X-axis.")
        x_axis = np.arange(len(weights))
        x_label = "Relative Layer Index"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x_axis, weights, color="#34495E", alpha=0.1)
    ax.plot(x_axis, weights, color="#2C3E50", linewidth=2.5)
    
    peak_idx = np.argmax(weights)
    peak_layer = x_axis[peak_idx]
    peak_val = weights[peak_idx]
    
    ax.annotate(f'Peak: {peak_layer}', xy=(peak_layer, peak_val), xytext=(peak_layer, peak_val + (max(weights)*0.1)),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), ha='center')
    
    ax.set_title("Layer Importance Profile")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Weight")
    plt.savefig(os.path.join(out_dir, "01_layer_profile.png"))
    plt.close()

def plot_orthogonality(data, out_dir):
    matrix = np.array(data["metrics"]["orthogonality_matrix"])
    traits = data["metrics"]["traits_ordered"]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                xticklabels=traits, yticklabels=traits, ax=ax)
    ax.set_title("Trait Orthogonality")
    plt.savefig(os.path.join(out_dir, "03_orthogonality_matrix.png"))
    plt.close()

def plot_signal_strength(data, out_dir):
    sigs = data["metrics"]["signal_strength"]
    sorted_s = sorted(sigs.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0].title() for x in sorted_s]
    vals = [x[1] for x in sorted_s]
    colors = [COLORS.get(x[0].capitalize(), "#7F8C8D") for x in sorted_s]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel("L2 Norm"); ax.set_title("Signal Strength")
    ax.invert_yaxis()
    plt.savefig(os.path.join(out_dir, "07_signal_strength.png"))
    plt.close()

def plot_emergence(data, out_dir):
    curves = data["metrics"]["emergence_curves"]
    if not curves: return
    
    # --- ROBUSTNESS FIX ---
    # Emergence curves come from activations, which might have different length than artifacts config
    activation_layers = data["metadata"]["activation_layers"]
    config_layers = data["metadata"]["layer_range"]
    
    # Pick the X-axis that matches the curve length
    sample_curve = next(iter(curves.values()))
    
    if len(sample_curve) == len(activation_layers):
        x_axis = activation_layers
        x_label = "Layer Number (Activations)"
    elif len(sample_curve) == len(config_layers):
        x_axis = config_layers
        x_label = "Layer Number (Config)"
    else:
        x_axis = np.arange(len(sample_curve))
        x_label = "Relative Layer Index"
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for t, curve in curves.items():
        c = COLORS.get(t.capitalize(), "#7F8C8D")
        ax.plot(x_axis, curve, marker='o', markersize=4, label=t.title(), color=c)
        
    ax.set_title("Signal Emergence (High vs Low Dist)")
    ax.set_xlabel(x_label); ax.set_ylabel("Euclidean Distance")
    ax.legend()
    plt.savefig(os.path.join(out_dir, "08_signal_emergence.png"))
    plt.close()

def plot_cluster_quality(data, out_dir):
    scores = data["metrics"]["cluster_quality_silhouette"]
    if not scores: return
    
    names = sorted(scores.keys())
    vals = [scores[n] for n in names]
    colors = ['#27AE60' if v > 0.1 else '#E74C3C' for v in vals]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, vals, color=colors)
    ax.axhline(0, color='black', lw=1)
    ax.set_title("Cluster Quality (Silhouette Score)")
    plt.savefig(os.path.join(out_dir, "09_cluster_quality.png"))
    plt.close()

# --- HELPER FOR PCA PLOTS (Requires raw artifacts still) ---
def plot_raw_artifacts_figures(artifacts, out_dir):
    subspace = artifacts["subspace"]
    directions = artifacts["trait_directions"]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0, color='grey', ls='--'); ax.axvline(0, color='grey', ls='--')
    
    max_val = 0
    for trait, vec in directions.items():
        vec = np.array(vec).flatten()
        x = np.dot(vec, subspace[:, 0])
        y = np.dot(vec, subspace[:, 1])
        c = COLORS.get(trait.capitalize(), "#7F8C8D")
        ax.arrow(0, 0, x, y, head_width=0.03, fc=c, ec=c, alpha=0.5)
        ax.scatter(x, y, color=c, s=150, ec='white', zorder=5)
        ax.text(x+0.04, y+0.04, trait.title(), color=c, fontweight='bold')
        max_val = max(max_val, abs(x), abs(y))
        
    limit = max_val * 1.3
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
    ax.set_title("PCA Biplot (PC1 vs PC2)")
    plt.savefig(os.path.join(out_dir, "02_pca_biplot.png"))
    plt.close()

# =============================================================================
# MAIN RUNNER
# =============================================================================

def load_pkl(path):
    with open(path, "rb") as f: return pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--activations", required=False, help="Optional for deep insights")
    parser.add_argument("--outdir", default="paper_results_final")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # 1. Load Data
    print("Loading raw files...")
    artifacts = load_pkl(args.artifacts)
    
    activations = None
    if args.activations and os.path.exists(args.activations):
        activations = load_pkl(args.activations)
    else:
        print("[WARN] No activations file found. Skipping deep/emergence plots.")
        
    # Check for variance file
    var_path = os.path.join(os.path.dirname(args.artifacts), "subspace_variance.json")
    var_stats = None
    if os.path.exists(var_path):
        with open(var_path, "r") as f: var_stats = json.load(f)

    # 2. Compute & JSON Dump
    # This creates the structured dictionary we use for plotting AND saves it
    master_data = compute_and_save_json(artifacts, activations, var_stats, args.outdir)
    
    # 3. Generate Visuals
    print("Generating figures...")
    plot_layer_profile(master_data, args.outdir)
    plot_orthogonality(master_data, args.outdir)
    plot_signal_strength(master_data, args.outdir)
    plot_emergence(master_data, args.outdir)
    plot_cluster_quality(master_data, args.outdir)
    
    # PCA Plots need the raw high-dim vectors which are in 'artifacts' but not 'master_data'
    plot_raw_artifacts_figures(artifacts, args.outdir)
    
    print(f"\n[SUCCESS] Analysis complete.\n  - Data: {os.path.join(args.outdir, 'experiment_data.json')}\n  - Figures: {args.outdir}/*.png")

if __name__ == "__main__":
    main()