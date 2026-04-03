
import os
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# --- 1. PUBLICATION STYLE CONFIG ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

# Trait Colors (Colorblind-friendly Palette)
COLORS = {
    "Openness": "#3366CC",          # Blue
    "Conscientiousness": "#DC3912", # Red
    "Extraversion": "#FF9900",      # Orange
    "Agreeableness": "#109618",     # Green
    "Neuroticism": "#990099",       # Purple
    "Default": "#7F8C8D"
}

def load_data(artifacts_path):
    print(f"[INFO] Loading artifacts from: {artifacts_path}")
    base_dir = os.path.dirname(artifacts_path)
    with open(artifacts_path, "rb") as f:
        data = pickle.load(f)
    
    # Check for variance stats
    var_path = os.path.join(base_dir, "subspace_variance.json")
    if os.path.exists(var_path):
        with open(var_path, "r") as f:
            data["variance_stats"] = json.load(f)
    else:
        data["variance_stats"] = None
    return data

# =============================================================================
# PART A: STANDARD VALIDATION PLOTS (Essential for Proof)
# =============================================================================

def plot_layer_profile(data, out_dir):
    """Figure 1: Where does personality live? (Area Chart)"""
    weights = np.array(data["layer_weights"])
    layer_range = data["config"]["layer_range"]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot Area
    ax.fill_between(layer_range, weights, color="#34495E", alpha=0.1)
    ax.plot(layer_range, weights, color="#2C3E50", linewidth=2.5)
    
    # Annotate Peak
    peak_idx = np.argmax(weights)
    peak_layer = layer_range[peak_idx]
    peak_val = weights[peak_idx]
    ax.annotate(f'Peak: Layer {peak_layer}', xy=(peak_layer, peak_val), xytext=(peak_layer, peak_val + 0.15),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                ha='center', fontweight='bold')
    
    ax.set_title("Layer Importance Profile", fontweight='bold')
    ax.set_xlabel("Model Layer")
    ax.set_ylabel("Learned Weight")
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(os.path.join(out_dir, "01_layer_profile.png"))
    plt.close()

def plot_pca_biplot(data, out_dir):
    """Figure 2: The Personality Space (Biplot with Arrows)"""
    subspace = data["subspace"]
    directions = data["trait_directions"]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Origin Lines
    ax.axhline(0, color='grey', lw=1, ls='--')
    ax.axvline(0, color='grey', lw=1, ls='--')
    
    max_val = 0
    for trait, vec in directions.items():
        vec = vec.flatten()
        # Project onto PC1 and PC2
        x = np.dot(vec, subspace[:, 0])
        y = np.dot(vec, subspace[:, 1])
        c = COLORS.get(trait.capitalize(), COLORS["Default"])
        
        # Arrow + Dot + Label
        ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.03, fc=c, ec=c, alpha=0.5, length_includes_head=True)
        ax.scatter(x, y, color=c, s=150, edgecolor='white', lw=1.5, zorder=5)
        ax.text(x + 0.04, y + 0.04, trait.title(), fontsize=11, fontweight='bold', color=c)
        
        max_val = max(max_val, abs(x), abs(y))
    
    limit = max_val * 1.3
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Personality Subspace Geometry (PC1 vs PC2)", fontweight='bold')
    
    plt.savefig(os.path.join(out_dir, "02_pca_biplot.png"))
    plt.close()

def plot_orthogonality_heatmap(data, out_dir):
    """Figure 3: Are traits independent? (Correlation Matrix)"""
    directions = data["trait_directions"]
    traits = sorted(directions.keys())
    n = len(traits)
    matrix = np.zeros((n, n))
    
    for i, t1 in enumerate(traits):
        for j, t2 in enumerate(traits):
            v1, v2 = directions[t1].flatten(), directions[t2].flatten()
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            matrix[i, j] = sim

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                xticklabels=[t.title() for t in traits],
                yticklabels=[t.title() for t in traits],
                cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title("Trait Orthogonality (Cosine Similarity)", fontweight='bold')
    plt.savefig(os.path.join(out_dir, "03_orthogonality_matrix.png"))
    plt.close()

# =============================================================================
# PART B: ADVANCED INTERPRETABILITY (Essential for Insights)
# =============================================================================

def plot_pareto_variance(data, out_dir):
    """Figure 4: The 'Knee' Plot (Variance Efficiency)"""
    if not data["variance_stats"]: return
    
    evr = np.array(data["variance_stats"]["explained_variance_ratio"])
    cum = np.array(data["variance_stats"]["cumulative"])
    x = np.arange(1, len(evr) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bars
    ax1.bar(x, evr, color="#95A5A6", alpha=0.5, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance")
    
    # Line
    ax2 = ax1.twinx()
    ax2.plot(x, cum, color="#C0392B", marker='o', lw=2, label="Cumulative")
    ax2.set_ylabel("Cumulative Variance")
    ax2.set_ylim(0, 1.05)
    
    # Thresholds (80%, 90%)
    for thr in [0.8, 0.9]:
        idx = np.argmax(cum >= thr)
        ax2.axhline(thr, color="black", ls=":", alpha=0.5)
        ax2.text(idx + 1.5, thr - 0.05, f"{thr*100}% at PC{idx+1}", fontweight='bold')

    plt.title("Dimensionality Efficiency (Pareto Chart)", fontweight='bold')
    plt.savefig(os.path.join(out_dir, "04_variance_pareto.png"))
    plt.close()

def plot_pc_loadings(data, out_dir):
    """Figure 5: What does PC1 mean? (Loadings Heatmap)"""
    subspace = data["subspace"]
    directions = data["trait_directions"]
    traits = sorted(directions.keys())
    pcs = [f"PC{i+1}" for i in range(min(5, subspace.shape[1]))] # Top 5 PCs
    
    loadings = np.zeros((len(traits), len(pcs)))
    for i, t in enumerate(traits):
        vec = directions[t].flatten()
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        for j in range(len(pcs)):
            loadings[i, j] = np.dot(vec, subspace[:, j]) # Dot product
            
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=pcs, yticklabels=[t.title() for t in traits], ax=ax)
    
    ax.set_title("Subspace Fingerprint (Trait-PC Correlations)", fontweight='bold')
    plt.savefig(os.path.join(out_dir, "05_pc_loadings.png"))
    plt.close()

def plot_dendrogram(data, out_dir):
    """Figure 6: Semantic Hierarchy (Clustering)"""
    directions = data["trait_directions"]
    traits = sorted(directions.keys())
    
    # Distance Matrix
    vecs = np.array([directions[t].flatten() for t in traits])
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    dists = 1 - np.dot(vecs, vecs.T)
    dists[dists < 0] = 0
    
    # Cluster
    linked = linkage(squareform(dists), 'ward')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(linked, labels=[t.title() for t in traits], orientation='top', leaf_font_size=12, ax=ax)
    
    ax.set_title("Semantic Hierarchy of Traits", fontweight='bold')
    ax.set_ylabel("Dissimilarity (Cosine Dist)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(os.path.join(out_dir, "06_semantic_dendrogram.png"))
    plt.close()

def plot_signal_strength(data, out_dir):
    """Figure 7: Which trait is strongest? (Bar Chart)"""
    subspace = data["subspace"]
    directions = data["trait_directions"]
    
    strengths = {}
    for t, vec in directions.items():
        # Project vector into subspace and measure length
        proj = subspace @ (subspace.T @ vec.flatten())
        strengths[t] = np.linalg.norm(proj)
        
    sorted_s = sorted(strengths.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0].title() for x in sorted_s]
    vals = [x[1] for x in sorted_s]
    colors = [COLORS.get(x[0].capitalize(), "#7F8C8D") for x in sorted_s]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel("Projection Magnitude (L2 Norm)")
    ax.set_title("Signal Strength in Subspace", fontweight='bold')
    ax.invert_yaxis() # Top trait at top
    
    plt.savefig(os.path.join(out_dir, "07_signal_strength.png"))
    plt.close()

# =============================================================================
# MAIN RUNNER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive visualization suite.")
    parser.add_argument("--artifacts", type=str, required=True, help="Path to artifacts.pkl")
    parser.add_argument("--outdir", type=str, default="figures_complete", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[START] Generating figures in {args.outdir}...")
    
    data = load_data(args.artifacts)
    
    # 1. Standard Plots
    plot_layer_profile(data, args.outdir)
    print("  -> Generated 01_layer_profile.png")
    
    plot_pca_biplot(data, args.outdir)
    print("  -> Generated 02_pca_biplot.png")
    
    plot_orthogonality_heatmap(data, args.outdir)
    print("  -> Generated 03_orthogonality_matrix.png")
    
    # 2. Advanced Plots
    plot_pareto_variance(data, args.outdir)
    print("  -> Generated 04_variance_pareto.png")
    
    plot_pc_loadings(data, args.outdir)
    print("  -> Generated 05_pc_loadings.png")
    
    plot_dendrogram(data, args.outdir)
    print("  -> Generated 06_semantic_dendrogram.png")
    
    plot_signal_strength(data, args.outdir)
    print("  -> Generated 07_signal_strength.png")
    
    print("\n[SUCCESS] All 7 figures generated successfully.")

if __name__ == "__main__":
    main()