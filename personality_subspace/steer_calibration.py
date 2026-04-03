
import argparse, os, json, math
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv


# plotting (optional)
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [12, 8]
sns.set_style("whitegrid")

# project imports
from .main import load_steerer
from .layer_selector import select_layers_for_prompt, SteerConfigPatch
import os

load_dotenv()
def make_clean_alpha_viz(df_alpha_sweep: pd.DataFrame, outdir: str, top_k: int = 2,
                         take_threshold: float = 1.2, min_kl_pos: float = 0.02):
    """
    Creates 4 concise figures from alpha sweep data:
      1) Faceted median+IQR ribbons of log_ratio vs alpha (top-K combos per trait)
      2) Sign-consistency heatmap (top-K combos per trait)
      3) Beeswarm of critical alpha (alpha* where steer 'takes')
      4) Cleveland dot plot of AUC of clipped log-ratio per combo
    """
    os.makedirs(outdir, exist_ok=True)
    df = df_alpha_sweep.copy()

    # Pre-compute helpful columns
    df["log_ratio"] = np.log(df["kl_pos"] / (df["kl_neg"] + 1e-9))
    df["sign_pos"] = (df["kl_pos"] > df["kl_neg"]).astype(float)

    # --- helper: clip extreme log-ratios so outliers don't dominate
    lo_clip, hi_clip = -np.log(1.5), np.log(2.5)  # ≈ [-0.405, 0.916]
    def _clip(s): 
        return np.clip(s, lo_clip, hi_clip)

    # --- aggregate by trait/combo/alpha
    agg = (df.groupby(["trait","combo","alpha"])
             .agg(
                 log_ratio_med=("log_ratio", lambda s: np.median(_clip(s))),
                 log_ratio_q1=("log_ratio", lambda s: np.quantile(_clip(s), 0.25)),
                 log_ratio_q3=("log_ratio", lambda s: np.quantile(_clip(s), 0.75)),
                 sign_rate=("sign_pos", "mean"),
                 kl_pos_med=("kl_pos","median"),
                 kl_neg_med=("kl_neg","median")
             ).reset_index())

    # --- AUC (clipped log-ratio) per trait/combo
    def _auc(g):
        g = g.sort_values("alpha")
        return np.trapz(_clip(g["log_ratio"].values), g["alpha"].values)
    auc = (df.groupby(["trait","combo"])
             .apply(_auc)
             .reset_index(name="auc"))

    # Pick top-K combos per trait by AUC
    top = (auc.sort_values(["trait","auc"], ascending=[True, False])
               .groupby("trait").head(top_k))
    top_set = set(map(tuple, top[["trait","combo"]].values))
    agg_top = agg[[ (t,c) in top_set for t,c in agg[["trait","combo"]].values ]].copy()

    # x is discrete (α values)
    alpha_vals = sorted(df["alpha"].unique())

    # 1) Median + IQR ribbons
    # small-multiples: rows = trait, cols = combo (top-K)
    grid_items = []
    for t in agg_top["trait"].unique():
        combos = agg_top[agg_top["trait"]==t]["combo"].unique().tolist()
        for c in combos:
            grid_items.append((t,c))
    n_rows = len(agg_top["trait"].unique())
    n_cols = max(top_k, 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0*n_cols, 3.7*n_rows), sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(-1,1)

    # map (trait,row) ordering stable
    trait_order = list(agg_top["trait"].unique())
    per_trait_combos = {t: list(agg_top[agg_top["trait"]==t]["combo"].unique()) for t in trait_order}

    for r, t in enumerate(trait_order):
        combos = per_trait_combos[t]
        for c_idx in range(n_cols):
            ax = axes[r, c_idx]
            ax.axhline(0.0, ls=":", lw=1, alpha=0.6)
            if c_idx >= len(combos):
                ax.set_axis_off()
                continue
            c = combos[c_idx]
            sub = agg_top[(agg_top["trait"]==t) & (agg_top["combo"]==c)].sort_values("alpha")
            x = sub["alpha"].values
            y = sub["log_ratio_med"].values
            q1 = sub["log_ratio_q1"].values
            q3 = sub["log_ratio_q3"].values
            ax.plot(x, y, marker="o", lw=2)
            ax.fill_between(x, q1, q3, alpha=0.2, edgecolor="none")
            ax.set_title(f"{t} — {c}")
            if c_idx == 0:
                ax.set_ylabel("median log(KL+ / KL−)  (IQR shaded)")
            ax.set_xlabel("alpha")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_median_iqr_logratio_small_multiples.png"), dpi=300)
    plt.close(fig)

    # 2) Sign-consistency heatmap che kk
    # one heatmap per trait; rows=top combos, cols=alpha
    for t in trait_order:
        sub = agg[(agg["trait"]==t) & (agg["combo"].isin(per_trait_combos[t]))]
        # keep only top_k by AUC
        topc = (auc[auc["trait"]==t]
                  .sort_values("auc", ascending=False)
                  .head(top_k)["combo"].tolist())
        sub = sub[sub["combo"].isin(topc)]
        pivot = (sub.pivot_table(index="combo", columns="alpha", values="sign_rate", aggfunc="mean")
                   .reindex(index=topc))  # preserve order
        plt.figure(figsize=(1.2*len(alpha_vals)+2, 1+1.2*len(topc)))
        sns.heatmap(pivot, vmin=0.0, vmax=1.0, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"Sign-consistency heatmap (KL+ > KL−) — {t}")
        plt.xlabel("alpha")
        plt.ylabel("combo")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"02_sign_consistency_{t}.png"), dpi=300)
        plt.close()

    # 3) Critical-alpha beeswarm ]]]]
    # alpha* = min alpha where log_ratio > log(take_threshold) and KL_pos > min_kl_pos
    tau = np.log(take_threshold)
    take = df[(df["log_ratio"] > tau) & (df["kl_pos"] > min_kl_pos)]
    crit = (take.groupby(["trait","combo","prompt_id"])["alpha"]
                 .min()
                 .reset_index(name="alpha_star"))
    # keep top combos per trait
    crit = crit[[ (t,c) in top_set for t,c in crit[["trait","combo"]].values ]]

    plt.figure(figsize=(10, 6))
    crit["group"] = crit["trait"] + " | " + crit["combo"]
    order = []
    for t in trait_order:
        for c in per_trait_combos[t][:top_k]:
            if (t,c) in top_set:
                order.append(f"{t} | {c}")
    sns.stripplot(data=crit, x="group", y="alpha_star", order=order, jitter=True, alpha=0.6)
    sns.boxplot(data=crit, x="group", y="alpha_star", order=order, showcaps=False, boxprops={"facecolor":"none"}, showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("critical alpha*")
    plt.title(f"Critical α (first α where steer ‘takes’: logratio>{take_threshold:.2f} & KL+>{min_kl_pos})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "03_critical_alpha_beeswarm.png"), dpi=300)
    plt.close()

    # 4) AUC summary 
   
    auc["rank"] = auc.groupby("trait")["auc"].rank(ascending=False, method="first")
    plt.figure(figsize=(10, 6))
    # order by trait then auc
    auc_sorted = (auc.sort_values(["trait","auc"], ascending=[True, False]))
    # draw per-trait bands
    y_labels = []
    y_pos = []
    y = 0
    for t, grp in auc_sorted.groupby("trait"):
        g = grp.sort_values("auc", ascending=True)  # smallest at bottom within trait block
        plt.hlines(y=np.arange(y, y+len(g)), xmin=0, xmax=g["auc"].values, linestyles=":")
        plt.plot(g["auc"].values, np.arange(y, y+len(g)), "o")
        for _, row in g.iterrows():
            y_labels.append(f"{t} | {row['combo']}")
            y_pos.append(y)
            y += 1
        # spacer
        y += 1
    plt.yticks(range(len(y_labels)), y_labels)
    plt.xlabel("AUC of clipped log-ratio (higher is better)")
    plt.title("Combo quality summary per trait")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "04_auc_dotplot.png"), dpi=300)
    plt.close()

    print(f"[viz] Saved plots in: {outdir}")




OCEAN_TRAITS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

TEST_PROMPTS: Dict[str, List[str]] = {
    'openness': [
        "Describe your ideal creative project.",
        "What new skill would you like to learn and why?",
        "Imagine exploring a completely unknown place."
    ],
    'conscientiousness': [
        "How do you organize your daily schedule?",
        "Describe your approach to meeting important deadlines.",
        "What does being responsible mean to you?"
    ],
    'extraversion': [
        "What makes a social gathering enjoyable for you?",
        "Describe your perfect weekend with friends.",
        "How do you recharge after social events?"
    ],
    'agreeableness': [
        "How do you handle disagreements with others?",
        "Describe a time you helped someone without expecting anything in return.",
        "What does cooperation mean to you?"
    ],
    'neuroticism': [
        "How do you deal with stressful situations?",
        "Describe a time you felt anxious and how you coped.",
        "What worries keep you up at night?"
    ]
}

def _kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0)
    q = q.clamp(min=eps, max=1.0)
    return torch.sum(p * (torch.log(p) - torch.log(q)))

def _format_for_chat(steerer, text: str, system: str = None) -> str:
    return steerer._format_prompt(text, use_chat_template=True, system=system)

@torch.no_grad()
def _next_token_kl(steerer, prompt: str, trait: str, alpha_eff: float, sign: int, system: str = None) -> float:
    """
    Next-token KL between baseline and steered logits, consistent with quick_steer/bench.
    """
    txt = _format_for_chat(steerer, prompt, system)
    # keep RMS cache consistent with generate()
    try:
        steerer._measure_layer_rms(txt)
    except Exception:
        pass

    enc = steerer.tok(txt, return_tensors="pt").to(steerer.device)
    out = steerer.model(**enc)
    p0 = torch.softmax(out.logits[:, -1, :].float(), dim=-1).squeeze(0)

    intensity = (sign * alpha_eff) / float(steerer.steer_gain)
    steerer._register(trait, intensity)
    try:
        out_s = steerer.model(**enc)
        p1 = torch.softmax(out_s.logits[:, -1, :].float(), dim=-1).squeeze(0)
    finally:
        steerer._clear()

    return float(_kl_div(p0, p1).item())

@torch.no_grad()
def calibrate_alpha_for_sign_in_context(
    steerer, prompt: str, trait: str, sign: int, target_kl: float,
    system: str = None, alpha_hi: float = 8.0, alpha_lo: float = 0.0,
    max_iters: int = 14, tol_frac: float = 0.05, expand_factor: float = 2.0,
    alpha_hi_cap: float = 64.0
) -> float:
    """
    Bisection on alpha_eff to hit next-token KL ~= target_kl for a given sign.
    Returns the unsigned alpha_eff (the "scale" printed in your [delta] logs).
    Mirrors bench/quick_steer logic.
    """
    assert target_kl > 0.0
    lo, hi = float(alpha_lo), float(alpha_hi)

    kl_hi = _next_token_kl(steerer, prompt, trait, hi, sign, system)
    while kl_hi < target_kl and hi < alpha_hi_cap:
        lo = hi
        hi *= float(expand_factor)
        kl_hi = _next_token_kl(steerer, prompt, trait, hi, sign, system)

    if kl_hi < target_kl and hi >= alpha_hi_cap:
        return hi

    _ = 0.0 if lo == 0.0 else _next_token_kl(steerer, prompt, trait, lo, sign, system)
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        kl_mid = _next_token_kl(steerer, prompt, trait, mid, sign, system)
        if abs(kl_mid - target_kl) <= tol_frac * target_kl:
            return mid
        if kl_mid < target_kl:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

@torch.no_grad()
def _kl_signed_on_layer(steerer, prompt: str, trait: str, layer: int, *,
                        system: str = None, alpha_test: float = 2.0) -> Tuple[float, float]:
    """
    Probe signed next-token KL on a single layer (used for the signed-KL filter).
    """
    with SteerConfigPatch(steerer, [layer], [1.0]):
        klp = _next_token_kl(steerer, prompt, trait, alpha_test, +1, system=system)
        kln = _next_token_kl(steerer, prompt, trait, alpha_test, -1, system=system)
    return float(klp), float(kln)

def _ensure_verified_and_floor(
    layers: List[int], weights: List[float], norms: Dict[int, float],
    verified: List[int], min_verified_weight: float
) -> Tuple[List[int], List[float]]:
    """
    Ensure at least one verified layer is in the mix and enforce a minimal
    total weight on verified layers.
    """
    if not verified or min_verified_weight <= 0.0:
        return layers, weights

    vset = set(verified)
    L = list(layers)
    W = np.asarray(weights, dtype=np.float32)
    W = W / (W.sum() + 1e-9)

    has_verified = any(Li in vset for Li in L)
    if not has_verified:
        # inject the best verified layer by runtime norm, evict the smallest weight slot
        v_best = max(verified, key=lambda Li: norms.get(Li, float("-inf")))
        j_evict = int(np.argmin(W))
        L[j_evict] = v_best

    v_mask = np.array([Li in vset for Li in L], dtype=bool)
    v_sum = float(W[v_mask].sum())
    if v_sum < min_verified_weight:
        nv_sum = float(W[~v_mask].sum())
        v_scale  = (min_verified_weight / (v_sum + 1e-9)) if v_sum > 0 else 0.0
        nv_scale = ((1.0 - min_verified_weight) / (nv_sum + 1e-9)) if nv_sum > 0 else 0.0
        W[v_mask]  *= v_scale
        W[~v_mask] *= nv_scale
        W = W / (W.sum() + 1e-9)

    return L, W.tolist()

@torch.no_grad()
def build_runtime_mix_like_quicksteer(
    steerer, prompt: str, trait: str,
    *, system: str,
    k_runtime: int, max_layers: int, prior_boost: float,
    temperature_sel: float, min_weight: float,
    layer_policy: str, min_verified_weight: float,
    signed_filter_alpha: float, log_level: str = "info"
) -> Tuple[List[int], List[float], int, Dict[int, float]]:
    """
    Reproduce the hybrid static+dynamic selection, policy overrides, sign calibration,
    signed-KL filter, and verified weight floor.
    Returns: (layers, weights, sgn, norms)
    """
    # 1) hybrid dynamic selection with prior boost
    layers, weights, norms = select_layers_for_prompt(
        steerer, prompt, trait,
        intensity=0.5,               # only for scoring; not used to steer
        system=system,
        k_runtime=k_runtime,
        max_layers=max_layers,
        prior_boost=prior_boost,
        temperature=temperature_sel,
        min_weight=min_weight
    )

    # 2) policy overrides
    verified = getattr(steerer, "_trait_layers", {}).get(trait.lower(), [])
    if layer_policy == "force_verified" and verified:
        layers = list(verified)[:max_layers]
        weights = [1.0 / len(layers)] * len(layers)
    elif layer_policy == "runtime_only":
        top_sorted = sorted(norms.items(), key=lambda kv: kv[1], reverse=True)
        layers = [L for (L, _) in top_sorted[:max_layers]]
        weights = [1.0 / len(layers)] * len(layers)

    # 3) global sign anchored on verified if available
    anchor_layers = list(verified[:max_layers]) if verified else layers
    anchor_weights = [1.0 / max(1, len(anchor_layers))] * max(1, len(anchor_layers))
    with SteerConfigPatch(steerer, anchor_layers, anchor_weights):
        sgn = int(np.sign(steerer._calibrate_polarity(trait)) or 1)

    # 4) signed-KL filter for runtime layers (agree with sgn)
    if layer_policy != "force_verified" and k_runtime > 0:
        top_sorted = sorted(norms.items(), key=lambda kv: kv[1], reverse=True)
        runtime_pool = [L for (L, _) in top_sorted if L not in set(verified)]
        keep = []
        for L in runtime_pool:
            with SteerConfigPatch(steerer, [L], [1.0]):
                klp = _next_token_kl(steerer, prompt, trait, signed_filter_alpha, +sgn, system)
                kln = _next_token_kl(steerer, prompt, trait, signed_filter_alpha, -sgn, system)
            prefer_plus = klp >= kln
            if (sgn > 0 and prefer_plus) or (sgn < 0 and not prefer_plus):
                keep.append(L)

        # compose: verified ∪ keep[:max_layers - len(verified)]
        new_layers = list(verified[:max_layers])
        for L in keep:
            if len(new_layers) >= max_layers:
                break
            if L not in new_layers:
                new_layers.append(L)
        if new_layers:
            layers = new_layers
            weights = [1.0 / len(layers)] * len(layers)

    # 5) verified weight floor (prefer_verified)
    if layer_policy == "prefer_verified" and verified:
        layers, weights = _ensure_verified_and_floor(layers, weights, norms, verified, min_verified_weight)

    return layers, weights, sgn, norms

# diagnose
@torch.no_grad()
def measure_layer_responsiveness_grid(
    steerer,
    prompt: str,
    trait: str,
    alpha_values: List[float],
    layer_pool: List[int],
    system: str = None
) -> pd.DataFrame:
    """
    Measure responsiveness of each layer across different alphas, using calibrated global sign.
    """
    results: List[Dict[str, Any]] = []

    # Calibrate sign anchored on verified (if present)
    verified = getattr(steerer, "_trait_layers", {}).get(trait.lower(), [])
    anchor_layers = verified[:1] if verified else [layer_pool[0]]
    anchor_weights = [1.0 / len(anchor_layers)] * len(anchor_layers)
    with SteerConfigPatch(steerer, anchor_layers, anchor_weights):
        sgn = int(np.sign(steerer._calibrate_polarity(trait)) or 1)

    for alpha in alpha_values:
        for layer in layer_pool:
            with SteerConfigPatch(steerer, [layer], [1.0]):
                kl = _next_token_kl(steerer, prompt, trait, alpha, sgn, system)
            results.append({
                'layer': layer,
                'alpha': alpha,
                'kl_divergence': kl,
                'trait': trait,
                'prompt': prompt[:60] + "..." if len(prompt) > 60 else prompt
            })

    return pd.DataFrame(results)

@torch.no_grad()
def alpha_sweep_experiment(
    steerer,
    prompt: str,
    trait: str,
    layers_combinations: Dict[str, Tuple[List[int], List[float]]],
    alpha_range: List[float],
    sgn: int,
    system: str = None
) -> pd.DataFrame:
    """
    Sweep alpha values for different layer combinations (each measured with calibrated sign).
    """
    results: List[Dict[str, Any]] = []
    for combo_name, (layers, weights) in layers_combinations.items():
        for alpha in alpha_range:
            with SteerConfigPatch(steerer, layers, weights):
                kl_pos = _next_token_kl(steerer, prompt, trait, alpha, +sgn, system)
                kl_neg = _next_token_kl(steerer, prompt, trait, alpha, -sgn, system)
            results.append({
                'combo': combo_name,
                'layers': str(layers),
                'weights': str([round(w, 3) for w in weights]),
                'alpha': alpha,
                'kl_pos': kl_pos,
                'kl_neg': kl_neg,
                'kl_ratio': (kl_pos / (kl_neg + 1e-9)),
                'trait': trait,
                'prompt': prompt[:60] + "..." if len(prompt) > 60 else prompt
            })
    return pd.DataFrame(results)

def _save_plot(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_visualizations(df_responsiveness: pd.DataFrame,
                                        df_alpha_sweep: pd.DataFrame,
                                        output_dir: str):
    """
    Create diagnostic figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Layer Responsiveness Heatmap (mean KL over prompts)
    plt.figure(figsize=(15, 10))
    pivot_data = df_responsiveness.pivot_table(
        index='layer', columns='alpha', values='kl_divergence', aggfunc='mean'
    )
    sns.heatmap(pivot_data, annot=False, cmap='viridis')
    plt.title('Layer Responsiveness (mean next-token KL) vs Layer vs Alpha')
    _save_plot(f'{output_dir}/layer_responsiveness_heatmap.png')

    # 2) Alpha Sweep per combo
    plt.figure(figsize=(14, 8))
    for combo in df_alpha_sweep['combo'].unique():
        combo_data = df_alpha_sweep[df_alpha_sweep['combo'] == combo]
        plt.plot(combo_data['alpha'], combo_data['kl_pos'],
                 label=f'{combo} (+)', marker='o', linewidth=2)
        plt.plot(combo_data['alpha'], combo_data['kl_neg'],
                 label=f'{combo} (-)', marker='s', linestyle='--', linewidth=2)
    plt.xlabel('Alpha')
    plt.ylabel('Next-token KL')
    plt.title('Alpha Sweep: KL(+/-) per Layer Mixture')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    _save_plot(f'{output_dir}/alpha_sweep_comparison.png')

    # 3) Layer Effectiveness Ranking (max KL across alphas/prompts)
    plt.figure(figsize=(12, 6))
    layer_effectiveness = df_responsiveness.groupby('layer')['kl_divergence'].max().sort_values(ascending=False)
    layer_effectiveness.head(20).plot(kind='bar', color='skyblue')
    plt.title('Top Responsive Layers (max next-token KL across prompts/alphas)')
    plt.xlabel('Layer')
    plt.ylabel('Max KL')
    plt.xticks(rotation=45)
    _save_plot(f'{output_dir}/layer_effectiveness_ranking.png')

    # 4) Trait-wise layer responsiveness (max over alpha, mean over prompts)
    if 'trait' in df_responsiveness.columns:
        plt.figure(figsize=(12, 8))
        trait_data = (df_responsiveness
                      .groupby(['trait', 'layer'])['kl_divergence']
                      .mean().reset_index())
        pivot_trait = trait_data.pivot_table(index='layer', columns='trait', values='kl_divergence')
        sns.heatmap(pivot_trait, annot=False, cmap='coolwarm')
        plt.title('Trait-wise Layer Responsiveness (mean KL over prompts)')
        _save_plot(f'{output_dir}/trait_layer_responsiveness.png')

def generate_diagnostic_report(steerer, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate comprehensive diagnostics across traits/prompts with runtime-consistent selection.
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    all_responsiveness_data: List[pd.DataFrame] = []
    all_alpha_sweep_data: List[pd.DataFrame] = []

    # Test parameters
    alpha_range = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
    layer_pool = list(steerer.cfg.layer_range)  # FIX: use configured layer range

    print("🚀 Starting Steering Diagnostics (runtime-consistent mixture)")
    print("=" * 70)

    traits_to_run = OCEAN_TRAITS if args.trait == "all" else [args.trait.lower()]

    for trait in traits_to_run:
        if trait not in TEST_PROMPTS:
            print(f"[WARN] No test prompts for trait={trait}; skipping.")
            continue

        print(f"\n🔍 Trait: {trait}")
        verified_layers = getattr(steerer, "_trait_layers", {}).get(trait.lower(), [])
        print(f"   verified layers: {verified_layers}")

        for i, prompt in enumerate(TEST_PROMPTS[trait]):
            print(f"   prompt {i+1}: {prompt[:80]}{'...' if len(prompt)>80 else ''}")

            # Build the SAME runtime mixture as quick_steer for curent prompt
            layers_rt, weights_rt, sgn, norms = build_runtime_mix_like_quicksteer(
                steerer, prompt, trait,
                system=args.system,
                k_runtime=args.k_runtime, max_layers=args.max_layers,
                prior_boost=args.prior_boost, temperature_sel=args.temperature_sel,
                min_weight=args.min_weight, layer_policy=args.layer_policy,
                min_verified_weight=args.min_verified_weight,
                signed_filter_alpha=args.signed_filter_alpha,
                log_level=args.log_level
            )

            # 1) Layer responsiveness grid (using calibrated sign
            if args.diagnostic_mode == "quick":
                lp = layer_pool[:min(16, len(layer_pool))]
            else:
                lp = layer_pool

            df_resp = measure_layer_responsiveness_grid(
                steerer, prompt, trait, alpha_range, lp, system=args.system
            )
            df_resp['prompt_id'] = f"{trait}_{i}"
            all_responsiveness_data.append(df_resp)

            # 2) Alpha sweep: compare mixtures
            verified_combo_layers = (verified_layers[:2] if verified_layers else layers_rt[:1])
            verified_combo_weights = ([0.5, 0.5] if len(verified_combo_layers) > 1 else [1.0])

            layers_combinations = {
                'hybrid_runtime': (layers_rt, weights_rt),
                'verified_only': (verified_combo_layers, verified_combo_weights),
                'runtime_top2': ((layers_rt[:2] if len(layers_rt) >= 2 else layers_rt[:1]),
                                 ([0.5, 0.5] if len(layers_rt) >= 2 else [1.0])),
            }

            df_sweep = alpha_sweep_experiment(
                steerer, prompt, trait, layers_combinations, alpha_range, sgn, system=args.system
            )
            df_sweep['prompt_id'] = f"{trait}_{i}"
            all_alpha_sweep_data.append(df_sweep)

    # Combine all data
    df_all_responsiveness = pd.concat(all_responsiveness_data, ignore_index=True) if all_responsiveness_data else pd.DataFrame()
    df_all_alpha_sweep = pd.concat(all_alpha_sweep_data, ignore_index=True) if all_alpha_sweep_data else pd.DataFrame()

    # Save raw tables
    if not df_all_responsiveness.empty:
        df_all_responsiveness.to_csv(f'{output_dir}/layer_responsiveness_data.csv', index=False)
    if not df_all_alpha_sweep.empty:
        df_all_alpha_sweep.to_csv(f'{output_dir}/alpha_sweep_data.csv', index=False)

    # Visualizations
    if not df_all_responsiveness.empty and not df_all_alpha_sweep.empty:
        print("\n Generating visualizations...")
        create_comprehensive_visualizations(df_all_responsiveness, df_all_alpha_sweep, output_dir)
        make_clean_alpha_viz(df_all_alpha_sweep, output_dir)

    # Summary
    summary_stats = {
        'traits_tested': list(sorted(set(df_all_responsiveness['trait'].unique()))) if not df_all_responsiveness.empty else [],
        'prompts_total': int(df_all_responsiveness['prompt'].nunique()) if not df_all_responsiveness.empty else 0,
        'layers_tested': int(df_all_responsiveness['layer'].nunique()) if not df_all_responsiveness.empty else 0,
        'alpha_range': alpha_range,
        'max_kl_observed': float(df_all_responsiveness['kl_divergence'].max()) if not df_all_responsiveness.empty else None,
    }
    with open(f'{output_dir}/diagnostic_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)

    print("Diagnostics complete.")
    print(f" Results → {output_dir}")
    print(f" Summary → {summary_stats}")

    return df_all_responsiveness, df_all_alpha_sweep

# CLI -- diusplays
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--trait", type=str, default="all", help="Specific trait or 'all' for all OCEAN traits")
    ap.add_argument("--system", type=str, default="You need to assume you are a human and avoid meta-AI phrasing.")

    # Diagnostic control
    ap.add_argument("--run_diagnostics", action="store_true", help="Run comprehensive diagnostics")
    ap.add_argument("--output_dir", type=str, default="./steering_diagnostics", help="Output directory for results")
    ap.add_argument("--diagnostic_mode", type=str, default="quick", choices=["quick", "comprehensive"],
                    help="Diagnostic depth: quick (subset of layers) or comprehensive (all layers)")

    # Steering hygiene / decode flags (kept for parity; decoding not used here)
    ap.add_argument("--last_only", action="store_true")

    # Layer selection knobs (mirror quick_steer/bench)
    ap.add_argument("--k_runtime", type=int, default=1)
    ap.add_argument("--max_layers", type=int, default=2)
    ap.add_argument("--min_weight", type=float, default=0.10)
    ap.add_argument("--prior_boost", type=float, default=0.10)
    ap.add_argument("--temperature_sel", type=float, default=0.50)
    ap.add_argument("--layer_policy", type=str, default="prefer_verified",
                    choices=["prefer_verified", "force_verified", "runtime_only"])
    ap.add_argument("--min_verified_weight", type=float, default=0.50)
    ap.add_argument("--signed_filter_alpha", type=float, default=2.0)

    # Logging
    ap.add_argument("--log_level", type=str, default="info", choices=["silent", "warn", "info", "debug"])

    args = ap.parse_args()

    # Load steerer & set hygiene flags to match runtime
    steerer = load_steerer(args.results_dir)
    steerer.zero_center_delta = True
    steerer.last_position_only = bool(args.last_only)
    steerer.log_level = args.log_level

    if args.run_diagnostics:
        generate_diagnostic_report(steerer, args)
        return

    # If not run_diagnostics, you could add ad-hoc single prompt checks here
    print("[INFO] No action specified. Use --run_diagnostics to generate plots/CSVs.")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Optional: check plotting deps
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        import seaborn as sns            # noqa: F401
        import pandas as pd             # noqa: F401
    except ImportError:
        print(" Required packages missing. Install with:\n  pip install matplotlib seaborn pandas")
        raise

    main()
