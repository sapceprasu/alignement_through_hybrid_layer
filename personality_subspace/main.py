# -*- coding: utf-8 -*-
import os, pickle, json, time
from typing import Dict
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os 
# import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

from .config import Config
from .utils import set_seeds, json_dump, CSVLogger
from .data import PersonalityDataset
from .extractor import MultiLayerActivationExtractor
from .optimizer import LayerWeightOptimizer
from .pas import PASBaseline
from .subspace import WeightedPersonalitySubspace
from .evaluators.evaluate import Evaluator
from .steering import PersonalitySteerer
from .layer_search import verify_best_layers
from .justify_layers import run_layer_justification
# from .direction_scrubber import DirectionScrubber, ScrubConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# import torch
# import numpy as np

def scrub_and_verify(results_directory, trait_dirs, threshold=0.4):
    """
    Safe-Scrub with MAGNITUDE PRESERVATION.
    1. Extracts Original Magnitude.
    2. Scrubs direction (Unit Norm).
    3. Restores Original Magnitude (So steering strength isn't lost).
    """
    print(f"\n{'='*80}")
    print(f"[VERIFICATION] SAFE ANCHOR SCRUBBING (Preserving Magnitude)")
    print(f"{'='*80}")
    
    keys = list(trait_dirs.keys())
    
    # 1. Capture Anchors AND Magnitudes
    anchors = {}
    magnitudes = {} 
    
    for k in keys:
        raw_v = torch.tensor(trait_dirs[k]["direction"]).float()
        mag = torch.norm(raw_v)
        magnitudes[k] = mag.item() # Save it for later
        
        # Normalize for the math part
        anchors[k] = raw_v / mag

    # 2. Prepare Working Copy
    cleaned_vecs = {k: anchors[k].clone() for k in keys}
    stats = []

    # 3. The Safe Loop (Direction Only)
    for target in keys:
        current_vec = cleaned_vecs[target]
        
        for interferer in keys:
            if target == interferer: continue
            
            overlap = torch.dot(current_vec, anchors[interferer]).item()
            
            if abs(overlap) > threshold:
                print(f"  -> Detected overlap: '{target}' vs '{interferer}' (Corr: {overlap:.2f})")
                
                # Projection subtraction
                proj = overlap * anchors[interferer]
                candidate_vec = current_vec - proj
                candidate_vec = candidate_vec / torch.norm(candidate_vec) # Keep unit for now
                
                # Retention Check
                retention = torch.dot(candidate_vec, anchors[target]).item()
                
                status = "SKIPPED"
                if retention < 0.60:
                    print(f"     [ABORT] Retention too low ({retention:.2f}).")
                    final_corr = overlap
                    status = "ABORTED"
                else:
                    print(f"     [OK] Scrub applied. Retention: {retention:.2f}")
                    current_vec = candidate_vec
                    final_corr = torch.dot(current_vec, anchors[interferer]).item()
                    status = "SUCCESS"

                stats.append({
                    "Target": target,
                    "Interferer": interferer,
                    "Corr_Before": overlap,
                    "Corr_After": final_corr,
                    "Retention": retention,
                    "Status": status
                })
        
        cleaned_vecs[target] = current_vec

    # 4. Save and Report
    # CRITICAL FIX: RE-APPLY MAGNITUDE 
    new_dirs = trait_dirs.copy()
    for k in keys:
        # Take the Clean Direction * Original Magnitude
        final_v = cleaned_vecs[k] * magnitudes[k] 
        
        new_dirs[k] = trait_dirs[k].copy()
        new_dirs[k]["direction"] = final_v.numpy()

    # Save Stats
    json_dump(stats, os.path.join(results_directory, "direction_scrubbing_stats.json"))
    
    return new_dirs

# -helpre
def _canonicalize_trait_signs(weighted_dirs, activs, layer_range, layer_weights, trait_names):
    """
    For each trait, compute a reference vector μ_high - μ_low aggregated across layers (weighted),
    and flip the learned direction if its dot product with the reference is negative.
    Returns a dict of {trait: +1/-1}.
    """
    signs = {}
    # layer_weights may be np.ndarray aligned with layer_range
    wlist = list(layer_weights) if hasattr(layer_weights, "__len__") else [1.0] * len(layer_range)

    for trait in trait_names:
        # 1) learned direction (same space as activs vectors)
        dir_vec = weighted_dirs.get(trait, {}).get("direction", None)
        if dir_vec is None:
            continue
        dir_vec = np.asarray(dir_vec, dtype=np.float32)

        # 2) build reference μ_high - μ_low aggregated over layers (weighted)
        ref = None
        have_any = False
        for idx, L in enumerate(layer_range):
            layer = activs.get(L, {})
            hi_key = f"{trait}_high"; lo_key = f"{trait}_low"
            if hi_key in layer and lo_key in layer:
                Xh = np.asarray(layer[hi_key], dtype=np.float32)
                Xl = np.asarray(layer[lo_key], dtype=np.float32)
                if Xh.size == 0 or Xl.size == 0: 
                    continue
                mu = Xh.mean(axis=0) - Xl.mean(axis=0)      # [d]
                wr = float(wlist[idx]) * mu
                ref = wr if ref is None else (ref + wr)
                have_any = True

        if not have_any or ref is None:
            # no reference found; leave as-is
            signs[trait] = +1
            continue

        ref = ref.astype(np.float32, copy=False)
        # 3) flip if misaligned
        dot = float(np.dot(dir_vec.ravel(), ref.ravel()))
        if np.isnan(dot) or np.isinf(dot):
            signs[trait] = +1
            continue

        if dot < 0.0:
            weighted_dirs[trait]["direction"] = (-dir_vec).astype(np.float32)
            signs[trait] = -1
        else:
            signs[trait] = +1

    return signs
# --- END OF HELPER ---


def run_pipeline(cfg: Config):
    set_seeds(cfg.seed)
    cfg.ensure_dirs()
    # breakpoint()
    # 0) dataset
    ds = PersonalityDataset(cfg)
    json_dump({"trait_distribution": ds.trait_levels, "total_samples": len(ds.data)},
              os.path.join(cfg.results_dir, "dataset_analysis.json"))
    balanced = ds.get_balanced()
    expected_min = 500 # adjust to your expectation (e.g., 5000)
    print(f"[INFO] balanced dataset size: {len(balanced)}")
    if len(balanced) < expected_min:
        raise RuntimeError(f"Dataset too small: {len(balanced)} (<{expected_min}). "
                        f"Check cfg.max_samples_per_group and data distribution.")

    print(f"[INFO] balanced dataset size: {len(balanced)}")

  # 1) activations
    extractor = MultiLayerActivationExtractor(cfg, results_dir=cfg.results_dir)
    activs = extractor.extract(balanced)

    # Joint standardize per (layer, trait)
    for L in cfg.layer_range:
        for trait in cfg.trait_mapping.values():
            hi_key, lo_key = f"{trait}_high", f"{trait}_low"
            if hi_key in activs[L] and lo_key in activs[L]:
                Xh, Xl = extractor.joint_standardize_layer_trait(activs[L][hi_key], activs[L][lo_key])
                activs[L][hi_key], activs[L][lo_key] = Xh, Xl

   

  
    print(f"\n[AUTO-SELECT] Ranking layers by separation score (Top {cfg.top_n_layers})...")
    
    layer_scores = {}

    for L in cfg.layer_range:
        # Safety check for missing layers
        if L not in activs: continue

        layer_distances = []
        for trait in cfg.trait_mapping.values():
            hi = activs[L][f"{trait}_high"]
            lo = activs[L][f"{trait}_low"]
            
            # Centroids & Distance
            mu_high = np.mean(hi, axis=0)
            mu_low  = np.mean(lo, axis=0)
            dist = np.linalg.norm(mu_high - mu_low)
            layer_distances.append(float(dist)) # Convert to float for JSON safety
        
        # Average score for this layer
        layer_scores[L] = np.mean(layer_distances)

    # 1. Sort layers by score (Highest to Lowest)
    sorted_layers = sorted(layer_scores.items(), key=lambda item: item[1], reverse=True)
    
    # 2. Slice the Top N
    top_n = cfg.top_n_layers
    best_layers_tuples = sorted_layers[:top_n]
    best_layers_indices = sorted([L for L, score in best_layers_tuples])
    
    # 3. Update Config
    cfg.layer_range = best_layers_indices

    # 4. SAVE TO JSON (The part you asked for!)
    selection_data = {
        "strategy": f"Top-{top_n} Selection",
        "selected_layers": cfg.layer_range,
        "selected_scores": {L: round(score, 4) for L, score in best_layers_tuples},
        "all_layer_scores": {L: round(score, 4) for L, score in sorted_layers}
    }
    
    json_path = os.path.join(cfg.results_dir, "layer_selection_scores.json")
    json_dump(selection_data, json_path)
    
    print(f"[INFO] Top {top_n} Layers: {cfg.layer_range}")
    print(f"[SAVE] Layer selection report -> {json_path}")

    # Save activations...
    with open(os.path.join(cfg.results_dir, "multi_layer_activations.pkl"), "wb") as f:
        pickle.dump(activs, f)


    # 2) layer weights
    opt = LayerWeightOptimizer(cfg)
    weights = opt.optimize(activs)
    json_dump(dict(zip([str(L) for L in cfg.layer_range], weights.tolist())),
              os.path.join(cfg.results_dir, "layer_weights.json"))
    print(f"[INFO] weights: {dict(zip(cfg.layer_range, np.round(weights,4)))}")

    # 3) PAS baseline
    pas = PASBaseline(cfg)
    best_layers, pas_dirs = pas.run(activs)
    json_dump(best_layers, os.path.join(cfg.results_dir, "pas_best_layers.json"))
    with open(os.path.join(cfg.results_dir, "pas_directions.pkl"), "wb") as f:
        pickle.dump(pas_dirs, f)



    # 4) weighted directions + subspace
    sub = WeightedPersonalitySubspace(cfg, weights)
    weighted_dirs = sub.compute_weighted_directions(activs)

   
    trait_names = list(cfg.trait_mapping.values())
    trait_signs = _canonicalize_trait_signs(weighted_dirs, activs, list(cfg.layer_range), weights, trait_names)
    json_dump(trait_signs, os.path.join(cfg.results_dir, "trait_signs.json"))
    print(f"[SIGN] canonical signs ( +1 high / -1 flipped ): {trait_signs}")

    # fixing the overlap in the directions and making sure that 2 direcctions donot 
    # weighted_dirs = scrub_and_verify(cfg.results_dir, weighted_dirs, threshold=0.4)
    # scrub_cfg = ScrubConfig(
    #     overlap_threshold=0.40,
    #     step_retention_floor=0.70,
    #     final_retention_floor=0.75,
    #     max_passes=2,
    #     use_clean_interferers=True,
    #     deterministic_order=True,
    #     out_subdir="dir_scrub"
    # )

    # scrubber = DirectionScrubber(cfg.results_dir, scrub_cfg)
    # weighted_dirs = scrubber.scrub(weighted_dirs)

    # print(f"[SCRUB] report written to: {os.path.join(cfg.results_dir, scrub_cfg.out_subdir)}")



    subspace, evr = sub.build(weighted_dirs)

    np.save(os.path.join(cfg.results_dir, "subspace.npy"), subspace)
    np.save(os.path.join(cfg.results_dir, "explained_variance_ratio.npy"), evr)
    json_dump({"explained_variance_ratio": evr.tolist(),
               "cumulative": np.cumsum(evr).tolist()},
              os.path.join(cfg.results_dir, "subspace_variance.json"))
    print(f"[INFO] EVR: {np.round(evr,4)} (cum {np.round(np.cumsum(evr),4)})")

    # 5) evaluation
    ev = Evaluator(cfg)
    align = ev.alignment(weighted_dirs, subspace, pas_dirs)
    cls   = ev.classification(activs, subspace, weights, best_layers)
    json_dump(align, os.path.join(cfg.results_dir, "alignment_metrics.json"))
    json_dump(cls,   os.path.join(cfg.results_dir, "classification_metrics.json"))

    # 6) artifacts 
    artifacts = {
        "subspace": subspace,
        "trait_directions": {t: v["direction"] for t, v in weighted_dirs.items()},
        "layer_weights": weights.tolist(),
        "trait_signs": trait_signs,
        "config": {
            "model_name": cfg.model_name,
            "layer_range": cfg.layer_range,
            "n_components": cfg.n_components,
        }
    }
    with open(os.path.join(cfg.results_dir, "artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    # 7) summary table
    print("\n=== Key Results Summary ===")
    print(f"{'Trait':<16} {'PAS-Layer':<9} {'AlignCos':<8} {'PASAlignCos':<11} {'PASAcc':<8} {'SubAcc':<8}")
    for trait in cfg.trait_mapping.values():
        a = align.get(trait, {})
        c = cls.get(trait, {})
        print(f"{trait:<16} {str(best_layers.get(trait,'-')):<9} "
              f"{a.get('cosine_similarity','-')!s:<8} {a.get('pas_cosine_similarity','-')!s:<11} "
              f"{c.get('pas_accuracy','-')!s:<8} {c.get('subspace_accuracy','-')!s:<8}")

    print("\n[OK] pipeline complete. results →", cfg.results_dir)


    # 3b) Verify best injection layers end-to-end with steerer + probes
    #     (fast: tiny probe prompts, 1-step decodes)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto").eval()
    tok   = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Load steerer from freshly produced artifacts
    with open(os.path.join(cfg.results_dir, "artifacts.pkl"), "rb") as f:
        arts = pickle.load(f)
    from .steering import PersonalitySteerer
    steerer = PersonalitySteerer(model, tok, arts["subspace"], arts["trait_directions"], cfg)
    verified = verify_best_layers(cfg, steerer)  # writes layer_verified.json
    best_layers = run_layer_justification(cfg, steerer)
    print("[LAYER-JUSTIFY] Best layers:", best_layers)


# personality_subspace/main.py
def load_steerer(results_dir: str, model_name: str = None) -> PersonalitySteerer:
    with open(os.path.join(results_dir, "artifacts.pkl"), "rb") as f:
        arts = pickle.load(f)
    mname = model_name or arts["config"]["model_name"]
    model = AutoModelForCausalLM.from_pretrained(
        mname, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    tok = AutoTokenizer.from_pretrained(mname)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = Config(
        model_name=mname,
        layer_range=arts["config"]["layer_range"],
        n_components=arts["config"]["n_components"],
    )
    cfg.layer_weights = arts.get("layer_weights")
    cfg.results_dir = results_dir

    steerer = PersonalitySteerer(model, tok, arts["subspace"], arts["trait_directions"], cfg)

    # Prefer verified layers if available; else PAS
    verified_path = os.path.join(results_dir, "layer_verified.json")
    pas_path = os.path.join(results_dir, "pas_best_layers.json")
    try:
        if os.path.exists(verified_path):
            with open(verified_path, "r") as f:
                steerer._trait_layers = json.load(f)
                print("layers to steer verified paths", steerer._trait_layers)
        elif os.path.exists(pas_path):
            with open(pas_path, "r") as f:
                steerer._trait_layers = json.load(f)
                print("layers to steer pas paths", steerer._trait_layers)
    except Exception:
        pass


    return steerer


if __name__ == "__main__":
    cfg = Config()
    run_pipeline(cfg)
