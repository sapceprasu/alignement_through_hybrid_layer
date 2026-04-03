
from typing import Dict, List, Tuple
import numpy as np
import torch

# ---------- formatting helper ----------
def _format_for_chat(steerer, text: str, system: str = None) -> str:
    return steerer._format_prompt(text, use_chat_template=True, system=system)

# ---------- per-prompt Δlogits for each layer -
@torch.no_grad()
def delta_logits_norms_for_prompt(
    steerer,
    prompt: str,
    trait: str,
    intensity: float,
    system: str = None,
    layer_pool: List[int] = None,
    skip_calibration: bool = True,
    last_only: bool = False,
) -> Dict[int, float]:
    """
    Returns {layer: ||Δlogits||_2} at the next-token position (end of prompt)
    for each layer in layer_pool (or cfg.layer_range).
    """
    dev = steerer.device
    txt = _format_for_chat(steerer, prompt, system)

    # Keep RMS in sync when alpha_mode == "frac"
    if getattr(steerer, "alpha_mode", "abs") == "frac":
        steerer._measure_layer_rms(txt)

    enc = steerer.tok(txt, return_tensors="pt").to(dev)
    base = steerer.model(**enc).logits[:, -1, :].float()

    norms: Dict[int, float] = {}

    # Save & patch config
    orig_mode    = getattr(steerer, "steer_mode", "weighted")
    orig_range   = list(steerer.cfg.layer_range)
    orig_weights = getattr(steerer.cfg, "layer_weights", None)

    pool = layer_pool or orig_range
    
    try:
        steerer.steer_mode = "weighted"
        for L in pool:
            # mask so only this layer is active
            steerer.cfg.layer_weights = [1.0 if x == L else 0.0 for x in orig_range]
            steerer.cfg.layer_range   = list(orig_range)

            # IMPORTANT: ensure the probe uses same “last_only” behavior if your hooks depend on it
            steerer._register(trait, intensity, skip_calibration=skip_calibration)
            try:
                hooked = steerer.model(**enc).logits[:, -1, :].float()
            finally:
                steerer._clear()

            norms[L] = float(torch.norm(hooked - base, p=2).item())
    finally:
        # restore
        steerer.steer_mode = orig_mode
        steerer.cfg.layer_range = orig_range
        steerer.cfg.layer_weights = orig_weights

    return norms

# ---------- verified layer list ----------
def _verified_layers(steerer, trait: str) -> List[int]:
    t = trait.lower()
    if hasattr(steerer, "_trait_layers") and t in steerer._trait_layers:
        v = steerer._trait_layers[t]
        if isinstance(v, int):
            return [int(v)]
        return [int(x) for x in v]
    return []

# ---------- main selection: union + softmax weights ----------
@torch.no_grad()
def select_layers_for_prompt(
    steerer,
    prompt: str,
    trait: str,
    intensity: float,
    system: str = None,
    k_runtime: int = 1,
    prior_boost: float = 0.20,
    temperature: float = 0.50,
    max_layers: int = 2,
    min_weight: float = 0.10,
    layer_policy: str = "prefer_verified",  # diff options for diff condition "auto" | "prefer_verified" | "force_verified"
) -> Tuple[List[int], List[float], Dict[int, float]]:
    """
    Returns (layers, weights, all_norms) for this prompt/trait.

    layer_policy:
      - "auto": current hybrid behavior; verified layers only get a prior and can be dropped.
      - "prefer_verified": always include all verified layers first (up to max_layers),
                          then fill with top runtime layers not in verified.
      - "force_verified": only use verified layers (up to max_layers), no runtime filling.
    """
    # 1) verified set from artifacts
    V = _verified_layers(steerer, trait)

    # 2) runtime Δlogits norms across full allowed pool
    norms = delta_logits_norms_for_prompt(steerer, prompt, trait, intensity, system)
    sorted_L = sorted(norms, key=lambda L: norms[L], reverse=True)

    # ---- POLICY: force_verified -------------------------------------------------
    if layer_policy == "force_verified":
        if not V:
            # fall back to best single runtime if no verified is available
            chosen = sorted_L[:max_layers] if sorted_L else []
        else:
            chosen = V[:max_layers]
        if not chosen:
            return [], [], norms
        # uniform weights
        w = [1.0 / len(chosen)] * len(chosen)
        return chosen, w, norms

    # ---- POLICY: prefer_verified (default) -------------------------------------
    if layer_policy == "prefer_verified":
        chosen: List[int] = []
        if V:
            chosen.extend(V[:max_layers])
        # fill remaining slots with top runtime not in verified
        if len(chosen) < max_layers:
            for L in sorted_L:
                if L not in chosen:
                    chosen.append(L)
                if len(chosen) >= max_layers:
                    break
        if not chosen:
            return [], [], norms

        # weights via softmax with prior for verified
        mx = max([norms[L] for L in chosen]) if chosen else 1.0
        raw = []
        for L in chosen:
            s = norms[L] / (mx + 1e-9)
            if V and (L in V):
                s += prior_boost
            raw.append(s)
        r = torch.tensor(raw, dtype=torch.float32)
        logits = r / max(temperature, 1e-6)
        w = torch.softmax(logits, dim=0).tolist()

        # ensure minimal weight and renormalize
        for i in range(len(w)):
            if w[i] < min_weight:
                w[i] = min_weight
        ssum = sum(w)
        w = [wi / ssum for wi in w]
        return chosen, w, norms

    # ---- POLICY: auto (legacy hybrid) ------------------------------------------
    # pick top-K runtime layers not in V
    R = []
    for L in sorted_L:
        if L not in V:
            R.append(L)
        if len(R) >= k_runtime:
            break

    # candidate union (ensure non-empty)
    C = V + [x for x in R if x not in V]
    if not C and sorted_L:
        C = [sorted_L[0]]

    # normalized scores + prior for verified
    mx = max([norms[L] for L in C]) if C else 1.0
    raw = []
    for L in C:
        s = norms[L] / (mx + 1e-9)
        if L in V:
            s += prior_boost
        raw.append(s)

    # softmax with temperature -> weights
    r = torch.tensor(raw, dtype=torch.float32)
    logits = r / max(temperature, 1e-6)
    w = torch.softmax(logits, dim=0).tolist()

    # prune tiny weights
    C2, W2 = [], []
    for L, wi in zip(C, w):
        if wi >= min_weight:
            C2.append(L); W2.append(wi)
    if not C2:
        idx = int(np.argmax(w))
        C2 = [C[idx]]; W2 = [1.0]

    # cap count and renormalize
    if len(C2) > max_layers:
        idxs = np.argsort(W2)[::-1][:max_layers]
        C2 = [C2[i] for i in idxs]
        W2 = [W2[i] for i in idxs]
        ssum = sum(W2)
        W2 = [wi/ssum for wi in W2]

    return C2, W2, norms

# ---------- context manager to temporarily apply weights ----------
class SteerConfigPatch:
    """
    Temporarily switch to 'weighted' mode with (layers, weights), then restore.
    """
    def __init__(self, steerer, layers: List[int], weights: List[float]):
        self.s = steerer
        self.orig_mode    = getattr(steerer, "steer_mode", "weighted")
        self.orig_range   = list(steerer.cfg.layer_range)
        self.orig_weights = getattr(steerer.cfg, "layer_weights", None)
        self.layers = list(layers)
        self.weights = list(weights)

    def __enter__(self):
        self.s.steer_mode = "weighted"
        self.s.cfg.layer_range = self.layers
        self.s.cfg.layer_weights = self.weights
        return self.s

    def __exit__(self, exc_type, exc, tb):
        self.s.steer_mode = self.orig_mode
        self.s.cfg.layer_range = self.orig_range
        self.s.cfg.layer_weights = self.orig_weights





# # -*- coding: utf-8 -*-
# from typing import Dict, List, Tuple
# import numpy as np
# import torch

# def _format_for_chat(steerer, text: str, system: str = None) -> str:
#     return steerer._format_prompt(text, use_chat_template=True, system=system)

# @torch.no_grad()
# def _signed_probe_kln(
#     steerer,
#     enc,                   
#     trait: str,
#     layers: list,
#     weights: list,
#     alpha: float,
#     k: int = 50,
# ) -> float:
#     """Returns KL_topk_norm for a single signed probe."""
#     # Base logits
#     logits_base = steerer.model(**enc).logits[:, -1, :].float().squeeze(0)
#     p = torch.softmax(logits_base, dim=-1)

#     # Top-k indices from base
#     k = min(int(k), p.numel())
#     idx = torch.topk(p, k=k, dim=0).indices

#     # Steered logits
#     with SteerConfigPatch(steerer, layers, weights):
#         steerer._register(trait, alpha, skip_calibration=True)
#         try:
#             logits_steer = steerer.model(**enc).logits[:, -1, :].float().squeeze(0)
#         finally:
#             steerer._clear()
#     q = torch.softmax(logits_steer, dim=-1)

#     # Restrict & renormalize
#     eps = 1e-12
#     p_k = (p[idx] / p[idx].sum().clamp_min(eps)).clamp_min(eps)
#     q_k = (q[idx] / q[idx].sum().clamp_min(eps)).clamp_min(eps)

#     # KL and entropy
#     kl = torch.sum(p_k * (torch.log(p_k) - torch.log(q_k))).item()
#     H = float(-(p_k * torch.log(p_k)).sum().item())
#     return float(kl / H) if H > 0 else 0.0

# @torch.no_grad()
# def set_polarity_from_signed_probe(
#     steerer,
#     prompt: str,
#     trait: str,
#     layers: List[int],
#     weights: List[float],
#     system: str = None,
#     alpha_probe: float = 0.20,
#     k: int = 50,
# ) -> int:
#     """Compute and set polarity for THIS layer mix via signed top-k KL probe."""
#     txt = steerer._format_prompt(prompt, use_chat_template=True, system=system)
    
#     # Ensure RMS measurement for fractional scaling
#     if getattr(steerer, "alpha_mode", "abs") == "frac":
#         steerer._measure_layer_rms(txt)
    
#     enc = steerer.tok(txt, return_tensors="pt").to(steerer.device)

#     # Measure KL for both directions
#     kln_pos = _signed_probe_kln(steerer, enc, trait, layers, weights, +alpha_probe, k=k)
#     kln_neg = _signed_probe_kln(steerer, enc, trait, layers, weights, -alpha_probe, k=k)
    
#     # Determine sign: +1 if positive steering causes more change, else -1
#     sgn = +1 if kln_pos >= kln_neg else -1
    
#     # Set polarity override
#     steerer.polarity_override[trait] = sgn
    
#     print(f"[polarity-probe] trait={trait} layers={layers} "
#           f"kln(+probe)={kln_pos:.6f} kln(-probe)={kln_neg:.6f} -> sign={sgn:+d}")
#     return sgn

# @torch.no_grad()
# def _delta_logits_norms_single_prompt(
#     steerer,
#     formatted_text: str,
#     trait: str,
#     intensity: float
# ) -> Dict[int, float]:
#     """Measure delta logits norms for single prompt."""
#     if getattr(steerer, "alpha_mode", "abs") == "frac":
#         steerer._measure_layer_rms(formatted_text)

#     enc = steerer.tok(formatted_text, return_tensors="pt").to(steerer.device)
#     base_logits = steerer.model(**enc).logits[:, -1, :].float()
    
#     norms: Dict[int, float] = {}
#     orig_mode = getattr(steerer, "steer_mode", "weighted")
#     orig_range = list(steerer.cfg.layer_range)
#     orig_weights = getattr(steerer.cfg, "layer_weights", None)

#     try:
#         steerer.steer_mode = "weighted"
#         for L in orig_range:
#             # Test each layer individually
#             steerer.cfg.layer_weights = [1.0 if x == L else 0.0 for x in orig_range]
#             steerer._register(trait, intensity, skip_calibration=True)
#             try:
#                 steered_logits = steerer.model(**enc).logits[:, -1, :].float()
#             finally:
#                 steerer._clear()
#             norms[L] = float(torch.norm(steered_logits - base_logits, p=2).item())
#     finally:
#         # Restore original settings
#         steerer.steer_mode = orig_mode
#         steerer.cfg.layer_range = orig_range
#         steerer.cfg.layer_weights = orig_weights

#     return norms

# @torch.no_grad()
# def delta_logits_norms_for_prompt(
#     steerer,
#     prompt: str,
#     trait: str,
#     intensity: float,
#     system: str = None,
#     probes: int = 3
# ) -> Dict[int, float]:
#     """Average delta logits norms across multiple prompt variations."""
#     # Create multiple prompt variations for stability
#     variations = [
#         steerer._format_prompt(prompt, use_chat_template=True, system=system)
#     ]
    
#     if probes >= 2:
#         variations.append(steerer._format_prompt("Respond briefly.", use_chat_template=True, system=system))
#     if probes >= 3:
#         tail = (prompt or "")[:80]
#         variations.append(steerer._format_prompt(f"Write about: {tail}", use_chat_template=True, system=system))

#     # Accumulate norms across variations
#     accumulated_norms: Dict[int, float] = {}
#     for variation in variations:
#         norms = _delta_logits_norms_single_prompt(steerer, variation, trait, intensity)
#         for L, value in norms.items():
#             accumulated_norms[L] = accumulated_norms.get(L, 0.0) + value

#     # Average the results
#     num_variations = len(variations)
#     return {L: value / num_variations for L, value in accumulated_norms.items()}

# def _verified_layers(steerer, trait: str) -> List[int]:
#     """Get verified layers for a trait from steerer artifacts."""
#     trait_lower = trait.lower()
#     if hasattr(steerer, "_trait_layers") and trait_lower in steerer._trait_layers:
#         layers = steerer._trait_layers[trait_lower]
#         if isinstance(layers, int):
#             return [int(layers)]
#         return [int(x) for x in layers]
#     return []

# @torch.no_grad()
# def select_layers_for_prompt(
#     steerer,
#     prompt: str,
#     trait: str,
#     intensity: float,
#     system: str = None,
#     *,
#     k_runtime: int = 3,
#     prior_boost: float = 0.05,
#     temperature: float = 0.80,
#     max_layers: int = 2,
#     min_weight: float = 0.10,
# ) -> Tuple[List[int], List[float], Dict[int, float]]:
#     """Hybrid layer selection combining verified layers with runtime responsiveness."""
    
#     # Get verified layers
#     verified_layers = _verified_layers(steerer, trait)
    
#     # Get runtime responsiveness norms
#     runtime_norms = delta_logits_norms_for_prompt(
#         steerer, prompt, trait, intensity, system=system, probes=3
#     )
    
#     if not runtime_norms:
#         # Fallback to verified layers or first layer
#         if verified_layers:
#             return verified_layers[:1], [1.0], {}
#         return [list(steerer.cfg.layer_range)[0]], [1.0], {}

#     # Sort layers by responsiveness
#     sorted_layers = sorted(runtime_norms, key=lambda L: runtime_norms[L], reverse=True)
    
#     # Select top runtime layers (excluding verified ones)
#     runtime_candidates = [L for L in sorted_layers if L not in verified_layers][:max(1, k_runtime)]

#     # Combine verified and runtime candidates
#     if verified_layers:
#         candidate_layers = [verified_layers[0]] + (runtime_candidates if runtime_candidates else [])
#     else:
#         candidate_layers = sorted_layers[:max_layers]

#     # Calculate weights using softmax with temperature
#     max_norm = max(runtime_norms[L] for L in candidate_layers)
#     raw_scores = []
#     for i, layer in enumerate(candidate_layers):
#         score = runtime_norms[layer] / (max_norm + 1e-9)
#         # Boost verified layer score
#         if verified_layers and i == 0 and layer == verified_layers[0]:
#             score += prior_boost
#         raw_scores.append(score)

#     # Apply temperature and softmax
#     logits = torch.tensor(raw_scores, dtype=torch.float32) / max(temperature, 1e-6)
#     weights = torch.softmax(logits, dim=0).tolist()

#     # Filter by minimum weight
#     final_layers, final_weights = [], []
#     for i, (layer, weight) in enumerate(zip(candidate_layers, weights)):
#         if (verified_layers and i == 0 and layer == verified_layers[0]) or weight >= min_weight:
#             final_layers.append(layer)
#             final_weights.append(weight)

#     # Ensure we have at least 2 layers if possible
#     if len(final_layers) == 1 and max_layers >= 2 and len(candidate_layers) >= 2:
#         final_layers.append(candidate_layers[1])
#         final_weights.append(max(1e-4, weights[1]))

#     # Limit to max_layers
#     if len(final_layers) > max_layers:
#         if verified_layers and final_layers[0] == verified_layers[0]:
#             # Keep verified layer, select top runtime layers
#             runtime_pairs = list(zip(final_layers[1:], final_weights[1:]))
#             runtime_pairs.sort(key=lambda x: x[1], reverse=True)
#             final_layers = [final_layers[0]] + [L for L, _ in runtime_pairs[:max_layers - 1]]
#             final_weights = [final_weights[0]] + [w for _, w in runtime_pairs[:max_layers - 1]]
#         else:
#             # Select top layers by weight
#             indices = np.argsort(final_weights)[::-1][:max_layers]
#             final_layers = [final_layers[i] for i in indices]
#             final_weights = [final_weights[i] for i in indices]

#     # Normalize weights
#     total_weight = sum(final_weights)
#     if total_weight > 0:
#         final_weights = [w / total_weight for w in final_weights]
#     else:
#         final_weights = [1.0 / len(final_weights)] * len(final_weights)

#     # Optional: Signed probe diagnostic
#     if getattr(steerer.cfg, "log_signed_probe", True) and final_layers:
#         try:
#             txt = steerer._format_prompt(prompt, use_chat_template=True, system=system)
#             enc = steerer.tok(txt, return_tensors="pt").to(steerer.device)
            
#             if getattr(steerer, "alpha_mode", "abs") == "frac":
#                 steerer._measure_layer_rms(txt)

#             kln_pos = _signed_probe_kln(steerer, enc, trait, final_layers, final_weights, +0.20, k=50)
#             kln_neg = _signed_probe_kln(steerer, enc, trait, final_layers, final_weights, -0.20, k=50)
            
#             print(f"[layer-probe] {trait} layers={final_layers} "
#                   f"weights={[round(w, 3) for w in final_weights]} "
#                   f"kln(+)= {kln_pos:.6f} kln(-)= {kln_neg:.6f}")
#         except Exception:
#             pass

#     # Auto-set polarity if configured
#     if (getattr(steerer.cfg, "auto_set_polarity_from_probe", True) and final_layers):
#         try:
#             set_polarity_from_signed_probe(steerer, prompt, trait, final_layers, final_weights, 
#                                          system=system, alpha_probe=0.20, k=50)
#         except Exception as e:
#             print(f"[warning] Failed to set polarity: {e}")

#     return final_layers, final_weights, runtime_norms

# class SteerConfigPatch:
#     """Context manager for temporary steering configuration."""
#     def __init__(self, steerer, layers: List[int], weights: List[float]):
#         self.steerer = steerer
#         self.layers = layers
#         self.weights = weights
#         self.original_mode = getattr(steerer, "steer_mode", "weighted")
#         self.original_range = list(steerer.cfg.layer_range)
#         self.original_weights = getattr(steerer.cfg, "layer_weights", None)

#     def __enter__(self):
#         self.steerer.steer_mode = "weighted"
#         self.steerer.cfg.layer_range = self.layers
#         self.steerer.cfg.layer_weights = self.weights
#         return self.steerer

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.steerer.steer_mode = self.original_mode
#         self.steerer.cfg.layer_range = self.original_range
#         self.steerer.cfg.layer_weights = self.original_weights