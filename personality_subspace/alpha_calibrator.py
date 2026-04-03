
from typing import Dict, Tuple, List, Optional
import math
import numpy as np
import torch

LN2 = math.log(2.0)

@torch.no_grad()
def _format_for_chat(steerer, text: str, system: str = None) -> str:
    return steerer._format_prompt(text, use_chat_template=True, system=system)

@torch.no_grad()
def _first_token_logits(steerer, formatted_prompt: str, do_measure_frac: bool = True) -> torch.Tensor:
    # Keep parity with generation path when alpha_mode == 'frac'
    if do_measure_frac and getattr(steerer, "alpha_mode", "abs") == "frac":
        steerer._measure_layer_rms(formatted_prompt)
    enc = steerer.tok(formatted_prompt, return_tensors="pt").to(steerer.device)
    out = steerer.model(**enc)
    return out.logits[:, -1, :].float().squeeze(0)


def _softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=-1)

def _entropy_nats(p: torch.Tensor) -> float:
    p = p.clamp_min(1e-12)
    return float((-p * p.log()).sum().item())

def _entropy_bits(p: torch.Tensor) -> float:
    return _entropy_nats(p) / LN2

def _topk_idx_from_probs(p: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, p.numel())
    return torch.topk(p, k=k, dim=0).indices

def _kl_nats(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    return float((p * (p.log() - q.log())).sum().item())

@torch.no_grad()
def _metrics_topk_norm(
    base_logits: torch.Tensor,
    steer_logits: torch.Tensor,
    k: int
) -> Dict[str, float]:
    """Compute normalized Top-k KL and normalized Top-k Δlogits (L2_z), plus ΔH in bits."""
    base_probs  = _softmax(base_logits)
    steer_probs = _softmax(steer_logits)

    idx = _topk_idx_from_probs(base_probs, k)
    # Renormalize on Top-k support
    p_k = (base_probs[idx]  / base_probs[idx].sum().clamp_min(1e-12)).clamp_min(1e-12)
    q_k = (steer_probs[idx] / steer_probs[idx].sum().clamp_min(1e-12)).clamp_min(1e-12)

    kl_topk_nats = _kl_nats(p_k, q_k)
    H_topk_nats  = _entropy_nats(p_k)
    kl_topk_norm = float(kl_topk_nats / max(H_topk_nats, 1e-12))

    # Δlogits on same support, normalized by base std over that support
    delta_topk = (steer_logits[idx] - base_logits[idx]).float()
    l2_topk = float(torch.norm(delta_topk, p=2).item())
    std_base = float(torch.std(base_logits[idx]).item())
    L2_z = float(l2_topk / (math.sqrt(len(idx)) * max(std_base, 1e-12)))

    # Entropy change (full support) in bits
    dH_bits = _entropy_bits(steer_probs) - _entropy_bits(base_probs)

    return dict(
        kl_topk_norm=kl_topk_norm,
        l2_z=L2_z,
        deltaH_bits=float(dH_bits),
        kl_topk_nats=float(kl_topk_nats),
        H_topk_nats=float(H_topk_nats)
    )



@torch.no_grad()
def _measure_once(
    steerer, formatted_prompt: str, trait: str, alpha: float, layers: List[int], weights: List[float]
) -> torch.Tensor:
    from .layer_selector import SteerConfigPatch
    with SteerConfigPatch(steerer, layers, weights):
        # MEASURE FIRST on a clean pass
        if getattr(steerer, "alpha_mode", "abs") == "frac":
            steerer._measure_layer_rms(formatted_prompt)
        # Install hook; DON'T re-measure again
        steerer._register(trait, alpha, skip_calibration =True )
        try:
            logits = _first_token_logits(steerer, formatted_prompt, do_measure_frac=False)
        finally:
            steerer._clear()
    return logits

def _pick_alpha_from_slopes(
    target_kln: float,
    slope_kln: float,
    target_l2z: float,
    slope_l2z: float,
    alpha_min: float,
    alpha_max: float,
) -> float:
    """
    Pick a conservative alpha that satisfies BOTH targets, using the measured slopes.

    - target_kln: desired normalized Top-k KL (unitless)
    - slope_kln:  measured (kln / alpha_probe)
    - target_l2z: desired normalized Δlogits magnitude
    - slope_l2z:  measured (l2z / alpha_probe)
    We take the smaller alpha that reaches either target, then clamp to [alpha_min, alpha_max].
    """
    eps = 1e-9
    a_from_kln = target_kln / max(slope_kln, eps)
    a_from_l2z = target_l2z / max(slope_l2z, eps)
    a = min(a_from_kln, a_from_l2z)
    return float(np.clip(a, alpha_min, alpha_max))

@torch.no_grad()
def calibrate_alpha_composite_for_prompt(
    steerer,
    prompt_text: str,
    trait: str,
    *,
    system: Optional[str],
    layers: List[int],
    weights: List[float],
    k: int = 50,
    alpha_probe: float = 0.05,
    target_kln: float =  0.20,   #0.10,      # normalized KL target
    target_l2z: float =  0.8,       #1.00      # normalized Δlogits target
    entropy_bits_cap: float =  0.05,   #0.15,
    alpha_min: float = 0.30,
    alpha_max: float = 4.00,
    retest_entropy_cap: bool = True
) -> Tuple[float, float, Dict]:
    """
    Returns (alpha_pos, alpha_neg, diagnostics).
    """
    formatted = _format_for_chat(steerer, prompt_text, system=system)
    base_logits = _first_token_logits(steerer, formatted, do_measure_frac=True)

    def slopes(dir_sign: int):
        probe_alpha = dir_sign * alpha_probe
        steer_logits = _measure_once(steerer, formatted, trait, probe_alpha, layers, weights)
        m = _metrics_topk_norm(base_logits, steer_logits, k)
        # Approx slopes per unit alpha
        s_kln = m["kl_topk_norm"] / max(alpha_probe, 1e-12)
        s_l2z = m["l2_z"]         / max(alpha_probe, 1e-12)
        return m, s_kln, s_l2z

    # +probe
    m_pos, s_kln_pos, s_l2z_pos = slopes(+1)
    # -probe
    m_neg, s_kln_neg, s_l2z_neg = slopes(-1)

    # Candidates (per direction) from both metrics; pick conservative (min)
    a_pos = _pick_alpha_from_slopes(target_kln, s_kln_pos, target_l2z, s_l2z_pos, alpha_min, alpha_max)
    a_neg = _pick_alpha_from_slopes(target_kln, s_kln_neg, target_l2z, s_l2z_neg, alpha_min, alpha_max)

    # Optional re-test to respect entropy cap
    def enforce_entropy_cap(alpha_eff: float) -> Tuple[float, float]:
        if not retest_entropy_cap:
            return alpha_eff, 0.0
        steer_logits = _measure_once(steerer, formatted, trait, alpha_eff, layers, weights)
        m = _metrics_topk_norm(base_logits, steer_logits, k)
        dH = abs(m["deltaH_bits"])
        if dH <= entropy_bits_cap:
            return alpha_eff, dH
        # Scale alpha down linearly to fit cap
        scale = max(entropy_bits_cap / max(dH, 1e-9), 1e-3)
        alpha_new = max(alpha_min, min(alpha_eff * scale, alpha_max))
        return alpha_new, dH

    a_pos_final, dH_pos_meas = enforce_entropy_cap(a_pos)
    a_neg_final, dH_neg_meas = enforce_entropy_cap(a_neg)

    diags = dict(
        prompt=prompt_text,
        trait=trait,
        system=system or "",
        k=k,
        alpha_probe=alpha_probe,
        targets=dict(kln=target_kln, l2z=target_l2z, entropy_bits_cap=entropy_bits_cap),
        probe_metrics=dict(pos=m_pos, neg=m_neg),
        slopes=dict(pos=dict(kln=s_kln_pos, l2z=s_l2z_pos),
                    neg=dict(kln=s_kln_neg, l2z=s_l2z_neg)),
        alpha_candidates=dict(pos=a_pos, neg=a_neg),
        alpha_final=dict(pos=a_pos_final, neg=a_neg_final),
        entropy_measured_bits=dict(pos=dH_pos_meas, neg=dH_neg_meas),
        layers=list(layers),
        weights=[float(w) for w in weights],
    )
    return float(a_pos_final), float(a_neg_final), diags

@torch.no_grad()
def calibrate_alpha_composite_for_prompt_bank(
    steerer,
    main_prompt: str,
    trait: str,
    *,
    system: Optional[str],
    layers: List[int],
    weights: List[float],
    bank_prompts: Optional[List[str]] = None,
    k: int = 50,
    alpha_probe: float = 0.05,
    target_kln: float = 0.01,
    target_l2z: float = 0.3,
    entropy_bits_cap: float = 0.05,
    alpha_min: float = 0.01,
    alpha_max: float = 2.00,
) -> Tuple[float, float, Dict]:
    """
    Median α over a small bank for robustness. Includes the main prompt by default.
    """
    prompts = [main_prompt]
    if not bank_prompts:
        # fall back to your config's small neutral bank if available
        try:
            bank_prompts = list(getattr(steerer.cfg.layer_search, "probe_prompts", []))
        except Exception:
            bank_prompts = []
    prompts.extend(p for p in bank_prompts if p.strip())

    alphas_pos, alphas_neg, recs = [], [], []

    for p in prompts:
        ap, an, d = calibrate_alpha_composite_for_prompt(
            steerer, p, trait, system=system, layers=layers, weights=weights,
            k=k, alpha_probe=alpha_probe,
            target_kln=target_kln, target_l2z=target_l2z,
            entropy_bits_cap=entropy_bits_cap,
            alpha_min=alpha_min, alpha_max=alpha_max,
        )
        alphas_pos.append(ap); alphas_neg.append(an); recs.append(d)

    a_pos = float(np.median(alphas_pos))
    a_neg = float(np.median(alphas_neg))
    summary = dict(
        prompts=prompts,
        per_prompt=recs,
        median_alpha=dict(pos=a_pos, neg=a_neg),
        all_alpha_pos=alphas_pos,
        all_alpha_neg=alphas_neg,
    )
    return a_pos, a_neg, summary
