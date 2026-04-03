import argparse, os, sys, json, math, time, csv, random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.nn import functional as F

from .main import load_steerer
from .layer_selector import select_layers_for_prompt, SteerConfigPatch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _format_for_chat(steerer, text: str, system: Optional[str] = None) -> str:
    return steerer._format_prompt(text, use_chat_template=True, system=system)


def _kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0)
    q = q.clamp(min=eps, max=1.0)
    return torch.sum(p * (torch.log(p) - torch.log(q)))


@torch.no_grad()
def _next_token_kl(
    steerer,
    prompt: str,
    trait: str,
    alpha_eff: float,
    sign: int,
    system: Optional[str] = None,
) -> float:
    """Next-token KL between base and steered (with intensity normalized by steer_gain)."""
    txt = _format_for_chat(steerer, prompt, system)
    if getattr(steerer, "alpha_mode", "abs") == "frac":
        steerer._measure_layer_rms(txt)

    enc = steerer.tok(txt, return_tensors="pt").to(steerer.device)
    logits_base = steerer.model(**enc).logits[:, -1, :].float()
    p0 = torch.softmax(logits_base, dim=-1).squeeze(0)

    intensity = (sign * alpha_eff) / float(steerer.steer_gain)
    steerer._register(trait, intensity)
    try:
        logits_steer = steerer.model(**enc).logits[:, -1, :].float()
        p1 = torch.softmax(logits_steer, dim=-1).squeeze(0)
    finally:
        steerer._clear()

    return float(_kl_div(p0, p1).item())


@torch.no_grad()
def _base_topk_entropy_bits(steerer, prompt: str, system: Optional[str], k: int = 50) -> float:
    """Entropy (in bits) of base next-token distribution restricted/renormalized to top-k of base."""
    txt = _format_for_chat(steerer, prompt, system)
    enc = steerer.tok(txt, return_tensors="pt").to(steerer.device)
    logits = steerer.model(**enc).logits[:, -1, :].float().squeeze(0)
    p = torch.softmax(logits, dim=-1)
    k = min(k, p.numel())
    idx = torch.topk(p, k=k).indices
    q = (p[idx] / (p[idx].sum().clamp_min(1e-12))).clamp_min(1e-12)
    # entropy in bits
    H = -torch.sum(q * torch.log2(q)).item()
    return float(H)


@torch.no_grad()
def _deterministic_generate(steerer, prompt: str, trait: str, alpha_eff: float, sign: int,
                            *, system: Optional[str], max_new_tokens: int) -> str:
    """Greedy-ish generation for micro PPL measurements (do_sample=False)."""
    txt = _format_for_chat(steerer, prompt, system)
    if getattr(steerer, "alpha_mode", "abs") == "frac":
        steerer._measure_layer_rms(txt)
    enc = steerer.tok(txt, return_tensors="pt").to(steerer.device)
    eos = steerer.tok.eos_token_id
    pad = steerer.tok.pad_token_id or eos

    if alpha_eff == 0.0:
        out = steerer.model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                     eos_token_id=eos, pad_token_id=pad, return_dict_in_generate=False)
    else:
        steerer._register(trait, (sign * alpha_eff) / float(steerer.steer_gain))
        try:
            out = steerer.model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                         eos_token_id=eos, pad_token_id=pad, return_dict_in_generate=False)
        finally:
            steerer._clear()
    gen = steerer.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()


@torch.no_grad()
def _sampled_generate(steerer, prompt: str, trait: str, alpha_eff: float, sign: int,
                      *, system: Optional[str], max_new_tokens: int,
                      temperature: float, top_p: float, top_k: int, repetition_penalty: float) -> str:
    """Sampling decode for final outputs."""
    txt = _format_for_chat(steerer, prompt, system)
    if getattr(steerer, "alpha_mode", "abs") == "frac":
        steerer._measure_layer_rms(txt)
    enc = steerer.tok(txt, return_tensors="pt").to(steerer.device)
    eos = steerer.tok.eos_token_id
    pad = steerer.tok.pad_token_id or eos

    if alpha_eff == 0.0:
        out = steerer.model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                                     repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
                                     eos_token_id=eos, pad_token_id=pad, return_dict_in_generate=False)
    else:
        steerer._register(trait, (sign * alpha_eff) / float(steerer.steer_gain))
        try:
            out = steerer.model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                                         repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
                                         eos_token_id=eos, pad_token_id=pad, return_dict_in_generate=False)
        finally:
            steerer._clear()
    gen = steerer.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()


@torch.no_grad()
def measure_ppl(model, tok, prompt: str, completion: str, device, max_len: int = 512) -> float:
    """Perplexity on completion conditioned on prompt (causal LM loss)."""
    text = prompt + completion
    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"][:, -max_len:].to(device)
    attn = enc["attention_mask"][:, -max_len:].to(device)
    # mask prompt positions
    prompt_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    n_prompt = prompt_ids.numel()
    labels = input_ids.clone()
    # mask everything up to the last n_completion tokens
    labels[:, :-min(labels.shape[1], enc["input_ids"].shape[1] - n_prompt)] = -100
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss.float().item()
    return float(math.exp(loss))


def trigram_repeat_rate(text: str) -> float:
    toks = text.split()
    if len(toks) < 3:
        return 0.0
    trigrams = [" ".join(toks[i:i+3]) for i in range(len(toks)-2)]
    total = len(trigrams)
    seen = {}
    repeat = 0
    for g in trigrams:
        seen[g] = seen.get(g, 0) + 1
        if seen[g] > 1:
            repeat += 1
    return repeat / total if total else 0.0


@torch.no_grad()
def mean_pooled_hidden(model, tok, text: str, device, max_tokens: int = 256) -> torch.Tensor:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
    need_reset = False
    if not getattr(model.config, "output_hidden_states", False):
        model.config.output_hidden_states = True
        need_reset = True
    out = model(**enc)
    hs = out.hidden_states[-1].float()  # [1, T, H]
    vec = hs.mean(dim=1).squeeze(0)     # [H]
    if need_reset:
        model.config.output_hidden_states = False
    return vec / (vec.norm(p=2) + 1e-9)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1.0 - torch.dot(a, b).item())


def token_overlap(prompt: str, text: str) -> float:
    def norm(s: str):
        return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in s).split()
    P, T = set(norm(prompt)), set(norm(text))
    return len(P & T) / max(1, len(P)) if P else 1.0


# Layer selection helpers

@torch.no_grad()
def _kl_signed_on_layer(steerer, sgn, prompt: str, trait: str, layer: int,
                        system: Optional[str] = None, alpha_test: float = 1.0) -> Tuple[float, float]:
    """Per-layer signed KLs at a small alpha for sign preference checks."""
    with SteerConfigPatch(steerer, [layer], [1.0]):
        klp = _next_token_kl(steerer, prompt, trait, alpha_test, +sgn, system=system)
        kln = _next_token_kl(steerer, prompt, trait, alpha_test, -sgn, system=system)
    return float(klp), float(kln)


def _enforce_verified_floor(layers: List[int], weights: List[float],
                            verified: List[int], min_verified_weight: float) -> List[float]:
    if not verified or min_verified_weight <= 0.0:
        return weights
    vset = set(verified)
    w = np.array(weights, dtype=np.float32)
    v_sum = float(np.sum([w[i] for i, L in enumerate(layers) if L in vset]))
    if v_sum >= min_verified_weight:
        return weights

    w_new = w.copy()
    # scale verified up
    scale_up = (min_verified_weight / max(v_sum, 1e-9))
    for i, L in enumerate(layers):
        if L in vset:
            w_new[i] = w[i] * scale_up

    # renormalize remaining mass to (1 - min_verified_weight)
    rem_target = 1.0 - min_verified_weight
    rem_idxs = [i for i, L in enumerate(layers) if L not in vset]
    rem_sum = float(np.sum(w[rem_idxs])) if rem_idxs else 0.0
    if rem_idxs and rem_sum > 0:
        for i in rem_idxs:
            w_new[i] = (w[i] / rem_sum) * rem_target

    s = float(np.sum(w_new))
    return (w_new / s).tolist() if s > 0 else weights


def _select_layers_hybrid(
    steerer, prompt: str, trait: str, *, system: Optional[str],
    k_runtime: int, max_layers: int, prior_boost: float, temperature_sel: float, min_weight: float
) -> Tuple[List[int], List[float], Dict[int, float]]:
    """Wrapper over your selector; returns layers, weights, norms (dict[layer]->score)."""
    layers, weights, norms = select_layers_for_prompt(
        steerer, prompt, trait, intensity=0.5,
        system=system, k_runtime=k_runtime,
        prior_boost=prior_boost, temperature=temperature_sel,
        max_layers=max_layers, min_weight=min_weight
    )
    return layers, weights, norms


def build_dual_mix(
    steerer, prompt: str, trait: str, sgn: int, verified: List[int], norms: Dict[int, float],
    *, system: Optional[str], max_layers: int, min_verified_weight: float, alpha_probe: float, log_level: str
) -> Tuple[List[int], List[float], List[int], List[float], Dict[int, Tuple[float, float]]]:
    """
    Build two sign-aware mixes (+ and -). Keeps verified layers in both and prefers
    runtime layers that have higher KL for the corresponding sign at a small probe alpha.
    Returns:
      layers_pos, weights_pos, layers_neg, weights_neg, per_layer_signed_kl (dict[layer]=(KL_pos, KL_neg))
    """
    top_sorted = [L for (L, _) in sorted(norms.items(), key=lambda kv: kv[1], reverse=True)]
    runtime_pool = [L for L in top_sorted if L not in set(verified)]

    per_layer = {}
    for L in runtime_pool[:8 * max_layers]:  # probe a small superset
        klp, kln = _kl_signed_on_layer(steerer, sgn, prompt, trait, L, system=system, alpha_test=alpha_probe)
        per_layer[int(L)] = (klp, kln)

    # start from verified anchors
    Lp = list(verified)
    Ln = list(verified)

    # add runtime layers that prefer each sign
    for L in runtime_pool:
        if len(Lp) >= max_layers and len(Ln) >= max_layers:
            break
        klp, kln = per_layer.get(int(L), (0.0, 0.0))
        if len(Lp) < max_layers and klp >= kln and L not in Lp:
            Lp.append(L)
        if len(Ln) < max_layers and kln >= klp and L not in Ln:
            Ln.append(L)

    # truncate
    Lp = Lp[:max_layers] if Lp else (verified[:max_layers] or runtime_pool[:max_layers])
    Ln = Ln[:max_layers] if Ln else (verified[:max_layers] or runtime_pool[:max_layers])

    Wp = [1.0 / len(Lp)] * len(Lp)
    Wn = [1.0 / len(Ln)] * len(Ln)

    if verified:
        Wp = _enforce_verified_floor(Lp, Wp, verified, min_verified_weight)
        Wn = _enforce_verified_floor(Ln, Wn, verified, min_verified_weight)

    if log_level in ("info", "debug"):
        print(f"[dual-mix] +layers={Lp} weights={[round(w,3) for w in Wp]}")
        print(f"[dual-mix] -layers={Ln} weights={[round(w,3) for w in Wn]}")

    return Lp, Wp, Ln, Wn, per_layer



# Frontier & target selection

def measure_frontier(
    steerer, prompt: str, trait: str, layers: List[int], weights: List[float], sign: int,
    *, system: Optional[str], alpha_grid: List[float], micro_len: int
) -> Tuple[List[float], List[float], List[float]]:
    """Return (alpha_grid, kls, ppl_ratios) for the given sign under fixed mix."""
    kls, ratios = [], []
    with SteerConfigPatch(steerer, layers, weights):
        # base micro text & ppl
        base_txt = _deterministic_generate(steerer, prompt, trait, 0.0, +1, system=system, max_new_tokens=micro_len)
        base_ppl = measure_ppl(steerer.model, steerer.tok, _format_for_chat(steerer, prompt, system), base_txt, steerer.device)
        for a in alpha_grid:
            kl = _next_token_kl(steerer, prompt, trait, a, sign, system)
            txt = _deterministic_generate(steerer, prompt, trait, a, sign, system=system, max_new_tokens=micro_len)
            ppl = measure_ppl(steerer.model, steerer.tok, _format_for_chat(steerer, prompt, system), txt, steerer.device)
            kls.append(float(kl))
            ratios.append(float(ppl / max(1e-6, base_ppl)))
    return list(alpha_grid), kls, ratios


def _elbow_from_frontier(kls: List[float], ratios: List[float], ok_idx: List[int]) -> float:
    """Discrete 'knee' via max curvature proxy on the quality-feasible subset."""
    if not ok_idx:
        return min(kls) if kls else 0.1
    if len(ok_idx) < 3:
        return kls[ok_idx[-1]]  # last feasible (most KL within quality bound)
    curv = []
    for j in range(1, len(ok_idx)-1):
        i1, i2, i3 = ok_idx[j-1], ok_idx[j], ok_idx[j+1]
        x1,y1 = kls[i1], ratios[i1]; x2,y2 = kls[i2], ratios[i2]; x3,y3 = kls[i3], ratios[i3]
        # curvature proxy: change in slope magnitude
        if (x2-x1) == 0 or (x3-x2) == 0:
            c = 0.0
        else:
            c = abs((y3-y2)/(x3-x2) - (y2-y1)/(x2-x1))
        curv.append((c, i2))
    curv.sort(key=lambda t: t[0], reverse=True)
    return kls[curv[0][1]] if curv else kls[ok_idx[-1]]


def auto_target_kl(
    alpha_grid: List[float], kls: List[float], ratios: List[float], base_entropy_bits_topk: float,
    *, tau: float, c_pos: float, c_neg: float, sign: int, kl_floor: float
) -> Tuple[float, Dict[str, float]]:
    """Choose target KL via elbow ∧ quality cap ∧ entropy cap ∧ floor."""
    # quality-feasible indices
    ok = [i for i, r in enumerate(ratios) if r <= tau]
    elbow = _elbow_from_frontier(kls, ratios, ok) if kls else 0.1
    c = c_pos if sign > 0 else c_neg
    entropy_cap = c * base_entropy_bits_topk
    chosen = max(kl_floor, min(elbow, entropy_cap))
    # If no feasible points, fall back to min KL in grid but respect floor
    if not ok:
        chosen = max(kl_floor, min(kls)) if kls else kl_floor
        reason = "no_quality_point"
        q_cap = None
    else:
        reason = "elbow∧quality∧entropy"
        q_cap = max(kls[i] for i in ok)
    policy = {
        "elbow_kl": float(elbow),
        "quality_cap_kl": float(q_cap) if q_cap is not None else None,
        "entropy_cap_kl": float(entropy_cap),
        "chosen_kl": float(chosen),
        "reason": reason,
        "base_entropy_bits_topk": float(base_entropy_bits_topk),
    }
    return chosen, policy


def calibrate_alpha_for_sign_in_context(
    steerer,
    prompt: str,
    trait: str,
    sign: int,
    target_kl: float,
    system: Optional[str] = None,
    alpha_hi: float = 8.0,
    alpha_lo: float = 0.0,
    max_iters: int = 14,
    tol_frac: float = 0.05,
    expand_factor: float = 2.0,
    alpha_hi_cap: float = 64.0,
) -> float:
    """Bisection on alpha_eff to hit next-token KL ~= target_kl for a given sign."""
    assert target_kl > 0.0
    lo, hi = float(alpha_lo), float(alpha_hi)

    # Expand until hi reaches target or cap
    kl_hi = _next_token_kl(steerer, prompt, trait, hi, sign, system)
    while kl_hi < target_kl and hi < alpha_hi_cap:
        lo = hi
        hi *= float(expand_factor)
        kl_hi = _next_token_kl(steerer, prompt, trait, hi, sign, system)

    if kl_hi < target_kl and hi >= alpha_hi_cap:
        return hi  # under-steer ceiling

    kl_lo = 0.0 if lo == 0.0 else _next_token_kl(steerer, prompt, trait, lo, sign, system)
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        kl_mid = _next_token_kl(steerer, prompt, trait, mid, sign, system)
        if abs(kl_mid - target_kl) <= tol_frac * target_kl:
            return mid
        if kl_mid < target_kl:
            lo, kl_lo = mid, kl_mid
        else:
            hi, kl_hi = mid, kl_mid
    return 0.5 * (lo + hi)



# One item end-to-end

def run_item(
    steerer,
    item: Dict,
    cfg: argparse.Namespace,
) -> Dict:
    prompt = item["prompt"]
    trait = (item.get("trait") or cfg.trait or "openness").lower()
    system = item.get("system", cfg.system)
    seed = int(item.get("seed", cfg.seed))

    set_seed(seed)
    # Freeze frac RMS for this prompt once so frontiers/calibration/final all align
    score_txt = _format_for_chat(steerer, prompt, system)
    if getattr(steerer, "alpha_mode", "abs") == "frac":
        steerer._measure_layer_rms(score_txt)

    # ---- 1) Base hybrid selection & global sign
    layers, weights, norms = _select_layers_hybrid(
        steerer, prompt, trait, system=system,
        k_runtime=cfg.k_runtime, max_layers=cfg.max_layers,
        prior_boost=cfg.prior_boost, temperature_sel=cfg.temperature_sel,
        min_weight=cfg.min_weight
    )
    verified = getattr(steerer, "_trait_layers", {}).get(trait.lower(), [])
    # Decide sign using verified anchor if present
    with SteerConfigPatch(steerer, (verified[:cfg.max_layers] or layers), [1.0 / len(verified[:cfg.max_layers] or layers)] * len(verified[:cfg.max_layers] or layers)):
        sgn = int(np.sign(steerer._calibrate_polarity(trait)) or 1)

    # ---- 2) Shared vs dual layer mix
    per_layer_signed_kl = {}
    if cfg.layer_mix_mode == "dual":
        Lp, Wp, Ln, Wn, per_layer_signed_kl = build_dual_mix(
            steerer, prompt, trait, sgn, verified, norms, system=system,
            max_layers=cfg.max_layers, min_verified_weight=cfg.min_verified_weight,
            alpha_probe=cfg.signed_filter_alpha, log_level=cfg.log_level
        )
    else:
        # shared
        if verified and cfg.min_verified_weight > 0.0:
            weights = _enforce_verified_floor(layers, weights, verified, cfg.min_verified_weight)
        Lp, Wp = layers, weights
        Ln, Wn = layers, weights

    # ---- 3) Frontier per sign
    alpha_grid = [float(x) for x in cfg.alpha_grid]
    micro_len = int(cfg.micro_len)

    Hp = _base_topk_entropy_bits(steerer, prompt, system, k=50)
    alpha_p_grid, kls_p, ratios_p = measure_frontier(steerer, prompt, trait, Lp, Wp, +sgn, system=system, alpha_grid=alpha_grid, micro_len=micro_len)
    kl_target_pos, policy_pos = auto_target_kl(alpha_p_grid, kls_p, ratios_p, Hp, tau=cfg.tau, c_pos=cfg.c_pos, c_neg=cfg.c_neg, sign=+sgn, kl_floor=cfg.kl_floor)

    Hn = Hp  # base entropy is same for both signs (same base next-token)
    alpha_n_grid, kls_n, ratios_n = measure_frontier(steerer, prompt, trait, Ln, Wn, -sgn, system=system, alpha_grid=alpha_grid, micro_len=micro_len)
    kl_target_neg, policy_neg = auto_target_kl(alpha_n_grid, kls_n, ratios_n, Hn, tau=cfg.tau, c_pos=cfg.c_pos, c_neg=cfg.c_neg, sign=-sgn, kl_floor=cfg.kl_floor)

    # ---- 4) Calibrate to hit each target + quality backoff
    with SteerConfigPatch(steerer, Lp, Wp):
        alpha_pos = calibrate_alpha_for_sign_in_context(
            steerer, prompt, trait, +sgn, kl_target_pos, system=system, alpha_hi=cfg.alpha_hi
        )
        # quality backoff loop
        for _ in range(3):
            kl_now = _next_token_kl(steerer, prompt, trait, alpha_pos, +sgn, system=system)
            txt_micro = _deterministic_generate(steerer, prompt, trait, alpha_pos, +sgn, system=system, max_new_tokens=micro_len)
            ppl_micro = measure_ppl(steerer.model, steerer.tok, _format_for_chat(steerer, prompt, system), txt_micro, steerer.device)
            base_txt = _deterministic_generate(steerer, prompt, trait, 0.0, +1, system=system, max_new_tokens=micro_len)
            base_ppl = measure_ppl(steerer.model, steerer.tok, _format_for_chat(steerer, prompt, system), base_txt, steerer.device)
            if ppl_micro / max(1e-6, base_ppl) <= cfg.tau:
                break
            alpha_pos *= 0.8

    with SteerConfigPatch(steerer, Ln, Wn):
        alpha_neg = calibrate_alpha_for_sign_in_context(
            steerer, prompt, trait, -sgn, kl_target_neg, system=system, alpha_hi=cfg.alpha_hi
        )
        for _ in range(3):
            kl_now = _next_token_kl(steerer, prompt, trait, alpha_neg, -sgn, system=system)
            txt_micro = _deterministic_generate(steerer, prompt, trait, alpha_neg, -sgn, system=system, max_new_tokens=micro_len)
            ppl_micro = measure_ppl(steerer.model, steerer.tok, _format_for_chat(steerer, prompt, system), txt_micro, steerer.device)
            base_txt = _deterministic_generate(steerer, prompt, trait, 0.0, +1, system=system, max_new_tokens=micro_len)
            base_ppl = measure_ppl(steerer.model, steerer.tok, _format_for_chat(steerer, prompt, system), base_txt, steerer.device)
            if ppl_micro / max(1e-6, base_ppl) <= cfg.tau:
                break
            alpha_neg *= 0.8

    # Final measured KLs with chosen alphas
    with SteerConfigPatch(steerer, Lp, Wp):
        kl_pos = _next_token_kl(steerer, prompt, trait, alpha_pos, +sgn, system=system)
    with SteerConfigPatch(steerer, Ln, Wn):
        kl_neg = _next_token_kl(steerer, prompt, trait, alpha_neg, -sgn, system=system)

    # ---- 5) Decode baseline / + / -
    if cfg.style == "balanced":
        temperature, top_p, top_k, rep = 0.4, 0.85, 50, 1.07
    else:
        temperature, top_p, top_k, rep = 0.7, 0.90, 50, 1.02

    base_txt = _sampled_generate(steerer, prompt, trait, 0.0, +1, system=system, max_new_tokens=cfg.max_new_tokens,
                                 temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=rep)
    pos_txt = None
    with SteerConfigPatch(steerer, Lp, Wp):
        pos_txt = _sampled_generate(steerer, prompt, trait, alpha_pos, +sgn, system=system, max_new_tokens=cfg.max_new_tokens,
                                    temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=rep)
    neg_txt = None
    with SteerConfigPatch(steerer, Ln, Wn):
        neg_txt = _sampled_generate(steerer, prompt, trait, alpha_neg, -sgn, system=system, max_new_tokens=cfg.max_new_tokens,
                                    temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=rep)

    # ---- 6) Metrics
    dev = steerer.device
    tok = steerer.tok
    mdl = steerer.model

    ppl_base = measure_ppl(mdl, tok, _format_for_chat(steerer, prompt, system), base_txt, dev)
    ppl_pos  = measure_ppl(mdl, tok, _format_for_chat(steerer, prompt, system), pos_txt,  dev)
    ppl_neg  = measure_ppl(mdl, tok, _format_for_chat(steerer, prompt, system), neg_txt,  dev)

    # Optional: final quality backoff (sampled text)
    for _ in range(2):
        ratio_pos = ppl_pos / max(1e-6, ppl_base)
        if ratio_pos > cfg.tau:
            alpha_pos *= 0.8
            with SteerConfigPatch(steerer, Lp, Wp):
                pos_txt = _sampled_generate(..., alpha_pos, +sgn, ...)
            ppl_pos = measure_ppl(mdl, tok, _format_for_chat(steerer, prompt, system), pos_txt, dev)
        else:
            break

    for _ in range(2):
        ratio_neg = ppl_neg / max(1e-6, ppl_base)
        if ratio_neg > cfg.tau:
            alpha_neg *= 0.8
            with SteerConfigPatch(steerer, Ln, Wn):
                neg_txt = _sampled_generate(..., alpha_neg, -sgn, ...)
            ppl_neg = measure_ppl(mdl, tok, _format_for_chat(steerer, prompt, system), neg_txt, dev)
        else:
            break


    rep_base = trigram_repeat_rate(base_txt)
    rep_pos  = trigram_repeat_rate(pos_txt)
    rep_neg  = trigram_repeat_rate(neg_txt)

    e_pos  = mean_pooled_hidden(mdl, tok, pos_txt, dev)
    e_neg  = mean_pooled_hidden(mdl, tok, neg_txt, dev)
    embed_dist = cosine_distance(e_pos, e_neg)

    topical_pos = token_overlap(prompt, pos_txt)
    topical_neg = token_overlap(prompt, neg_txt)

    # Gates
    quality_ok = (
        (ppl_pos / max(1e-6, ppl_base) <= cfg.ppl_ratio_max) and
        (ppl_neg / max(1e-6, ppl_base) <= cfg.ppl_ratio_max) and
        (rep_pos <= cfg.repeat_rate_max) and
        (rep_neg <= cfg.repeat_rate_max)
    )
    separation_ok = (embed_dist >= cfg.embed_dist_min)

    final_pass = quality_ok and separation_ok  # KL selection already enforced by auto policy/backoff

    # ---- 7) Record
    rec = {
        "id": item.get("id"),
        "trait": trait,
        "prompt": prompt,
        "system": system,
        "seed": seed,

        "layer_mix_mode": cfg.layer_mix_mode,
        "layers_pos": Lp, "weights_pos": [float(w) for w in Wp],
        "layers_neg": Ln, "weights_neg": [float(w) for w in Wn],
        "per_layer_signed_kl": {str(k): [float(v[0]), float(v[1])] for k, v in per_layer_signed_kl.items()},

        "alpha_grid": [float(a) for a in alpha_grid],
        "frontier_pos": {"kls": kls_p, "ppl_ratio": ratios_p},
        "frontier_neg": {"kls": kls_n, "ppl_ratio": ratios_n},
        "target_policy_pos": policy_pos,
        "target_policy_neg": policy_neg,

        "alpha_pos": float(alpha_pos),
        "alpha_neg": float(alpha_neg),
        "kl_pos": float(kl_pos),
        "kl_neg": float(kl_neg),

        "text_base": base_txt,
        "text_pos": pos_txt,
        "text_neg": neg_txt,

        "ppl_base": float(ppl_base),
        "ppl_pos": float(ppl_pos),
        "ppl_neg": float(ppl_neg),
        "repeat_base": float(rep_base),
        "repeat_pos": float(rep_pos),
        "repeat_neg": float(rep_neg),
        "embed_dist_pos_neg": float(embed_dist),
        "topical_pos": float(topical_pos),
        "topical_neg": float(topical_neg),

        "quality_ok": bool(quality_ok),
        "separation_ok": bool(separation_ok),
        "pass": bool(final_pass),
    }
    return rec



# I/O helpers

def read_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out


def append_jsonl(path: str, row: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv_summary(path: str, rows: List[Dict]):
    if not rows:
        return
    keys = [
        "id","trait","pass","quality_ok","separation_ok",
        "alpha_pos","alpha_neg","kl_pos","kl_neg",
        "embed_dist_pos_neg","ppl_base","ppl_pos","ppl_neg",
        "repeat_pos","repeat_neg","topical_pos","topical_neg",
        "layers_pos","weights_pos","layers_neg","weights_neg","layer_mix_mode"
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in keys}
            w.writerow(row)



# MAIN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--data", type=str, required=True,
                    help="JSONL with fields: id, prompt [, trait] [, system] [, seed]")
    ap.add_argument("--out_dir", type=str, required=True)

    # global defaults (can be overridden per item)
    ap.add_argument("--trait", type=str, default=None, help="Trait to use if row doesn't include one.")
    ap.add_argument("--system", type=str, default="Write naturally; avoid meta-AI phrasing.")
    ap.add_argument("--seed", type=int, default=1234)

    # layer policy & safety
    ap.add_argument("--layer_mix_mode", type=str, default="dual", choices=["shared","dual"])
    ap.add_argument("--k_runtime", type=int, default=2)
    ap.add_argument("--max_layers", type=int, default=2)
    ap.add_argument("--prior_boost", type=float, default=0.10)
    ap.add_argument("--temperature_sel", type=float, default=0.50)
    ap.add_argument("--min_weight", type=float, default=0.10)
    ap.add_argument("--min_verified_weight", type=float, default=0.60)
    ap.add_argument("--signed_filter_alpha", type=float, default=1.0)

    # injection
    ap.add_argument("--injection_point", type=str, default="post", choices=["post","mha","mlp","final_norm"])
    ap.add_argument("--alpha_mode", type=str, default="frac", choices=["abs", "frac"])
    ap.add_argument("--last_only", action="store_true", default=False)

    # alpha/KL frontier & policy
    ap.add_argument("--alpha_grid", type=str, default="0.25,0.5,1.0,2.0")
    ap.add_argument("--alpha_hi", type=float, default=4.0)
    ap.add_argument("--micro_len", type=int, default=60)
    ap.add_argument("--tau", type=float, default=1.35)       # quality cap (PPL ratio)
    ap.add_argument("--c_pos", type=float, default=0.35)     # entropy cap coefficient for +
    ap.add_argument("--c_neg", type=float, default=0.25)     # entropy cap coefficient for -
    ap.add_argument("--kl_floor", type=float, default=0.08)  # lower bound for target KL

    # decoding & length
    ap.add_argument("--style", type=str, default="balanced", choices=["balanced","strong"])
    ap.add_argument("--max_new_tokens", type=int, default=180)

    # pass/fail thresholds
    ap.add_argument("--embed_dist_min", type=float, default=0.10)
    ap.add_argument("--repeat_rate_max", type=float, default=0.20)
    ap.add_argument("--ppl_ratio_max", type=float, default=1.35)

    # logging
    ap.add_argument("--log_level", type=str, default="info", choices=["silent","warn","info","debug"])
    args = ap.parse_args()

    # parse alpha grid
    args.alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "results.jsonl")
    out_csv   = os.path.join(args.out_dir, "summary.csv")

    # load steerer once
    steerer = load_steerer(args.results_dir)
    steerer.injection_point = args.injection_point
    steerer.alpha_mode = args.alpha_mode
    steerer.last_position_only = bool(args.last_only)
    steerer.log_level = args.log_level

    rows = read_jsonl(args.data)
    print(f"[bench] loaded {len(rows)} items")

    all_results = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        try:
            rec = run_item(steerer, row, args)
            append_jsonl(out_jsonl, rec)
            all_results.append(rec)
        except Exception as e:
            err = {"id": row.get("id"), "prompt": row.get("prompt"), "error": repr(e), "pass": False}
            append_jsonl(out_jsonl, err)
            all_results.append(err)

        if i % 10 == 0:
            print(f"[bench] {i}/{len(rows)} done...")

    # CSV summary
    valid = [r for r in all_results if "error" not in r]
    write_csv_summary(out_csv, valid)

    # aggregates
    if valid:
        pass_rate = np.mean([1.0 if r.get("pass") else 0.0 for r in valid])
        med_embed = float(np.median([r.get("embed_dist_pos_neg", 0.0) for r in valid]))
        alp = [r.get("alpha_pos", 0.0) for r in valid] + [r.get("alpha_neg", 0.0) for r in valid]
        kls = [r.get("kl_pos", 0.0) for r in valid] + [r.get("kl_neg", 0.0) for r in valid]
        print(f"\n[bench] done in {time.time()-t0:.1f}s")
        print(f"[bench] pass_rate={pass_rate:.3f}  median_embed_dist={med_embed:.3f}")
        print(f"[bench] median_alpha={np.median(alp):.3f}  median_kl={np.median(kls):.3f}")
        # small diagnostic on negative steering quality (often the hard part)
        neg_ratios = [r["ppl_neg"]/max(1e-6,r["ppl_base"]) for r in valid]
        print(f"[bench] median_ppl_ratio_neg={np.median(neg_ratios):.3f}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
