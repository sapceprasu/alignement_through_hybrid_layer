
from typing import Dict, List
import json, os
import numpy as np
import torch

def _format_chat(steerer, text: str) -> str:
    return steerer._format_prompt(text, use_chat_template=True, system=None)

@torch.no_grad()
def _first_token_probs(steerer, txt: str) -> torch.Tensor:
    enc = steerer.tok(
        txt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=getattr(steerer.cfg, "max_length", 1024),
    ).to(steerer.device)

    out = steerer.model.generate(
        **enc,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=False,
        eos_token_id=steerer.tok.eos_token_id,
        pad_token_id=steerer.tok.pad_token_id or steerer.tok.eos_token_id,
    )
    return out.scores[0].float().softmax(dim=-1).squeeze(0)

def _delta_logits_L2(steerer, txt: str, trait: str, layer: int, alpha: float) -> float:
    # kept for compatibility (verify_best_layers uses the batched path)
    steerer._measure_layer_rms(txt)
    old_layers = getattr(steerer, "_trait_layers", {}).get(trait, None)
    if not hasattr(steerer, "_trait_layers"):
        steerer._trait_layers = {}
    steerer._trait_layers[trait] = [layer]
    p0 = _first_token_probs(steerer, txt)
    steerer._register(trait, alpha)
    try:
        p1 = _first_token_probs(steerer, txt)
    finally:
        steerer._clear()
    if old_layers is None:
        del steerer._trait_layers[trait]
    else:
        steerer._trait_layers[trait] = old_layers
    return torch.norm(p1 - p0, p=2).item()

def _first_token_KL(steerer, txt: str, trait: str, layer: int, alpha: float) -> float:
    steerer._measure_layer_rms(txt)
    old_layers = getattr(steerer, "_trait_layers", {}).get(trait, None)
    if not hasattr(steerer, "_trait_layers"):
        steerer._trait_layers = {}
    steerer._trait_layers[trait] = [layer]
    p0 = _first_token_probs(steerer, txt)
    steerer._register(trait, alpha)
    try:
        p1 = _first_token_probs(steerer, txt)
    finally:
        steerer._clear()
    if old_layers is None:
        del steerer._trait_layers[trait]
    else:
        steerer._trait_layers[trait] = old_layers
    eps = 1e-9
    p0 = p0.clamp_min(eps)
    p1 = p1.clamp_min(eps)
    return float(torch.sum(p0 * (torch.log(p0) - torch.log(p1))).item())

@torch.no_grad()
def _p0p1_for_layer_batched(steerer, txts: List[str], trait: str, layer: int, alpha: float):
    """
    Returns list of (p0, p1) tensors for each txt, computed with 2 batched generate() calls.
    """
    # tokenize once
    enc = steerer.tok(
        txts, return_tensors="pt", padding=True, truncation=True,
        max_length=getattr(steerer.cfg, "max_length", 1024)
    ).to(steerer.device)

    # measure RMS per text (cheap, keeps your normalization logic)
    for t in txts:
        steerer._measure_layer_rms(t)

    # force candidate layer
    old_layers = getattr(steerer, "_trait_layers", {}).get(trait, None)
    if not hasattr(steerer, "_trait_layers"):
        steerer._trait_layers = {}
    steerer._trait_layers[trait] = [layer]

    # baseline
    out0 = steerer.model.generate(
        **enc,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=False,
        eos_token_id=steerer.tok.eos_token_id,
        pad_token_id=steerer.tok.pad_token_id or steerer.tok.eos_token_id,
    )
    p0 = out0.scores[0].float().softmax(dim=-1)  # [B, V]

    # steered
    steerer._register(trait, alpha)  # pass sign=+1 if your _register requires it
    try:
        out1 = steerer.model.generate(
            **enc,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            eos_token_id=steerer.tok.eos_token_id,
            pad_token_id=steerer.tok.pad_token_id or steerer.tok.eos_token_id,
        )
        p1 = out1.scores[0].float().softmax(dim=-1)  # [B, V]
    finally:
        steerer._clear()

    # restore
    if old_layers is None:
        del steerer._trait_layers[trait]
    else:
        steerer._trait_layers[trait] = old_layers

    return [(p0[i], p1[i]) for i in range(p0.size(0))]

def verify_best_layers(cfg, steerer) -> Dict[str, List[int]]:
    """
    Returns: {trait: [best_layer,...]}
    Writes JSON to <results_dir>/layer_verified.json
    """
    # align runtime with intended settings
    steerer.steer_mode = getattr(cfg.layer_search, "eval_steer_mode", "weighted")
    steerer.injection_point = cfg.layer_search.eval_injection_point

    # sanity: device align
    assert next(steerer.model.parameters()).device.type == steerer.device.type, "model/device mismatch"
    tmp = steerer.tok("hi", return_tensors="pt").to(steerer.device)
    assert tmp.input_ids.device.type == steerer.device.type, "encodings not on device"

    trait2best = {}
    probe_texts = [_format_chat(steerer, p) for p in cfg.layer_search.probe_prompts]
    alpha = float(cfg.layer_search.alpha_probe)
    w = cfg.layer_search.metric_weights

    for trait in cfg.trait_mapping.values():
        scores = []
        for L in cfg.layer_range:
            pairs = _p0p1_for_layer_batched(steerer, probe_texts, trait, L, alpha)
            if not pairs:
                scores.append((0.0, L))
                continue
            # metrics
            d_l2 = float(np.mean([torch.norm(p1 - p0, p=2).item() for p0, p1 in pairs]))
            eps = 1e-9
            fkl = float(np.mean([
                torch.sum(
                    p0.clamp_min(eps) * (torch.log(p0.clamp_min(eps)) - torch.log(p1.clamp_min(eps)))
                ).item() for p0, p1 in pairs
            ]))
            flip = float(np.mean([
                int(torch.argmax(p0).item() != torch.argmax(p1).item())
                for p0, p1 in pairs
            ]))
            s = (w["delta_l2"] * d_l2 + w["first_kl"] * fkl + w["flip"] * flip)
            scores.append((s, L))
        scores.sort(reverse=True)
        trait2best[trait] = [L for _, L in scores[:max(1, int(cfg.layer_search.top_k))]]

    out_path = os.path.join(cfg.results_dir, "layer_verified.json")
    with open(out_path, "w") as f:
        json.dump(trait2best, f, indent=2)
    return trait2best
