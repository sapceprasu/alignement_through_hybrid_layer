
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime
import argparse
import json
import re
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

# --- your modules (unchanged) ---
from ..main import load_steerer
from ..layer_selector import select_layers_for_prompt, SteerConfigPatch

TRAIT_ALL = ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]
DATASET_TAG = "gsm8k-cot8"

DEFAULT_SYSTEM = (
    "Think step by step and finish with the number after a '####' marker. "
    "Output only your reasoning and the final line '#### <number>'."
)

# --------------------------
# Helpers: filenames, formatting, parsing
# --------------------------
def _model_name_for_filename(steerer) -> str:
    name = getattr(getattr(steerer, "model", None), "name_or_path", None)
    if not name:
        cfg = getattr(getattr(steerer, "model", None), "config", None)
        name = getattr(cfg, "_name_or_path", None)
    if not name:
        name = "model"
    return name.strip().replace("/", "__").replace(" ", "_")

def _format_for_chat(steerer, text: str, system: str = None) -> str:
    return steerer._format_prompt(text, use_chat_template=True, system=system)

def _ensure_frac_rms_if_needed(steerer, formatted_prompt: str):
    if getattr(steerer, "alpha_mode", "abs") == "frac":
        steerer._measure_layer_rms(formatted_prompt)

# robust number parsing
_NUM_RE = re.compile(r"[+\-−]?\d[\d,]*(?:\.\d+)?(?:[eE][+\-]?\d+)?")

def _normalize_number(s: str) -> str:
    s = s.strip().replace("−","-").replace(",", "")
    if re.fullmatch(r"[+\-]?\d+\.", s):
        s = s[:-1]
    return s

def extract_final_number_after_hashes(text: str) -> str:
    # 1) canonical pattern
    m = re.search(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$", text)
    if m:
        return _normalize_number(m.group(1))
    # 2) sometimes the model prints "####" then the number on next line
    m = re.search(r"####\s*$", text)
    if m:
        tail = text[m.end():]
        n2 = _NUM_RE.search(tail)
        if n2:
            return _normalize_number(n2.group(0))
    # 3) fallback: last number anywhere
    nums = _NUM_RE.findall(text)
    if nums:
        return _normalize_number(nums[-1])
    return ""

def equal_numbers(a: str, b: str, tol: float = 0.0) -> bool:
    if not a or not b:
        return False
    if tol <= 0:
        return a == b
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False

# --------------------------
# Build the 8-shot CoT prompt
# --------------------------
def _format_exemplar(q: str, a_cot: str) -> str:
    q = q.strip()
    a_cot = a_cot.strip()
    return f"Q: {q}\nA: {a_cot}\n\n"

def _build_cot8_prompt(exemplars: List[dict], test_question: str) -> str:
    blocks = []
    for ex in exemplars:
        blocks.append(_format_exemplar(ex["question"], ex["answer"]))
    return "".join(blocks) + f"Q: {test_question.strip()}\nA:"

def _load_exemplars_from_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    out = []
    for it in items:
        q = (it.get("question") or "").strip()
        a = (it.get("answer") or "").strip()
        if q and a:
            out.append({"question": q, "answer": a})
    if len(out) < 8:
        raise ValueError(f"Need at least 8 exemplars; got {len(out)} in {path}")
    return out[:8]

def _load_canonical_or_fallback_exemplars(exemplars_path: str = "") -> List[dict]:
    if exemplars_path:
        return _load_exemplars_from_json(exemplars_path)
    ds_train = load_dataset("gsm8k", "main", split="train")
    exemplars = []
    for i in range(8):
        ex = ds_train[i]
        exemplars.append({"question": ex["question"], "answer": ex["answer"]})
    return exemplars

# --------------------------
# Generation (sampling per LLaMA recipe)
# --------------------------
@torch.no_grad()
def generate_once(
    steerer,
    formatted_prompt: str,
    *,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
):
    """Return (raw_full_text, new_text_only)."""
    dev = steerer.device
    tok = steerer.tok
    enc = tok(formatted_prompt, return_tensors="pt").to(dev)
    prompt_len = enc["input_ids"].shape[1]
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id or eos_id

    out = steerer.model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=(temperature if do_sample else None),
        top_p=(top_p if do_sample else None),
        top_k=(top_k if do_sample else None),
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        return_dict_in_generate=False,
    )
    full_text = tok.decode(out[0], skip_special_tokens=True)
    new_ids = out[0][prompt_len:]
    new_text = tok.decode(new_ids, skip_special_tokens=True).strip()
    return full_text, new_text

def majority_vote(strings: List[str]) -> str:
    if not strings:
        return ""
    cnt = Counter(strings)
    best = max(cnt.items(), key=lambda kv: (kv[1], -strings.index(kv[0])))
    return best[0]

# --------------------------
# Tiny first-token probe (stability/strength)
# --------------------------
@torch.no_grad()
def _first_token_logits(steerer, formatted_prompt: str, layers=None, weights=None, trait=None, alpha=None):
    """
    Returns the *next-token* logits at the first generation step,
    either baseline (no hooks) or with a temporary (layers, weights, trait, alpha) install.
    """
    tok = steerer.tok
    dev = steerer.device
    enc = tok(formatted_prompt, return_tensors="pt").to(dev)
    _ensure_frac_rms_if_needed(steerer, formatted_prompt)

    if layers is None:
        # baseline
        logits = steerer.model(**enc).logits[:, -1, :].float()
        return logits

    with SteerConfigPatch(steerer, layers, weights):
        steerer._register(trait, alpha)
        try:
            logits = steerer.model(**enc).logits[:, -1, :].float()
        finally:
            steerer._clear()
    return logits

def _kl_pq(p, q, eps=1e-8):
    p = torch.softmax(p, dim=-1)
    q = torch.softmax(q, dim=-1)
    return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=-1)

def _entropy(p, eps=1e-8):
    p = torch.softmax(p, dim=-1)
    return -torch.sum(p * torch.log(p + eps), dim=-1)

@torch.no_grad()
def probe_polarity_stability(
    steerer,
    formatted_prompt: str,
    trait: str,
    layers: List[int],
    weights: List[float],
    alpha_signed: float,
    *,
    kl_warn: float = 2.0,
    dH_warn: float = 0.5
) -> Tuple[List[int], List[float], Dict[str, float]]:
    """
    Checks first-token KL and entropy change for the given sign.
    If it's too spiky, try softening by down-weighting or dropping the second layer.
    Returns (possibly adjusted) layers/weights and probe metrics.
    """
    base_logits = _first_token_logits(steerer, formatted_prompt)
    steer_logits = _first_token_logits(steerer, formatted_prompt, layers, weights, trait, alpha_signed)

    kl = float(_kl_pq(base_logits, steer_logits).item())
    H0 = float(_entropy(base_logits).item())
    H1 = float(_entropy(steer_logits).item())
    dH = H1 - H0

    adjusted_layers = list(layers)
    adjusted_weights = list(weights)

    # If too aggressive, try dropping/shrinking the second layer (keep verified anchor)
    if kl > kl_warn or dH > dH_warn:
        if len(adjusted_layers) >= 2:
            # Keep slot 0; reduce slot 1, renormalize
            adjusted_weights[1] *= 0.25
            s = sum(adjusted_weights)
            adjusted_weights = [w / s for w in adjusted_weights]

            # Re-probe
            steer_logits2 = _first_token_logits(
                steerer, formatted_prompt, adjusted_layers, adjusted_weights, trait, alpha_signed
            )
            kl2 = float(_kl_pq(base_logits, steer_logits2).item())
            H12 = float(_entropy(steer_logits2).item())
            dH2 = H12 - H0

            if kl2 > kl_warn or dH2 > dH_warn:
                # Drop second layer entirely
                adjusted_layers = [adjusted_layers[0]]
                adjusted_weights = [1.0]
                steer_logits3 = _first_token_logits(
                    steerer, formatted_prompt, adjusted_layers, adjusted_weights, trait, alpha_signed
                )
                kl3 = float(_kl_pq(base_logits, steer_logits3).item())
                H13 = float(_entropy(steer_logits3).item())
                dH3 = H13 - H0
                return adjusted_layers, adjusted_weights, {"kl": kl3, "dH": dH3, "adjusted": True}

            return adjusted_layers, adjusted_weights, {"kl": kl2, "dH": dH2, "adjusted": True}

    return adjusted_layers, adjusted_weights, {"kl": kl, "dH": dH, "adjusted": False}

# --------------------------
# Base vs Steered (maj@1)
# --------------------------
@torch.no_grad()
def predict_number_base_maj1(
    steerer,
    formatted_prompt: str,
    *,
    num_samples: int = 5,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
):
    raw_samples = []       # full texts
    extracted_numbers = [] # extracted numbers
    _ensure_frac_rms_if_needed(steerer, formatted_prompt)
    for _ in range(max(1, num_samples)):
        full, new_only = generate_once(
            steerer, formatted_prompt,
            max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, top_k=top_k
        )
        raw_samples.append(full)  # keep full context for auditing
        extracted_numbers.append(extract_final_number_after_hashes(full))
    return majority_vote(extracted_numbers), raw_samples, extracted_numbers

@torch.no_grad()
def predict_number_steered_maj1(
    steerer,
    prompt_text_for_selection: str,
    formatted_prompt: str,
    trait: str,
    alpha_signed: float,
    *,
    num_samples: int = 5,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    system_line: str = None,
    polarity_stability_guard: bool = True,
    cooler_neg_decode: bool = False,
):
    # 1) Hybrid selection on the QUESTION (unchanged)
    layers, weights, all_norms = select_layers_for_prompt(
        steerer, prompt_text_for_selection, trait, intensity=abs(alpha_signed),
        system=system_line, k_runtime=2, prior_boost=0.15,
        temperature=0.50, max_layers=2, min_weight=0.25
    )

    # 2) Optional polarity-stability guard (especially helpful for negative sign)
    if polarity_stability_guard and alpha_signed < 0:
        layers, weights, probe = probe_polarity_stability(
            steerer, formatted_prompt, trait, layers, weights, alpha_signed,
            kl_warn=2.0, dH_warn=0.5
        )
        # (You could print probe if you want real-time diagnostics.)

    # 3) Decode (optionally slightly cooler for negative sign)
    t_use, p_use = temperature, top_p
    if cooler_neg_decode and alpha_signed < 0:
        t_use = max(0.1, temperature - 0.1)
        p_use = max(0.1, top_p - 0.05)

    raw_samples = []
    extracted_numbers = []
    with SteerConfigPatch(steerer, layers, weights):
        steerer._register(trait, alpha_signed)
        try:
            for _ in range(max(1, num_samples)):
                full, new_only = generate_once(
                    steerer, formatted_prompt,
                    max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=t_use, top_p=p_use, top_k=top_k
                )
                raw_samples.append(full)
                extracted_numbers.append(extract_final_number_after_hashes(full))
        finally:
            steerer._clear()

    return majority_vote(extracted_numbers), layers, weights, all_norms, raw_samples, extracted_numbers

# --------------------------
# Evaluate ALL traits (single CSV/JSON)
# --------------------------
def evaluate_gsm8k_cot8_alltraits(
    results_dir: str,
    alpha_pos: float,
    *,
    alpha_neg: float = None,            # NEW: separate magnitude for −α (defaults to alpha_pos)
    traits: List[str] = None,           # default: all five
    limit: int = 0,                     # 0 = all test items
    system_line: str = DEFAULT_SYSTEM,
    out_dir_name: str = "gsm8k_eval",
    num_samples: int = 5,               # maj@1 pool size
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    float_tolerance: float = 0.0,
    polarity_overrides: Dict[str, int] = None,
    exemplars_path: str = "",           # optional canonical 8-shot file
    cooler_neg_decode: bool = False,    # NEW: cooler decoding on −α to help fluency
):
    # Load steerer once
    steerer = load_steerer(results_dir)

    # Optional polarity overrides (e.g., set +1 for all but neuroticism)
    if polarity_overrides:
        for t, s in polarity_overrides.items():
            steerer.polarity_override[t.lower()] = int(s)

    if not traits:
        traits = TRAIT_ALL

    alpha_neg = alpha_pos if alpha_neg is None else float(alpha_neg)

    # Dataset & exemplars
    ds_test = load_dataset("gsm8k", "main", split="test")
    exemplars = _load_canonical_or_fallback_exemplars(exemplars_path)
    total_items = len(ds_test)
    n_eval = total_items if limit <= 0 else min(limit, total_items)
    print(f"[GSM8K-CoT8] test total={total_items}; evaluating per trait n={n_eval}")
    print(f"[GSM8K-CoT8] exemplars: {'file:'+exemplars_path if exemplars_path else 'train[:8] fallback'}")

    # Output files (single CSV/JSON across all traits)
    model_tag = _model_name_for_filename(steerer)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(results_dir) / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / f"{model_tag}__alpha+{alpha_pos:+}_-{alpha_neg:+}__{DATASET_TAG}__{stamp}.csv"
    json_path = out_dir / f"{model_tag}__alpha+{alpha_pos:+}_-{alpha_neg:+}__{DATASET_TAG}__{stamp}.json"
    audit_path = out_dir / f"{model_tag}__alpha+{alpha_pos:+}_-{alpha_neg:+}__{DATASET_TAG}__{stamp}.audit.jsonl"

    import csv
    fieldnames = [
        "trait", "index",
        "gold", "pred_base", "pred_pos", "pred_neg",
        "acc_base", "acc_pos", "acc_neg",
        "layers_pos", "weights_pos", "layers_neg", "weights_neg"
    ]
    csv_f = open(csv_path, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    writer.writeheader(); csv_f.flush()
    audit_f = open(audit_path, "a", encoding="utf-8")

    # Accumulators
    per_trait_summary = {}
    overall_base = overall_pos = overall_neg = overall_total = 0

    # Iterate traits
    for trait in traits:
        base_correct = pos_correct = neg_correct = 0
        print(f"\n=== Trait: {trait} | alpha_pos={alpha_pos:+} | alpha_neg={alpha_neg:+} | num_samples={num_samples} ===")

        for i in tqdm(range(n_eval), desc=f"GSM8K-CoT8[{trait}]", leave=False):
            ex = ds_test[i]
            question = ex["question"]
            gold = extract_final_number_after_hashes(ex["answer"])

            # Build final 8-shot prompt and format with chat template
            cot8_prompt = _build_cot8_prompt(exemplars, question)
            formatted_prompt = _format_for_chat(steerer, cot8_prompt, system_line)
            _ensure_frac_rms_if_needed(steerer, formatted_prompt)

            # --- Baseline maj@1 ---
            pred_base, base_raws, base_nums = predict_number_base_maj1(
                steerer, formatted_prompt,
                num_samples=num_samples, max_new_tokens=1024,
                temperature=temperature, top_p=top_p, top_k=top_k
            )

            # --- Steered maj@1 with HYBRID selection ( +alpha_pos ) ---
            pred_pos, layers_pos, weights_pos, _ , pos_raws, pos_nums = predict_number_steered_maj1(
                steerer,
                prompt_text_for_selection=question,
                formatted_prompt=formatted_prompt,
                trait=trait, alpha_signed=+alpha_pos,
                num_samples=num_samples, max_new_tokens=1024,
                temperature=temperature, top_p=top_p, top_k=top_k,
                system_line=system_line,
                polarity_stability_guard=False,      # positive side usually stable
                cooler_neg_decode=False
            )

            # --- Steered maj@1 with HYBRID selection ( -alpha_neg ) ---
            pred_neg, layers_neg, weights_neg, _, neg_raws, neg_nums = predict_number_steered_maj1(
                steerer,
                prompt_text_for_selection=question,
                formatted_prompt=formatted_prompt,
                trait=trait, alpha_signed=-alpha_neg,
                num_samples=num_samples, max_new_tokens=1024,
                temperature=temperature, top_p=top_p, top_k=top_k,
                system_line=system_line,
                polarity_stability_guard=True,       # NEW: stabilize negative
                cooler_neg_decode=cooler_neg_decode  # NEW: optionally cooler decode for −α
            )

            # --- Audit row ---
            audit_f.write(json.dumps({
                "trait": trait,
                "index": i,
                "gold": gold,
                "layers_pos": layers_pos,
                "weights_pos": [round(w, 3) for w in weights_pos],
                "layers_neg": layers_neg,
                "weights_neg": [round(w, 3) for w in weights_neg],
                "pred_base": pred_base,
                "pred_pos":  pred_pos,
                "pred_neg":  pred_neg,
                # Keep only first few samples to limit file size
                "base_samples_raw": base_raws[:3],
                "pos_samples_raw":  pos_raws[:3],
                "neg_samples_raw":  neg_raws[:3],
                "base_samples_extracted": base_nums[:3],
                "pos_samples_extracted":  pos_nums[:3],
                "neg_samples_extracted":  neg_nums[:3],
            }) + "\n")

            # --- Accuracy bits ---
            b_ok = int(equal_numbers(pred_base, gold, tol=float_tolerance))
            p_ok = int(equal_numbers(pred_pos,  gold, tol=float_tolerance))
            n_ok = int(equal_numbers(pred_neg,  gold, tol=float_tolerance))
            base_correct += b_ok; pos_correct += p_ok; neg_correct += n_ok

            writer.writerow({
                "trait": trait,
                "index": i,
                "gold": gold,
                "pred_base": pred_base,
                "pred_pos":  pred_pos,
                "pred_neg":  pred_neg,
                "acc_base": b_ok,
                "acc_pos":  p_ok,
                "acc_neg":  n_ok,
                "layers_pos": ",".join(map(str, layers_pos)),
                "weights_pos": ",".join([f"{w:.3f}" for w in weights_pos]),
                "layers_neg": ",".join(map(str, layers_neg)),
                "weights_neg": ",".join([f"{w:.3f}" for w in weights_neg]),
            })
            csv_f.flush()

        per_trait_summary[trait] = {
            "n_items": n_eval,
            "acc_base": round(base_correct / max(n_eval,1), 4),
            "acc_pos":  round(pos_correct  / max(n_eval,1), 4),
            "acc_neg":  round(neg_correct  / max(n_eval,1), 4),
        }

        overall_base += base_correct
        overall_pos  += pos_correct
        overall_neg  += neg_correct
        overall_total += n_eval

    csv_f.close(); audit_f.close()

    overall = {
        "results_dir": results_dir,
        "model_name": _model_name_for_filename(steerer),
        "dataset": DATASET_TAG,
        "alpha_pos": alpha_pos,
        "alpha_neg": alpha_neg,
        "n_per_trait": n_eval,
        "num_samples": num_samples,
        "decode": {"temperature": temperature, "top_p": top_p, "top_k": top_k,
                   "cooler_neg_decode": bool(cooler_neg_decode)},
        "float_tolerance": float_tolerance,
        "overall": {
            "acc_base": round(overall_base / max(overall_total,1), 4),
            "acc_pos":  round(overall_pos  / max(overall_total,1), 4),
            "acc_neg":  round(overall_neg  / max(overall_total,1), 4),
            "total_items": int(overall_total),
        },
        "per_trait": per_trait_summary,
        "audit_log": str(audit_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print("\n=== GSM8K (8-shot CoT maj@1) summary — ALL TRAITS ===")
    print(json.dumps(overall["overall"], indent=2))
    print(f"\n[Saved] CSV:  {csv_path}")
    print(f"[Saved] JSON: {json_path}")
    print(f"[Audit] JSONL: {audit_path}")

# --------------------------
# CLI
# --------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--alpha", type=float, required=True,
                    help="Magnitude for +alpha (positive steering).")
    ap.add_argument("--alpha_neg", type=float, default=None,
                    help="Optional magnitude for negative steering (defaults to --alpha).")
    ap.add_argument("--traits", type=str, nargs="*", default=None,
                    help="Subset of traits to run. Default = all five.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all test items per trait")
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM)
    ap.add_argument("--out_dir_name", type=str, default="gsm8k_eval")
    ap.add_argument("--polarity_overrides", type=str, default="",
                    help='JSON string, e.g. \'{"openness":1,"conscientiousness":1,"extraversion":1,"agreeableness":1}\'')
    ap.add_argument("--num_samples", type=int, default=5, help="maj@1 sample count")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--float_tolerance", type=float, default=0.0)
    ap.add_argument("--exemplars_path", type=str, default="",
                    help="Optional path to 8 CoT exemplars JSON [{'question':..., 'answer':...}, ...].")
    ap.add_argument("--cooler_neg_decode", action="store_true",
                    help="Decode −alpha with slightly lower temp and top_p for fluency.")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pol = {}
    if args.polarity_overrides:
        try:
            pol = json.loads(args.polarity_overrides)
        except Exception as e:
            raise ValueError(f"Invalid polarity_overrides JSON: {e}")

    evaluate_gsm8k_cot8_alltraits(
        results_dir=args.results_dir,
        alpha_pos=float(args.alpha),
        alpha_neg=(None if args.alpha_neg is None else float(args.alpha_neg)),
        traits=args.traits,
        limit=int(args.limit),
        system_line=args.system,
        out_dir_name=args.out_dir_name,
        num_samples=int(args.num_samples),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        float_tolerance=float(args.float_tolerance),
        polarity_overrides=pol,
        exemplars_path=args.exemplars_path,
        cooler_neg_decode=bool(args.cooler_neg_decode),
    )
