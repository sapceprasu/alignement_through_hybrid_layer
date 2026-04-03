import argparse, os
from typing import List, Optional, Tuple

import torch

from .main import load_steerer
from .layer_selector import delta_logits_norms_for_prompt, SteerConfigPatch

# ---------- 10 inline prompts (5 persona + 5 math from easy→hard) ----------
PROMPTS: List[str] = [
    # Persona-style
    "Describe your ideal weekend in two sentences.",
    "Give one sentence of advice for staying organized.",
    "When a new colleague joins your team, and you have a busy day with tight deadlines, how do you balance your tasks while also taking the initiative to introduce yourself and get to know them?",
    "After attending a lively event where you watched exciting races, you have the option to either join a group of friends for an after-party or go home to relax alone. What do you choose to do next and why?",
    "Complete: My friends would describe me as",
    "You spent some time with Remy's family after they lost their only son. Although you didn't know most of them, how did you engage with the group and contribute to the conversation during your time there?",
    "In one sentence, describe your communication style.",
    "You are playing basketball with a group of friends and accidentally chip someone's tooth. How do you handle the situation? Do you take the lead in addressing it, and if so, how?",
    "After giving your speech about climate change to a large and attentive audience, you notice a group of people who seem interested in discussing your ideas further. How would you approach this situation?",
    # Math (simple → harder)
    "Compute 37 + 58 and explain in one short sentence.",
    "Solve for x: 2x + 5 = 19 (one line).",
    "Is 97 a prime number? Answer yes/no and justify briefly.",
    "Differentiate f(x) = x^3 - 4x + 2 (show the derivative only).",
    "A store sells apples at $3 each and oranges at $2 each. With $20, what's the maximum apples you can buy if you must buy at least 2 oranges? Explain in one sentence.",
]

ALPHA_EFF = 4           # fixed effective alpha BEFORE steer_gain
MAX_NEW_TOKENS = 160
SYSTEM_MSG = "You are a helpful, candid assistant who writes like a human."
INTENSITY_FOR_RUNTIME_SCAN = 0.5  # small probe intensity for Δlogits norms

@torch.no_grad()
def _paired(
    steerer,
    prompt: str,
    trait: str,
    alpha_signed: float,
    max_new_tokens: int,
    system: Optional[str],
) -> Tuple[str, str, str]:
    """Deterministic baseline / + / - generations (no sampling flags)."""
    gen_kwargs = dict(max_new_tokens=max_new_tokens, use_chat_template=True, system=system, do_sample=False)
    base = steerer.generate(prompt, trait, intensity=0.0, **gen_kwargs)
    plus = steerer.generate(prompt, trait, intensity=+alpha_signed, **gen_kwargs)
    minus = steerer.generate(prompt, trait, intensity=-alpha_signed, **gen_kwargs)
    return base, plus, minus

@torch.no_grad()
def _assert_hook_moves_logits(steerer, prompt: str, trait: str, intensity: float = 6.0):
    txt = steerer._format_prompt(prompt, use_chat_template=True, system=None)
    enc = steerer.tok(txt, return_tensors="pt").to(steerer.device)
    base = steerer.model(**enc).logits[:, -1, :].float()
    steerer._register(trait, intensity)
    try:
        hooked = steerer.model(**enc).logits[:, -1, :].float()
    finally:
        steerer._clear()
    diff = hooked - base
    l2 = float(torch.norm(diff, p=2).item())
    mam = float(diff.abs().max().item())
    print(f"[sanity] Δlogits L2={l2:.4f}  max|Δ|={mam:.4f}")

def _verified_layers(steerer, trait: str):
    return list(getattr(steerer, "_trait_layers", {}).get(trait.lower(), []))

def _compute_polarity(steerer, trait: str, verified_layers: list) -> int:
    if verified_layers:
        with SteerConfigPatch(steerer, [verified_layers[0]], [1.0]):
            return steerer._calibrate_polarity(trait)
    return steerer._calibrate_polarity(trait)

def _write_header(f, title: str, results_dir: str, trait: str, verified_layers, sgn: int, alpha_signed: float):
    f.write(f"# {title}\n")
    f.write(f"# results_dir: {results_dir}\n")
    f.write(f"# trait: {trait}\n")
    f.write(f"# verified_layers: {verified_layers if verified_layers else 'None'}\n")
    f.write(f"# polarity: {'+1' if sgn>0 else '-1'}\n")
    f.write(f"# alpha_eff: {ALPHA_EFF}  (alpha_signed={alpha_signed:.4f})\n")
    f.write(f"# deterministic decoding, max_new_tokens={MAX_NEW_TOKENS}\n\n")

def _run_verified_only(steerer, trait: str, verified_layers: list, alpha_signed: float, out_path: str):
    """Use ONLY verified layers (uniform weights)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        _write_header(f, "QuickSteer (verified-only)", steerer.cfg.results_dir, trait, verified_layers, 1, alpha_signed)
        if not verified_layers:
            f.write("[WARN] No verified layers found; skipping verified-only runs.\n")
            return
        weights = [1.0 / len(verified_layers)] * len(verified_layers)
        for i, prompt in enumerate(PROMPTS, start=1):
            f.write(f"\n===================== PROMPT {i} =====================\n")
            f.write(prompt.strip() + "\n")
            with SteerConfigPatch(steerer, verified_layers, weights):
                base, up, down = _paired(
                    steerer, prompt, trait, alpha_signed,
                    max_new_tokens=MAX_NEW_TOKENS, system=SYSTEM_MSG
                )
            f.write("\n[Baseline]\n")
            f.write(base.strip() + "\n")
            f.write(f"\n[Steered +{trait}]\n")
            f.write(up.strip() + "\n")
            f.write(f"\n[Steered -{trait}]\n")
            f.write(down.strip() + "\n")
    print(f"[OK] wrote (verified-only) → {out_path}")

def _run_verified_plus_dynamic(steerer, trait: str, v_pick: Optional[int], alpha_signed: float, out_path: str):
    """Per-prompt: add the most reactive runtime layer by Δlogits L2; 50/50 weights with verified."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        _write_header(f, "QuickSteer (verified + dynamic 50/50)", steerer.cfg.results_dir, trait, [v_pick] if v_pick is not None else [], 1, alpha_signed)
        for i, prompt in enumerate(PROMPTS, start=1):
            # pick dynamic layer
            norms = delta_logits_norms_for_prompt(
                steerer, prompt=prompt, trait=trait,
                intensity=INTENSITY_FOR_RUNTIME_SCAN, system=SYSTEM_MSG
            )
            dynamic_layer = None
            if norms:
                for L, _ in sorted(norms.items(), key=lambda kv: kv[1], reverse=True):
                    if v_pick is None or L != v_pick:
                        dynamic_layer = L
                        break
                if dynamic_layer is None:
                    dynamic_layer = max(norms, key=norms.get)

            # choose mix
            if v_pick is not None and dynamic_layer is not None:
                layers = [v_pick, dynamic_layer]
                weights = [0.75, 0.25]
            elif v_pick is not None:
                layers, weights = [v_pick], [1.0]
            elif dynamic_layer is not None:
                layers, weights = [dynamic_layer], [1.0]
            else:
                layers, weights = [], []

            f.write(f"\n===================== PROMPT {i} =====================\n")
            f.write(prompt.strip() + "\n")
            f.write(f"[selection] verified={v_pick} dynamic={dynamic_layer} (ΔL2={norms.get(dynamic_layer, 0.0):.3f} if chosen) weights={weights}\n")

            if layers:
                with SteerConfigPatch(steerer, layers, weights):
                    base, up, down = _paired(
                        steerer, prompt, trait, alpha_signed,
                        max_new_tokens=MAX_NEW_TOKENS, system=SYSTEM_MSG
                    )
            else:
                base, up, down = _paired(
                    steerer, prompt, trait, alpha_signed,
                    max_new_tokens=MAX_NEW_TOKENS, system=SYSTEM_MSG
                )
            f.write("\n[Baseline]\n")
            f.write(base.strip() + "\n")
            f.write(f"\n[Steered +{trait}]\n")
            f.write(up.strip() + "\n")
            f.write(f"\n[Steered -{trait}]\n")
            f.write(down.strip() + "\n")
    print(f"[OK] wrote (verified + dynamic) → {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Quick steer: verified-only AND verified+dynamic (50/50)")
    ap.add_argument("--results_dir", type=str, required=True, help="Folder with artifacts.pkl (and layer_verified.json)")
    ap.add_argument("--trait", type=str, required=True, help="openness/conscientiousness/extraversion/agreeableness/neuroticism")
    ap.add_argument("--out_base", type=str, required=True, help="Output base path (script writes two files with suffixes)")
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(args.out_base))
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)

    # Load steerer (prefers verified layers automatically)
    steerer = load_steerer(args.results_dir)
    steerer.last_position_only = True     # stable & fast
    steerer.zero_center_delta = True
    steerer.log_level = "warn"

    # Quick sanity
    _assert_hook_moves_logits(steerer, "Say hello in one word.", args.trait, intensity=6.0)

    # Verified layers & polarity
    verified = _verified_layers(steerer, args.trait)
    v_pick = verified[0] if verified else None
    sgn = _compute_polarity(steerer, args.trait, verified)
    alpha_signed = (ALPHA_EFF / float(steerer.steer_gain)) * sgn

    # A) Verified-only
    out_verified = f"{args.out_base}__verified.txt"
    _run_verified_only(steerer, args.trait, verified, alpha_signed, out_verified)

    # B) Verified + dynamic
    out_mixed = f"{args.out_base}__mixed.txt"
    _run_verified_plus_dynamic(steerer, args.trait, v_pick, alpha_signed, out_mixed)

    print(f"[DONE] wrote:\n  {out_verified}\n  {out_mixed}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# """
# Steer a list of (trait, prompt) from prompts.jsonl and save rich outputs, with tqdm and per-item append.

# Modes:
#   - hybrid  : verified layer + runtime argmax Δlogits L2  (weights 0.75 / 0.25)
#   - verified: verified-only (uniform over verified layers)

# Deterministic decoding (no sampling flags). Appends one JSON line per item, flushes to disk immediately,
# and supports --resume to skip items already processed.

# Optionally, run GPT judging inline: after each steering output, immediately call GPT judge and save results.
#   --judge_model gpt-4o-mini [--judge_out_dir ...] [--judge_skip_fluency] [--judge_limit_per_trait N]
# """

# import argparse, os, json, sys, time
# from typing import Optional, Tuple, List, Dict, Iterable

# import torch
# from tqdm.auto import tqdm

# from .main import load_steerer
# from .layer_selector import delta_logits_norms_for_prompt, SteerConfigPatch
# from . import result_gpt_eval as gpteval  # <-- import eval engine

# # --------------------------------------------------
# # Constants / defaults
# # --------------------------------------------------
# ALPHA_EFF = 4.5
# MAX_NEW_TOKENS = 180
# SYSTEM_TEMPLATE = (
#     "You are a helpful, candid assistant who responds as a human. "
#     "Stay natural and avoid meta-AI phrasing."
# )
# INTENSITY_FOR_RUNTIME_SCAN = 0.7
# WEIGHTS_HYBRID = (0.70, 0.30)

# # --------------------------------------------------
# # Helpers
# # --------------------------------------------------
# @torch.no_grad()
# def _paired(steerer, prompt: str, trait: str, alpha_signed: float, system: Optional[str]) -> Tuple[str, str, str]:
#     gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, use_chat_template=True, system=system, do_sample=False)
#     base = steerer.generate(prompt, trait, intensity=0.0, **gen_kwargs)
#     plus = steerer.generate(prompt, trait, intensity=+alpha_signed, **gen_kwargs)
#     minus = steerer.generate(prompt, trait, intensity=-alpha_signed, **gen_kwargs)
#     return base, plus, minus

# @torch.no_grad()
# def _assert_hook_moves_logits(steerer, prompt: str, trait: str, intensity: float = 6.0):
#     txt = steerer._format_prompt(prompt, use_chat_template=True, system=None)
#     enc = steerer.tok(txt, return_tensors="pt").to(steerer.device)
#     base = steerer.model(**enc).logits[:, -1, :].float()
#     steerer._register(trait, intensity)
#     try:
#         hooked = steerer.model(**enc).logits[:, -1, :].float()
#     finally:
#         steerer._clear()
#     diff = hooked - base
#     l2 = float(torch.norm(diff, p=2).item())
#     mam = float(diff.abs().max().item())
#     print(f"[sanity] Δlogits L2={l2:.4f}  max|Δ|={mam:.4f}")

# def _verified_layers(steerer, trait: str) -> List[int]:
#     return list(getattr(steerer, "_trait_layers", {}).get(trait.lower(), []))

# def _compute_polarity_with_anchor(steerer, trait: str, verified_layers: List[int]) -> int:
#     if verified_layers:
#         with SteerConfigPatch(steerer, [verified_layers[0]], [1.0]):
#             return steerer._calibrate_polarity(trait)
#     return steerer._calibrate_polarity(trait)

# def _pick_dynamic_layer(steerer, prompt: str, trait: str, exclude: Optional[int], system_msg: str) -> Tuple[Optional[int], Dict[int, float]]:
#     norms = delta_logits_norms_for_prompt(
#         steerer, prompt=prompt, trait=trait,
#         intensity=INTENSITY_FOR_RUNTIME_SCAN, system=system_msg
#     )
#     dynamic_layer = None
#     if norms:
#         for L, _ in sorted(norms.items(), key=lambda kv: kv[1], reverse=True):
#             if exclude is None or L != exclude:
#                 dynamic_layer = L
#                 break
#         if dynamic_layer is None:
#             dynamic_layer = max(norms, key=norms.get)
#     return dynamic_layer, norms

# def _iter_jsonl(path: str) -> Iterable[dict]:
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             s = line.strip()
#             if not s:
#                 continue
#             try:
#                 yield json.loads(s)
#             except Exception as e:
#                 print(f"[WARN] bad JSONL line in {path}: {e}", file=sys.stderr)

# def _count_lines(path: str) -> int:
#     with open(path, "r", encoding="utf-8") as f:
#         return sum(1 for _ in f)

# def _load_done_keys(out_path: str) -> set:
#     done = set()
#     if not os.path.exists(out_path):
#         return done
#     with open(out_path, "r", encoding="utf-8") as f:
#         for line in f:
#             s = line.strip()
#             if not s:
#                 continue
#             try:
#                 obj = json.loads(s)
#                 t = str(obj.get("trait","")).strip().lower()
#                 p = str(obj.get("prompt","")).strip()
#                 if t and p:
#                     done.add((t, p))
#             except Exception:
#                 pass
#     return done

# def _append_jsonl(out_path: str, rec: dict):
#     with open(out_path, "a", encoding="utf-8") as fout:
#         fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
#         fout.flush()
#         os.fsync(fout.fileno())

# # --------------------------------------------------
# # Main
# # --------------------------------------------------
# def main():
#     ap = argparse.ArgumentParser(description="Steer prompts from JSONL; save outputs; optional inline GPT judging.")
#     ap.add_argument("--results_dir", type=str, required=True)
#     ap.add_argument("--in", dest="inp", type=str, required=True)
#     ap.add_argument("--out", type=str, required=True)
#     ap.add_argument("--mode", type=str, default="hybrid", choices=["hybrid","verified"])
#     ap.add_argument("--resume", action="store_true")

#     ap.add_argument("--limit_per_trait", type=int, default=0)

#     # inline judge
#     ap.add_argument("--judge_model", type=str, default=None)
#     ap.add_argument("--judge_out_dir", type=str, default=None)
#     ap.add_argument("--judge_skip_fluency", action="store_true")
#     ap.add_argument("--judge_limit_per_trait", type=int, default=0)

#     args = ap.parse_args()
#     os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

#     total = _count_lines(args.inp)
#     steerer = load_steerer(args.results_dir)
#     steerer.last_position_only = True
#     steerer.zero_center_delta = True
#     steerer.log_level = "warn"
#     steerer.injection_point = "final_norm"
#     _assert_hook_moves_logits(steerer, "Say hello in one word.", "openness", intensity=6.0)

#     done_keys = _load_done_keys(args.out) if args.resume else set()
#     trait_polarity: Dict[str, int] = {}
#     counts_per_trait: Dict[str, int] = {}

#     with tqdm(total=total, desc="Steering prompts", unit="item") as pbar:
#         for row in _iter_jsonl(args.inp):
#             trait = str(row.get("trait","")).strip().lower()
#             prompt = str(row.get("prompt","")).strip()
#             pbar.set_postfix_str(trait[:10] if trait else "-")

#             if not trait or not prompt:
#                 pbar.update(1); continue

#             if args.limit_per_trait > 0:
#                 c = counts_per_trait.get(trait, 0)
#                 if c >= args.limit_per_trait:
#                     pbar.update(1); continue
#                 counts_per_trait[trait] = c + 1

#             key = (trait, prompt)
#             if key in done_keys:
#                 pbar.update(1); continue

#             system_msg = SYSTEM_TEMPLATE.format(trait=trait)
#             verified = _verified_layers(steerer, trait)
#             v_pick = verified[0] if verified else None

#             if trait not in trait_polarity:
#                 sgn = _compute_polarity_with_anchor(steerer, trait, verified)
#                 trait_polarity[trait] = sgn
#             else:
#                 sgn = trait_polarity[trait]

#             alpha_signed = (ALPHA_EFF / float(steerer.steer_gain)) * sgn

#             try:
#                 if args.mode == "verified":
#                     layers = verified[:] if verified else []
#                     weights = [1.0 / len(layers)] * len(layers) if layers else []
#                     selection_info = {"mode": "verified", "verified_layers": layers, "weights": weights}
#                     if layers:
#                         with SteerConfigPatch(steerer, layers, weights):
#                             base_txt, pos_txt, neg_txt = _paired(steerer, prompt, trait, alpha_signed, system_msg)
#                     else:
#                         base_txt, pos_txt, neg_txt = _paired(steerer, prompt, trait, alpha_signed, system_msg)
#                 else:
#                     dyn_layer, norms = _pick_dynamic_layer(steerer, prompt, trait, v_pick, system_msg)
#                     if v_pick is not None and dyn_layer is not None:
#                         layers = [v_pick, dyn_layer]; weights = [WEIGHTS_HYBRID[0], WEIGHTS_HYBRID[1]]
#                     elif v_pick is not None:
#                         layers, weights = [v_pick], [1.0]
#                     elif dyn_layer is not None:
#                         layers, weights = [dyn_layer], [1.0]
#                     else:
#                         layers, weights = [], []
#                     selection_info = {
#                         "mode": "hybrid",
#                         "verified_layer": v_pick,
#                         "dynamic_layer": dyn_layer,
#                         "dynamic_delta_l2": float(norms.get(dyn_layer, 0.0)) if isinstance(norms, dict) and dyn_layer in norms else 0.0,
#                         "weights": weights
#                     }
#                     if layers:
#                         with SteerConfigPatch(steerer, layers, weights):
#                             base_txt, pos_txt, neg_txt = _paired(steerer, prompt, trait, alpha_signed, system_msg)
#                     else:
#                         base_txt, pos_txt, neg_txt = _paired(steerer, prompt, trait, alpha_signed, system_msg)

#                 out_rec = {
#                     "trait": trait, "prompt": prompt,
#                     "polarity": int(sgn),
#                     "alpha_eff": float(ALPHA_EFF),
#                     "alpha_signed": float(alpha_signed),
#                     "injection_point": steerer.injection_point,
#                     "last_position_only": bool(steerer.last_position_only),
#                     "text_base": base_txt, "text_pos": pos_txt, "text_neg": neg_txt,
#                     "selection": selection_info,
#                 }
#             except Exception as e:
#                 out_rec = {"trait": trait, "prompt": prompt, "error": f"{type(e).__name__}: {e}"}

#             # write output
#             _append_jsonl(args.out, out_rec)
#             done_keys.add(key)

#             # inline judge
#             if args.judge_model:
#                 judge_dir = args.judge_out_dir or (os.path.dirname(args.out) + "/judgments")
#                 os.makedirs(judge_dir, exist_ok=True)
#                 gpteval.run_gpt_eval_results(
#                     results_jsonl=args.out,
#                     out_dir=judge_dir,
#                     model=args.judge_model,
#                     traits=["auto"],
#                     limit_per_trait=args.judge_limit_per_trait,
#                     skip_fluency=args.judge_skip_fluency,
#                 )

#             pbar.update(1)

#     print(f"[OK] wrote outputs → {args.out}")
#     if args.judge_model:
#         print(f"[judge] results saved under {args.judge_out_dir or (os.path.dirname(args.out) + '/judgments')}")


# if __name__ == "__main__":
#     torch.set_grad_enabled(False)
#     os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
#     os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
#     os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
#     main()
