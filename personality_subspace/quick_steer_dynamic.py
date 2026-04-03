
import argparse, os, json, sys
from typing import Optional, Tuple, List, Dict, Iterable

import torch
from tqdm.auto import tqdm

from .main import load_steerer
from .layer_selector import delta_logits_norms_for_prompt, SteerConfigPatch
from . import result_gpt_eval as gpteval  # optional end-of-run evaluator


# Constants / defaults
ALPHA_EFF = 6.5

MAX_NEW_TOKENS = 180
SYSTEM_TEMPLATE = (
    "You are an assistant who responds as a human. "
    "Stay natural and avoid meta-AI phrasing."
)  
INTENSITY_FOR_RUNTIME_SCAN = 6.5
WEIGHTS_HYBRID = (.7, .3)


# Helpers

@torch.no_grad()
def _paired(steerer, prompt: str, trait: str, alpha_signed: float, system: Optional[str]) -> Tuple[str, str, str]:
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, use_chat_template=True, system=system, do_sample=False)
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

def _verified_layers(steerer, trait: str) -> List[int]:
    return list(getattr(steerer, "_trait_layers", {}).get(trait.lower(), []))

def _compute_polarity_with_anchor(steerer, trait: str, verified_layers: List[int]) -> int:
    if verified_layers:
        with SteerConfigPatch(steerer, [verified_layers[0]], [1.0]):
            return steerer._calibrate_polarity(trait)
    return steerer._calibrate_polarity(trait)

def _pick_dynamic_layer(steerer, prompt: str, trait: str, exclude: Optional[int], system_msg: str) -> Tuple[Optional[int], Dict[int, float]]:
    norms = delta_logits_norms_for_prompt(
        steerer, prompt=prompt, trait=trait,
        intensity=INTENSITY_FOR_RUNTIME_SCAN, system=system_msg
    )
    dynamic_layer = None
    if norms:
        for L, _ in sorted(norms.items(), key=lambda kv: kv[1], reverse=True):
            if exclude is None or L != exclude:
                dynamic_layer = L
                break
        if dynamic_layer is None:
            dynamic_layer = max(norms, key=norms.get)
    return dynamic_layer, norms or {}

def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                print(f"[WARN] bad JSONL line in {path}: {e}", file=sys.stderr)

def _count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def _load_done_keys(out_path: str) -> set:
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                t = str(obj.get("trait","")).strip().lower()
                p = str(obj.get("prompt","")).strip()
                if t and p:
                    done.add((t, p))
            except Exception:
                pass
    return done

def _append_jsonl(out_path: str, rec: dict):
    with open(out_path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        os.fsync(fout.fileno())

# Main

def main():
    ap = argparse.ArgumentParser(description="Steer prompts from JSONL; save outputs; optional end-of-run GPT judging.")
    ap.add_argument("--results_dir", type=str, required=True, help="Folder with artifacts.pkl (and layer_verified.json)")
    ap.add_argument("--in", dest="inp", type=str, required=True, help="Input JSONL with at least {'trait','prompt'}")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL (append-per-item)")
    ap.add_argument("--mode", type=str, default="hybrid", choices=["hybrid","verified"],
                    help="hybrid=(0.70 verified + 0.30 dynamic) per prompt; verified=verified-only")
    ap.add_argument("--resume", action="store_true", help="Skip items already present in --out (by exact trait+prompt).")

    # Limit: process at most N prompts per trait (0 = all). Counts only items actually processed this run.
    ap.add_argument("--limit_per_trait", type=int, default=0)

    # Optional: run a single judging pass AFTER steering completes
    ap.add_argument("--judge_after", action="store_true",
                    help="Run GPT judging once after steering (never inside the loop).")
    ap.add_argument("--judge_model", type=str, default=None)
    ap.add_argument("--judge_out_dir", type=str, default=None)
    ap.add_argument("--judge_skip_fluency", action="store_true")
    ap.add_argument("--judge_limit_per_trait", type=int, default=0)
    ap.add_argument("--judge_no_timestamp", action="store_true",
                    help="If set, evaluator writes 'latest' files (no timestamp), overwriting previous run.")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    total = _count_lines(args.inp)

    # Load steerer
    steerer = load_steerer(args.results_dir)
    steerer.last_position_only = True   # this was true before setting this to false
    steerer.zero_center_delta = False
    steerer.log_level = "warn"
    # steerer.injection_point = "final_norm"
    _assert_hook_moves_logits(steerer, "Say hello in one word.", "openness", intensity=6.0)

    # Resume supportsteer_gain
    done_keys = _load_done_keys(args.out) if args.resume else set()
    if done_keys:
        print(f"[resume] found {len(done_keys)} completed items in {args.out}; will skip duplicates.")

    trait_polarity: Dict[str, int] = {}
    counts_processed_this_run: Dict[str, int] = {}

    with tqdm(total=total, desc="Steering prompts", unit="item") as pbar:
        for row in _iter_jsonl(args.inp):
            # required
            trait = str(row.get("trait","")).strip().lower()
            prompt = str(row.get("prompt","")).strip()

            pbar.set_postfix_str((trait or "-")[:12])

            # invalid rows
            if not trait or not prompt:
                pbar.update(1)
                continue

            key = (trait, prompt)

            # resume skip (does not consume limit)
            if key in done_keys:
                pbar.update(1)
                continue

            # enforce per-trait limit on items we will actually process now
            if args.limit_per_trait > 0:
                if counts_processed_this_run.get(trait, 0) >= args.limit_per_trait:
                    pbar.update(1)
                    continue

            # system message (kept generic)
            system_msg = SYSTEM_TEMPLATE

            # verified/dynamic selection + polarity
            verified = _verified_layers(steerer, trait)
            v_pick = verified[0] if verified else None

            if trait not in trait_polarity:
                sgn = _compute_polarity_with_anchor(steerer, trait, verified)
                trait_polarity[trait] = sgn
            else:
                sgn = trait_polarity[trait]

            alpha_signed = (ALPHA_EFF / float(steerer.steer_gain)) * sgn
            # alpha_signed = (float(steerer.steer_gain)) * sgn

            try:
                if args.mode == "verified":
                    layers = verified[:] if verified else []
                    weights = [1.0 / len(layers)] * len(layers) if layers else []
                    selection_info = {"mode": "verified", "verified_layers": layers, "weights": weights}
                    if layers:
                        with SteerConfigPatch(steerer, layers, weights):
                            base_txt, pos_txt, neg_txt = _paired(steerer, prompt, trait, alpha_signed, system_msg)
                    else:
                        base_txt, pos_txt, neg_txt = _paired(steerer, prompt, trait, alpha_signed, system_msg)
                else:
                    dyn_layer, norms = _pick_dynamic_layer(steerer, prompt, trait, v_pick, system_msg)
                    if v_pick is not None and dyn_layer is not None:
                        layers = [v_pick, dyn_layer]; weights = [WEIGHTS_HYBRID[0], WEIGHTS_HYBRID[1]]
                    elif v_pick is not None:
                        layers, weights = [v_pick], [1.0]
                    elif dyn_layer is not None:
                        layers, weights = [dyn_layer], [1.0]
                    else:
                        layers, weights = [], []
                    selection_info = {
                        "mode": "hybrid",
                        "verified_layer": v_pick,
                        "dynamic_layer": dyn_layer,
                        "dynamic_delta_l2": float(norms.get(dyn_layer, 0.0)) if dyn_layer in norms else 0.0,
                        "weights": weights
                    }
                    if layers:
                        with SteerConfigPatch(steerer, layers, weights):
                            base_txt, pos_txt, neg_txt = _paired(steerer, prompt, trait, alpha_signed, system_msg)
                    else:
                        base_txt, pos_txt, neg_txt = _paired(steerer, prompt, trait, alpha_signed, system_msg)

                # keep all non-core fields for BFI bookkeeping
                core_keys = {"trait","prompt"}
                meta = {k: v for k, v in row.items() if k not in core_keys}

                out_rec = {
                    "trait": trait,
                    "prompt": prompt,
                    "polarity": int(sgn),
                    "alpha_eff": float(ALPHA_EFF),
                    "alpha_signed": float(alpha_signed),
                    "injection_point": steerer.injection_point,
                    "last_position_only": bool(steerer.last_position_only),
                    "text_base": base_txt,
                    "text_pos": pos_txt,
                    "text_neg": neg_txt,
                    "selection": selection_info,
                    "meta": meta,
                }

            except Exception as e:
                out_rec = {"trait": trait, "prompt": prompt, "error": f"{type(e).__name__}: {e}"}

            # append + bookkeeping
            _append_jsonl(args.out, out_rec)
            done_keys.add(key)
            counts_processed_this_run[trait] = counts_processed_this_run.get(trait, 0) + 1
            pbar.update(1)

    print(f"[OK] wrote outputs → {args.out}")

    # single judging pass (optional)
    if args.judge_after and args.judge_model:
        judge_dir = args.judge_out_dir or (os.path.dirname(args.out) + "/judgments")
        os.makedirs(judge_dir, exist_ok=True)
        print(f"[judge] running GPT judging once on {args.out} with model={args.judge_model} ...")
        gpteval.run_gpt_eval_results(
            results_jsonl=args.out,
            out_dir=judge_dir,
            model=args.judge_model,
            traits=["auto"],
            limit_per_trait=args.judge_limit_per_trait,
            skip_fluency=args.judge_skip_fluency,
            no_timestamp=args.judge_no_timestamp,
        )
        print(f"[judge] results saved under {judge_dir}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
