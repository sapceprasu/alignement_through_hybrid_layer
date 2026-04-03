
import argparse, os, json, sys, csv, re
from typing import Optional, Tuple, List, Dict
from collections import defaultdict

import torch
from tqdm.auto import tqdm
from datasets import load_dataset

from ..main import load_steerer
from ..layer_selector import delta_logits_norms_for_prompt, SteerConfigPatch

# --------------------------------------------------
# Constants / defaults (kept close to quick_steer)
# --------------------------------------------------
ALPHA_EFF = 5.0
MAX_NEW_TOKENS = 180
SYSTEM_TEMPLATE = (
    "You are an assistant who answers multiple-choice questions like a careful human examinee. "
    "Read the question and options. Then output **only** a single letter (A, B, C, or D). "
    "Do not include any explanation or extra text."
)
INTENSITY_FOR_RUNTIME_SCAN = 0.7
WEIGHTS_HYBRID = (0.70, 0.30)   # verified / dynamic

CHOICE_LETTERS = ["A", "B", "C", "D"]
LETTER_RE = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)

# --------------------------------------------------
# ARC helpers
# --------------------------------------------------
def _load_arc(split: str):
    """
    Load ARC-Challenge as a HuggingFace dataset.
    split: 'train' | 'validation' | 'test'
    (Use 'validation' or 'test' for evaluation.)
    """
    ds = load_dataset("ai2_arc", "ARC-Challenge", split=split)
    if len(ds) == 0:
        raise RuntimeError(f"No rows for ARC-Challenge split={split}.")
    return ds

def _arc_row_to_qa(ex: dict) -> Tuple[str, List[str], str]:
    """
    Normalize ARC row → (question, [A..D], gold_letter)
    Fields:
      - ex['question']: string stem
      - ex['choices']: dict with keys: 'label' (list of 'A'..'D'), 'text' (list of strings)
      - ex['answerKey']: 'A'|'B'|'C'|'D'
    """
    q = str(ex.get("question") or "").strip()

    ch = ex.get("choices") or {}
    labels = ch.get("label") or []
    texts  = ch.get("text")  or []
    pairs = [(str(l), str(t)) for l, t in zip(labels, texts) if l and t]
    # sort by A,B,C,D to ensure stable ordering
    order = {k: i for i, k in enumerate(CHOICE_LETTERS)}
    pairs = sorted(pairs, key=lambda kv: order.get(kv[0], 999))
    opts = [t for _, t in pairs]

    gold = str(ex.get("answerKey") or "").strip().upper()
    return q, opts, gold

def _format_question(q: str, choices: List[str]) -> str:
    """Build a single-turn question string with options in A–D order."""
    lines = [q.strip()]
    for i, ch in enumerate(choices):
        letter = CHOICE_LETTERS[i] if i < len(CHOICE_LETTERS) else f"Option{i+1}"
        lines.append(f"{letter}. {ch.strip()}")
    lines.append("\nAnswer with one letter (A/B/C/D) only.")
    return "\n".join(lines)

def _parse_letter(text: str) -> Optional[str]:
    """Extract first A/B/C/D letter from model output."""
    if not isinstance(text, str):
        return None
    m = LETTER_RE.search(text.strip())
    if not m:
        return None
    return m.group(1).upper()

# --------------------------------------------------
# Steering helpers
# --------------------------------------------------
@torch.no_grad()
def _paired_answer(steerer, prompt: str, trait: str, alpha_signed: float, system: Optional[str]) -> Tuple[str, str, str]:
    """
    Deterministic BASE / POS / NEG generations; returns raw strings.
    """
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

def _append_jsonl(out_path: str, rec: dict):
    with open(out_path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        os.fsync(fout.fileno())

def _load_done_ids(out_path: str) -> set:
    """
    For resume: record example indices already written.
    (ARC-Challenge has no per-subject split, so we track by global index.)
    """
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
                idx = obj.get("index")
                if isinstance(idx, int):
                    done.add(idx)
            except Exception:
                pass
    return done

def _write_summary(summary_path_json: str, summary_path_csv: str, overall: Dict[str, float]):
    # JSON
    with open(summary_path_json, "w", encoding="utf-8") as f:
        json.dump({"overall": overall}, f, indent=2, ensure_ascii=False)
    # CSV
    with open(summary_path_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "base_acc", "pos_acc", "neg_acc", "pos_minus_base", "base_minus_neg", "pos_minus_neg"])
        w.writerow([
            int(overall["n"]),
            f'{overall["base_acc"]:.4f}', f'{overall["pos_acc"]:.4f}', f'{overall["neg_acc"]:.4f}',
            f'{overall["pos_minus_base"]:.4f}', f'{overall["base_minus_neg"]:.4f}', f'{overall["pos_minus_neg"]:.4f}',
        ])

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ARC-Challenge evaluation with base / +trait / -trait steering.")
    ap.add_argument("--results_dir", type=str, required=True, help="Folder with artifacts.pkl and layer_verified.json.")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path (one line per example).")
    ap.add_argument("--trait", type=str, required=True,
                    choices=["openness","conscientiousness","extraversion","agreeableness","neuroticism"],
                    help="Trait to steer.")
    ap.add_argument("--mode", type=str, default="hybrid", choices=["hybrid","verified"],
                    help="hybrid=(0.70 verified + 0.30 dynamic) per prompt; verified=verified-only")
    ap.add_argument("--split", type=str, default="validation", choices=["validation","test"],
                    help="Which ARC-Challenge split to use.")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N items (0 = all).")
    ap.add_argument("--resume", action="store_true", help="Skip examples already present in --out (by index).")

    # steering constants (aligned with quick_steer behavior)
    ap.add_argument("--alpha_eff", type=float, default=ALPHA_EFF, help="Effective alpha before steer_gain.")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Load steerer
    steerer = load_steerer(args.results_dir)
    steerer.last_position_only = True
    steerer.zero_center_delta = True
    steerer.log_level = "warn"
    _assert_hook_moves_logits(steerer, "Pick A/B/C/D only.", args.trait, intensity=6.0)

    # Resume bookkeeping
    done_ids = _load_done_ids(args.out) if args.resume else set()
    if done_ids:
        print(f"[resume] found {len(done_ids)} completed examples in {args.out}; will skip duplicates.")

    # Polarity & alpha
    trait = args.trait.strip().lower()
    verified = _verified_layers(steerer, trait)
    v_pick = verified[0] if verified else None
    if verified:
        with SteerConfigPatch(steerer, [verified[0]], [1.0]):
            sgn = steerer._calibrate_polarity(trait)
    else:
        sgn = steerer._calibrate_polarity(trait)
    # alpha_signed = (float(args.alpha_eff) / float(steerer.steer_gain)) * sgn
    alpha_signed = (float(steerer.steer_gain)) * sgn

    # Load ARC split
    ds = _load_arc(args.split)
    total_len = len(ds) if args.limit in (0, None) else min(args.limit, len(ds))

    # Accumulators
    totals = {"base": 0, "pos": 0, "neg": 0, "n": 0}

    with tqdm(total=total_len, desc=f"ARC-{args.split}", unit="q") as pbar:
        for idx, ex in enumerate(ds):
            if args.limit and totals["n"] >= args.limit:
                break

            if args.resume and idx in done_ids:
                pbar.update(1)
                continue

            q, choices, gold_letter = _arc_row_to_qa(ex)
            if not q or len(choices) < 4 or gold_letter not in CHOICE_LETTERS:
                pbar.update(1)
                continue

            system_msg = SYSTEM_TEMPLATE
            user_prompt = _format_question(q, choices)

            # Select layers (mode)
            try:
                if args.mode == "verified":
                    layers = verified[:] if verified else []
                    weights = [1.0 / len(layers)] * len(layers) if layers else []
                    selection_info = {"mode": "verified", "verified_layers": layers, "weights": weights}
                    if layers:
                        with SteerConfigPatch(steerer, layers, weights):
                            base_txt, pos_txt, neg_txt = _paired_answer(steerer, user_prompt, trait, alpha_signed, system_msg)
                    else:
                        base_txt, pos_txt, neg_txt = _paired_answer(steerer, user_prompt, trait, alpha_signed, system_msg)
                else:
                    # hybrid
                    dyn_layer, norms = _pick_dynamic_layer(steerer, user_prompt, trait, v_pick, system_msg)
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
                            base_txt, pos_txt, neg_txt = _paired_answer(steerer, user_prompt, trait, alpha_signed, system_msg)
                    else:
                        base_txt, pos_txt, neg_txt = _paired_answer(steerer, user_prompt, trait, alpha_signed, system_msg)

                # Parse letters
                base_letter = _parse_letter(base_txt)
                pos_letter  = _parse_letter(pos_txt)
                neg_letter  = _parse_letter(neg_txt)

                # Correctness
                base_ok = int(base_letter == gold_letter)
                pos_ok  = int(pos_letter  == gold_letter)
                neg_ok  = int(neg_letter  == gold_letter)

                # Accumulate
                totals["n"] += 1
                totals["base"] += base_ok
                totals["pos"]  += pos_ok
                totals["neg"]  += neg_ok

                # Write row
                rec = {
                    "index": idx,
                    "trait": trait,
                    "prompt": user_prompt,
                    "choices": choices,
                    "gold": gold_letter,
                    "text_base": base_txt,
                    "text_pos": pos_txt,
                    "text_neg": neg_txt,
                    "answer_base": base_letter,
                    "answer_pos": pos_letter,
                    "answer_neg": neg_letter,
                    "correct_base": bool(base_ok),
                    "correct_pos": bool(pos_ok),
                    "correct_neg": bool(neg_ok),
                    "alpha_eff": float(args.alpha_eff),
                    "alpha_signed": float(alpha_signed),
                    "selection": selection_info,
                }
                _append_jsonl(args.out, rec)

            except Exception as e:
                rec = {
                    "index": idx,
                    "trait": trait,
                    "prompt": user_prompt,
                    "error": f"{type(e).__name__}: {e}"
                }
                _append_jsonl(args.out, rec)

            pbar.update(1)

    # Build summary
    def _safe_div(a, b):
        return (a / b) if (b and b > 0) else 0.0

    overall = {
        "n": totals["n"],
        "base_acc": _safe_div(totals["base"], totals["n"]),
        "pos_acc":  _safe_div(totals["pos"],  totals["n"]),
        "neg_acc":  _safe_div(totals["neg"],  totals["n"]),
    }
    overall.update({
        "pos_minus_base": overall["pos_acc"] - overall["base_acc"],
        "base_minus_neg": overall["base_acc"] - overall["neg_acc"],
        "pos_minus_neg":  overall["pos_acc"] - overall["neg_acc"],
    })

    base = os.path.splitext(args.out)[0]
    _write_summary(base + ".summary.json", base + ".summary.csv", overall)

    print("\n=== ARC-Challenge steering summary ===")
    print(f"n={overall['n']}  base={overall['base_acc']:.4f}  pos={overall['pos_acc']:.4f}  neg={overall['neg_acc']:.4f}  "
          f"Δ(+−base)={overall['pos_minus_base']:.4f}  Δ(base−−)={overall['base_minus_neg']:.4f}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
