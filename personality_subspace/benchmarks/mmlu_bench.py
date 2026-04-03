# personality_subspace/mmlu_eval.py
"""
MMLU (+/- steering) evaluator using your PersonalitySteerer.

What this does
--------------
- Loads MMLU (HuggingFace: cais/mmlu) by subject.
- For each question, produces deterministic BASE / POS / NEG answers:
  * BASE: intensity=0.0
  * POS : intensity=+alpha_signed (based on trait polarity & steer_gain)
  * NEG : intensity=-alpha_signed
- Parses A/B/C/D from the model output and computes accuracy for each arm.
- Appends one JSON line per example to --out (resume-safe, flush-on-write).
- At the end, writes:
  * <out>.summary.json        (overall & per-subject accuracies)
  * <out>.summary.csv         (same in CSV)

Designed to mirror your quick_steer settings:
- deterministic generation (no sampling flags)
- hybrid or verified selection (0.70 / 0.30 by default)
- system prompt is natural but strict about "one-letter only" answer
- same polarity / layer logic and ALPHA_EFF handling

Install deps
------------
pip install datasets

Example runs
------------
# Evaluate conscientiousness steering on a small slice of 5 questions per subject (validation split)
python -m personality_subspace.mmlu_eval \
  --results_dir MetaLlama3_8B_results_20k_subspace \
  --out runs/mmlu_conscientiousness.jsonl \
  --trait conscientiousness \
  --mode hybrid \
  --split validation \
  --limit_per_subject 5

# Verified-only, full test split on a few chosen subjects
python -m personality_subspace.mmlu_eval \
  --results_dir MetaLlama3_8B_results_20k_subspace \
  --out runs/mmlu_extraversion_verified.jsonl \
  --trait extraversion \
  --mode verified \
  --split test \
  --subjects abstract_algebra college_biology \
  --limit_per_subject 25 \
  --resume

Notes
-----
- This script does *not* call GPT judging (MMLU is accuracy-based).
- You must pass --trait (openness/conscientiousness/extraversion/agreeableness/neuroticism).
- Uses deterministic decoding and your steerer’s chat template path.
"""

import argparse, os, json, sys, csv, re
from typing import Optional, Tuple, List, Dict, Iterable
from collections import defaultdict

import torch
from tqdm.auto import tqdm
from datasets import load_dataset

from ..main import load_steerer
from ..layer_selector import delta_logits_norms_for_prompt, SteerConfigPatch

# --------------------------------------------------
# Constants / defaults (kept close to your quick_steer)
# --------------------------------------------------
ALPHA_EFF = 4.0                 # aligns with your snippet
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


# ------------------------- Subjects & data -------------------------
def list_all_mmlu_subjects(split: str = "validation") -> List[str]:
    """Return the canonical list of subjects by loading the 'all' config."""
    all_ds = load_dataset("cais/mmlu", "all", split=split)
    return sorted(set(all_ds["subject"]))

def validate_subjects(requested: List[str], split: str) -> List[str]:
    available = set(list_all_mmlu_subjects(split))
    missing = [s for s in requested if s not in available]
    if missing:
        raise ValueError(
            "Invalid MMLU subject(s): " + ", ".join(missing) +
            "\nTip: run `python -c \"from datasets import load_dataset; "
            "print(sorted(set(load_dataset('cais/mmlu', 'all', split='validation')['subject'])))\"`"
        )
    return requested

def iter_subjects(split: str, subjects: Optional[List[str]]) -> List[str]:
    if subjects:
        return validate_subjects(subjects, split)
    return list_all_mmlu_subjects(split)

def load_subject(split: str, subject: str):
    """Load a single subject slice for the given split (correct HF usage)."""
    ds = load_dataset("cais/mmlu", subject, split=split)
    if len(ds) == 0:
        raise RuntimeError(f"No rows for subject={subject} split={split}.")
    return ds


# ------------------------- Prompting & parsing -------------------------
def format_question(q: str, choices: List[str]) -> str:
    """Build a single-turn question string with options in a stable format."""
    lines = [q.strip()]
    for i, ch in enumerate(choices):
        letter = CHOICE_LETTERS[i] if i < len(CHOICE_LETTERS) else f"Option{i+1}"
        lines.append(f"{letter}. {ch.strip()}")
    lines.append("\nAnswer with one letter (A/B/C/D) only.")
    return "\n".join(lines)

def parse_letter(text: str) -> Optional[str]:
    """Extract the first A/B/C/D letter from model output."""
    if not isinstance(text, str):
        return None
    m = LETTER_RE.search(text.strip())
    if not m:
        return None
    return m.group(1).upper()


# ------------------------- Steering helpers -------------------------
@torch.no_grad()
def paired_answer(steerer, prompt: str, trait: str, alpha_signed: float, system: Optional[str]) -> Tuple[str, str, str]:
    """Deterministic BASE / POS / NEG generations; returns raw strings."""
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, use_chat_template=True, system=system, do_sample=False)
    base = steerer.generate(prompt, trait, intensity=0.0, **gen_kwargs)
    plus = steerer.generate(prompt, trait, intensity=+alpha_signed, **gen_kwargs)
    minus = steerer.generate(prompt, trait, intensity=-alpha_signed, **gen_kwargs)
    return base, plus, minus

@torch.no_grad()
def assert_hook_moves_logits(steerer, prompt: str, trait: str, intensity: float = 6.0):
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

def verified_layers(steerer, trait: str) -> List[int]:
    return list(getattr(steerer, "_trait_layers", {}).get(trait.lower(), []))

def compute_polarity_with_anchor(steerer, trait: str, verified: List[int]) -> int:
    if verified:
        with SteerConfigPatch(steerer, [verified[0]], [1.0]):
            return steerer._calibrate_polarity(trait)
    return steerer._calibrate_polarity(trait)

def pick_dynamic_layer(steerer, prompt: str, trait: str, exclude: Optional[int], system_msg: str) -> Tuple[Optional[int], Dict[int, float]]:
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


# ------------------------- IO helpers -------------------------
def append_jsonl(out_path: str, rec: dict):
    with open(out_path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        os.fsync(fout.fileno())

def load_done_pairs(out_path: str) -> set:
    """For resume: record (subject, idx) pairs already written."""
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
                sub = str(obj.get("subject","")).strip()
                idx = obj.get("index")
                if sub and isinstance(idx, int):
                    done.add((sub, idx))
            except Exception:
                pass
    return done

def write_summary(summary_path_json: str, summary_path_csv: str, per_subject: Dict[str, Dict[str, float]]):
    overall = per_subject.pop("__overall__", None)
    # JSON
    blob = {"overall": overall, "per_subject": per_subject}
    with open(summary_path_json, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, ensure_ascii=False)
    # CSV
    with open(summary_path_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "n", "base_acc", "pos_acc", "neg_acc",
                    "pos_minus_base", "base_minus_neg", "pos_minus_neg"])
        if overall:
            w.writerow(["__OVERALL__", int(overall["n"]),
                        f'{overall["base_acc"]:.4f}', f'{overall["pos_acc"]:.4f}', f'{overall["neg_acc"]:.4f}',
                        f'{overall["pos_minus_base"]:.4f}', f'{overall["base_minus_neg"]:.4f}', f'{overall["pos_minus_neg"]:.4f}'])
        for sub, met in sorted(per_subject.items()):
            w.writerow([sub, int(met["n"]),
                        f'{met["base_acc"]:.4f}', f'{met["pos_acc"]:.4f}', f'{met["neg_acc"]:.4f}',
                        f'{met["pos_minus_base"]:.4f}', f'{met["base_minus_neg"]:.4f}', f'{met["pos_minus_neg"]:.4f}'])


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser(description="MMLU evaluation with base / +trait / -trait steering.")
    ap.add_argument("--results_dir", type=str, required=True, help="Folder with artifacts.pkl and layer_verified.json.")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path (one line per example).")
    ap.add_argument("--trait", type=str, required=True,
                    choices=["openness","conscientiousness","extraversion","agreeableness","neuroticism"],
                    help="Trait to steer.")
    ap.add_argument("--mode", type=str, default="hybrid", choices=["hybrid","verified"],
                    help="hybrid=(0.70 verified + 0.30 dynamic) per prompt; verified=verified-only")
    ap.add_argument("--split", type=str, default="validation", choices=["validation","test"],
                    help="Which MMLU split to use (cais/mmlu).")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="Optional list of MMLU subjects to run (default = all).")
    ap.add_argument("--limit_per_subject", type=int, default=0,
                    help="Process at most N items per subject (0 = all).")
    ap.add_argument("--resume", action="store_true", help="Skip examples already in --out (by subject,index).")
    ap.add_argument("--alpha_eff", type=float, default=ALPHA_EFF, help="Effective alpha before steer_gain.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Load steerer
    steerer = load_steerer(args.results_dir)
    steerer.last_position_only = True
    steerer.zero_center_delta = True
    steerer.log_level = "warn"
    # steerer.injection_point = "final_norm"  # uncomment if you prefer
    assert_hook_moves_logits(steerer, "Pick A/B/C/D only.", args.trait, intensity=6.0)

    # Resume bookkeeping
    done_pairs = load_done_pairs(args.out) if args.resume else set()
    if done_pairs:
        print(f"[resume] found {len(done_pairs)} completed examples in {args.out}; will skip duplicates.")

    # Polarity & alpha
    trait = args.trait.strip().lower()
    verified = verified_layers(steerer, trait)
    v_pick = verified[0] if verified else None
    sgn = compute_polarity_with_anchor(steerer, trait, verified)
    # alpha_signed = (float(args.alpha_eff) / float(steerer.steer_gain)) * sgn
    alpha_signed = (float(steerer.steer_gain)) * sgn
    # Subjects
    subjects = iter_subjects(args.split, args.subjects)
    if not subjects:
        print("[WARN] No subjects found for the given split/filters.")
        return

    # Accumulators
    totals = {"base": 0, "pos": 0, "neg": 0, "n": 0}
    per_subject_acc = defaultdict(lambda: {"base": 0, "pos": 0, "neg": 0, "n": 0})

    # Per subject loop
    for subject in subjects:
        ds = load_subject(args.split, subject)
        if len(ds) == 0:
            continue
        if args.limit_per_subject and args.limit_per_subject > 0:
            ds = ds.select(range(min(args.limit_per_subject, len(ds))))
        wrote = 0

        with tqdm(total=len(ds), desc=f"{subject}", unit="q") as pbar:
            for idx, ex in enumerate(ds):
                # Resume check
                if args.resume and (subject, idx) in done_pairs:
                    pbar.update(1)
                    continue

                # --- Normalize fields (robust to int-vs-letter answers) ---
                q = str(ex.get("question") or ex.get("input") or "").strip()

                # choices as a flat list of 4 strings
                choices = ex.get("choices")
                if not choices:
                    # rare/legacy shape; try to build from A/B/C/D keys if present
                    choices = [ex.get(k) for k in ["A", "B", "C", "D"] if ex.get(k) is not None]
                choices = [str(c) for c in (choices or [])]

                # gold may be 'A'/'B'/'C'/'D' or 0..3
                gold_raw = ex.get("answer", ex.get("target", None))
                gold_letter = None
                if isinstance(gold_raw, str):
                    g = gold_raw.strip().upper()
                    if g in ("A", "B", "C", "D"):
                        gold_letter = g
                    elif g.isdigit():
                        gi = int(g)
                        if 0 <= gi < 4:
                            gold_letter = ["A", "B", "C", "D"][gi]
                elif isinstance(gold_raw, int):
                    if 0 <= gold_raw < 4:
                        gold_letter = ["A", "B", "C", "D"][gold_raw]

                # sanity filters (with counts for diagnostics)
                bad_reason = None
                if not q:
                    bad_reason = "empty_question"
                elif len(choices) < 4:
                    bad_reason = f"choices_len_{len(choices)}"
                elif gold_letter not in ("A", "B", "C", "D"):
                    bad_reason = f"bad_gold_{gold_raw!r}"

                if bad_reason:
                    # optional: track skips; comment out if you don't want logs
                    # print(f"[skip] {subject}[{idx}] reason={bad_reason}")
                    pbar.update(1)
                    continue

                if gold_letter not in CHOICE_LETTERS or len(choices) < 4 or not q:
                    # Skip malformed rows, but still advance tqdm
                    pbar.update(1)
                    continue

                user_prompt = format_question(q, choices)
                system_msg = SYSTEM_TEMPLATE

                try:
                    # Layer selection
                    if args.mode == "verified":
                        layers = verified[:] if verified else []
                        weights = [1.0 / len(layers)] * len(layers) if layers else []
                        selection_info = {"mode": "verified", "verified_layers": layers, "weights": weights}
                        if layers:
                            with SteerConfigPatch(steerer, layers, weights):
                                base_txt, pos_txt, neg_txt = paired_answer(steerer, user_prompt, trait, alpha_signed, system_msg)
                        else:
                            base_txt, pos_txt, neg_txt = paired_answer(steerer, user_prompt, trait, alpha_signed, system_msg)
                    else:
                        # hybrid
                        dyn_layer, norms = pick_dynamic_layer(steerer, user_prompt, trait, v_pick, system_msg)
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
                                base_txt, pos_txt, neg_txt = paired_answer(steerer, user_prompt, trait, alpha_signed, system_msg)
                        else:
                            base_txt, pos_txt, neg_txt = paired_answer(steerer, user_prompt, trait, alpha_signed, system_msg)

                    # Parse letters
                    base_letter = parse_letter(base_txt)
                    pos_letter  = parse_letter(pos_txt)
                    neg_letter  = parse_letter(neg_txt)

                    # Correctness
                    base_ok = int(base_letter == gold_letter)
                    pos_ok  = int(pos_letter  == gold_letter)
                    neg_ok  = int(neg_letter  == gold_letter)

                    # Accumulate
                    totals["n"] += 1
                    totals["base"] += base_ok
                    totals["pos"]  += pos_ok
                    totals["neg"]  += neg_ok

                    per_subject_acc[subject]["n"] += 1
                    per_subject_acc[subject]["base"] += base_ok
                    per_subject_acc[subject]["pos"]  += pos_ok
                    per_subject_acc[subject]["neg"]  += neg_ok

                    # Write row
                    rec = {
                        "subject": subject,
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
                    append_jsonl(args.out, rec)
                    wrote += 1

                except Exception as e:
                    # Log error but keep going (still append a record to ease debugging)
                    rec = {
                        "subject": subject,
                        "index": idx,
                        "trait": trait,
                        "prompt": user_prompt,
                        "error": f"{type(e).__name__}: {e}"
                    }
                    append_jsonl(args.out, rec)

                pbar.update(1)

        print(f"[mmlu] subject={subject} wrote={wrote} rows -> {args.out}")

    # Build summary
    def _safe_div(a, b):
        return (a / b) if (b and b > 0) else 0.0

    per_subject_summary = {}
    for sub, acc in per_subject_acc.items():
        n = acc["n"]
        b = _safe_div(acc["base"], n)
        p = _safe_div(acc["pos"],  n)
        n_ = _safe_div(acc["neg"],  n)
        per_subject_summary[sub] = {
            "n": n,
            "base_acc": b,
            "pos_acc": p,
            "neg_acc": n_,
            "pos_minus_base": p - b,
            "base_minus_neg": b - n_,
            "pos_minus_neg":  p - n_,
        }

    overall = {
        "n": sum(v["n"] for v in per_subject_acc.values()),
        "base_acc": _safe_div(sum(v["base"] for v in per_subject_acc.values()),
                             sum(v["n"] for v in per_subject_acc.values())),
        "pos_acc":  _safe_div(sum(v["pos"]  for v in per_subject_acc.values()),
                             sum(v["n"] for v in per_subject_acc.values())),
        "neg_acc":  _safe_div(sum(v["neg"]  for v in per_subject_acc.values()),
                             sum(v["n"] for v in per_subject_acc.values())),
    }
    overall.update({
        "pos_minus_base": overall["pos_acc"] - overall["base_acc"],
        "base_minus_neg": overall["base_acc"] - overall["neg_acc"],
        "pos_minus_neg":  overall["pos_acc"] - overall["neg_acc"],
    })
    per_subject_summary["__overall__"] = overall

    base = os.path.splitext(args.out)[0]
    write_summary(base + ".summary.json", base + ".summary.csv", per_subject_summary)

    print("\n=== MMLU steering summary ===")
    print(f"Overall n={overall['n']}  base={overall['base_acc']:.4f}  pos={overall['pos_acc']:.4f}  neg={overall['neg_acc']:.4f}  "
          f"Δ(+−base)={overall['pos_minus_base']:.4f}  Δ(base−−)={overall['base_minus_neg']:.4f}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
