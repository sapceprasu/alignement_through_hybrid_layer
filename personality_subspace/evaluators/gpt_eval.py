import os, re, json, argparse, time, math, glob
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env

# ---------- trait factors (shown in the judge prompt) ----------
TRAIT_FACTORS: Dict[str, str] = {
    "openness": (
        "curiosity, imagination, creativity, intellectual exploration, appreciation of art and beauty, "
        "and willingness to try new experiences"
    ),
    "conscientiousness": (
        "organization, diligence, reliability, goal orientation, self-discipline, and careful planning"
    ),
    "extraversion": (
        "sociability, assertiveness, enthusiasm, positive affect, talkativeness, and energy in social settings"
    ),
    "agreeableness": (
        "empathy, kindness, cooperativeness, trust, altruism, and consideration for others"
    ),
    "neuroticism": (
        "emotional instability, anxiety, moodiness, vulnerability to stress, and negative affect"
    ),
}

JUDGE_TEMPLATE = (
    "Please act as an impartial judge and evaluate the personality trait of {BFI} exhibited in the response "
    "provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as "
    "{factors} demonstrated in the response.\n\n"
    "Begin your evaluation by providing a short explanation of no more than 30 words. Be as objective as possible. "
    "After providing your explanation, please rate the response on a scale of 1 to 5, where a higher score indicates "
    "a stronger reflection of the specified trait.\n\n"
    "Please follow this format for your rating: “[[rating]]”, for example: “Rating: [[5]]”.\n"
    "[Question]{question}[The Start of Assistant’s Answer]{answer}[The End of Assistant’s Answer]"
)

# ---------- small utils ----------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError as e:
                print(f"[WARN] bad json in {path.name}: {e}")
    return out

# parse question from templated prompt
_Q_RE = re.compile(r"###\s*Question:\s*(.*?)\s*###\s*Response:", re.DOTALL | re.IGNORECASE)

def _extract_question(item: Dict[str, Any]) -> str:
    # prefer explicit field if present
    q = (item.get("question") or "").strip()
    if q:
        return q
    # handle either 'prompt' or 'prompt_full'
    p = (item.get("prompt") or item.get("prompt_full") or "").strip()
    if p:
        m = _Q_RE.search(p)
        if m:
            return m.group(1).strip()
        return p[:400]
    return ""

# unify fields across both schemas
def _answers_from_row(r: Dict[str, Any]) -> Tuple[str, str, str]:
    # flat
    base = (r.get("base_text") or "").strip()
    pos  = (r.get("pos_text") or r.get("positive_text") or "").strip()
    neg  = (r.get("neg_text") or r.get("negative_text") or "").strip()
    # nested
    if not (base or pos or neg):
        tx = r.get("texts") or {}
        base = (tx.get("base") or "").strip()
        pos  = (tx.get("positive") or "").strip()
        neg  = (tx.get("negative") or "").strip()
    return base, pos, neg

def _floatsafe(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        v = float(x)
        if math.isnan(v): return None
        return v
    except Exception:
        return None

def _alpha_from_row(r: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    a_pos = r.get("alpha_pos")
    a_neg = r.get("alpha_neg")
    if a_pos is None or a_neg is None:
        a = r.get("alpha") or {}
        a_pos = a_pos if a_pos is not None else a.get("pos")
        a_neg = a_neg if a_neg is not None else a.get("neg")
    return _floatsafe(a_pos), _floatsafe(a_neg)

def _kl_from_row(r: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    fk_pos = r.get("first_token_kl_pos")
    fk_neg = r.get("first_token_kl_neg")
    ck_pos = r.get("cumulative_kl_pos")
    ck_neg = r.get("cumulative_kl_neg")
    if any(v is None for v in (fk_pos, fk_neg, ck_pos, ck_neg)):
        f = r.get("first_token_kl") or {}
        c = r.get("cumulative_kl") or {}
        fk_pos = fk_pos if fk_pos is not None else f.get("pos")
        fk_neg = fk_neg if fk_neg is not None else f.get("neg")
        ck_pos = ck_pos if ck_pos is not None else c.get("pos")
        ck_neg = ck_neg if ck_neg is not None else c.get("neg")
    return _floatsafe(fk_pos), _floatsafe(fk_neg), _floatsafe(ck_pos), _floatsafe(ck_neg)

# rating parser (accepts [[3]] anywhere)
_RATING_RE = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*\]")

def _parse_rating(text: str) -> Optional[float]:
    if not text:
        return None
    m = _RATING_RE.search(text)
    if not m:
        m = re.search(r"Rating\s*:\s*\[\[\s*(\d+(?:\.\d+)?)\s*\]\]", text, flags=re.I)
        if not m:
            return None
    try:
        v = float(m.group(1))
        return float(np.clip(v, 1.0, 5.0))
    except Exception:
        return None

# ---------- find latest benchmark JSONL for an injection mode ----------
def _latest_bench_jsonl(results_dir: Path, inj: str) -> Optional[Path]:
    candidates: List[Path] = []
    # simple layout
    candidates += list((results_dir / f"benchmark_{inj}").glob(f"bench_{inj}_*.jsonl"))
    # mainlike layout
    # candidates += list((results_dir / f"benchmarks_mainlike_{inj}").glob(f"bench_{inj}_*.jsonl"))
    # last-resort search
    if not candidates:
        candidates += list(results_dir.rglob(f"bench_{inj}_*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

# ---------- OpenAI client ----------
def _openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Install `openai` (>=1.0) and set OPENAI_API_KEY") from e
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI()

def _judge_once(client, model: str, trait: str, question: str, answer: str,
                retries: int = 3, sleep: float = 1.5) -> Dict[str, Any]:
    prompt = JUDGE_TEMPLATE.format(
        BFI=trait.title(),
        factors=TRAIT_FACTORS.get(trait.lower(), "core behavioral indicators of the specified trait"),
        question=question,
        answer=answer,
    )
    last_err = None
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a meticulous, impartial research rater."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            rating = _parse_rating(text)
            # take first line as the "explanation" (judge is asked to keep ≤30 words)
            expl = text.splitlines()[0].strip() if text else ""
            return {"raw": text, "explanation": expl, "rating": rating}
        except Exception as e:
            last_err = e
            time.sleep(sleep)
    return {"raw": None, "explanation": None, "rating": None, "error": str(last_err) if last_err else "unknown"}

# ---------- stats ----------
def _mean_var(xs: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    arr = np.array([x for x in xs if isinstance(x, (int, float)) and np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return None, None, 0
    return float(arr.mean()), float(arr.var(ddof=0)), int(arr.size)

# ---------- main processing ----------
def _auto_traits(rows: List[Dict[str, Any]]) -> List[str]:
    ts = sorted({(r.get("trait") or "").strip().lower() for r in rows if r.get("trait")})
    # keep only known traits; if unknowns exist, include them with generic factors
    known = [t for t in ts if t]
    return known

def run_gpt_eval(results_dir: str, inj_list: List[str], model: str, traits: List[str], limit_per_trait: int):
    client = _openai_client()

    for inj in inj_list:
        bench = _latest_bench_jsonl(Path(results_dir), inj)
        if not bench:
            print(f"[WARN] no benchmark JSONL found for inj={inj} under {results_dir}")
            continue
        print(f"[INFO] Using benchmark file: {bench}")
        rows = _read_jsonl(bench)
        print(f"[INFO] Loaded {len(rows)} rows.")

        # dynamic traits
        if not traits or (len(traits) == 1 and traits[0].lower() == "auto"):
            traits_run = _auto_traits(rows)
        else:
            traits_run = [t.lower() for t in traits]

        if not traits_run:
            print(f"[WARN] No traits detected in file. Nothing to do for inj={inj}.")
            continue

        out_dir = Path(results_dir) / "judgments" / inj
        _ensure_dir(out_dir)

        combined_summary: List[Dict[str, Any]] = []

        # group rows by trait (won't fail if some traits missing)
        rows_by_trait: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            t = (r.get("trait") or "").strip().lower()
            if not t:
                continue
            rows_by_trait.setdefault(t, []).append(r)

        for trait in traits_run:
            trait_rows = rows_by_trait.get(trait, [])
            if not trait_rows:
                print(f"[INFO] inj={inj}: no rows for trait={trait}; skipping.")
                continue

            if limit_per_trait and len(trait_rows) > limit_per_trait:
                trait_rows = trait_rows[:limit_per_trait]

            print(f"\n=== [{inj}] Trait: {trait} | judging {len(trait_rows)} items ===")

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_jsonl = out_dir / f"{trait}_base_{stamp}.jsonl"
            pos_jsonl  = out_dir / f"{trait}_pos_{stamp}.jsonl"
            neg_jsonl  = out_dir / f"{trait}_neg_{stamp}.jsonl"
            merged_jsonl = out_dir / f"{trait}_merged_{stamp}.jsonl"

            base_csv = out_dir / f"{trait}_base_{stamp}.csv"
            pos_csv  = out_dir / f"{trait}_pos_{stamp}.csv"
            neg_csv  = out_dir / f"{trait}_neg_{stamp}.csv"
            merged_csv = out_dir / f"{trait}_merged_{stamp}.csv"

            # open files
            bjf = open(base_jsonl, "w", encoding="utf-8")
            pjf = open(pos_jsonl,  "w", encoding="utf-8")
            njf = open(neg_jsonl,  "w", encoding="utf-8")
            mjf = open(merged_jsonl,"w", encoding="utf-8")

            import csv
            bcsv = csv.DictWriter(open(base_csv, "w", encoding="utf-8", newline=""),
                                  fieldnames=["inj","trait","index","question","base_rating","base_explanation"])
            pcsv = csv.DictWriter(open(pos_csv,  "w", encoding="utf-8", newline=""),
                                  fieldnames=["inj","trait","index","question","pos_rating","pos_explanation"])
            ncsv = csv.DictWriter(open(neg_csv,  "w", encoding="utf-8", newline=""),
                                  fieldnames=["inj","trait","index","question","neg_rating","neg_explanation"])
            mcsv = csv.DictWriter(open(merged_csv,"w", encoding="utf-8", newline=""),
                                  fieldnames=["inj","trait","index","question",
                                              "base_rating","pos_rating","neg_rating",
                                              "base_explanation","pos_explanation","neg_explanation"])
            for w in (bcsv, pcsv, ncsv, mcsv): w.writeheader()

            base_scores: List[Optional[float]] = []
            pos_scores:  List[Optional[float]] = []
            neg_scores:  List[Optional[float]] = []

            for r in tqdm(trait_rows, desc=f"{inj}:{trait}", leave=False):
                idx = r.get("index")
                q = _extract_question(r)
                base_ans, pos_ans, neg_ans = _answers_from_row(r)
                a_pos, a_neg = _alpha_from_row(r)
                fk_pos, fk_neg, ck_pos, ck_neg = _kl_from_row(r)

                row_meta = {
                    "inj": inj, "trait": trait, "index": idx, "question": q,
                    "alpha_pos": a_pos, "alpha_neg": a_neg,
                    "first_token_kl_pos": fk_pos, "first_token_kl_neg": fk_neg,
                    "cumulative_kl_pos": ck_pos, "cumulative_kl_neg": ck_neg,
                }

                # base
                base_rating = None; base_expl = None; base_raw = None
                if base_ans:
                    jr = _judge_once(client, model, trait, q, base_ans)
                    base_rating = jr["rating"]; base_expl = jr["explanation"]; base_raw = jr["raw"]
                base_scores.append(base_rating)
                bjf.write(json.dumps({**row_meta, "answer": base_ans, "base_rating": base_rating,
                                      "base_explanation": base_expl, "judge_raw": base_raw},
                                     ensure_ascii=False) + "\n")
                bcsv.writerow({"inj": inj, "trait": trait, "index": idx, "question": q,
                               "base_rating": "" if base_rating is None else f"{base_rating:.3f}",
                               "base_explanation": (base_expl or "")})

                # positive
                pos_rating = None; pos_expl = None; pos_raw = None
                if pos_ans:
                    jr = _judge_once(client, model, trait, q, pos_ans)
                    pos_rating = jr["rating"]; pos_expl = jr["explanation"]; pos_raw = jr["raw"]
                pos_scores.append(pos_rating)
                pjf.write(json.dumps({**row_meta, "answer": pos_ans, "pos_rating": pos_rating,
                                      "pos_explanation": pos_expl, "judge_raw": pos_raw},
                                     ensure_ascii=False) + "\n")
                pcsv.writerow({"inj": inj, "trait": trait, "index": idx, "question": q,
                               "pos_rating": "" if pos_rating is None else f"{pos_rating:.3f}",
                               "pos_explanation": (pos_expl or "")})

                # negative
                neg_rating = None; neg_expl = None; neg_raw = None
                if neg_ans:
                    jr = _judge_once(client, model, trait, q, neg_ans)
                    neg_rating = jr["rating"]; neg_expl = jr["explanation"]; neg_raw = jr["raw"]
                neg_scores.append(neg_rating)
                njf.write(json.dumps({**row_meta, "answer": neg_ans, "neg_rating": neg_rating,
                                      "neg_explanation": neg_expl, "judge_raw": neg_raw},
                                     ensure_ascii=False) + "\n")
                ncsv.writerow({"inj": inj, "trait": trait, "index": idx, "question": q,
                               "neg_rating": "" if neg_rating is None else f"{neg_rating:.3f}",
                               "neg_explanation": (neg_expl or "")})

                # merged view row
                mjf.write(json.dumps({
                    **row_meta,
                    "base_rating": base_rating, "pos_rating": pos_rating, "neg_rating": neg_rating,
                    "base_explanation": base_expl, "pos_explanation": pos_expl, "neg_explanation": neg_expl,
                    "base_text": base_ans, "pos_text": pos_ans, "neg_text": neg_ans,
                }, ensure_ascii=False) + "\n")
                mcsv.writerow({
                    "inj": inj, "trait": trait, "index": idx, "question": q,
                    "base_rating": "" if base_rating is None else f"{base_rating:.3f}",
                    "pos_rating":  "" if pos_rating  is None else f"{pos_rating:.3f}",
                    "neg_rating":  "" if neg_rating  is None else f"{neg_rating:.3f}",
                    "base_explanation": base_expl or "",
                    "pos_explanation":  pos_expl  or "",
                    "neg_explanation":  neg_expl  or "",
                })

            # close files
            for f in (bjf, pjf, njf, mjf): f.close()

            # per-trait summary
            mb, vb, nb = _mean_var(base_scores)
            mp, vp, np_ = _mean_var(pos_scores)
            mn, vn, nn_ = _mean_var(neg_scores)

            # “sum of opposing aspects” (you asked for mean & variance summed)
            combined_sum_mean = None if (mp is None or mn is None) else float(mp + mn)
            combined_sum_variance = None if (vp is None or vn is None) else float(vp + vn)

            # also report an inverted version (common in papers): pos + (6 - neg)
            combined_inverted_mean = None if (mp is None or mn is None) else float(mp + (6.0 - mn))
            combined_inverted_variance = combined_sum_variance  # same variance under (6 - x)

            summary = {
                "inj": inj,
                "trait": trait,
                "n": {"base": nb, "pos": np_, "neg": nn_},
                "mean": {"base": mb, "pos": mp, "neg": mn},
                "variance": {"base": vb, "pos": vp, "neg": vn},
                "deltas": {
                    "pos_minus_base": None if (mp is None or mb is None) else float(mp - mb),
                    "base_minus_neg": None if (mb is None or mn is None) else float(mb - mn),
                    "pos_minus_neg":  None if (mp is None or mn is None) else float(mp - mn),
                },
                "combined": {
                    "mean_sum_raw": combined_sum_mean,
                    "mean_sum_inverted": combined_inverted_mean,
                    "variance_sum": combined_sum_variance,
                    "variance_sum_inverted": combined_inverted_variance,
                    "note": "mean_sum_raw = mean_pos + mean_neg; mean_sum_inverted = mean_pos + (6 - mean_neg); variance_sum = var_pos + var_neg.",
                },
            }

            (out_dir / f"{trait}_summary_{stamp}.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
            combined_summary.append(summary)

        # combined summary for this inj
        (Path(results_dir) / "judgments" / inj / f"_combined_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
         ).write_text(json.dumps(combined_summary, indent=2, ensure_ascii=False))

        # pretty print
        print(f"\n=== Summary ({inj}) ===")
        for s in combined_summary:
            t = s["trait"]
            mb = s["mean"]["base"]; mp = s["mean"]["pos"]; mn = s["mean"]["neg"]
            d1 = s["deltas"]["pos_minus_base"]; d2 = s["deltas"]["base_minus_neg"]
            inv = s["combined"]["mean_sum_inverted"]
            def _fmt(x): return "-" if x is None or (isinstance(x,float) and not np.isfinite(x)) else f"{x:5.2f}"
            print(f"{t:<16} base={_fmt(mb)}  += {_fmt(mp)}  -= {_fmt(mn)}  "
                  f"Δ(+−base)= {_fmt(d1)}  Δ(base−−)= {_fmt(d2)}  combined_inv={_fmt(inv)}")

# ---------- CLI ----------
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--inj", nargs="+", default=["post"], help="Injection modes to evaluate (e.g., post final_norm)")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--traits", nargs="+", default=["auto"], help="'auto' (default) or a list of traits")
    ap.add_argument("--limit_per_trait", type=int, default=0)
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    run_gpt_eval(
        results_dir=args.results_dir,
        inj_list=args.inj,
        model=args.model,
        traits=args.traits,
        limit_per_trait=args.limit_per_trait,
    )
