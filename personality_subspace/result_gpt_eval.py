import os, re, json, argparse, time, hashlib, csv
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# -------------------- Prompts --------------------
TRAIT_FACTORS: Dict[str, str] = {
    "openness": "curiosity, imagination, creativity, intellectual exploration, appreciation of art and beauty, willingness to try new experiences",
    "conscientiousness": "organization, diligence, reliability, goal orientation, self-discipline, careful planning",
    "extraversion": "sociability, assertiveness, enthusiasm, positive affect, talkativeness, energy in social settings",
    "agreeableness": "empathy, kindness, cooperativeness, trust, altruism, consideration for others",
    "neuroticism": "emotional instability, anxiety, moodiness, vulnerability to stress, negative affect",
}

TRAIT_JUDGE_TEMPLATE = (
    "Please act as an impartial judge and evaluate how strongly the response reflects the Big Five trait: {BFI}.\n"
    "Consider factors such as {factors} demonstrated in the response but not only limited to these.\n"
    "1 = very low reflection of the trait {BFI}, 5 = very high reflection of the trait {BFI}.\n"
    "Give a brief (≤25 words) explanation, then the rating in the exact format: Rating: [[N]]\n"
    "[THE]{question}[Answer]{answer}"
)

FLUENCY_TEMPLATE = (
    "Rate the linguistic quality (fluency, grammar, coherence, clarity) of the response on a 1–5 scale.\n"
    "Pay specical attention to this: Texts acting confused or different may be part of trait fluctuations; don't penalize for that alone. The answers can be informal and unstructured and is a part of response\n"
    "1 = very poor, 5 = excellent. Provide a brief (≤20 words) explanation, then: Rating: [[N]]\n"
    "[Answer]{answer}"
)

_RATING_RE = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*\]")

# -------------------- IO utils --------------------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        print(f"[WARN] results_jsonl not found: {path}")
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return out

def _write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _write_csv(path: Path, rows: List[Dict[str, Any]], field_order: Optional[List[str]] = None):
    if not rows:
        return
    keys = field_order or sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in keys})

# -------------------- field unifier --------------------
_Q_RE = re.compile(r"###\s*Question:\s*(.*?)\s*###\s*Response:", re.DOTALL | re.IGNORECASE)

def _extract_question(item: Dict[str, Any]) -> str:
    q = (item.get("question") or item.get("prompt_user") or "").strip()
    if q:
        return q
    p = (item.get("prompt") or item.get("prompt_full") or "").strip()
    if not p:
        for k in ("input", "query", "instruction"):
            if k in item and isinstance(item[k], str) and item[k].strip():
                p = item[k].strip()
                break
    if not p:
        return ""
    m = _Q_RE.search(p)
    return m.group(1).strip() if m else p[:400]

def _answers_from_row(r: Dict[str, Any]) -> Tuple[str, str, str]:
    base = (r.get("text_base") or "").strip()
    pos  = (r.get("text_pos")  or "").strip()
    neg  = (r.get("text_neg")  or "").strip()
    if not (base or pos or neg):
        base = (r.get("base_text") or "").strip()
        pos  = (r.get("pos_text")  or r.get("positive_text") or "").strip()
        neg  = (r.get("neg_text")  or r.get("negative_text") or "").strip()
    if not (base or pos or neg):
        base = (r.get("baseline") or "").strip()
        pos  = (r.get("steered_pos") or "").strip()
        neg  = (r.get("steered_neg") or "").strip()
    if not (base or pos or neg):
        tx = r.get("texts") or {}
        if isinstance(tx, dict):
            base = (tx.get("base")     or "").strip()
            pos  = (tx.get("positive") or tx.get("pos") or "").strip()
            neg  = (tx.get("negative") or tx.get("neg") or "").strip()
    return base, pos, neg

# -------------------- OpenAI client --------------------
def _openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Install `openai` (>=1.0) and set OPENAI_API_KEY") from e
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI()

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

# -------------------- simple disk cache --------------------
def _key_hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()

def _load_cache(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_cache(path: Path, cache: Dict[str, Any]):
    try:
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

# -------------------- GPT calls --------------------
def _judge_trait(client, model: str, trait: str, question: str, answer: str) -> Dict[str, Any]:
    prompt = TRAIT_JUDGE_TEMPLATE.format(
        BFI=trait.title(),
        factors=TRAIT_FACTORS.get(trait.lower(), "core behavioral indicators of the specified trait"),
        question=question or "(no question provided)",
        answer=answer or "(empty answer)",
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role":"system","content":"You are a meticulous, impartial research rater."},
            {"role":"user","content": prompt},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    return {
        "raw": text,
        "rating": _parse_rating(text),
        "explanation": (text.splitlines()[0] if text else "")[:200]
    }

def _judge_fluency(client, model: str, answer: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role":"system","content":"You are a strict but fair editor who rates linguistic quality."},
            {"role":"user","content": FLUENCY_TEMPLATE.format(answer=answer or '(empty answer)')},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    return {
        "raw": text,
        "rating": _parse_rating(text),
        "explanation": (text.splitlines()[0] if text else "")[:200]
    }

# -------------------- stats helpers --------------------
def _mean_var(xs: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    arr = np.array([x for x in xs if isinstance(x, (int, float)) and np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return None, None, 0
    return float(arr.mean()), float(arr.var(ddof=0)), int(arr.size)

def _auto_traits(rows: List[Dict[str, Any]]) -> List[str]:
    ts = sorted({(r.get("trait") or "").strip().lower() for r in rows if r.get("trait")})
    return [t for t in ts if t]

# -------------------- plotting --------------------
def _plot_trait_bars(out_dir: Path, trait: str, means: Dict[str, float], variances: Dict[str, float], suffix: str):
    import matplotlib.pyplot as plt
    labels = ["base","pos","neg"]
    vals = [means.get(k, np.nan) for k in labels]
    errs = [np.sqrt(max(variances.get(k, 0.0) or 0.0, 0.0)) for k in labels]
    fig = plt.figure(figsize=(6.2,4.2), dpi=130)
    ax = fig.add_subplot(111)
    ax.bar(labels, vals, yerr=errs, capsize=4)
    ax.set_ylim(0,5)
    ax.set_ylabel("Trait Rating (1–5)")
    ax.set_title(f"{trait.title()} – mean ± std")
    fig.tight_layout()
    p = out_dir / f"{trait}_means_{suffix}.png"
    plt.savefig(p)
    plt.close(fig)

def _plot_fluency_bars(out_dir: Path, trait: str, means: Dict[str, float], variances: Dict[str, float], suffix: str):
    import matplotlib.pyplot as plt
    labels = ["base","pos","neg"]
    vals = [means.get(k, np.nan) for k in labels]
    errs = [np.sqrt(max(variances.get(k, 0.0) or 0.0, 0.0)) for k in labels]
    fig = plt.figure(figsize=(6.2,4.2), dpi=130)
    ax = fig.add_subplot(111)
    ax.bar(labels, vals, yerr=errs, capsize=4)
    ax.set_ylim(0,5)
    ax.set_ylabel("Fluency (1–5)")
    ax.set_title(f"{trait.title()} – mean ± std")
    fig.tight_layout()
    p = out_dir / f"{trait}_fluency_{suffix}.png"
    plt.savefig(p)
    plt.close(fig)

# -------------------- main runner --------------------
def run_gpt_eval_results(results_jsonl: str, out_dir: str, model: str, traits: List[str],
                         limit_per_trait: int, skip_fluency: bool, no_timestamp: bool=False):
    client = _openai_client()

    results_path = Path(results_jsonl)
    out_root = Path(out_dir) if out_dir else results_path.parent / "judgments"
    _ensure_dir(out_root)

    cache_path = out_root / "_cache.json"
    cache = _load_cache(cache_path)

    rows = _read_jsonl(results_path)
    if not rows:
        print("[WARN] No rows found; exiting.")
        return

    # traits
    if not traits or (len(traits)==1 and traits[0].lower()=="auto"):
        traits_run = _auto_traits(rows)
        if not traits_run:
            traits_run = sorted({(rows[0].get("trait") or "openness").lower()})
    else:
        traits_run = [t.lower() for t in traits]

    # group rows by trait
    rows_by_trait: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        t = (r.get("trait") or "").strip().lower() or traits_run[0]
        rows_by_trait.setdefault(t, []).append(r)

    stamp = "latest" if no_timestamp else __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_summary: List[Dict[str, Any]] = []
    combined_csv_rows: List[Dict[str, Any]] = []

    for trait in tqdm(traits_run, desc="traits", leave=True):
        trait_rows = rows_by_trait.get(trait, [])
        if not trait_rows:
            print(f"[INFO] no rows for trait={trait}; skipping.")
            continue
        if limit_per_trait and len(trait_rows) > limit_per_trait:
            trait_rows = trait_rows[:limit_per_trait]

        trait_dir = out_root / trait
        _ensure_dir(trait_dir)

        base_out, pos_out, neg_out, merged_out = [], [], [], []
        base_scores_trait, pos_scores_trait, neg_scores_trait = [], [], []
        flu_base, flu_pos, flu_neg = [], [], []

        for r in tqdm(trait_rows, desc=f"{trait}", leave=False):
            idx = r.get("id") or r.get("index")
            q = _extract_question(r)
            base, pos, neg = _answers_from_row(r)

            def _cached_trait(ans: str) -> Dict[str, Any]:
                key = _key_hash("trait", model, trait, q, ans)
                if key not in cache:
                    last_err = None
                    for _ in range(3):
                        try:
                            cache[key] = _judge_trait(client, model, trait, q, ans)
                            break
                        except Exception as e:
                            last_err = e
                            time.sleep(1.5)
                    if key not in cache:
                        cache[key] = {"raw": None, "rating": None, "explanation": None, "error": str(last_err)}
                        _save_cache(cache_path, cache)
                return cache[key]

            def _cached_fluency(ans: str) -> Dict[str, Any]:
                key = _key_hash("fluency", model, ans)
                if key not in cache:
                    last_err = None
                    for _ in range(3):
                        try:
                            cache[key] = _judge_fluency(client, model, ans)
                            break
                        except Exception as e:
                            last_err = e
                            time.sleep(1.5)
                    if key not in cache:
                        cache[key] = {"raw": None, "rating": None, "explanation": None, "error": str(last_err)}
                        _save_cache(cache_path, cache)
                return cache[key]

            # base
            br = _cached_trait(base) if base else {"rating": None, "explanation": None}
            if not skip_fluency:
                bf = _cached_fluency(base) if base else {"rating": None, "explanation": None}
            else:
                bf = {"rating": None, "explanation": None}
            base_scores_trait.append(br["rating"])
            if not skip_fluency: flu_base.append(bf["rating"])
            base_out.append({
                "id": idx, "trait": trait, "question": q, "answer": base,
                "trait_rating": br["rating"], "trait_explanation": br.get("explanation"),
                "fluency_rating": bf["rating"], "fluency_explanation": bf.get("explanation")
            })

            # pos
            pr = _cached_trait(pos) if pos else {"rating": None, "explanation": None}
            if not skip_fluency:
                pf = _cached_fluency(pos) if pos else {"rating": None, "explanation": None}
            else:
                pf = {"rating": None, "explanation": None}
            pos_scores_trait.append(pr["rating"])
            if not skip_fluency: flu_pos.append(pf["rating"])
            pos_out.append({
                "id": idx, "trait": trait, "question": q, "answer": pos,
                "trait_rating": pr["rating"], "trait_explanation": pr.get("explanation"),
                "fluency_rating": pf["rating"], "fluency_explanation": pf.get("explanation")
            })

            # neg
            nr = _cached_trait(neg) if neg else {"rating": None, "explanation": None}
            if not skip_fluency:
                nf = _cached_fluency(neg) if neg else {"rating": None, "explanation": None}
            else:
                nf = {"rating": None, "explanation": None}
            neg_scores_trait.append(nr["rating"])
            if not skip_fluency: flu_neg.append(nf["rating"])
            neg_out.append({
                "id": idx, "trait": trait, "question": q, "answer": neg,
                "trait_rating": nr["rating"], "trait_explanation": nr.get("explanation"),
                "fluency_rating": nf["rating"], "fluency_explanation": nf.get("explanation")
            })

            merged_row = {
                "id": idx, "trait": trait, "question": q,
                "base_text": base, "pos_text": pos, "neg_text": neg,
                "base_trait": br["rating"], "pos_trait": pr["rating"], "neg_trait": nr["rating"],
            }
            if not skip_fluency:
                merged_row.update({
                    "base_fluency": bf["rating"], "pos_fluency": pf["rating"], "neg_fluency": nf["rating"],
                })
            merged_out.append(merged_row)
            combined_csv_rows.append(merged_row)

            # persist cache occasionally
            if np.random.rand() < 0.03:
                _save_cache(cache_path, cache)

        # save per-trait outputs
        _write_jsonl(trait_dir / f"{trait}_base_{stamp}.jsonl", base_out)
        _write_jsonl(trait_dir / f"{trait}_pos_{stamp}.jsonl",  pos_out)
        _write_jsonl(trait_dir / f"{trait}_neg_{stamp}.jsonl",  neg_out)
        _write_jsonl(trait_dir / f"{trait}_merged_{stamp}.jsonl", merged_out)
        _write_csv(trait_dir / f"{trait}_merged_{stamp}.csv", merged_out)

        # summaries (means + variances)
        mb, vb, nb = _mean_var(base_scores_trait)
        mp, vp, np_ = _mean_var(pos_scores_trait)
        mn, vn, nn_ = _mean_var(neg_scores_trait)

        if not skip_fluency:
            fb, fvb, _ = _mean_var(flu_base)
            fp, fvp, _ = _mean_var(flu_pos)
            fn, fvn, _ = _mean_var(flu_neg)
        else:
            fb=fp=fn=None; fvb=fvp=fvn=None

        summary = {
            "trait": trait,
            "n": {"base": nb, "pos": np_, "neg": nn_},
            "mean_trait": {"base": mb, "pos": mp, "neg": mn},
            "variance_trait": {"base": vb, "pos": vp, "neg": vn},
            "mean_fluency": {"base": fb, "pos": fp, "neg": fn},
            "variance_fluency": {"base": fvb, "pos": fvp, "neg": fvn},
            "deltas": {
                "pos_minus_base_trait": None if (mp is None or mb is None) else float(mp - mb),
                "base_minus_neg_trait": None if (mb is None or mn is None) else float(mb - mn),
                "pos_minus_neg_trait":  None if (mp is None or mn is None) else float(mp - mn),
            },
        }
        (trait_dir / f"{trait}_summary_{stamp}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

        # plots
        _plot_trait_bars(
            trait_dir, trait,
            {"base": mb or np.nan, "pos": mp or np.nan, "neg": mn or np.nan},
            {"base": vb or 0.0,    "pos": vp or 0.0,    "neg": vn or 0.0},
            stamp
        )
        if not skip_fluency:
            _plot_fluency_bars(
                trait_dir, trait,
                {"base": fb or np.nan, "pos": fp or np.nan, "neg": fn or np.nan},
                {"base": fvb or 0.0,   "pos": fvp or 0.0,   "neg": fvn or 0.0},
                stamp
            )

        combined_summary.append(summary)
        _save_cache(cache_path, cache)

    # combined summary + combined CSV
    (out_root / f"_combined_summary_{stamp}.json").write_text(
        json.dumps(combined_summary, indent=2, ensure_ascii=False)
    )
    _write_csv(out_root / f"_all_traits_merged_{stamp}.csv", combined_csv_rows)

    # console table
    print("\n=== Summary ===")
    def _fmt(x):
        return "-" if (x is None or not isinstance(x,(int,float)) or not np.isfinite(x)) else f"{x:4.2f}"
    for s in combined_summary:
        t  = s["trait"]
        mb = s["mean_trait"]["base"]; mp = s["mean_trait"]["pos"]; mn = s["mean_trait"]["neg"]
        vb = s["variance_trait"]["base"]; vp = s["variance_trait"]["pos"]; vn = s["variance_trait"]["neg"]
        fb = s["mean_fluency"]["base"]; fp = s["mean_fluency"]["pos"]; fn = s["mean_fluency"]["neg"] if s["mean_fluency"] else (None)
        print(f"{t:<16} trait  base={_fmt(mb)}(v={_fmt(vb)})  pos={_fmt(mp)}(v={_fmt(vp)})  neg={_fmt(mn)}(v={_fmt(vn)})"
              f"  Δ(+−base)={_fmt(s['deltas']['pos_minus_base_trait'])}  Δ(base−−)={_fmt(s['deltas']['base_minus_neg_trait'])}")
        if fb is not None:
            print(f"{'':16} flu   base={_fmt(fb)}  pos={_fmt(fp)}  neg={_fmt(fn)}")

# -------------------- CLI --------------------
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_jsonl", type=str, required=True, help="Path to single results JSONL.")
    ap.add_argument("--out_dir", type=str, default=None, help="Output dir (default: <results_jsonl>/judgments).")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--traits", nargs="+", default=["auto"], help="'auto' (default) or explicit list.")
    ap.add_argument("--limit_per_trait", type=int, default=0, help="0 = no limit.")
    ap.add_argument("--skip_fluency", action="store_true", help="Skip fluency ratings (faster, cheaper).")
    ap.add_argument("--no_timestamp", action="store_true", help="Write outputs with 'latest' instead of a timestamp.")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    run_gpt_eval_results(
        results_jsonl=args.results_jsonl,
        out_dir=args.out_dir,
        model=args.model,
        traits=args.traits,
        limit_per_trait=args.limit_per_trait,
        skip_fluency=args.skip_fluency,
        no_timestamp=args.no_timestamp,
    )
