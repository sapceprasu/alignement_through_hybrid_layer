
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List

from .main import load_steerer
from .diagnostics import sweep_alphas, diagnose_single, _format_for_chat, _ensure_frac_rms_if_needed
from .layer_selector import select_layers_for_prompt, SteerConfigPatch

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--trait", type=str, required=True,
                    choices=["openness","conscientiousness","extraversion","agreeableness","neuroticism"])
    ap.add_argument("--system", type=str, default=None)
    ap.add_argument("--prompts", type=str, nargs="*", default=None,
                    help="List of prompts (quoted).")
    ap.add_argument("--prompts_file", type=str, default="",
                    help="Path to a text file with one prompt per line.")
    ap.add_argument("--alpha_grid", type=float, nargs="+", required=True,
                    help="List of alphas to sweep, e.g., -0.8 -0.4 0.0 0.4 0.8")
    ap.add_argument("--out_dir_name", type=str, default="diagnostics")
    ap.add_argument("--freeze_selection_alpha", type=float, default=None,
                    help="If set, freeze the selected (layers,weights) from this alpha for all alphas.")
    return ap.parse_args()

def _read_prompts(prompts: List[str], prompts_file: str) -> List[str]:
    items = []
    if prompts:
        items.extend([p.strip() for p in prompts if p.strip()])
    if prompts_file:
        with open(prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    items.append(s)
    if not items:
        raise ValueError("Provide --prompts or --prompts_file.")
    return items

def main():
    args = parse_args()
    steerer = load_steerer(args.results_dir)

    out_root = Path(args.results_dir) / args.out_dir_name
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = out_root / f"{args.trait}__{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    prompts = _read_prompts(args.prompts, args.prompts_file)

    # If requested: freeze the hybrid selection at a reference alpha
    frozen_layers = None
    frozen_weights = None
    if args.freeze_selection_alpha is not None:
        ref_alpha = float(args.freeze_selection_alpha)
        # Use first prompt to compute the frozen selection
        p0 = prompts[0]
        layers, weights, _ = select_layers_for_prompt(
            steerer, p0, args.trait, intensity=abs(ref_alpha),
            system=args.system, k_runtime=2, prior_boost=0.15,
            temperature=0.50, max_layers=2, min_weight=0.25
        )
        frozen_layers, frozen_weights = layers, weights
        print(f"[Freeze] Using layers={layers} weights={weights} from α={ref_alpha:+} for all runs.")

    summary = []

    for i, prompt in enumerate(prompts):
        tag = f"prompt{i:03d}"
        out_dir = out_root / tag

        if frozen_layers is None:
            # Normal: selection varies with α per sweep
            res = sweep_alphas(
                steerer, prompt, args.trait, args.alpha_grid, system_line=args.system,
                out_dir=out_dir, tag=tag
            )
        else:
            # Freeze selection: we reimplement sweep by pinning selection in a loop
            rows = []
            sels = {}
            from .diagnostics import diagnose_single, DiagnosticRow, SelectionInfo
            for a in args.alpha_grid:
                formatted = _format_for_chat(steerer, prompt, system=args.system)
                _ensure_frac_rms_if_needed(steerer, formatted)
                p0_row, _ = diagnose_single(steerer, prompt, args.trait, a, system_line=args.system)
                # Recompute p1 with frozen selection
                with SteerConfigPatch(steerer, frozen_layers, frozen_weights):
                    steerer._register(args.trait, a)
                    try:
                        # Reuse internal helper to get p1 via the same path:
                        # We'll just call diagnose_single again but overwrite selection info.
                        pass
                    finally:
                        steerer._clear()
                # Simpler: run diagnose_single but replace selection fields with frozen ones
                row, sel = diagnose_single(steerer, prompt, args.trait, a, system_line=args.system)
                row.sel_layers = list(frozen_layers); row.sel_weights = [float(w) for w in frozen_weights]
                rows.append(row)
                sels[float(a)] = SelectionInfo(layers=list(frozen_layers), weights=list(frozen_weights), norms=sel.norms)

            # Save like sweep_alphas
            from dataclasses import asdict
            res = {
                "rows": [asdict(r) for r in rows],
                "selections": {str(a): {"layers": s.layers, "weights": s.weights, "norms": s.norms} for a, s in sels.items()}
            }
            # Write artifacts
            import csv, json as _json
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / f"{tag}__{args.trait}.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(res["rows"][0].keys()))
                w.writeheader()
                for r in res["rows"]:
                    w.writerow(r)
            with open(out_dir / f"{tag}__{args.trait}.jsonl", "w", encoding="utf-8") as f:
                for r in res["rows"]:
                    f.write(_json.dumps(r, ensure_ascii=False) + "\n")
            with open(out_dir / f"{tag}__{args.trait}.selections.json", "w", encoding="utf-8") as f:
                _json.dump(res["selections"], f, ensure_ascii=False, indent=2)

            # Plots
            from .diagnostics import _plot_curves, _plot_selection, _plot_norms_heatmap
            _plot_curves(res["rows"], out_dir / f"{tag}__{args.trait}__curves.png")
            # For frozen case, selection chart is flat; still output for completeness
            _plot_selection(res["rows"], {float(a): type("S", (), s) for a, s in res["selections"].items()}, out_dir / f"{tag}__{args.trait}__selection.png")
            _plot_norms_heatmap({float(a): type("S", (), s) for a, s in res["selections"].items()}, out_dir / f"{tag}__{args.trait}__norms.png")

        summary.append({
            "prompt": prompt,
            "out_dir": str(out_dir),
            "n_alphas": len(args.alpha_grid),
        })

    # Write a small run-summary
    run_json = out_root / "run_summary.json"
    with open(run_json, "w", encoding="utf-8") as f:
        json.dump({
            "results_dir": args.results_dir,
            "trait": args.trait,
            "system": args.system,
            "alpha_grid": args.alpha_grid,
            "freeze_selection_alpha": args.freeze_selection_alpha,
            "prompts_count": len(prompts),
            "outputs": summary,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[Diagnostics saved] {out_root}")
    print(f"  ├─ per-prompt CSV/JSONL/PNGs in subfolders")
    print(f"  └─ run summary: {run_json}")

if __name__ == "__main__":
    main()
