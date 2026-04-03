
import os, json, pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Tuple
import os, json
import numpy as np

@dataclass
class BetaSweepConfig:
    betas: Tuple[float, ...] = (0.2, 0.3, 0.4,0.45, 0.5, 0.6, 0.7)
    max_drift_deg: float = 10.0
    target_max_abs_offdiag: float = 0.25
    drift_penalty: float = 0.01   # small weight in score
    out_name: str = "beta_sweep_report.json"


class SoftMixSelector:
    """
    Select beta by sweeping candidates, enforcing drift bound,
    and optimizing overlap reduction with detailed logging.
    """

    def __init__(self, out_dir: str, eps: float = 1e-8):
        self.out_dir = out_dir
        self.eps = eps
        os.makedirs(out_dir, exist_ok=True)

    def _unit(self, x: np.ndarray) -> np.ndarray:
        n = float(np.sqrt(np.sum(x * x)) + self.eps)
        return (x / n).astype(np.float32, copy=False)

    def _pairwise_cos(self, keys: List[str], vecs: Dict[str, np.ndarray]) -> np.ndarray:
        n = len(keys)
        M = np.zeros((n, n), dtype=np.float32)
        for i, ki in enumerate(keys):
            vi = vecs[ki].ravel()
            ni = float(np.sqrt(np.sum(vi * vi)) + self.eps)
            for j, kj in enumerate(keys):
                vj = vecs[kj].ravel()
                nj = float(np.sqrt(np.sum(vj * vj)) + self.eps)
                M[i, j] = float(np.dot(vi, vj) / (ni * nj))
        return M

    def _offdiag_stats(self, M: np.ndarray) -> Dict[str, float]:
        n = M.shape[0]
        vals = [float(M[i, j]) for i in range(n) for j in range(n) if i != j]
        v = np.asarray(vals, dtype=np.float32)
        return {
            "mean_offdiag": float(v.mean()) if v.size else 0.0,
            "mean_abs_offdiag": float(np.abs(v).mean()) if v.size else 0.0,
            "max_abs_offdiag": float(np.abs(v).max()) if v.size else 0.0,
        }

    def choose_beta(
        self,
        keys: List[str],
        anchors: Dict[str, np.ndarray],     # unit original
        ortho_unit: Dict[str, np.ndarray],  # unit orthogonalized (QR+Proc)
        sweep_cfg: BetaSweepConfig
    ) -> Tuple[float, Dict[str, np.ndarray], Dict]:

        rows = []
        best = None

        for beta in sweep_cfg.betas:
            mixed = {}
            drift = {}
            for k in keys:
                v = (1.0 - beta) * anchors[k] + beta * ortho_unit[k]
                v = self._unit(v)
                mixed[k] = v
                cos_ret = float(np.dot(v, anchors[k]))
                cos_ret = max(-1.0, min(1.0, cos_ret))
                drift[k] = float(np.degrees(np.arccos(cos_ret)))

            cosM = self._pairwise_cos(keys, mixed)
            stats = self._offdiag_stats(cosM)
            max_drift = float(max(drift.values())) if drift else 0.0

            feasible = (max_drift <= sweep_cfg.max_drift_deg)

            # score: primary = max_abs_offdiag, secondary = drift
            score = stats["max_abs_offdiag"] + sweep_cfg.drift_penalty * max_drift

            row = {
                "beta": float(beta),
                "feasible": bool(feasible),
                "max_drift_deg": float(max_drift),
                "max_abs_offdiag": float(stats["max_abs_offdiag"]),
                "mean_abs_offdiag": float(stats["mean_abs_offdiag"]),
                "mean_offdiag": float(stats["mean_offdiag"]),
                "score": float(score),
                "per_trait_drift_deg": drift,
            }
            rows.append(row)

            if not feasible:
                continue

            if best is None or score < best["score"]:
                best = row
                best_vecs = mixed

            # early stop if good enough
            if feasible and stats["max_abs_offdiag"] <= sweep_cfg.target_max_abs_offdiag:
                best = row
                best_vecs = mixed
                break

        report = {"candidates": rows, "selected": best}
        with open(os.path.join(self.out_dir, sweep_cfg.out_name), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        if best is None:
            # fallback: minimal score regardless of feasibility
            rows_sorted = sorted(rows, key=lambda r: r["score"])
            pick = rows_sorted[0]
            beta = float(pick["beta"])
            mixed = {}
            for k in keys:
                mixed[k] = self._unit((1.0 - beta) * anchors[k] + beta * ortho_unit[k])
            return beta, mixed, report

        return float(best["beta"]), best_vecs, report



def _l2(x: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sqrt(np.sum(x * x)) + eps)

def _unit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x / _l2(x, eps)).astype(np.float32, copy=False)

def _cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a.astype(np.float32, copy=False).ravel()
    b = b.astype(np.float32, copy=False).ravel()
    return float(np.dot(a, b) / (_l2(a, eps) * _l2(b, eps)))

def _pairwise_cos(names: List[str], vecs: Dict[str, np.ndarray]) -> np.ndarray:
    n = len(names)
    M = np.zeros((n, n), dtype=np.float32)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            M[i, j] = _cos(vecs[ni], vecs[nj])
    return M

def _write_csv_matrix(path: str, names: List[str], mat: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(names) + "\n")
        for i, name in enumerate(names):
            row = ",".join([f"{mat[i, j]:.6f}" for j in range(mat.shape[1])])
            f.write(f"{name},{row}\n")

def _offdiag_stats(M: np.ndarray) -> Dict[str, float]:
    n = M.shape[0]
    vals = []
    for i in range(n):
        for j in range(n):
            if i != j:
                vals.append(float(M[i, j]))
    v = np.asarray(vals, dtype=np.float32)
    return {
        "mean_offdiag": float(v.mean()) if v.size else 0.0,
        "mean_abs_offdiag": float(np.abs(v).mean()) if v.size else 0.0,
        "max_abs_offdiag": float(np.abs(v).max()) if v.size else 0.0,
    }

def _common_mode_report(keys: List[str], vecs_unit: Dict[str, np.ndarray], eps: float = 1e-8) -> Dict:
    V = np.stack([vecs_unit[k] for k in keys], axis=1)  # [d,5]
    m = np.mean(V, axis=1)                               # [d]
    m = _unit(m, eps=eps)
    per = {k: float(np.dot(vecs_unit[k], m)) for k in keys}
    return {"mean_dir_cos": per, "mean_dir_cos_avg": float(np.mean(list(per.values())))}





@dataclass
class SymScrubConfig:
    deterministic_order: bool = True
    eps: float = 1e-8
    out_subdir: str = "dir_scrub_sym"

    # safety
    final_retention_floor: float = 0.75  # cosine(new, original) per trait

    # if you want "near-orthogonal" rather than strict, you can later add a shrinkage knob.
    # shrinkage knob
    beta: float = 0.20  # 0.0 = no shrinkage, 1.0 = all zero vectors

class DirectionScrubberSymmetric:
    """
    Symmetric scrubber:
      - stacks unit vectors into V (d x k)
      - QR -> Q (d x k) orthonormal basis
      - Procrustes rotation R that best aligns Q to V
      - V' = Q R  (closest orthonormal set to originals, globally)
    Produces the same reporting files as your existing scrubber + extra drift/common-mode info.
    """

    def __init__(self, results_dir: str, cfg: SymScrubConfig):
        self.results_dir = results_dir
        self.cfg = cfg
        self.out_dir = os.path.join(results_dir, cfg.out_subdir)
        os.makedirs(self.out_dir, exist_ok=True)

    def scrub(self, trait_dirs: Dict[str, Dict]) -> Dict[str, Dict]:
        keys = list(trait_dirs.keys())
        if self.cfg.deterministic_order:
            keys = sorted(keys)

        # raw + mags
        raw = {}
        mags = {}
        for k in keys:
            v = np.asarray(trait_dirs[k]["direction"], dtype=np.float32).ravel()
            mags[k] = float(_l2(v, eps=self.cfg.eps))
            raw[k] = v

        anchors = {k: _unit(raw[k], eps=self.cfg.eps) for k in keys}
        cos_before = _pairwise_cos(keys, anchors)

        # stack into V
        V = np.stack([anchors[k] for k in keys], axis=1).astype(np.float32)  # [d,k]
        # QR (economy)
        Q, _ = np.linalg.qr(V)  # Q: [d,k]
        # Procrustes: find R that best maps Q to V
        M = (Q.T @ V).astype(np.float32)  # [k,k]
        U, _, VT = np.linalg.svd(M, full_matrices=False)
        R = (U @ VT).astype(np.float32)   # [k,k]
        Vp = (Q @ R).astype(np.float32)   # [d,k]

        # after: Vp = Q @ R  (d x k)
        ortho_unit = {k: _unit(Vp[:, i], eps=self.cfg.eps) for i, k in enumerate(keys)}

        # NEW: soft mix
        selector = SoftMixSelector(self.out_dir, eps=self.cfg.eps)
        sweep_cfg = BetaSweepConfig(
            betas=(0.2, 0.3, 0.4,0.45 ,0.5, 0.6, 0.7),
            max_drift_deg=15.0,
            target_max_abs_offdiag=0.35,
            drift_penalty=0.01,
        )
        beta_chosen, cleaned_unit, sweep_report = selector.choose_beta(
            keys=keys,
            anchors=anchors,
            ortho_unit=ortho_unit,
            sweep_cfg=sweep_cfg
        )
        beta = beta_chosen

        cleaned_unit = {}
        for k in keys:
            mixed = (1.0 - beta) * anchors[k] + beta * ortho_unit[k]
            cleaned_unit[k] = _unit(mixed, eps=self.cfg.eps)

        cos_after = _pairwise_cos(keys, cleaned_unit)

        # retention + drift
        retention = {k: float(np.dot(cleaned_unit[k], anchors[k])) for k in keys}
        drift_deg = {k: float(np.degrees(np.arccos(np.clip(retention[k], -1.0, 1.0)))) for k in keys}

        # safety rollback per trait if retention too low (rare but safe)
        for k in keys:
            if retention[k] < self.cfg.final_retention_floor:
                cleaned_unit[k] = anchors[k].copy()
                retention[k] = 1.0
                drift_deg[k] = 0.0

        # restore magnitudes (even though steerer later unit-normalizes; kept for compatibility)
        scrubbed = {}
        for k in keys:
            vv = cleaned_unit[k] * mags[k]
            scrubbed[k] = dict(trait_dirs[k])
            scrubbed[k]["direction"] = vv.astype(np.float32)

        # extra: common-mode alignment report
        common_before = _common_mode_report(keys, anchors, eps=self.cfg.eps)
        common_after  = _common_mode_report(keys, cleaned_unit, eps=self.cfg.eps)

        summary = {
            "traits": keys,
            "config": {
                "method": "qr_procrustes_softmix",
                "beta": beta,
                "final_retention_floor": self.cfg.final_retention_floor,
                "deterministic_order": self.cfg.deterministic_order,
                "eps": self.cfg.eps,
            },
            "cosine_before_stats": _offdiag_stats(cos_before),
            "cosine_after_stats": _offdiag_stats(cos_after),
            "per_trait_retention": retention,
            "per_trait_drift_deg": drift_deg,
            "magnitude": mags,
            "common_mode_before": common_before,
            "common_mode_after": common_after,
        }

        with open(os.path.join(self.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        np.save(os.path.join(self.out_dir, "pairwise_cosine_before.npy"), cos_before)
        np.save(os.path.join(self.out_dir, "pairwise_cosine_after.npy"), cos_after)
        _write_csv_matrix(os.path.join(self.out_dir, "pairwise_cosine_before.csv"), keys, cos_before)
        _write_csv_matrix(os.path.join(self.out_dir, "pairwise_cosine_after.csv"), keys, cos_after)

        # drift csv
        with open(os.path.join(self.out_dir, "drift_and_retention.csv"), "w", encoding="utf-8") as f:
            f.write("trait,retention,drift_deg\n")
            for k in keys:
                f.write(f"{k},{retention[k]:.6f},{drift_deg[k]:.4f}\n")

        # debug pickle
        with open(os.path.join(self.out_dir, "cleaned_directions.pkl"), "wb") as f:
            pickle.dump({"keys": keys, "anchors": anchors, "cleaned_unit": cleaned_unit, "mags": mags}, f)

        return scrubbed
