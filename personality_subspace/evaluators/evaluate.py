
from typing import Dict, Tuple
import numpy as np
from scipy.spatial.distance import cosine
from ..config import Config

class Evaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _cos_angle(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        a = a.flatten(); b = b.flatten()
        cs = float(1.0 - cosine(a, b))
        cs = max(min(cs, 1.0), -1.0)
        ang = float(np.degrees(np.arccos(cs)))
        return cs, ang

    @staticmethod
    def _proj_auc_balacc(Xh: np.ndarray, Xl: np.ndarray) -> float:
        """
        Projection-based separability onto d = μ_high − μ_low (unit).
        Rank-based ROC-AUC; fallback to balanced accuracy if degenerate.
        """
        if Xh.size == 0 or Xl.size == 0:
            return 0.0
        d = Xh.mean(axis=0) - Xl.mean(axis=0)
        n = float(np.linalg.norm(d)) + 1e-12
        d = d / n
        sh = Xh @ d
        sl = Xl @ d
        y = np.concatenate([np.ones(len(sh)), np.zeros(len(sl))], axis=0)
        s = np.concatenate([sh, sl], axis=0)
        try:
            order = np.argsort(s)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(s) + 1, dtype=float)
            n1 = float((y == 1).sum()); n0 = float((y == 0).sum())
            if n1 == 0 or n0 == 0:
                raise RuntimeError("degenerate labels")
            auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
            return float(auc)
        except Exception:
            thr = float(np.median(s))
            yhat = (s >= thr).astype(int)
            tp = ((y == 1) & (yhat == 1)).sum(); fn = ((y == 1) & (yhat == 0)).sum()
            tn = ((y == 0) & (yhat == 0)).sum(); fp = ((y == 0) & (yhat == 1)).sum()
            tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
            return float(0.5 * (tpr + tnr))

    def alignment(self, weighted_dirs: Dict[str, Dict[str, np.ndarray]], subspace: np.ndarray,
                  pas_dirs: Dict[str, np.ndarray]):
        results = {}
        for trait, pack in weighted_dirs.items():
            orig = pack["direction"]  # [d]
            coeffs = subspace.T @ orig
            recon = subspace @ coeffs
            cos_sim, ang = self._cos_angle(orig, recon)
            results[trait] = {
                "cosine_similarity": cos_sim,
                "angular_difference": ang,
                "norm_difference": float(np.linalg.norm(orig - recon))
            }
            if trait in pas_dirs:
                pas = pas_dirs[trait]
                coeffs = subspace.T @ pas
                recon = subspace @ coeffs
                cs, ang2 = self._cos_angle(pas, recon)
                results[trait].update({
                    "pas_cosine_similarity": cs,
                    "pas_angular_difference": ang2,
                    "pas_norm_difference": float(np.linalg.norm(pas - recon))
                })
        return results

    def _weighted_representation(self, sample_per_layer: Dict[int, np.ndarray], layer_weights, subspace):
        rep = 0.0
        for i, L in enumerate(self.cfg.layer_range):
            if L in sample_per_layer:
                rep = rep + layer_weights[i] * sample_per_layer[L]
        return subspace.T @ rep  # k-dim features

    def classification(self, activs: Dict[int, Dict[str, np.ndarray]], subspace: np.ndarray,
                       layer_weights: np.ndarray, pas_best_layers: Dict[str, int]):
        out = {}
        for trait in self.cfg.trait_mapping.values():
            hi_key, lo_key = f"{trait}_high", f"{trait}_low"
            cls_res = {}

            # PAS single-layer: projection separability
            Lpas = pas_best_layers.get(trait, None)
            if Lpas is not None and hi_key in activs[Lpas] and lo_key in activs[Lpas]:
                Xh, Xl = activs[Lpas][hi_key], activs[Lpas][lo_key]
                if len(Xh) >= 10 and len(Xl) >= 10:
                    cls_res["pas_accuracy"] = float(self._proj_auc_balacc(Xh, Xl))

            # Subspace weighted-multilayer: project each layer into subspace with its weight
            Xh_sub, Xl_sub = [], []
            for i, L in enumerate(self.cfg.layer_range):
                if hi_key in activs[L]:
                    Xh_sub.append((activs[L][hi_key] * layer_weights[i]) @ subspace)
                if lo_key in activs[L]:
                    Xl_sub.append((activs[L][lo_key] * layer_weights[i]) @ subspace)
            if Xh_sub and Xl_sub:
                Xh_cat = np.vstack(Xh_sub)
                Xl_cat = np.vstack(Xl_sub)
                cls_res["subspace_accuracy"] = float(self._proj_auc_balacc(Xh_cat, Xl_cat))

            if cls_res:
                out[trait] = cls_res
        return out
