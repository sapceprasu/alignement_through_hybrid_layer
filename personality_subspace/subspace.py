# -*- coding: utf-8 -*-
from typing import Dict, Tuple
import numpy as np
from sklearn.decomposition import PCA
from .config import Config

class WeightedPersonalitySubspace:
    """
    Combines per-layer means with learned weights, then forms trait direction vector (high-low),
    L2-normalizes it, and builds a PCA subspace in *model space* (no feature standardization to
    keep basis aligned with model hidden states).
    """
    def __init__(self, cfg: Config, layer_weights: np.ndarray):
        self.cfg = cfg
        self.layer_weights = layer_weights
        self.subspace: np.ndarray = None
        self.pca: PCA = None

    def compute_weighted_directions(self, activs) -> Dict[str, Dict[str, np.ndarray]]:
        out = {}
        for trait in self.cfg.trait_mapping.values():
            hi_key, lo_key = f"{trait}_high", f"{trait}_low"
            hi_means, lo_means = [], []
            print("self.cfg.high_rep :", self.cfg.layer_range)
            for i, L in enumerate(self.cfg.layer_range):
                if hi_key in activs[L] and lo_key in activs[L]:
                    Xh, Xl = activs[L][hi_key], activs[L][lo_key]
                    # joint standardization before averaging
                    Xh, Xl = Xh, Xl  # already standardized upstream; if not, consider applying here
                    mh, ml = Xh.mean(axis=0), Xl.mean(axis=0)
                    hi_means.append(self.layer_weights[i] * mh)
                    lo_means.append(self.layer_weights[i] * ml)
            if not hi_means:
                continue
            comb_hi = np.sum(hi_means, axis=0)
            comb_lo = np.sum(lo_means, axis=0)
            direction = comb_hi - comb_lo
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            out[trait] = {"direction": direction, "combined_high": comb_hi, "combined_low": comb_lo}
        return out

    def build(self, weighted_dirs: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        if not weighted_dirs:
            raise RuntimeError("No weighted directions to build subspace.")
        D = np.stack([v["direction"] for v in weighted_dirs.values()], axis=0)  # [T, d]
        # Optionally mean-center across traits:
        D = D - D.mean(axis=0, keepdims=True)
        self.pca = PCA(n_components=min(self.cfg.n_components, D.shape[0], D.shape[1]), svd_solver="full")
        self.pca.fit(D)
        self.subspace = self.pca.components_.T  # [d, k]
        return self.subspace, self.pca.explained_variance_ratio_
