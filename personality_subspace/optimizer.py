
import numpy as np
from typing import Dict
from .config import Config

class LayerWeightOptimizer:
    """
    Closed-form, fast layer weights:
    S_L = mean_trait || μ_high(L, t) − μ_low(L, t) ||_2
    w = softmax(S / τ). Optionally sparsify top-k (commented).
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.layer_weights = None

    def optimize(self, activs: Dict[int, Dict[str, np.ndarray]]) -> np.ndarray:
        layers = list(self.cfg.layer_range)
        print(f"[LayerWeightOptimizer] scoring {len(layers)} layers...")
        k = len(layers)
        scores = np.zeros(k, dtype=np.float64)
        n_traits = 0

        for trait in self.cfg.trait_mapping.values():
            hi_key, lo_key = f"{trait}_high", f"{trait}_low"
            have_any = False
            for i, L in enumerate(layers):
                if hi_key in activs[L] and lo_key in activs[L]:
                    Xh = activs[L][hi_key]; Xl = activs[L][lo_key]
                    if Xh.size == 0 or Xl.size == 0:
                        continue
                    d = Xh.mean(axis=0) - Xl.mean(axis=0)
                    scores[i] += float(np.linalg.norm(d))
                    have_any = True
            if have_any:
                n_traits += 1

        if n_traits == 0:
            print("[WARN] no traits found for weight scoring; using uniform.")
            self.layer_weights = np.ones(k, dtype=np.float64) / k
            return self.layer_weights

        scores /= float(n_traits)
        # temperature (lower => peakier). use median-based scale.
        tau = max(np.median(scores[scores > 0]) * 0.5, 1e-6) if np.any(scores > 0) else 1.0
        z = (scores / tau)
        z = z - z.max()  # stabilize
        w = np.exp(z); w = w / (w.sum() + 1e-12)

        # optional sparsity:
        # m = min(3, k)
        # top = np.argsort(-scores)[:m]
        # mask = np.zeros_like(w); mask[top] = 1.0
        # w = w * mask; w = w / (w.sum() + 1e-12)

        self.layer_weights = w.astype(np.float64)
        return self.layer_weights
