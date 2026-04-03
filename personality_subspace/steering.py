# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Tuple
import os, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = ["PersonalitySteerer", "SteerConfigPatch"]

# ------------------------ helpers ------------------------
def _add_delta_to_output(out, delta: torch.Tensor):
    """
    Add delta to the hidden-states part of a decoder layer output.
    Updated to preserve Tuple vs ModelOutput types strictly.
    """
    # 1. Handle Standard Tensor
    if torch.is_tensor(out):
        return out + delta.to(out.dtype).to(out.device)
    
    # 2. Handle Tuple/List (Standard HuggingFace)
    # We explicitly verify we have a container with a tensor at index 0
    if isinstance(out, (tuple, list)) and len(out) >= 1 and torch.is_tensor(out[0]):
        hs = out[0]
        # Cast delta to exact dtype of hidden state (bfloat16 safe)
        d = delta.to(hs.dtype).to(hs.device)
        
        # Create the new hidden state
        new_hs = hs + d
        
        # CRITICAL FIX: Reconstruct using the exact original type
        # If it was a specific subclass of tuple, this keeps it.
        # If it was a list, this keeps it.
        return type(out)((new_hs,) + tuple(out[1:]))

    # 3. Fallback for unexpected types
    # If we can't safely modify it, we return it untouched and warn.
    # This prevents "gibberish" by effectively turning off steering rather than breaking the model.
    print(f"[CRITICAL WARN] Output type {type(out)} unknown. Steering skipped to prevent corruption.")
    return out


# def _add_delta_to_output(out, delta: torch.Tensor):
#     """
#     Add delta to the hidden-states part of a decoder layer output.
#     Handles both Tensor and Tuple outputs (Universal compatibility).
#     """
#     # Case A: Standard Tensor (Old models)
#     if torch.is_tensor(out):
#         return out + delta.to(out.dtype).to(out.device)
    
#     # Case B: Tuple (Gemma 3, Llama 3, Mistral)
#     # The hidden states are always at index 0
#     if isinstance(out, tuple) and len(out) >= 1 and torch.is_tensor(out[0]):
#         hs = out[0]
#         # Robust casting to ensure no float16/bfloat16 mismatches
#         d = delta.to(hs.dtype).to(hs.device)
#         new_hs = hs + d
#         return (new_hs,) + out[1:]
        
#     # Case C: Fallback/Error
#     print(f"[CRITICAL WARN] Steering failed! Output type {type(out)} not recognized or index 0 not tensor.")
#     return out

# ------------------------ simple config patcher ------------------------

class SteerConfigPatch:
    """
    Context manager to temporarily set (layer_range, layer_weights, steer_mode='weighted')
    for composite steering / calibration code.
    """
    def __init__(self, steerer, layers: List[int], weights: Optional[List[float]] = None):
        self.steerer = steerer
        self.layers = list(layers)
        self.weights = list(weights) if weights is not None else None
        # save
        self._old_range = list(steerer.cfg.layer_range)
        self._old_weights = getattr(steerer.cfg, "layer_weights", None)
        self._old_mode = steerer.steer_mode

    def __enter__(self):
        self.steerer.cfg.layer_range = list(self.layers)
        setattr(self.steerer.cfg, "layer_weights", list(self.weights) if self.weights is not None else None)
        self.steerer.steer_mode = "weighted"
        return self.steerer

    def __exit__(self, exc_type, exc, tb):
        self.steerer.cfg.layer_range = list(self._old_range)
        if self._old_weights is None and hasattr(self.steerer.cfg, "layer_weights"):
            try:
                delattr(self.steerer.cfg, "layer_weights")
            except Exception:
                setattr(self.steerer.cfg, "layer_weights", None)
        else:
            setattr(self.steerer.cfg, "layer_weights", self._old_weights)
        self.steerer.steer_mode = self._old_mode
        return False

# ------------------------ steerer ------------------------

class PersonalitySteerer:
    """
    Hook-based steering: inject a delta into residual stream at selected sites.
    """

    # ---- lightweight logging gate ----
    _LVL = {"silent": 0, "warn": 1, "info": 2, "debug": 3}

    def _log(self, level: str, msg: str):
        if self._LVL.get(self.log_level, 2) >= self._LVL.get(level, 2):
            print(msg)

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        subspace: Optional[np.ndarray],
        trait_dirs: Dict[str, np.ndarray],
        cfg
    ):
        self.model = model.eval()
        self.tok   = tokenizer
        self.cfg   = cfg
        self.device = next(self.model.parameters()).device
        self.dtype  = next(self.model.parameters()).dtype

        # --- logging controls (from cfg, optional) ---
        self.log_level      = getattr(self.cfg, "log_level", "info")   # "silent"|"warn"|"info"|"debug"
        self.hook_log_every = int(getattr(self.cfg, "hook_log_every", 0))  # 0 => only first hit per site

        # Always create hooks list first so _clear() is safe
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._dbg_hits: Dict[str, int] = {}

        # --------- knobs ---------
        self.steer_mode: str = "pas"                                       # "pas" or "weighted"
        self.injection_point: str = getattr(self.cfg, "injection_point", "post")
        valid_injection_points = {"post", "mha", "mlp", "final_norm"}
        if self.injection_point not in valid_injection_points:
            raise ValueError(
                f"Invalid injection_point='{self.injection_point}'. "
                f"Valid options: {sorted(valid_injection_points)}"
            )
        print("[steerer] using PersonalitySteerer from", __file__)

        # "abs": alpha is raw units; "frac": alpha is fraction of residual RMS
        self.alpha_mode: str = getattr(self.cfg, "alpha_mode", "abs")

        
        self.last_position_only: bool = bool(getattr(self.cfg, "last_position_only", True))
        self.delta_cap_ratio: float   = float(getattr(self.cfg, "delta_cap_ratio", 0.0))  # 0 => off
        self.steer_tokens: int        = int(getattr(self.cfg, "steer_tokens", 0))         # 0 => unlimited
        # dynamic fractional scaling at final_norm (no prior RMS pass needed)
        self.frac_dynamic: bool       = bool(getattr(self.cfg, "frac_dynamic", True))
        # Whether to mean-center the injected vector before adding
        self.zero_center_delta: bool  = bool(getattr(self.cfg, "zero_center_delta", False))
        # runtime counters
        self._steer_budget_remaining: Optional[int] = None

        # --- Build orthonormal subspace (QR in fp32 on CPU), then move to model device/dtype.
        self.subspace = None
        if subspace is not None:
            subspace_np = subspace if isinstance(subspace, np.ndarray) else np.asarray(subspace)
            subspace_t = torch.tensor(subspace_np, dtype=torch.float32, device="cpu")
            q, _ = torch.linalg.qr(subspace_t, mode="reduced")
            self.subspace = q.to(self.device, dtype=self.dtype)

        # Pre-project trait directions into subspace and unit-normalize
        self.trait_unit = {}
        for t, v in trait_dirs.items():
            vv = torch.tensor(v, device=self.device, dtype=self.dtype)
            if self.subspace is not None:
                vv = self.subspace @ (self.subspace.T @ vv)
            nrm = vv.norm(p=2)
            if float(nrm) < 1e-12:
                vv = torch.ones_like(vv)
                nrm = vv.norm(p=2)
            vv = vv / (nrm + 1e-12)
            self.trait_unit[t.lower()] = vv
        

        self._layers = []
        # lets define differnt priorities for different models
        # PRIORITY 1: Standard LLMs (Llama, Mistral, Qwen, Gemma 2)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self._layers = self.model.model.layers
            
        # PRIORITY 2: Gemma 3 / Multimodal Models
        # We only look here if Priority 1 failed.
        # Path A: Gemma 3 often hides the LLM inside 'language_model'
        elif hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
             self._layers = self.model.model.language_model.layers
        # Path B: Alternative wrapping
        elif hasattr(self.model, "language_model") and hasattr(self.model.language_model, "layers"):
            self._layers = self.model.language_model.layers

        # PRIORITY 3: Older Architectures
        elif hasattr(self.model, "model") and hasattr(self.model.model, "decoder") and hasattr(self.model.model.decoder, "layers"):
            self._layers = self.model.model.decoder.layers

        # PRIORITY 4: "Nuclear" Fallback Search
        # Only runs if the model is completely unrecognizable 
        if len(self._layers) == 0:
            print("[steerer] WARN: Standard layer paths failed. Searching recursively...")
            candidates = []
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.ModuleList) and "layers" in name.lower():
                    candidates.append((len(module), module))
            
            if candidates:
                # Sort by length
                candidates.sort(key=lambda x: x[0], reverse=True)
                print(f"[steerer] FOUND layers via search. Count: {candidates[0][0]}")
                self._layers = candidates[0][1]

        # Final sanity check
        self._layers = list(self._layers)
        if len(self._layers) == 0:
            raise RuntimeError(f"CRITICAL: Could not find any layers in {type(self.model)}. Hooks cannot fire.")
            
        print(f"[steerer] Hooks attached to {len(self._layers)} layers.")
        # ------------------------------------------------------------------

        # sanity: cfg.layer_range must be 1-based
        n_layers = len(self._layers)
        for L in self.cfg.layer_range:
            if not (1 <= L <= n_layers):
                raise ValueError(f"Layer {L} out of range 1..{n_layers}")

        # final RMSNorm 
        self._final_norm = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            self._final_norm = self.model.model.norm

        # PAS/verified best layers 
        self._trait_layers = {}
        results_dir = getattr(self.cfg, "results_dir", None)
        if results_dir:
            # PAS
            path = os.path.join(results_dir, "pas_best_layers.json")
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        best = json.load(f)
                    for t, L in best.items():
                        if isinstance(L, int):
                            self._trait_layers[t.lower()] = [L]
            except Exception:
                pass
            # Verified
            try:
                path2 = os.path.join(results_dir, "layer_verified.json")
                if os.path.exists(path2):
                    with open(path2, "r", encoding="utf-8") as f:
                        best = json.load(f)
                    for t, L in best.items():
                        key = str(t).lower()
                        if isinstance(L, int):
                            self._trait_layers[key] = [int(L)]
                        elif isinstance(L, list) and len(L) > 0:
                            self._trait_layers[key] = [int(x) for x in L]
                    self._log("info", f"[steerer] verified layers: {self._trait_layers}")
            except Exception as e:
                self._log("warn", f"[steerer] could not read layer_verified.json: {e}")

        # caches
        self._layer_rms: Dict[int, float] = {}   # per-prompt residual RMS
        self._polarity_cache: Dict[str, int] = {}
        self._polarity_sig: Dict[str, tuple] = {}

        # steer gain & override
        self.steer_gain = float(getattr(self.cfg, "steer_gain", 16) or 16)
        self.polarity_override: Dict[str, int] = dict(getattr(self.cfg, "polarity_override", {}) or {})

    # ---------- helpers fxnsss ----------

    def _layers_for_trait(self, trait: str) -> List[int]:
        return self._trait_layers.get(trait.lower(), list(self.cfg.layer_range))

    def _clear(self):
        for h in getattr(self, "_hooks", []):
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []
        self._dbg_hits.clear()
        self._steer_budget_remaining = None

    def _should_hook_log(self, hid: str) -> bool:
        c = self._dbg_hits.get(hid, 0) + 1
        self._dbg_hits[hid] = c
        if self.hook_log_every <= 0:
            return c == 1
        return c == 1 or (c % self.hook_log_every == 0)

    def _budget_allows_and_tick(self, like: torch.Tensor) -> bool:
        """Coarse budget: limit number of *applications* of delta."""
        if self._steer_budget_remaining is None:
            return True
        if self._steer_budget_remaining <= 0:
            return False
        self._steer_budget_remaining -= 1
        return True

    def _cap_delta(self, delta: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Cap per-position L2 of delta to <= ratio * per-position L2 of like."""
        r = float(getattr(self, "delta_cap_ratio", 0.0) or 0.0)
        if r <= 0.0:
            return delta
        if delta.shape != like.shape:
            delta = delta.expand_as(like)
        xnorm = torch.linalg.vector_norm(like, dim=-1)            # 
        dnorm = torch.linalg.vector_norm(delta, dim=-1) + 1e-12   # 
        scale = torch.clamp(r * xnorm / dnorm, max=1.0).unsqueeze(-1)
        return delta * scale

    # ---------- per-prompt RMS ----------
    @torch.no_grad()
    def _measure_layer_rms(self, formatted_text: str):
        self._layer_rms = {}
        if self.alpha_mode != "frac":
            return
        if self.injection_point == "final_norm" and self.frac_dynamic:
            return

        enc = self.tok(formatted_text, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        inject_where = self.injection_point

        handles = []

        def make_block_post_hook(L):
            def hook(_m, _inputs, output):
                x = output if torch.is_tensor(output) else output[0]
                self._layer_rms[L] = x.detach().float().pow(2).mean().sqrt().item()
                return output
            return hook

        def make_mha_hook(L):
            def hook(_m, _inputs, output):
                x = output if torch.is_tensor(output) else output[0]
                self._layer_rms[L] = x.detach().float().pow(2).mean().sqrt().item()
                return output
            return hook

        def make_mlp_prehook(L):
            def prehook(_m, inputs):
                x = inputs[0]
                self._layer_rms[L] = x.detach().float().pow(2).mean().sqrt().item()
                return inputs
            return prehook

        def make_final_norm_hook():
            def hook(_m, _inputs, output):
                x = output if torch.is_tensor(output) else output[0]
                self._layer_rms[0] = x.detach().float().pow(2).mean().sqrt().item()
                return output
            return hook

        if inject_where == "post":
            for L in list(self.cfg.layer_range):
                handles.append(self._layers[L - 1].register_forward_hook(make_block_post_hook(L)))
        elif inject_where == "mha":
            for L in list(self.cfg.layer_range):
                handles.append(self._layers[L - 1].self_attn.register_forward_hook(make_mha_hook(L)))
        elif inject_where == "mlp":
            for L in list(self.cfg.layer_range):
                handles.append(self._layers[L - 1].mlp.register_forward_pre_hook(make_mlp_prehook(L)))
        elif inject_where == "final_norm" and self._final_norm is not None:
            handles.append(self._final_norm.register_forward_hook(make_final_norm_hook()))
        else:
            for L in list(self.cfg.layer_range):
                handles.append(self._layers[L - 1].register_forward_hook(make_block_post_hook(L)))

        _ = self.model(**enc)
        for h in handles:
            h.remove()

        sample_keys = sorted(self._layer_rms.keys())
        sample_preview = [(k, round(self._layer_rms[k], 3)) for k in sample_keys[:3]]
        self._log("debug", f"[rms] inj={self.injection_point} keys={sample_keys[:10]} sample={sample_preview}")

    # ---------- polarity calibration ----------

    @torch.no_grad()
    def _calibrate_polarity(self, trait: str) -> int:
        trait = trait.lower()
        if trait in self._polarity_cache:
            return self._polarity_cache[trait]

        prompts = [
            "Write one honest sentence about your day.",
            "Respond briefly and naturally.",
        ]

        @torch.no_grad()
        def next_probs(text):
            s = self._format_prompt(text, True)
            if self.alpha_mode == "frac":
                self._measure_layer_rms(s)
            enc = self.tok(s, return_tensors="pt").to(self.device)
            out = self.model(**enc)
            return torch.softmax(out.logits[:, -1, :].float(), dim=-1).squeeze(0)

        base = [next_probs(p) for p in prompts]

        def kl_dir(sign):
            self._register(trait, 0.75 * sign, skip_calibration=True)
            try:
                hooked = [next_probs(p) for p in prompts]
            finally:
                self._clear()
            kls = []
            eps = 1e-9
            for b, h in zip(base, hooked):
                kls.append(torch.sum(b * (torch.log(b + eps) - torch.log(h + eps))).item())
            return float(np.mean(kls))

        pos = kl_dir(+1)
        neg = kl_dir(-1)
        sgn = +1 if pos >= neg else -1
        self._polarity_cache[trait] = sgn
        self._log("debug", f"[polarity] trait={trait} posKL={pos:.4g} negKL={neg:.4g} sgn={sgn:+d}")
        return sgn

    # ---------- alpha calibration ----------

    @torch.no_grad()
    def calibrate_alpha(
        self,
        trait: str,
        prompts: List[str],
        *,
        target_kl: float = 0.10,
        tol: float = 0.01,
        alpha_lo: float = 0.0,
        alpha_hi: float = 8.0,
        topk: Optional[int] = None,
        max_steps: int = 12,
        use_chat_template: bool = True,
        system: Optional[str] = None
    ) -> Tuple[float, Dict[str, float]]:
        dev = self.device
        trait = trait.lower()

        if trait in getattr(self, "polarity_override", {}):
            sgn = +1 if int(self.polarity_override[trait]) >= 0 else -1
        else:
            sgn = self._calibrate_polarity(trait)

        def next_probs(text: str) -> torch.Tensor:
            s = self._format_prompt(text, use_chat_template, system=system)
            if self.alpha_mode == "frac":
                self._measure_layer_rms(s)
            enc = self.tok(s, return_tensors="pt").to(dev)
            out = self.model(**enc)
            return torch.softmax(out.logits[:, -1, :].float(), dim=-1).squeeze(0)

        bases = [next_probs(p) for p in prompts]

        def steered_probs(text: str, alpha_signed: float) -> torch.Tensor:
            s = self._format_prompt(text, use_chat_template, system=system)
            if self.alpha_mode == "frac":
                self._measure_layer_rms(s)
            enc = self.tok(s, return_tensors="pt").to(dev)
            self._register(trait, 1.0, skip_calibration=True, alpha_override=alpha_signed)
            try:
                out = self.model(**enc)
            finally:
                self._clear()
            logits = out.logits[:, -1, :].float().squeeze(0)
            if topk is None:
                return torch.softmax(logits, dim=-1)
            with torch.no_grad():
                p = bases[0]
                idx = torch.topk(p, k=min(topk, p.numel())).indices
                logits_k = logits[idx]
                qk = torch.softmax(logits_k, dim=-1)
                full = torch.zeros_like(p)
                full[idx] = qk
                return full

        def avg_kl_for_alpha(a_signed: float) -> float:
            eps = 1e-9
            kls = []
            for b, text in zip(bases, prompts):
                q = steered_probs(text, a_signed).clamp_min(eps)
                p = b.clamp_min(eps)
                kls.append(torch.sum(p * (torch.log(p) - torch.log(q))).item())
            return float(np.mean(kls))

        if target_kl <= tol:
            return float(sgn * 0.0), {"sgn": sgn, "alpha": 0.0, "kl": 0.0}

        lo, hi = float(alpha_lo), float(max(alpha_lo, alpha_hi))
        for _ in range(6):
            kl_hi = avg_kl_for_alpha(sgn * hi)
            if kl_hi >= target_kl or hi > 1e3:
                break
            hi *= 2.0

        best_alpha = hi
        best_kl = avg_kl_for_alpha(sgn * best_alpha)
        for _ in range(max_steps):
            mid = 0.5 * (lo + hi)
            kl_mid = avg_kl_for_alpha(sgn * mid)
            if abs(kl_mid - target_kl) < abs(best_kl - target_kl):
                best_alpha, best_kl = mid, kl_mid
            if abs(kl_mid - target_kl) <= tol:
                best_alpha, best_kl = mid, kl_mid
                break
            if kl_mid < target_kl:
                lo = mid
            else:
                hi = mid

        alpha_eff = float(sgn * best_alpha)
        return alpha_eff, {"sgn": float(sgn), "alpha_abs": float(best_alpha), "alpha": float(alpha_eff), "kl": float(best_kl)}

    # ---------- injection fxn ----------

    def _register(
        self,
        trait: str,
        intensity: float,
        *,
        skip_calibration: bool = False,
        alpha_override: Optional[float] = None
    ):
        self._clear()
        self._steer_budget_remaining = int(self.steer_tokens) if int(self.steer_tokens) > 0 else None

        trait = trait.lower()
        if trait not in self.trait_unit:
            raise ValueError(f"Unknown trait '{trait}'")

        v_unit = self.trait_unit[trait]

        cur_layers = list(self.cfg.layer_range)
        cur_weights = list(getattr(self.cfg, "layer_weights", []) or [])
        if self.steer_mode == "pas":
            cur_layers = self._layers_for_trait(trait)
            cur_weights = []
        sig = (
            self.injection_point,
            self.alpha_mode,
            self.steer_mode,
            tuple(cur_layers),
            tuple(cur_weights),
            bool(self.last_position_only),
            float(self.delta_cap_ratio),
        )
        prev = self._polarity_sig.get(trait)
        if prev != sig:
            self._polarity_cache.pop(trait, None)
            self._polarity_sig[trait] = sig

        val = None
        try:
            val = getattr(self, "polarity_override", {}).get(trait, None)
        except Exception:
            val = None

        if val is not None:
            sgn = +1 if int(val) >= 0 else -1
        elif skip_calibration:
            sgn = +1
        else:
            sgn = self._calibrate_polarity(trait)

        if alpha_override is not None:
            alpha = float(alpha_override)
        else:
            alpha = sgn * float(intensity) * float(self.steer_gain)

        self._alpha_last = alpha
        inj = self.injection_point
        mode = self.steer_mode
        layer_weights = getattr(self.cfg, "layer_weights", None)

        if mode == "pas":
            layers = self._layers_for_trait(trait)
            weights = None
        else:
            layers = list(self.cfg.layer_range)
            weights = layer_weights

        ZERO_CENTER_DELTA = self.zero_center_delta

        def _make_vec(scale: float, v: torch.Tensor) -> torch.Tensor:
            vec = (scale * v).to(self.dtype).to(self.device)
            if ZERO_CENTER_DELTA:
                vec = vec - vec.mean()
            return vec

        def _vec_to_full_delta(vec: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
            if like.ndim >= 3 and self.last_position_only:
                delta_full = torch.zeros_like(like)
                delta_full[:, -1, :] = vec.to(like.dtype).to(like.device)
                return delta_full
            d = vec
            while d.ndim < like.ndim:
                d = d.unsqueeze(0)
            return d.to(like.dtype).to(like.device)


# ---- hook factories with rate-limited prints ----
        def post_hook_factory(L, w):
            def hook_fn(_m, _x, out):
                # 1. Get the input tensor safely
                x = out if torch.is_tensor(out) else out[0]
                
                if not self._budget_allows_and_tick(x):
                    return out
                
                # 2. Calculate the Steering Vector
                rms = self._layer_rms.get(L, 1.0) if self.alpha_mode == "frac" else 1.0
                scale = alpha * rms * w
                
                vec = _make_vec(scale, v_unit)
                delta = _vec_to_full_delta(vec, x)
                delta = self._cap_delta(delta, x)
                
                # 3. Apply the Steering
                new_out = _add_delta_to_output(out, delta)
                
                # DIAGNOSTIC PROBE --- per generation stats needs checking
          
                if L == self.cfg.layer_range[0] and not hasattr(self, "_has_printed_diag"):
                    # Use float32 for calculations
                    x_f = x.detach().float()
                    d_f = delta.detach().float()
                    v_f = v_unit.detach().float()
                    
                    # A. MAGNITUDE (Volume)
                    # We measure the last token (the one generating the next word)
                    x_last = x_f[0, -1, :] 
                    d_last = d_f[0, -1, :]
                    
                    x_norm = x_last.norm().item()
                    d_norm = d_last.norm().item()
                    ratio = d_norm / (x_norm + 1e-9)
                    
                    # B. ALIGNMENT (Direction)
                    # Cosine: 1.0 = Perfect Alignment, 0.0 = Orthogonal (90 degrees), -1.0 = Opposite
                    x_new = new_out if torch.is_tensor(new_out) else new_out[0]
                    x_new_last = x_new.detach().float()[0, -1, :]
                    
                    # How much did we rotate the thought vector?
                    cos_before = torch.nn.functional.cosine_similarity(x_last.unsqueeze(0), v_f.unsqueeze(0)).item()
                    cos_after  = torch.nn.functional.cosine_similarity(x_new_last.unsqueeze(0), v_f.unsqueeze(0)).item()
                    rotation   = cos_after - cos_before

                    print(f"\n[DIAGNOSTIC] Layer {L} | Gain: {self.steer_gain} | Mode: {self.alpha_mode}")
                    print(f"   VOLUME (Ratio):  {ratio:.4f}  (Target: 0.05 - 0.15)")
                    print(f"   SIGNAL (Cosine): {cos_before:.4f} -> {cos_after:.4f} (Delta: {rotation:+.4f})")
                    
                    # C. THE DIAGNOSIS
                    if ratio > 0.15 and rotation < 0.02:
                        print("   >>> FAIL CAUSE: 'ALL STATIC'. You are shouting (High Volume) but saying nothing (Low Rotation).")
                        print("       The vector is orthogonal to the model's meaning. It adds noise but doesn't steer.")
                    
                    elif ratio < 0.02 and rotation < 0.01:
                        print("   >>> FAIL CAUSE: 'WHISPER'. Too weak to matter.")
                        
                    elif ratio > 0.20:
                        print("   >>> FAIL CAUSE: 'BROKEN SPEAKER'. Volume is unsafe (>20%). Gibberish guaranteed.")
                        
                    elif rotation > 0.05:
                        print("   >>> STATUS: WORKING. Significant rotation achieved.")
                        
                    else:
                        print("   >>> STATUS: UNKNOWN. Borderline.")

                    self._has_printed_diag = True
                # -----------------------------------------------------------

                return new_out
            return hook_fn

        # ---- hook factories with rate-limited prints ----
        # def post_hook_factory(L, w):
        #     def hook_fn(_m, _x, out):
        #         x = out if torch.is_tensor(out) else out[0]
                
        #         # --- DEBUG PROBE ---
        #         if L == self.cfg.layer_range[0] and self.hook_log_every >= 0:
        #             if not hasattr(self, "_has_printed_hook"):
        #                 print(f"\n[DEBUG HOOK] Layer {L} is ACTIVE. Input shape: {x.shape}, Dtype: {x.dtype}")
        #                 self._has_printed_hook = True
        #         # -------------------

        #         if not self._budget_allows_and_tick(x):
        #             return out
                
        #         scale = alpha * (self._layer_rms.get(L, 1.0) if self.alpha_mode == "frac" else 1.0) * w
                
        #         vec = _make_vec(scale, v_unit)
        #         delta = _vec_to_full_delta(vec, x)
        #         delta = self._cap_delta(delta, x)
                
        #         hid = f"L{L}-post"
        #         if self._should_hook_log(hid):
        #             sgn_chr = "+" if scale >= 0 else "-"
        #             self._log("info", f"[delta] inj=post L={L} w={w:.3f} scale={abs(scale):.4f} ({sgn_chr}) last_only={bool(self.last_position_only)} cap={float(self.delta_cap_ratio):.1f}")
        #         return _add_delta_to_output(out, delta)
        #     return hook_fn

        def mha_hook_factory(L, w):
            def hook_fn(_m, _x, out):
                x = out if torch.is_tensor(out) else out[0]
                if not self._budget_allows_and_tick(x):
                    return out
                scale = alpha * (self._layer_rms.get(L, 1.0) if self.alpha_mode == "frac" else 1.0) * w
                vec = _make_vec(scale, v_unit)
                delta = _vec_to_full_delta(vec, x)
                delta = self._cap_delta(delta, x)
                hid = f"L{L}-mha"
                if self._should_hook_log(hid):
                    sgn_chr = "+" if scale >= 0 else "-"
                    self._log("info", f"[delta] inj=mha  L={L} w={w:.3f} scale={abs(scale):.4f} ({sgn_chr}) last_only={bool(self.last_position_only)} cap={float(self.delta_cap_ratio):.1f}")
                return _add_delta_to_output(out, delta)
            return hook_fn

        def mlp_prehook_factory(L, w):
            def pre_fn(_m, inputs):
                x = inputs[0]
                if not self._budget_allows_and_tick(x):
                    return inputs
                scale = alpha * (self._layer_rms.get(L, 1.0) if self.alpha_mode == "frac" else 1.0) * w
                vec = _make_vec(scale, v_unit)
                delta = _vec_to_full_delta(vec, x)
                delta = self._cap_delta(delta, x)
                hid = f"L{L}-mlp"
                if self._should_hook_log(hid):
                    sgn_chr = "+" if scale >= 0 else "-"
                    self._log("info", f"[delta] inj=mlp  L={L} w={w:.3f} scale={abs(scale):.4f} ({sgn_chr}) last_only={bool(self.last_position_only)} cap={float(self.delta_cap_ratio):.1f}")
                x = x + delta
                return (x, *inputs[1:])
            return pre_fn

        if inj == "final_norm":
            if self._final_norm is None:
                raise RuntimeError("final_norm requested but model.model.norm not found")
            def final_forward_hook(_m, _inputs, output):
                like = output if torch.is_tensor(output) else output[0]
                if not self._budget_allows_and_tick(like):
                    return output
                if self.alpha_mode == "frac":
                    if self.frac_dynamic:
                        if like.ndim >= 3:
                            rms = like[..., -1, :].detach().float().pow(2).mean().sqrt()
                        else:
                            rms = like.detach().float().pow(2).mean().sqrt()
                        layer_scale = float(rms.item())
                    else:
                        layer_scale = float(self._layer_rms.get(0, 1.0))
                else:
                    layer_scale = 1.0
                scale = alpha * layer_scale
                vec = _make_vec(scale, v_unit)
                delta = _vec_to_full_delta(vec, like)
                delta = self._cap_delta(delta, like)
                if self._should_hook_log("final_norm"):
                    sgn_chr = "+" if scale >= 0 else "-"
                    self._log("info", f"[delta] inj=final_norm scale={abs(scale):.4f} ({sgn_chr})")
                return _add_delta_to_output(output, delta)
            self._hooks.append(self._final_norm.register_forward_hook(final_forward_hook))
            return

        # otherwise per-layer hooks
        weights = (layer_weights or [1.0] * len(layers)) if mode == "weighted" else (weights or [1.0] * len(layers))
        for i, L in enumerate(layers):
            w = float(weights[i]) if i < len(weights) else 1.0
            block = self._layers[L - 1]
            if inj == "post":
                self._hooks.append(block.register_forward_hook(post_hook_factory(L, w)))
            elif inj == "mha":
                self._hooks.append(block.self_attn.register_forward_hook(mha_hook_factory(L, w)))
            elif inj == "mlp":
                self._hooks.append(block.mlp.register_forward_pre_hook(mlp_prehook_factory(L, w)))
            else:
                self._hooks.append(block.register_forward_hook(post_hook_factory(L, w)))

    # ---------- prompt formatting & public generate ----------

    def _format_prompt(self, prompt: str, use_chat_template: bool = True, system: str = None) -> str:
        if use_chat_template and hasattr(self.tok, "apply_chat_template"):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    @torch.no_grad()
    def generate(self, prompt: str, trait: str, intensity: float = 1.0, temperature: float = 0.8,
                 max_new_tokens: int = 120, use_chat_template: bool = True,
                 alpha_override: Optional[float] = None, **kw) -> str:
        text = self._format_prompt(prompt, use_chat_template=use_chat_template, system=kw.get("system"))
        self._measure_layer_rms(text)
        self._register(trait, intensity, alpha_override=alpha_override)
        try:
            enc = self.tok(text, return_tensors="pt").to(self.device)
            prompt_len = enc["input_ids"].shape[1]
            eos_id = self.tok.eos_token_id
            pad_id = self.tok.pad_token_id or eos_id
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=kw.get("do_sample", True),
                temperature=kw.get("temperature", 0.4),
                top_p=kw.get("top_p", 0.95),
                top_k=kw.get("top_k", 50),
                repetition_penalty=kw.get("repetition_penalty", 1.1),
                no_repeat_ngram_size=kw.get("no_repeat_ngram_size", 3),
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                return_dict_in_generate=False
            )
            gen_ids = out[0][prompt_len:]
            return self.tok.decode(gen_ids, skip_special_tokens=True)
        finally:
            self._clear()

    def steer(self, *a, **k):
        return self.generate(*a, **k)
