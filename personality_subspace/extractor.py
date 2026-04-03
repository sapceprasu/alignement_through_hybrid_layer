
from typing import Dict, List, Any
import os, pickle
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from .config import Config
from .utils import json_dump

class MultiLayerActivationExtractor:
    """
    Extract last-token hidden states.
    UPDATED: Includes Debug Probes to verify model output.
    """
    def __init__(self, cfg: Config, results_dir: str):
        self.cfg = cfg
        self.results_dir = results_dir
        
        print(f"[DEBUG] Loading model: {cfg.model_name}")
        # Using bfloat16 is CRITICAL for Gemma/Llama-3 to avoid NaNs
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        
        # LEFT PADDING is safer for extraction because the last token is always at index -1
        self.tok.padding_side = "left"  

        # --- Robust Device Detection ---
        if cfg.device is None:
            if hasattr(self.model, "device"):
                self.dev = self.model.device
            else:
                self.dev = next(self.model.parameters()).device
        else:
            self.dev = torch.device(cfg.device)
        print(f"[DEBUG] Model device: {self.dev}")

        # --- Sanity Check / Layer Counting ---
        try:
            # We trust the config for layer count to avoid running a fragile dummy pass
            self.num_layers = getattr(self.model.config, "num_hidden_layers", None)
            if self.num_layers is None:
                self.num_layers = getattr(self.model.config, "n_layer", 32)
            print(f"[DEBUG] Detected {self.num_layers} layers.")
        except Exception as e:
            print(f"[WARNING] Could not detect layer count: {e}. Defaulting to 32.")
            self.num_layers = 32

    def _to_device(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.dev) for k, v in inputs.items()}

    def _get_hidden_states(self, texts: List[str]) -> Dict[int, np.ndarray]:
        enc = self.tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.cfg.max_length
        )
        enc = self._to_device(enc)
        
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        
        hs = out.hidden_states 
        
        last_ix = -1 
        
        layer_acts = {}
        for L in self.cfg.layer_range:
            if L >= len(hs): continue
            
            H = hs[L] # [Batch, Seq, Dim]
            
            # Extract last token simply using -1
            last = H[:, last_ix, :] 
            
            # Cast to float32 for storage (numpy hates bfloat16)
            layer_acts[L] = last.to("cpu", dtype=torch.float32).numpy()
            
        return layer_acts

    def extract(self, balanced_samples: List[Dict[str, Any]]) -> Dict[int, Dict[str, np.ndarray]]:
        per_layer: Dict[int, Dict[str, List[np.ndarray]]] = {
            L: {f"{t}_{lv}": [] for t in self.cfg.trait_mapping.values() for lv in ["high", "low"]}
            for L in self.cfg.layer_range
        }

        texts = [s["text"] for s in balanced_samples]
        meta  = [(s["trait"], s["level"]) for s in balanced_samples]

        print(f"[DEBUG] Starting extraction loop on {len(texts)} texts...")

        for i in tqdm(range(0, len(texts), self.cfg.batch_size), desc="Extracting activations"):
            batch_texts = texts[i:i+self.cfg.batch_size]
            batch_meta  = meta[i:i+self.cfg.batch_size]
            
            layer_acts = self._get_hidden_states(batch_texts)
            B = len(batch_texts)
            
            for L, acts in layer_acts.items():
                if L not in per_layer: continue

                # --- THE PROBE: PRINT RAW DATA TO CHECK IF MODEL IS DEAD ---
                # We check the first layer of the very first batch.
                if i == 0 and L == self.cfg.layer_range[0]:
                    print(f"\n[DEBUG PROBE] Layer {L} | Batch 0")
                    print(f"   Shape: {acts.shape}")
                    print(f"   Mean:  {acts.mean():.6f}")
                    print(f"   Std:   {acts.std():.6f}")
                    print(f"   First Sample (Partial): {acts[0][:8]}") # Print first 8 numbers
                    
                    if np.all(acts == 0):
                        print("\n!!! CRITICAL WARNING: ACTIVATIONS ARE ALL ZEROS !!!")
                        print("    This means you are likely extracting padding tokens or the layer index is wrong.")
                    
                    if np.isnan(acts).any():
                        print("\n!!! CRITICAL WARNING: ACTIVATIONS CONTAIN NaNs !!!")
                        print("    Model is unstable. Ensure you are using bfloat16.")
                # -----------------------------------------------------------

                for j in range(B):
                    trait, level = batch_meta[j]
                    per_layer[L][f"{trait}_{level}"].append(acts[j])

            # Checkpoint
            if ((i // self.cfg.batch_size) + 1) % self.cfg.checkpoint_every == 0:
                ck = os.path.join(self.results_dir, "ckpt_activations.pkl")
                with open(ck, "wb") as f:
                    pickle.dump(per_layer, f)
                print(f"[CKPT] saved partial activations → {ck}")

        # Stack to arrays
        out: Dict[int, Dict[str, np.ndarray]] = {L: {} for L in self.cfg.layer_range}
        for L in self.cfg.layer_range:
            if L not in per_layer: continue
            for key, lst in per_layer[L].items():
                if len(lst):
                    out[L][key] = np.stack(lst, axis=0)
        
        summary = {L: {k: (v.shape[0] if isinstance(v, np.ndarray) else 0) for k, v in out[L].items()} for L in out}
        json_dump(summary, os.path.join(self.results_dir, "activations_summary.json"))
        return out

    @staticmethod
    def joint_standardize_layer_trait(X_high: np.ndarray, X_low: np.ndarray):
        X_all = np.vstack([X_high, X_low])
        mu = X_all.mean(axis=0)
        sd = X_all.std(axis=0) + 1e-8
        
        # Debug check for standardizer
        if np.isnan(mu).any() or np.isnan(sd).any():
             print("[DEBUG] NaN detected during standardization step.")
             
        return (X_high - mu) / sd, (X_low - mu) / sd