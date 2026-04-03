# -*- coding: utf-8 -*-
import json, os
from typing import List, Dict, Any, Optional
import numpy as np
from .config import Config
from .utils import json_dump

class PersonalityDataset:
    def __init__(self, cfg: Config):
        # breakpoint()
        self.cfg = cfg
        self.trait_mapping = cfg.trait_mapping
        self.data = self._load_data()
        self.trait_levels = self._analyze_trait_distribution()

    def _load_data(self) -> List[Dict[str, Any]]:
        path = self.cfg.dataset_path
        data = []
        if not os.path.exists(path):
            print(f"[ERROR] dataset not found: {path}")
            return data

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    entry = json.loads(line)
                    trait = entry["trait"].lower()
                    level = entry["level"].lower()
                    text  = entry["text"]

                    # normalize trait name
                    for code, name in self.trait_mapping.items():
                        if trait in (name.lower(), code.lower()):
                            trait = name.lower()
                            break
                    data.append({"trait": trait, "level": level, "text": text})
                except Exception as e:
                    print(f"[WARN] bad line {i}: {e}")
        return data

    def _analyze_trait_distribution(self) -> Dict[str, int]:
        counts = {}
        for e in self.data:
            k = f"{e['trait']}_{e['level']}"
            counts[k] = counts.get(k, 0) + 1
        return counts

    def get_trait_samples(self, trait: str, level: str) -> List[Dict[str, Any]]:
        trait = trait.lower()
        valid = []
        for e in self.data:
            if e["level"] != level:
                continue
            if e["trait"] == trait:
                valid.append(e)
        return valid

    def get_balanced(self) -> List[Dict[str, Any]]:
        """Balanced by (trait, level); optional cap to control memory."""
        print(f"[DEBUG] max_samples_per_group={self.cfg.max_samples_per_group}")
        for trait in self.trait_mapping.values():
            for level in ["high","low"]:
                n = len(self.get_trait_samples(trait, level))
        print(f"[DEBUG] group {trait}_{level}: {n} raw")
        out = []
        cap = self.cfg.max_samples_per_group
        for trait in self.trait_mapping.values():
            for level in ["high", "low"]:
                group = self.get_trait_samples(trait, level)
                if cap is not None and len(group) > cap:
                    rng = np.random.default_rng(self.cfg.seed)
                    idx = rng.choice(len(group), size=cap, replace=False)
                    group = [group[i] for i in idx]
                out.extend(group)
                # rng.shuffle(out)
        return out

    def save_analysis(self, path: str):
        analysis = {
            "total_samples": len(self.data),
            "trait_distribution": self.trait_levels,
            "unique_traits": sorted({e["trait"] for e in self.data}),
            "unique_levels": sorted({e["level"] for e in self.data}),
            "sample_texts": [e["text"][:120] + "..." for e in self.data[:5]],
        }
        json_dump(analysis, path)
        print(f"[INFO] dataset analysis → {path}")
