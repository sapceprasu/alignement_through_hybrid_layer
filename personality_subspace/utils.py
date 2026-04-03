# -*- coding: utf-8 -*-
import json, os, random, csv
from typing import Any, Dict
import numpy as np
import torch

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def json_dump(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def json_load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class CSVLogger:
    def __init__(self, path: str, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        self._init_file()

    def _init_file(self):
        new = not os.path.exists(self.path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            if new:
                w.writeheader()

    def log(self, row: Dict[str, Any]):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writerow(row)
