# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional



@dataclass
class LayerSearchConfig:
    # tiny probe set; keep small to be cheap
    probe_prompts: List[str] = field(default_factory=lambda: [
        "Write one honest sentence about your day.",
        "Respond briefly and naturally.",
        "I have a big task due Friday. Continue: My plan is",
        "Describe your ideal weekend in one sentence.",
        "Give a short tip for staying organized.",
        "What motivates you to do your best work?",
        "How do you handle stressful situations?",      
        "What qualities do you value in a friend?",
    ])
    # intensity used during verification (kept modest)
    alpha_probe: float = 6.0
    # weights to combine metrics: Δlogits L2, first-token KL, flip-rate
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "delta_l2": 0.5, "first_kl": 0.4, "flip": 0.2
    })
    # choose top-1 or small top-k
    top_k: int = 1
    # evaluation injection point & mode, to mirror your intended runtime
    eval_injection_point: str = "post"      # "post" | "mlp" | "mha" | "final_norm"
    eval_steer_mode: str = "pas"            # "pas" | "weighted"


@dataclass
class OptimizationConfig:
    maxiter: int = 1000
    reg_lambda: float = 0.5       # L1 strength
    sparsity_weight: float = 0.3  # entropy penalty weight
    specialization_weight: float = 0.1  # encourages different trait distances

@dataclass
class Config:
    # model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    # model_name: str = "google/gemma-3-4b-it"
    model_name: str = "google/gemma-3-27b-it"
    # model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_name: str = "mistralai/Ministral-8B-Instruct-2410"
    # model_name: str = "openai/gpt-oss-20b"
    dataset_path: str = "personality_subspace/data/big5_chat_traitwise_5000.jsonl"
    # results_dir: str = "testnew_Meta-Llama-3-8B-Instruct_results"
    results_dir: str = "gemma-3-27b-it_results"
    # results_dir: str = "Ministral-8B-Instruct-2410_results"
    # results_dir: str = "gpt_oss_20b"
    # layer_range:  List[int] = field(default_factory=lambda: [10, 15, 18, 21, 24])  # 1-based indexing (embeddings=0)
    layer_range: List[int] = field(default_factory=lambda: list(range(10,44)))
    top_n_layers: int = 7
    n_components: int = 3
    separation_threshold: float = 3.5
    # high_rep_layers : List[int] =field(default_factory=list)
    device: Optional[str] = None  # if None, inferred from model's embeddings device
    batch_size: int = 8
    max_length: int = 128
    min_samples_per_trait: int = 0
    max_samples_per_group: Optional[int] = None   # cap per (trait,level) to control RAM
    checkpoint_every: int = 100  # save checkpoint every N batches
    seed: int = 42

    trait_mapping: Dict[str, str] = field(default_factory=lambda: {
        "O": "openness",
        "C": "conscientiousness",
        "E": "extraversion",
        "A": "agreeableness",
        "N": "neuroticism"
    })

    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    layer_search: LayerSearchConfig = field(default_factory=LayerSearchConfig)

    def ensure_dirs(self):
        os.makedirs(self.results_dir, exist_ok=True)
