"""Minimal loader for BoLT's DMCurriculumMO emulator (https://github.com/chewwt/bolt).

The bolt-bench package requires botorch>=0.16.1 / torch>=2.9.1, which conflicts with
this repo's pinned botorch==0.10.0 (whose FixedNoiseGP the GP wrapper depends on).
The emulator itself has no botorch dependency: it is a pretrained 3-layer MLP hosted
on the HuggingFace Hub, so this module reproduces bolt's MLPFunction/FeatureNet
forward pass and un-standardization exactly (see bolt/functions/mlp.py and
bolt/_utils.py; the data-mixture problem has no categorical inputs, so the feature
encoding is the identity).

Output is numerically identical to bolt.DMCurriculumMO(noise_std=None, negate=False):
a (N, 3) tensor of [IFEval, MATH-500, MBPP+] scores for a (N, 6) two-simplex input
[IF_1, Math_1, Code_1, IF_2, Math_2, Code_2].
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

HF_REPO = "chewwt/dm_qwen4b_emulator"


class DMCurriculumMOEmulator(nn.Module):

    def __init__(self, hf_repo: str = HF_REPO):
        super().__init__()
        model_path = hf_hub_download(hf_repo, "model.safetensors")
        config_path = hf_hub_download(hf_repo, "config.json")
        csv_path = hf_hub_download(hf_repo, "model_standardize.csv")

        with open(config_path, "r") as f:
            config = json.load(f)
        standardize = pd.read_csv(csv_path)
        self.y_mean = torch.tensor(standardize["y_mean"].to_numpy())
        self.y_std = torch.tensor(standardize["y_std"].to_numpy())

        self.input_dim = config["input_dim"]
        hidden_dim = config["hidden_dim"]
        output_dim = config["output_dim"]
        # Same architecture (and state-dict keys) as bolt's FeatureNet.
        self.mlp = nn.Sequential(
            nn.Linear(int(self.input_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.load_state_dict(load_file(model_path))
        self.eval()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the noise-free emulator on a (N, 6) two-simplex input."""
        self._validate_simplex_product(X)
        model_dtype = next(self.mlp.parameters()).dtype
        with torch.no_grad():
            pred = self.mlp(X.to(dtype=model_dtype)).to(dtype=X.dtype)
        return pred * self.y_std.to(X.dtype) + self.y_mean.to(X.dtype)

    @staticmethod
    def _validate_simplex_product(X: torch.Tensor, eps: float = 1e-5) -> None:
        if ((X[:, :3].sum(dim=1) - 1).abs() > eps).any() or (
                (X[:, 3:].sum(dim=1) - 1).abs() > eps).any():
            raise ValueError("first 3 and last 3 parameters need be from a simplex (sum to 1)")
