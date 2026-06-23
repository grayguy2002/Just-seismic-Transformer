"""Token ablation wrapper — zero specific condition token positions."""

from __future__ import annotations

import torch
import torch.nn as nn

from .condition_encoder import SeismicConditionEncoder


class AblationConditionEncoder(nn.Module):
    """Wrap a SeismicConditionEncoder, zeroing selected token indices."""

    def __init__(self, base_encoder: SeismicConditionEncoder, dropped_tokens: list[int]):
        super().__init__()
        self.encoder = base_encoder
        self.dropped = sorted(dropped_tokens)
        self.n_tokens = base_encoder.n_tokens

        # Rebuild group_token_indices excluding dropped tokens
        self.group_token_indices = {}
        for group, indices in base_encoder.group_token_indices.items():
            remaining = [i for i in indices if i not in self.dropped]
            if remaining:
                self.group_token_indices[group] = remaining

        self.token_names = base_encoder.token_names
        self.hidden_dim = base_encoder.hidden_dim
        self.encoder_version = base_encoder.encoder_version

    def forward(self, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = self.encoder(cond)
        for idx in self.dropped:
            tokens[:, idx, :] = 0.0
        return tokens
