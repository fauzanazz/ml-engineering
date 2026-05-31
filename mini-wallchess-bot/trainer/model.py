"""Value + policy net. A small MLP — the input is already a flat hand-built
feature vector, so a couple of dense layers is enough and keeps the wasm/candle
inference cheap.

The tensor names here (`l1`, `l2`, `policy`, `value`) are the contract with the
Rust candle loader (`core/src/net.rs`): same names, same shapes.
"""

import torch
import torch.nn as nn

from encoding import ACTION_COUNT, FEATURE_LEN

HIDDEN = 256


class WallNet(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.l1 = nn.Linear(FEATURE_LEN, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.policy = nn.Linear(hidden, ACTION_COUNT)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.relu(self.l1(x))
        h = torch.relu(self.l2(h))
        # policy: raw logits (loss applies log_softmax); value: tanh to [-1, 1]
        return self.policy(h), torch.tanh(self.value(h)).squeeze(-1)
