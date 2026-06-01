"""Value + policy nets for WallChess.

Two architectures:
  WallNet    — 2-layer MLP over the flat 300-dim feature vector. Fast, simple.
               Tensor contract: l1, l2, policy, value.
  WallNetCNN — CNN encoder + MLP head over the same flat features (reshaped
               internally). Captures spatial wall-cluster patterns that the MLP
               cannot represent (e.g. whether a cage has a gap).
               Tensor contract: conv1, conv2, conv3, conv_out, fc_scalar,
               fc_head, policy, value.

The Rust candle loader (core/src/net.rs) auto-detects which arch by checking
for conv1.weight in the safetensors file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import ACTION_COUNT, FEATURE_LEN

HIDDEN = 256

# CNN board encoding constants (must match core/src/net.rs board_tensor logic)
CNN_BOARD_CHANNELS = 4   # my_pawn, opp_pawn, h_walls, v_walls
BOARD_SIZE = 9
SCALAR_LEN = 7           # my_walls/10, opp_walls/10, race_margin/16, 4 progress flags


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


class WallNetCNN(nn.Module):
    """CNN encoder over the 9×9 board + scalar head for non-spatial features.

    Input: same flat 300-dim feature vector as WallNet — reshaped internally.
    Board channels (4, 9, 9):
      ch0: my pawn (one-hot, me-frame)
      ch1: opp pawn (one-hot, me-frame)
      ch2: h-walls (8×8 anchor bits placed in top-left of 9×9)
      ch3: v-walls (same)
    Scalar inputs (7):
      my_walls/10, opp_walls/10, race_margin/16, 4 progress flags

    Three conv layers (3×3, padding=1) keep spatial resolution at 9×9.
    A 1×1 conv compresses channels before flattening — avoids a huge linear.
    """

    def __init__(self, channels: int = 32, scalar_hidden: int = 32, head_hidden: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(CNN_BOARD_CHANNELS, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        # 1×1 conv: halve channels before flatten to keep the flat→linear layer small
        self.conv_out = nn.Conv2d(channels, channels // 2, 1)

        board_flat = (channels // 2) * BOARD_SIZE * BOARD_SIZE  # e.g. 16*81=1296
        self.fc_scalar = nn.Linear(SCALAR_LEN, scalar_hidden)
        self.fc_head = nn.Linear(board_flat + scalar_hidden, head_hidden)
        self.policy = nn.Linear(head_hidden, ACTION_COUNT)
        self.value = nn.Linear(head_hidden, 1)

    def _board_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape flat features → 4-channel 9×9 board tensor.

        Feature layout (from features.rs, 0-indexed):
          0..81   my pawn one-hot (9×9 me-frame, row-major)
          81..162 opp pawn one-hot
          162..226 h-walls 64 bits = 8×8 anchor grid, row-major
          226..290 v-walls 64 bits
        """
        B = x.shape[0]
        my_pawn = x[:, :81].reshape(B, 1, BOARD_SIZE, BOARD_SIZE)
        opp_pawn = x[:, 81:162].reshape(B, 1, BOARD_SIZE, BOARD_SIZE)
        # Wall bits: 8×8 anchor grid placed in top-left corner of 9×9 spatial grid
        h = torch.zeros(B, 1, BOARD_SIZE, BOARD_SIZE, device=x.device, dtype=x.dtype)
        h[:, 0, :8, :8] = x[:, 162:226].reshape(B, 8, 8)
        v = torch.zeros(B, 1, BOARD_SIZE, BOARD_SIZE, device=x.device, dtype=x.dtype)
        v[:, 0, :8, :8] = x[:, 226:290].reshape(B, 8, 8)
        return torch.cat([my_pawn, opp_pawn, h, v], dim=1)  # [B, 4, 9, 9]

    def _scalar_vec(self, x: torch.Tensor) -> torch.Tensor:
        """Extract non-spatial scalar features.

        Feature indices (from features.rs):
          290: my_walls_left / 10
          291: opp_walls_left / 10
          295: race_margin / 16  (opp_dist - my_dist)
          296..300: 4 progress flags
        """
        return torch.cat([x[:, 290:292], x[:, 295:300]], dim=1)  # [B, 7]

    def forward(self, x: torch.Tensor):
        board = self._board_tensor(x)
        h = F.relu(self.conv1(board))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv_out(h))
        h_flat = h.flatten(1)

        s = F.relu(self.fc_scalar(self._scalar_vec(x)))

        combined = F.relu(self.fc_head(torch.cat([h_flat, s], dim=1)))
        return self.policy(combined), torch.tanh(self.value(combined)).squeeze(-1)
