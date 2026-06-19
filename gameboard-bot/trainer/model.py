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


# ---------------------------------------------------------------------------
# ResTNet POC — interleaved Residual (R) + Transformer (T) blocks.
#
# Motivation (docs/restnet-architecture-research.md): MLP and pure CNN are
# spatially myopic — they miss non-local patterns like an incomplete cage
# (a wall + a distant gap that together trap the opponent). ResTNet, from
# "Bridging Local and Global Knowledge via Transformer in Board Games"
# (arXiv 2410.05347), interleaves conv residual blocks (local pattern
# extraction) with Transformer blocks (global attention across all 81 board
# squares). Paper rules baked in here:
#   - start with R blocks (T-first underperforms)
#   - interleave R/T, do not stack T at the end (RRT pattern, e.g. RRTRRT)
#   - one-to-one row-major token<->square mapping preserves board coords
#
# This is a POC: NOT WASM/Rust-deployable (net.rs has no Transformer path).
# Intended use is offline teacher for distillation into the deployed MLP if it
# wins on the cage probe / arena.
# ---------------------------------------------------------------------------

def _board_tensor(x: torch.Tensor) -> torch.Tensor:
    """Flat features → [B, 4, 9, 9] board (shared with WallNetCNN layout)."""
    B = x.shape[0]
    my_pawn = x[:, :81].reshape(B, 1, BOARD_SIZE, BOARD_SIZE)
    opp_pawn = x[:, 81:162].reshape(B, 1, BOARD_SIZE, BOARD_SIZE)
    h = torch.zeros(B, 1, BOARD_SIZE, BOARD_SIZE, device=x.device, dtype=x.dtype)
    h[:, 0, :8, :8] = x[:, 162:226].reshape(B, 8, 8)
    v = torch.zeros(B, 1, BOARD_SIZE, BOARD_SIZE, device=x.device, dtype=x.dtype)
    v[:, 0, :8, :8] = x[:, 226:290].reshape(B, 8, 8)
    return torch.cat([my_pawn, opp_pawn, h, v], dim=1)


def _scalar_vec(x: torch.Tensor) -> torch.Tensor:
    """Flat features → [B, 7] non-spatial scalars (shared with WallNetCNN)."""
    return torch.cat([x[:, 290:292], x[:, 295:300]], dim=1)


class ResBlock(nn.Module):
    """Standard 2-conv residual block, BN, 3×3, padding=1 (keeps 9×9)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class TBlock(nn.Module):
    """Pre-norm Transformer block over 81 board tokens (row-major).

    Converts [B, C, 9, 9] → [B, 81, C], applies MHA + FFN with a learned
    absolute positional embedding (one token per square), then reshapes back.
    Learned absolute pos-embedding stands in for the paper's relative encoding
    — adequate for a fixed 9×9 board and simpler for a POC.
    """

    def __init__(self, channels: int, heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.tokens = BOARD_SIZE * BOARD_SIZE  # 81
        self.pos = nn.Parameter(torch.zeros(1, self.tokens, channels))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Linear(channels * mlp_ratio, channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)          # [B, 81, C] row-major
        t = t + self.pos
        n = self.norm1(t)
        a, _ = self.attn(n, n, n, need_weights=False)
        t = t + a
        t = t + self.ff(self.norm2(t))
        return t.transpose(1, 2).reshape(B, C, H, W)


class WallNetResT(nn.Module):
    """Interleaved R/T tower over the 9×9 board + scalar head.

    Tensor-contract prefix `rest_` so the Rust loader (which only knows mlp/cnn)
    safely refuses this file rather than misreading it.
    """

    def __init__(self, channels: int = 64, heads: int = 4,
                 blocks: str = "RRTRRT", head_hidden: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(CNN_BOARD_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        if blocks[0] != "R":
            raise ValueError(f"block string must start with R (got {blocks!r})")
        tower = []
        for b in blocks:
            if b == "R":
                tower.append(ResBlock(channels))
            elif b == "T":
                tower.append(TBlock(channels, heads))
            else:
                raise ValueError(f"bad block {b!r} in {blocks!r}; use R/T only")
        self.tower = nn.ModuleList(tower)

        self.conv_out = nn.Conv2d(channels, channels // 2, 1)
        board_flat = (channels // 2) * BOARD_SIZE * BOARD_SIZE
        self.fc_scalar = nn.Linear(SCALAR_LEN, 32)
        self.fc_head = nn.Linear(board_flat + 32, head_hidden)
        self.policy = nn.Linear(head_hidden, ACTION_COUNT)
        self.value = nn.Linear(head_hidden, 1)

    def forward(self, x):
        h = self.stem(_board_tensor(x))
        for blk in self.tower:
            h = blk(h)
        h = F.relu(self.conv_out(h)).flatten(1)
        s = F.relu(self.fc_scalar(_scalar_vec(x)))
        combined = F.relu(self.fc_head(torch.cat([h, s], dim=1)))
        return self.policy(combined), torch.tanh(self.value(combined)).squeeze(-1)
