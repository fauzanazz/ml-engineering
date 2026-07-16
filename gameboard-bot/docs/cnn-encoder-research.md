# CNN Encoder Research Note

## Current MLP Architecture and Its Weaknesses

### Architecture (as of 2026-06-01)

```
Input: 462-dim flat vector (features.rs)
  - 81: my pawn one-hot (9×9 me-frame)
  - 81: opp pawn one-hot
  - 64: horizontal wall bits (8×8 anchor grid)
  - 64: vertical wall bits
  - 1: my walls_left / 10
  - 1: opp walls_left / 10
  - 1: bias constant / side-to-move plane source
  - 1: my BFS distance to goal / 16
  - 1: opp BFS distance / 16
  - 1: race margin / 16
  - 4: progress flags (can step toward goal in each direction?)
  - 81: my BFS distance-to-goal map / 16
  - 81: opponent BFS distance-to-goal map / 16
Hidden (MLP default): Linear(462→256) → ReLU → Linear(256→256) → ReLU
Heads: policy Linear(256→209), value Linear(256→1)+tanh
```

Arena result: **~20% win rate vs depth-3 alpha-beta (600 MCTS sims).**

### Observed Failure Mode: Incomplete Cage

Game analysis (2026-06-01 replay vs human, net plays North):

The net discovered a genuine cage strategy — it placed horizontal walls across row 2
(r2,c5,h at ply 1; r2,c3,h at ply 3) to trap South near the starting row. This is
tactically sound and more sophisticated than the heuristic's reactive BFS-based walls.

But the cage was incomplete: South escaped via column 9 because the net placed r2,c7,h
at ply 21, AFTER South had already reached r2,c7 at ply 20 — one ply too late.

The net also spent 4 walls in 9 plies, leaving a −4 wall deficit that made recovery
impossible.

**Root cause: the MLP has no spatial sense.**

The flat input treats wall at (r2,c5) and wall at (r7,c5) as unrelated inputs. The
net cannot "see" that three walls at (r2,c3), (r2,c5), (r2,c7) form a contiguous
barrier, or that a gap at (r2,c7) makes the whole structure exploitable.

A 2-layer MLP could in theory learn these combinations, but:
1. It needs to memorize every specific wall pattern (~128-bit wall state space).
2. There is no inductive bias toward "nearby walls interact."
3. The flat encoding gives the net no gradient signal about spatial contiguity.

### What the MLP CAN learn (already handles well)

- BFS distance advantage (explicit features, no learning needed)
- Wall count parity (direct scalar features)
- Basic racing vs blocking tradeoff
- Immediate tactical shots (1-2 ply combinations)

### What the MLP CANNOT learn efficiently

- Whether a wall cluster has a gap (spatial contiguity)
- Diagonal/lateral escape routes
- The difference between a tight cage and a leaky one
- How wall positions near row 2 interact vs walls near row 7

---

## Why CNN Encoder

A small CNN treats the board as a 9×9 image. Each cell "sees" its neighbors via the
convolution kernel. The network learns:

- Which wall configurations form complete barriers (local kernel captures adjacency)
- Where escape routes exist (pooling / deeper layers aggregate spatial context)
- How my pawn position relates to nearby wall patterns (same spatial frame)

### Proposed architecture

```
Board representation: 9×9 spatial grid with C channels
  ch0: my pawn position (9×9 float)
  ch1: opp pawn position (9×9 float)
  ch2: horizontal walls — h_wall[r,c] = 1.0 if wall at anchor (r,c) (8×8, padded to 9×9)
  ch3: vertical walls   — v_wall[r,c] = 1.0
  ch4: my BFS distance-to-goal map
  ch5: opp BFS distance-to-goal map
  ch6: side-to-move / bias constant plane
CNN head: 
  Conv2d(C→32, 3×3, pad=1) → BN → ReLU  # local wall patterns
  Conv2d(32→64, 3×3, pad=1) → BN → ReLU # intermediate features
  Conv2d(64→64, 3×3, pad=1) → BN → ReLU # deeper spatial context
  → flatten → Linear(64*9*9 → 256)

Scalar head (features that aren't spatial):
  [my_walls/10, opp_walls/10, race_margin/16, 4×progress_flags] → Linear(7→32)

Combined:
  concat(CNN_out, scalar_out) → Linear(288→256) → ReLU
  → policy Linear(256→209)
  → value  Linear(256→1) + tanh

Total params: ~400-500K (fits WASM budget)
```

### Expected improvements

- Cage completeness detection: CNN kernel over row 2 channels will fire on
  "wall here, gap there" patterns
- Transfer across board positions: same 3×3 kernel applies everywhere on the 9×9
  board, generalizing from row-2 cage patterns to row-5 cage patterns
- Literature: CNN-based Quoridor agents reported ~40-50% vs depth-3 alpha-beta
  (vs current 20% MLP baseline)

### References

- AlphaViT (2024): Vision Transformer for board games, outperforms CNN on multiple sizes
  https://arxiv.org/pdf/2408.13871
- AlphaVile (CNN+Transformer): +180 Elo in chess vs AlphaZero baseline
- Quoridor-AI survey: CNN and MCTS approaches, 40-50% vs shallow alpha-beta
  https://www.researchgate.net/publication/363456787_Quoridor-AI
- Three-head network (Q-value head): +5-10% MCTS efficiency
  https://openreview.net/forum?id=BJxvH1BtDS

---

## Implementation Plan

1. `core/src/features.rs` — encoder now emits 462 features: legacy 300 + two BFS maps
2. `trainer/model.py` — `WallNetCNN` consumes a 7×9×9 spatial tensor
3. `trainer/train.py` — `--arch cnn --policy-loss-weight 0` trains value-only first
4. `core/src/net.rs` — Candle loader runs CNN as both `PolicyValue` and alpha-beta `Evaluator`
5. `trainer/autoloop.py` — resumable data → train → arena-gate → promote loop
6. Arena eval: target >30% vs d3 at 600 sims

Key invariant: me-frame mirroring still required (North board flipped 180°
before encoding, same as current features.rs logic).
