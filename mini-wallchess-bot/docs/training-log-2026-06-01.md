# Training Log — 2026-06-01

Goal: improve the net bot using AlphaZero-style MCTS + self-play until it can beat the human player.

---

## Baseline

Pre-session state:
- Deployed net: 256-hidden MLP, 300-dim features, ~195K params
- Performance: ~50% vs depth-2 heuristic (weak)
- Key prior fix (from previous session): `w_wall=100` in eval.rs, path features added (BFS dist + 4 progress flags), `BOT_DEPTH=3`

---

## Step 1 — Game analysis (replay 1: human vs heuristic)

Analyzed `wallchess-bot-2026-06-01-v9z1ou.jsonl`.

**Result:** human (south) beat heuristic (north), 59 plies.

**Heuristic weaknesses:**
- Built left-center funnel (6 walls at r:3-7, c:3-5,v) but left right side (c:6+) open
- Depth-3 horizon: can't see 5+ ply escape route
- Burned all 10 walls; no reserve after ply 39

**Human weaknesses:**
- Ply 6: wall r:1,c:5,h — useless (behind own start row), win_prob 56→44
- Ply 14: wall r:1,c:3,h — same problem
- Walked into north's funnel (ply 16), lost 12 plies escaping
- South found right-flank escape (c:8) and won

---

## Step 2 — Architecture audit

Identified MLP (256-hidden, 2 layers) with 300-dim flat features. Performance ceiling from:
1. Weak training data (d2 expert data dominated 55% of 54K samples)
2. No best-checkpoint saving (overfitting)
3. No LR schedule

---

## Step 3 — Training pipeline fixes

**`trainer/train.py`:**
- Added cosine annealing LR (`T_max=epochs, eta_min=1e-5`)
- Added best-checkpoint saving (track `val_policy + value_weight * val_value`)

**`core/src/bin/search_data.rs`:**
- Full-coverage policy targets: included ALL legal moves in pi target
  - Top-24 d3 candidates: scored with softmax(score/policy_scale)
  - All other legal moves: floor weight = exp(-4) ≈ 0.018 each
  - Explicit negative signal for non-productive walls (fixes untrained action space)

**`core/src/bin/selfplay_data.rs`:**
- Same floor-coverage fix for MCTS visit targets
  - `floor_visits=0.1` per unvisited legal move
  - Prevents random probability mass on non-visited actions

**`core/src/bin/net_arena.rs`:**
- Added `no_guard` flag (7th positional arg = `1`) for raw net evaluation

---

## Step 4 — Data generation

### sd-full (full-coverage search_data, d3)
6 shards × 10K = **60K samples**
- `search_data 10000 3 /tmp/sd-full-N.jsonl 12 350 80 24`
- Random openings 0-12 plies, top-24 d3 candidates + floor for all others
- Policy: top move ~58% prob, floor moves ~0.025%

### sd-deep (deeper mid-game positions)
4 shards × 10K = **40K samples**
- `search_data 10000 3 /tmp/sd-deep-N.jsonl 30 350 80 24`
- `max_random_plies=30` vs default 12 — samples mid/endgame positions
- Higher value stdev (0.387 vs 0.241) — more decisive positions

### sp-iter1 (heuristic MCTS selfplay, 40 sims)
4 shards × ~49K = **196K samples**
- `selfplay_data 2000 40 /tmp/sp-iter1-N.jsonl`
- `HEURISTIC_DEPTH=1, HEURISTIC_VALUE_SCALE=400` (default)
- Soft MCTS policy targets + floor for unvisited actions

### net-sp (net-guided selfplay iter2)
4 shards × ~7K = **28K samples**
- `selfplay_data 300 200 /tmp/net-sp2-N.jsonl /tmp/wallnet-v6.safetensors`
- v6 net, 200 sims per move — higher quality than heuristic MCTS

---

## Step 5 — Training iterations

All models: hidden=512, epochs=60-100, batch=512-1024, lr=2e-3 cosine, value_weight=0.5

| Net | Key data | Best val_policy | Arena vs d3 (200 sims) |
|-----|----------|-----------------|------------------------|
| iter0 | expert-d2 + misc (54K) | 1.22 | 7.5% |
| iter1 | +196K flat selfplay (250K) | 1.78 | 13.3% |
| v2 | 26K sd-iter1×5 + selfplay (493K eff) | 2.19 | 19.5% |
| v5 | 60K sd-full×5, no selfplay | 2.35 | 5.5% ← value overfit |
| v6 | sd-full×4 + sd-deep×3 + selfplay (724K eff) | 2.19 | ~19% |
| v7 | v6 + net-sp2×3 | 2.24 | 11% ← net-sp hurt |
| v9 | sd-full×4 + sd-deep-partial×6 + selfplay | 2.19 | 18-19% |
| v10 | sd-full×4 + sd-deep-full×6 + selfplay | 2.11 | 14.5% ← deep-only overfit |

**Key lessons:**
- Removing flat selfplay → value head overfit to random positions (v5: 5.5%)
- Net self-play with weak net → mode collapse (v7: 11%)
- Deep-position data alone → early-game policy degrades (v10: 14.5%)
- **Best recipe**: sd-full×4 + search-d3×3 + selfplay-h×2 + flat-selfplay×1

---

## Step 6 — Sim count tuning

| Sims | North win% | South win% | Notes |
|------|-----------|-----------|-------|
| 200 | 15% | 23% | South-biased |
| 400 | 15% | 23% | similar |
| 600 | 20% | 21% | **balanced** ✓ |

**Decision:** `NET_SIMS = 600` in `webui/src/game/api.ts`

More sims help North more because wall blocking requires deeper tree search.

---

## Step 7 — Deploy v6

- Weights: `/tmp/wallnet-v6.safetensors` → `webui/public/wallnet.safetensors` (2.0MB, 512-hidden)
- WASM net rebuilt: `wasm-pack build wasm --target web --release --features net`
- Files copied to `webui/src/wasm-net/`
- NET_SIMS: 200 → **600**
- UI: "MCTS Net" toggle in vs-Bot mode

---

## Step 8 — Game analysis (replay 2: human vs net bot)

Analyzed `wallchess-bot-2026-06-01-vn3ppt.jsonl`.

**Result:** human (south) beat net (north), 59 plies.

**Net behavior:**
- Discovered **cage strategy** (novel — not in heuristic): placed h-walls across row 2 to trap south near start row
- South stuck on row 1 lateral movement for ~12 plies (r1,c2→r1,c7) — strategy working
- BUT: placed r2,c7,h at ply 21, AFTER south reached r2,c7 at ply 20 (one ply too late)
- Spent 4 walls in first 9 plies → −4 wall deficit → unrecoverable endgame

**Root cause:** MLP has no spatial sense. Wall at (r2,c5) and gap at (r2,c7) are unrelated inputs — net can't detect incomplete cage.

**Comparison vs heuristic:**
- Heuristic: reactive BFS walls, boring but consistent
- Net: creative cage (better idea) but bad execution timing
- User finds d3 heuristic **harder** to beat than net

---

## Step 9 — CNN encoder implementation

Research note: `docs/cnn-encoder-research.md`

**`trainer/model.py` — WallNetCNN:**
```
Input: same 300-dim flat features (reshaped internally)
Board: 4-channel 9×9 (my_pawn, opp_pawn, h_walls, v_walls)
CNN:   Conv2d(4→32,3×3) → Conv2d(32→32,3×3) × 2 → Conv2d(32→16,1×1) → flatten
Scalars: [walls_left×2, race_margin, 4 progress flags] → FC(7→32)
Head:  concat → FC(1328→256) → policy(209) + value(1)
Params: 414K
```

**`core/src/net.rs` — auto-detect:**
- Checks `conv1.weight` key → CNN path
- Else → MLP path
- Same public `NetEvaluator` API, WASM unchanged

**`trainer/train.py`:**
- `--arch mlp|cnn` flag
- `--cnn-channels N` flag
- `--device auto|cpu|mps|cuda` flag (MPS auto-selected on macOS)

---

## Step 10 — CNN evaluation

| Model | val_policy | Arena @200sim | Time/move |
|-------|-----------|---------------|-----------|
| MLP v6 | 2.22 | 19.5% | 26ms |
| CNN v1/v2 | **2.17** | 16.7% | 165ms |

**Finding:** CNN fits training data better (2.17 < 2.22) but plays worse because:
1. BFS distances already explicit in flat features → CNN can't improve on BFS
2. 6x slower on CPU → WASM deployment impractical
3. MLP generalizes better to actual game positions despite worse val_policy

**Decision:** CNN stays in codebase as experimental architecture. MLP v6 remains deployed.

---

## Current state

**Deployed:** v6 MLP (512-hidden, 524K params) at 600 sims
**Win rate:** ~20% vs d3 heuristic, balanced North/South

**Ceiling reason:** Net imitates d3 via search_data. Can't exceed d3 by imitation.

**Next steps to break 20% ceiling:**
1. Expert d3 game data (`expert_data 3` — full game trajectories showing racing + walling balance)
2. 3-5 more self-play iterations (each ~1hr)
3. Alternatively: distillation (use CNN offline as stronger teacher → train MLP on CNN's data)

---

## Data inventory (in `/tmp`)

| File | Samples | Format | Quality |
|------|---------|--------|---------|
| sd-full-{1-6}.jsonl | 60K | d3 full-coverage search | ★★★★ |
| sd-deep-full-{1-4}.jsonl | 40K | d3 deep-position search | ★★★★ |
| sp-iter1-{1-4}.jsonl | 196K | 40-sim heuristic MCTS | ★★ |
| net-sp2-{1-4}.jsonl | 28K | 200-sim net MCTS (v6) | ★★★ |

Permanent: `data/search-d3-{1-9}.jsonl` (9K), `data/selfplay-h-{1-9}.jsonl` (15K), `data/expert-d2-{1-5}.jsonl` (30K)

---

## Sequel (2026-06-03) — net shelved, search side won

The "Next steps to break 20% ceiling" above were all pursued and all failed (see
`restnet-architecture-research.md` — 8 approaches, net capped ~42% vs d3). The net
is a proven dead end by imitation/self-play. Effort redirected to the **search
side**, which succeeded: `search_moves` BFS-path wall pruning made **alpha-beta d10
playable in WASM (<1s/move)**, monotonic D4–D10. Strongest deployed bot is now
`ab-d10`, not the net. See `../memory/project_search_optimization.md` and the
EPILOGUE in `restnet-architecture-research.md`.
