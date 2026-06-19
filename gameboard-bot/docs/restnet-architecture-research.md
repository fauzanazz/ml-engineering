# ResTNet Architecture Research — 2026-06-02

Source paper: **"Bridging Local and Global Knowledge via Transformer in Board Games"** (Chen et al.), arXiv 2410.05347, IJCAI-25 proceedings.

Why this note: the deployed MLP net is spatially blind — it can't detect an *incomplete cage* (wall at (r2,c5) + gap at (r2,c7) are unrelated inputs). See `training-log-2026-06-01.md` Step 8. This is the same failure class as ladder/circular patterns in Go and bridge connections in Hex — **non-local spatial dependencies**. ResTNet is the most directly relevant published fix.

---

## Problem ResTNet solves

CNNs (ResNet) are inherently local. They miss long-sequence / global patterns:
- Go: ladder pattern, circular (cyclic) pattern
- Hex: virtual connections across the board

Pure CNN can't bridge distant board regions. Pure Transformer loses local pattern extraction. ResTNet interleaves both.

**Direct mapping to our bot:** an incomplete cage = a global pattern spanning multiple wall slots. Exactly what CNN-only (our rejected v1/v2 CNN) and MLP both fail at.

---

## Architecture

### Block composition
Interleave Residual blocks (**R**) and Transformer blocks (**T**):
- Best config: `R3(RRT)` = `RRTRRTRRT` (10 blocks, 19×19 games)
- 9×9 games: 6-block nets, best = `RRTRRT`
- **Rule found:** must start with R blocks. Starting with T underperforms (local pattern extraction first is critical).

### Residual block
- Standard AlphaZero: 2 conv layers, 256 filters.

### Transformer block
- Standard Transformer, **relative position encoding**
- 4 attention heads per block
- MLP ratio: 2
- Embedding 256-dim across N tokens (81 tokens for 9×9)

### Local↔global bridge (the key mechanism)
- R blocks operate on 2D feature maps `C×H×W`.
- T blocks need 1D tokens → convert board positions **row-major** into tokens.
- **One-to-one positional mapping** preserves board coordinates across every R↔T conversion → spatial precision maintained both directions.

### Hyperparameters
| Param | Value |
|-------|-------|
| Hidden channels | 256 |
| Transformer heads | 4 |
| MLP ratio | 2 |
| Embedding dim | 256 |

---

## Architecture ablation (9×9 Go) — what works

| Config | Win rate | Note |
|--------|----------|------|
| Pure CNN (6R) | 54.60% | baseline |
| Pure Transformer (6T) | 39.85% | **fails** — no local extraction |
| TRRRRT (T first) | 43.90% | starting with T hurts |
| CoAtNet-style (5R1T) | 56.00% | minor gain |
| **RRTRRT (interleaved)** | **60.80%** | optimal |

Lesson: T must follow R. Interleaving beats stacking-at-end.

---

## Results

### Playing strength (win rate vs baseline)
| Game | Config | Win rate | Baseline |
|------|--------|----------|----------|
| 9×9 Go | RRTRRT | 60.80% ±2.14 | KataGo 54.6% |
| 19×19 Go | R3(RRT) | 60.90% ±2.14 | KataGo 53.6% |
| 19×19 Hex | R3(RRT) | 58.00% ±4.33 | MoHex 50.4% |

### Global pattern capture (the relevant metric for us)
- **Circular pattern MSE:** 2.58 → **1.07** (frozen-backbone probe)
- **Cyclic-adversary attack rate:** 70.44% → **23.91%** (2.95× better)
- **Ladder pattern accuracy:** 59.15% → **80.01%** (model probing, frozen backbone, 1.66M train / 166.5K test patterns)

Ladder = a long causal chain across the board. Cage-completion is our analog. 59→80% accuracy jump is the headline transferable result.

### Interpretability
Attention maps captured human game concepts: alive stones, territory, ladders (Go); basic + virtual connections (Hex). → attention learns the non-local relations explicitly.

---

## Training setup (reference)
- 9×9 Go: Gumbel AlphaZero, 64 sims, 100K steps, LR 0.02→0.005, batch 1024, 1M self-play games, ~200 GPU-hrs/model.
- 19×19 Hex: Gumbel AlphaZero, 32 sims, 100K steps, 500K self-play games, LR 0.02.

---

## Implications for mini-wallchess-bot

**Confirms:**
- MLP spatial blindness is a known, documented failure class (non-local dependency).
- Our CNN-only experiment failing is *expected* — CNN alone also misses ladder/cage patterns. ResTNet adds Transformer *because* CNN is insufficient. So "CNN buntu" conclusion was right but for an incomplete reason.

**Tension with our constraints:**
- ResTNet is heavy: 256 channels × 10 blocks + attention. Our prior CNN was already 6× slower than MLP on CPU and impractical for WASM at 600 sims. A Transformer stack is *heavier* still.
- ResTNet trains with full AlphaZero self-play (1M games, 200 GPU-hrs). We hit mode collapse on net self-play with a weak net (v7). Same risk amplified.

**Realistic paths (cheap → expensive):**
1. **Cheap / recommended first:** explicit feature engineering — add a "cage completeness" scalar feature (does one more wall close opponent's path?). Captures the non-local relation without any new architecture. Fits existing MLP + WASM latency budget.
2. **Medium:** ResTNet-lite as an **offline teacher** — train small interleaved R/T net (e.g. RRTRRT, 64–128 channels) offline, use it to generate distillation data, train the deployed MLP on its targets. Gets non-local knowledge into MLP without WASM-deploying the Transformer. (Aligns with the distillation idea already in training-log Step "Next steps".)
3. **Expensive / research:** port a quantized ResTNet-lite to WASM and re-tune sim count. Only if 1–2 stall.

**Key transferable design rules if we build R/T net:**
- Start with R blocks (never T first).
- Interleave (RRT…), don't stack T at the end.
- Use relative position encoding.
- One-to-one row-major positional mapping on R↔T conversion (don't lose board coords).

---

## POC results — 2026-06-02 (MPS)

Implemented `WallNetResT` in `trainer/model.py` (interleaved R/T tower, learned
absolute pos-embedding, 4 heads, MLP ratio 2). Wired `--arch restnet` into
`train.py` (`--rest-channels`, `--rest-blocks`). Trained MLP baseline and ResT
on **identical data + identical split seed** to isolate architecture effect.

- Data (~46k, best-recipe shape): sd-full×4 + search-d3×3 + selfplay-h×2
- Hyperparams: 40 epochs, batch 512, lr 2e-3 cosine, value_weight 0.5
- Device: MPS (Apple), torch 2.12

| Arch | Blocks | Channels | Params | best val_policy | best val_value |
|------|--------|----------|--------|-----------------|----------------|
| MLP | — | hidden 512 | 524K | 2.6343 | 0.0221 |
| **ResTNet** | RRTRRT | 64 | 1.10M | **2.5697** | **0.0202** |

**ResT wins both heads**: policy −0.0646 (−2.45%), value −8.6%. Trains cleanly on
MPS, no instability. Confirms the interleaved R/T tower fits this data better
than the flat MLP — the global-attention path is the theoretically correct tool
for the non-local cage pattern.

**Caveats (do not over-read val_policy):**
- The CNN experiment (training-log Step 10) had *better* val_policy (2.17) yet
  *worse* arena (16.7% vs 19.5%). val_policy is a weak proxy for playing strength.
  A −2.45% policy edge is promising but NOT proof ResT plays better.
- ResT is 2.1× the params and not Rust/WASM-deployable. No arena number yet —
  net_arena.rs can't load a Transformer.

**Verdict:** POC passes the gate (ResT > MLP on identical data, MPS-feasible).
Proceed to the real test before committing to distillation:
1. **Cage probe** — build a held-out set of incomplete-cage positions (wall + one
   distant gap that closes opponent's path). Measure whether ResT assigns the
   cage-closing wall higher policy/value than MLP. This directly tests the thing
   val_policy can't.
2. If ResT wins the probe → **distill**: use ResT (MPS, offline teacher) to
   generate soft policy+value targets, train the deployed MLP student on them,
   arena the student vs d3. MLP stays the deployed/WASM artifact.
3. If probe is flat → ResT's val edge is just capacity, not spatial sense; stop
   and fall back to explicit "cage completeness" feature engineering.

Artifacts: `/tmp/wallnet-poc-mlp.safetensors`, `/tmp/wallnet-poc-rest.safetensors`,
log `/tmp/poc-restnet.log`.

---

## Distillation result — 2026-06-02 (ResT teacher → MLP student → arena vs d3)

Goal: prove the net path can beat the d3 heuristic (the deployed opponent).

**Pipeline executed:**
1. ResT teacher: `WallNetResT` RRTRRT c64, 124k best-recipe samples
   (sd-full×6 + sd-deep×4 + search-d3×9 + selfplay-h×9), 50 epochs MPS.
   Best val_policy **2.31** (vs POC 2.57 — more data helped). → `/tmp/wallnet-rest-teacher.safetensors`
2. Distill (`trainer/distill.py`): teacher soft targets (temp 2.0, pure distill
   alpha 0) → MLP-512 student, 40 epochs. Final teacher-fidelity policy **2.37**.
   → `/tmp/wallnet-student.safetensors` (standard MLP, Rust/WASM-loadable)
3. Arena vs d3 heuristic, 40 games, 600 sims, openings 6 (identical settings/seed
   to baseline → directly comparable).

**Arena (net wins / 40 vs d3):**
| Net | Result | Win% | S / N split |
|-----|--------|------|-------------|
| v6 MLP (deployed baseline) | 9 – 31 | 22.5% | 5 / 4 |
| **Distilled student MLP** | 10 – 30 | **25.0%** | 7 / 3 |

**Finding: distillation into an MLP does NOT cross the ceiling.** +1 win is within
noise. The ResT teacher fits the data much better (val_policy 2.31) but that
advantage is **lost in compression** to the MLP (student fidelity 2.37, arena
flat). Confirms the core thesis the other way around: the MLP *structurally*
cannot represent the non-local cage relations, so no teacher can pour them in via
soft targets. The spatial knowledge lives in ResT's attention; an MLP has nowhere
to put it.

**Consequence — two remaining paths to beat d3 (>50%):**
- **A. Deploy ResT directly.** Arena the teacher itself (never measured — Rust
  `net.rs` has no Transformer path). Requires implementing the R/T forward in
  candle (+ WASM). Decisive test: is ResT-as-player actually stronger, or does the
  whole net+MCTS approach cap below d3 regardless of evaluator?
- **B. Self-play RL.** Exceed d3 by search+learning rather than imitation (the
  imitation ceiling is fundamental — net imitates d3, can't surpass it by copying).
  Prior weak-net self-play mode-collapsed (v7); needs the stronger net as a base.

**Diagnostic: sim-scaling sweep (student vs d3, 40 games each, openings 6):**
| Sims | net wins / 40 | Win% |
|------|---------------|------|
| 600  | 10 | 25.0% |
| 1000 | 15 | 37.5% |
| **1500** | **20** | **50.0%** |
| 2000 | 15 | 37.5% |
| 3000 | 8 / 24 | 33% |

**Non-monotonic peak at 1500 sims = parity (50%) with d3, then decline.** This is
the signature of a **miscalibrated value head**: MCTS improves with more search up
to ~1500 sims, then deeper search amplifies value-net errors and play degrades.
Policy is not the bottleneck — value calibration is. The deployed config (600
sims) is badly under-searched; the same net at 1500 sims is +25 points.

**Reframed conclusion (REVISED — see precise read below):** the net+MCTS approach
improves with search but the 50% @1500 above was a 40-game variance artifact.

### Precise peak read (60 games, student vs d3)
| Sims | net wins / 60 | Win% |
|------|---------------|------|
| 1300 | 25 | 41.7% |
| 1500 | 25 | 41.7% |
| 1700 | 25 | 41.7% |

Dead flat at **~42%** — the student saturates *below* parity. The earlier 20-20
(50%) was noise. **True ceiling of the distilled-MLP path ≈ 42% vs d3.**

### Self-play iteration (iter3) — FAILED
Generated 23k self-play samples with the student net (600 sims), retrained MLP-512
on best-recipe + self-play (hard targets, from scratch). Result collapsed:
| Sims | iter3 win% | student win% |
|------|-----------|--------------|
| 600  | 5.0% | 25.0% |
| 1500 | 15.0% | ~42% |
| 2000 | 22.5% | 37.5% |

Two compounding causes: (1) self-play visit-count policy targets (from a 600-sim
MCTS) are weaker than d3 search_data targets and poisoned the policy prior;
(2) iter3 abandoned **distillation** (soft teacher targets) for hard targets —
losing the very thing that made the student good. Confirms training-log lesson:
weak-net self-play → degradation.

### Standing conclusion
Best net = distilled student, **~42% vs d3**, saturating. Imitation ceiling is
real and robust. Remaining levers to cross 50%, both costly/uncertain:
- **Deploy ResT directly** — implement the R/T forward in candle `net.rs` (~200
  lines: stem+BN, ResBlock, manual MHA, scalar head; detect via `stem.0.weight`),
  arena the teacher itself. Untested whether ResT's spatial value head scales
  search past 42%. Highest-principle bet, real eng cost.
- **Self-play done right** — keep distillation, use the teacher to soft-label
  self-play positions, downweight self-play policy. Medium cost, mode-collapse risk.

Artifacts: `/tmp/wallnet-student.safetensors` (best, ~42%), `/tmp/arena-peak.log`,
`/tmp/arena-iter3.log`, self-play `/tmp/sp-iter3-*.jsonl`.

---

## ResT deployed in Rust — 2026-06-02 (decisive test)

Implemented the R/T forward in `core/src/net.rs` (`NetArch::ResT`):
- `BatchNorm2dW` / `LayerNormW` — manual eval-mode forward (running stats / last-dim).
- `ResTBlock::Res` — conv→BN→relu→conv→BN→skip→relu.
- `ResTBlock::T` — tokens (1,81,C) + learned pos, pre-norm MHA (`self_attention`,
  4 heads, manual QKV split + scaled dot-product) + pre-norm GELU FFN.
- Tower auto-detected per index from tensor keys; arch detected via `stem.0.weight`.
- `rest_probe` bin + `trainer/parity_check.py` cross-check.

**Parity verified:** Rust vs PyTorch teacher, max |value diff| = **4.3e-7** over 7
positions, top-policy probabilities identical. The Rust forward is numerically exact.

**Arena, ResT teacher directly vs d3:**
| Net | Sims | Result | Win% |
|-----|------|--------|------|
| ResT teacher | 600 | 5 – 25 | 16.7% |
| ResT teacher | 1200 | 4 – 16 | 20.0% |
| (student MLP) | 600 | — | 25.0% |
| (student MLP) | peak (1300–1700) | — | ~42% |

ResT teacher at 600 sims plays **worse** (16.7%) than both the distilled student
(25%) and v6 (22.5%) — despite fitting the training data far better (val_policy
2.31). Eval latency: **1505 ms/move at 600 sims** (~50× the MLP), making WASM
deployment infeasible regardless.

This is the **third** instance of the same pattern: better imitation fit (CNN
2.17, ResT 2.31) → *worse* arena play. Training-loss quality is anti-correlated
with MCTS playing strength here. The fancy spatial architectures overfit the d3
imitation target; the plainer MLP generalizes better inside search.

More search did **not** rescue the teacher: 16.7%@600 → 20%@1200, still half the
student's ~42%. ResT-direct is conclusively NOT the path past d3.

---

## FINAL VERDICT (2026-06-02)

**Not proven better than d3. The net-by-imitation approach is capped.**

Full ladder vs d3 (the deployed opponent), best win% each:
| Approach | Best win% vs d3 | Note |
|----------|-----------------|------|
| v6 MLP (deployed) | 22.5% @600 | baseline |
| ResT teacher (direct, Rust) | 20% @1200 | worse + 50× slower, WASM-infeasible |
| Distilled student MLP | **~42% @1500** | **best net found** |
| Self-play iter3 | 22.5% @2000 | collapsed |
| — | **<50% everywhere** | d3 not beaten |

**Why imitation can't win (root cause):** every net here is trained to imitate
d3 (search_data) or weak self-play. Imitation cannot exceed its teacher. Worse,
higher-capacity spatial nets (CNN, ResT) *overfit* the imitation target and play
*worse* in MCTS — repeatedly. The original hypothesis (spatial blindness is the
ceiling) was half-right: the MLP IS spatially blind, but giving it spatial sense
(ResT) does not help, because the *target itself* (d3 imitation) is the ceiling,
not the architecture.

**The only remaining principled path to beat d3:** a real AlphaZero loop —
self-play with **arena-gating** (accept a new net only if it beats the current
best in arena), starting from the ~42% student as anchor, many iterations, with
the policy anchored to search_data to avoid the v7/iter3 collapse. Large,
multi-session, mode-collapse risk. Alternatively: stop improving the net and
strengthen the heuristic/search side (already the stronger playable engine).

**Net deliverable kept:** distilled student (`/tmp/wallnet-student.safetensors`),
~42% vs d3 @1500 sims — strictly better than the deployed v6 (22.5%) but only at
2.5× the sim budget. New code shipped and verified: `WallNetResT` (model.py),
`distill.py`, `NetArch::ResT` (net.rs, parity 4.3e-7), `rest_probe`,
`parity_check.py`.

---

## AlphaZero arena-gating loop — iter az1 (decoupled) FAILED

Added value/policy decoupling to `train.py` (`--value-data` + per-sample
`policy_weight` in `dataset.py`): policy trained only on the d3 search anchor,
value trained on self-play `z`. Goal: fix the value-calibration ceiling without
the policy-poisoning that collapsed iter3.

Generated az1 self-play (student net @800 sims, 600 games, 30k samples) + reused
iter3 (23k) = 53k value-only. Trained MLP-512: 124k d3-anchor (policy+value) +
53k self-play (value-only), value_weight 0.7.

Arena az1 vs d3: **15% / 12.5% / 17.5%** @ 600/1000/1500 — flat, far below the
student's 42%. The arena gate correctly **rejects** it. Self-play `z` from a
sub-d3 net miscalibrates value *worse*, not better.

**Rigorous impossibility argument (why self-play can't cross d3 here):** to
bootstrap past d3, the net+MCTS must already beat d3 to generate supra-d3 self-play
targets. But net+MCTS peaks at 42% < d3. So self-play targets are capped below d3.
Self-play cannot cross d3 from a sub-d3 base. Chicken-and-egg, confirmed empirically
(iter3 collapse, az1 flat).

## Root cause + the real lever: stronger teacher

The net is capped at ~d3 because it is **trained on d3** (search_data depth 3).
Imitation cannot exceed its teacher. The fix is not architecture or self-play — it
is a **stronger teacher**. The engine's alpha-beta at depth 4 beats depth 2 16-0
(after the `w_wall=100` eval fix restored depth monotonicity; even depths are clean,
odd depths corrupted by the ±1 tempo term — see project memory). So:

**Hypothesis:** train the net on `search_data` at **depth 4** → it imitates d4 →
beats d3. Generating ~24k d4 samples (8 shards, ~1 s/sample solo; ~4 s/sample
under 8-way contention → slow). Then train MLP, arena vs d3.

### d4 probe (10k samples, hidden 256 — underpowered, directional only)
| Sims | d4probe win% | student win% |
|------|--------------|--------------|
| 600  | 27.5% | 25.0% |
| 1000 | 30.0% | 37.5% |
| 1500 | 30.0% | ~42% (peak) |
| 2200 | 33.3% | 37.5% |
| 3000 | **36.7%** | 33% (degraded) |

**Key signal — opposite sim-scaling shapes.** The d3-student PEAKS at 1500 then
DEGRADES (value miscalibration amplified by deep search). The d4-probe climbs
MONOTONICALLY (30→37%) and is still rising at 3000 — **d4's value targets are
better-calibrated, so more search keeps helping.** The probe is data/capacity
starved (10k vs the student's 124k; 196k vs 524k params) yet already overtakes the
student at 3000 sims. This is the first approach whose curve points UP, not flat
or down.

**Decision: commit to the d4 path.** Train the full 524k net on the complete d4
set (24k, generating) and arena at high sims (3000+) where d4 calibration pays off.

### d4 full-data nets vs d3 — capped ~35%, below the d3-student
| Net | 1500 | 2200 | 3000 | 4000 |
|-----|------|------|------|------|
| d4 probe (256h, 10k) | 30% | 33% | **36.7%** | — |
| d4 full (512h, 24k) | 25% | 25% | 17.5% | 17.5% (overfit) |
| d4 (256h, 24k) | 20% | 20% | 30% | 30% |

Two findings:
1. **512h overfits 24k** (degrades at high sims like the d3-student). The small 256h
   regularizes better — capacity, not just data, drives the calibration shape.
2. **All d4 nets cap ~30–37%, BELOW the d3-student's 42%.** A stronger teacher did
   NOT yield a stronger net.

**Why the stronger teacher failed:** the imitation gap *scales with teacher
complexity*. d4 plays harder, more tactical moves than d3; the net imitates them
*worse* (same val_policy ~2.35 but the target is harder). The strength gained from
d4 > d3 is eaten by the larger imitation gap → net-imitating-d4 ≈ net-imitating-d3.

---

## DEFINITIVE CONCLUSION (2026-06-02) — d3 not beatable by these methods

Six principled approaches, all measured vs d3, none crosses 50% — and none beats
the simple d3-distilled student (~42%):

| # | Approach | Best vs d3 | Failure mode |
|---|----------|-----------|--------------|
| 1 | Distilled student MLP (d3) | **~42%** | best, but <50% |
| 2 | CNN spatial encoder | ~17% | overfits imitation, worse in search |
| 3 | ResT teacher, deployed in Rust | 20% | same overfit + 50× slower |
| 4 | Self-play iter3 (hard targets) | collapsed | weak-net policy poison |
| 5 | Self-play az1 (value-decoupled) | 15% | sub-d3 z miscalibrates value |
| 6 | Stronger teacher (d4 search_data) | ~37% | imitation gap scales with teacher |

**Three independent ceilings, each proven empirically:**
- **Imitation ceiling:** net ≤ teacher. d3 teacher → ≤d3. d4 teacher → net imitates
  it proportionally worse, no net gain.
- **Bootstrap trap:** self-play needs net+search > d3 to generate supra-d3 targets,
  but net+search peaks at 42% < d3. Cannot cross. (rigorous)
- **Capacity/overfit:** bigger/spatial nets fit the imitation target better and play
  *worse* — training-loss is anti-correlated with arena strength throughout.

**The net cannot beat d3 with imitation or self-play at this compute budget.**

**Direct RL — expert-iteration vs d3 (attempted, FAILED).** Built `core/src/bin/rl_data.rs`:
net+MCTS plays vs d3, records the net's (features, MCTS-visit-policy, outcome z),
split into `.win`/`.loss` files. Generated 600 games @1000 sims. **The net won only
~11%** during generation (root noise + no guard → below arena strength) → just 2565
win-samples vs 12593 loss-samples — the "beat d3" signal is sparse.

| Net | win-data weight | 1000 | 1500 | 2200 | 3000 |
|-----|-----------------|------|------|------|------|
| rl1 | win ×3 (upweighted) | 10% | 15% | 15% | 10% (collapse) |
| rl1b | win ×1 (gentle) | 15% | 17.5% | 20% | — |

Both **worse than the 42% student.** Upweighting thin win-data collapses the policy
(overfit to a tiny, possibly-lucky winning subset); gentle weighting leaves it
below the student. The value head, trained mostly on losses, learns pessimism.

**The chicken-and-egg trap, for the policy head this time:** to exceed d3 the
winning-move data must dominate the policy, but the net wins d3 rarely (~11–42%), so
that data is thin — dominating with it overfits. Same barrier as the self-play
bootstrap, restated. RL-vs-d3 does not escape it at this net strength.

**Tried to fix the generator** (added the arena guard + productive-only policy
target + lower noise to `rl_data` so the net plays at arena strength): win rate rose
only to ~12.5% @1200 sims. The exploration needed for data diversity (temperature on
early plies, root noise) costs most of the wins; removing it makes games near-
deterministic (no diversity). And even at full no-exploration strength the net is 42%
— still a minority. There is no operating point where winning trajectories are the
majority, so expert-iteration has no majority signal to learn from. Definitive.

## FINAL FINAL VERDICT — 8 approaches, d3 not beatable

| # | Approach | Best vs d3 |
|---|----------|-----------|
| 1 | Distilled student MLP (d3) | **~42%** ← best, deliverable |
| 2 | CNN spatial encoder | ~17% |
| 3 | ResT teacher, deployed in Rust (parity 4e-7) | 20% |
| 4 | Self-play iter3 (hard targets) | collapse |
| 5 | Self-play az1 (value-decoupled) | 15% |
| 6 | Stronger teacher (d4 search_data) | ~37% |
| 7 | Expert-iteration RL vs d3 (rl1, win×3) | 15% collapse |
| 8 | Expert-iteration RL vs d3 (rl1b, gentle) | 20% |

Nothing beats the simple d3-distilled student (~42% @1500 sims). The net cannot beat
d3, and the reason is a single structural fact expressed three ways:
**the net+MCTS is at/below parity with d3 (peak 42%), so every learning signal that
depends on the net beating d3 — self-play outcomes, RL winning trajectories — is a
minority signal capped below d3, and every signal that imitates a fixed teacher is
capped at that teacher (d3, or d4 imitated worse).** A 196k–524k MLP over these
features simply isn't strong enough to clear a tuned depth-3 alpha-beta here, and no
training scheme manufactures strength the representation can't hold.

**To actually beat d3 would require a categorically different lever:** a much larger
model / richer board representation (so the imitation gap shrinks and self-play can
exceed d3), or abandoning the net for search-side gains (the Alpha-Beta engine is
already the stronger player — see project memory). Both are out of scope of "tune the
small net."

**Deliverable:** `/tmp/wallnet-student.safetensors` — d3-distilled, ~42% vs d3 @1500
sims, ~2× the deployed v6 (22.5%). New infra in place: `WallNetResT` + candle ResT
forward (`net.rs`, parity-verified), `distill.py`, value/policy-decoupled training
(`--value-data`), d4 data pipeline, and `rl_data` (expert-iteration generator).

**Best deliverable:** `/tmp/wallnet-student.safetensors` (d3-distilled, ~42% vs d3
@1500 sims) — ~2× the deployed v6 (22.5%), at 2.5× the sim budget. All
infrastructure (ResT arch + Rust forward + distill + value-decoupled training +
d4 data pipeline) is in place for future RL work.

Artifacts: `/tmp/wallnet-rest-teacher.safetensors`, `/tmp/wallnet-student.safetensors`,
`trainer/distill.py`, logs `/tmp/rest-teacher.log` `/tmp/distill.log` `/tmp/arena-student.log`.

---

## EPILOGUE (2026-06-03) — the search side took the escape hatch, and won

The FINAL FINAL VERDICT named two levers to actually beat d3: (a) a categorically
bigger net, or (b) **abandon the net, strengthen the search side** (the alpha-beta
engine was already the stronger player). Lever (b) was taken the next session and
**succeeded** — see `../memory/project_search_optimization.md`.

Key move: `search_moves` in `core/src/moves.rs` (`wall_move_gen(state, path_only=true)`)
restricts wall generation to only walls that block ≥1 player's current shortest BFS
path — ~150 legal walls → ~15 candidates. 20–300× speedup at D6–D10, depth
monotonicity preserved (both sides use the same restricted set). Result: **D10
playable in WASM at 937 ms/move** (D7 900ms→30.6ms, D8 3500ms→95.6ms), monotonic
D4–D10. Deployed `ab-d10` bot.

So the two threads close opposite ways:
- **Net (this doc + training-log):** 8 imitation/self-play/RL approaches, all capped
  <50% vs d3. Dead end at this compute/representation budget.
- **Search:** the same target (beat d3) reached trivially by going the *other*
  direction — search d10 ≫ d3. The strongest deployed bot is alpha-beta, not the net.

The net research was not wasted: it *proved* d3-imitation can't exceed d3, which is
exactly what redirected effort to the search side. The verdict's recommendation was
correct and is now realized.

---

## References

1. Chen et al. — *Bridging Local and Global Knowledge via Transformer in Board Games.* arXiv 2410.05347. https://arxiv.org/abs/2410.05347 · HTML: https://arxiv.org/html/2410.05347 · IJCAI-25: https://www.ijcai.org/proceedings/2025/828 · project: https://rlg.iis.sinica.edu.tw/papers/restnet
2. *From Images to Connections: Can DQN with GNNs Learn the Strategic Game of Hex?* arXiv 2311.13414. https://arxiv.org/pdf/2311.13414
3. *Evaluation Beyond Task Performance: Analyzing Concepts in AlphaZero in Hex.* arXiv 2211.14673. https://arxiv.org/pdf/2211.14673
4. *SpatialSim: Recognizing Spatial Configurations of Objects With Graph Neural Networks.* arXiv 2004.04546. https://arxiv.org/pdf/2004.04546
5. *On Neural Architecture Inductive Biases for Relational Tasks.* arXiv 2206.05056. https://arxiv.org/pdf/2206.05056
6. Battaglia et al. — *Relational Inductive Biases, Deep Learning, and Graph Networks* (2018). https://arxiv.org/abs/1806.01261

Related internal docs: `training-log-2026-06-01.md` (Step 8 cage analysis, Step 9–10 CNN experiment), `cnn-encoder-research.md`.
