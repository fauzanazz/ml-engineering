# Wall Chess ML Bot — Build Plan

> Status: proposal / roadmap. Game is Quoridor (9×9, 10 walls each, race to opposite row, no-trap rule).
> Goal: replace the hand-tuned leaf evaluator with a learned one, deployable in the browser via WASM.

---

## 1. Strategy: Hybrid (AlphaZero teacher → NNUE student)

Two networks with two different jobs:

| Role | Net | Where it runs | Job |
|------|-----|---------------|-----|
| **Teacher** | AlphaZero (ResNet policy+value) + MCTS | Offline, Python + GPU | Self-play to generate strong positions and labels. Strength ceiling. |
| **Student** | NNUE (small, int8, efficiently-updatable) | In-repo, Rust → WASM | Ships in the browser. Plugs into existing alpha-beta. Speed. |
| **Bridge** | Distillation | Offline | Train student to match teacher's value judgments. |

**Why hybrid:** AlphaZero gives the strongest possible training signal but MCTS + ResNet is too heavy for a no-server browser bot. NNUE is built for exactly the deploy constraint (CPU/WASM, runs inside alpha-beta search at depth) but trains better against a strong teacher than against itself from scratch. Take the ceiling for training, take the speed for deploy.

**Rejected alternatives:** GNN (elegant — walls = graph edge edits — but heavier inference, weaker WASM story, fewer references), Transformer (overkill at 9×9, no efficient-update trick). Could revisit GNN as an alternate *teacher* later; not on the critical path.

---

## 2. What already exists (reuse, don't rebuild)

The Rust core was written with this swap in mind.

- **`core/src/eval.rs`** — `trait Evaluator { fn eval(&self, state, side) -> i32 }`. Explicit ML seam: *"same search, swap the scoring of leaf states."* `Heuristic` is the current impl. Output is centi-steps, squashed by `win_prob`. **The student net implements this trait.**
- **`core/src/state.rs`** — bit-packed `State`: `pawns: [Cell; 2]`, `h_walls: u64`, `v_walls: u64`, `walls_left: [u8; 2]`, `turn`, `winner`. This *is* the sparse feature source for both nets.
- **`core/src/moves.rs`** — `legal_moves`, `distance_to_goal` (BFS). The BFS distance is the dominant Quoridor feature; feed it to the nets as input, don't make them relearn it.
- **`core/src/search.rs`** — negamax alpha-beta, iterative deepening, transposition table, pluggable `&dyn Evaluator`. **Untouched by this project** — only the evaluator changes.
- **`wasm/src/lib.rs`** — wasm-bindgen surface (`analyze_state`, `top_moves_js`, `generate_graph_js`, `choose_move`). Deserializes the TS `GameState` JSON, runs the real search, returns move + 0..100 win split. New eval ships through here.
- **`webui/src/game/engine.ts`** (+ `bot.ts`, `api.ts`, `validate.ts`) — reference TS rules engine the Rust core mirrors. Source of truth for rule parity checks.
- **`core/src/bin/{selfplay,arena,diag,graphcount}.rs`** — **already exist**. `selfplay` and `arena` are the seed of the strength-measurement harness — extend them to pit any two `Evaluator`s and report winrate/ELO instead of rebuilding from scratch.

---

## 3. State / action representation (shared contract)

Defining this once, up front, so teacher and student agree.

### Input planes (teacher) / sparse features (student)
Derived from `State`:

1. South pawn position — 9×9 one-hot
2. North pawn position — 9×9 one-hot
3. Horizontal walls — 8×8 (from `h_walls` u64)
4. Vertical walls — 8×8 (from `v_walls` u64)
5. Walls-left South — scalar, broadcast to a plane (or bucketed one-hot 0..10)
6. Walls-left North — scalar, broadcast
7. Side to move — single bit / constant plane
8. **BFS distance-to-goal map, mover** — 9×9, from `distance_to_goal`
9. **BFS distance-to-goal map, opponent** — 9×9

Always encode **from the side-to-move's perspective** (canonical orientation: flip the board so the mover always races "up"). Halves what the net must learn and matches negamax convention.

### Action space (teacher policy head)
Fixed, ~140 discrete actions:
- Pawn moves: up to 12 (4 orthogonal + jump + diagonal-jump variants)
- Horizontal wall placements: 64 (8×8 anchors)
- Vertical wall placements: 64 (8×8 anchors)

Mask illegal actions with `legal_moves` before softmax. Fixed index mapping must be identical in Python and Rust — define it once in a shared spec (see §8).

---

## 4. Phase plan

### Phase 0 — Foundations & contract (prereq)
- [ ] Write `docs/encoding-spec.md`: exact plane order, action index mapping, perspective-canonicalization rule. Single source of truth for both languages.
- [ ] Decide the Rust↔Python bridge (see §8). Recommendation: **PyO3 binding over the existing Rust core** so Python self-play uses the *same* rules engine — zero reimplementation drift.
- [ ] Build the strength-measurement harness: head-to-head match runner + ELO/winrate, extending the existing `core/src/bin/{selfplay,arena}.rs`. This is how every later phase is judged.
- [ ] Lock the current `Heuristic` as the **baseline opponent** and the **bootstrap label source**.

**Exit:** can play `Heuristic` vs `Heuristic` (and later any two evaluators) N games and report winrate/ELO.

### Phase 1 — AlphaZero teacher (offline, Python)
- [ ] Model: ResNet trunk (e.g. 6–10 blocks, 64–128 channels — small board, keep modest), policy head (~140 logits) + value head (scalar tanh).
- [ ] MCTS: PUCT, net-guided priors, value-head leaf eval (no random rollouts), Dirichlet noise at root, temperature schedule.
- [ ] Self-play loop: generate games → store (position, MCTS visit-count policy, game outcome).
- [ ] Train: policy loss (cross-entropy vs visit counts) + value loss (MSE vs outcome) + L2.
- [ ] Iterate: new net replaces self-play net when it beats it in the harness (gating).
- [ ] Bootstrap option: seed early self-play / pretraining against `Heuristic` to skip the coldest start.

**Exit:** teacher net beats `Heuristic` decisively (target ≥ ~80% winrate) in the harness.

**References:** public AlphaZero-Quoridor implementations exist — use as architecture/hyperparameter reference, not copy-paste.

### Phase 2 — Distillation dataset
- [ ] Run teacher self-play (and/or teacher vs varied opponents) to dump a large, *diverse* position set.
- [ ] For each position store: canonical features (§3), teacher value (and optionally teacher policy / MCTS eval), final game outcome.
- [ ] Ensure coverage: openings, midgame wall-fights, near-terminal races, low-wall endgames. Avoid a dataset that is all one game phase.
- [ ] Hold out a validation split for honest student eval.

**Exit:** dataset of ≥ ~1M positions with teacher labels, version-pinned.

### Phase 3 — NNUE student (train in Python, run in Rust)
- [ ] Architecture: sparse feature transformer (wide first layer = "accumulator") → 2–3 small dense layers → scalar. Designed for int8 quantization.
- [ ] Train: regression to teacher value (MSE), optionally mixed with game outcome. Quantization-aware training (QAT) so int8 export holds accuracy.
- [ ] Export weights to a compact binary format readable by Rust (versioned header).
- [ ] Implement `struct NnueNet` + `impl Evaluator for NnueNet` in `core/src/`. Output centi-steps so `win_prob` and `search.rs` are unchanged.
- [ ] Implement the **efficient accumulator update**: alpha-beta children differ by one move (one pawn step or one wall bit). Update the accumulator incrementally on make/unmake instead of recomputing — the whole reason NNUE is fast in search.

**Exit:** `NnueNet` loads weights, scores positions matching the Python net within quantization tolerance; unit-tested for parity.

### Phase 4 — Integration & WASM
- [ ] Wire `NnueNet` through `wasm/src/lib.rs` (load weights blob, expose as the engine's evaluator). Keep `Heuristic` selectable for A/B and fallback.
- [ ] Bundle weights for the browser (size budget: aim small — quantized NNUE is tens–hundreds of KB, not MB).
- [ ] Benchmark in-browser: nodes/sec, achievable depth, move latency. Confirm it's playable (sub-second move at reasonable depth).
- [ ] Verify rule parity Rust ↔ `webui/src/game/engine.ts` still holds.

**Exit:** browser bot plays with the learned eval, no server, within latency budget.

### Phase 5 — Measure, tune, iterate
- [ ] Head-to-head: `NnueNet` vs `Heuristic` vs teacher (sanity: teacher ≥ student ≥ heuristic).
- [ ] Tune search depth / time control for the student's eval cost.
- [ ] Optional loop: stronger teacher → re-distill → stronger student.
- [ ] Optional: small per-difficulty configs (depth/temperature) for casual vs hard bot in the UI.

**Exit:** student beats `Heuristic` by a clear margin and is the default browser bot.

---

## 5. Repo layout (proposed additions)

```
core/src/
  eval.rs            # add NnueNet: Evaluator alongside Heuristic
  nnue.rs            # weight loading + accumulator + incremental update (new)
wasm/src/lib.rs      # expose evaluator selection + weights blob
ml/                  # new — Python training project (uv)
  pyproject.toml
  encoding/          # plane/action encoding, MUST match docs/encoding-spec.md
  teacher/           # AlphaZero net, MCTS, self-play loop
  distill/           # dataset dump + NNUE training + QAT + export
  bridge/            # PyO3 bindings to wallchess-core (or rules reimpl)
  eval_harness/      # head-to-head, ELO
docs/
  ml-bot-plan.md     # this file
  encoding-spec.md   # shared contract (new, Phase 0)
weights/             # versioned exported NNUE blobs
```

Python tooling: **uv** (per workspace convention). Rust unchanged toolchain.

---

## 6. Critical risks & mitigations

| Risk | Mitigation |
|------|------------|
| Rules drift between Python self-play and Rust engine | PyO3 over the *same* Rust core — one rules implementation. Parity test vs `engine.ts`. |
| Encoding mismatch (plane/action order) Python vs Rust | `docs/encoding-spec.md` is the single contract; round-trip test a few hand-built positions in both languages. |
| Quantization tears down student accuracy | Quantization-aware training; parity test int8 Rust output vs float Python output. |
| AlphaZero self-play too slow / expensive | Small ResNet (board is tiny), modest sim count, bootstrap from `Heuristic`, batch MCTS leaf eval. |
| WASM too slow at depth | NNUE incremental accumulator is the whole point; benchmark early in Phase 4, cap depth via time control. |
| Weights bloat the web bundle | int8 + small net; budget tens–hundreds of KB; lazy-load the blob. |
| "Is it actually better?" ambiguity | Strength harness built in Phase 0, used as the gate for every phase. |

---

## 7. Definition of done

- Browser plays the bot with the learned NNUE eval, no server, sub-second moves at a playable depth.
- `NnueNet` beats `Heuristic` by a clear, measured margin in the head-to-head harness.
- `Heuristic` remains available as baseline / fallback.
- Reproducible pipeline: self-play → dataset → distill → export → Rust → WASM, documented.

---

## 8. Open decisions (resolve in Phase 0)

1. **Bridge:** PyO3 binding to `wallchess-core` (recommended, no rules drift) **vs** reimplement rules in Python (simpler infra, drift risk) **vs** subprocess/FFI over a serialized protocol.
2. **Teacher framework:** PyTorch (default) vs JAX.
3. **Policy in student?** Pure value NNUE (simplest, fits existing alpha-beta) vs also distill a policy for move ordering / shallower search. Start value-only.
4. **Outcome vs teacher-value labels** for distillation: start with teacher value, A/B a mix with game outcomes.
5. **Bootstrap depth:** how much `Heuristic`-seeded pretraining before pure self-play.
