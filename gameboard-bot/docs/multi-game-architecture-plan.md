# Multi-game architecture plan

> Status: LANDED (phases 0–2, 4–5); phase 3 DEFERRED; phase 6 = this doc set. Full suite 62 passed / 1 ignored, golden snapshot byte-identical, checkers perft d1–d8 exact.
> Goal: turn the single-game Wall Chess engine into a game-agnostic platform — one shared search/eval/match/NN pipeline that any two-player, perfect-information, zero-sum board game plugs into — **without** duplicating the engine and **without** changing one byte of the deployed Wall Chess bot's behavior.

---

## 1. The problem

`gameboard-core` (née `wallchess-core`) was a Wall Chess engine: negamax + alpha-beta + iterative deepening + transposition table + a pruning bundle + an arena match driver + an NN seam, all written against concrete `State`/`Move`/`Side` types from Wall Chess. Adding a second game (International Draughts) the obvious way — copy the engine, swap the rules — would fork search, eval, the arena, and the NN pipeline into two drifting codebases.

The constraint that shapes everything: **the deployed Wall Chess bot must not change.** Its strength was tuned and validated empirically (Gen-2 beats d12-v2 at 81%/200g), and the search is *deterministic* — the webui and the arena rely on byte-for-byte reproducibility. So the refactor has a hard invariant: a golden `bestmove` snapshot taken before any change must stay identical after every change.

**Decision:** extract one trait seam — `Game` — at exactly the `State`/`Move` boundary. Everything above it (search, arena, score scale, NN feature contract) becomes generic; everything that names a wall, a pawn jump, a flying king, a goal row, or a board dimension stays per-game behind the trait.

---

## 2. The seam: the `Game` trait

`core/src/game.rs` defines the contract. Implementors are zero-size marker types (`struct WallChess;`, `struct Checkers;`) that tie together a `State`, a `Move`, and the rules/encoding functions. All trait methods are associated (no `&self`) — a game has no instance state, only its rules.

```rust
pub trait Game: Sized + 'static {
    type State: Clone + Copy + PartialEq + Eq + Hash + Debug;
    type Move:  Clone + Copy + PartialEq + Eq + Hash + Debug;

    const ID: &'static str;
    const ACTION_COUNT: usize;      // dense policy-head width (NN)
    const FEATURE_LEN: usize;       // NN feature-vector length
    const MOVE_INDEX_SPACE: usize;  // size of the history/killer table

    fn initial() -> Self::State;
    fn turn(s: &Self::State) -> Player;
    fn winner(s: &Self::State) -> Option<Player>;
    fn is_terminal(s: &Self::State) -> bool;
    fn apply(s: &Self::State, mv: Self::Move) -> Self::State;
    fn null_move(s: &Self::State) -> Self::State;
    fn is_legal(s: &Self::State, mv: Self::Move) -> bool;

    fn legal_moves(s: &Self::State) -> Vec<Self::Move>;
    fn search_moves(s: &Self::State) -> Vec<Self::Move>;
    fn search_moves_wide(s: &Self::State) -> Vec<Self::Move>;
    fn capture_moves(_s: &Self::State) -> Vec<Self::Move> { Vec::new() } // default: none
    fn is_quiet(mv: Self::Move) -> bool;

    fn hash(s: &Self::State) -> u64;                 // never 0 (empty-slot sentinel)
    fn move_order_index(mv: Self::Move) -> usize;    // < MOVE_INDEX_SPACE
    fn immediate_winning_move(s: &Self::State) -> Option<Self::Move>;

    fn state_key(s: &Self::State) -> String;
    fn parse_state_key(k: &str) -> Option<Self::State>;
}
```

The generic player marker replaces every per-game `Side`/`Color`:

```rust
pub enum Player { P0 = 0, P1 = 1 }  // P0 moves first from the initial position
```

The score scale is shared so the search's mate-zone guards (`score.abs() >= WIN_SCORE`) mean the same thing in every game:

```rust
pub const WIN_SCORE:   i32 = 1_000_000;
pub const ENDGAME_WIN: i32 = WIN_SCORE - 1000; // 999_000 — provably resolved but not yet terminal
```

`Evaluator` and `Encoder` carry their game as an associated type (`type G: Game`) rather than a second type parameter. This is the trick that keeps every existing call site unchanged (§3).

```rust
pub trait Evaluator { type G: Game; fn eval(&self, s: &<Self::G as Game>::State, p: Player) -> i32; }
pub trait Encoder   { type G: Game; /* encode / action_index / index_to_move / mirror_move */ }
```

### Why generics, not trait objects

This is the load-bearing design decision. We did **not** make `Game`/`Move` into `dyn` trait objects.

| Concern | `dyn Game` (trait objects) | Generics (`Search<E: Evaluator>`) — chosen |
|---------|----------------------------|---------------------------------------------|
| `Move` storage | Must box / erase — `Move` can't stay a `Copy` value in const-sized tables | `Move: Copy` lives directly in the TT slots and killer arrays |
| TT / history tables | Sized at runtime, pointer-chased | Const-sized per game (`MOVE_INDEX_SPACE`), monomorphized |
| Determinism | Vtable dispatch + heap erasure perturb the hot loop | Compiles to one fully-inlined engine per game — byte-for-byte identical to the old code |
| Dispatch cost | One indirect call per node | Zero — static dispatch, fully inlined |

The trade is one `match game_id { .. }` at the binary boundary (CLI/wasm) instead of dynamic dispatch in the inner loop. Game selection happens **once**, at a binary edge, then the whole search monomorphizes. The deployed bot's reproducibility — the thing the webui and arena depend on — is *preserved by construction*: the monomorphized `Search<Heuristic>` is the same machine code path as before the refactor.

---

## 3. What stays generic vs. what is per-game

**The boundary is exactly the concrete `State`/`Move` types.**

| Generic (game-agnostic, written once) | Per-game (behind `Game`) |
|---------------------------------------|--------------------------|
| `core/src/search.rs` — negamax/alpha-beta, iterative deepening, TT, killers/history, PVS/NMP/RFP/razor/futility/LMP/LMR/aspiration, quiescence | move generation, legality, BFS distance, jump/flying-king rules, no-trap rule |
| `core/src/arena.rs` — `BotConfig<E>`, `play_game<E>`, `play_match<E>`, `Outcome` | terminal detection, winner, board dims |
| `core/src/game.rs` — `Game`/`Evaluator`/`Encoder`/`Player`, score scale | the eval weights, the feature layout |
| score scale (`WIN_SCORE`, `ENDGAME_WIN`) | what counts as a resolved endgame |
| NN feature *contract* (`Encoder`) | the actual tensor planes / action index |

**One exception, deliberately left non-generic:** the AlphaZero-style policy/value MCTS seam (`PolicyValue` in `core/src/mcts.rs`) is still bound to Wall Chess. The `Encoder` contract is already game-generic, but `Mcts`/`net`/graph are not. Genericizing them is deferred to when checkers NN training actually begins — there is no second consumer yet, so the abstraction would be speculative.

### How the seam keeps every call site unchanged

`Search` keeps **one** type parameter and resolves its game *through the evaluator*:

```rust
pub struct Search<'a, E: Evaluator> { /* tt: Vec<TtSlot<Mv<E>>>, history: Vec<i32>, .. */ }

// projections used throughout search.rs / arena.rs:
type Gm<E> = <E as Evaluator>::G;
type St<E> = <Gm<E> as Game>::State;
type Mv<E> = <Gm<E> as Game>::Move;
```

Because the game is reached via `E::G`, every existing caller that wrote `Search<'_, Heuristic>` compiles unchanged — no turbofish, no second type argument at the call site. Two mechanical changes were forced by going generic:

- **TT slot** became `TtSlot<M: Copy>` (generic over the move type).
- **History table** moved from a fixed `[i32; N]` array to `Vec<i32>` sized at `<Gm<E> as Game>::MOVE_INDEX_SPACE`. Associated-const array lengths are not nameable in generic Rust, so the array became a `Vec` — the *values* are identical, so search results are unchanged.

Also added in the same pass: a `pub quiescence` flag + new `qsearch()` method (mandatory-capture quiescence — no stand-pat while captures exist) and a `SearchConfig::draughts()` preset (`pvs + aspiration + lmr + quiescence` ON, `null_move` OFF) plus a `WC_QUIESCENCE` env knob. Quiescence is **inert for Wall Chess** because `capture_moves` defaults to an empty `Vec`, so the deployed engine is untouched.

---

## 4. The crate layout that actually landed

Flat modules at the crate root. The generic engine and the per-game glue sit side by side:

```
core/src/
  game.rs        # GENERIC: Game / Evaluator / Encoder / Player / score scale
  search.rs      # GENERIC: Search<E>, SearchConfig (+ draughts preset, qsearch)
  arena.rs       # GENERIC: BotConfig<E>, play_game/play_match, Outcome
  wallchess.rs   # GLUE:    struct WallChess; impl Game + impl Encoder
  checkers.rs    # GLUE:    struct Checkers; impl Game + impl Encoder + CheckersHeuristic
  checkers/tests.rs  # International draughts rule-fidelity + perft tests
  state.rs moves.rs eval.rs action.rs features.rs mcts.rs net.rs books.rs  # Wall Chess internals
  bin/checkers.rs    # new CLI: perft / bestmove / selfplay
```

`wallchess.rs` hoists the previously-private `state_hash → hash`, `history_idx → move_order_index`, and `immediate_winning_move` out of `search.rs` into the `Game` impl (byte-identical bodies), plus `pub fn side_to_player` / `player_to_side`. Player mapping: **Wall Chess** South=P0, North=P1; **Draughts** White=P0, Black=P1.

Game dimensions, as implemented:

| Game | `ID` | `ACTION_COUNT` | `FEATURE_LEN` | `MOVE_INDEX_SPACE` |
|------|------|----------------|---------------|--------------------|
| Wall Chess | `"wallchess"` | 209 | 300 | 384 |
| Checkers (Intl. 10×10) | `"checkers"` | 2500 (`N*N`) | 308 (`6*N+8`) | 2500 (`from*50 + to`) |

### Deferred: the `games/` subdirectory nesting

The original plan put each game under `games/wallchess/` and `games/checkers/`. **Decision: DEFERRED.** It is pure cosmetic file-move churn — every move would be golden-guarded with no behavior change, and checkers already proves multi-game isolation works from flat modules. Not worth the diff noise. The flat layout is the one that landed.

---

## 5. Phased order (as executed)

- [x] **Phase 0 — golden snapshots.** Capture `bestmove` output for a fixed position set + baseline tests, so every later phase is guarded against drift. **Exit:** `core/tests/golden.rs` + `golden_snapshot.txt` committed; suite green.
- [x] **Phase 1 — the seam.** Add `core/src/game.rs`: `Game` / `Evaluator` / `Encoder` / `Player` / `WIN_SCORE` / `ENDGAME_WIN`. No engine changes yet. **Exit:** trait compiles; golden byte-identical.
- [x] **Phase 2 — genericize Search + arena.** `Search<'a, E: Evaluator>` resolving its game via `E::G`; `TtSlot<M>`; history → `Vec<i32>` at `MOVE_INDEX_SPACE`; `qsearch` + `draughts()` preset; `arena.rs` → `BotConfig<E>` / `play_game<E>` / `play_match<E>` / `Outcome`. **Exit:** every `Search<'_, Heuristic>` caller unchanged; golden byte-identical; suite green.
- [x] **Phase 3 — relocate into `games/`. DEFERRED.** Judged cosmetic, golden-guarded file moves; flat modules retained. **Exit:** explicitly not done; documented as deferred.
- [x] **Phase 4 — rename to `gameboard-core`.** Package `wallchess-core → gameboard-core`, lib `wallchess_core → gameboard_core`, across `Cargo.toml`, all bins, `tests/golden.rs`, and the wasm dep + `wasm/src/lib.rs`. The wasm **package** name stays `wallchess-wasm` (per-game wasm bundles deferred). **Exit:** crate + wasm compile against renamed dep; golden byte-identical.
- [x] **Phase 5 — checkers engine.** `struct Checkers` impl `Game` + `Encoder` + `CheckersHeuristic`; `core/src/checkers/tests.rs`; new `core/src/bin/checkers.rs` (perft/bestmove/selfplay). **Exit:** perft d1–d6 in fast tests + d7/d8 behind `--ignored`, all exact (see §6); suite 62 passed / 1 ignored.
- [x] **Phase 6 — documentation.** This plan + the per-component briefs. **Exit:** doc set written.

---

## 6. Verification state

- **Golden snapshot:** byte-identical after every phase — the deployed Wall Chess bot is provably unchanged.
- **Full suite:** 62 passed / 1 ignored (the ignored one is the slow checkers perft d7/d8).
- **Checkers perft (the fidelity oracle):** d1=9, d2=81, d3=658, d4=4265, d5=27117, d6=167140 (fast tests); d7=1049442, d8=6483961 (the 4+-capture dedup tripwire, `--ignored`). All exact.
- **wasm:** compiles against the renamed `gameboard-core` dependency.

---

## 7. Out of scope this session / next phase

Honest NOT-LANDED list — these are real follow-ups, not done:

- **wasm per-game bundles.** The wasm package is still `wallchess-wasm`, Wall-Chess-only. No checkers wasm surface yet.
- **webui game-picker + per-game Board.** The UI is single-game; no game-selection dropdown, no checkers board renderer.
- **trainer per-game `GameSpec`.** The Python trainer is Wall-Chess-bound; no abstraction to point it at a second game's encoder/action space.
- **NN training for checkers.** No teacher/student nets, no self-play data, no learned evaluator for draughts — `CheckersHeuristic` is hand-written.
- **Genericizing `Mcts` / `net` / graph over `G`.** The `PolicyValue` MCTS seam, the candle net path, and the state-graph generator are still Wall-Chess-bound. The `Encoder` contract is generic; its consumers are not. Deferred until checkers NN training begins.
