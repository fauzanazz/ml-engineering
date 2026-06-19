# Game trait design

> Status: LANDED — `Game` / `Evaluator` / `Encoder` are the live seam; two games (Wall Chess, International Draughts) ship behind it; full suite 62 passed / 1 ignored, golden snapshot byte-identical, checkers perft d1–8 EXACT.
> Goal: document the trait surface that lets one generic search/eval/match/NN pipeline serve any two-player, perfect-information, zero-sum board game — and the rules that keep adding a game from disturbing the deployed bot's numerics.

---

## The seam

The boundary runs **exactly at the concrete `State`/`Move` types** (`core/src/game.rs`). Above them — negamax/alpha-beta, the transposition table, killers/history, LMR/NMP/PVS, the arena match driver — everything is generic over `Game`. Below them — anything that names a wall, a pawn jump, a flying king, a goal row, or a board dimension — is per-game logic behind the trait.

Dispatch is **monomorphized, not dynamic**: `Search<E: Evaluator>` resolves its game through `E::G`, so each concrete game compiles to its own fully optimized engine — no boxing, no `Move`-erasure. The `Copy` value-type `Move` and the const-sized tables the deployed bot relies on are preserved. Game selection happens once, at a binary boundary (`match game_id { .. }`).

---

## The `Game` trait

A `Game` implementor is a zero-size marker type (e.g. `struct WallChess;`, `struct Checkers;`). All methods are associated (no `&self`): the game has no instance state, only its rules.

```rust
pub trait Game: Sized + 'static {
    // --- associated types ---
    /// Bit-packed, cheap-to-copy board state.
    type State: Clone + Copy + PartialEq + Eq + std::hash::Hash + std::fmt::Debug;
    /// The action type. `Copy` so the search keeps it in const-sized tables.
    type Move:  Clone + Copy + PartialEq + Eq + std::hash::Hash + std::fmt::Debug;

    // --- identity / dims ---
    const ID: &'static str;          // stable id at CLI/wasm dispatch + in data files
    const ACTION_COUNT: usize;       // dense policy-head width (NN action space)
    const FEATURE_LEN: usize;        // NN feature-vector length produced by the Encoder
    const MOVE_INDEX_SPACE: usize;   // size of history/move-ordering table; > every move_order_index

    // --- rules ---
    fn initial() -> Self::State;                              // start position
    fn turn(s: &Self::State) -> Player;                       // side to move
    fn winner(s: &Self::State) -> Option<Player>;            // decided winner, else None (ongoing OR drawn)
    fn is_terminal(s: &Self::State) -> bool;                  // search stops here (won/lost/drawn)
    fn apply(s: &Self::State, mv: Self::Move) -> Self::State; // apply move assumed legal (flip turn, update board)
    fn null_move(s: &Self::State) -> Self::State;             // flip side to move, board unchanged (NMP only)
    fn is_legal(s: &Self::State, mv: Self::Move) -> bool;     // validate a candidate move

    // --- move generation ---
    fn legal_moves(s: &Self::State) -> Vec<Self::Move>;       // all legal moves for the side to move
    fn search_moves(s: &Self::State) -> Vec<Self::Move>;      // pruned candidate set for interior nodes
    fn search_moves_wide(s: &Self::State) -> Vec<Self::Move>; // wider candidate set, root only
    fn capture_moves(_s: &Self::State) -> Vec<Self::Move> {   // tactical moves for quiescence; DEFAULT: none
        Vec::new()
    }
    fn is_quiet(mv: Self::Move) -> bool;                      // quiet ⇒ eligible for LMR/LMP/futility

    // --- search hooks ---
    fn hash(s: &Self::State) -> u64;                          // 64-bit TT hash; NEVER returns 0 (empty-slot sentinel)
    fn move_order_index(mv: Self::Move) -> usize;            // compact index into history/killer tables; < MOVE_INDEX_SPACE
    fn immediate_winning_move(s: &Self::State) -> Option<Self::Move>; // one-ply forced win, lets root short-circuit

    // --- serialization (graph dedup / UI parity / books) ---
    fn state_key(s: &Self::State) -> String;                 // canonical string key
    fn parse_state_key(k: &str) -> Option<Self::State>;      // inverse of state_key
}
```

### Notes on a few methods

- **`winner` vs `is_terminal`** are split deliberately. Games with an O(1) terminal marker (Wall Chess stores `s.winner`) stay cheap; games whose terminality is "side to move has no move" (draughts) answer `is_terminal` with an early-exit probe instead of full move generation. `winner` returns `None` for *both* ongoing and drawn — a draw has no winner.
- **`search_moves` / `search_moves_wide`** let a game narrow the branching factor (Wall Chess drops walls that cut no shortest path; draughts currently returns the full legal set at every node, root or interior, and lets alpha-beta do the work). Either may equal `legal_moves`.
- **`capture_moves` default is empty.** A game without a capture notion (Wall Chess) never enters quiescence — qsearch finds no captures and returns the static eval. Draughts overrides it with the mandatory-capture set.
- **`hash` never returns 0** because 0 is the TT empty-slot sentinel; both impls fold a final `if h == 0 { 1 }`.

---

## `Evaluator` and `Encoder`

```rust
/// Leaf evaluator: scores a position from `p`'s POV in the negamax convention.
/// Antisymmetric on non-terminal states: eval(s, P0) == -eval(s, P1).
/// Terminal positions score ±WIN_SCORE.
pub trait Evaluator {
    type G: Game;
    fn eval(&self, state: &<Self::G as Game>::State, p: Player) -> i32;
}

/// NN feature-encoding contract: the me-frame tensor layout that must stay
/// byte-identical to the Python trainer's encoder.
pub trait Encoder {
    type G: Game;
    fn encode(state: &<Self::G as Game>::State) -> Vec<f32>;          // flat vector, len FEATURE_LEN, side-to-move ("me") frame
    fn action_index(mv: <Self::G as Game>::Move) -> usize;           // dense action index, < ACTION_COUNT
    fn index_to_move(i: usize) -> <Self::G as Game>::Move;           // inverse; NO legality check — intersect with legal set
    fn mirror_move(p: Player, mv: <Self::G as Game>::Move) -> <Self::G as Game>::Move; // absolute → me-frame of p (involution)
}
```

### Why `Evaluator` carries its game as an associated type

`Evaluator::G` is an associated type, not a second type parameter on `Search`. This is what keeps **`Search<E: Evaluator>` single-param**: the search resolves its game through `E::G`, so construction (`Search::with_config(&heuristic, cfg)`) infers the game from the evaluator with **no turbofish** at the call site. Every existing caller that wrote `Search<'_, Heuristic>` compiles unchanged. State/Move are `Copy`, so the search uses const-sized tables and monomorphizes per game — preserving the byte-for-byte determinism the deployed bot relies on.

> The AlphaZero-style policy/value MCTS seam (`PolicyValue`) currently lives in `core/src/mcts.rs`, **bound to Wall Chess**. Genericizing it over `Game` is **DEFERRED** to when checkers NN training begins. The `Encoder` contract above is already game-generic.

---

## `Player` and the score scale

```rust
pub enum Player { P0 = 0, P1 = 1 }   // P0 = side that moves first from initial()
// Player::other() flips; Player::idx() returns 0/1.

pub const WIN_SCORE:   i32 = 1_000_000;            // terminal magnitude
pub const ENDGAME_WIN: i32 = WIN_SCORE - 1000;     // 999_000 — provably resolved, NOT terminal
```

`Player` is the generic, game-neutral replacement for any per-game `Side`/`Color`. **P0 is the side that moves first from the initial position.**

**Negamax meaning of the consts:**
- **`WIN_SCORE`** — large finite magnitude for *terminal* states, kept below i32 saturation so alpha-beta windows never overflow. Shared across every game so the search's `score.abs() >= WIN_SCORE` mate guards mean the same thing everywhere. Terminal nodes score `±WIN_SCORE`.
- **`ENDGAME_WIN`** — decisive magnitude for a *provably resolved but not-yet-terminal* position (a won race, a won material endgame). **Strictly below `WIN_SCORE`** so a real terminal win is always preferred and the mate-zone guards are not tripped by a resolved-but-non-terminal node.

### Colour → Player mapping

| Game | Colour | Player | Notes |
|------|--------|--------|-------|
| Wall Chess | South | `P0` | South moves first; `side_to_player` / `player_to_side` in `wallchess.rs` |
| Wall Chess | North | `P1` | |
| Draughts | White | `P0` | White moves first, up the board; `color_to_player` / `player_to_color` in `checkers.rs` |
| Draughts | Black | `P1` | Black moves down the board |

---

## What stays generic vs. what's per-game

| Generic (above the seam) | Where | Per-game (behind the trait) |
|--------------------------|-------|-----------------------------|
| `Search<'a, E: Evaluator>` (one type param via `E::G`) | `core/src/search.rs` | `legal_moves` / `search_moves` / `search_moves_wide` / `capture_moves` |
| `SearchConfig` (pvs, null_move, rfp, razor, futility, lmp, lmr, aspiration, **quiescence**) + presets | `core/src/search.rs` | `is_quiet` (drives LMR/LMP/futility eligibility) |
| TT slot `TtSlot<M>` (generic over `Move`) | `core/src/search.rs` | `hash` (TT key, never 0) |
| Killers + history table (`Vec<i32>` sized at `MOVE_INDEX_SPACE`) | `core/src/search.rs` | `move_order_index` (slot into history/killers) |
| `qsearch()` — mandatory-capture quiescence (no stand-pat while captures exist) | `core/src/search.rs` | `capture_moves` (empty default ⇒ qsearch inert) |
| `BotConfig<E>`, `play_game<E>`, `play_match<E>` | `core/src/arena.rs` | `Evaluator::eval`, `apply`, `turn`, `winner`, `is_terminal` |
| `enum Outcome { P0Win, P1Win, Draw }` | `core/src/arena.rs` | `initial`, `state_key` / `parse_state_key` |
| `Player`, `WIN_SCORE`, `ENDGAME_WIN` | `core/src/game.rs` | `Encoder` layout (`encode` / `action_index` / `index_to_move` / `mirror_move`) |

> The history table was moved from a fixed `[i32; N]` array to a `Vec<i32>` sized at `MOVE_INDEX_SPACE` because assoc-const lengths are not nameable in generic Rust. The values are identical, so results are unchanged.

> `SearchConfig::draughts()` preset turns on pvs + aspiration + lmr + **quiescence** and leaves null_move **OFF** (forced-capture sequences make a passed turn unsound). The `WC_QUIESCENCE` env var toggles the flag for Wall Chess, where it is inert (empty `capture_moves`).

---

## Per-game const dimensions

| Game | `ID` | `ACTION_COUNT` | `FEATURE_LEN` | `MOVE_INDEX_SPACE` |
|------|------|----------------|---------------|--------------------|
| Wall Chess | `"wallchess"` | 209 | 300 | 384 |
| Draughts | `"checkers"` | 2500 | 308 | 2500 |

Derivations from the code:
- **Wall Chess** (`wallchess.rs`): re-exports `ACTION_COUNT` (209), `FEATURE_LEN` (300) from `action`/`features`; `MOVE_INDEX_SPACE = HISTORY_SIZE = 384`. History layout: `Pawn(to)` → `to.r*16 + to.c` (0..=153), `Wall(r,c,H)` → 256..=319, `Wall(r,c,V)` → 320..=383.
- **Draughts** (`checkers.rs`, all *provisional*, revisited when training starts): `ACTION_COUNT = N*N = 2500` (from×to = 50×50); `FEATURE_LEN = 6*N + 8 = 308` (six 50-wide planes + an 8-scalar tail); `MOVE_INDEX_SPACE = N*N = 2500` (`from*50 + to`). `move_order_index` = `from*50 + to`; distinct capture *targets* from the same square share a history slot — acceptable for move ordering.

---

## The "relocate without numeric edits" rule

When the per-game rules modules move (Wall Chess's `state`/`moves`/`action`/`features` are slated to relocate under `games/wallchess/`; the `games/` nesting was **DEFERRED** as pure golden-guarded file-move churn — checkers already proves multi-game isolation), the following must stay **byte-identical**:

- **`state_key` / `parse_state_key`** — these are the canonical keys for graph dedup, UI parity, and opening books. A changed key invalidates stored data.
- **`hash`** — the TT key; both impls reproduce a fixed mixing sequence (Wall Chess reproduces the previously-deployed `search::state_hash` byte-for-byte) and never return 0.
- **`move_order_index`** and the **const dims** (`ACTION_COUNT` / `FEATURE_LEN` / `MOVE_INDEX_SPACE`) — these size const tables and the NN tensors that must match the Python trainer.

**Why:** the deployed bot is deterministic. A file move that perturbs any of these numbers silently corrupts arena results, TT hits, or trainer/encoder parity. The golden snapshot test is the guard — it stayed byte-identical after every phase of this refactor.

### How `is_quiet` / `capture_moves` drive the search

- **`is_quiet(mv)`** gates the reduction/pruning heuristics. A quiet move is eligible for **LMR** (late-move reduction), **LMP** (late-move pruning), and futility reduction. Tactical moves return `false` so the search never reduces or skips them: Wall Chess pawn advances (`is_quiet = matches!(mv, Move::Wall(_))` — only wall placements are quiet) and draughts captures (`is_quiet = mv.captured == 0`).
- **`capture_moves(s)`** feeds **quiescence**: `qsearch()` is mandatory-capture (no stand-pat while captures exist). The empty default makes qsearch inert for any capture-less game, so Wall Chess's behaviour is unchanged; draughts returns the deduplicated max-count capture set, so a quiet leaf returns its static eval immediately and a tactical leaf resolves the capture sequence before scoring.
