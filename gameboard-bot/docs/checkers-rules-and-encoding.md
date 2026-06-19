# Checkers rules and encoding

> Status: LANDED — `Checkers` is the second `Game` on the shared engine; rules + eval + provisional encoding implemented and perft-verified (full suite 62 passed / 1 ignored).
> Goal: document the International draughts ruleset as implemented, the bitboard/move-gen design that makes perft exact, and the (explicitly provisional) NN encoding that will be revisited at training time.

---

## The ruleset (FMJD International draughts)

The implementation targets the FMJD International ruleset. Source of truth: `core/src/checkers.rs`.

- **01 — Board.** 10×10, 50 dark playable squares, PDN-numbered 1..50 (internal indices `0..49`). `pub const N: usize = 50;`.
- **02 — Pieces.** 20 men per side at the start. White's 20 men occupy indices `30..=49` (`WHITE_INIT`); Black's occupy `0..=19` (`BLACK_INIT`).
- **03 — Side / direction.** White = `P0` moves first, **up** the board (toward decreasing row, row 0); Black = `P1` moves down (toward row 9). "Up" = decreasing row.
- **04 — Man movement.** Men step **one** diagonal square **forward only** (`WHITE_FWD = [UP_LEFT, UP_RIGHT]`, `BLACK_FWD = [DOWN_LEFT, DOWN_RIGHT]`).
- **05 — Man capture.** Men **capture in all four directions** (forward *and* backward). A man jumps an adjacent enemy with the square immediately beyond empty.
- **06 — Flying kings.** Kings slide and capture **any distance** along a diagonal: scan past empties to the first occupied square; if it is an un-captured enemy with ≥1 empty square beyond, **every** empty landing beyond it is legal.
- **07 — Mandatory capture.** If any capture exists, **only** captures are legal (`legal_moves` returns `captures` when non-empty, else `quiets`).
- **08 — Maximum count.** A capture must take the **maximum number** of pieces. Count only — **no material/majority/value tiebreak**: distinct sequences taking equal counts are *all* legal.
- **09 — Promotion on stop.** A man promotes **only if it stops on its far rank**. Passing *through* the back rank mid-capture does **not** promote (the "passing"/Turkish rule). White promotes on row 0 (`WHITE_PROMO = 0b11111`, indices `0..=4`); Black on row 9 (`BLACK_PROMO = 0b11111 << 45`, indices `45..=49`).
- **10 — Captured pieces stay as blockers.** Jumped pieces remain on the board (blocking landings) until the *whole* move resolves, and **cannot be jumped twice** — tracked in the `captured` mask during generation, removed in one shot in `apply`.
- **11 — No move = loss.** Side to move with no legal move (capture or quiet) **loses**.
- **12 — Draw.** 25 king-only non-capturing moves per side ⇒ draw: `pub const DRAW_PLIES: u16 = 50;` (50 plies).

---

## State: bitboard packing

```rust
pub struct State {
    pub white: u64,   // White occupancy over 50 dark squares
    pub black: u64,   // Black occupancy
    pub kings: u64,   // crowned subset of EITHER colour
    pub stm: Color,   // side to move
    pub idle: u16,    // plies since last capture or man move
}
```

**Decision:** three `u64` occupancy boards over the 50 dark squares plus a colour-agnostic `kings` mask. A man of colour `c` is `color_bb & !kings`; a king is `color_bb & kings`. `occ() = white | black`.

**Why:** keeping `kings` as a subset of occupancy (rather than separate man/king boards per colour) lets `apply` move/crown/capture with plain bit ops and makes the `wm = white & !kings` style splits in eval/encode one AND each.

`idle` is the draw counter: reset to 0 on any capture **or any man move**; incremented only on a quiet king move (the sole remaining case). `Color` is a 2-valued `enum { White, Black }` with `other()`; `color_to_player`/`player_to_color` bridge to the shared `Player` seam (White↔P0, Black↔P1).

---

## Geometry: the `NEIGHBOR[50][4]` table

```rust
static NEIGHBOR: [[i8; 4]; N] = build_neighbors();   // const fn
fn nb(sq: i32, dir: usize) -> i32   // -> neighbour index, or -1 off-board
```

Directions are indexed `0..4`: `UP_LEFT=0, UP_RIGHT=1, DOWN_LEFT=2, DOWN_RIGHT=3`. `NEIGHBOR[i][dir]` is the dark-square index one diagonal step from `i` in `dir`, or `-1` off-board. The table is built at compile time via `const fn build_neighbors()` over the `(row, col)` mapping:

```rust
const fn idx_to_rc(i: usize) -> (i32, i32) {
    let r = (i / 5) as i32;
    let within = (i % 5) as i32;
    let c = if r % 2 == 0 { within * 2 + 1 } else { within * 2 };
    (r, c)
}
```

Five dark squares per row; even rows carry dark squares on odd columns and vice-versa (PDN numbering, square 1 = top row leftmost dark square). `rc_to_idx` is the inverse for in-bounds squares.

**Walking a ray is repeated application of `nb` in the same `dir`** — directions stay consistent step to step, so a flying-king scan is just a `while sq >= 0 { sq = nb(sq, dir); }` loop.

**Why a table over a u128 shift trick:** on a 10×10 dark-square board the index step for a single diagonal move **alternates between 5 and 6 by row parity** (the within-row offset shifts because even/odd rows place dark squares on opposite columns). A single fixed shift mask cannot express that parity-dependent stride; the precomputed table sidesteps the whole problem and turns a step into one array load. It also gives clean `-1` off-board sentinels at the edges.

---

## Move: a multi-jump is ONE move

```rust
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub captured: u64,   // union of EVERY jumped square; 0 == quiet
}
```

**Decision:** an entire multi-jump sequence collapses to a single `Move`. `from` is the origin, `to` the **final** landing, and `captured` is the bitwise union of every square jumped over the whole sequence. A quiet move has `captured == 0` (this is exactly `Game::is_quiet`).

Generation is recursive:
- `extend_man` — for each of the 4 directions, jump an un-captured enemy (`enemy & overbit != 0 && captured & overbit == 0`) when the landing is empty in `blockers`; recurse with `captured | overbit`. When no further jump is possible and `captured != 0`, push the completed `Move`.
- `extend_king` — scan past empties to the first occupied square; if it is an un-captured enemy, **every** empty square beyond branches a `Move` (and recurses for chained jumps).

`blockers` is the original occupancy **minus the origin square**, so re-landing on the vacated origin is allowed; captured pieces stay *in* `blockers` (they block landings) and *in* `captured` (so they cannot be jumped twice) until the move resolves. This directly encodes rule **10**.

---

## The dedup + max-count filter (why perft is exact)

```rust
pub fn captures(s: &State) -> Vec<Move> {
    // ... extend_man / extend_king collect every maximal sequence into `raw` ...
    let maxn = raw.iter().map(|m| m.captured.count_ones()).max().unwrap();
    raw.retain(|m| m.captured.count_ones() == maxn);   // MAX-COUNT (rule 08)
    raw.sort_unstable_by_key(|m| (m.from, m.to, m.captured));
    raw.dedup();                                        // DEDUP by (from,to,captured)
    raw
}
```

Two post-passes after raw collection:

1. **Max-count filter** keeps only sequences capturing `maxn = max(count_ones)` pieces — count only, no tiebreak (rule **08**). Equal-count alternatives all survive (rule: *no value tiebreak*).
2. **Dedup by `(from, to, captured)`.** Distinct jump *orders* that take the **same set** of pieces and end on the **same** landing are the **same legal move** and must be counted once. Sort then `dedup` gives a stable, deterministic order.

**Why it is load-bearing:** without the `(from,to,captured)` dedup, positions reachable by multiple jump orderings inflate the move count, and perft drifts above the published oracle. The d8 case (`6483961`) is the tripwire: the test comment calls it "the 4+-capture dedup tripwire" because that is where multi-order 4-captures first appear in bulk. The published counts only reproduce *with* this filter.

`has_capture` is a cheaper "does any capture exist?" single-jump probe used by terminal detection; `quiets` generates man single-steps-forward and flying-king slides, used only when `captures` is empty.

---

## apply(): promotion-on-stop + idle reset

`State::apply(mv)` (assumes legality):

1. Relocate the moving piece (`white`/`black`, and `kings` if it was already a king).
2. Remove the whole `captured` set in one shot from `white`, `black`, `kings` (they blocked until now — rule **10**).
3. **Promote only if a man `stops` on its far rank:** `if !was_king && (tbit & promo) != 0 { kings |= tbit; }`. A man that *passed through* the back rank mid-capture lands on `to` (off the rank) and is not crowned (rule **09**).
4. **`idle`:** reset to 0 on a capture **or any man move** (`mv.captured != 0 || !was_king`); otherwise (quiet king move) `self.idle + 1`.
5. Flip `stm`.

`null_move()` flips `stm` only (null-move-pruning hook; board untouched).

---

## Terminal, draw, and evaluation

**Terminal / winner** (`Game` impl):
- `is_terminal` ⇔ `s.idle >= DRAW_PLIES || !any_legal_move(s)`.
- `winner`: draw (`None`) when `idle >= DRAW_PLIES`; `None` when a legal move exists; otherwise the side to move has no move and **loses**, so the winner is `color_to_player(s.stm.other())`.
- `any_legal_move = has_capture(s) || has_quiet(s)`.

**Evaluator** `CheckersHeuristic` (negamax-antisymmetric, terminal `±WIN_SCORE`):

| Weight | Default | Env override |
|--------|---------|--------------|
| `man` | 100 | `GB_CK_MAN` |
| `king` | 300 | `GB_CK_KING` |
| `advance` | 4 | `GB_CK_ADV` |
| `back_rank` | 6 | `GB_CK_BACK` |

`white_score(s)` (positive = good for White):
- Material: `±man` per man, `±king` per king.
- **Advancement:** `advance * (9 - r)` per White man (advances toward row 0); `advance * r` per Black man (toward row 9), subtracted.
- **Back-rank guard:** `+back_rank` per White man still on row 9 (`wm & BLACK_PROMO`); `-back_rank` per Black man still on row 0 (`bm & WHITE_PROMO`) — home-rank men deny opponent promotions.

`eval(state, p)` returns `0` on a draw (`idle >= DRAW_PLIES`), `±WIN_SCORE` on no-legal-move (side to move is the loser), else `white_score` flipped to `p`'s view (`P0` = white, `P1` = negated). Weights come from `from_env()` (parity with Wall Chess's `WC_*` knobs).

---

## Search integration

The generic negamax + alpha-beta + transposition table + arena driver are reused unchanged via the `E::G` associated-type seam — **no checkers-specific search code exists**. The per-game `Game` hooks the search relies on:

- `search_moves` / `search_moves_wide` both return the full `legal_moves` set (no interior move pruning yet; alpha-beta does the heavy lifting).
- `capture_moves` returns `captures(s)` — drives **forced-capture quiescence**: empty in a quiet position, so qsearch returns the static eval immediately.
- `is_quiet(mv) == (mv.captured == 0)` — LMR/LMP may reduce quiet moves; **captures are never reduced**.
- `move_order_index(mv) = from*50 + to` (`0..2499`) for history move-ordering (distinct capture targets from the same origin share a slot — fine for ordering).
- `hash` — MurmurHash3-style mix of `white`/`black`/`kings`/`stm`/`idle`, never 0 (returns 1 if the mix is 0).
- `immediate_winning_move` returns `None` (one-ply forced-win detection left to the search).
- `state_key` = `"{white:x}.{black:x}.{kings:x}.{w|b}.{idle}"`; `parse_state_key` is its strict inverse.

The recommended preset is `SearchConfig::draughts()` (PVS + aspiration + LMR + quiescence ON, **null-move OFF**, `WC_QUIESCENCE` env), per the search.rs seam notes — NMP is unsafe in draughts because zugzwang-like forced-capture positions are common.

---

## NN encoding (PROVISIONAL — revisit at training)

> **This entire section is EXPLICITLY PROVISIONAL.** The dims and the action mapping are placeholders chosen to compile and round-trip; they will be reconsidered when training actually starts. The source marks them "provisional NN dims (revisited when training starts)".

| Const | Value | Definition |
|-------|-------|------------|
| `ACTION_COUNT` | 2500 | `N * N` — `from*50 + to` |
| `MOVE_INDEX_SPACE` | 2500 | `N * N` |
| `FEATURE_LEN` | 308 | `6 * N + 8` = 6 planes × 50 + 8 scalars |

**Action space = `from*50 + to` (2500), PROVISIONAL.**
**Why over `from × dir × dist`:** flying-king captures *bend* — a king's path can change direction across a multi-jump, so there is no single `(dir, dist)` that describes a capture move. `from*50 + to` always identifies the endpoints; `captured` is unknown from the index alone (`index_to_move` sets `captured: 0`), so callers must **intersect with the legal set** to recover the full move.

**Feature vector (308), me-frame, PROVISIONAL.** Six 50-wide planes — my men, my kings, opp men, opp kings, all men, all kings — followed by 8 scalars: side-to-move bias (`1.0`), normalized `idle / DRAW_PLIES`, then `me_men/me_kings/opp_men/opp_kings` counts each `/20.0`, then two `0.0` padding slots.

**Me-frame is a 180° board rotation, `i → 49 - i`,** applied to all of `me`/`opp`/`kings` so the side to move always views from the same orientation. `mirror_move` is the matching **involution** (`from → 49-from`, `to → 49-to`, and each captured bit `sq → 49-sq`); applying it twice is the identity.

---

## Perft oracle

`perft(s, depth)` is a pure move-tree enumeration (no draw/terminal pruning).

| Depth | Nodes |
|-------|-------|
| d1 | 9 |
| d2 | 81 |
| d3 | 658 |
| d4 | 4265 |
| d5 | 27117 |
| d6 | 167140 |
| d7 | 1049442 |
| d8 | 6483961 |

Sources (initial-position counts, Ed Gilbert / Feike Boomstra, reproduced by BikDam):
- https://damforum.nl/viewtopic.php?t=2308
- https://aartbik.blogspot.com/2012/10/bikdam-international-checkers.html

**CI runs d1–d6** in `perft_initial_shallow`; **d7 + d8** are in `perft_initial_deep`, marked `#[ignore]` (slow). d8 is "the 4+-capture dedup tripwire". Reproduce:

```sh
cd core && cargo build --release --bin checkers && ./target/release/checkers perft 7
```

(The `checkers` CLI bin exposes `perft`/`bestmove`/`selfplay`. Run the ignored deep perft with `cargo test -- --ignored`.)

---

## Rule tests that landed (`core/src/checkers/tests.rs`)

| Test | Asserts |
|------|---------|
| `perft_initial_shallow` | d1–d6 exact |
| `perft_initial_deep` (`#[ignore]`) | d7=1049442, d8=6483961 |
| `capture_is_mandatory` | a capture exists ⇒ only captures are legal |
| `max_count_filter_drops_shorter` | the 2-chain beats the 1-jump; exactly 1 move, `captured.count_ones()==2` |
| `no_value_tiebreak_between_equal_counts` | two equal-count 1-captures (king + man) both kept |
| `man_captures_backward` | a man jumps an enemy behind it (down-right) |
| `flying_king_distance_and_multiple_landings` | king jumps at distance with 2 distinct landings |
| `king_cannot_cross_two_enemies` | two adjacent victims ⇒ no capture |
| `king_cannot_jump_own_piece` | own piece blocks the ray |
| `man_promotes_on_stopping_back_rank` | quiet step onto row 0 crowns |
| `man_does_not_promote_passing_through_back_rank` | 2-chain touching but not stopping on row 0 ⇒ no crown |
| `no_legal_move_is_a_loss` | boxed-in side to move ⇒ terminal, opponent wins |
| `idle_counter_draws_and_resets` | `idle >= DRAW_PLIES` draws (eval 0); quiet king move +1; man move resets to 0 |
| `eval_is_antisymmetric_on_nonterminal` | `eval(P0) == -eval(P1)`, up-material ⇒ `> 0` |
| `state_key_round_trips` | `parse_state_key(state_key(s)) == s` |
| `action_index_round_trips_endpoints` | `action_index < ACTION_COUNT`; `index_to_move` recovers `(from, to)` |
| `mirror_move_is_an_involution` | double-mirror is identity (incl. `captured`) |
| `encode_has_expected_length` | feature vector length `== FEATURE_LEN` (308) |
| `shared_search_and_arena_play_checkers` | generic `Search`/`play_game` drive checkers with no game-specific search code |
| `random_playout_preserves_occupancy_invariants` | 200 deterministic random playouts: no colour overlap, no king on empty, exact piece-count bookkeeping |

**Result:** rules, terminal/draw, eval antisymmetry, serialization, provisional encoding round-trips, and shared-engine reuse are all covered; perft d1–d6 gate CI, d7/d8 verify the dedup oracle on demand. Full workspace suite: **62 passed / 1 ignored**.
