# Refactor risk and determinism

> Status: LANDED. Multi-game refactor complete; golden gate green on every phase.
> Goal: prove the multi-game refactor did not change a single move the deployed Wall Chess bot plays, and document the methodology that guarantees it.

---

## Why determinism is the contract

The deployed Gen-2 Wall Chess bot is a pure function of the position. Its search
(`core/src/search.rs`) has **no source of nondeterminism**:

- **No RNG.** Move ordering is TT-move → killers → history (`history` table),
  all derived from the search itself. The only `xorshift64` in the codebase is in
  the golden test's position generator, never in the engine.
- **No threads.** Search is single-threaded; arena parallelism (`xargs -P`) runs
  whole games as separate processes, never inside one search.
- **No clock.** Depth is fixed (no time-based cutoff in the deployed path).
- **State/Move are `Copy`**, tables are const-sized per game, and the search
  monomorphizes per game — so a given `(state, depth, eval, config)` always
  yields the same `(best, score)`, byte for byte.

This is the whole reason a refactor of this size is *safe to attempt*: if the
output is a deterministic function of the inputs, then "did not change behaviour"
is a decidable property, checkable by a frozen-output gate. The refactor was a
seam-insertion exercise (introduce the `Game`/`Evaluator`/`Encoder` traits,
genericize `Search` over `E::G`, rename the crate, add checkers) — none of which
*should* touch what Wall Chess plays. The gate turns "should" into "did".

**Decision:** treat any golden mismatch as a refactor bug, never as an
acceptable drift. The only sanctioned way to change the fixture is an
*intentional* behaviour change with `REGEN_GOLDEN=1`.

---

## The golden-snapshot gate

Source: `core/tests/golden.rs`. Fixture: `core/tests/golden_snapshot.txt`
(committed). Single test: `deployed_search_is_byte_identical`.

### Corpus — 24 deterministic positions

```rust
const DEPTH: u8 = 6;
const N_POSITIONS: usize = 24;
```

`corpus()` seeds `xorshift64` with `0xC0FFEE_1234_5678` and plays random legal
playouts of length `4 + (next() % 22)` plies from `State::initial()`, discarding
any line where someone already won. Position 0 is always the initial state. The
seed is fixed, so the 24-position corpus is identical on every machine and every
run — no fixture drift from the generator itself.

### Both engines scored at depth 6

Each `snapshot_line()` scores the position with **two** engines and writes both:

| Engine | Eval | Config | Mirrors |
|--------|------|--------|---------|
| Default | `Heuristic::default()` | `SearchConfig::default()` (LMR + TT on, all newer prunings off) | the previously-deployed baseline search |
| **Gen-2 (deployed)** | `Heuristic::new(50, 120, true)` | `SearchConfig::aggressive()` | the wasm `*_gen2` path |

```rust
let g2_eval = Heuristic::new(50, 120, true);   // w_path=50, w_wall=120, exact endgame
let g2_cfg  = SearchConfig::aggressive();      // pvs+nmp+rfp+razor+futility+lmp+lmr+asp
let g2 = Search::with_config(&g2_eval, g2_cfg).search(state, DEPTH);
```

One fixture line is tab-separated:

```
state_key \t default.best \t default.score \t gen2.best \t gen2.score
```

First fixture lines (`core/tests/golden_snapshot.txt`):

```
s|15|95|1010|	P25	50	P25	50
s|16|95|87|h18,h38,v47,v66,h85	P26	150	P26	170
```

Note line 5/6 carry the exact-endgame Gen-2 scores (`998993` / `-998993`,
i.e. `ENDGAME_WIN`-band) where the default eval reads only `±350` / `±50` — the
fixture captures the Gen-2 endgame divergence, so a regression in *either*
engine trips the gate.

### Fail-loud on a missing or empty fixture

The test reads the committed fixture and asserts it is non-empty before
comparing; an absent file `expect()`s with the regen hint, an empty file fails
the `assert!(!expected.is_empty(), ...)`. It then compares **line by line**, so a
mismatch names the exact position index, and round-trips each `state_key`
through `parse_state_key` so a parser regression also trips here.

```rust
assert_eq!(got, want, "GOLDEN MISMATCH at position {i}\n  got:  {got}\n  want: {want}");
```

### Regenerating (intentional changes only)

```bash
REGEN_GOLDEN=1 cargo test --test golden
```

Only ever run this when you *mean* to change what the bot plays. Never to paper
over a refactor regression — that defeats the entire gate.

---

## Per-phase verdicts

Every phase below was validated by the same gate plus the full suite. The full
suite is **62 passed / 1 ignored**; the golden snapshot is **byte-identical**
after each phase.

| Phase | What changed | Golden | Suite | Verdict |
|-------|--------------|:------:|:-----:|---------|
| 0 — gate first | Add `tests/golden.rs` + commit fixture before touching code | baseline | 62/1 | **LANDED** — frozen baseline established |
| 1 — seam | Add `game.rs`: `Game`/`Evaluator`/`Encoder` traits, `Player`, `WIN_SCORE`/`ENDGAME_WIN` | byte-identical | 62/1 | **LANDED** — additive only |
| 2 — genericize | `Search<'a, E: Evaluator>` over `E::G`; `TtSlot<M>`; history → `Vec<i32>`; `arena.rs` generic + `Outcome` | byte-identical | 62/1 | **LANDED** — monomorphizes per game |
| 3 — per-game glue | `wallchess.rs` (`impl Game` + `impl Encoder`), hoist `state_hash`→`hash`, `history_idx`→`move_order_index`, `immediate_winning_move` | byte-identical | 62/1 | **LANDED**; games/ subdir nesting **DEFERRED** (golden-guarded file-move churn only) |
| 4 — crate rename | `wallchess-core`→`gameboard-core`, lib `wallchess_core`→`gameboard_core` across Cargo.toml/bins/tests/wasm | byte-identical | 62/1 | **LANDED**; wasm package name stays `wallchess-wasm` (per-game bundles **DEFERRED**) |
| 5 — checkers | `checkers.rs` + tests; `SearchConfig::draughts()`; `quiescence` flag + `qsearch()`; new `bin/checkers.rs` | byte-identical | 62/1 | **LANDED**; perft d1–8 EXACT |

**Why byte-identical holds through Phase 2's generics:** `Search` keeps one type
param and resolves its game via the `E::G` associated type, so every existing
`Search<'_, Heuristic>` caller compiles unchanged. State/Move are `Copy`, tables
are const-sized, and the search monomorphizes per game — there is no dynamic
dispatch on the hot path to perturb ordering or timing-sensitive behaviour.

---

## The three subtle changes — why each is provably inert

These are the only edits in the refactor that *could*, in principle, have
perturbed Wall Chess output. Each is inert, and the golden gate confirms it.

### 1. History table: fixed array → `Vec<i32>`

The history heuristic table moved from a fixed `[i32; N]` to:

```rust
// core/src/search.rs
history: Vec<i32>,
// allocated:
history: vec![0; <Gm<E> as Game>::MOVE_INDEX_SPACE],
```

**Why:** an associated-const array length (`Gm<E>::MOVE_INDEX_SPACE`) is not
nameable as an array length in generic Rust. **Inert because:** the length is
identical (`MOVE_INDEX_SPACE`, =384 for Wall Chess), every slot is zero-init, and
indexing is unchanged (`move_order_index(mv)`). The only difference is heap vs
stack storage — values written, read, and ordered are bit-for-bit the same.

### 2. Quiescence flag — default OFF, Wall Chess never enters qsearch

`SearchConfig::quiescence: bool` defaults to `false`
(`SearchConfig::default()` and therefore `aggressive()` via `..default()`). At a
leaf:

```rust
if depth == 0 {
    return if self.config.quiescence {
        self.qsearch(state, alpha, beta, ply)
    } else {
        self.eval.eval(state, <Gm<E> as Game>::turn(state))
    };
}
```

**Inert because:** the deployed Gen-2 config (`aggressive()`) leaves
`quiescence: false`, so the leaf takes the *exact same* `eval.eval(...)` branch
as before. Even if it were on, `qsearch` is driven by `Game::capture_moves`,
whose **default impl returns an empty `Vec`** — Wall Chess has no captures, so
qsearch would immediately resolve to the static eval. Two independent reasons it
cannot move a Wall Chess line. (Draughts opts in via `SearchConfig::draughts()`,
which is the only consumer of the new path.)

### 3. `derive(Hash)` added to the Wall Chess `Move`

The `Game` trait requires `Move: Hash`, so `#[derive(Hash)]` was added to the
Wall Chess `Move`.

**Inert because:** the engine's transposition table is array-indexed by
`Game::hash(state)` (the hoisted, byte-identical `state_hash`), **not** by hashing
`Move`. The `Hash` impl is a compile-time obligation only — nothing in the
deployed search path calls it. Runtime behaviour is unchanged.

---

## Operational rules (do not break the gate from the outside)

These come from hard-won incidents; the gate cannot catch them because they
corrupt the experiment, not the source.

- **Never `cargo build` mid-match.** A rebuild swaps the `bestmove` binary while a
  match is running and silently corrupts results — the search is deterministic, so
  always verify reproducibility instead of rebuilding. Build *before* the match,
  not during.
- **Run full game counts; no early stop.** A favourable partial read is sampling
  bias, not a result. (Historical example: a PVS change looked like 55% at 81
  games and settled to 50% at 200.) Let the full count finish before judging.

---

## Reproduce the gate

```bash
cd core && cargo test --test golden
```

Green = the deployed Wall Chess bot plays the identical move and score it played
before the refactor, for all 24 frozen positions, under both the default and the
deployed Gen-2 engine. Red names the exact position that drifted.
