# International Draughts Gen-1 bot

> Status: LANDED. Gen-1 = default hand evaluator + draughts search; validated;
> harness `core/src/bin/checkers_arena.rs` committed. No neural net.
> Goal: establish and validate the first-generation checkers bot (search + hand
> eval), pick a deploy depth, and record — with data — why **no weight tweak in
> the 4-parameter eval space beats the default at the depth that ships**.

---

## What Gen-1 is

A pure search + hand-evaluation bot on the shared `gameboard-core` engine — the
same negamax/alpha-beta the deployed Wall Chess Gen-2 bot uses, reached through
the `Game`/`Evaluator` seam with zero checkers-specific search code.

| Component | Gen-1 setting | Source |
|-----------|---------------|--------|
| Evaluator | `CheckersHeuristic` default — man **100**, king **300**, advance **4**, back_rank **6** | `core/src/checkers.rs` |
| Search    | `SearchConfig::draughts()` — PVS + aspiration + LMR + quiescence ON, null-move OFF | `core/src/search.rs` |
| Deploy depth | **d10** (recommended); d8 low-latency; d12 expert | this doc |
| Eval terms | material (man/king) + advancement + back-rank guard | `white_score()` |

`king/man = 3.0` matches the conventional draughts king≈3-men valuation and the
arena's material tiebreak. Weights are overridable at runtime via
`GB_CK_MAN`/`GB_CK_KING`/`GB_CK_ADV`/`GB_CK_BACK` (parity with Wall Chess's
`WC_*` knobs).

---

## Methodology — the validation harness

`core/src/bin/checkers_arena.rs`. The shared `arena::play_match` always starts
from `State::initial()`, and the search is **deterministic** — so it would
replay one identical game N times. The harness fixes this and adds the pieces a
heuristic-bot shoot-out needs:

- **Random openings.** Each game plays a short random opening (default 6 plies)
  before the bots take over, so a match is a genuine sample of distinct games.
- **Seeded + reproducible.** Every game is a pure function of a `SplitMix`-spread
  seed; re-running the same `(seed, games)` reproduces the result bit-for-bit.
- **Parallel-safe.** Games are independent, so they fan out across threads; the
  aggregate is identical regardless of thread timing.
- **Seat-balanced.** A and B swap White/Black every game to cancel the
  first-move advantage.
- **Full counts, no early stop** ([[feedback_no_early_stop_arena]]): a favourable
  partial read is sampling bias. Every table below is the full game count.
- **Random opponent.** A uniform-random legal mover — not expressible as an
  `Evaluator`, so it can't ride the generic match loop; the harness special-cases
  it as the sanity floor.

Score% = win + ½·draw (draughts is draw-heavy; ply-cap overruns are broken by
weighted material, king≈3 men, so the sample isn't drowned in cap-draws).

---

## Step 1 — Sanity floor: crush a random mover

```bash
checkers_arena vsrandom 8 200 1 6
```

| Bot | Opponent | Score | Record |
|-----|----------|------:|--------|
| d8  | uniform-random | **100.0%** | 200W-0L-0D / 200 |

**Result:** total. The engine never loses a game to random play. Floor cleared.

---

## Step 2 — Depth is the strength lever

Deeper search vs shallower, **identical default weights**, random openings.

```bash
checkers_arena ladder 160 1 6              # d8/d6/d4 rungs
checkers_arena duel 100 300 4 6 10  100 300 4 6 8  120 1 6   # d10 vs d8
checkers_arena duel 100 300 4 6 12  100 300 4 6 10 120 1 6   # d12 vs d10
```

| Matchup | Score (deeper) | Record | Games |
|---------|---------------:|--------|------:|
| d4 vs d2  | 87.5% | 133W-13L-14D | 160 |
| d6 vs d4  | 85.3% | 126W-13L-21D | 160 |
| d8 vs d6  | 80.6% | 117W-19L-24D | 160 |
| d10 vs d8 | ~72.5% | (s1 70.8% · s2 74.2%) | 120×2 |
| d12 vs d10 | ~70.3% | (s1 73.8% · s2 66.7%) | 120×2 |

**Result:** strictly monotonic — every extra 2 plies wins ~70–87% of games. The
search is the dominant strength lever, and it keeps paying off well past d8.
This is the load-bearing evidence that the generic engine is sound on a second
game.

---

## Step 3 — Weight sweep, and the overfit lesson

Each candidate vs the default `100/300/4/6` baseline, `man` fixed at 100 (unit).

```bash
checkers_arena tune 6 160 1 6                      # seed-1 sweep
checkers_arena duel <cand> 6  100 300 4 6 6  160 <seed> 6   # cross-seed
```

Seed-1 sweep (`tune`, d6) flagged two apparent winners — `back20` at **61.9%**
and `adv8` at 52.8%. Both **evaporated under cross-seed validation** (3 seeds,
480 games each):

| Candidate (vs default, d6) | seed1 | aggregate 3-seed | every seed ≥50%? |
|----------------------------|------:|-----------------:|:----------------:|
| `adv8`        (100/300/8/6)  | 52.8% | 49.4% | no |
| `back20`      (100/300/4/20) | **61.9%** | 51.6% | no |
| `back28`      (100/300/4/28) | 57.2% | 51.6% | no |
| `adv8+back20` (100/300/8/20) | 52.8% | 48.5% | no |
| `adv8+back28` (100/300/8/28) | 51.9% | **52.8%** | **yes** (51.9/54.7/51.9) |

`adv8+back28` was the lone candidate positive on every seed (and +55.1% over a
fresh 400-game seed at d6) — so it looked like the Gen-1 pick. Then it was tested
at the depth that ships:

| `adv8+back28` vs default | seed1 | seed2 | seed3 |
|--------------------------|------:|------:|------:|
| **at d8 (deploy depth)** | 47.5% | 45.0% | 44.7% |

**The d6 gain reverses at d8.** The advance/back-rank bias that nudges a shallow
search toward sound shapes actively *misleads* a deeper one, which already sees
those consequences in its tree and is dragged off-line by the inflated term.

**Decision:** keep the default `100/300/4/6`. In this 4-parameter space no weight
change survives at deploy depth; the apparent wins were a shallow-depth +
single-seed artifact. The honest Gen-1 eval is the clean default — and the clear
path to a stronger bot is **depth and richer eval terms (Gen-2)**, not retuning
these four.

---

## Step 4 — Move-time and deploy depth

Average over a full self-play game (`checkers selfplay D`, both sides):

| Depth | Self-play game | Avg / move | vs next-shallower |
|------:|---------------|-----------:|------------------:|
| d8  | 87 plies / 195 ms  | ~2.2 ms | 80.6% (vs d6) |
| d10 | 196 plies / 2.85 s | ~14.5 ms | ~72.5% (vs d8) |
| d12 | 115 plies / 6.12 s | ~53 ms  | ~70.3% (vs d10) |

All comfortably interactive on a single core (worst-case tactical nodes run
higher than the average, but stay well within a snappy UI budget through d10).

**Decision — deploy depth d10:** a ~72% edge over d8 for ~15 ms/move average.
d8 is the ultra-low-latency fallback; d12 is an "expert" tier (stronger still,
~50 ms/move) — mirroring the Wall Chess default-depth + expert-tier pattern
([[project_search_optimization]]).

---

## Reproduce

```bash
cd core
cargo build --release --bin checkers_arena --bin checkers   # build BEFORE matches
./target/release/checkers_arena vsrandom 8 200 1 6          # sanity floor
./target/release/checkers_arena ladder 160 1 6             # depth ladder
./target/release/checkers_arena tune 6 160 1 6            # weight sweep
# cross-seed a candidate (A) vs default (B): man king adv back depth ×2, games seed open
./target/release/checkers_arena duel 100 300 8 28 8  100 300 4 6 8  160 2 6
```

Never `cargo build` mid-match — a rebuild swaps the binary and silently corrupts
results ([[feedback_no_rebuild_during_matches]]); the search is deterministic, so
verify reproducibility instead of rebuilding.

---

## Limitations / what Gen-2 would add

Gen-1 is a deliberately minimal material+advancement+back-rank eval. It has **no**
mobility, centre control, king safety/activity, tempo, formation (bridge/phalanx),
or endgame knowledge — the flat weight space in Step 3 is a direct symptom. A
checkers Gen-2 would add those terms and re-tune at deploy depth; a checkers
*neural* bot (the deferred path) would need the training stack — selfplay
data-gen → encoder parity (the provisional 308-feature / 2500-action encoding) →
trainer GameSpec → candle inference — none of which exists yet
([[project_gameboard_multigame]]).
