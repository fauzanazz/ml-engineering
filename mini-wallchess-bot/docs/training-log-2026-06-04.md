# Training Log — 2026-06-04

Goal: Stockfish-style path — make the alpha-beta search insanely efficient, then
land only changes that win head-to-head against the deployed champion. Iterate.

Deployed champion at session start: **D12-V2** = depth-12 iterative deepening with
a 600k-node budget (`webui/src/game/bots.ts`, `nodeLimit: 600_000`). The working-tree
search already had the "A+B+C" stack: array-indexed TT, killers, history, LMR,
BFS-path wall pruning (`search_moves`).

---

## Iteration 1 — PVS + null-move pruning + aspiration windows

### Change (`core/src/search.rs`)
- **PVS (principal variation search):** move 0 searched with the full `[alpha,beta]`
  window; every later move scouted with a null window `[-alpha-1, -alpha]` and
  re-searched at full width only when the scout beats alpha. Applied at root and in
  `negamax`, composed with the existing LMR re-search ladder.
- **Null-move pruning:** before the move loop, if `static_eval >= beta` at
  `depth >= 3`, try `null_move()` (skip the turn) at `depth - 1 - R` with `R = 2`.
  If the opponent still can't pull the score below beta with a free move, the
  position is winning → return beta. Guarded off near terminal scores
  (`beta.abs() < WIN_SCORE/2`) to avoid win-in-N distortion. Sound in a pure race
  game: passing is strictly bad (you lose a tempo), so there is no zugzwang trap.
- **Aspiration windows:** iterative deepening re-roots depth `d>4` inside
  `[score-δ, score+δ]` (δ=50, ~half a path step), doubling δ on fail-high/low and
  widening to the full window after the window blows past WIN_SCORE.
- Added `State::null_move()` in `core/src/state.rs`.

### Verification harness (new, this session)
The arena binary plays one compiled engine against itself at two depths — it can
only measure depth scaling, not whether a code change adds Elo. To compare
new-search vs old-search at *identical* depth+budget I added:
- `core/src/bin/bestmove.rs` — stdin/stdout move engine (`<depth> [node_limit]`).
- `core/src/bin/xmatch.rs` — referee that drives two external engine binaries over
  balanced randomized openings, alternating sides, owning all game logic.

Baseline `bm_old` built by reverting the three search features; `bm_new` is the full
change. Both at `12 600000` (exact champion config).

### Results (new vs old, identical D12 / 600k budget)

| Seed / open | Games | New (A) | Old (B) | Draw | New % |
|-------------|-------|---------|---------|------|-------|
| 42 / open-6 | 100   | 57      | 43      | 0    | 57%   |
| 99 / open-8 | 200   | 100     | 99      | 1    | 50%   |
| **combined**| **300** | **157** | **142** | **1** | **52.5%** |

Combined **52.5% over 300 games — 0.87σ, NOT significant** (p≈0.19). The change is
roughly Elo-neutral, not a proven gain.

**Process note / lesson (re-learned the hard way):** an early read of the seed-99 run
at 81 games showed 44-36 (55%) and I nearly landed on it. The FULL 200 games came back
100-99 — the back half reversed. Stopping a match early because the partial looks good
is exactly the sampling bias the "every change verified by match score" rule exists to
prevent. Always run the pre-committed game count.

Depth-scaling sanity (new search both sides): new d10 vs d6, 100 games = 84% ✓
(confirms depth still scales; says nothing about the PVS/null-move change).

Note: the 80%-arena gate in the goal applies vs *weaker* reference bots (depth gap).
Against an equal depth+budget opponent that already has TT+killers+LMR, ~57% is the
honest ceiling for a single search edge — this is Stockfish-style compounding, not a
handicap match.

### Ablation
`bm_noasp` (PVS + null-move, aspiration removed) vs `bm_old`, seed 99 / open-8, to
test whether aspiration windows help or just add re-search overhead on the noisy
path-race eval.

| Engine vs old | Games | Win % |
|---------------|-------|-------|
| bm_noasp      | 15    | 60% (9-6, sample too small to conclude) |

The ablation was stopped early (CPU contention with the primary match). 15 games is
too few to separate aspiration's contribution from noise; it is at least not visibly
worse than the full bundle, so the full change ships as-is. A clean aspiration ablation
is deferred to a later iteration.

### Decision — NOT LANDED (held for a decisive match)
52.5% / 300 games (0.87σ) does not clear the "proven stronger" bar. The change is sound
(PVS and null-move are correct, node-saving prunings; aspiration is standard) and not
harmful — but not a demonstrated Elo gain either. The seed-42 (57%) vs seed-99 (50%)
split suggests it may help in shallow/wall-heavy openings and be neutral in deeper ones,
but that is post-hoc slicing, not evidence.

Action: WASM reverted to the baseline (pre-change) search until a decisive larger match
(fresh seeds, full pre-committed game count, no early stop) settles whether the change
is +Elo or neutral. The `core/src/search.rs` change stays in the working tree as an
unproven experiment with this log explaining its status.

### Next steps
- Decisive match: ≥300 games, 3+ fresh seeds, both open-6 and open-8, no early stop.
- If still ~50–52%: the bundle is neutral; either drop it or ablate to find the one
  component (likely null-move) that does pay off at the 600k budget and keep only that.
- Aspiration is the prime suspect for adding re-search overhead on the noisy path eval.
- Bigger levers regardless: eval upgrade, wall-threat extension.
