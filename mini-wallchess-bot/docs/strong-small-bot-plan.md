# Strong Small Bot Plan

## Goal

Build a Wall Chess bot that is clearly stronger than the depth-2 heuristic while
remaining small and fast enough for browser/WASM play.

The bot should not rely on a large neural net. The target shape is a compact
hybrid:

1. Opening graph: precomputed strong opening states with sparse positional
   encodings and action indices, split-friendly for parallel training/search.
2. Midgame search: small value-policy net inside MCTS, optionally with cached
   opening priors/values when the state is in the graph.
3. Endgame compression: compact winning/race result hints for low-branching
   late states, so the engine does not spend network/search budget rediscovering
   forced races.

## Current Baseline

Committed v2 net (`public/wallnet.safetensors`) is better than v1 but still loses
to the heuristic. The first reproducible gate is now `core/src/bin/net_arena.rs`.

Observed on native release:

```bash
cargo run --release --features net --bin net_arena -- ../webui/public/wallnet.safetensors 6 200 2 140 0
```

Result: net 0 - 6 heuristic, no draws, no race caps.

```bash
cargo run --release --features net --bin net_arena -- ../webui/public/wallnet.safetensors 8 200 2 140 4
```

Result: net 0 - 8 heuristic, one race-scored cap.

## Gates

Do not mark the bot stronger from single-opening anecdotes. A candidate must pass
all of these before replacing the shipped net:

- Strength: positive win rate vs heuristic depth 2 with alternating sides.
- Robustness: positive win rate on randomized opening prefixes.
- Latency: average move inference/search stays browser-playable at the chosen sim
  budget; native release timing from `net_arena` is only a lower bound.
- Size: model plus cache assets remain small enough for lazy browser loading.
- Honesty: report natural wins, race-scored wins, draws, average plies, and
  per-engine timing.

## Data Artifacts

### Opening Graph

`core/src/bin/opening_graph.rs` writes JSONL:

- `meta`: format, feature/action lengths, generation knobs, counts.
- `node`: key, ply, expanded flag, sparse positional encoding `pe`.
- `edge`: from, to, me-frame action index `a`, heuristic score.

Smoke command:

```bash
cargo run --release --bin opening_graph -- /private/tmp/wallchess-opening-smoke.jsonl 3 5 2 250 500
```

This file is meant to be sharded by line ranges or key prefixes for parallel
labeling and training.

### Endgame Compression

Next target: generate compact late-state hints keyed by `state_key`. Start with
race-like states where both players have few or no walls, then expand to low-wall
positions. Store only decisive states and best action/result distance, not full
trees. The consumer should treat this as an exact or high-confidence override
before spending MCTS simulations.

## Next Engineering Steps

1. Add an endgame hint generator and compact reader.
2. Teach `selfplay_data` to seed games from opening graph nodes, not only random
   plies, so training covers hard opening branches in parallel.
3. Add candidate model configs smaller/faster than the current 195k-param MLP
   and measure policy/value loss against arena strength, not loss alone.
4. Integrate opening/endgame hints into net-MCTS selection.
5. Run the full gate matrix and commit only candidates that improve measured
   strength without violating latency/size constraints.
