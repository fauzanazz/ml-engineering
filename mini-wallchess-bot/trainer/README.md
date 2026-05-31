# Wall Chess self-play training loop

Replaces the depth-2 negamax teacher (a permanent ceiling) with MCTS self-play:
search quality scales with simulations + a learned policy, so the net can
surpass the heuristic instead of regressing to it.

```
 ┌─ selfplay_data ──► selfplay.jsonl ──► train.py ──► wallnet.safetensors ─┐
 │  (Rust, MCTS)        (state, π, z)     (PyTorch)      (candle weights)   │
 └──────────────────  next iteration uses the new net  ◄────────────────────┘
```

Each JSONL record is one move from a self-play game, in the side-to-move
**me-frame**:

- `f`  — feature vector, length `FEATURE_LEN` (293). See `encoding.py` /
  `core/src/features.rs`.
- `pi` — MCTS visit-count policy, sparse `[[action_index, prob], ...]` over the
  `ACTION_COUNT` (209) action space. See `core/src/action.rs`.
- `z`  — game outcome from that state's POV: `+1` win, `-1` loss, `0` draw.

The Rust action/feature layouts in `core/src/{action,features}.rs` and the
Python copies in `encoding.py` **must stay identical**; the net reads garbage
otherwise.

## One iteration

```bash
# 0. build the Rust tools
cd core
cargo build --release --bin selfplay_data                 # heuristic bootstrap
cargo build --release --bin expert_data                   # fast alpha-beta traces
cargo build --release --bin search_data                   # supervised search labels
cargo build --release --features net --bin selfplay_data   # net-driven (iter ≥1)
cargo build --release --features net --bin bestmove_net

# 1. generate self-play data
#    iter 0 — heuristic bootstrap (no weights arg):
./target/release/selfplay_data 200 200 ../selfplay.jsonl
#    iter ≥1 — drive self-play with the previous net:
./target/release/selfplay_data 200 200 ../selfplay.jsonl ../wallnet.safetensors
#    optional — seed games from a precomputed opening graph:
OPENING_GRAPH=../opening_graph.jsonl ./target/release/selfplay_data 200 200 ../selfplay.jsonl
#    optional — make heuristic bootstrap labels stronger but slower:
HEURISTIC_DEPTH=2 OPENING_GRAPH=../opening_graph.jsonl ./target/release/selfplay_data 50 80 ../selfplay-hd2.jsonl

#    fast baseline-imitation data from exact alpha-beta traces:
OPENING_GRAPH=../opening_graph.jsonl ./target/release/expert_data 100 2 ../expert-depth2.jsonl 140

#    stronger local teacher labels over pruned candidate moves:
OPENING_GRAPH=../opening_graph.jsonl ./target/release/search_data 1000 3 ../search-depth3.jsonl 0 350 70 8

# 2. train (uv manages the env)
cd ../trainer
uv run train.py --data ../selfplay.jsonl --out ../wallnet.safetensors --epochs 20
# smaller/faster candidate:
uv run train.py --data ../selfplay.jsonl --out ../wallnet-h128.safetensors --epochs 20 --hidden 128
# mixed-data candidate:
uv run train.py --data ../expert-depth2.jsonl ../selfplay.jsonl ../search-depth3.jsonl --out ../wallnet-h128-mix.safetensors --epochs 28 --hidden 128

# 3. sanity-check the net inside MCTS
cd ../core
cargo run --release --features net --bin bestmove_net -- ../wallnet.safetensors 400

# 4. go to step 1 with the new weights. Repeat.
```

`bestmove_net` prints the top moves by visit count — eyeball that the net plays
sensibly before spending compute on the next data round.

## Notes / next steps

- The bootstrap heuristic always lets South win, so iter-0 value targets are
  near-degenerate (`z≈+1`). Diversity appears once the net drives self-play
  (games start drawing / both sides win) — that is the loop working.
- `win_prob`'s `k` (display calibration in `core/src/eval.rs`) can be fit to the
  `z` outcomes here once enough games accumulate.
- `OPENING_GRAPH=/path/to/opening_graph.jsonl` makes `selfplay_data` sample start
  states from `core/src/bin/opening_graph.rs` node records instead of uniform
  random opening plies. Use this to parallelize and rebalance hard opening
  branches.
- `expert_data` is the cheap baseline-imitation stage: one alpha-beta move per
  turn, final race-scored `z`. It is useful when MCTS bootstrap is too expensive
  at higher heuristic depths.
- `search_data` emits supervised labels from a stronger search over a pruned
  candidate set. It is much cheaper than scoring every legal wall at depth 3+,
  but still should be sharded with `SEARCH_DATA_SEED` for larger runs.
- `train.py --data` accepts multiple JSONL files and concatenates them in memory,
  so shards can be mixed without creating temporary merged files.
- `--hidden` controls the MLP width. The Rust `NetEvaluator` infers the width
  from safetensors shapes, so candidates smaller than the default 256-hidden net
  can be evaluated without a Rust/WASM architecture edit.
- wasm wiring: `NetEvaluator::from_buffer` already takes raw safetensors bytes
  (no filesystem), so exposing a `wasm-bindgen` entry that accepts the fetched
  buffer + runs MCTS is the remaining browser integration step.
