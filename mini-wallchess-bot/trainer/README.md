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
cargo build --release --features net --bin selfplay_data   # net-driven (iter ≥1)
cargo build --release --features net --bin bestmove_net

# 1. generate self-play data
#    iter 0 — heuristic bootstrap (no weights arg):
./target/release/selfplay_data 200 200 ../selfplay.jsonl
#    iter ≥1 — drive self-play with the previous net:
./target/release/selfplay_data 200 200 ../selfplay.jsonl ../wallnet.safetensors

# 2. train (uv manages the env)
cd ../trainer
uv run train.py --data ../selfplay.jsonl --out ../wallnet.safetensors --epochs 20

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
- wasm wiring: `NetEvaluator::from_buffer` already takes raw safetensors bytes
  (no filesystem), so exposing a `wasm-bindgen` entry that accepts the fetched
  buffer + runs MCTS is the remaining browser integration step.
