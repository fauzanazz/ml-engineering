# Gameboard trainer

Python training utilities for policy/value nets that consume Rust-emitted JSONL.

Wall Chess remains the only complete training loop. International Draughts now has trainer-side encoding helpers and dimensional contracts, but no data emitter or learned evaluator yet.

## Data contract

Each JSONL record is one move from a self-play or search-labelled game in side-to-move me-frame:

- `f`: dense feature vector.
- `pi`: sparse policy target as `[[action_index, probability], ...]`.
- `z`: game outcome from that state's side-to-move point of view: `+1`, `-1`, or `0`.

Current specs in `encoding.py`:

| Game | `--game` | Feature length | Action count | Status |
| --- | --- | ---: | ---: | --- |
| Wall Chess | `wallchess` | 462 | 209 | Spatial CNN loop landed. |
| International Draughts | `checkers` | 308 | 2500 | Encoder/action trainer helpers landed; data generation and NN inference deferred. |

The Rust layouts and Python copies must stay identical. A mismatch silently trains weights that score the wrong moves.

## Wall Chess loop

```bash
# 0. build Rust data tools
cd ../core
cargo build --release --bin selfplay_data
cargo build --release --bin expert_data
cargo build --release --bin search_data
cargo build --release --bin counter_book
cargo build --release --features net --bin selfplay_data
cargo build --release --features net --bin bestmove_net

# 1. generate data
./target/release/selfplay_data 200 200 ../selfplay.jsonl
OPENING_GRAPH=../opening_graph.jsonl ./target/release/expert_data 100 2 ../expert-depth2.jsonl 140
OPENING_GRAPH=../opening_graph.jsonl ./target/release/search_data 1000 3 ../search-depth3.jsonl 0 350 70 8

# 2. train
cd ../trainer
uv run train.py --game wallchess --data ../selfplay.jsonl --out ../wallnet.safetensors --epochs 20
uv run train.py --game wallchess --data ../expert-depth2.jsonl ../selfplay.jsonl ../search-depth3.jsonl --out ../wallnet-h128-mix.safetensors --epochs 28 --hidden 128

# 3. inspect inside Rust MCTS
cd ../core
cargo run --release --features net --bin bestmove_net -- ../wallnet.safetensors 400
```

`train.py --data` accepts multiple JSONL files and concatenates them in memory. `--value-data` adds value-only rows with policy loss disabled.

## Autonomous Wall Chess CNN loop

```bash
uv run python autoloop.py
```

Default mode runs forever and resumes from `../runs/wallchess-cnn-loop/state.json`.
Each iteration builds data from heuristic alpha-beta/search labels plus
previous-net MCTS self-play, trains a CNN policy+value model
(`--policy-loss-weight 0.25` by default), gates it with MCTS `net_arena`,
and atomically promotes only passing candidates to
`../runs/wallchess-cnn-loop/best/current.safetensors`.

Each candidate warm-starts from the previous candidate (or promoted best model) and trains on the current data plus the recent replay window (`--replay-iters`, default 8), so learning compounds without lowering the promotion gate.

Data generation is parallelized across search/expert/self-play jobs; self-play and arena gates are sharded by `--selfplay-shards` and `--arena-shards` without changing total game counts.

## Checkers readiness

Available now:

- `CHECKERS = GameSpec("checkers", feature_len=308, action_count=2500)`.
- `checkers_encode(state)` mirrors `impl Encoder for Checkers`.
- `checkers_action_index(move)` and `checkers_index_to_move(index)` use `from*50 + to`.
- `checkers_mirror_move(move)` mirrors endpoints and captured squares with `i ↔ 49-i`.
- `checkers_state_key(state)` / `checkers_parse_state_key(key)` use `white.black.kings.stm.idle` hex format.
- `SelfPlayDataset(..., spec=CHECKERS)` validates draughts feature/action dimensions.
- `train.py --game checkers --arch mlp ...` can train from compatible JSONL once a data emitter exists.

Deferred before a real draughts NN:

1. Emit checkers training JSONL from Rust search/self-play.
2. Add a Rust/WASM learned-evaluator path for checkers.
3. Promote the Rust `FEATURE_LEN=308` / `ACTION_COUNT=2500` comments from provisional after golden-vector parity is generated from real data.

## Focused self-checks

```bash
uv run python checkers_selfcheck.py
```

This checks the trainer-side checkers encoding contract without running training.
