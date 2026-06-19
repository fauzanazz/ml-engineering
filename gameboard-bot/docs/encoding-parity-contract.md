# Encoding parity contract

> Status: Wall Chess parity LANDED (deployed). International Draughts encoding PROVISIONAL — Rust-only; Python/TS sides NOT yet written.
> Goal: keep the NN feature vector and the action index byte-identical across Rust, the Python trainer, and the TS webui, per game — so a net trained on one encoding is scored on the same encoding everywhere.

---

## Why byte-identity is a contract, not a convention

The net is a pure function of an `f32` vector. Three independent reimplementations produce that vector:

- **Rust** — `core/src/features.rs::encode` + `core/src/action.rs` (Wall Chess); `impl Encoder for Checkers` in `core/src/checkers.rs` (draughts). This is what the search and the webui inference path consume.
- **Python trainer** — `trainer/encoding.py` (referenced from the Rust headers as the byte-for-byte counterpart). This is what produces the weights.
- **TS webui** — the browser inference path that feeds positions to the exported net.

If any field shifts by one slot, changes scale, or mirrors differently, **nothing crashes**. The net silently reads garbage in the moved channels and emits a plausible-but-wrong score. There is no exception, no log line — only a weaker bot. Hence parity is enforced by structural guards (below), not by trust.

The frame convention is the most error-prone part: both games encode from the **side-to-move's "me vs them" view**, so a reimplementation that forgets to mirror the non-moving side produces a vector that looks valid and scores wrong only for one colour.

---

## Per-game encoding inventory

| Game | Rust source | `FEATURE_LEN` | `ACTION_COUNT` | `MOVE_INDEX_SPACE` | Status |
|---|---|---|---|---|---|
| Wall Chess | `core/src/features.rs`, `core/src/action.rs` | **300** | **209** | 384 | **LANDED** — deployed contract, Python + TS parity exist |
| International Draughts | `core/src/checkers.rs` (`impl Encoder`) | **308** | **2500** | 2500 | **PROVISIONAL** — Rust-only; Python/TS NOT written |

`ACTION_COUNT` is the policy-head width (one logit per action). `MOVE_INDEX_SPACE` is the move-ordering history-table width and may differ from `ACTION_COUNT` (Wall Chess: 384 vs 209). For draughts both are `N*N = 2500`.

### Wall Chess feature layout (`FEATURE_LEN = 300`)

`81 + 81 + 64 + 64 + 3 + 3 + 4`, all in the me-frame (board mirrored if the side to move is North):

| Range | Field | Notes |
|---|---|---|
| `0..81` | my pawn, one-hot over 81 cells | index `(r-1)*9 + (c-1)` |
| `81..162` | opponent pawn, one-hot over 81 cells | |
| `162..226` | horizontal walls, 64 bits | 8×8 anchor grid, mirrored to me-frame |
| `226..290` | vertical walls, 64 bits | |
| `290` | my walls left / 10 | |
| `291` | opponent walls left / 10 | |
| `292` | constant `1.0` | bias / side-to-move marker |
| `293` | my shortest-path distance to goal / 16 | `DIST_NORM = 16.0` |
| `294` | opponent shortest-path distance to goal / 16 | |
| `295` | race margin `(opp_dist - my_dist) / 16` | positive ⇒ I lead |
| `296..300` | 4 progress flags `[toward-goal, away, right, left]` | `1.0` iff a legal pawn step that way strictly cuts my distance |

### Wall Chess action layout (`ACTION_COUNT = 209`)

`81 + 64 + 64` (`PAWN_ACTIONS + 2*WALL_ACTIONS`):

| Range | Action | Index formula |
|---|---|---|
| `0..81` | pawn destination cell | `(r-1)*9 + (c-1)`, r,c ∈ 1..=9 |
| `81..145` | horizontal wall | `81 + (r-1)*8 + (c-1)`, r,c ∈ 1..=8 |
| `145..209` | vertical wall | `145 + (r-1)*8 + (c-1)`, r,c ∈ 1..=8 |

### Draughts feature layout (`FEATURE_LEN = 308`, PROVISIONAL)

`6 * N + 8` = six 50-wide planes over the dark squares + 8 scalars. Built in the me-frame by rotating the board 180° (`i → 49-i`) so the side to move always views from the same orientation:

| Range | Plane / scalar |
|---|---|
| `0..50` | my men |
| `50..100` | my kings |
| `100..150` | opp men |
| `150..200` | opp kings |
| `200..250` | all men (`me_men \| opp_men`) |
| `250..300` | all kings (`me_kings \| opp_kings`) |
| `300` | side-to-move bias, constant `1.0` |
| `301` | `idle / DRAW_PLIES` (= `idle / 50`) |
| `302` | my men count / 20 |
| `303` | my kings count / 20 |
| `304` | opp men count / 20 |
| `305` | opp kings count / 20 |
| `306` | `0.0` (reserved pad) |
| `307` | `0.0` (reserved pad) |

**Note:** the brief described the tail as 8 scalars; the code emits exactly 8 (`300..308`), of which the last two are reserved `0.0` pads. Any Python/TS port must reproduce both pad slots verbatim — they count toward `FEATURE_LEN`.

### Draughts action layout (`ACTION_COUNT = 2500`, PROVISIONAL)

`from*50 + to`, range `0..2500`. `index_to_move` recovers `from`/`to` but **not** `captured` (the index discards it); callers must intersect the decoded `(from, to)` with the legal set to recover the captured mask. Distinct capture targets that share a `(from, to)` collapse to one slot.

---

## Parity guards and the failure mode each catches

The guards below are the only thing that converts a silent mis-score into a hard test failure. The Rust tests that exist **today** are Wall Chess's (`core/src/action.rs`, `core/src/features.rs` test modules) plus checkers `perft`.

| Guard | Property checked | Failure mode it catches |
|---|---|---|
| **action_index round-trip** | `index_to_move(action_index(mv)) == mv` (endpoints preserved) for every legal move | A shifted base offset or wrong stride in the policy layout — would map a logit to the wrong move. |
| **mirror_move involution** | `mirror_move(side, mirror_move(side, mv)) == mv` | A me-frame mirror that is not self-inverse — would desync the encoded board from the decoded action for one colour. |
| **state_key round-trip** | `parse_state_key(state_key(s)) == s` | A lossy or ambiguous canonical key — corrupts graph dedup, opening books, and UI/engine position parity. |
| **PERFT** | move-tree node counts match a known oracle (language-independent) | Any move-generation divergence between Rust and a reimplementation — the encoding is meaningless if the legal-move set itself differs. The fidelity backstop for checkers. |

### action_index round-trip

- **Wall Chess:** `roundtrip_all_legal_moves` in `core/src/action.rs` walks 40 plies of a real game and asserts `index_to_move(action_index(mv)) == mv` and `i < ACTION_COUNT` for every legal move; `action_count_is_209` pins the width.
- **Draughts:** `action_index`/`index_to_move` exist (`from*50+to`), but no cross-language round-trip test exists yet. Endpoints round-trip; `captured` does **not** (by design — see above), so the test must assert endpoint equality after a legal-set intersection, not raw `Move` equality.

### mirror_move involution

- **Wall Chess:** `mirror_move` is documented as an involution (the same op applied with the same side decodes it). `me_frame_is_side_invariant_at_start` checks the symmetric start encodes identically for both sides — a proxy that catches a broken mirror.
- **Draughts:** `mirror_move` is the 180° rotation `i ↔ 49-i` on both endpoints and the captured mask; self-inverse by construction. No explicit involution test yet.

### state_key round-trip

- **Wall Chess:** seam method `state_key` / `parse_state_key`.
- **Draughts:** `state_key` = `"{white:x}.{black:x}.{kings:x}.{stm}.{idle}"` (hex bitboards, `stm ∈ {w,b}`, decimal `idle`); `parse_state_key` rejects extra fields. **Note:** the brief omitted `idle` and the hex format — the code includes both, and `idle` is load-bearing (it drives the draw counter and is part of the position identity). Any port must preserve the exact `.`-delimited hex format.

### PERFT (checkers move-gen oracle)

`perft(s, depth)` in `core/src/checkers.rs` is a pure legal-sequence enumeration (no draw/terminal pruning). Verified EXACT d1–d8 against the known International-draughts node counts. This is the language-independent oracle: before trusting any draughts encoding port, prove the porting language reproduces these perft numbers, because the action space is only meaningful over a matching legal-move set.

---

## Honest status: what must be added when checkers training starts

Only the **Rust** side of the checkers encoder exists. The Python trainer and TS webui have **no** draughts feature/action code, and the cross-language parity tests today are **wallchess-only**. Before draughts training:

1. **Port `impl Encoder for Checkers::encode` to `trainer/encoding.py`** — all 308 fields including the two reserved `0.0` pads, the me-frame 180° rotation, and the exact plane order (my men / my kings / opp men / opp kings / all men / all kings).
2. **Port the action layout** (`from*50 + to`) to Python and TS; add a round-trip test asserting endpoint preservation after legal-set intersection (raw `Move` equality fails because `captured` is dropped).
3. **Add a mirror_move involution test** for draughts in Rust and the port (`i ↔ 49-i` on both endpoints + captured mask).
4. **Add a state_key round-trip test** for the `white.black.kings.stm.idle` hex format, including the extra-field rejection.
5. **Cross-language perft check** — reproduce the Rust d1–d8 perft counts in any language that generates draughts moves, as the move-gen oracle underpinning the action space.
6. **Promote `FEATURE_LEN=308` / `ACTION_COUNT=2500` from PROVISIONAL to LANDED** only once a generated golden vector matches byte-for-byte across Rust ↔ Python ↔ TS. The dims are marked provisional in the code (`const FEATURE_LEN`, `const ACTION_COUNT` carry "revisited when training starts") and may change on first real training.
