"""Action/feature layouts that must match the Rust engine.

Wall Chess remains the default because existing datasets, models, and Rust
weights use that contract. Checkers helpers are the trainer-side port of
`impl Encoder for Checkers` in `core/src/checkers.rs`; keep them byte-for-byte
aligned before any draughts training run.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GameSpec:
    id: str
    feature_len: int
    action_count: int


SIZE = 9

# features.rs — 300 legacy scalar/bit features, then two 9×9 BFS distance maps:
# my distance-to-goal map and opponent distance-to-goal map, both in me-frame.
WALLCHESS_FEATURE_LEN = 81 + 81 + 64 + 64 + 3 + 3 + 4 + 81 + 81  # 462

# action.rs
PAWN_ACTIONS = SIZE * SIZE              # 81
WALL_ACTIONS = (SIZE - 1) * (SIZE - 1)  # 64
WALLCHESS_ACTION_COUNT = PAWN_ACTIONS + 2 * WALL_ACTIONS  # 209

CHECKERS_N = 50
CHECKERS_FEATURE_LEN = 6 * CHECKERS_N + 8  # 308
CHECKERS_ACTION_COUNT = CHECKERS_N * CHECKERS_N  # from*50 + to
CHECKERS_DRAW_PLIES = 50

WALLCHESS = GameSpec("wallchess", WALLCHESS_FEATURE_LEN, WALLCHESS_ACTION_COUNT)
CHECKERS = GameSpec("checkers", CHECKERS_FEATURE_LEN, CHECKERS_ACTION_COUNT)
GAMES = {g.id: g for g in (WALLCHESS, CHECKERS)}

# Backwards-compatible defaults used by the existing Wall Chess trainer.
FEATURE_LEN = WALLCHESS.feature_len
ACTION_COUNT = WALLCHESS.action_count


def get_game_spec(game: str) -> GameSpec:
    try:
        return GAMES[game]
    except KeyError as exc:
        raise ValueError(f"unknown game {game!r}; expected one of {sorted(GAMES)}") from exc


def _bitboard(squares: int | list[int] | tuple[int, ...] | set[int]) -> int:
    if isinstance(squares, int):
        return squares
    bb = 0
    for sq in squares:
        if sq < 0 or sq >= CHECKERS_N:
            raise ValueError(f"checkers square out of range: {sq}")
        bb |= 1 << int(sq)
    return bb


def _squares(bb: int) -> list[int]:
    return [i for i in range(CHECKERS_N) if (bb >> i) & 1]


def _checkers_color(stm: str) -> str:
    if stm in ("w", "white"):
        return "white"
    if stm in ("b", "black"):
        return "black"
    raise ValueError(f"bad checkers side to move: {stm!r}")


def _mirror_bits(bb: int) -> int:
    out = 0
    for sq in _squares(bb):
        out |= 1 << (49 - sq)
    return out


def checkers_initial_state() -> dict[str, Any]:
    return {
        "white": list(range(30, 50)),
        "black": list(range(0, 20)),
        "kings": [],
        "stm": "white",
        "idle": 0,
    }


def checkers_state_key(state: dict[str, Any]) -> str:
    white = _bitboard(state["white"])
    black = _bitboard(state["black"])
    kings = _bitboard(state.get("kings", []))
    stm = "w" if _checkers_color(state["stm"]) == "white" else "b"
    idle = int(state.get("idle", 0))
    return f"{white:x}.{black:x}.{kings:x}.{stm}.{idle}"


def checkers_parse_state_key(key: str) -> dict[str, Any] | None:
    parts = key.split(".")
    if len(parts) != 5:
        return None
    white, black, kings, stm, idle = parts
    if stm not in ("w", "b"):
        return None
    try:
        return {
            "white": _squares(int(white, 16)),
            "black": _squares(int(black, 16)),
            "kings": _squares(int(kings, 16)),
            "stm": "white" if stm == "w" else "black",
            "idle": int(idle),
        }
    except ValueError:
        return None


def checkers_encode(state: dict[str, Any]) -> list[float]:
    white = _bitboard(state["white"])
    black = _bitboard(state["black"])
    kings = _bitboard(state.get("kings", []))
    if _checkers_color(state["stm"]) == "white":
        me, opp = white, black
    else:
        me, opp = black, white

    me = _mirror_bits(me)
    opp = _mirror_bits(opp)
    kings = _mirror_bits(kings)

    me_men = me & ~kings
    me_kings = me & kings
    opp_men = opp & ~kings
    opp_kings = opp & kings

    features: list[float] = []
    for plane in (
        me_men,
        me_kings,
        opp_men,
        opp_kings,
        me_men | opp_men,
        me_kings | opp_kings,
    ):
        features.extend(float((plane >> i) & 1) for i in range(CHECKERS_N))

    features.extend(
        [
            1.0,
            float(state.get("idle", 0)) / CHECKERS_DRAW_PLIES,
            me_men.bit_count() / 20.0,
            me_kings.bit_count() / 20.0,
            opp_men.bit_count() / 20.0,
            opp_kings.bit_count() / 20.0,
            0.0,
            0.0,
        ]
    )
    return features


def checkers_action_index(move: dict[str, Any]) -> int:
    return int(move["from"]) * CHECKERS_N + int(move["to"])


def checkers_index_to_move(index: int) -> dict[str, Any]:
    if index < 0 or index >= CHECKERS_ACTION_COUNT:
        raise ValueError(f"checkers action index out of range: {index}")
    return {"from": index // CHECKERS_N, "to": index % CHECKERS_N, "captured": []}


def _captured_bits(captured: int | list[int] | tuple[int, ...] | set[int]) -> int:
    return _bitboard(captured)


def checkers_mirror_move(move: dict[str, Any]) -> dict[str, Any]:
    captured = move.get("captured", [])
    mirrored = _mirror_bits(_captured_bits(captured))
    return {
        "from": 49 - int(move["from"]),
        "to": 49 - int(move["to"]),
        "captured": mirrored if isinstance(captured, int) else _squares(mirrored),
    }
