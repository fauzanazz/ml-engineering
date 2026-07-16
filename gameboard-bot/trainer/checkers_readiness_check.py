"""Checkers trainer-encoding readiness self-check.

Assert-based sanity pass over `encoding.py`'s checkers helpers and
`SelfPlayDataset`'s multi-game `spec=` support — the trainer-side half of the
parity guards in `../docs/encoding-parity-contract.md` (state_key round-trip
incl. extra-field rejection, action_index round-trip, mirror_move involution)
plus the feature/action dimensions the checkers `GameSpec` promises.

Usage: uv run python checkers_readiness_check.py
"""

import json
import tempfile
from pathlib import Path

from encoding import (
    CHECKERS,
    WALLCHESS,
    checkers_action_index,
    checkers_encode,
    checkers_index_to_move,
    checkers_initial_state,
    checkers_mirror_move,
    checkers_parse_state_key,
    checkers_state_key,
    get_game_spec,
)

INITIAL_KEY = "3ffffc0000000.fffff.0.w.0"


def check_game_spec():
    assert get_game_spec("checkers") is CHECKERS
    assert get_game_spec("wallchess") is WALLCHESS
    assert CHECKERS.feature_len == 308, CHECKERS.feature_len
    assert CHECKERS.action_count == 2500, CHECKERS.action_count


def check_initial_state_key():
    key = checkers_state_key(checkers_initial_state())
    assert key == INITIAL_KEY, f"initial state key {key!r} != {INITIAL_KEY!r}"


def check_state_key_round_trip():
    states = [
        checkers_initial_state(),
        {"white": [22, 30], "black": [5, 13], "kings": [22], "stm": "black", "idle": 7},
    ]
    for state in states:
        key = checkers_state_key(state)
        parsed = checkers_parse_state_key(key)
        assert parsed is not None, f"failed to parse {key!r}"
        assert checkers_state_key(parsed) == key, f"round trip drifted for {key!r}"

    # A trailing field beyond white.black.kings.stm.idle must be rejected, not
    # silently ignored — a lossy/ambiguous key corrupts graph dedup and books.
    extra = checkers_state_key(checkers_initial_state()) + ".7"
    assert checkers_parse_state_key(extra) is None, "extra-field key was accepted"


def check_encode_shape_and_initial_invariants():
    f = checkers_encode(checkers_initial_state())
    assert len(f) == CHECKERS.feature_len, f"feature len {len(f)} != {CHECKERS.feature_len}"

    # Plane order: my men / my kings / opp men / opp kings / all men / all
    # kings, 50-wide each. White to move at the start: 20 men per side, no
    # kings yet.
    planes = [sum(f[i : i + 50]) for i in range(0, 300, 50)]
    assert planes == [20.0, 0.0, 20.0, 0.0, 40.0, 0.0], f"plane sums {planes}"

    # Scalar tail: bias, idle/DRAW_PLIES, my-men/20, my-kings/20, opp-men/20,
    # opp-kings/20, then two reserved 0.0 pads.
    assert f[300:308] == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], f"tail {f[300:308]}"


def check_action_index_round_trip():
    for i in range(CHECKERS.action_count):
        mv = checkers_index_to_move(i)
        assert checkers_action_index(mv) == i, f"round trip broke at index {i}: {mv}"

    for frm, to in ((0, 0), (0, 49), (49, 0), (49, 49), (13, 27)):
        idx = frm * 50 + to
        assert checkers_action_index({"from": frm, "to": to}) == idx
        mv = checkers_index_to_move(idx)
        assert (mv["from"], mv["to"]) == (frm, to), mv


def check_mirror_move_involution():
    # captured as a list of squares.
    mv = {"from": 13, "to": 11, "captured": [12]}
    once = checkers_mirror_move(mv)
    assert once == {"from": 36, "to": 38, "captured": [37]}, once
    twice = checkers_mirror_move(once)
    assert twice == mv, f"involution failed: {mv} -> {once} -> {twice}"

    # captured as an int bitmask.
    mv_bb = {"from": 5, "to": 9, "captured": 1 << 7}
    once_bb = checkers_mirror_move(mv_bb)
    twice_bb = checkers_mirror_move(once_bb)
    assert twice_bb == mv_bb, f"bitmask involution failed: {mv_bb} -> {once_bb} -> {twice_bb}"

    # quiet move (no captured key at all) must round-trip too.
    quiet = {"from": 30, "to": 25}
    twice_quiet = checkers_mirror_move(checkers_mirror_move(quiet))
    assert twice_quiet == {"from": 30, "to": 25, "captured": []}, twice_quiet


def check_selfplay_dataset_accepts_checkers_spec():
    from dataset import SelfPlayDataset  # deferred: pulls in torch

    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "one.jsonl"
        good.write_text(
            json.dumps({"f": [0.0] * CHECKERS.feature_len, "pi": [[0, 1.0]], "z": 1.0}) + "\n"
        )
        ds = SelfPlayDataset(str(good), spec=CHECKERS)
        assert len(ds) == 1, len(ds)
        feat, pol, val, _pw = ds[0]
        assert tuple(feat.shape) == (CHECKERS.feature_len,), feat.shape
        assert tuple(pol.shape) == (CHECKERS.action_count,), pol.shape
        assert val.item() == 1.0, val.item()

        # A record shaped for wallchess (462 features) must be rejected under
        # the checkers spec instead of silently loading garbage.
        bad = Path(tmp) / "bad.jsonl"
        bad.write_text(
            json.dumps({"f": [0.0] * WALLCHESS.feature_len, "pi": [[0, 1.0]], "z": 1.0}) + "\n"
        )
        try:
            SelfPlayDataset(str(bad), spec=CHECKERS)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "SelfPlayDataset accepted a wallchess-length feature vector under spec=CHECKERS"
            )


CHECKS = [
    check_game_spec,
    check_initial_state_key,
    check_state_key_round_trip,
    check_encode_shape_and_initial_invariants,
    check_action_index_round_trip,
    check_mirror_move_involution,
    check_selfplay_dataset_accepts_checkers_spec,
]


if __name__ == "__main__":
    for check in CHECKS:
        check()
        print(f"ok  {check.__name__}")
    print(f"\n{len(CHECKS)} checks passed — checkers trainer encoding is training-ready.")
