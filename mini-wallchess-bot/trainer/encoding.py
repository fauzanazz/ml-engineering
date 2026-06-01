"""Action/feature layout — MUST match the Rust engine.

Mirror of `core/src/action.rs` and `core/src/features.rs`. If you change a
constant here, change it there (and vice-versa) or the net will read garbage.
"""

SIZE = 9

# features.rs — pawns + walls + walls-left/bias (293), then path features:
# my_dist/16, opp_dist/16, race_margin/16, and 4 me-frame progress flags.
FEATURE_LEN = 81 + 81 + 64 + 64 + 3 + 3 + 4  # 300

# action.rs
PAWN_ACTIONS = SIZE * SIZE              # 81
WALL_ACTIONS = (SIZE - 1) * (SIZE - 1)  # 64
ACTION_COUNT = PAWN_ACTIONS + 2 * WALL_ACTIONS  # 209
