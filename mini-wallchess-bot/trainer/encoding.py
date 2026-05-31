"""Action/feature layout — MUST match the Rust engine.

Mirror of `core/src/action.rs` and `core/src/features.rs`. If you change a
constant here, change it there (and vice-versa) or the net will read garbage.
"""

SIZE = 9

# features.rs
FEATURE_LEN = 81 + 81 + 64 + 64 + 3  # 293

# action.rs
PAWN_ACTIONS = SIZE * SIZE              # 81
WALL_ACTIONS = (SIZE - 1) * (SIZE - 1)  # 64
ACTION_COUNT = PAWN_ACTIONS + 2 * WALL_ACTIONS  # 209
