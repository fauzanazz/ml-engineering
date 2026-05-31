//! Canonical action space: a fixed, dense index for every possible move so a
//! policy head can emit one logit per action. MUST stay in lock-step with the
//! Python trainer (`trainer/encoding.py`) — same layout, same size.
//!
//! Layout (total [`ACTION_COUNT`] = 209):
//!   - `0..81`    pawn destination cell, index `(r-1)*9 + (c-1)`, r,c in 1..=9
//!   - `81..145`  horizontal wall, index `81 + (r-1)*8 + (c-1)`, r,c in 1..=8
//!   - `145..209` vertical wall,   index `145 + (r-1)*8 + (c-1)`, r,c in 1..=8

use crate::state::{Cell, Move, Orientation, Wall, SIZE};

pub const PAWN_ACTIONS: usize = (SIZE as usize) * (SIZE as usize); // 81
pub const WALL_ACTIONS: usize = ((SIZE - 1) as usize) * ((SIZE - 1) as usize); // 64
pub const ACTION_COUNT: usize = PAWN_ACTIONS + 2 * WALL_ACTIONS; // 209

const H_BASE: usize = PAWN_ACTIONS; // 81
const V_BASE: usize = PAWN_ACTIONS + WALL_ACTIONS; // 145

/// Dense index for a move. Inputs are assumed in-bounds (legal moves always are).
pub fn action_index(mv: Move) -> usize {
    match mv {
        Move::Pawn(c) => ((c.r - 1) as usize) * (SIZE as usize) + (c.c - 1) as usize,
        Move::Wall(w) => {
            let off = ((w.r - 1) as usize) * ((SIZE - 1) as usize) + (w.c - 1) as usize;
            match w.o {
                Orientation::H => H_BASE + off,
                Orientation::V => V_BASE + off,
            }
        }
    }
}

/// Inverse of [`action_index`]. Returns the move an index denotes (no legality
/// check — the caller intersects with `legal_moves`).
pub fn index_to_move(i: usize) -> Move {
    if i < PAWN_ACTIONS {
        let r = (i / (SIZE as usize)) as u8 + 1;
        let c = (i % (SIZE as usize)) as u8 + 1;
        Move::Pawn(Cell::new(r, c))
    } else {
        let (base, o) = if i < V_BASE {
            (H_BASE, Orientation::H)
        } else {
            (V_BASE, Orientation::V)
        };
        let off = i - base;
        let r = (off / ((SIZE - 1) as usize)) as u8 + 1;
        let c = (off % ((SIZE - 1) as usize)) as u8 + 1;
        Move::Wall(Wall { r, c, o })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::legal_moves;
    use crate::state::State;

    #[test]
    fn roundtrip_all_legal_moves() {
        // Every legal move on a busy mid-game position round-trips through the
        // index and lands back on itself, all indices inside the action space.
        let mut state = State::initial();
        let mut rng: u64 = 0xdead_beef_cafe_f00d;
        for _ in 0..40 {
            if state.winner.is_some() {
                break;
            }
            for mv in legal_moves(&state) {
                let i = action_index(mv);
                assert!(i < ACTION_COUNT, "index {i} out of range for {mv:?}");
                assert_eq!(index_to_move(i), mv, "roundtrip failed for {mv:?}");
            }
            let moves = legal_moves(&state);
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            state = state.apply(moves[(rng as usize) % moves.len()]);
        }
    }

    #[test]
    fn action_count_is_209() {
        assert_eq!(ACTION_COUNT, 209);
    }
}
