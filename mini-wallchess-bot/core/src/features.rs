//! State → flat feature vector for the value/policy net. MUST match the Python
//! trainer (`trainer/encoding.py`) byte-for-byte in length and field order.
//!
//! Layout (total [`FEATURE_LEN`] = 293), all from the side-to-move's frame so the
//! net sees one consistent "me vs them" view regardless of which colour moves:
//!   - `0..81`    my pawn, one-hot over 81 cells (board mirrored if I am North)
//!   - `81..162`  opponent pawn, one-hot over 81 cells
//!   - `162..226` horizontal walls, 64 bits (mirrored to my frame)
//!   - `226..290` vertical walls, 64 bits
//!   - `290`      my walls left / 10
//!   - `291`      opponent walls left / 10
//!   - `292`      constant 1.0 (bias / "side to move" marker, always me)

use crate::state::{Cell, Move, Side, State, Wall, SIZE};

pub const FEATURE_LEN: usize = 81 + 81 + 64 + 64 + 3;

#[inline]
fn cell_index(r: u8, c: u8) -> usize {
    ((r - 1) as usize) * (SIZE as usize) + (c - 1) as usize
}

/// Mirror a cell so the side-to-move always advances "up the board" (toward
/// increasing row). South already does; North's board is flipped vertically and
/// horizontally so its goal looks like South's.
#[inline]
fn to_me_frame(side: Side, cell: Cell) -> Cell {
    match side {
        Side::South => cell,
        Side::North => Cell::new(SIZE + 1 - cell.r, SIZE + 1 - cell.c),
    }
}

/// Map an absolute-frame move into the side-to-move's me-frame, so action
/// indices line up with the me-frame [`encode`]. Identity for South. Negation of
/// this (decode) is the same op applied with the same side — it is an involution.
pub fn mirror_move(side: Side, mv: Move) -> Move {
    match side {
        Side::South => mv,
        Side::North => match mv {
            Move::Pawn(c) => Move::Pawn(Cell::new(SIZE + 1 - c.r, SIZE + 1 - c.c)),
            Move::Wall(w) => Move::Wall(Wall {
                r: SIZE - w.r,
                c: SIZE - w.c,
                o: w.o,
            }),
        },
    }
}

/// Encode `state` from the side-to-move's perspective into [`FEATURE_LEN`] f32s.
pub fn encode(state: &State) -> Vec<f32> {
    let mut f = vec![0.0f32; FEATURE_LEN];
    let me = state.turn;
    let opp = me.other();

    let mp = to_me_frame(me, state.pawn(me));
    let op = to_me_frame(me, state.pawn(opp));
    f[cell_index(mp.r, mp.c)] = 1.0;
    f[81 + cell_index(op.r, op.c)] = 1.0;

    // Walls live on the 8x8 anchor grid. Under the North mirror an anchor (r,c)
    // maps to (SIZE-r, SIZE-c) and its orientation is preserved (H stays H, V
    // stays V — a 180° rotation keeps axis alignment).
    for r in 1..=(SIZE - 1) {
        for c in 1..=(SIZE - 1) {
            let (ar, ac) = match me {
                Side::South => (r, c),
                Side::North => (SIZE - r, SIZE - c),
            };
            let bit = ((ar - 1) as usize) * ((SIZE - 1) as usize) + (ac - 1) as usize;
            if state.has_h_wall(r, c) {
                f[162 + bit] = 1.0;
            }
            if state.has_v_wall(r, c) {
                f[226 + bit] = 1.0;
            }
        }
    }

    f[290] = state.walls_left[me.idx()] as f32 / 10.0;
    f[291] = state.walls_left[opp.idx()] as f32 / 10.0;
    f[292] = 1.0;
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_and_initial_symmetry() {
        let s = State::initial();
        let f = encode(&s);
        assert_eq!(f.len(), FEATURE_LEN);
        // exactly two pawns set, equal walls
        let pawn_bits: f32 = f[0..162].iter().sum();
        assert_eq!(pawn_bits, 2.0);
        assert_eq!(f[290], 1.0);
        assert_eq!(f[291], 1.0);
    }

    #[test]
    fn me_frame_is_side_invariant_at_start() {
        // Initial position is symmetric, so South-to-move and North-to-move (same
        // board, flipped turn) must produce identical me-frame encodings.
        let mut s = State::initial();
        let fs = encode(&s);
        s.turn = Side::North;
        let fnorth = encode(&s);
        assert_eq!(fs, fnorth, "symmetric start must look identical to both sides");
    }
}
