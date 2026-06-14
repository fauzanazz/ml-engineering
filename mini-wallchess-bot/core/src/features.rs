//! State → flat feature vector for the value/policy net. MUST match the Python
//! trainer (`trainer/encoding.py`) byte-for-byte in length and field order.
//!
//! Layout (total [`FEATURE_LEN`] = 300), all from the side-to-move's frame so the
//! net sees one consistent "me vs them" view regardless of which colour moves:
//!   - `0..81`    my pawn, one-hot over 81 cells (board mirrored if I am North)
//!   - `81..162`  opponent pawn, one-hot over 81 cells
//!   - `162..226` horizontal walls, 64 bits (mirrored to my frame)
//!   - `226..290` vertical walls, 64 bits
//!   - `290`      my walls left / 10
//!   - `291`      opponent walls left / 10
//!   - `292`      constant 1.0 (bias / "side to move" marker, always me)
//!   - `293`      my shortest-path distance to goal / 16
//!   - `294`      opponent shortest-path distance to goal / 16
//!   - `295`      race margin (opp_dist - my_dist) / 16  (positive ⇒ I lead)
//!   - `296..300` 4 progress flags: for each me-frame direction
//!                [toward-goal, away, right, left], 1.0 iff a legal pawn step
//!                that way strictly reduces my distance-to-goal. Gives the MLP
//!                the path gradient directly instead of forcing it to recompute
//!                shortest paths from one-hot cells (the old finishing weakness).

use crate::moves::distance_to_goal;
use crate::state::{Cell, Move, Side, State, Wall, SIZE};

pub const FEATURE_LEN: usize = 81 + 81 + 64 + 64 + 3 + 3 + 4; // 300

/// BFS distance normalizer. Path lengths run ~8..~30; /16 keeps the common range
/// in [0,1] without clamping the rare long detour.
const DIST_NORM: f32 = 16.0;
/// Fallback distance for the (rule-impossible) trapped state, kept finite so the
/// feature stays bounded if a malformed position is ever encoded.
const UNREACHABLE_DIST: u16 = 40;

/// The four me-frame step directions `[toward-goal, away, right, left]` expressed
/// as absolute `(dr, dc)` deltas. North's me-frame is a 180° rotation, so its
/// "toward-goal" is absolute `-r` and "right" is absolute `-c`.
#[inline]
fn me_frame_dirs(side: Side) -> [(i16, i16); 4] {
    match side {
        Side::South => [(1, 0), (-1, 0), (0, 1), (0, -1)],
        Side::North => [(-1, 0), (1, 0), (0, -1), (0, 1)],
    }
}

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

    // Path features. `distance_to_goal` is frame-agnostic (a scalar BFS length),
    // so the raw value is correct without mirroring.
    let me_pos = state.pawn(me);
    let me_dist = distance_to_goal(state, me_pos, me.goal_row()).unwrap_or(UNREACHABLE_DIST);
    let opp_dist =
        distance_to_goal(state, state.pawn(opp), opp.goal_row()).unwrap_or(UNREACHABLE_DIST);
    f[293] = me_dist as f32 / DIST_NORM;
    f[294] = opp_dist as f32 / DIST_NORM;
    f[295] = (opp_dist as f32 - me_dist as f32) / DIST_NORM;

    // For each me-frame direction, flag a legal single step that strictly cuts my
    // distance-to-goal. Simple adjacency (matches the BFS step model); pawn jumps
    // over the opponent are rare and already covered by the global distance.
    for (i, (dr, dc)) in me_frame_dirs(me).into_iter().enumerate() {
        let n = Cell::new((me_pos.r as i16 + dr) as u8, (me_pos.c as i16 + dc) as u8);
        if n.in_bounds() && !state.is_blocked(me_pos, n) {
            if let Some(nd) = distance_to_goal(state, n, me.goal_row()) {
                if nd < me_dist {
                    f[296 + i] = 1.0;
                }
            }
        }
    }
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
        // start: both pawns 8 steps from goal, race even.
        assert_eq!(f[293], 8.0 / 16.0);
        assert_eq!(f[294], 8.0 / 16.0);
        assert_eq!(f[295], 0.0);
        // only the toward-goal step makes progress at the start cell.
        assert_eq!(f[296], 1.0);
        assert_eq!(&f[297..300], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn me_frame_is_side_invariant_at_start() {
        // Initial position is symmetric, so South-to-move and North-to-move (same
        // board, flipped turn) must produce identical me-frame encodings.
        let mut s = State::initial();
        let fs = encode(&s);
        s.turn = Side::North;
        let fnorth = encode(&s);
        assert_eq!(
            fs, fnorth,
            "symmetric start must look identical to both sides"
        );
    }
}
