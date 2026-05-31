//! Move generation, BFS distance, and legality — ports the reference TS engine.

use crate::state::{Cell, Move, Orientation, Side, State, Wall, SIZE};

const DIRS: [(i16, i16); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

#[inline]
fn step(c: Cell, dr: i16, dc: i16) -> Cell {
    Cell::new((c.r as i16 + dr) as u8, (c.c as i16 + dc) as u8)
}

/// Legal pawn destinations for `side`: orthogonal step, straight jump over the
/// opponent, or diagonal jump when the straight jump is blocked.
pub fn pawn_moves(state: &State, side: Side) -> Vec<Cell> {
    let pos = state.pawn(side);
    let opp = state.pawn(side.other());
    let mut out: Vec<Cell> = Vec::with_capacity(5);
    let push = |cell: Cell, out: &mut Vec<Cell>| {
        if cell.in_bounds() && !out.contains(&cell) {
            out.push(cell);
        }
    };

    for (dr, dc) in DIRS {
        let s = step(pos, dr, dc);
        if !s.in_bounds() || state.is_blocked(pos, s) {
            continue;
        }
        if s != opp {
            push(s, &mut out);
            continue;
        }
        // opponent on `s` -> try to jump straight over
        let beyond = step(s, dr, dc);
        if beyond.in_bounds() && !state.is_blocked(s, beyond) {
            push(beyond, &mut out);
        } else {
            // straight jump blocked -> diagonal jumps around the opponent
            let perps: [(i16, i16); 2] = if dr == 0 {
                [(1, 0), (-1, 0)]
            } else {
                [(0, 1), (0, -1)]
            };
            for (pr, pc) in perps {
                let diag = step(s, pr, pc);
                if diag.in_bounds() && !state.is_blocked(s, diag) {
                    push(diag, &mut out);
                }
            }
        }
    }
    out
}

/// BFS shortest path length to `goal_row`, ignoring the opponent pawn.
/// Returns `None` if unreachable (used for wall-validity and heuristics).
pub fn distance_to_goal(state: &State, from: Cell, goal_row: u8) -> Option<u16> {
    let key = |c: Cell| (c.r as usize) * 16 + c.c as usize;
    let mut seen = [false; 16 * 16];
    let mut frontier = vec![from];
    seen[key(from)] = true;
    let mut dist: u16 = 0;
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for cell in &frontier {
            if cell.r == goal_row {
                return Some(dist);
            }
            for (dr, dc) in DIRS {
                let n = step(*cell, dr, dc);
                if !n.in_bounds() || state.is_blocked(*cell, n) || seen[key(n)] {
                    continue;
                }
                seen[key(n)] = true;
                next.push(n);
            }
        }
        frontier = next;
        dist += 1;
    }
    None
}

#[inline]
pub fn has_path(state: &State, from: Cell, goal_row: u8) -> bool {
    distance_to_goal(state, from, goal_row).is_some()
}

/// Can `side` legally place `wall`? Mirrors canPlaceWall: bounds, no overlap /
/// crossing, and must not trap either player.
pub fn can_place_wall(state: &State, wall: Wall, side: Side) -> bool {
    if state.walls_left[side.idx()] == 0 {
        return false;
    }
    if wall.r < 1 || wall.r > SIZE - 1 || wall.c < 1 || wall.c > SIZE - 1 {
        return false;
    }
    // An anchor holds at most one wall (h or v) -> identical-centre conflict.
    if state.has_h_wall(wall.r, wall.c) || state.has_v_wall(wall.r, wall.c) {
        return false;
    }
    match wall.o {
        Orientation::H => {
            // two horizontals on the same line overlapping a column
            if state.has_h_wall(wall.r, wall.c.wrapping_sub(1))
                || state.has_h_wall(wall.r, wall.c + 1)
            {
                return false;
            }
        }
        Orientation::V => {
            // two verticals on the same line overlapping a row
            if state.has_v_wall(wall.r.wrapping_sub(1), wall.c)
                || state.has_v_wall(wall.r + 1, wall.c)
            {
                return false;
            }
        }
    }
    // must not trap either player
    let probe = state.apply_wall_unchecked(wall, side);
    has_path(&probe, probe.pawn(Side::South), Side::South.goal_row())
        && has_path(&probe, probe.pawn(Side::North), Side::North.goal_row())
}

impl State {
    /// Place a wall without legality checks (turn unchanged) — for probing.
    pub(crate) fn apply_wall_unchecked(&self, wall: Wall, side: Side) -> State {
        let mut next = *self;
        let bit = 1u64 << (((wall.r - 1) * 8 + (wall.c - 1)) as u64);
        match wall.o {
            Orientation::H => next.h_walls |= bit,
            Orientation::V => next.v_walls |= bit,
        }
        next.walls_left[side.idx()] = next.walls_left[side.idx()].saturating_sub(1);
        next
    }
}

pub fn is_legal(state: &State, mv: Move) -> bool {
    if state.winner.is_some() {
        return false;
    }
    match mv {
        Move::Pawn(to) => pawn_moves(state, state.turn).iter().any(|&c| c == to),
        Move::Wall(w) => can_place_wall(state, w, state.turn),
    }
}

/// All legal moves for the side to move (pawn steps first, then walls).
pub fn legal_moves(state: &State) -> Vec<Move> {
    let mut out: Vec<Move> = pawn_moves(state, state.turn)
        .into_iter()
        .map(Move::Pawn)
        .collect();
    if state.walls_left[state.turn.idx()] > 0 {
        for r in 1..=SIZE - 1 {
            for c in 1..=SIZE - 1 {
                for o in [Orientation::H, Orientation::V] {
                    let w = Wall { r, c, o };
                    if can_place_wall(state, w, state.turn) {
                        out.push(Move::Wall(w));
                    }
                }
            }
        }
    }
    out
}
