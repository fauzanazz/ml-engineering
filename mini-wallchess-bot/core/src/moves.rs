//! Move generation, BFS distance, and legality — ports the reference TS engine.

use crate::state::{Cell, Move, Orientation, Side, State, Wall, SIZE};

const DIRS: [(i16, i16); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

// ── Bitboard BFS constants ─────────────────────────────────────────────────
//
// Cell (r, c) (1-indexed) → bit (r-1)*9 + (c-1) in a u128.
// 81 cells fit in bits 0..=80; bits 81..=127 are always 0.
//
// Step masks: BASE_CAN_{UP,DOWN,RIGHT,LEFT} capture board-boundary constraints.
// H/V walls further restrict these per-call in `step_masks()`.

// Row 9 cells: bits 72..80
const ROW9_MASK: u128 = 0x1FF_u128 << 72;
// Row 1 cells: bits 0..8
const ROW1_MASK: u128 = 0x1FF_u128;
// Column 9 cells: bits 8, 17, 26, 35, 44, 53, 62, 71, 80
const COL9_MASK: u128 = (1u128 << 8)
    | (1u128 << 17)
    | (1u128 << 26)
    | (1u128 << 35)
    | (1u128 << 44)
    | (1u128 << 53)
    | (1u128 << 62)
    | (1u128 << 71)
    | (1u128 << 80);
// Column 1 cells: bits 0, 9, 18, 27, 36, 45, 54, 63, 72
const COL1_MASK: u128 = 1u128
    | (1u128 << 9)
    | (1u128 << 18)
    | (1u128 << 27)
    | (1u128 << 36)
    | (1u128 << 45)
    | (1u128 << 54)
    | (1u128 << 63)
    | (1u128 << 72);
const BOARD_MASK: u128 = (1u128 << 81) - 1;
const BASE_CAN_UP: u128 = BOARD_MASK & !ROW9_MASK;
const BASE_CAN_DOWN: u128 = BOARD_MASK & !ROW1_MASK;
const BASE_CAN_RIGHT: u128 = BOARD_MASK & !COL9_MASK;
const BASE_CAN_LEFT: u128 = BOARD_MASK & !COL1_MASK;

/// Compute per-direction step-permission masks from the placed walls.
///
/// H-wall at anchor (r, c) (1-indexed, r ∈ 1..8, c ∈ 1..8):
///   Blocks rows r↔r+1 at columns c and c+1.
///   → cells (r,c) and (r,c+1) can't step up; cells (r+1,c) and (r+1,c+1) can't step down.
///
/// V-wall at anchor (r, c):
///   Blocks cols c↔c+1 at rows r and r+1.
///   → cells (r,c) and (r+1,c) can't step right; cells (r,c+1) and (r+1,c+1) can't step left.
///
/// Anchor bit b → row ar = b/8, col ac = b%8 (0-indexed).
/// Cell bit = ar*9 + ac.
#[inline]
fn step_masks(h_walls: u64, v_walls: u64) -> (u128, u128, u128, u128) {
    let mut can_up = BASE_CAN_UP;
    let mut can_down = BASE_CAN_DOWN;
    let mut can_right = BASE_CAN_RIGHT;
    let mut can_left = BASE_CAN_LEFT;

    let mut hw = h_walls;
    while hw != 0 {
        let b = hw.trailing_zeros() as usize;
        hw &= hw - 1;
        let cb = (b / 8) * 9 + (b % 8); // cell (r,c) index
        can_up &= !(3u128 << cb); // (r,c) and (r,c+1) can't step up
        can_down &= !(3u128 << (cb + 9)); // (r+1,c) and (r+1,c+1) can't step down
    }

    let mut vw = v_walls;
    while vw != 0 {
        let b = vw.trailing_zeros() as usize;
        vw &= vw - 1;
        let cb = (b / 8) * 9 + (b % 8);
        can_right &= !(1u128 << cb) & !(1u128 << (cb + 9)); // (r,c) and (r+1,c)
        can_left &= !(1u128 << (cb + 1)) & !(1u128 << (cb + 10)); // (r,c+1) and (r+1,c+1)
    }

    (can_up, can_down, can_right, can_left)
}

/// Bit index for wall anchor (r,c) — mirrors `anchor_bit` in state.rs.
/// Caller must ensure r,c ∈ 1..=SIZE-1.
#[inline]
fn anchor_bit_local(r: u8, c: u8) -> u64 {
    1u64 << (((r - 1) * 8 + (c - 1)) as u64)
}

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

/// Bitboard BFS: shortest path length to `goal_row`, ignoring the opponent pawn.
/// Returns `None` if unreachable (used for wall-validity and heuristics).
///
/// All frontier cells at one distance are represented as a single u128 bitmask
/// and expanded in ~11 bitwise operations per level — roughly 10× faster than
/// cell-by-cell BFS, with zero heap allocation.
pub fn distance_to_goal(state: &State, from: Cell, goal_row: u8) -> Option<u16> {
    let start = (from.r - 1) as u32 * 9 + (from.c - 1) as u32;
    let goal_mask: u128 = 0x1FF_u128 << ((goal_row - 1) as u32 * 9);

    let mut frontier: u128 = 1u128 << start;
    if frontier & goal_mask != 0 {
        return Some(0);
    }
    let mut seen = frontier;
    let (can_up, can_down, can_right, can_left) = step_masks(state.h_walls, state.v_walls);
    let mut dist: u16 = 0;

    loop {
        let new = (((frontier & can_up) << 9)
            | ((frontier & can_down) >> 9)
            | ((frontier & can_right) << 1)
            | ((frontier & can_left) >> 1))
            & !seen;
        if new == 0 {
            return None;
        }
        dist += 1;
        if new & goal_mask != 0 {
            return Some(dist);
        }
        seen |= new;
        frontier = new;
    }
}

#[inline]
pub fn has_path(state: &State, from: Cell, goal_row: u8) -> bool {
    distance_to_goal(state, from, goal_row).is_some()
}

/// Cheap placement checks (no BFS): walls_left, bounds, centre conflict,
/// overlap/crossing. Does NOT verify the trap-connectivity constraint.
fn wall_placeable_ignoring_trap(state: &State, wall: Wall, side: Side) -> bool {
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
            !state.has_h_wall(wall.r, wall.c.wrapping_sub(1))
                && !state.has_h_wall(wall.r, wall.c + 1)
        }
        Orientation::V => {
            !state.has_v_wall(wall.r.wrapping_sub(1), wall.c)
                && !state.has_v_wall(wall.r + 1, wall.c)
        }
    }
}

#[inline]
fn edge_blockers(prev: Cell, cur: Cell) -> (u64, u64) {
    let dc = cur.c as i16 - prev.c as i16;
    if dc == 0 {
        let r = prev.r.min(cur.r);
        let c = prev.c;
        let mut h = 0;
        if c <= SIZE - 1 {
            h |= anchor_bit_local(r, c);
        }
        if c >= 2 {
            h |= anchor_bit_local(r, c - 1);
        }
        (h, 0)
    } else {
        let r = prev.r;
        let c = prev.c.min(cur.c);
        let mut v = 0;
        if r <= SIZE - 1 {
            v |= anchor_bit_local(r, c);
        }
        if r >= 2 {
            v |= anchor_bit_local(r - 1, c);
        }
        (0, v)
    }
}

/// BFS to `goal_row` with parent tracking; reconstructs one shortest path and
/// returns `(distance, h_blockers, v_blockers)`:
///   • `distance`   — steps to reach `goal_row` (same as `distance_to_goal`).
///   • `h_blockers` — H-wall anchors whose wall would cut an edge on that path.
///   • `v_blockers` — V-wall anchors whose wall would cut an edge on that path.
///
/// Returns `None` if the goal is unreachable.
///
/// Geometry (from `is_blocked`):
///   • H-wall at (r,c) cuts vertical edges (r,c)↔(r+1,c) and (r,c+1)↔(r+1,c+1).
///   • V-wall at (r,c) cuts horizontal edges (r,c)↔(r,c+1) and (r+1,c)↔(r+1,c+1).
/// Reverse map for path-edge → blocking anchors:
///   • vertical edge (r,c)↔(r+1,c)   → H-anchors (r,c) and (r,c-1).
///   • horizontal edge (r,c)↔(r,c+1) → V-anchors (r,c) and (r-1,c).
pub fn path_blockers(state: &State, from: Cell, goal_row: u8) -> Option<(u16, u64, u64)> {
    let key = |c: Cell| (c.r as usize) * 16 + c.c as usize;
    // parent[key(n)] = key(parent of n). Valid keys for in-bounds cells: 17..=153.
    // 255 is the safe sentinel (never a valid in-bounds key).
    let mut parent = [255u8; 16 * 16];
    let mut seen = [false; 16 * 16];
    // Stack-allocated frontier/next — no heap allocation
    let sentinel = Cell::new(0, 0);
    let mut frontier = [sentinel; 81];
    let mut frontier_len = 1usize;
    let mut next = [sentinel; 81];
    frontier[0] = from;
    seen[key(from)] = true;
    let mut goal_cell: Option<Cell> = None;
    let mut dist: u16 = 0;

    'bfs: loop {
        let mut next_len = 0usize;
        for i in 0..frontier_len {
            let cell = frontier[i];
            if cell.r == goal_row {
                goal_cell = Some(cell);
                break 'bfs;
            }
            for (dr, dc) in DIRS {
                let n = step(cell, dr, dc);
                let nk = key(n);
                if !n.in_bounds() || state.is_blocked(cell, n) || seen[nk] {
                    continue;
                }
                seen[nk] = true;
                parent[nk] = key(cell) as u8;
                next[next_len] = n;
                next_len += 1;
            }
        }
        if next_len == 0 {
            break;
        }
        frontier[..next_len].copy_from_slice(&next[..next_len]);
        frontier_len = next_len;
        dist += 1;
    }

    let goal = goal_cell?;
    let mut h_blockers: u64 = 0;
    let mut v_blockers: u64 = 0;

    let mut cur = goal;
    while cur != from {
        let pk = parent[key(cur)] as usize;
        let prev = Cell::new((pk / 16) as u8, (pk % 16) as u8);

        let (h, v) = edge_blockers(prev, cur);
        h_blockers |= h;
        v_blockers |= v;
        cur = prev;
    }

    Some((dist, h_blockers, v_blockers))
}

/// Blocker union for every shortest path to `goal_row`.
/// Search uses this wider set so it can consider walls that cut an alternate
/// equally-short route, not only the arbitrary path reconstructed by BFS.
fn all_shortest_path_blockers(state: &State, from: Cell, goal_row: u8) -> Option<(u16, u64, u64)> {
    let key = |c: Cell| (c.r as usize) * 16 + c.c as usize;
    let sentinel = Cell::new(0, 0);
    let mut dist = [u8::MAX; 16 * 16];
    let mut frontier = [sentinel; 81];
    let mut next = [sentinel; 81];
    let mut frontier_len = 1usize;
    let mut goal_dist: Option<u8> = if from.r == goal_row { Some(0) } else { None };

    frontier[0] = from;
    dist[key(from)] = 0;

    while frontier_len > 0 {
        let mut next_len = 0usize;
        for i in 0..frontier_len {
            let cell = frontier[i];
            let cell_dist = dist[key(cell)];
            if goal_dist.is_some_and(|d| cell_dist >= d) {
                continue;
            }
            for (dr, dc) in DIRS {
                let n = step(cell, dr, dc);
                if !n.in_bounds() || state.is_blocked(cell, n) {
                    continue;
                }
                let nk = key(n);
                let nd = cell_dist + 1;
                if dist[nk] != u8::MAX {
                    continue;
                }
                dist[nk] = nd;
                if n.r == goal_row {
                    goal_dist = Some(goal_dist.map_or(nd, |old| old.min(nd)));
                }
                next[next_len] = n;
                next_len += 1;
            }
        }
        frontier[..next_len].copy_from_slice(&next[..next_len]);
        frontier_len = next_len;
    }

    let goal_dist = goal_dist?;
    let mut stack = [sentinel; 81];
    let mut stack_len = 0usize;
    let mut seen = [false; 16 * 16];
    for c in 1..=SIZE {
        let goal = Cell::new(goal_row, c);
        if dist[key(goal)] == goal_dist {
            stack[stack_len] = goal;
            stack_len += 1;
            seen[key(goal)] = true;
        }
    }

    let mut h_blockers = 0u64;
    let mut v_blockers = 0u64;
    while stack_len > 0 {
        stack_len -= 1;
        let cur = stack[stack_len];
        if cur == from {
            continue;
        }
        let cur_dist = dist[key(cur)];
        for (dr, dc) in DIRS {
            let prev = step(cur, dr, dc);
            if !prev.in_bounds() || state.is_blocked(prev, cur) {
                continue;
            }
            let pk = key(prev);
            if dist[pk].saturating_add(1) != cur_dist {
                continue;
            }
            let (h, v) = edge_blockers(prev, cur);
            h_blockers |= h;
            v_blockers |= v;
            if !seen[pk] {
                seen[pk] = true;
                stack[stack_len] = prev;
                stack_len += 1;
            }
        }
    }

    Some((goal_dist as u16, h_blockers, v_blockers))
}

/// Thin wrapper returning only the blocker bitsets (used in `legal_moves`).
fn shortest_path_blockers(state: &State, from: Cell, goal_row: u8) -> Option<(u64, u64)> {
    path_blockers(state, from, goal_row).map(|(_, h, v)| (h, v))
}

/// Can `side` legally place `wall`? Mirrors canPlaceWall: bounds, no overlap /
/// crossing, and must not trap either player.
pub fn can_place_wall(state: &State, wall: Wall, side: Side) -> bool {
    if !wall_placeable_ignoring_trap(state, wall, side) {
        return false;
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
///
/// Wall legality uses a BFS-pruned fast path: compute each pawn's shortest-path
/// blocker bitsets once per call; a wall whose anchor bit is absent from both
/// pawn's blocker sets cannot disconnect anyone and is accepted without running
/// the full trap-check BFS.  Walls that DO touch a path still run `has_path`
/// only for the touched pawn(s).  Expected ~75-90% of walls skip BFS.
pub fn legal_moves(state: &State) -> Vec<Move> {
    wall_move_gen(state, false, false)
}

/// Candidate moves for search: pawn moves + walls that cut at least one
/// player's current shortest path.  Typically ~20 wall options instead of
/// ~150, giving a large branching-factor reduction at high depths.
/// Non-path walls (cutting neither path) are skipped as they are almost
/// never the best move — they use a wall without gaining distance on either
/// player.  Legality (no-trap) is still checked for both players.
pub fn search_moves(state: &State) -> Vec<Move> {
    wall_move_gen(state, true, false)
}

/// Wider root candidate set: pawn moves + walls that cut any shortest path.
/// Interior search keeps [`search_moves`] because recomputing the full union at
/// every node costs more than it saves.
pub fn search_moves_wide(state: &State) -> Vec<Move> {
    wall_move_gen(state, true, true)
}

fn wall_move_gen(state: &State, path_only: bool, all_shortest: bool) -> Vec<Move> {
    let mut out: Vec<Move> = pawn_moves(state, state.turn)
        .into_iter()
        .map(Move::Pawn)
        .collect();
    if state.walls_left[state.turn.idx()] > 0 {
        // Blocker bitsets: u64::MAX as fallback forces full BFS (safe but slow).
        let (sh, sv) = if path_only && all_shortest {
            all_shortest_path_blockers(state, state.pawn(Side::South), Side::South.goal_row())
                .map(|(_, h, v)| (h, v))
                .unwrap_or((u64::MAX, u64::MAX))
        } else {
            shortest_path_blockers(state, state.pawn(Side::South), Side::South.goal_row())
                .unwrap_or((u64::MAX, u64::MAX))
        };
        let (nh, nv) = if path_only && all_shortest {
            all_shortest_path_blockers(state, state.pawn(Side::North), Side::North.goal_row())
                .map(|(_, h, v)| (h, v))
                .unwrap_or((u64::MAX, u64::MAX))
        } else {
            shortest_path_blockers(state, state.pawn(Side::North), Side::North.goal_row())
                .unwrap_or((u64::MAX, u64::MAX))
        };

        for r in 1..=SIZE - 1 {
            for c in 1..=SIZE - 1 {
                for o in [Orientation::H, Orientation::V] {
                    let w = Wall { r, c, o };
                    if !wall_placeable_ignoring_trap(state, w, state.turn) {
                        continue;
                    }
                    let bit = anchor_bit_local(r, c);
                    let (touches_s, touches_n) = match o {
                        Orientation::H => (sh & bit != 0, nh & bit != 0),
                        Orientation::V => (sv & bit != 0, nv & bit != 0),
                    };

                    if !touches_s && !touches_n {
                        // Wall doesn't cut either shortest path.
                        // search mode: skip (not immediately useful).
                        // full-legal mode: accept without trap-check BFS.
                        if !path_only {
                            out.push(Move::Wall(w));
                        }
                        continue;
                    }

                    let probe = state.apply_wall_unchecked(w, state.turn);
                    let s_ok = !touches_s
                        || has_path(&probe, probe.pawn(Side::South), Side::South.goal_row());
                    let n_ok = !touches_n
                        || has_path(&probe, probe.pawn(Side::North), Side::North.goal_row());
                    if s_ok && n_ok {
                        out.push(Move::Wall(w));
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::State;

    fn o_key(o: Orientation) -> u8 {
        match o {
            Orientation::H => 0,
            Orientation::V => 1,
        }
    }

    #[test]
    fn all_shortest_path_blockers_matches_opening_file() {
        let state = State::initial();
        let (dist, h, v) =
            all_shortest_path_blockers(&state, state.pawn(Side::South), Side::South.goal_row())
                .expect("opening path exists");

        let mut expected_h = 0u64;
        for r in 1..=SIZE - 1 {
            expected_h |= anchor_bit_local(r, 4);
            expected_h |= anchor_bit_local(r, 5);
        }
        assert_eq!(dist, 8);
        assert_eq!(h, expected_h);
        assert_eq!(v, 0);
    }

    #[test]
    fn all_shortest_path_blockers_superset_single_path_blockers() {
        let mut rng: u64 = 0x4d59_5df4_d0f3_3173;
        let mut next_rng = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        for _ in 0..120 {
            let mut state = State::initial();
            for _ in 0..40 {
                for side in [Side::South, Side::North] {
                    let Some((one_dist, one_h, one_v)) =
                        path_blockers(&state, state.pawn(side), side.goal_row())
                    else {
                        continue;
                    };
                    let (all_dist, all_h, all_v) =
                        all_shortest_path_blockers(&state, state.pawn(side), side.goal_row())
                            .expect("single path implies at least one shortest path");
                    assert_eq!(all_dist, one_dist);
                    assert_eq!(all_h & one_h, one_h);
                    assert_eq!(all_v & one_v, one_v);
                }

                if state.winner.is_some() {
                    break;
                }
                let moves = legal_moves(&state);
                if moves.is_empty() {
                    break;
                }
                state = state.apply(moves[(next_rng() as usize) % moves.len()]);
            }
        }
    }

    /// Optimized legal_moves must return the identical wall set as the
    /// reference brute-force can_place_wall scan on many pseudo-random positions.
    #[test]
    fn legal_moves_matches_reference() {
        let mut rng: u64 = 0x9e3779b97f4a7c15;
        let mut next_rng = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..300 {
            let mut state = State::initial();
            for _ in 0..60 {
                if state.winner.is_some() {
                    break;
                }
                // Reference: brute-force can_place_wall for every anchor.
                let mut ref_walls: Vec<Wall> = Vec::new();
                if state.walls_left[state.turn.idx()] > 0 {
                    for r in 1..=SIZE - 1 {
                        for c in 1..=SIZE - 1 {
                            for o in [Orientation::H, Orientation::V] {
                                let w = Wall { r, c, o };
                                if can_place_wall(&state, w, state.turn) {
                                    ref_walls.push(w);
                                }
                            }
                        }
                    }
                }
                // Optimized.
                let moves = legal_moves(&state);
                let mut new_walls: Vec<Wall> = moves
                    .iter()
                    .filter_map(|m| {
                        if let Move::Wall(w) = m {
                            Some(*w)
                        } else {
                            None
                        }
                    })
                    .collect();
                ref_walls.sort_by_key(|w| (w.r, w.c, o_key(w.o)));
                new_walls.sort_by_key(|w| (w.r, w.c, o_key(w.o)));
                assert_eq!(
                    ref_walls, new_walls,
                    "wall set mismatch at state {:?}",
                    state
                );

                if moves.is_empty() {
                    break;
                }
                let mv = moves[(next_rng() as usize) % moves.len()];
                state = state.apply(mv);
            }
        }
    }
}
