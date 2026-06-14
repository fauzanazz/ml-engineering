//! Negamax + alpha-beta with iterative deepening. The transposition table is
//! a fixed-size array indexed by a cheap multiplicative hash — much faster than
//! HashMap for the tight inner loop (no pointer-chasing, no SipHash overhead).
//!
//! Move ordering (priority, highest first):
//!   1. TT move from prior iteration.
//!   2. Killer moves — quiet moves that caused a beta cutoff at this ply.
//!   3. History heuristic — moves that caused cutoffs in other nodes.
//!
//! Late-move reductions (LMR): wall moves past the first FULL_DEPTH_MOVES are
//! searched at depth-2 first; re-searched at full depth only if they beat alpha.

use crate::eval::{Evaluator, WIN_SCORE};
use crate::moves::{legal_moves, pawn_moves, search_moves, search_moves_wide};
use crate::state::{Move, Orientation, State};

// ── constants ─────────────────────────────────────────────────────────────────

const MAX_PLY: usize = 64; // search tree depth ceiling
const FULL_DEPTH_MOVES: usize = 4; // first N moves always searched at full depth
const LMR_MIN_DEPTH: u8 = 4; // don't reduce at depth < this
const LMR_REDUCTION: u8 = 1; // extra plies below (depth-1) for reduced search

// TT: 2^18 = 262 144 slots × ~24 bytes ≈ 6 MB per Search instance.
const TT_BITS: usize = 18;
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: usize = TT_SIZE - 1;

// ── TT ────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Default)]
enum Bound {
    Exact,
    Lower,
    #[default]
    Upper,
}

/// One slot in the array-indexed transposition table.
/// `depth == 0` is the empty/unoccupied sentinel.
#[derive(Clone, Copy, Default)]
struct TtSlot {
    key: u64,  // state hash; 0 means empty
    depth: u8, // 0 means empty
    score: i32,
    bound: Bound,
    best: Option<Move>,
}

/// Fast 64-bit hash of a State for array-TT indexing.
/// Mixes h_walls, v_walls, pawn positions, wall counts, and turn.
/// Never returns 0 (0 is the empty-slot sentinel).
#[inline]
fn state_hash(s: &State) -> u64 {
    let mut h = s.h_walls;
    h ^= s.v_walls.wrapping_mul(0x9e3779b97f4a7c15u64);
    h ^= (s.pawns[0].r as u64 * 10 + s.pawns[0].c as u64).wrapping_mul(0x517cc1b727220a95u64) << 32;
    h ^= (s.pawns[1].r as u64 * 10 + s.pawns[1].c as u64).wrapping_mul(0xbf58476d1ce4e5b9u64) << 16;
    h ^= (s.walls_left[0] as u64) << 56;
    h ^= (s.walls_left[1] as u64) << 48;
    h ^= (s.turn as u64) << 63;
    // finalizer for good bit distribution
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9u64);
    h ^= h >> 27;
    if h == 0 {
        1
    } else {
        h
    }
}

// ── History index ─────────────────────────────────────────────────────────────

/// Compact index into the history table.
/// Pawn(to)       → 0..=153   (to.r * 16 + to.c)
/// Wall(r,c,H)    → 256..=319
/// Wall(r,c,V)    → 320..=383
const HISTORY_SIZE: usize = 384;

#[inline]
fn history_idx(mv: Move) -> usize {
    match mv {
        Move::Pawn(c) => c.r as usize * 16 + c.c as usize,
        Move::Wall(w) => match w.o {
            Orientation::H => 256 + (w.r as usize - 1) * 8 + (w.c as usize - 1),
            Orientation::V => 320 + (w.r as usize - 1) * 8 + (w.c as usize - 1),
        },
    }
}

// ── Search ────────────────────────────────────────────────────────────────────

pub struct Search<'a, E: Evaluator> {
    eval: &'a E,
    tt: Vec<TtSlot>,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: [i32; HISTORY_SIZE],
    node_limit: Option<u64>,
    stopped: bool,
    pub nodes: u64,
}

pub struct SearchResult {
    pub best: Option<Move>,
    /// Score from the side-to-move perspective at the root.
    pub score: i32,
    pub depth: u8,
    pub nodes: u64,
    pub stopped: bool,
}

impl<'a, E: Evaluator> Search<'a, E> {
    pub fn new(eval: &'a E) -> Self {
        Search {
            eval,
            tt: vec![TtSlot::default(); TT_SIZE],
            killers: [[None; 2]; MAX_PLY],
            history: [0; HISTORY_SIZE],
            node_limit: None,
            stopped: false,
            nodes: 0,
        }
    }

    /// Iterative deepening to `max_depth`. Returns the best root move and its
    /// score (side-to-move POV).
    pub fn search(&mut self, state: &State, max_depth: u8) -> SearchResult {
        self.stopped = false;
        if max_depth == 0 {
            return SearchResult {
                best: None,
                score: self.eval.eval(state, state.turn),
                depth: 0,
                nodes: self.nodes,
                stopped: false,
            };
        }
        if let Some(mv) = immediate_winning_move(state) {
            return SearchResult {
                best: Some(mv),
                score: WIN_SCORE,
                depth: 1,
                nodes: self.nodes,
                stopped: false,
            };
        }
        let mut best: Option<Move> = None;
        let mut score = 0;
        let mut reached = 0;
        for depth in 1..=max_depth {
            let (s, m) = self.root(state, depth, best);
            if self.stopped {
                break;
            }
            score = s;
            best = m.or(best);
            reached = depth;
            if score.abs() >= WIN_SCORE {
                break; // forced result found, deeper search is wasted
            }
        }
        SearchResult {
            best,
            score,
            depth: reached,
            nodes: self.nodes,
            stopped: self.stopped,
        }
    }

    /// Iterative deepening with a relative node budget. If the budget trips in
    /// the middle of a depth, the previous completed depth is returned.
    pub fn search_with_node_limit(
        &mut self,
        state: &State,
        max_depth: u8,
        node_limit: u64,
    ) -> SearchResult {
        let old_limit = self.node_limit;
        self.node_limit = Some(self.nodes.saturating_add(node_limit));
        let result = self.search(state, max_depth);
        self.node_limit = old_limit;
        result
    }

    fn root(&mut self, state: &State, depth: u8, hint: Option<Move>) -> (i32, Option<Move>) {
        let mut alpha = -WIN_SCORE * 2;
        let beta = WIN_SCORE * 2;
        let mut best_move = None;
        let mut best_score = -WIN_SCORE * 2;
        // Prefer the prior iteration's best move, then any TT move for this state.
        let hash = state_hash(state);
        let tt_move = {
            let slot = self.tt[hash as usize & TT_MASK]; // Copy
            hint.or_else(|| {
                if slot.key == hash && slot.depth > 0 {
                    slot.best
                } else {
                    None
                }
            })
        };
        for mv in self.order_moves(state, tt_move, 0, true) {
            let child = state.apply(mv);
            let s = -self.negamax(&child, depth - 1, -beta, -alpha, 1);
            if self.stopped {
                break;
            }
            if s > best_score {
                best_score = s;
                best_move = Some(mv);
            }
            if s > alpha {
                alpha = s;
            }
        }
        (best_score, best_move)
    }

    /// Score every legal move at full width (no alpha-beta at the root so each
    /// move gets an exact comparable score), sorted best-first from the
    /// side-to-move's perspective. Used by the graph to prune to the strong
    /// moves only. Shares one transposition table across all children.
    pub fn ranked(&mut self, state: &State, depth: u8) -> Vec<(Move, i32)> {
        let alpha = -WIN_SCORE * 2;
        let beta = WIN_SCORE * 2;
        let mut out: Vec<(Move, i32)> = legal_moves(state)
            .into_iter()
            .map(|mv| {
                let child = state.apply(mv);
                let s = -self.negamax(&child, depth.saturating_sub(1), -beta, -alpha, 1);
                (mv, s)
            })
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1));
        out
    }

    fn negamax(&mut self, state: &State, depth: u8, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        if self.over_budget() {
            self.stopped = true;
            return self.eval.eval(state, state.turn);
        }
        self.nodes += 1;
        let alpha_orig = alpha;
        let hash = state_hash(state);
        let idx = hash as usize & TT_MASK;

        let mut tt_move = None;
        {
            let slot = self.tt[idx]; // Copy — no live borrow held into recursive calls
            if slot.key == hash && slot.depth > 0 {
                tt_move = slot.best;
                if slot.depth >= depth {
                    match slot.bound {
                        Bound::Exact => return slot.score,
                        Bound::Lower if slot.score >= beta => return slot.score,
                        Bound::Upper if slot.score <= alpha => return slot.score,
                        _ => {}
                    }
                }
            }
        }

        if state.winner.is_some() || depth == 0 {
            // eval is from the side *to move* at this node — negamax convention.
            return self.eval.eval(state, state.turn);
        }

        let mut best = -WIN_SCORE * 2;
        let mut best_move = None;
        let safe_ply = ply.min(MAX_PLY - 1);

        for (move_idx, mv) in self
            .order_moves(state, tt_move, safe_ply, false)
            .into_iter()
            .enumerate()
        {
            let child = state.apply(mv);

            // Late-move reductions: after the first few moves, reduce quiet
            // (wall) moves.  If the reduced search beats alpha, re-search full.
            let is_wall = matches!(mv, Move::Wall(_));
            let score = if depth >= LMR_MIN_DEPTH
                && move_idx >= FULL_DEPTH_MOVES
                && is_wall
                && !self.is_killer(mv, safe_ply)
            {
                let full_depth = depth - 1;
                let reduced_depth = full_depth.saturating_sub(LMR_REDUCTION);
                let s_reduced = -self.negamax(&child, reduced_depth, -beta, -alpha, ply + 1);
                if s_reduced > alpha {
                    // Reduced search beat alpha: re-search at full depth.
                    -self.negamax(&child, full_depth, -beta, -alpha, ply + 1)
                } else {
                    s_reduced
                }
            } else {
                -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1)
            };

            if self.stopped {
                break;
            }

            if score > best {
                best = score;
                best_move = Some(mv);
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                // Beta cutoff: update killers + history.
                self.update_killers(mv, safe_ply);
                self.history[history_idx(mv)] =
                    self.history[history_idx(mv)].saturating_add((depth as i32) * (depth as i32));
                break;
            }
        }

        if self.stopped {
            return best;
        }

        let bound = if best <= alpha_orig {
            Bound::Upper
        } else if best >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        // Depth-preferred replacement: keep the deeper (more expensive) analysis.
        let cur = self.tt[idx]; // Copy
        if cur.key != hash || depth >= cur.depth {
            self.tt[idx] = TtSlot {
                key: hash,
                depth,
                score: best,
                bound,
                best: best_move,
            };
        }
        best
    }

    // ── Killer helpers ────────────────────────────────────────────────────────

    #[inline]
    fn is_killer(&self, mv: Move, ply: usize) -> bool {
        self.killers[ply][0] == Some(mv) || self.killers[ply][1] == Some(mv)
    }

    fn update_killers(&mut self, mv: Move, ply: usize) {
        if self.killers[ply][0] != Some(mv) {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = Some(mv);
        }
    }

    // ── Move ordering ─────────────────────────────────────────────────────────

    /// Return legal moves in priority order:
    ///   TT move → killer[0] → killer[1] → rest sorted by history (descending).
    /// Sorts in-place to avoid a second Vec allocation.
    fn order_moves(
        &self,
        state: &State,
        tt_move: Option<Move>,
        ply: usize,
        wide: bool,
    ) -> Vec<Move> {
        let mut moves = if wide {
            search_moves_wide(state)
        } else {
            search_moves(state)
        };
        let k = &self.killers[ply];
        moves.sort_by_cached_key(|&mv| {
            std::cmp::Reverse(self.move_order_score(state, mv, tt_move, k))
        });
        moves
    }

    #[inline]
    fn over_budget(&self) -> bool {
        self.node_limit.is_some_and(|limit| self.nodes >= limit)
    }

    fn move_order_score(
        &self,
        _state: &State,
        mv: Move,
        tt_move: Option<Move>,
        killers: &[Option<Move>; 2],
    ) -> i64 {
        if Some(mv) == tt_move {
            return 8_000_000_000;
        } else if killers[0] == Some(mv) {
            return 7_000_000_000;
        } else if killers[1] == Some(mv) {
            return 6_000_000_000;
        }

        self.history[history_idx(mv)] as i64
    }
}

fn immediate_winning_move(state: &State) -> Option<Move> {
    pawn_moves(state, state.turn)
        .into_iter()
        .find(|to| to.r == state.turn.goal_row())
        .map(Move::Pawn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Heuristic;
    use crate::moves::is_legal;
    use crate::state::{Cell, Side};

    #[test]
    fn immediate_goal_move_short_circuits_search() {
        let state = State {
            pawns: [Cell::new(8, 5), Cell::new(9, 1)],
            h_walls: 0,
            v_walls: 0,
            walls_left: [0, 0],
            turn: Side::South,
            winner: None,
        };
        let eval = Heuristic::default();
        let mut search = Search::new(&eval);
        let result = search.search(&state, 10);

        assert_eq!(result.best, Some(Move::Pawn(Cell::new(9, 5))));
        assert_eq!(result.score, WIN_SCORE);
        assert_eq!(result.depth, 1);
        assert_eq!(result.nodes, 0);
        assert!(!result.stopped);
    }

    #[test]
    fn node_limit_keeps_last_completed_depth() {
        let state = State::initial();
        let eval = Heuristic::default();
        let mut search = Search::new(&eval);
        let result = search.search_with_node_limit(&state, 10, 1_000);

        assert!(result.stopped);
        assert!(result.depth > 0);
        assert!(result.nodes <= 1_000);
        assert!(result.best.is_some_and(|mv| is_legal(&state, mv)));
    }
}
