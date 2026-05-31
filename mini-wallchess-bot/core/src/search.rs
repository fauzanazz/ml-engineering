//! Negamax + alpha-beta with iterative deepening. The transposition table is
//! the lazily-built "graph cache": equal states reached by different move orders
//! share one entry.

use std::collections::HashMap;

use crate::eval::{Evaluator, WIN_SCORE};
use crate::moves::legal_moves;
use crate::state::{Move, State};

#[derive(Clone, Copy)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TtEntry {
    depth: u8,
    score: i32,
    bound: Bound,
    /// Best move found at this node — replayed first on revisits so iterative
    /// deepening turns last iteration's PV into this iteration's ordering.
    best: Option<Move>,
}

pub struct Search<'a, E: Evaluator> {
    eval: &'a E,
    tt: HashMap<State, TtEntry>,
    pub nodes: u64,
}

pub struct SearchResult {
    pub best: Option<Move>,
    /// Score from the side-to-move perspective at the root.
    pub score: i32,
    pub depth: u8,
    pub nodes: u64,
}

impl<'a, E: Evaluator> Search<'a, E> {
    pub fn new(eval: &'a E) -> Self {
        Search {
            eval,
            tt: HashMap::new(),
            nodes: 0,
        }
    }

    /// Iterative deepening to `max_depth`. Returns the best root move and its
    /// score (side-to-move POV).
    pub fn search(&mut self, state: &State, max_depth: u8) -> SearchResult {
        let mut best: Option<Move> = None;
        let mut score = 0;
        let mut reached = 0;
        for depth in 1..=max_depth {
            let (s, m) = self.root(state, depth, best);
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
        }
    }

    fn root(&mut self, state: &State, depth: u8, hint: Option<Move>) -> (i32, Option<Move>) {
        let mut alpha = -WIN_SCORE * 2;
        let beta = WIN_SCORE * 2;
        let mut best_move = None;
        let mut best_score = -WIN_SCORE * 2;
        // Prefer the prior iteration's best move, then any TT move for this state.
        let tt_move = hint.or_else(|| self.tt.get(state).and_then(|e| e.best));
        for mv in order_moves(state, tt_move) {
            let child = state.apply(mv);
            let s = -self.negamax(&child, depth - 1, -beta, -alpha);
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
                let s = -self.negamax(&child, depth.saturating_sub(1), -beta, -alpha);
                (mv, s)
            })
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1));
        out
    }

    fn negamax(&mut self, state: &State, depth: u8, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        let alpha_orig = alpha;

        let mut tt_move = None;
        if let Some(e) = self.tt.get(state) {
            tt_move = e.best;
            if e.depth >= depth {
                match e.bound {
                    Bound::Exact => return e.score,
                    Bound::Lower if e.score >= beta => return e.score,
                    Bound::Upper if e.score <= alpha => return e.score,
                    _ => {}
                }
            }
        }

        if state.winner.is_some() || depth == 0 {
            // eval is from the side *to move* at this node — negamax convention.
            return self.eval.eval(state, state.turn);
        }

        let mut best = -WIN_SCORE * 2;
        let mut best_move = None;
        for mv in order_moves(state, tt_move) {
            let child = state.apply(mv);
            let s = -self.negamax(&child, depth - 1, -beta, -alpha);
            if s > best {
                best = s;
                best_move = Some(mv);
            }
            if s > alpha {
                alpha = s;
            }
            if alpha >= beta {
                break; // beta cutoff
            }
        }

        let bound = if best <= alpha_orig {
            Bound::Upper
        } else if best >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        // Depth-preferred replacement: keep the deeper (more expensive) analysis
        // rather than letting a shallow revisit clobber it.
        let keep = self
            .tt
            .get(state)
            .map(|e| depth >= e.depth)
            .unwrap_or(true);
        if keep {
            self.tt.insert(
                *state,
                TtEntry {
                    depth,
                    score: best,
                    bound,
                    best: best_move,
                },
            );
        }
        best
    }
}

/// Static move ordering: try `tt_move` first (its subtree already scored well),
/// then the natural pawn-before-wall order from `legal_moves`. Cheap — no extra
/// BFS — but enough to make alpha-beta cut the bulk of a ~130-wide tree.
fn order_moves(state: &State, tt_move: Option<Move>) -> Vec<Move> {
    let mut moves = legal_moves(state);
    if let Some(m) = tt_move {
        if let Some(i) = moves.iter().position(|&x| x == m) {
            moves.swap(0, i);
        }
    }
    moves
}
