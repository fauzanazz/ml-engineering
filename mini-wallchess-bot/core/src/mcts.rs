//! PUCT Monte-Carlo Tree Search over the [`PolicyValue`] seam.
//!
//! This is the label source that replaces the depth-2 negamax teacher: search
//! quality scales with simulation count and a learned policy, so distilling the
//! visit-count distribution can surpass the heuristic instead of regressing to
//! it. The seam means iteration 0 bootstraps from the hand heuristic and later
//! iterations plug a trained net into the exact same search.

use crate::action::{action_index, ACTION_COUNT};
use crate::eval::{Heuristic, WIN_SCORE};
use crate::moves::legal_moves;
use crate::search::Search;
use crate::state::{Move, State};

/// Leaf model for MCTS: a value estimate and per-action priors.
pub trait PolicyValue {
    /// Returns `(value, priors)` where `value` is in `[-1, 1]` from `state`'s
    /// side-to-move POV, and `priors` is a length-[`ACTION_COUNT`] vector in the
    /// **absolute** frame (not the me-frame). Priors need not sum to 1 over legal
    /// moves — the search renormalizes over the legal set.
    fn evaluate(&self, state: &State) -> (f32, Vec<f32>);
}

#[derive(Clone)]
struct Edge {
    mv: Move,
    prior: f32,
    child: Option<usize>,
    n: u32,
    w: f32, // summed value from the parent's side-to-move POV
}

struct Node {
    state: State,
    terminal: bool,
    /// Value from this node's side-to-move POV, set when the node is evaluated.
    value: f32,
    edges: Vec<Edge>,
}

/// MCTS configuration.
#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    pub sims: u32,
    pub c_puct: f32,
    /// Root exploration: mix `root_noise` of a uniform distribution into the root
    /// priors so self-play doesn't collapse onto one line. 0 disables.
    pub root_noise: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            sims: 200,
            c_puct: 1.5,
            root_noise: 0.25,
        }
    }
}

pub struct Mcts<'a, P: PolicyValue> {
    eval: &'a P,
    cfg: MctsConfig,
    nodes: Vec<Node>,
}

impl<'a, P: PolicyValue> Mcts<'a, P> {
    pub fn new(eval: &'a P, cfg: MctsConfig) -> Self {
        Mcts {
            eval,
            cfg,
            nodes: Vec::new(),
        }
    }

    /// Run the configured simulations from `root` and return each legal root
    /// move with its visit count — the visit distribution is the policy target.
    pub fn run(&mut self, root: &State) -> Vec<(Move, u32)> {
        self.nodes.clear();
        let r = self.add_node(*root);
        self.evaluate_node(r);
        if self.cfg.root_noise > 0.0 {
            self.add_root_noise(r);
        }

        for _ in 0..self.cfg.sims {
            let mut path: Vec<(usize, usize)> = Vec::new();
            let mut cur = r;
            loop {
                let node = &self.nodes[cur];
                if node.terminal || node.edges.is_empty() {
                    break;
                }
                let e = self.select_edge(cur);
                path.push((cur, e));
                match self.nodes[cur].edges[e].child {
                    Some(ch) => cur = ch,
                    None => {
                        let st = self.nodes[cur].state.apply(self.nodes[cur].edges[e].mv);
                        let ch = self.add_node(st);
                        self.nodes[cur].edges[e].child = Some(ch);
                        self.evaluate_node(ch);
                        cur = ch;
                        break;
                    }
                }
            }
            let leaf_value = self.nodes[cur].value;
            self.backup(&path, leaf_value);
        }

        self.nodes[r]
            .edges
            .iter()
            .map(|e| (e.mv, e.n))
            .collect()
    }

    fn add_node(&mut self, state: State) -> usize {
        let terminal = state.winner.is_some();
        // A terminal node means the side to move has already lost (the opponent
        // just reached its goal), so its value is -1 from the to-move POV.
        let value = if terminal { -1.0 } else { 0.0 };
        self.nodes.push(Node {
            state,
            terminal,
            value,
            edges: Vec::new(),
        });
        self.nodes.len() - 1
    }

    /// Expand a non-terminal node: query the model, attach legal-move edges with
    /// renormalized priors, and record the node's value.
    fn evaluate_node(&mut self, idx: usize) {
        if self.nodes[idx].terminal {
            return;
        }
        let state = self.nodes[idx].state;
        let (value, priors) = self.eval.evaluate(&state);
        let moves = legal_moves(&state);
        let mut edges: Vec<Edge> = Vec::with_capacity(moves.len());
        let mut sum = 0.0f32;
        for mv in moves {
            let p = priors[action_index(mv)].max(0.0);
            sum += p;
            edges.push(Edge {
                mv,
                prior: p,
                child: None,
                n: 0,
                w: 0.0,
            });
        }
        // Renormalize over legal moves; fall back to uniform if the model gave
        // all-zero mass to the legal set.
        if sum > 0.0 {
            for e in &mut edges {
                e.prior /= sum;
            }
        } else {
            let u = 1.0 / edges.len() as f32;
            for e in &mut edges {
                e.prior = u;
            }
        }
        self.nodes[idx].edges = edges;
        self.nodes[idx].value = value.clamp(-1.0, 1.0);
    }

    fn add_root_noise(&mut self, idx: usize) {
        let eps = self.cfg.root_noise;
        let n = self.nodes[idx].edges.len();
        if n == 0 {
            return;
        }
        let u = 1.0 / n as f32;
        for e in &mut self.nodes[idx].edges {
            e.prior = (1.0 - eps) * e.prior + eps * u;
        }
    }

    fn select_edge(&self, idx: usize) -> usize {
        let node = &self.nodes[idx];
        let sum_n: u32 = node.edges.iter().map(|e| e.n).sum();
        let sqrt_sum = (sum_n as f32).max(1.0).sqrt();
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (i, e) in node.edges.iter().enumerate() {
            let q = if e.n == 0 { 0.0 } else { e.w / e.n as f32 };
            let u = self.cfg.c_puct * e.prior * sqrt_sum / (1.0 + e.n as f32);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    /// Propagate `leaf_value` (leaf side-to-move POV) up the path, flipping sign
    /// each ply so every edge accumulates value from its own parent's POV.
    fn backup(&mut self, path: &[(usize, usize)], leaf_value: f32) {
        let mut value = leaf_value;
        for &(node, edge) in path.iter().rev() {
            value = -value;
            let e = &mut self.nodes[node].edges[edge];
            e.n += 1;
            e.w += value;
        }
    }
}

/// Iteration-0 bootstrap: derive value and priors from the hand heuristic via a
/// shallow negamax. Lets self-play start before any net is trained.
pub struct HeuristicPolicy {
    h: Heuristic,
    /// Depth used to score moves for the priors (1 = one-ply lookahead).
    prior_depth: u8,
    /// Eval units that map to a tanh of ~0.76 (one e-fold). Larger = flatter.
    value_scale: f32,
}

impl Default for HeuristicPolicy {
    fn default() -> Self {
        HeuristicPolicy {
            h: Heuristic::default(),
            prior_depth: 1,
            value_scale: 400.0,
        }
    }
}

impl PolicyValue for HeuristicPolicy {
    fn evaluate(&self, state: &State) -> (f32, Vec<f32>) {
        let mut priors = vec![0.0f32; ACTION_COUNT];
        if state.winner.is_some() {
            return (-1.0, priors);
        }
        let mut s = Search::new(&self.h);
        // Value: shallow search score from the side-to-move POV, squashed.
        let score = s.search(state, self.prior_depth.max(1)).score;
        let value = if score.abs() >= WIN_SCORE {
            (score.signum() as f32).clamp(-1.0, 1.0)
        } else {
            (score as f32 / self.value_scale).tanh()
        };
        // Priors: softmax over per-move scores from the side-to-move POV.
        let ranked = s.ranked(state, self.prior_depth.max(1));
        let max = ranked.iter().map(|(_, sc)| *sc).max().unwrap_or(0) as f32;
        for (mv, sc) in ranked {
            let logit = (sc as f32 - max) / self.value_scale;
            priors[action_index(mv)] = logit.exp();
        }
        (value, priors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::is_legal;
    use crate::state::Side;

    #[test]
    fn visits_concentrate_on_a_legal_move() {
        let policy = HeuristicPolicy::default();
        let cfg = MctsConfig {
            sims: 120,
            c_puct: 1.5,
            root_noise: 0.0,
        };
        let mut mcts = Mcts::new(&policy, cfg);
        let state = State::initial();
        let visits = mcts.run(&state);
        let total: u32 = visits.iter().map(|(_, n)| n).sum();
        assert_eq!(total, cfg.sims, "every sim should visit one root edge");
        let (best, _) = visits.iter().max_by_key(|(_, n)| *n).unwrap();
        assert!(is_legal(&state, *best), "MCTS best move must be legal");
        // from the opening the strong move is a forward pawn advance
        match best {
            Move::Pawn(c) => assert_eq!(c.r, 2),
            other => panic!("expected pawn advance from opening, got {other:?}"),
        }
    }

    #[test]
    fn finds_immediate_winning_move() {
        // South one step from the goal row must get the lion's share of visits on
        // the winning advance.
        // Put South one row from goal but off-column so the straight goal cell is
        // free (North stays on its home column 5).
        let mut state = State::initial();
        state.pawns[Side::South.idx()] = crate::state::Cell::new(8, 3);
        let policy = HeuristicPolicy::default();
        let mut mcts = Mcts::new(&policy, MctsConfig { sims: 200, c_puct: 1.5, root_noise: 0.0 });
        let visits = mcts.run(&state);
        let (best, n) = visits.iter().max_by_key(|(_, n)| *n).unwrap();
        // best move must land on the goal row (an immediate win)
        match best {
            Move::Pawn(c) => assert_eq!(c.r, 9, "winning move steps onto goal row"),
            other => panic!("expected winning pawn advance, got {other:?}"),
        }
        assert!(*n > 100, "winning move should dominate visits, got {n}");
    }
}
