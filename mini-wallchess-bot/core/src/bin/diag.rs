//! Search diagnostic: node counts and best move per depth from a fixed
//! position. If node counts barely grow with depth, the search isn't recursing.

use wallchess_core::{eval::Heuristic, search::Search, State};

fn main() {
    let state = State::initial();
    let h = Heuristic::default();
    for depth in 1..=4u8 {
        let mut s = Search::new(&h);
        let res = s.search(&state, depth);
        println!(
            "depth {depth}  nodes {:>8}  score {:>6}  best {:?}",
            res.nodes, res.score, res.best
        );
    }
}
