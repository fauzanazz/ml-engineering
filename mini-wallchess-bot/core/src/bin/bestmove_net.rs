//! Run MCTS with the trained net and print the best opening move. Proves the
//! candle inference path end-to-end. Build with `--features net`.
//!
//! Usage: bestmove_net <weights.safetensors> [sims]

use wallchess_core::{net::NetEvaluator, Mcts, MctsConfig, State};

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a
        .next()
        .expect("usage: bestmove_net <weights.safetensors> [sims]");
    let sims: u32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(200);

    let net = NetEvaluator::load(&path).expect("load net weights");
    let cfg = MctsConfig {
        sims,
        root_noise: 0.0,
        ..MctsConfig::default()
    };
    let mut mcts = Mcts::new(&net, cfg);
    let state = State::initial();
    let mut visits = mcts.run(&state);
    visits.sort_by_key(|(_, n)| std::cmp::Reverse(*n));

    println!("top moves by MCTS visits ({sims} sims, net policy):");
    for (mv, n) in visits.iter().take(5) {
        println!("  {n:>4}  {mv:?}");
    }
}
