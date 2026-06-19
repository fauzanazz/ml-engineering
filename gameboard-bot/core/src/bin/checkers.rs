//! International draughts CLI on the shared `gameboard-core` engine.
//!
//! Demonstrates that the second game reuses the generic search + arena with no
//! game-specific search code, and reproduces the perft oracle from the terminal.
//!
//! Usage:
//!   checkers perft [maxDepth=7]     # move-tree counts (the fidelity oracle)
//!   checkers bestmove [depth=12]    # best opening move (draughts search preset)
//!   checkers selfplay [depth=6]     # one self-play game to a decisive/draw end
//!
//! Eval weights honour the `GB_CK_*` env vars; search flags honour `WC_*`.

use std::time::Instant;

use gameboard_core::arena::{play_game, BotConfig};
use gameboard_core::checkers::{legal_moves, perft, CheckersHeuristic, State};
use gameboard_core::game::Game;
use gameboard_core::search::{Search, SearchConfig};
use gameboard_core::Checkers;

fn parse<T: std::str::FromStr>(s: Option<&String>, default: T) -> T {
    s.and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("perft");

    match cmd {
        "perft" => {
            let max: u32 = parse(args.get(2), 7);
            // Verified oracle (Gilbert/Boomstra/BikDam) for cross-checking.
            let oracle = [0u64, 9, 81, 658, 4265, 27117, 167140, 1049442, 6483961];
            for d in 1..=max {
                let t0 = Instant::now();
                let n = perft(&State::initial(), d);
                let tag = oracle
                    .get(d as usize)
                    .map(|&o| if o == n { "ok" } else { "MISMATCH" })
                    .unwrap_or("?");
                println!("perft({d}) = {n:>12}   [{tag}]   {:?}", t0.elapsed());
            }
        }
        "bestmove" => {
            let depth: u8 = parse(args.get(2), 12);
            let h = CheckersHeuristic::from_env();
            let mut s = Search::with_config(&h, SearchConfig::draughts());
            let init = State::initial();
            let t0 = Instant::now();
            let res = s.search(&init, depth);
            let legal = legal_moves(&init).len();
            println!(
                "game={} depth={} legal_moves={legal}\nbest={:?} score={} reached_depth={} nodes={} time={:?}",
                Checkers::ID,
                depth,
                res.best,
                res.score,
                res.depth,
                res.nodes,
                t0.elapsed()
            );
        }
        "selfplay" => {
            let depth: u8 = parse(args.get(2), 6);
            let h = CheckersHeuristic::from_env();
            let cfg = BotConfig::new(h, depth);
            let t0 = Instant::now();
            let (outcome, plies) = play_game(&cfg, &cfg, 400);
            println!(
                "self-play (depth {depth}): {outcome:?} after {plies} plies  ({:?})",
                t0.elapsed()
            );
        }
        other => {
            eprintln!("unknown command {other:?}");
            eprintln!("usage: checkers [perft N | bestmove DEPTH | selfplay DEPTH]");
            std::process::exit(2);
        }
    }
}
