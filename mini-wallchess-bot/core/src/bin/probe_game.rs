//! Throwaway diagnostic: play one game between two depths and print every move
//! with both pawns' shortest-path distance, to see WHY deeper search misplays.
//! Usage: probe_game <southDepth> <northDepth> [max_plies]

use wallchess_core::arena::BotConfig;
use wallchess_core::eval::Heuristic;
use wallchess_core::moves::distance_to_goal;
use wallchess_core::state::{Move, Side, State};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let sd: u8 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let nd: u8 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    let max: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(120);

    let h = Heuristic::default();
    let south = BotConfig::new(h, sd);
    let north = BotConfig::new(h, nd);

    let mut state = State::initial();
    let mut ply = 0u32;
    println!("South=d{sd}  North=d{nd}");
    while state.winner.is_none() && ply < max {
        let cfg = if state.turn == Side::South { &south } else { &north };
        let mv = match cfg.choose(&state) {
            Some(m) => m,
            None => break,
        };
        let who = if state.turn == Side::South { "S" } else { "N" };
        let desc = match mv {
            Move::Pawn(c) => format!("move({},{})", c.r, c.c),
            Move::Wall(w) => format!("WALL({},{},{:?})", w.r, w.c, w.o),
        };
        state = state.apply(mv);
        let ds = distance_to_goal(&state, state.pawn(Side::South), Side::South.goal_row())
            .unwrap_or(999);
        let dn = distance_to_goal(&state, state.pawn(Side::North), Side::North.goal_row())
            .unwrap_or(999);
        println!(
            "ply {ply:>3} {who} {desc:<18} | distS={ds} distN={dn} wallsS={} wallsN={}",
            state.walls_left[0], state.walls_left[1]
        );
        ply += 1;
    }
    println!("winner={:?} plies={ply}", state.winner);
}
