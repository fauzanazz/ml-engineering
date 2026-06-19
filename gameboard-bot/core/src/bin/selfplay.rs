//! Quick self-play smoke: heuristic engine vs itself, prints the 0..100 split
//! each ply. Sanity check that search + rules terminate in a legal game.

use gameboard_core::{analyze, moves::is_legal, Side, State};

fn main() {
    let depth: u8 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(2);
    let k = 200.0;

    let mut state = State::initial();
    let mut ply = 0;
    while state.winner.is_none() && ply < 300 {
        let (best, south, north) = analyze(&state, depth, k);
        let mv = best.expect("no legal move");
        assert!(is_legal(&state, mv), "engine produced illegal move");
        println!(
            "ply {ply:3}  turn {:?}  S {south:3} / N {north:3}  -> {mv:?}",
            state.turn
        );
        state = state.apply(mv);
        ply += 1;
    }
    match state.winner {
        Some(Side::South) => println!("SOUTH wins in {ply} plies"),
        Some(Side::North) => println!("NORTH wins in {ply} plies"),
        None => println!("no winner after {ply} plies (cap reached)"),
    }
}
