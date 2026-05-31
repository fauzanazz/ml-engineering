//! Head-to-head arena CLI. Usage: arena <depthA> <depthB> [games]
//! Decisive search-health check: a deeper bot should beat a shallower one.

use wallchess_core::arena::{play_game, BotConfig, Outcome};
use wallchess_core::eval::Heuristic;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let da: u8 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let db: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let games: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);

    let h = Heuristic::default();
    let a = BotConfig::new(h, da);
    let b = BotConfig::new(h, db);

    let (mut aw, mut bw, mut draws) = (0u32, 0u32, 0u32);
    for g in 0..games {
        let a_is_south = g % 2 == 0;
        let (south, north) = if a_is_south { (&a, &b) } else { (&b, &a) };
        let (outcome, plies) = play_game(south, north, 300);
        let tag = match outcome {
            Outcome::SouthWin if a_is_south => {
                aw += 1;
                "A"
            }
            Outcome::NorthWin if !a_is_south => {
                aw += 1;
                "A"
            }
            Outcome::Draw => {
                draws += 1;
                "draw"
            }
            _ => {
                bw += 1;
                "B"
            }
        };
        println!("game {g}: A_south={a_is_south} plies={plies} winner={tag}");
    }
    println!(
        "TOTAL A(d{da}) {aw} - {bw} B(d{db})  draws={draws}  (games={games})"
    );
}
