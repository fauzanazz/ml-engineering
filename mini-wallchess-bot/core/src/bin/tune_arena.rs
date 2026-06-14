//! Eval-weight tuning arena. Pit a candidate (depth/w_path/w_wall) against a
//! baseline, alternating sides. A healthy eval makes the DEEPER bot win.
//! Usage: tune_arena <depthA> <wpathA> <wwallA> <depthB> <wpathB> <wwallB> [games]

use wallchess_core::arena::{play_game, BotConfig, Outcome};
use wallchess_core::eval::Heuristic;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let g = |i: usize, d: i64| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (da, wpa, wwa) = (g(1, 4) as u8, g(2, 50) as i32, g(3, 4) as i32);
    let (db, wpb, wwb) = (g(4, 2) as u8, g(5, 50) as i32, g(6, 4) as i32);
    let games = g(7, 20) as u32;

    let a_cfg = BotConfig::new(
        Heuristic {
            w_path: wpa,
            w_wall: wwa,
        },
        da,
    );
    let b_cfg = BotConfig::new(
        Heuristic {
            w_path: wpb,
            w_wall: wwb,
        },
        db,
    );

    let (mut aw, mut bw, mut draws) = (0u32, 0u32, 0u32);
    for gm in 0..games {
        let a_south = gm % 2 == 0;
        let (south, north) = if a_south {
            (&a_cfg, &b_cfg)
        } else {
            (&b_cfg, &a_cfg)
        };
        let (outcome, _) = play_game(south, north, 200);
        match outcome {
            Outcome::SouthWin if a_south => aw += 1,
            Outcome::NorthWin if !a_south => aw += 1,
            Outcome::Draw => draws += 1,
            _ => bw += 1,
        }
    }
    println!(
        "A(d{da},wp{wpa},ww{wwa}) {aw} - {bw} B(d{db},wp{wpb},ww{wwb})  draws={draws} games={games}"
    );
}
