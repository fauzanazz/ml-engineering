//! Head-to-head arena CLI.
//! Usage: arena <depthA> <depthB> [games [open_plies [seed]]]
//!
//! `open_plies`  random plies played before the bots take over (0 = fixed
//!               initial position, which makes every even/odd pair identical).
//!               Default 6 gives ~6 distinct opening positions per seed.
//! `seed`        xorshift64 seed; change to verify robustness.  Default 1.

use std::time::Instant;

use gameboard_core::arena::{BotConfig, Outcome};
use gameboard_core::distance_to_goal;
use gameboard_core::eval::Heuristic;
use gameboard_core::legal_moves;
use gameboard_core::search::Search;
use gameboard_core::state::{Side, State};

fn xorshift(rng: &mut u64) -> u64 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    *rng
}

/// Play `open_plies` random PAWN-ONLY moves from the initial position, retrying
/// until both pawns have equal BFS distance to their goals (±1 step).
/// Equal race start ensures deeper search wins by strategy, not by inherited advantage.
fn random_opening(open_plies: u32, rng: &mut u64) -> Option<State> {
    use gameboard_core::state::Move;
    for _attempt in 0..200 {
        let mut state = State::initial();
        let mut ok = true;
        for _ in 0..open_plies {
            if state.winner.is_some() {
                ok = false;
                break;
            }
            let moves: Vec<Move> = legal_moves(&state)
                .into_iter()
                .filter(|m| matches!(m, Move::Pawn(_)))
                .collect();
            if moves.is_empty() {
                ok = false;
                break;
            }
            let mv = moves[(xorshift(rng) as usize) % moves.len()];
            state = state.apply(mv);
        }
        if !ok {
            continue;
        }
        // Accept only balanced openings: |dist_south - dist_north| <= 1.
        let ds = distance_to_goal(&state, state.pawn(Side::South), Side::South.goal_row())
            .unwrap_or(999) as i32;
        let dn = distance_to_goal(&state, state.pawn(Side::North), Side::North.goal_row())
            .unwrap_or(999) as i32;
        if (ds - dn).abs() <= 1 {
            return Some(state);
        }
    }
    Some(State::initial()) // fallback: initial position is perfectly balanced
}

/// Play one game starting from `start`; returns (winner_tag, plies, mean_ms_per_move).
fn play_timed(
    south: &BotConfig<Heuristic>,
    north: &BotConfig<Heuristic>,
    start: State,
    max_plies: u32,
) -> (Outcome, u32, f64) {
    let mut state = start;
    let mut ply = 0u32;
    let mut total_ms = 0f64;
    let mut move_count = 0u32;

    while state.winner.is_none() && ply < max_plies {
        let cfg = if state.turn == Side::South {
            south
        } else {
            north
        };
        let t0 = Instant::now();
        let mut s = Search::new(&cfg.eval);
        let mv = match s.search(&state, cfg.depth).best {
            Some(m) => m,
            None => break,
        };
        total_ms += t0.elapsed().as_secs_f64() * 1000.0;
        move_count += 1;
        state = state.apply(mv);
        ply += 1;
    }

    let mean_ms = if move_count > 0 {
        total_ms / move_count as f64
    } else {
        0.0
    };
    let outcome = match state.winner {
        Some(Side::South) => Outcome::P0Win,
        Some(Side::North) => Outcome::P1Win,
        None => Outcome::Draw,
    };
    (outcome, ply, mean_ms)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let da: u8 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let db: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let games: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);
    let open_plies: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6);
    let seed: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

    let eval = Heuristic::default();
    let a = BotConfig::new(eval, da);
    let b = BotConfig::new(eval, db);

    let (mut aw, mut bw, mut draws) = (0u32, 0u32, 0u32);
    let mut total_mean_ms = 0f64;
    let mut rng = seed;

    for g in 0..games {
        let start = match random_opening(open_plies, &mut rng) {
            Some(s) => s,
            None => State::initial(), // fallback (game ended during opening)
        };
        let a_is_south = g % 2 == 0;
        let (south, north) = if a_is_south { (&a, &b) } else { (&b, &a) };
        let (outcome, plies, mean_ms) = play_timed(south, north, start, 300);
        total_mean_ms += mean_ms;

        let tag = match outcome {
            Outcome::P0Win if a_is_south => {
                aw += 1;
                "A"
            }
            Outcome::P1Win if !a_is_south => {
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
        println!(
            "game {g:>2}: A_south={a_is_south} plies={plies:>3} mean_ms={mean_ms:>6.1} winner={tag}"
        );
    }

    let overall_mean_ms = total_mean_ms / games as f64;
    println!(
        "TOTAL A(d{da}) {aw} - {bw} B(d{db})  draws={draws}  (games={games} open={open_plies} seed={seed})  mean_ms/move={overall_mean_ms:.1}"
    );
}
