//! Net-vs-heuristic arena with race-scored caps and per-engine timing.
//!
//! Usage:
//!   net_arena <weights.safetensors> [games] [sims] [heuristic_depth] [max_plies] [opening_plies]

use std::time::{Duration, Instant};

use wallchess_core::{
    distance_to_goal, eval::Heuristic, legal_moves, net::NetEvaluator, Mcts, MctsConfig, Move,
    Search, Side, State,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Player {
    Net,
    Heuristic,
}

#[derive(Clone, Copy, Debug)]
enum ResultKind {
    Natural(Side),
    Race(Side),
    Draw,
}

#[derive(Clone, Copy, Debug, Default)]
struct Timing {
    moves: u32,
    total: Duration,
    max: Duration,
}

impl Timing {
    fn add(&mut self, elapsed: Duration) {
        self.moves += 1;
        self.total += elapsed;
        self.max = self.max.max(elapsed);
    }

    fn avg_ms(self) -> f64 {
        if self.moves == 0 {
            0.0
        } else {
            self.total.as_secs_f64() * 1000.0 / f64::from(self.moves)
        }
    }

    fn max_ms(self) -> f64 {
        self.max.as_secs_f64() * 1000.0
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct MatchStats {
    net_wins: u32,
    heuristic_wins: u32,
    draws: u32,
    natural: u32,
    race_scored: u32,
    plies: u32,
    net_timing: Timing,
    heuristic_timing: Timing,
}

#[derive(Clone, Copy, Debug)]
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn index(&mut self, len: usize) -> usize {
        (self.next() % len as u64) as usize
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let weights = args.next().expect(
        "usage: net_arena <weights.safetensors> [games] [sims] [heuristic_depth] [max_plies] [opening_plies]",
    );
    let games: u32 = parse_arg(args.next(), 20);
    let sims: u32 = parse_arg(args.next(), 200);
    let heuristic_depth: u8 = parse_arg(args.next(), 2);
    let max_plies: u32 = parse_arg(args.next(), 140);
    let opening_plies: u32 = parse_arg(args.next(), 0);

    let net = NetEvaluator::load(&weights).expect("load net weights");
    let heuristic = Heuristic::default();
    let mut rng = Rng(0x51f1_7e5d_9a11_2026);
    let mut stats = MatchStats::default();

    for game in 0..games {
        let net_side = if game % 2 == 0 {
            Side::South
        } else {
            Side::North
        };
        let mut state = State::initial();
        play_random_opening(&mut state, opening_plies, &mut rng);

        let mut plies = 0u32;
        while state.winner.is_none() && plies < max_plies {
            let player = if state.turn == net_side {
                Player::Net
            } else {
                Player::Heuristic
            };
            let start = Instant::now();
            let mv = match player {
                Player::Net => choose_net(&net, &state, sims),
                Player::Heuristic => choose_heuristic(&heuristic, &state, heuristic_depth),
            };
            let elapsed = start.elapsed();
            match player {
                Player::Net => stats.net_timing.add(elapsed),
                Player::Heuristic => stats.heuristic_timing.add(elapsed),
            }
            let Some(mv) = mv else {
                break;
            };
            state = state.apply(mv);
            plies += 1;
        }

        let result = state.winner.map(ResultKind::Natural).unwrap_or_else(|| {
            race_winner(&state)
                .map(ResultKind::Race)
                .unwrap_or(ResultKind::Draw)
        });
        record_result(&mut stats, result, net_side, plies);
        println!(
            "game {game:>3}: net={:?} plies={plies:>3} result={}",
            net_side,
            result_label(result, net_side)
        );
    }

    println!(
        "TOTAL net {} - {} heuristic  draws={}  games={}  natural={}  race_scored={}  avg_plies={:.1}",
        stats.net_wins,
        stats.heuristic_wins,
        stats.draws,
        games,
        stats.natural,
        stats.race_scored,
        f64::from(stats.plies) / f64::from(games.max(1)),
    );
    println!(
        "TIMING net avg={:.2}ms max={:.2}ms moves={}  heuristic avg={:.2}ms max={:.2}ms moves={}",
        stats.net_timing.avg_ms(),
        stats.net_timing.max_ms(),
        stats.net_timing.moves,
        stats.heuristic_timing.avg_ms(),
        stats.heuristic_timing.max_ms(),
        stats.heuristic_timing.moves,
    );
}

fn parse_arg<T: std::str::FromStr>(arg: Option<String>, default: T) -> T {
    arg.and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn choose_net(net: &NetEvaluator, state: &State, sims: u32) -> Option<Move> {
    let mut mcts = Mcts::new(
        net,
        MctsConfig {
            sims,
            root_noise: 0.0,
            ..MctsConfig::default()
        },
    );
    mcts.run(state)
        .into_iter()
        .max_by_key(|(_, visits)| *visits)
        .map(|(mv, _)| mv)
}

fn choose_heuristic(heuristic: &Heuristic, state: &State, depth: u8) -> Option<Move> {
    let mut search = Search::new(heuristic);
    search.search(state, depth).best
}

fn play_random_opening(state: &mut State, opening_plies: u32, rng: &mut Rng) {
    for _ in 0..opening_plies {
        if state.winner.is_some() {
            return;
        }
        let moves = legal_moves(state);
        if moves.is_empty() {
            return;
        }
        *state = state.apply(moves[rng.index(moves.len())]);
    }
}

fn race_winner(state: &State) -> Option<Side> {
    let far = u16::MAX;
    let dist =
        |side: Side| distance_to_goal(state, state.pawn(side), side.goal_row()).unwrap_or(far);
    let south = dist(Side::South);
    let north = dist(Side::North);
    match south.cmp(&north) {
        std::cmp::Ordering::Less => Some(Side::South),
        std::cmp::Ordering::Greater => Some(Side::North),
        std::cmp::Ordering::Equal => None,
    }
}

fn record_result(stats: &mut MatchStats, result: ResultKind, net_side: Side, plies: u32) {
    stats.plies += plies;
    match result {
        ResultKind::Natural(winner) => {
            stats.natural += 1;
            record_winner(stats, winner, net_side);
        }
        ResultKind::Race(winner) => {
            stats.race_scored += 1;
            record_winner(stats, winner, net_side);
        }
        ResultKind::Draw => stats.draws += 1,
    }
}

fn record_winner(stats: &mut MatchStats, winner: Side, net_side: Side) {
    if winner == net_side {
        stats.net_wins += 1;
    } else {
        stats.heuristic_wins += 1;
    }
}

fn result_label(result: ResultKind, net_side: Side) -> &'static str {
    match result {
        ResultKind::Natural(winner) if winner == net_side => "net:natural",
        ResultKind::Natural(_) => "heuristic:natural",
        ResultKind::Race(winner) if winner == net_side => "net:race",
        ResultKind::Race(_) => "heuristic:race",
        ResultKind::Draw => "draw",
    }
}
