//! Alpha-beta arena gate for CNN value nets.
//!
//! Usage:
//!   net_search_arena <candidate.safetensors> [opponent.safetensors|heuristic|-] [games] [depth] [max_plies] [opening_plies] [threshold]
//!
//! The candidate always uses `NetEvaluator` as a leaf [`Evaluator`] inside the
//! existing alpha-beta [`Search`]. The opponent is either the hand heuristic or a
//! previously promoted net, using the same search depth/config.

use gameboard_core::{
    distance_to_goal, eval::Heuristic, legal_moves, net::NetEvaluator, Move, Search, SearchConfig,
    Side, State,
};

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

enum Opponent {
    Heuristic(Heuristic),
    Net(NetEvaluator),
}

#[derive(Default)]
struct Stats {
    candidate_wins: u32,
    opponent_wins: u32,
    draws: u32,
    plies: u32,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let candidate_path = args.next().expect(
        "usage: net_search_arena <candidate.safetensors> [opponent.safetensors|heuristic|-] [games] [depth] [max_plies] [opening_plies] [threshold]",
    );
    let opponent_arg = args.next().unwrap_or_else(|| "heuristic".to_string());
    let games: u32 = parse_arg(args.next(), 20);
    let depth: u8 = parse_arg(args.next(), 2);
    let max_plies: u32 = parse_arg(args.next(), 140);
    let opening_plies: u32 = parse_arg(args.next(), 4);
    let threshold: f32 = parse_arg(args.next(), 0.60);

    let candidate = NetEvaluator::load(&candidate_path).expect("load candidate net");
    let opponent = if opponent_arg == "heuristic" || opponent_arg == "-" {
        Opponent::Heuristic(Heuristic::default())
    } else {
        Opponent::Net(NetEvaluator::load(&opponent_arg).expect("load opponent net"))
    };
    let cfg = SearchConfig::from_env();
    let seed = std::env::var("NET_SEARCH_ARENA_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|s| *s != 0)
        .unwrap_or(0xa6e1_5eed_2026_0707);
    let mut rng = Rng(seed);
    let mut stats = Stats::default();

    for game in 0..games {
        let candidate_side = if game % 2 == 0 {
            Side::South
        } else {
            Side::North
        };
        let mut state = State::initial();
        play_random_opening(&mut state, opening_plies, &mut rng);
        let mut plies = 0u32;
        while state.winner.is_none() && plies < max_plies {
            let mv = if state.turn == candidate_side {
                choose_net(&candidate, &state, depth, cfg)
            } else {
                choose_opponent(&opponent, &state, depth, cfg)
            };
            let Some(mv) = mv else { break };
            state = state.apply(mv);
            plies += 1;
        }
        stats.plies += plies;
        let result = match state.winner.or_else(|| race_winner(&state)) {
            Some(winner) if winner == candidate_side => {
                stats.candidate_wins += 1;
                "candidate"
            }
            Some(_) => {
                stats.opponent_wins += 1;
                "opponent"
            }
            None => {
                stats.draws += 1;
                "draw"
            }
        };
        println!("game {game:>3}: candidate={candidate_side:?} plies={plies:>3} result={result}");
    }

    let score = (stats.candidate_wins as f32 + 0.5 * stats.draws as f32) / games.max(1) as f32;
    let promote = score >= threshold;
    println!(
        "RESULT {{\"candidate_wins\":{},\"opponent_wins\":{},\"draws\":{},\"games\":{},\"candidate_score\":{:.4},\"threshold\":{:.4},\"promote\":{}}}",
        stats.candidate_wins,
        stats.opponent_wins,
        stats.draws,
        games,
        score,
        threshold,
        promote
    );
}

fn parse_arg<T: std::str::FromStr>(arg: Option<String>, default: T) -> T {
    arg.and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn choose_net(net: &NetEvaluator, state: &State, depth: u8, cfg: SearchConfig) -> Option<Move> {
    let mut search = Search::with_config(net, cfg);
    search.search(state, depth).best
}

fn choose_opponent(opp: &Opponent, state: &State, depth: u8, cfg: SearchConfig) -> Option<Move> {
    match opp {
        Opponent::Heuristic(h) => {
            let mut search = Search::with_config(h, cfg);
            search.search(state, depth).best
        }
        Opponent::Net(net) => choose_net(net, state, depth, cfg),
    }
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
    match dist(Side::South).cmp(&dist(Side::North)) {
        std::cmp::Ordering::Less => Some(Side::South),
        std::cmp::Ordering::Greater => Some(Side::North),
        std::cmp::Ordering::Equal => None,
    }
}
