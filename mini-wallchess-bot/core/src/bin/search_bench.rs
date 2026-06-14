//! Search benchmark for browser-strength gates.
//!
//! Usage:
//!   search_bench [--depths 8,9,10,11,12] [--positions 12]
//!                [--open-plies 10] [--seed 1] [--node-limit 0]

use std::time::Instant;

use wallchess_core::eval::Heuristic;
use wallchess_core::moves::{search_moves, search_moves_wide};
use wallchess_core::{distance_to_goal, legal_moves, Move, Search, Side, State};

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[derive(Default)]
struct DepthStats {
    ms: Vec<f64>,
    nodes: Vec<u64>,
    stopped: u32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let depths = parse_depths(arg_value(&args, "--depths").as_deref().unwrap_or("8,9,10"));
    let positions = arg_value(&args, "--positions")
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let open_plies = arg_value(&args, "--open-plies")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let seed = arg_value(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let node_limit = arg_value(&args, "--node-limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let states = bench_positions(positions, open_plies, seed);
    let eval = Heuristic::default();
    let mut stats: Vec<DepthStats> = depths.iter().map(|_| DepthStats::default()).collect();

    println!(
        "bench positions={} open_plies={} seed={} node_limit={} depths={:?}",
        states.len(),
        open_plies,
        seed,
        node_limit,
        depths
    );

    for (pos_idx, state) in states.iter().enumerate() {
        let candidates = search_moves(state).len();
        let wide_candidates = search_moves_wide(state).len();
        println!(
            "pos {pos_idx:>2}: turn={:?} candidates={candidates} wide_candidates={wide_candidates}",
            state.turn
        );
        for (depth_idx, depth) in depths.iter().copied().enumerate() {
            let mut search = Search::new(&eval);
            let start = Instant::now();
            let result = if node_limit > 0 {
                search.search_with_node_limit(state, depth, node_limit)
            } else {
                search.search(state, depth)
            };
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            stats[depth_idx].ms.push(elapsed_ms);
            stats[depth_idx].nodes.push(result.nodes);
            if result.stopped {
                stats[depth_idx].stopped += 1;
            }
            println!(
                "  d{depth:<2} reached={:<2} stopped={} ms={:>8.2} nodes={:>9} best={:?}",
                result.depth, result.stopped, elapsed_ms, result.nodes, result.best
            );
        }
    }

    for (depth, s) in depths.iter().zip(stats.iter_mut()) {
        s.ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s.nodes.sort_unstable();
        let mean_ms = s.ms.iter().sum::<f64>() / s.ms.len().max(1) as f64;
        let mean_nodes = s.nodes.iter().sum::<u64>() / s.nodes.len().max(1) as u64;
        println!(
            "SUMMARY d{depth:<2} mean_ms={:>8.2} p95_ms={:>8.2} p99_ms={:>8.2} mean_nodes={:>9} p95_nodes={:>9} stopped={}",
            mean_ms,
            percentile(&s.ms, 0.95),
            percentile(&s.ms, 0.99),
            mean_nodes,
            percentile_u64(&s.nodes, 0.95),
            s.stopped
        );
    }
}

fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn parse_depths(raw: &str) -> Vec<u8> {
    raw.split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect()
}

fn bench_positions(count: usize, open_plies: u32, seed: u64) -> Vec<State> {
    let mut rng = Rng(seed.max(1));
    let mut states = Vec::with_capacity(count.max(1));
    states.push(State::initial());
    while states.len() < count {
        states.push(random_balanced_opening(open_plies, &mut rng));
    }
    states
}

fn random_balanced_opening(open_plies: u32, rng: &mut Rng) -> State {
    for _ in 0..200 {
        let mut state = State::initial();
        let mut ok = true;
        for _ in 0..open_plies {
            if state.winner.is_some() {
                ok = false;
                break;
            }
            let moves: Vec<Move> = legal_moves(&state)
                .into_iter()
                .filter(|mv| matches!(mv, Move::Pawn(_)))
                .collect();
            if moves.is_empty() {
                ok = false;
                break;
            }
            state = state.apply(moves[(rng.next() as usize) % moves.len()]);
        }
        if ok && balanced(&state) {
            return state;
        }
    }
    State::initial()
}

fn balanced(state: &State) -> bool {
    let south = side_dist(state, Side::South);
    let north = side_dist(state, Side::North);
    (south - north).abs() <= 1
}

fn side_dist(state: &State, side: Side) -> i32 {
    distance_to_goal(state, state.pawn(side), side.goal_row()).unwrap_or(999) as i32
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).ceil() as usize;
    sorted[idx]
}

fn percentile_u64(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).ceil() as usize;
    sorted[idx]
}
