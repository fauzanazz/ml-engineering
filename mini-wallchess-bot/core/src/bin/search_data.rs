//! Supervised teacher-data generator for the small net.
//!
//! Unlike `selfplay_data`, these labels come directly from a stronger
//! alpha-beta search. The expensive search happens offline; the browser bot only
//! pays for the compact net and optional books.
//!
//! Usage:
//!   search_data [samples] [depth] [out.jsonl] [max_random_plies] [value_scale] [policy_scale] [candidates]
//!
//! Set `OPENING_GRAPH=/path/to/opening_graph.jsonl` to seed from precomputed
//! opening nodes. Set `SEARCH_DATA_SEED` to split deterministic parallel shards.

use std::fs::{self, File};
use std::io::{BufWriter, Write};

use wallchess_core::{
    action_index, distance_to_goal, encode, eval::Heuristic, eval::WIN_SCORE, legal_moves,
    mirror_move, parse_state_key, Move, Search, State,
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

fn main() {
    let mut args = std::env::args().skip(1);
    let samples: usize = parse_arg(args.next(), 10_000);
    let depth: u8 = parse_arg(args.next(), 3);
    let out = args
        .next()
        .unwrap_or_else(|| "search_data.jsonl".to_string());
    let max_random_plies: u32 = parse_arg(args.next(), 12);
    let value_scale: f32 = parse_arg(args.next(), 350.0);
    let policy_scale: f32 = parse_arg(args.next(), 80.0);
    let candidates: usize = parse_arg(args.next(), 24);

    let opening_states = load_opening_states();
    let seed = std::env::var("SEARCH_DATA_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|s| *s != 0)
        .unwrap_or(0xf00d_cafe_1234_5678);
    let mut rng = Rng(seed);
    let heuristic = Heuristic::default();
    let mut search = Search::new(&heuristic);
    let file = File::create(&out).expect("create search-data output");
    let mut w = BufWriter::new(file);

    let mut written = 0usize;
    let mut attempts = 0usize;
    while written < samples && attempts < samples.saturating_mul(50) {
        attempts += 1;
        let mut state = start_state(&opening_states, &mut rng);
        play_random_tail(&mut state, max_random_plies, &mut rng);
        if state.winner.is_some() {
            continue;
        }

        let ranked = ranked_candidates(&mut search, &state, depth, candidates);
        if ranked.is_empty() {
            continue;
        }
        let best_score = ranked[0].1;
        let z = score_to_value(best_score, value_scale);
        let all_moves = legal_moves(&state);
        let policy = policy_target(&state, &ranked, &all_moves, policy_scale);
        write_record(&mut w, z, &encode(&state), &policy);
        written += 1;

        if written % 1000 == 0 {
            eprintln!("wrote {written}/{samples} samples");
        }
    }

    w.flush().expect("flush search data");
    eprintln!(
        "wrote {written} samples to {out} depth={depth} candidates={candidates} attempts={attempts} nodes={}",
        search.nodes
    );
}

fn parse_arg<T: std::str::FromStr>(arg: Option<String>, default: T) -> T {
    arg.and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn load_opening_states() -> Vec<State> {
    let Ok(path) = std::env::var("OPENING_GRAPH") else {
        return Vec::new();
    };
    let text = fs::read_to_string(&path).expect("read OPENING_GRAPH");
    let states: Vec<State> = text
        .lines()
        .filter(|line| line.contains("\"type\":\"node\""))
        .filter_map(extract_key)
        .filter_map(parse_state_key)
        .filter(|state| state.winner.is_none())
        .collect();
    eprintln!("loaded {} opening graph states from {path}", states.len());
    states
}

fn extract_key(line: &str) -> Option<&str> {
    let start = line.find("\"key\":\"")? + "\"key\":\"".len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}

fn start_state(opening_states: &[State], rng: &mut Rng) -> State {
    if opening_states.is_empty() {
        State::initial()
    } else {
        opening_states[rng.index(opening_states.len())]
    }
}

fn play_random_tail(state: &mut State, max_plies: u32, rng: &mut Rng) {
    let plies = if max_plies == 0 {
        0
    } else {
        (rng.next() % (max_plies as u64 + 1)) as u32
    };
    for _ in 0..plies {
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

fn ranked_candidates(
    search: &mut Search<'_, Heuristic>,
    state: &State,
    depth: u8,
    candidates: usize,
) -> Vec<(Move, i32)> {
    let opp = state.turn.other();
    let opp_dist_before =
        distance_to_goal(state, state.pawn(opp), opp.goal_row()).unwrap_or(u16::MAX);

    let mut shallow = search.ranked(state, 1);
    shallow.retain(|(mv, _)| match mv {
        Move::Pawn(_) => true,
        Move::Wall(_) => {
            let child = state.apply(*mv);
            let after =
                distance_to_goal(&child, child.pawn(opp), opp.goal_row()).unwrap_or(u16::MAX);
            after > opp_dist_before
        }
    });
    shallow.truncate(candidates.max(1));

    let mut ranked: Vec<(Move, i32)> = shallow
        .into_iter()
        .map(|(mv, _)| {
            let child = state.apply(mv);
            let score = -search.search(&child, depth.saturating_sub(1)).score;
            (mv, score)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    ranked
}

fn score_to_value(score: i32, value_scale: f32) -> f32 {
    if score >= WIN_SCORE {
        1.0
    } else if score <= -WIN_SCORE {
        -1.0
    } else {
        (score as f32 / value_scale.max(1.0)).tanh()
    }
}

fn policy_target(
    state: &State,
    ranked: &[(wallchess_core::Move, i32)],
    all_moves: &[Move],
    policy_scale: f32,
) -> Vec<(usize, f32)> {
    use std::collections::HashSet;
    let scale = policy_scale.max(1.0);
    let max = ranked[0].1 as f32;
    // Score for unlisted moves: floor well below the worst candidate so they
    // get ~0 probability in the softmax target, giving explicit negative signal.
    let floor = ranked.last().map(|(_, s)| *s as f32).unwrap_or(max) - 4.0 * scale;

    let mut weighted: Vec<(usize, f32)> = Vec::new();
    let mut total = 0.0f32;
    let mut listed: HashSet<usize> = HashSet::new();

    for (mv, score) in ranked {
        let weight = ((*score as f32 - max) / scale).clamp(-40.0, 0.0).exp();
        let idx = action_index(mirror_move(state.turn, *mv));
        total += weight;
        weighted.push((idx, weight));
        listed.insert(idx);
    }
    // Add all remaining legal moves with floor weight (explicit near-zero target).
    let floor_weight = ((floor - max) / scale).clamp(-40.0, 0.0).exp();
    for mv in all_moves {
        let idx = action_index(mirror_move(state.turn, *mv));
        if listed.contains(&idx) {
            continue;
        }
        total += floor_weight;
        weighted.push((idx, floor_weight));
    }
    weighted
        .into_iter()
        .filter(|(_, weight)| *weight > 0.0)
        .map(|(idx, weight)| (idx, weight / total))
        .collect()
}

fn write_record(w: &mut impl Write, z: f32, features: &[f32], policy: &[(usize, f32)]) {
    let f: Vec<String> = features.iter().map(|x| format!("{x:.4}")).collect();
    let pi: Vec<String> = policy
        .iter()
        .map(|(i, p)| format!("[{i},{p:.5}]"))
        .collect();
    writeln!(
        w,
        "{{\"z\":{z:.5},\"f\":[{}],\"pi\":[{}]}}",
        f.join(","),
        pi.join(",")
    )
    .expect("write record");
}
