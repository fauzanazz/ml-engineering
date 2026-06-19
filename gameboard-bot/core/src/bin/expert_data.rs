//! Fast expert-trace data from alpha-beta search.
//!
//! This is cheaper than MCTS self-play: each turn records the move chosen by a
//! fixed-depth search, then backfills the final race-scored outcome. It is a
//! practical baseline-imitation stage before policy/value self-play.
//!
//! Usage:
//!   expert_data [games] [depth] [out.jsonl] [max_plies]
//!
//! Set `OPENING_GRAPH=/path/to/opening_graph.jsonl` to sample start states from
//! precomputed opening nodes. Set `EXPERT_DATA_SEED` for parallel shards.

use std::fs::{self, File};
use std::io::{BufWriter, Write};

use wallchess_core::{
    action_index, distance_to_goal, encode, eval::Heuristic, mirror_move, parse_state_key, Search,
    Side, State,
};

struct Sample {
    features: Vec<f32>,
    action: usize,
    turn: Side,
}

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
    let games: u32 = parse_arg(args.next(), 100);
    let depth: u8 = parse_arg(args.next(), 2);
    let out = args
        .next()
        .unwrap_or_else(|| "expert_data.jsonl".to_string());
    let max_plies: u32 = parse_arg(args.next(), 140);

    let opening_states = load_opening_states();
    let seed = std::env::var("EXPERT_DATA_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|s| *s != 0)
        .unwrap_or(0xace5_2026_5eed_f00d);
    let mut rng = Rng(seed);
    let heuristic = Heuristic::default();
    let file = File::create(&out).expect("create expert-data output");
    let mut w = BufWriter::new(file);
    let mut total_samples = 0u64;

    for game in 0..games {
        let mut state = start_state(&opening_states, &mut rng);
        let mut samples = Vec::new();
        let mut plies = 0u32;

        while state.winner.is_none() && plies < max_plies {
            let mut search = Search::new(&heuristic);
            let Some(mv) = search.search(&state, depth).best else {
                break;
            };
            samples.push(Sample {
                features: encode(&state),
                action: action_index(mirror_move(state.turn, mv)),
                turn: state.turn,
            });
            state = state.apply(mv);
            plies += 1;
        }

        let winner = state.winner.or_else(|| race_winner(&state));
        for sample in &samples {
            let z = match winner {
                Some(side) if side == sample.turn => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            };
            write_record(&mut w, z, &sample.features, sample.action);
            total_samples += 1;
        }

        let result = match winner {
            Some(Side::South) => "S",
            Some(Side::North) => "N",
            None => "draw",
        };
        eprintln!(
            "game {:>4}/{games} plies {plies:>3} result {result} samples {}",
            game + 1,
            samples.len()
        );
    }

    w.flush().expect("flush expert data");
    eprintln!("wrote {total_samples} samples to {out}");
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

fn race_winner(state: &State) -> Option<Side> {
    let far = u16::MAX;
    let d = |side: Side| distance_to_goal(state, state.pawn(side), side.goal_row()).unwrap_or(far);
    let (ds, dn) = (d(Side::South), d(Side::North));
    match ds.cmp(&dn) {
        std::cmp::Ordering::Less => Some(Side::South),
        std::cmp::Ordering::Greater => Some(Side::North),
        std::cmp::Ordering::Equal => None,
    }
}

fn write_record(w: &mut impl Write, z: f32, features: &[f32], action: usize) {
    let f: Vec<String> = features.iter().map(|x| format!("{x:.4}")).collect();
    writeln!(
        w,
        "{{\"z\":{z:.1},\"f\":[{}],\"pi\":[[{action},1.00000]]}}",
        f.join(","),
    )
    .expect("write record");
}
