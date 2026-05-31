//! Self-play data generator. Plays MCTS-vs-MCTS games and writes one JSONL
//! record per move: `{"z": outcome, "f": [features], "pi": [[action, prob], ...]}`
//! all in the side-to-move me-frame, ready for the Python trainer.
//!
//! Usage: selfplay_data [games] [sims] [out.jsonl] [weights.safetensors]
//!   games    number of self-play games          (default 50)
//!   sims     MCTS simulations per move           (default 160)
//!   out      output path                         (default selfplay.jsonl)
//!   weights  optional net (needs `--features net`); omit to use the heuristic
//!            bootstrap. Passing the previous iteration's weights here is how the
//!            self-play loop compounds: net -> data -> train -> better net.
//!
//! `z` is the final outcome from the recorded state's side-to-move POV
//! (+1 win, -1 loss, 0 draw) — the value-head target. `pi` is the visit-count
//! distribution — the policy-head target.

use std::fs::{self, File};
use std::io::{BufWriter, Write};

use wallchess_core::{
    action_index, distance_to_goal, encode, legal_moves, mirror_move, parse_state_key,
    HeuristicPolicy, Mcts, MctsConfig, Move, PolicyValue, Side, State,
};

/// One training sample, missing only the outcome until the game ends.
struct Sample {
    features: Vec<f32>,
    policy: Vec<(usize, f32)>, // (me-frame action index, probability)
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
    fn unit(&mut self) -> f32 {
        // 24 random bits -> [0,1)
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

const MAX_PLIES: u32 = 80; // cap games short and score the cap by race progress
                           // (see `race_winner`), so stalling/wall-spam becomes a
                           // loss, not a passive draw. Net learns wasted walls hurt.
const TEMP_PLIES: u32 = 16; // sample proportionally this many plies, then argmax
const OPENING_MIN: u32 = 2; // randomized opening: min unrecorded random plies
const OPENING_MAX: u32 = 8; // ...max. Diversifies positions so games don't all
                            // collapse to the same South rush -> value head sees
                            // both sides win, learns the board not the turn bit.

fn main() {
    let mut a = std::env::args().skip(1);
    let games: u32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(50);
    let sims: u32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(160);
    let out = a.next().unwrap_or_else(|| "selfplay.jsonl".to_string());
    let weights = a.next();

    let cfg = MctsConfig {
        sims,
        ..MctsConfig::default()
    };
    let opening_states = load_opening_states();
    let file = File::create(&out).expect("create output file");
    let mut w = BufWriter::new(file);

    match weights {
        #[cfg(feature = "net")]
        Some(path) => {
            let net = wallchess_core::net::NetEvaluator::load(&path).expect("load net weights");
            eprintln!("self-play policy: net {path}");
            play_games(&net, games, cfg, &opening_states, &mut w, &out);
        }
        #[cfg(not(feature = "net"))]
        Some(_) => {
            eprintln!("weights given but built without `--features net`; rebuild to use a net");
            std::process::exit(2);
        }
        None => {
            let prior_depth = env_parse("HEURISTIC_DEPTH", 1);
            let value_scale = env_parse("HEURISTIC_VALUE_SCALE", 400.0);
            eprintln!(
                "self-play policy: heuristic bootstrap depth={prior_depth} value_scale={value_scale}"
            );
            play_games(
                &HeuristicPolicy::new(prior_depth, value_scale),
                games,
                cfg,
                &opening_states,
                &mut w,
                &out,
            );
        }
    }
}

fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn play_games<P: PolicyValue>(
    policy: &P,
    games: u32,
    cfg: MctsConfig,
    opening_states: &[State],
    w: &mut BufWriter<File>,
    out: &str,
) {
    // Per-shard seed via env so parallel processes don't replay identical games.
    let seed = std::env::var("SELFPLAY_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|s| *s != 0) // xorshift is stuck at 0
        .unwrap_or(0x1234_5678_9abc_def0);
    let mut rng = Rng(seed);
    let mut total_samples = 0u64;

    for g in 0..games {
        let mut mcts = Mcts::new(policy, cfg);
        let mut state = pick_start_state(opening_states, &mut rng);
        let mut samples: Vec<Sample> = Vec::new();
        let mut ply = 0u32;

        if opening_states.is_empty() {
            // Randomized opening: play k uniform-random legal plies (unrecorded).
            let span = (OPENING_MAX - OPENING_MIN + 1) as u64;
            let opening = OPENING_MIN + (rng.next() % span) as u32;
            for _ in 0..opening {
                if state.winner.is_some() {
                    break;
                }
                let moves = legal_moves(&state);
                if moves.is_empty() {
                    break;
                }
                let pick = (rng.next() % moves.len() as u64) as usize;
                state = state.apply(moves[pick]);
                ply += 1;
            }
        }

        while state.winner.is_none() && ply < MAX_PLIES {
            let visits = mcts.run(&state);
            let total: u32 = visits.iter().map(|(_, n)| n).sum();
            if total == 0 {
                break;
            }

            // Policy target: visit counts -> probabilities, in the me-frame.
            let policy_target: Vec<(usize, f32)> = visits
                .iter()
                .filter(|(_, n)| *n > 0)
                .map(|(mv, n)| {
                    let idx = action_index(mirror_move(state.turn, *mv));
                    (idx, *n as f32 / total as f32)
                })
                .collect();
            samples.push(Sample {
                features: encode(&state),
                policy: policy_target,
                turn: state.turn,
            });

            // Move selection: proportional to visits early (exploration), then
            // greedily pick the most-visited move.
            let chosen = if ply < TEMP_PLIES {
                sample_proportional(&visits, total, &mut rng)
            } else {
                visits
                    .iter()
                    .max_by_key(|(_, n)| *n)
                    .map(|(m, _)| *m)
                    .unwrap()
            };
            state = state.apply(chosen);
            ply += 1;
        }

        // Backfill outcome from each sample's own side-to-move POV. A game that
        // hit the ply cap with no natural winner is decided by race progress so
        // passivity loses instead of drawing.
        let winner = state.winner.or_else(|| race_winner(&state));
        for s in &samples {
            let z = match winner {
                Some(win) if win == s.turn => 1.0f32,
                Some(_) => -1.0,
                None => 0.0,
            };
            write_record(w, z, &s.features, &s.policy);
            total_samples += 1;
        }

        let result = match winner {
            Some(Side::South) => "S",
            Some(Side::North) => "N",
            None => "draw",
        };
        eprintln!(
            "game {:>4}/{games}  plies {ply:>3}  result {result}  samples {}",
            g + 1,
            samples.len()
        );
    }

    w.flush().expect("flush");
    eprintln!("wrote {total_samples} samples to {out}");
}

fn pick_start_state(opening_states: &[State], rng: &mut Rng) -> State {
    if opening_states.is_empty() {
        return State::initial();
    }
    let pick = (rng.next() % opening_states.len() as u64) as usize;
    opening_states[pick]
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

/// Decide a capped (no natural winner) game by race progress: whoever is fewer
/// BFS steps from their goal wins. Equal distance stays a true draw.
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

fn sample_proportional(visits: &[(Move, u32)], total: u32, rng: &mut Rng) -> Move {
    let mut target = rng.unit() * total as f32;
    for (mv, n) in visits {
        target -= *n as f32;
        if target <= 0.0 {
            return *mv;
        }
    }
    visits.last().map(|(m, _)| *m).unwrap()
}

fn write_record(w: &mut impl Write, z: f32, features: &[f32], policy: &[(usize, f32)]) {
    let f: Vec<String> = features.iter().map(|x| format!("{x:.4}")).collect();
    let pi: Vec<String> = policy
        .iter()
        .map(|(i, p)| format!("[{i},{p:.5}]"))
        .collect();
    writeln!(
        w,
        "{{\"z\":{z:.1},\"f\":[{}],\"pi\":[{}]}}",
        f.join(","),
        pi.join(",")
    )
    .expect("write record");
}

#[cfg(test)]
mod tests {
    use super::*;
    use wallchess_core::Cell;

    #[test]
    fn parses_initial_state_key() {
        let state = parse_state_key("s|15|95|1010|").expect("parse initial");
        assert_eq!(state.turn, Side::South);
        assert_eq!(state.pawn(Side::South), Cell::new(1, 5));
        assert_eq!(state.pawn(Side::North), Cell::new(9, 5));
        assert_eq!(state.walls_left, [10, 10]);
        assert_eq!(state.winner, None);
    }

    #[test]
    fn parses_walls_and_single_digit_wall_counts() {
        let state = parse_state_key("n|15|95|910|h11,v22").expect("parse walls");
        assert_eq!(state.turn, Side::North);
        assert_eq!(state.walls_left, [9, 10]);
        assert!(state.has_h_wall(1, 1));
        assert!(state.has_v_wall(2, 2));
    }
}
