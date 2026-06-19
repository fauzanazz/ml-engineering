//! Expert-iteration RL data generator: the net (MCTS) plays VS the d3 heuristic
//! and we record the NET's moves labelled by the GAME OUTCOME vs d3.
//!
//! Unlike `selfplay_data` (net vs net — bootstrap-capped) and `search_data`
//! (imitate the teacher — imitation-capped), here the opponent is the fixed
//! strong target (d3). Positions come from the distribution that matters for
//! beating d3, and the outcome z is a real signal about beating d3.
//!
//! Output is SPLIT by outcome so training can upweight winning behaviour:
//!   <out>.win.jsonl   net-move samples from games the net WON  (policy + value)
//!   <out>.loss.jsonl  net-move samples from games the net LOST (value only)
//! Train: --data <anchor> <out>.win.jsonl   --value-data <out>.loss.jsonl
//!
//! Usage: rl_data <net.safetensors> [games] [sims] [out_prefix] [hdepth]
//!   (needs --features net)

use std::fs::File;
use std::io::{BufWriter, Write};

use wallchess_core::{
    action_index, distance_to_goal, encode, eval::Heuristic, legal_moves, mirror_move, Mcts,
    MctsConfig, Move, PolicyValue, Search, Side, State,
};

const MAX_PLIES: u32 = 140;
const TEMP_PLIES: u32 = 8; // light exploration early, then argmax
const OPENING_MIN: u32 = 2;
const OPENING_MAX: u32 = 8;
const FLOOR_VISITS: f32 = 0.1;

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

struct Sample {
    features: Vec<f32>,
    policy: Vec<(usize, f32)>,
}

fn main() {
    #[cfg(not(feature = "net"))]
    {
        eprintln!("rebuild with --features net");
        std::process::exit(2);
    }
    #[cfg(feature = "net")]
    {
        let mut a = std::env::args().skip(1);
        let net_path = a
            .next()
            .expect("usage: rl_data <net> [games] [sims] [out_prefix] [hdepth]");
        let games: u32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(100);
        let sims: u32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(800);
        let out_prefix = a.next().unwrap_or_else(|| "/tmp/rl".to_string());
        let hdepth: u8 = a.next().and_then(|s| s.parse().ok()).unwrap_or(3);

        let net = wallchess_core::net::NetEvaluator::load(&net_path).expect("load net");
        let heuristic = Heuristic::default();
        let cfg = MctsConfig {
            sims,
            root_noise: 0.05,
            ..MctsConfig::default()
        };

        let seed = std::env::var("RL_SEED")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|s| *s != 0)
            .unwrap_or(0xC0FF_EE12_3456_789A);
        let mut rng = Rng(seed);

        let win_file = File::create(format!("{out_prefix}.win.jsonl")).expect("create win file");
        let loss_file = File::create(format!("{out_prefix}.loss.jsonl")).expect("create loss file");
        let mut win_w = BufWriter::new(win_file);
        let mut loss_w = BufWriter::new(loss_file);

        let (mut net_wins, mut net_losses, mut draws) = (0u32, 0u32, 0u32);
        let (mut win_samples, mut loss_samples) = (0u64, 0u64);

        for g in 0..games {
            let net_side = if g % 2 == 0 { Side::South } else { Side::North };
            let mut state = State::initial();

            // randomized opening (unrecorded)
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
                state = state.apply(moves[(rng.next() % moves.len() as u64) as usize]);
            }

            let mut mcts = Mcts::new(&net, cfg);
            let mut samples: Vec<Sample> = Vec::new();
            let mut ply = 0u32;

            while state.winner.is_none() && ply < MAX_PLIES {
                let chosen = if state.turn == net_side {
                    let visits = mcts.run(&state);
                    let total: u32 = visits.iter().map(|(_, n)| n).sum();
                    if total == 0 {
                        break;
                    }
                    // Record policy over PRODUCTIVE moves only (matches the guarded play
                    // the net is actually rewarded for) — non-productive walls get floor.
                    samples.push(Sample {
                        features: encode(&state),
                        policy: visit_policy(&state, &visits, total),
                    });
                    // Play at arena strength: guarded MCTS (override a non-productive top
                    // move with the best productive one) so the net wins ~as in arena and
                    // the win-data is dense, not the ~11% the handicapped raw net gives.
                    if ply < TEMP_PLIES {
                        sample_proportional(&visits, total, &mut rng)
                    } else {
                        guarded_move(&state, &visits)
                    }
                } else {
                    match Search::new(&heuristic).search(&state, hdepth).best {
                        Some(m) => m,
                        None => break,
                    }
                };
                state = state.apply(chosen);
                ply += 1;
            }

            let winner = state.winner.or_else(|| race_winner(&state));
            let (z, w): (f32, &mut BufWriter<File>) = match winner {
                Some(win) if win == net_side => {
                    net_wins += 1;
                    win_samples += samples.len() as u64;
                    (1.0, &mut win_w)
                }
                Some(_) => {
                    net_losses += 1;
                    loss_samples += samples.len() as u64;
                    (-1.0, &mut loss_w)
                }
                None => {
                    draws += 1;
                    continue;
                }
            };
            for s in &samples {
                write_record(w, z, &s.features, &s.policy);
            }
            if g % 20 == 0 {
                eprintln!("game {g}/{games} net={net_side:?} plies={ply} W{net_wins}-L{net_losses}-D{draws}");
            }
        }
        win_w.flush().ok();
        loss_w.flush().ok();
        eprintln!(
            "DONE: net {net_wins}-{net_losses} (draws {draws}) of {games}  win_samples={win_samples} loss_samples={loss_samples}"
        );
    }
}

// Policy target over PRODUCTIVE moves only: non-productive moves (e.g. walls that
// neither advance the net nor delay d3) are dropped to the floor. This teaches the
// guarded, winning behaviour rather than the raw MCTS top that the guard overrides.
#[cfg(feature = "net")]
fn visit_policy(state: &State, visits: &[(Move, u32)], _total: u32) -> Vec<(usize, f32)> {
    let prod: Vec<(Move, u32)> = visits
        .iter()
        .copied()
        .filter(|(mv, n)| *n > 0 && is_productive(state, *mv))
        .collect();
    let prod_total: u32 = prod.iter().map(|(_, n)| n).sum();
    use std::collections::HashSet;
    let kept: HashSet<usize> = prod
        .iter()
        .map(|(mv, _)| action_index(mirror_move(state.turn, *mv)))
        .collect();
    let floored = legal_moves(state)
        .iter()
        .filter(|mv| !kept.contains(&action_index(mirror_move(state.turn, **mv))))
        .count() as f32;
    // Fall back to all visited if nothing is productive (rare).
    let (src, src_total): (Vec<(Move, u32)>, u32) = if prod_total > 0 {
        (prod, prod_total)
    } else {
        let v: Vec<(Move, u32)> = visits.iter().copied().filter(|(_, n)| *n > 0).collect();
        let t = v.iter().map(|(_, n)| n).sum();
        (v, t)
    };
    let total_eff = src_total as f32 + FLOOR_VISITS * floored;
    let mut pol: Vec<(usize, f32)> = src
        .iter()
        .map(|(mv, n)| {
            (
                action_index(mirror_move(state.turn, *mv)),
                *n as f32 / total_eff,
            )
        })
        .collect();
    for mv in legal_moves(state) {
        let idx = action_index(mirror_move(state.turn, mv));
        if !pol.iter().any(|(i, _)| *i == idx) {
            pol.push((idx, FLOOR_VISITS / total_eff));
        }
    }
    pol
}

// Best-visited move, overriding a non-productive top pick with the best productive
// one (mirrors net_arena's guard). Keeps the net at arena strength.
#[cfg(feature = "net")]
fn guarded_move(state: &State, visits: &[(Move, u32)]) -> Move {
    let best = visits
        .iter()
        .max_by_key(|(_, n)| *n)
        .map(|(m, _)| *m)
        .unwrap();
    if is_productive(state, best) {
        return best;
    }
    visits
        .iter()
        .filter(|(mv, n)| *n > 0 && is_productive(state, *mv))
        .max_by_key(|(_, n)| *n)
        .map(|(m, _)| *m)
        .or_else(|| {
            legal_moves(state)
                .into_iter()
                .filter(|mv| is_productive(state, *mv))
                .max_by_key(|mv| {
                    let c = state.apply(*mv);
                    side_dist(&c, state.turn.other()) as i32 - side_dist(&c, state.turn) as i32
                })
        })
        .unwrap_or(best)
}

#[cfg(feature = "net")]
fn is_productive(state: &State, mv: Move) -> bool {
    let side = state.turn;
    let (bm, bo) = (side_dist(state, side), side_dist(state, side.other()));
    let child = state.apply(mv);
    if child.winner == Some(side) {
        return true;
    }
    side_dist(&child, side) < bm || side_dist(&child, side.other()) > bo
}

#[cfg(feature = "net")]
fn side_dist(state: &State, side: Side) -> u16 {
    distance_to_goal(state, state.pawn(side), side.goal_row()).unwrap_or(u16::MAX)
}

#[cfg(feature = "net")]
fn sample_proportional(visits: &[(Move, u32)], total: u32, rng: &mut Rng) -> Move {
    let mut r = (rng.next() % total.max(1) as u64) as i64;
    for (mv, n) in visits {
        r -= *n as i64;
        if r < 0 {
            return *mv;
        }
    }
    visits[0].0
}

#[cfg(feature = "net")]
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

#[cfg(feature = "net")]
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
    .expect("write");
}
