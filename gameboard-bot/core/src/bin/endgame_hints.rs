//! Generate compressed no-wall endgame race hints.
//!
//! Usage:
//!   endgame_hints [out.jsonl] [depth] [min_abs_score]
//!
//! This is deliberately compact: one record per decisive no-wall state, keyed by
//! `state_key`, with the me-frame best action and a side-relative score.

use std::fs::File;
use std::io::{BufWriter, Write};

use gameboard_core::{
    action_index, distance_to_goal, mirror_move, state_key, Cell, Heuristic, Search, Side, State,
};

const SIZE: u8 = 9;

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args
        .next()
        .unwrap_or_else(|| "endgame_hints.jsonl".to_string());
    let depth: u8 = parse_arg(args.next(), 8);
    let min_abs_score: i32 = parse_arg(args.next(), 300);

    let hints = generate(depth, min_abs_score);
    write_jsonl(&out, &hints, depth, min_abs_score);
    eprintln!(
        "wrote {} no-wall endgame hints to {out} (depth={depth}, min_abs_score={min_abs_score})",
        hints.len()
    );
}

fn parse_arg<T: std::str::FromStr>(arg: Option<String>, default: T) -> T {
    arg.and_then(|s| s.parse().ok()).unwrap_or(default)
}

struct Hint {
    key: String,
    action: usize,
    winner: Side,
    score: i32,
    dist_me: u16,
    dist_opp: u16,
}

fn generate(depth: u8, min_abs_score: i32) -> Vec<Hint> {
    let heuristic = Heuristic::default();
    let mut hints = Vec::new();
    for south_r in 1..=SIZE {
        for south_c in 1..=SIZE {
            let south = Cell::new(south_r, south_c);
            if south.r == Side::South.goal_row() {
                continue;
            }
            for north_r in 1..=SIZE {
                for north_c in 1..=SIZE {
                    let north = Cell::new(north_r, north_c);
                    if north.r == Side::North.goal_row() || north == south {
                        continue;
                    }
                    for turn in [Side::South, Side::North] {
                        let state = State {
                            pawns: [south, north],
                            h_walls: 0,
                            v_walls: 0,
                            walls_left: [0, 0],
                            turn,
                            winner: None,
                        };
                        if let Some(hint) = solve_hint(&heuristic, &state, depth, min_abs_score) {
                            hints.push(hint);
                        }
                    }
                }
            }
        }
    }
    hints
}

fn solve_hint(heuristic: &Heuristic, state: &State, depth: u8, min_abs_score: i32) -> Option<Hint> {
    let mut search = Search::new(heuristic);
    let result = search.search(state, depth);
    if result.score.abs() < min_abs_score {
        return None;
    }
    let mv = result.best?;
    let winner = if result.score > 0 {
        state.turn
    } else {
        state.turn.other()
    };
    let me = state.turn;
    let opp = me.other();
    Some(Hint {
        key: state_key(state),
        action: action_index(mirror_move(me, mv)),
        winner,
        score: result.score,
        dist_me: distance_to_goal(state, state.pawn(me), me.goal_row()).unwrap_or(u16::MAX),
        dist_opp: distance_to_goal(state, state.pawn(opp), opp.goal_row()).unwrap_or(u16::MAX),
    })
}

fn write_jsonl(out: &str, hints: &[Hint], depth: u8, min_abs_score: i32) {
    let file = File::create(out).expect("create endgame hints");
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "{{\"type\":\"meta\",\"format\":\"wallchess-endgame-hints-v1\",\"scope\":\"no-walls\",\"depth\":{depth},\"min_abs_score\":{min_abs_score},\"hints\":{}}}",
        hints.len()
    )
    .expect("write meta");
    for hint in hints {
        writeln!(
            w,
            "{{\"type\":\"hint\",\"key\":\"{}\",\"a\":{},\"winner\":\"{}\",\"score\":{},\"dist_me\":{},\"dist_opp\":{}}}",
            hint.key,
            hint.action,
            side_label(hint.winner),
            hint.score,
            hint.dist_me,
            hint.dist_opp,
        )
        .expect("write hint");
    }
    w.flush().expect("flush endgame hints");
}

fn side_label(side: Side) -> &'static str {
    match side {
        Side::South => "south",
        Side::North => "north",
    }
}
