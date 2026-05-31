//! Generate a compact counter-book against the default heuristic opponent.
//!
//! The opponent is modeled as deterministic alpha-beta. The book stores only
//! our side-to-move states, so runtime lookup is an O(1) state-key check before
//! falling back to the small net.
//!
//! Usage:
//!   counter_book [out.jsonl] [bot_turns] [candidates] [opponent_depth] [line_plies]

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

use wallchess_core::{
    action_index, distance_to_goal, eval::Heuristic, eval::WIN_SCORE, mirror_move, state_key,
    Evaluator, Move, Search, Side, State,
};

#[derive(Clone, Copy)]
struct Entry {
    score: i32,
    best: Option<Move>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args
        .next()
        .unwrap_or_else(|| "counter_book.jsonl".to_string());
    let bot_turns: u8 = parse_arg(args.next(), 5);
    let candidates: usize = parse_arg(args.next(), 8);
    let opponent_depth: u8 = parse_arg(args.next(), 2);
    let line_plies: u32 = parse_arg(args.next(), 0);

    let heuristic = Heuristic::default();
    let mut memo: HashMap<(State, Side, u8), Entry> = HashMap::new();
    for bot_side in [Side::South, Side::North] {
        let Some(start) = first_bot_state(bot_side, opponent_depth, &heuristic) else {
            continue;
        };
        let score = solve(
            start,
            bot_side,
            bot_turns,
            candidates,
            opponent_depth,
            &heuristic,
            &mut memo,
        );
        eprintln!("solved bot={bot_side:?} score={score}");
        if line_plies > 0 {
            extend_principal_line(
                start,
                bot_side,
                bot_turns,
                candidates,
                opponent_depth,
                line_plies,
                &heuristic,
                &mut memo,
            );
        }
    }

    write_jsonl(
        &out,
        &memo,
        bot_turns,
        candidates,
        opponent_depth,
        line_plies,
    );
    eprintln!("wrote counter book to {out}");
}

fn parse_arg<T: std::str::FromStr>(arg: Option<String>, default: T) -> T {
    arg.and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn first_bot_state(bot_side: Side, opponent_depth: u8, heuristic: &Heuristic) -> Option<State> {
    let mut state = State::initial();
    while state.turn != bot_side && state.winner.is_none() {
        let mv = choose_heuristic(&state, opponent_depth, heuristic)?;
        state = state.apply(mv);
    }
    Some(state)
}

fn solve(
    state: State,
    bot_side: Side,
    bot_turns: u8,
    candidates: usize,
    opponent_depth: u8,
    heuristic: &Heuristic,
    memo: &mut HashMap<(State, Side, u8), Entry>,
) -> i32 {
    if let Some(score) = terminal_score(&state, bot_side) {
        return score;
    }
    if bot_turns == 0 {
        return leaf_score(&state, bot_side, heuristic);
    }
    let key = (state, bot_side, bot_turns);
    if let Some(entry) = memo.get(&key) {
        return entry.score;
    }

    let mut best_score = -WIN_SCORE * 2;
    let mut best_move = None;
    for mv in candidate_moves(&state, candidates) {
        let child = state.apply(mv);
        let score = match terminal_score(&child, bot_side) {
            Some(score) => score,
            None => match opponent_reply(child, bot_side, opponent_depth, heuristic) {
                Some(reply_state) => solve(
                    reply_state,
                    bot_side,
                    bot_turns - 1,
                    candidates,
                    opponent_depth,
                    heuristic,
                    memo,
                ),
                None => leaf_score(&child, bot_side, heuristic),
            },
        };
        if score > best_score {
            best_score = score;
            best_move = Some(mv);
        }
    }

    memo.insert(
        key,
        Entry {
            score: best_score,
            best: best_move,
        },
    );
    best_score
}

fn extend_principal_line(
    mut state: State,
    bot_side: Side,
    bot_turns: u8,
    candidates: usize,
    opponent_depth: u8,
    line_plies: u32,
    heuristic: &Heuristic,
    memo: &mut HashMap<(State, Side, u8), Entry>,
) {
    let mut plies = 0u32;
    while state.winner.is_none() && plies < line_plies {
        if state.turn != bot_side {
            let Some(next) = opponent_reply(state, bot_side, opponent_depth, heuristic) else {
                break;
            };
            state = next;
            plies += 1;
            continue;
        }
        solve(
            state,
            bot_side,
            bot_turns,
            candidates,
            opponent_depth,
            heuristic,
            memo,
        );
        let Some(best) = memo
            .get(&(state, bot_side, bot_turns))
            .and_then(|entry| entry.best)
        else {
            break;
        };
        state = state.apply(best);
        plies += 1;
    }
    eprintln!(
        "extended bot={bot_side:?} plies={plies} winner={:?} race={}",
        state.winner,
        race_score(&state, bot_side)
    );
}

fn opponent_reply(
    state: State,
    bot_side: Side,
    opponent_depth: u8,
    heuristic: &Heuristic,
) -> Option<State> {
    if state.turn == bot_side || state.winner.is_some() {
        return Some(state);
    }
    let mv = choose_heuristic(&state, opponent_depth, heuristic)?;
    Some(state.apply(mv))
}

fn choose_heuristic(state: &State, depth: u8, heuristic: &Heuristic) -> Option<Move> {
    let mut search = Search::new(heuristic);
    search.search(state, depth).best
}

fn candidate_moves(state: &State, candidates: usize) -> Vec<Move> {
    let heuristic = Heuristic::default();
    let mut search = Search::new(&heuristic);
    let opp = state.turn.other();
    let opp_dist_before =
        distance_to_goal(state, state.pawn(opp), opp.goal_row()).unwrap_or(u16::MAX);
    let mut ranked = search.ranked(state, 1);
    ranked.retain(|(mv, _)| match mv {
        Move::Pawn(_) => true,
        Move::Wall(_) => {
            let child = state.apply(*mv);
            let after =
                distance_to_goal(&child, child.pawn(opp), opp.goal_row()).unwrap_or(u16::MAX);
            after > opp_dist_before
        }
    });
    ranked
        .into_iter()
        .take(candidates.max(1))
        .map(|(mv, _)| mv)
        .collect()
}

fn terminal_score(state: &State, bot_side: Side) -> Option<i32> {
    state.winner.map(|winner| {
        if winner == bot_side {
            WIN_SCORE
        } else {
            -WIN_SCORE
        }
    })
}

fn leaf_score(state: &State, bot_side: Side, heuristic: &Heuristic) -> i32 {
    if let Some(score) = terminal_score(state, bot_side) {
        return score;
    }
    let race = race_score(state, bot_side) * 100;
    race + heuristic.eval(state, bot_side) / 4
}

fn race_score(state: &State, side: Side) -> i32 {
    let far = u16::MAX;
    let d = |s: Side| distance_to_goal(state, state.pawn(s), s.goal_row()).unwrap_or(far) as i32;
    d(side.other()) - d(side)
}

fn write_jsonl(
    out: &str,
    memo: &HashMap<(State, Side, u8), Entry>,
    bot_turns: u8,
    candidates: usize,
    opponent_depth: u8,
    line_plies: u32,
) {
    let mut by_key: HashMap<String, (State, Side, u8, i32, Move)> = HashMap::new();
    for ((state, bot_side, depth), entry) in memo {
        let Some(best) = entry.best else {
            continue;
        };
        let key = state_key(state);
        let replace = by_key
            .get(&key)
            .map(|(_, _, existing_depth, _, _)| depth > existing_depth)
            .unwrap_or(true);
        if replace {
            by_key.insert(key, (*state, *bot_side, *depth, entry.score, best));
        }
    }

    let mut rows: Vec<(String, State, Side, u8, i32, Move)> = by_key
        .into_iter()
        .map(|(key, (state, bot_side, depth, score, best))| {
            (key, state, bot_side, depth, score, best)
        })
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    let file = File::create(out).expect("create counter book");
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "{{\"type\":\"meta\",\"format\":\"wallchess-counter-book-v1\",\"bot_turns\":{bot_turns},\"candidates\":{candidates},\"opponent_depth\":{opponent_depth},\"line_plies\":{line_plies},\"entries\":{}}}",
        rows.len()
    )
    .expect("write meta");
    for (key, state, bot_side, depth, score, best) in rows {
        let action = action_index(mirror_move(state.turn, best));
        writeln!(
            w,
            "{{\"type\":\"book\",\"key\":\"{key}\",\"a\":{action},\"score\":{score},\"bot\":\"{}\",\"depth\":{depth}}}",
            side_label(bot_side)
        )
        .expect("write row");
    }
    w.flush().expect("flush counter book");
}

fn side_label(side: Side) -> &'static str {
    match side {
        Side::South => "south",
        Side::North => "north",
    }
}
