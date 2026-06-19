//! Stdin/stdout move engine for cross-binary matches (referee = `xmatch`).
//!
//! Usage: bestmove <depth> [node_limit]
//!   node_limit 0 (or omitted) → fixed-depth search; >0 → budgeted search.
//!
//! Protocol (one game position per line on stdin):
//!   h_walls v_walls p0r p0c p1r p1c wl0 wl1 turn
//!     - h_walls/v_walls: u64 wall bitsets
//!     - p0=South pawn, p1=North pawn
//!     - wl0/wl1: walls left for South/North
//!     - turn: 0=South, 1=North
//! Response (one line per query):
//!   P r c        (pawn move to r,c)
//!   H r c        (horizontal wall anchored r,c)
//!   V r c        (vertical wall anchored r,c)
//!   NONE         (no legal move)

use std::io::{self, BufRead, Write};

use gameboard_core::eval::Heuristic;
use gameboard_core::search::{Search, SearchConfig};
use gameboard_core::state::{Cell, Move, Orientation, Side, State};

fn parse_state(line: &str) -> Option<State> {
    let f: Vec<&str> = line.split_whitespace().collect();
    if f.len() != 9 {
        return None;
    }
    let h_walls: u64 = f[0].parse().ok()?;
    let v_walls: u64 = f[1].parse().ok()?;
    let p0 = Cell::new(f[2].parse().ok()?, f[3].parse().ok()?);
    let p1 = Cell::new(f[4].parse().ok()?, f[5].parse().ok()?);
    let wl0: u8 = f[6].parse().ok()?;
    let wl1: u8 = f[7].parse().ok()?;
    let turn = if f[8] == "0" { Side::South } else { Side::North };
    Some(State {
        pawns: [p0, p1],
        h_walls,
        v_walls,
        walls_left: [wl0, wl1],
        turn,
        winner: None,
    })
}

fn fmt_move(mv: Move) -> String {
    match mv {
        Move::Pawn(c) => format!("P {} {}", c.r, c.c),
        Move::Wall(w) => {
            let o = match w.o {
                Orientation::H => "H",
                Orientation::V => "V",
            };
            format!("{} {} {}", o, w.r, w.c)
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let depth: u8 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let node_limit: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    // Eval + pruning config both come from env vars (WC_EVAL_* / WC_*), logged
    // to stderr so the stdout move protocol stays clean for the xmatch referee.
    // A stripped environment reproduces the deployed default eval exactly.
    let eval = Heuristic::from_env();
    let config = SearchConfig::from_env();
    eprintln!("bestmove eval: {eval:?} | config: {}", config.summary());

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let state = match parse_state(&line) {
            Some(s) => s,
            None => {
                writeln!(out, "NONE").ok();
                out.flush().ok();
                continue;
            }
        };
        let mut s = Search::with_config(&eval, config);
        let best = if node_limit > 0 {
            s.search_with_node_limit(&state, depth, node_limit).best
        } else {
            s.search(&state, depth).best
        };
        match best {
            Some(mv) => writeln!(out, "{}", fmt_move(mv)).ok(),
            None => writeln!(out, "NONE").ok(),
        };
        out.flush().ok();
    }
}
