//! Count unique states reachable when the graph is generated with the same
//! pruning the UI uses (opening book + top-K, useless walls + blunders dropped),
//! expanded breadth-first to depth D. Transpositions are deduped by state.

use std::collections::{HashMap, HashSet, VecDeque};

use wallchess_core::{top_moves, State};

fn main() {
    let max_depth: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(8);
    let k: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(3);
    let margin: i32 = std::env::args()
        .nth(3)
        .and_then(|a| a.parse().ok())
        .unwrap_or(80);
    let depth: u8 = 2; // search depth used to rank moves

    let root = State::initial();
    let mut seen: HashSet<State> = HashSet::new();
    let mut by_depth: HashMap<usize, usize> = HashMap::new();
    let mut edges: u64 = 0;
    let mut q: VecDeque<(State, usize)> = VecDeque::new();
    seen.insert(root);
    q.push_back((root, 0));
    *by_depth.entry(0).or_default() += 1;

    while let Some((s, d)) = q.pop_front() {
        if d >= max_depth {
            continue;
        }
        for r in top_moves(&s, depth, k, margin, 200.0) {
            edges += 1;
            let child = s.apply(r.mv);
            if seen.insert(child) {
                *by_depth.entry(d + 1).or_default() += 1;
                q.push_back((child, d + 1));
            }
        }
    }

    println!("pruning: top-K={k}, blunder margin={margin}, rank depth={depth}");
    println!("expanded breadth-first to depth {max_depth}\n");
    let mut cum = 0usize;
    for d in 0..=max_depth {
        let n = *by_depth.get(&d).unwrap_or(&0);
        cum += n;
        println!("depth {d:2}: {n:>7} new states   (cumulative {cum})");
    }
    println!("\nTOTAL unique states: {}", seen.len());
    println!("TOTAL edges:         {edges}");
}
