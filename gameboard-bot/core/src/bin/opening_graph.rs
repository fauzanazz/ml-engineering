//! Generate a compact opening graph for parallel training/search jobs.
//!
//! Usage:
//!   opening_graph [out.jsonl] [max_depth] [top_k] [rank_depth] [margin] [max_nodes]
//!
//! Output is JSONL:
//!   - one meta record
//!   - node records with sparse positional encoding `pe=[[feature_index,value], ...]`
//!   - edge records with me-frame action index `a`

use std::collections::{HashSet, VecDeque};
use std::fs::File;
use std::io::{BufWriter, Write};

use wallchess_core::{
    action_index, encode, eval::Heuristic, legal_moves, mirror_move, state_key, Search, State,
    ACTION_COUNT, FEATURE_LEN,
};

#[derive(Clone)]
struct Node {
    key: String,
    state: State,
    ply: u32,
    expanded: bool,
}

struct Edge {
    from: String,
    to: String,
    action: usize,
    score: i32,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args
        .next()
        .unwrap_or_else(|| "opening_graph.jsonl".to_string());
    let max_depth: u32 = parse_arg(args.next(), 6);
    let top_k: usize = parse_arg(args.next(), 8);
    let rank_depth: u8 = parse_arg(args.next(), 2);
    let margin: i32 = parse_arg(args.next(), 300);
    let max_nodes: usize = parse_arg(args.next(), 10_000);

    let (nodes, edges, capped) = generate(max_depth, top_k, rank_depth, margin, max_nodes);
    write_jsonl(
        &out, &nodes, &edges, capped, max_depth, top_k, rank_depth, margin, max_nodes,
    );
    eprintln!(
        "wrote {} nodes and {} edges to {out} (capped={capped})",
        nodes.len(),
        edges.len()
    );
}

fn parse_arg<T: std::str::FromStr>(arg: Option<String>, default: T) -> T {
    arg.and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn generate(
    max_depth: u32,
    top_k: usize,
    rank_depth: u8,
    margin: i32,
    max_nodes: usize,
) -> (Vec<Node>, Vec<Edge>, bool) {
    let heuristic = Heuristic::default();
    let mut search = Search::new(&heuristic);
    let root = State::initial();
    let root_key = state_key(&root);

    let mut nodes = vec![Node {
        key: root_key.clone(),
        state: root,
        ply: 0,
        expanded: false,
    }];
    let mut edges = Vec::new();
    let mut seen = HashSet::from([root_key]);
    let mut queue = VecDeque::from([0usize]);
    let mut capped = false;

    while let Some(idx) = queue.pop_front() {
        let state = nodes[idx].state;
        let ply = nodes[idx].ply;
        if ply >= max_depth || state.winner.is_some() {
            continue;
        }

        let ranked = ranked_moves(&mut search, &state, rank_depth, top_k, margin);
        let new_children = ranked
            .iter()
            .filter(|(mv, _)| !seen.contains(&state_key(&state.apply(*mv))))
            .count();
        if nodes.len() + new_children > max_nodes {
            capped = true;
            break;
        }

        let from = nodes[idx].key.clone();
        for (mv, score) in ranked {
            let child = state.apply(mv);
            let to = state_key(&child);
            if seen.insert(to.clone()) {
                nodes.push(Node {
                    key: to.clone(),
                    state: child,
                    ply: ply + 1,
                    expanded: false,
                });
                queue.push_back(nodes.len() - 1);
            }
            edges.push(Edge {
                from: from.clone(),
                to,
                action: action_index(mirror_move(state.turn, mv)),
                score,
            });
        }
        nodes[idx].expanded = true;
    }

    (nodes, edges, capped)
}

fn ranked_moves(
    search: &mut Search<'_, Heuristic>,
    state: &State,
    rank_depth: u8,
    top_k: usize,
    margin: i32,
) -> Vec<(wallchess_core::Move, i32)> {
    let alpha = -1_000_000 * 2;
    let beta = 1_000_000 * 2;
    let mut ranked: Vec<_> = legal_moves(state)
        .into_iter()
        .map(|mv| {
            let child = state.apply(mv);
            let score = -search.search(&child, rank_depth.saturating_sub(1)).score;
            (mv, score)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    let best = ranked.first().map(|(_, score)| *score).unwrap_or(alpha);
    ranked
        .into_iter()
        .filter(|(_, score)| best - *score <= margin || *score >= beta)
        .take(top_k.max(1))
        .collect()
}

fn write_jsonl(
    out: &str,
    nodes: &[Node],
    edges: &[Edge],
    capped: bool,
    max_depth: u32,
    top_k: usize,
    rank_depth: u8,
    margin: i32,
    max_nodes: usize,
) {
    let file = File::create(out).expect("create opening graph");
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "{{\"type\":\"meta\",\"format\":\"wallchess-opening-graph-v1\",\"feature_len\":{FEATURE_LEN},\"action_count\":{ACTION_COUNT},\"max_depth\":{max_depth},\"top_k\":{top_k},\"rank_depth\":{rank_depth},\"margin\":{margin},\"max_nodes\":{max_nodes},\"nodes\":{},\"edges\":{},\"capped\":{capped}}}",
        nodes.len(),
        edges.len(),
    )
    .expect("write meta");

    for node in nodes {
        writeln!(
            w,
            "{{\"type\":\"node\",\"key\":\"{}\",\"ply\":{},\"expanded\":{},\"pe\":[{}]}}",
            node.key,
            node.ply,
            node.expanded,
            sparse_encoding(&node.state)
        )
        .expect("write node");
    }

    for edge in edges {
        writeln!(
            w,
            "{{\"type\":\"edge\",\"from\":\"{}\",\"to\":\"{}\",\"a\":{},\"score\":{}}}",
            edge.from, edge.to, edge.action, edge.score
        )
        .expect("write edge");
    }
    w.flush().expect("flush opening graph");
}

fn sparse_encoding(state: &State) -> String {
    encode(state)
        .into_iter()
        .enumerate()
        .filter(|(_, value)| *value != 0.0)
        .map(|(idx, value)| format!("[{idx},{value:.4}]"))
        .collect::<Vec<_>>()
        .join(",")
}
