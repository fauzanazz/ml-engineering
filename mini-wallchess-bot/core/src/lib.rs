//! Wall Chess (Quoridor) engine + search core.
//!
//! Modules:
//!   - `state`  : bit-packed board, move application
//!   - `moves`  : move generation, BFS distance, legality
//!   - `eval`   : leaf evaluation (`Evaluator` trait + hand heuristic)
//!   - `search` : negamax + alpha-beta + iterative deepening + transposition table

pub mod action;
pub mod arena;
pub mod books;
pub mod eval;
pub mod features;
pub mod mcts;
#[cfg(feature = "net")]
pub mod net;
pub mod moves;
pub mod search;
pub mod state;

pub use action::{action_index, index_to_move, ACTION_COUNT};
pub use books::EndgameBook;
pub use eval::{win_prob, Evaluator, Heuristic};
pub use features::{encode, mirror_move, FEATURE_LEN};
pub use mcts::{HeuristicPolicy, Mcts, MctsConfig, PolicyValue};
pub use moves::{distance_to_goal, legal_moves, pawn_moves};
pub use search::Search;
pub use state::{Cell, Move, Orientation, Side, State, Wall};

/// Pick the best move for the side to move and report the 0..100 win split.
/// Returns `(best_move, south_score, north_score)` with the two scores summing
/// to 100 — matching the "north 23 / south 77" framing.
pub fn analyze(state: &State, depth: u8, k: f64) -> (Option<Move>, u8, u8) {
    let h = Heuristic::default();
    let mut s = Search::new(&h);
    let res = s.search(state, depth);
    // res.score is from the side-to-move POV; convert to SOUTH's POV.
    let south_eval = match state.turn {
        Side::South => res.score,
        Side::North => -res.score,
    };
    let south = win_prob(south_eval, k);
    (res.best, south, 100 - south)
}

/// A pruned, scored successor for the state graph: the move, the resulting
/// state's SOUTH win-chance (0..100), and the side-to-move score it earned.
pub struct RankedMove {
    pub mv: Move,
    pub south: u8,
    pub score: i32,
}

/// Opening book — the perfect early-game move, memorized so no search is needed.
///
/// Wall Chess opens as a pure race: with no walls down and a clear lane, the
/// only non-blundering move is to advance the pawn straight toward your goal.
/// We return that move while the position is still a clean straight-line race
/// (pawn on its home column, full lane ahead, no walls placed yet). This makes
/// the opening of the graph a single canonical line instead of a search result
/// that the side-bias in the eval could otherwise distort.
pub fn opening_move(state: &State) -> Option<Move> {
    // walls are bit-packed; no walls placed yet means both boards are empty
    if state.winner.is_some() || state.h_walls != 0 || state.v_walls != 0 {
        return None;
    }
    let side = state.turn;
    let me = state.pawn(side);
    let mid = (state::SIZE + 1) / 2; // home column = 5
    if me.c != mid {
        return None;
    }
    // straight advance toward the goal row
    let next_r = if side.goal_row() > me.r {
        me.r + 1
    } else {
        me.r.checked_sub(1)?
    };
    // stay on book only while we haven't reached contact with the opponent's
    // half (after that, walls/jumps matter and we should actually search)
    let opp = state.pawn(side.other());
    if next_r == opp.r && me.c == opp.c {
        return None;
    }
    Some(Move::Pawn(state::Cell::new(next_r, me.c)))
}

/// Core ranking + pruning over a *shared* `Search`, returning kept moves with
/// their side-to-move score. Sharing one `Search` lets the transposition table
/// persist across calls — when generating the whole graph, transposed subtrees
/// are searched once, not once per arrival path (the compute-ahead saving).
/// Steps: opening book → score all → drop useless walls → drop blunders
/// (> `margin`) → tie-simplify (one representative per equal score) → take `k`.
fn rank_prune(
    s: &mut Search<'_, Heuristic>,
    state: &State,
    depth: u8,
    k: usize,
    margin: i32,
) -> Vec<(Move, i32)> {
    use moves::distance_to_goal;
    if state.winner.is_some() {
        return Vec::new();
    }
    // opening book: one perfect move, scored cheaply via the shared search.
    if let Some(mv) = opening_move(state) {
        let child = state.apply(mv);
        let score = -s.search(&child, depth.saturating_sub(1)).score;
        return vec![(mv, score)];
    }
    let opp = state.turn.other();
    let opp_dist_before =
        distance_to_goal(state, state.pawn(opp), opp.goal_row()).unwrap_or(u16::MAX);

    let ranked = s.ranked(state, depth);

    // keep pawn moves + walls that actually delay the opponent
    let useful: Vec<(Move, i32)> = ranked
        .into_iter()
        .filter(|(mv, _)| match mv {
            Move::Pawn(_) => true,
            Move::Wall(_) => {
                let child = state.apply(*mv);
                let after =
                    distance_to_goal(&child, child.pawn(opp), opp.goal_row()).unwrap_or(u16::MAX);
                after > opp_dist_before
            }
        })
        .collect();

    let best = useful.first().map(|x| x.1).unwrap_or(0);

    // Drop blunders (worse than `best` by more than `margin`), then dedup by the
    // *resulting state* — distinct moves that reach the same position collapse to
    // one, but two different positions that happen to share a score are both
    // kept. Deduping by score (the old behaviour) silently discarded distinct
    // lines whenever their i32 scores collided.
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    useful
        .into_iter()
        .filter(|(_, sc)| *sc >= best - margin)
        .filter(|(mv, _)| seen.insert(state_key(&state.apply(*mv))))
        .take(k)
        .collect()
}

/// Convert a side-to-move move score (after `state.turn` plays it) to that
/// child's SOUTH win-chance 0..100.
#[inline]
fn south_win(turn: Side, score: i32, win_k: f64) -> u8 {
    let south_eval = match turn {
        Side::South => score,
        Side::North => -score,
    };
    win_prob(south_eval, win_k)
}

/// Top moves for the state graph — pruned so the graph stays light and shows
/// strong play only. See [`rank_prune`] for the pipeline. `margin` is in the
/// eval's centi-step units (≈100 per board step).
pub fn top_moves(state: &State, depth: u8, k: usize, margin: i32, win_k: f64) -> Vec<RankedMove> {
    let h = Heuristic::default();
    let mut s = Search::new(&h);
    rank_prune(&mut s, state, depth, k, margin)
        .into_iter()
        .map(|(mv, score)| RankedMove {
            mv,
            south: south_win(state.turn, score, win_k),
            score,
        })
        .collect()
}

/// Canonical state key — must match the TS `stateKey` (webui/src/game/engine.ts)
/// exactly so a Rust-generated graph and the UI's lazy expansion share one key
/// space (travel + dedup work across both). Walls sorted so placement order
/// doesn't matter.
pub fn state_key(state: &State) -> String {
    let mut walls: Vec<(u8, u8, char)> = Vec::new();
    for r in 1..=(state::SIZE - 1) {
        for c in 1..=(state::SIZE - 1) {
            if state.has_h_wall(r, c) {
                walls.push((r, c, 'h'));
            }
            if state.has_v_wall(r, c) {
                walls.push((r, c, 'v'));
            }
        }
    }
    walls.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let w = walls
        .iter()
        .map(|(r, c, o)| format!("{o}{r}{c}"))
        .collect::<Vec<_>>()
        .join(",");
    let t = match state.turn {
        Side::South => "s",
        Side::North => "n",
    };
    let p = &state.pawns;
    format!(
        "{t}|{}{}|{}{}|{}{}|{w}",
        p[0].r, p[0].c, p[1].r, p[1].c, state.walls_left[0], state.walls_left[1]
    )
}

/// Parse a [`state_key`] string back into a [`State`].
///
/// The key format is shared with the web UI and generated graph files, so graph
/// consumers should use this instead of carrying their own partial parser.
pub fn parse_state_key(key: &str) -> Option<State> {
    let mut parts = key.split('|');
    let turn = match parts.next()? {
        "s" => Side::South,
        "n" => Side::North,
        _ => return None,
    };
    let south = parse_cell(parts.next()?)?;
    let north = parse_cell(parts.next()?)?;
    let walls_left = parse_walls_left(parts.next()?)?;
    let walls = parts.next().unwrap_or("");
    let mut state = State {
        pawns: [south, north],
        h_walls: 0,
        v_walls: 0,
        walls_left,
        turn,
        winner: None,
    };
    for wall in walls.split(',').filter(|w| !w.is_empty()) {
        add_wall_bits(&mut state, parse_wall(wall)?);
    }
    state.winner = if state.pawn(Side::South).r == Side::South.goal_row() {
        Some(Side::South)
    } else if state.pawn(Side::North).r == Side::North.goal_row() {
        Some(Side::North)
    } else {
        None
    };
    Some(state)
}

fn parse_cell(raw: &str) -> Option<Cell> {
    let bytes = raw.as_bytes();
    if bytes.len() != 2 {
        return None;
    }
    let r = digit(bytes[0])?;
    let c = digit(bytes[1])?;
    Some(Cell::new(r, c))
}

fn parse_walls_left(raw: &str) -> Option<[u8; 2]> {
    match raw.len() {
        2 => Some([digit(raw.as_bytes()[0])?, digit(raw.as_bytes()[1])?]),
        3 if raw.starts_with("10") => Some([10, digit(raw.as_bytes()[2])?]),
        3 if raw.ends_with("10") => Some([digit(raw.as_bytes()[0])?, 10]),
        4 if raw == "1010" => Some([10, 10]),
        _ => None,
    }
}

fn parse_wall(raw: &str) -> Option<Wall> {
    let bytes = raw.as_bytes();
    if bytes.len() != 3 {
        return None;
    }
    let o = match bytes[0] {
        b'h' => Orientation::H,
        b'v' => Orientation::V,
        _ => return None,
    };
    Some(Wall {
        r: digit(bytes[1])?,
        c: digit(bytes[2])?,
        o,
    })
}

fn digit(byte: u8) -> Option<u8> {
    if byte.is_ascii_digit() {
        Some(byte - b'0')
    } else {
        None
    }
}

fn add_wall_bits(state: &mut State, wall: Wall) {
    let bit = 1u64 << (((wall.r - 1) * 8 + (wall.c - 1)) as u64);
    match wall.o {
        Orientation::H => state.h_walls |= bit,
        Orientation::V => state.v_walls |= bit,
    }
}

/// One node in a precomputed state graph.
pub struct GraphNode {
    pub key: String,
    pub state: State,
    pub ply: u32,
    /// SOUTH win-chance 0..100 for this position, computed canonically from this
    /// node's own state (not inherited from the edge that first reached it).
    pub south: u8,
    /// True once all of this node's ranked children are in the graph. False for
    /// depth-limited frontier nodes and for nodes left unexpanded by the node cap
    /// — lets a consumer tell "no strong moves" from "not expanded yet".
    pub expanded: bool,
}

/// One directed edge: the move from `from` reaches `to`.
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub mv: Move,
}

/// A fully precomputed, pruned state graph.
pub struct Graph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    /// True if `max_nodes` was hit before reaching `max_depth` (graph truncated).
    pub capped: bool,
}

/// Generate the whole pruned state graph from `root`, breadth-first, deduping
/// transpositions by [`state_key`]. One [`Search`] is reused across the entire
/// BFS so its transposition table amortizes search across shared subtrees.
///
/// The two caps are the precompute / compute-ahead balance dials:
///   - `max_depth`  — how many plies ahead we precompute (more = more CPU now,
///     deeper instant navigation later),
///   - `max_nodes`  — hard ceiling on graph size (the memory budget); BFS stops
///     and sets `capped` when hit, so the system never blows past its budget.
/// `rank_depth` is the per-node search depth used to rank/prune moves.
pub fn generate_graph(
    root: &State,
    rank_depth: u8,
    k: usize,
    margin: i32,
    win_k: f64,
    max_depth: u32,
    max_nodes: usize,
) -> Graph {
    use std::collections::{HashMap, VecDeque};

    let h = Heuristic::default();
    let mut s = Search::new(&h); // shared TT across the whole BFS

    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut edges: Vec<GraphEdge> = Vec::new();
    let mut index: HashMap<String, ()> = HashMap::new();
    // Queue carries each node's index in `nodes` so we can flip `expanded` once
    // its children are fully materialized.
    let mut q: VecDeque<(State, u32, usize)> = VecDeque::new();

    let root_key = state_key(root);
    let (_b, root_south, _n) = analyze(root, rank_depth, win_k);
    index.insert(root_key.clone(), ());
    nodes.push(GraphNode {
        key: root_key.clone(),
        state: *root,
        ply: 0,
        south: root_south,
        expanded: false,
    });
    q.push_back((*root, 0, 0));

    let mut capped = false;
    while let Some((st, ply, idx)) = q.pop_front() {
        if ply >= max_depth {
            continue; // frontier node: stays `expanded = false`
        }
        let ranked = rank_prune(&mut s, &st, rank_depth, k, margin);

        // Atomic cap: expand a node fully or not at all, so no node is left with
        // a random subset of its children. Count the genuinely new children
        // first; if they don't all fit, stop without touching this node.
        let new_children = ranked
            .iter()
            .filter(|(mv, _)| !index.contains_key(&state_key(&st.apply(*mv))))
            .count();
        if nodes.len() + new_children > max_nodes {
            capped = true;
            break; // this node stays `expanded = false`
        }

        let from = nodes[idx].key.clone();
        for (mv, _score) in ranked {
            let child = st.apply(mv);
            let ck = state_key(&child);
            if !index.contains_key(&ck) {
                index.insert(ck.clone(), ());
                // Canonical valuation: score the child from its OWN state at the
                // same `rank_depth` as every other node (root included), so a
                // node's win% is path-independent, not the score of whichever
                // edge happened to reach it first.
                let child_score = s.search(&child, rank_depth).score;
                let cidx = nodes.len();
                nodes.push(GraphNode {
                    key: ck.clone(),
                    state: child,
                    ply: ply + 1,
                    south: south_win(child.turn, child_score, win_k),
                    expanded: false,
                });
                q.push_back((child, ply + 1, cidx));
            }
            edges.push(GraphEdge {
                from: from.clone(),
                to: ck,
                mv,
            });
        }
        nodes[idx].expanded = true;
    }

    Graph {
        nodes,
        edges,
        capped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_is_symmetric() {
        let s = State::initial();
        assert_eq!(s.pawn(Side::South), Cell::new(1, 5));
        assert_eq!(s.pawn(Side::North), Cell::new(9, 5));
        assert_eq!(s.walls_left, [10, 10]);
        assert_eq!(s.turn, Side::South);
        // both start 8 steps from goal
        assert_eq!(
            moves::distance_to_goal(&s, s.pawn(Side::South), Side::South.goal_row()),
            Some(8)
        );
        assert_eq!(
            moves::distance_to_goal(&s, s.pawn(Side::North), Side::North.goal_row()),
            Some(8)
        );
    }

    #[test]
    fn opening_pawn_has_three_moves() {
        let s = State::initial();
        // (1,5): can go up, left, right — not off the bottom edge.
        let mv = pawn_moves(&s, Side::South);
        assert_eq!(mv.len(), 3);
        assert!(mv.contains(&Cell::new(2, 5)));
        assert!(mv.contains(&Cell::new(1, 4)));
        assert!(mv.contains(&Cell::new(1, 6)));
    }

    #[test]
    fn horizontal_wall_blocks_vertical_step() {
        let mut s = State::initial();
        s.pawns[Side::South.idx()] = Cell::new(4, 5);
        // wall anchor (4,5) spans cols 5,6 between rows 4 and 5
        s = s.apply_wall_unchecked(Wall { r: 4, c: 5, o: Orientation::H }, Side::South);
        assert!(s.is_blocked(Cell::new(4, 5), Cell::new(5, 5)));
        assert!(!s.is_blocked(Cell::new(4, 5), Cell::new(4, 6)));
    }

    #[test]
    fn wall_cannot_fully_trap_a_player() {
        // Box SOUTH's pawn into a corner-ish trap attempt; the last sealing
        // wall must be rejected by the no-trap rule.
        let mut s = State::initial();
        s.pawns[Side::South.idx()] = Cell::new(1, 1);
        // surround (1,1): wall above it (h at (1,1)) leaves the side exit open,
        // wall to its right (v at (1,1)) seals it -> illegal.
        s = s.apply_wall_unchecked(Wall { r: 1, c: 1, o: Orientation::H }, Side::North);
        let seal = Wall { r: 1, c: 1, o: Orientation::V };
        assert!(!moves::can_place_wall(&s, seal, Side::North));
    }

    #[test]
    fn state_key_matches_ts_format_for_initial() {
        // TS: "s|15|95|1010|" for the opening (turn s, pawns (1,5)/(9,5),
        // walls_left 10/10, no walls).
        let s = State::initial();
        assert_eq!(state_key(&s), "s|15|95|1010|");
    }

    #[test]
    fn parse_state_key_reads_initial_state() {
        let state = parse_state_key("s|15|95|1010|").expect("parse initial");
        assert_eq!(state.turn, Side::South);
        assert_eq!(state.pawn(Side::South), Cell::new(1, 5));
        assert_eq!(state.pawn(Side::North), Cell::new(9, 5));
        assert_eq!(state.walls_left, [10, 10]);
        assert_eq!(state.winner, None);
    }

    #[test]
    fn parse_state_key_reads_walls_and_counts() {
        let state = parse_state_key("n|15|95|910|h11,v22").expect("parse walls");
        assert_eq!(state.turn, Side::North);
        assert_eq!(state.walls_left, [9, 10]);
        assert!(state.has_h_wall(1, 1));
        assert!(state.has_v_wall(2, 2));
    }

    #[test]
    fn generate_graph_caps_and_dedups() {
        let root = State::initial();
        // small depth: opening is the straight-line book, so the graph is a
        // short chain (1 node per ply) — exercises dedup + edges.
        let g = generate_graph(&root, 2, 5, 400, 200.0, 4, 1000);
        assert!(!g.capped);
        assert_eq!(g.nodes.len(), 5, "depth-4 book chain = 5 nodes");
        assert_eq!(g.edges.len(), 4);
        // node cap is honoured
        let capped = generate_graph(&root, 2, 5, 400, 200.0, 20, 50);
        assert!(capped.capped);
        assert!(capped.nodes.len() <= 50);
    }

    #[test]
    fn opening_is_a_single_memorized_advance() {
        let s = State::initial();
        // book hit: exactly one move, the straight advance
        let book = opening_move(&s);
        assert_eq!(book, Some(Move::Pawn(Cell::new(2, 5))));
        let top = top_moves(&s, 2, 3, 80, 200.0);
        assert_eq!(top.len(), 1, "opening should be one perfect move");
        assert_eq!(top[0].mv, Move::Pawn(Cell::new(2, 5)));
    }

    #[test]
    fn off_book_once_a_wall_is_placed() {
        let mut s = State::initial();
        s = s.apply_wall_unchecked(
            Wall { r: 4, c: 4, o: Orientation::H },
            Side::North,
        );
        assert_eq!(opening_move(&s), None, "walls on board -> must search");
    }

    #[test]
    fn top_moves_prunes_to_k_and_drops_useless_walls() {
        let s = State::initial();
        // full legal move count is large (pawn + ~128 wall placements)
        let all = legal_moves(&s).len();
        assert!(all > 100, "expected many raw moves, got {all}");
        // pruned graph view: at most k, and from the opening every wall is
        // "useless" (does not yet lengthen a straight-line opponent path that
        // far away) so the strong moves are pawn advances.
        let top = top_moves(&s, 2, 3, 80, 200.0);
        assert!(top.len() <= 3, "expected <=3, got {}", top.len());
        assert!(!top.is_empty(), "expected at least one move");
        // best move from the opening is advancing the pawn forward
        match top[0].mv {
            Move::Pawn(c) => assert_eq!(c.r, 2),
            other => panic!("expected pawn advance, got {other:?}"),
        }
    }

    #[test]
    fn analyze_scores_sum_to_100() {
        let s = State::initial();
        let (_m, south, north) = analyze(&s, 2, 200.0);
        assert_eq!(south as u16 + north as u16, 100);
    }

    #[test]
    fn search_prefers_advancing_when_unobstructed() {
        let s = State::initial();
        let (best, _s, _n) = analyze(&s, 2, 200.0);
        // with no walls in play, the engine should step the pawn forward
        match best {
            Some(Move::Pawn(c)) => assert_eq!(c.r, 2),
            other => panic!("expected forward pawn move, got {other:?}"),
        }
    }
}
