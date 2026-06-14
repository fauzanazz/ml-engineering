//! WASM bridge between the browser UI and the Rust `wallchess-core` engine.
//!
//! The UI speaks the TS `GameState` / `Move` JSON shapes (see
//! webui/src/game/engine.ts). We deserialize those, run the real search, and
//! return the chosen move plus the 0..100 win split.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wallchess_core::{
    analyze, analyze_with_node_limit, generate_graph, top_moves,
    state::{Cell, Orientation, Side, State},
};

// ---- inbound DTOs (match the TS GameState JSON exactly) --------------------

#[derive(Deserialize)]
struct CellJs {
    r: u8,
    c: u8,
}

#[derive(Deserialize)]
struct WallJs {
    r: u8,
    c: u8,
    o: String, // "h" | "v"
}

#[derive(Deserialize)]
struct PawnsJs {
    south: CellJs,
    north: CellJs,
}

#[derive(Deserialize)]
struct WallsLeftJs {
    south: u8,
    north: u8,
}

#[derive(Deserialize)]
struct StateJs {
    pawns: PawnsJs,
    walls: Vec<WallJs>,
    #[serde(rename = "wallsLeft")]
    walls_left: WallsLeftJs,
    turn: String,           // "south" | "north"
    winner: Option<String>, // "south" | "north" | null
}

// ---- outbound DTOs (match the TS Move JSON) -------------------------------

#[derive(Serialize)]
struct CellOut {
    r: u8,
    c: u8,
}

#[derive(Serialize)]
struct WallOut {
    r: u8,
    c: u8,
    o: &'static str,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum MoveOut {
    Move { to: CellOut },
    Wall { wall: WallOut },
}

#[derive(Serialize)]
struct RankedOut {
    #[serde(rename = "move")]
    mv: MoveOut,
    south: u8,
    score: i32,
}

#[derive(Serialize)]
struct Analysis {
    /// `null` only if the position is terminal / has no legal move.
    #[serde(rename = "move")]
    mv: Option<MoveOut>,
    /// SOUTH win chance 0..100; `north = 100 - south`.
    south: u8,
    north: u8,
}

#[derive(Serialize)]
struct BudgetAnalysis {
    #[serde(rename = "move")]
    mv: Option<MoveOut>,
    south: u8,
    north: u8,
    depth: u8,
    stopped: bool,
    nodes: u64,
}

// ---- graph DTOs (full GameState per node so the UI can render each board) --

#[derive(Serialize)]
struct PawnsOut {
    south: CellOut,
    north: CellOut,
}

#[derive(Serialize)]
struct WallsLeftOut {
    south: u8,
    north: u8,
}

#[derive(Serialize)]
struct StateOut {
    pawns: PawnsOut,
    walls: Vec<WallOut>,
    #[serde(rename = "wallsLeft")]
    walls_left: WallsLeftOut,
    turn: &'static str,
    winner: Option<&'static str>,
}

#[derive(Serialize)]
struct GraphNodeOut {
    key: String,
    ply: u32,
    south: u8,
    expanded: bool,
    state: StateOut,
}

#[derive(Serialize)]
struct GraphEdgeOut {
    from: String,
    to: String,
    #[serde(rename = "move")]
    mv: MoveOut,
}

#[derive(Serialize)]
struct GraphOut {
    nodes: Vec<GraphNodeOut>,
    edges: Vec<GraphEdgeOut>,
    capped: bool,
}

fn side_str(s: Side) -> &'static str {
    match s {
        Side::South => "south",
        Side::North => "north",
    }
}

fn state_to_out(st: &State) -> StateOut {
    let mut walls: Vec<WallOut> = Vec::new();
    for r in 1..=8u8 {
        for c in 1..=8u8 {
            if st.has_h_wall(r, c) {
                walls.push(WallOut { r, c, o: "h" });
            }
            if st.has_v_wall(r, c) {
                walls.push(WallOut { r, c, o: "v" });
            }
        }
    }
    StateOut {
        pawns: PawnsOut {
            south: CellOut {
                r: st.pawn(Side::South).r,
                c: st.pawn(Side::South).c,
            },
            north: CellOut {
                r: st.pawn(Side::North).r,
                c: st.pawn(Side::North).c,
            },
        },
        walls,
        walls_left: WallsLeftOut {
            south: st.walls_left[0],
            north: st.walls_left[1],
        },
        turn: side_str(st.turn),
        winner: st.winner.map(side_str),
    }
}

// ---- conversion -----------------------------------------------------------

fn parse_side(s: &str) -> Result<Side, String> {
    match s {
        "south" => Ok(Side::South),
        "north" => Ok(Side::North),
        other => Err(format!("bad side: {other}")),
    }
}

fn to_core(js: StateJs) -> Result<State, String> {
    let mut h_walls: u64 = 0;
    let mut v_walls: u64 = 0;
    for w in &js.walls {
        if !(1..=8).contains(&w.r) || !(1..=8).contains(&w.c) {
            return Err(format!("wall anchor out of range: ({},{})", w.r, w.c));
        }
        let bit = 1u64 << (((w.r - 1) * 8 + (w.c - 1)) as u64);
        match w.o.as_str() {
            "h" => h_walls |= bit,
            "v" => v_walls |= bit,
            other => return Err(format!("bad wall orientation: {other}")),
        }
    }
    let winner = match js.winner.as_deref() {
        None => None,
        Some(s) => Some(parse_side(s)?),
    };
    Ok(State {
        pawns: [
            Cell::new(js.pawns.south.r, js.pawns.south.c),
            Cell::new(js.pawns.north.r, js.pawns.north.c),
        ],
        h_walls,
        v_walls,
        walls_left: [js.walls_left.south, js.walls_left.north],
        turn: parse_side(&js.turn)?,
        winner,
    })
}

fn move_to_out(mv: wallchess_core::Move) -> MoveOut {
    match mv {
        wallchess_core::Move::Pawn(c) => MoveOut::Move {
            to: CellOut { r: c.r, c: c.c },
        },
        wallchess_core::Move::Wall(w) => MoveOut::Wall {
            wall: WallOut {
                r: w.r,
                c: w.c,
                o: match w.o {
                    Orientation::H => "h",
                    Orientation::V => "v",
                },
            },
        },
    }
}

// ---- public API -----------------------------------------------------------

/// Analyze a position: return the engine's best move and the 0..100 win split.
/// `state` is the TS `GameState` object; `depth` is the search depth (plies).
/// `k` controls the logistic squash (larger = scores look closer to 50/50).
#[wasm_bindgen]
pub fn analyze_state(state: JsValue, depth: u8, k: f64) -> Result<JsValue, JsValue> {
    let js: StateJs = serde_wasm_bindgen::from_value(state)
        .map_err(|e| JsValue::from_str(&format!("bad state: {e}")))?;
    let core = to_core(js).map_err(|e| JsValue::from_str(&e))?;
    let (best, south, north) = analyze(&core, depth, k);
    let out = Analysis {
        mv: best.map(move_to_out),
        south,
        north,
    };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&format!("encode: {e}")))
}

/// Analyze with a node budget. Returns the deepest completed depth plus budget
/// metadata so the browser can use a high max depth with stable tail latency.
#[wasm_bindgen]
pub fn analyze_state_budgeted(
    state: JsValue,
    depth: u8,
    node_limit: u64,
    k: f64,
) -> Result<JsValue, JsValue> {
    let js: StateJs = serde_wasm_bindgen::from_value(state)
        .map_err(|e| JsValue::from_str(&format!("bad state: {e}")))?;
    let core = to_core(js).map_err(|e| JsValue::from_str(&e))?;
    let (best, south, north, reached, stopped, nodes) =
        analyze_with_node_limit(&core, depth, node_limit, k);
    let out = BudgetAnalysis {
        mv: best.map(move_to_out),
        south,
        north,
        depth: reached,
        stopped,
        nodes,
    };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&format!("encode: {e}")))
}

/// Pruned successors for the state graph: top-`k` strong moves (depth `depth`),
/// useless walls and blunders (worse than best by > `margin`) removed. Each
/// entry carries the move + the resulting SOUTH win-chance, already scored —
/// the graph needs no separate analysis pass.
#[wasm_bindgen]
pub fn top_moves_js(
    state: JsValue,
    depth: u8,
    k: usize,
    margin: i32,
    win_k: f64,
) -> Result<JsValue, JsValue> {
    let js: StateJs = serde_wasm_bindgen::from_value(state)
        .map_err(|e| JsValue::from_str(&format!("bad state: {e}")))?;
    let core = to_core(js).map_err(|e| JsValue::from_str(&e))?;
    let ranked = top_moves(&core, depth, k, margin, win_k);
    let out: Vec<RankedOut> = ranked
        .into_iter()
        .map(|r| RankedOut {
            mv: move_to_out(r.mv),
            south: r.south,
            score: r.score,
        })
        .collect();
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&format!("encode: {e}")))
}

/// Generate the whole pruned state graph from `state` in one call. BFS happens
/// entirely in Rust (one shared transposition table), so the UI makes a single
/// call instead of thousands of round-trips. `max_depth` controls how far ahead
/// we precompute; `max_nodes` is the hard memory ceiling (sets `capped` if hit).
#[wasm_bindgen]
pub fn generate_graph_js(
    state: JsValue,
    rank_depth: u8,
    k: usize,
    margin: i32,
    win_k: f64,
    max_depth: u32,
    max_nodes: usize,
) -> Result<JsValue, JsValue> {
    let js: StateJs = serde_wasm_bindgen::from_value(state)
        .map_err(|e| JsValue::from_str(&format!("bad state: {e}")))?;
    let core = to_core(js).map_err(|e| JsValue::from_str(&e))?;
    let g = generate_graph(&core, rank_depth, k, margin, win_k, max_depth, max_nodes);
    let out = GraphOut {
        nodes: g
            .nodes
            .iter()
            .map(|n| GraphNodeOut {
                key: n.key.clone(),
                ply: n.ply,
                south: n.south,
                expanded: n.expanded,
                state: state_to_out(&n.state),
            })
            .collect(),
        edges: g
            .edges
            .into_iter()
            .map(|e| GraphEdgeOut {
                from: e.from,
                to: e.to,
                mv: move_to_out(e.mv),
            })
            .collect(),
        capped: g.capped,
    };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&format!("encode: {e}")))
}

/// Convenience: just the move (TS `Move` JSON), matching the old `chooseMove`.
#[wasm_bindgen]
pub fn choose_move(state: JsValue, depth: u8) -> Result<JsValue, JsValue> {
    let js: StateJs = serde_wasm_bindgen::from_value(state)
        .map_err(|e| JsValue::from_str(&format!("bad state: {e}")))?;
    let core = to_core(js).map_err(|e| JsValue::from_str(&e))?;
    let (best, _s, _n) = analyze(&core, depth, 200.0);
    match best {
        Some(mv) => serde_wasm_bindgen::to_value(&move_to_out(mv))
            .map_err(|e| JsValue::from_str(&format!("encode: {e}"))),
        None => Err(JsValue::from_str("no legal move")),
    }
}

// ---- trained-net bot (feature `net`) --------------------------------------

/// MCTS bot backed by the trained value/policy net. Construct once from the
/// fetched safetensors bytes, then call `best_move` per position. Built only
/// with `--features net` (pulls candle into the wasm bundle).
#[cfg(feature = "net")]
#[wasm_bindgen]
pub struct NetBot {
    eval: wallchess_core::net::NetEvaluator,
}

#[cfg(feature = "net")]
#[wasm_bindgen]
impl NetBot {
    /// Load the net from raw safetensors bytes (the browser fetches the file and
    /// hands over the `Uint8Array`).
    #[wasm_bindgen(constructor)]
    pub fn new(weights: &[u8]) -> Result<NetBot, JsValue> {
        let eval = wallchess_core::net::NetEvaluator::from_buffer(weights)
            .map_err(|e| JsValue::from_str(&format!("load net: {e}")))?;
        Ok(NetBot { eval })
    }

    /// Run `sims` MCTS simulations from `state` and return the best move plus the
    /// 0..100 SOUTH/NORTH win split (from the net's value head).
    #[wasm_bindgen]
    pub fn best_move(&self, state: JsValue, sims: u32) -> Result<JsValue, JsValue> {
        use wallchess_core::{Mcts, MctsConfig, PolicyValue};
        let js: StateJs = serde_wasm_bindgen::from_value(state)
            .map_err(|e| JsValue::from_str(&format!("bad state: {e}")))?;
        let core = to_core(js).map_err(|e| JsValue::from_str(&e))?;

        let cfg = MctsConfig {
            sims,
            root_noise: 0.0,
            ..MctsConfig::default()
        };
        let mut mcts = Mcts::new(&self.eval, cfg);
        let visits = mcts.run(&core);
        let best = visits
            .iter()
            .max_by_key(|(_, n)| *n)
            .map(|(m, _)| move_to_out(*m));

        // Win split from the value head (side-to-move POV -> SOUTH POV -> 0..100).
        let (value, _) = self.eval.evaluate(&core);
        let south_value = match core.turn {
            Side::South => value,
            Side::North => -value,
        };
        let south = (((south_value + 1.0) * 50.0).round()).clamp(0.0, 100.0) as u8;

        let out = Analysis {
            mv: best,
            south,
            north: 100 - south,
        };
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&format!("encode: {e}")))
    }
}
