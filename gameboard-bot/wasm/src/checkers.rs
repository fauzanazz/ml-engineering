//! WASM bridge for International Draughts (10×10 checkers).
//!
//! Unlike Wall Chess — whose browser UI carries a full parallel TS rules engine
//! and only asks WASM for the bot's move — draughts drives *every* rule from
//! Rust: the mandatory-maximal-capture move generator, flying-king slides, and
//! promotion-on-stop are subtle and already perft-validated in `gameboard-core`,
//! so duplicating them in TypeScript would be a fidelity hazard. The browser
//! therefore treats this module as the engine: initial position, legal moves,
//! move application, terminal/winner status, and the Gen-1 bot's analysis all
//! come from here. The TS layer is a thin renderer.
//!
//! State is exchanged as occupied dark-square indices (0..49) rather than raw
//! 64-bit bitboards, so the JSON stays JS-number-safe and trivial to render.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use gameboard_core::checkers::{
    legal_moves, player_to_color, CheckersHeuristic, Color, Move, State,
};
use gameboard_core::eval::win_prob;
use gameboard_core::game::{Game, Player};
use gameboard_core::search::{Search, SearchConfig};
use gameboard_core::Checkers;

/// Number of playable dark squares (mirrors `gameboard_core::checkers::N`).
const N: u8 = 50;

// ---- DTOs (the TS contract; see webui/src/game/checkers.ts) ----------------

#[derive(Deserialize)]
struct StateJs {
    /// Dark-square indices (0..49) occupied by White.
    white: Vec<u8>,
    /// Dark-square indices occupied by Black.
    black: Vec<u8>,
    /// Indices that are crowned (subset of `white ∪ black`).
    kings: Vec<u8>,
    /// "white" | "black" — side to move.
    stm: String,
    /// Plies since the last capture or man move (the draw counter).
    idle: u16,
}

#[derive(Serialize)]
struct StateOut {
    white: Vec<u8>,
    black: Vec<u8>,
    kings: Vec<u8>,
    stm: &'static str,
    idle: u16,
}

#[derive(Deserialize)]
struct MoveJs {
    from: u8,
    to: u8,
    /// Indices of every jumped square (empty for a quiet move).
    #[serde(default)]
    captured: Vec<u8>,
}

#[derive(Serialize)]
struct MoveOut {
    from: u8,
    to: u8,
    captured: Vec<u8>,
}

#[derive(Serialize)]
struct StatusOut {
    /// No legal move for the side to move, or the idle-draw cap was reached.
    terminal: bool,
    /// "white" | "black" winner, or `null` (ongoing or drawn).
    winner: Option<&'static str>,
    /// Terminal with no winner (the 25-move idle-king rule).
    draw: bool,
}

#[derive(Serialize)]
struct AnalysisOut {
    /// `null` only if the position is terminal / has no legal move.
    #[serde(rename = "move")]
    mv: Option<MoveOut>,
    /// White win chance 0..100; `black = 100 - white`.
    white: u8,
    black: u8,
    /// Deepest iterative-deepening depth actually completed.
    depth: u8,
    /// True if the node budget tripped before `depth` finished.
    stopped: bool,
    nodes: u64,
}

// ---- bitboard <-> index conversions ----------------------------------------

/// Set-bit indices of a 50-bit board, ascending. Mirrors the engine's `bits`.
fn board_to_indices(mut b: u64) -> Vec<u8> {
    let mut out = Vec::new();
    while b != 0 {
        out.push(b.trailing_zeros() as u8);
        b &= b - 1;
    }
    out
}

/// Fold dark-square indices into a 50-bit board, rejecting out-of-range squares.
fn indices_to_board(ixs: &[u8]) -> Result<u64, String> {
    let mut b = 0u64;
    for &i in ixs {
        if i >= N {
            return Err(format!("square index out of range: {i}"));
        }
        b |= 1u64 << i;
    }
    Ok(b)
}

fn parse_color(s: &str) -> Result<Color, String> {
    match s {
        "white" => Ok(Color::White),
        "black" => Ok(Color::Black),
        other => Err(format!("bad side: {other}")),
    }
}

fn color_str(c: Color) -> &'static str {
    match c {
        Color::White => "white",
        Color::Black => "black",
    }
}

fn player_str(p: Player) -> &'static str {
    color_str(player_to_color(p))
}

fn to_core(js: StateJs) -> Result<State, String> {
    let white = indices_to_board(&js.white)?;
    let black = indices_to_board(&js.black)?;
    let kings = indices_to_board(&js.kings)?;
    if white & black != 0 {
        return Err("a square is occupied by both colours".into());
    }
    if kings & !(white | black) != 0 {
        return Err("a king square has no piece".into());
    }
    Ok(State {
        white,
        black,
        kings,
        stm: parse_color(&js.stm)?,
        idle: js.idle,
    })
}

fn state_to_out(s: &State) -> StateOut {
    StateOut {
        white: board_to_indices(s.white),
        black: board_to_indices(s.black),
        kings: board_to_indices(s.kings),
        stm: color_str(s.stm),
        idle: s.idle,
    }
}

fn move_to_core(js: MoveJs) -> Result<Move, String> {
    Ok(Move {
        from: js.from,
        to: js.to,
        captured: indices_to_board(&js.captured)?,
    })
}

fn move_to_out(mv: Move) -> MoveOut {
    MoveOut {
        from: mv.from,
        to: mv.to,
        captured: board_to_indices(mv.captured),
    }
}

fn encode<T: Serialize>(v: &T) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(v).map_err(|e| JsValue::from_str(&format!("encode: {e}")))
}

fn decode_state(state: JsValue) -> Result<State, JsValue> {
    let js: StateJs = serde_wasm_bindgen::from_value(state)
        .map_err(|e| JsValue::from_str(&format!("bad state: {e}")))?;
    to_core(js).map_err(|e| JsValue::from_str(&e))
}

// ---- public API ------------------------------------------------------------

/// The starting position: 20 men per side, White to move.
#[wasm_bindgen]
pub fn ck_initial() -> Result<JsValue, JsValue> {
    encode(&state_to_out(&State::initial()))
}

/// Every legal move for the side to move. Captures are mandatory and maximal, so
/// when any capture exists this returns *only* captures — the UI never needs to
/// know the rule, it just offers what comes back.
#[wasm_bindgen]
pub fn ck_legal_moves(state: JsValue) -> Result<JsValue, JsValue> {
    let core = decode_state(state)?;
    let moves: Vec<MoveOut> = legal_moves(&core).into_iter().map(move_to_out).collect();
    encode(&moves)
}

/// Apply a move. Validated against the legal set (so an illegal/forged move is
/// rejected rather than silently mutating the board), then the engine relocates
/// the piece, removes captures, crowns on promotion, and flips the side.
#[wasm_bindgen]
pub fn ck_apply(state: JsValue, mv: JsValue) -> Result<JsValue, JsValue> {
    let core = decode_state(state)?;
    let m: MoveJs = serde_wasm_bindgen::from_value(mv)
        .map_err(|e| JsValue::from_str(&format!("bad move: {e}")))?;
    let m = move_to_core(m).map_err(|e| JsValue::from_str(&e))?;
    if !legal_moves(&core).contains(&m) {
        return Err(JsValue::from_str("illegal move"));
    }
    encode(&state_to_out(&core.apply(m)))
}

/// Terminal / winner status for the position (game-over detection for the UI).
#[wasm_bindgen]
pub fn ck_status(state: JsValue) -> Result<JsValue, JsValue> {
    let core = decode_state(state)?;
    let terminal = Checkers::is_terminal(&core);
    let winner = Checkers::winner(&core).map(player_str);
    encode(&StatusOut {
        terminal,
        winner,
        draw: terminal && winner.is_none(),
    })
}

/// Run the Gen-1 bot (default eval + the draughts search preset: PVS +
/// aspiration + LMR + quiescence) and return its best move plus the White/Black
/// win split. `node_limit == 0` means a fixed-depth search; `> 0` budgets nodes
/// for stable tail latency. `k` is the logistic squash for the win meter.
#[wasm_bindgen]
pub fn ck_analyze(
    state: JsValue,
    depth: u8,
    node_limit: u64,
    k: f64,
) -> Result<JsValue, JsValue> {
    let core = decode_state(state)?;
    let h = CheckersHeuristic::default();
    let mut s = Search::with_config(&h, SearchConfig::draughts());
    let res = if node_limit > 0 {
        s.search_with_node_limit(&core, depth, node_limit)
    } else {
        s.search(&core, depth)
    };
    // res.score is from the side-to-move POV; convert to White's POV.
    let white_eval = match core.stm {
        Color::White => res.score,
        Color::Black => -res.score,
    };
    let white = win_prob(white_eval, k);
    encode(&AnalysisOut {
        mv: res.best.map(move_to_out),
        white,
        black: 100 - white,
        depth: res.depth,
        stopped: res.stopped,
        nodes: res.nodes,
    })
}
