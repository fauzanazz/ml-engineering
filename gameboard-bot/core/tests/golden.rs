//! Golden determinism gate for the deployed Wall Chess search.
//!
//! The multi-game refactor (introducing the `Game` trait + genericizing the
//! search) must NOT change a single move the engine plays. Search is fully
//! deterministic (no rng / threads / clock), so we freeze the best move + score
//! over a fixed position set for BOTH the default engine and the deployed Gen-2
//! engine, and assert byte-identical output after every refactor phase.
//!
//! Regenerate the frozen baseline (only when an intentional behavior change is
//! made — never to paper over a refactor regression):
//!     REGEN_GOLDEN=1 cargo test --test golden
//!
//! The fixture (`tests/golden_snapshot.txt`) is committed; an empty/absent file
//! makes the check fail loudly rather than silently passing.

use std::fs;
use std::path::PathBuf;

use gameboard_core::eval::Heuristic;
use gameboard_core::search::{Search, SearchConfig};
use gameboard_core::state::{Move, Orientation, State};
use gameboard_core::{legal_moves, parse_state_key, state_key};

const DEPTH: u8 = 6;
const N_POSITIONS: usize = 24;

fn fmt_move(mv: Move) -> String {
    match mv {
        Move::Pawn(c) => format!("P{}{}", c.r, c.c),
        Move::Wall(w) => {
            let o = match w.o {
                Orientation::H => "H",
                Orientation::V => "V",
            };
            format!("{}{}{}", o, w.r, w.c)
        }
    }
}

fn fmt_opt(mv: Option<Move>) -> String {
    mv.map(fmt_move).unwrap_or_else(|| "NONE".to_string())
}

/// Deterministic mid-game position corpus: xorshift64 random legal playouts of
/// varying length, stopping before anyone wins. Identical every run.
fn corpus() -> Vec<State> {
    let mut rng: u64 = 0xC0FFEE_1234_5678;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut out = vec![State::initial()];
    while out.len() < N_POSITIONS {
        let mut state = State::initial();
        let plies = 4 + (next() as usize % 22);
        for _ in 0..plies {
            if state.winner.is_some() {
                break;
            }
            let moves = legal_moves(&state);
            if moves.is_empty() {
                break;
            }
            state = state.apply(moves[(next() as usize) % moves.len()]);
        }
        if state.winner.is_none() {
            out.push(state);
        }
    }
    out
}

/// One snapshot line: state_key | default(move,score) | gen2(move,score).
fn snapshot_line(state: &State) -> String {
    let def_eval = Heuristic::default();
    let def_cfg = SearchConfig::default();
    let def = Search::with_config(&def_eval, def_cfg).search(state, DEPTH);

    // Deployed Gen-2 engine: reweighted eval (w_path=50, w_wall=120, exact
    // endgame) + aggressive pruning bundle. Mirrors the wasm `*_gen2` path.
    let g2_eval = Heuristic::new(50, 120, true);
    let g2_cfg = SearchConfig::aggressive();
    let g2 = Search::with_config(&g2_eval, g2_cfg).search(state, DEPTH);

    format!(
        "{}\t{}\t{}\t{}\t{}",
        state_key(state),
        fmt_opt(def.best),
        def.score,
        fmt_opt(g2.best),
        g2.score
    )
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden_snapshot.txt")
}

#[test]
fn deployed_search_is_byte_identical() {
    let lines: Vec<String> = corpus().iter().map(snapshot_line).collect();
    let current = lines.join("\n");

    if std::env::var("REGEN_GOLDEN").is_ok() {
        fs::write(fixture_path(), format!("{current}\n")).expect("write golden fixture");
        eprintln!("REGEN_GOLDEN: wrote {} lines", lines.len());
        return;
    }

    let expected = fs::read_to_string(fixture_path())
        .expect("golden_snapshot.txt missing — run `REGEN_GOLDEN=1 cargo test --test golden`");
    let expected = expected.trim_end();
    assert!(!expected.is_empty(), "golden fixture is empty");

    // Compare line-by-line so a mismatch points at the exact position.
    let exp_lines: Vec<&str> = expected.lines().collect();
    assert_eq!(
        exp_lines.len(),
        lines.len(),
        "position count drift: fixture {} vs current {}",
        exp_lines.len(),
        lines.len()
    );
    for (i, (got, want)) in lines.iter().zip(exp_lines.iter()).enumerate() {
        // Round-trip the key so a parse regression also trips here.
        let key = got.split('\t').next().unwrap();
        assert!(parse_state_key(key).is_some(), "position {i} key unparseable: {key}");
        assert_eq!(got, want, "GOLDEN MISMATCH at position {i}\n  got:  {got}\n  want: {want}");
    }
}
