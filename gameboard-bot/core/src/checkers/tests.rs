//! International draughts rule-fidelity + perft tests.
//!
//! Perft is the external oracle: initial-position move-tree counts verified by
//! Ed Gilbert / Feike Boomstra and reproduced by BikDam. They only reproduce
//! with the `(from,to,captured)` max-count dedup in [`super::captures`].
//!   d1=9 d2=81 d3=658 d4=4265 d5=27117 d6=167140 d7=1049442 d8=6483961

use super::*;

/// Build a state from square-index lists (idle = 0).
fn st(white: &[u8], black: &[u8], kings: &[u8], stm: Color) -> State {
    let bb = |v: &[u8]| v.iter().fold(0u64, |a, &i| a | (1u64 << i));
    State {
        white: bb(white),
        black: bb(black),
        kings: bb(kings),
        stm,
        idle: 0,
    }
}

// ── Perft (the fidelity oracle) ─────────────────────────────────────────────

#[test]
fn perft_initial_shallow() {
    let s = State::initial();
    assert_eq!(perft(&s, 1), 9, "d1");
    assert_eq!(perft(&s, 2), 81, "d2");
    assert_eq!(perft(&s, 3), 658, "d3");
    assert_eq!(perft(&s, 4), 4265, "d4");
    assert_eq!(perft(&s, 5), 27117, "d5");
    assert_eq!(perft(&s, 6), 167140, "d6");
}

#[test]
#[ignore = "slow: perft d7/d8 — run with `cargo test -- --ignored`"]
fn perft_initial_deep() {
    let s = State::initial();
    assert_eq!(perft(&s, 7), 1049442, "d7");
    assert_eq!(perft(&s, 8), 6483961, "d8 — the 4+-capture dedup tripwire");
}

// ── Mandatory capture + max-count + no tiebreak ─────────────────────────────

#[test]
fn capture_is_mandatory() {
    // White man at 22 with a black man at 18 and an empty 13 beyond → must jump.
    let s = st(&[22], &[18], &[], Color::White);
    let moves = legal_moves(&s);
    assert!(!moves.is_empty());
    assert!(
        moves.iter().all(|m| m.captured != 0),
        "a capture exists, so only captures are legal: {moves:?}"
    );
}

#[test]
fn max_count_filter_drops_shorter() {
    // From 22 a 2-chain exists up-right (over 18→land 13, over 9→land 4) and a
    // 1-jump exists down-left (over 27→land 31). Only the 2-chain is legal.
    let s = st(&[22], &[18, 9, 27], &[], Color::White);
    let moves = legal_moves(&s);
    assert_eq!(moves.len(), 1, "{moves:?}");
    let m = moves[0];
    assert_eq!(m.from, 22);
    assert_eq!(m.to, 4);
    assert_eq!(m.captured.count_ones(), 2);
    assert_eq!(m.captured, (1u64 << 18) | (1u64 << 9));
}

#[test]
fn no_value_tiebreak_between_equal_counts() {
    // Two distinct 1-captures: a king (down-left, over 27→31) and a man
    // (up-right, over 18→13). International keeps both — count only, no value.
    let s = st(&[22], &[18, 27], &[27], Color::White);
    let moves = legal_moves(&s);
    assert_eq!(moves.len(), 2, "{moves:?}");
    assert!(moves.iter().all(|m| m.captured.count_ones() == 1));
    let targets: Vec<u8> = moves.iter().map(|m| m.to).collect();
    assert!(targets.contains(&13) && targets.contains(&31), "{targets:?}");
}

#[test]
fn man_captures_backward() {
    // White man at 22, black man at 28 (down-right, behind), empty 33 beyond.
    let s = st(&[22], &[28], &[], Color::White);
    let moves = legal_moves(&s);
    assert_eq!(moves.len(), 1);
    assert_eq!(moves[0].to, 33);
    assert_eq!(moves[0].captured, 1u64 << 28);
}

// ── Flying king ─────────────────────────────────────────────────────────────

#[test]
fn flying_king_distance_and_multiple_landings() {
    // King at 22, empty 18 between, black man at 13, empties 9 and 4 beyond.
    let s = st(&[22], &[13], &[22], Color::White);
    let moves = legal_moves(&s);
    assert_eq!(moves.len(), 2, "{moves:?}");
    assert!(moves.iter().all(|m| m.captured == 1u64 << 13));
    let tos: Vec<u8> = moves.iter().map(|m| m.to).collect();
    assert!(tos.contains(&9) && tos.contains(&4), "{tos:?}");
}

#[test]
fn king_cannot_cross_two_enemies() {
    // 18 and 13 both occupied: first victim 18 has no empty landing beyond.
    let s = st(&[22], &[18, 13], &[22], Color::White);
    assert!(captures(&s).is_empty());
}

#[test]
fn king_cannot_jump_own_piece() {
    let s = st(&[22, 18], &[], &[22], Color::White);
    assert!(captures(&s).is_empty());
}

// ── Promotion (stop vs pass-through) ────────────────────────────────────────

#[test]
fn man_promotes_on_stopping_back_rank() {
    // White man at 6 steps quietly up-left to 0 (row 0) → crowned.
    let s = st(&[6], &[], &[], Color::White);
    let m = Move {
        from: 6,
        to: 0,
        captured: 0,
    };
    let after = s.apply(m);
    assert!(after.white & (1 << 0) != 0);
    assert!(after.kings & (1 << 0) != 0, "stopping on row 0 promotes");
}

#[test]
fn man_does_not_promote_passing_through_back_rank() {
    // White man at 13 must take the 2-chain 13→(over 8)→2 [row 0] →(over 7)→11.
    // It touches the back rank at 2 but does not STOP there → no promotion.
    let s = st(&[13], &[8, 7], &[], Color::White);
    let moves = legal_moves(&s);
    assert_eq!(moves.len(), 1, "{moves:?}");
    let m = moves[0];
    assert_eq!(m.to, 11);
    assert_eq!(m.captured, (1u64 << 8) | (1u64 << 7));
    let after = s.apply(m);
    assert!(after.white & (1 << 11) != 0);
    assert!(
        after.kings & (1 << 11) == 0,
        "passing through row 0 must NOT promote"
    );
}

// ── Terminal / draw ─────────────────────────────────────────────────────────

#[test]
fn no_legal_move_is_a_loss() {
    // White's lone man at 49 is boxed in: 43/44 occupied, no landing beyond.
    let s = st(&[49], &[43, 44, 38], &[], Color::White);
    assert!(!any_legal_move(&s));
    assert!(Checkers::is_terminal(&s));
    assert_eq!(Checkers::winner(&s), Some(Player::P1), "stm (White) loses");
}

#[test]
fn idle_counter_draws_and_resets() {
    // Hitting the idle cap is a draw (no winner), terminal, eval 0.
    let mut s = st(&[22], &[0], &[22, 0], Color::White);
    s.idle = DRAW_PLIES;
    assert!(Checkers::is_terminal(&s));
    assert_eq!(Checkers::winner(&s), None);
    assert_eq!(CheckersHeuristic::default().eval(&s, Player::P0), 0);

    // A quiet king move increments idle; a man move / capture resets it.
    let king_only = st(&[22], &[0], &[22], Color::White);
    let km = Move {
        from: 22,
        to: 17,
        captured: 0,
    };
    assert_eq!(king_only.apply(km).idle, 1, "quiet king move advances idle");

    let with_man = State {
        idle: 7,
        ..st(&[22], &[0], &[], Color::White)
    };
    let man_step = Move {
        from: 22,
        to: 17,
        captured: 0,
    };
    assert_eq!(with_man.apply(man_step).idle, 0, "man move resets idle");
}

// ── Evaluation ──────────────────────────────────────────────────────────────

#[test]
fn eval_is_antisymmetric_on_nonterminal() {
    // White up a man → strictly good for White, mirror-opposite for Black.
    let s = st(&[22, 30, 31], &[5, 6], &[], Color::White);
    assert!(any_legal_move(&s));
    let h = CheckersHeuristic::default();
    let a = h.eval(&s, Player::P0);
    let b = h.eval(&s, Player::P1);
    assert_eq!(a, -b);
    assert!(a > 0, "White is up material: {a}");
}

// ── Serialization / encoding ────────────────────────────────────────────────

#[test]
fn state_key_round_trips() {
    let mut states = vec![State::initial()];
    states.push(st(&[22, 30], &[5, 13], &[22], Color::Black));
    for s in states {
        let k = Checkers::state_key(&s);
        assert_eq!(Checkers::parse_state_key(&k), Some(s), "key={k}");
    }
}

#[test]
fn action_index_round_trips_endpoints() {
    let s = st(&[22], &[18, 9, 27], &[], Color::White);
    for m in legal_moves(&s) {
        let i = <Checkers as Encoder>::action_index(m);
        assert!(i < Checkers::ACTION_COUNT);
        let back = <Checkers as Encoder>::index_to_move(i);
        assert_eq!((back.from, back.to), (m.from, m.to));
    }
}

#[test]
fn mirror_move_is_an_involution() {
    let m = Move {
        from: 13,
        to: 11,
        captured: (1u64 << 8) | (1u64 << 7),
    };
    let twice =
        <Checkers as Encoder>::mirror_move(Player::P0, <Checkers as Encoder>::mirror_move(Player::P0, m));
    assert_eq!(twice, m);
}

#[test]
fn encode_has_expected_length() {
    assert_eq!(Checkers::encode(&State::initial()).len(), Checkers::FEATURE_LEN);
}

// ── Shared engine reuse (the no-duplication payoff) ─────────────────────────

#[test]
fn shared_search_and_arena_play_checkers() {
    use crate::arena::{play_game, BotConfig};
    use crate::search::Search;

    // The generic negamax engine (G inferred via CheckersHeuristic::G) picks a
    // legal move from the opening — no checkers-specific search code exists.
    let h = CheckersHeuristic::default();
    let mut s = Search::new(&h);
    let best = s.search(&State::initial(), 5).best.expect("engine returns a move");
    assert!(legal_moves(&State::initial()).contains(&best), "illegal: {best:?}");

    // The generic arena match driver plays a full self-play game to a result.
    // (debug_assert inside play_game rejects any illegal move it is handed.)
    let cfg = BotConfig::new(h, 3);
    let (_outcome, plies) = play_game(&cfg, &cfg, 200);
    assert!(plies > 0, "self-play produced no moves");
}

// ── Invariants over a deterministic random playout ──────────────────────────

#[test]
fn random_playout_preserves_occupancy_invariants() {
    let mut rng: u64 = 0xDA5_C0FFEE_u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for _ in 0..200 {
        let mut s = State::initial();
        for _ in 0..120 {
            if Checkers::is_terminal(&s) {
                break;
            }
            let moves = legal_moves(&s);
            assert!(!moves.is_empty(), "non-terminal but no moves");
            let m = moves[(next() as usize) % moves.len()];
            let before = s.occ().count_ones();
            let ncaptured = m.captured.count_ones();
            assert!(ncaptured < before, "captured more pieces than on board");
            s = s.apply(m);
            // No square is both colours; every king sits on an occupied square.
            assert_eq!(s.white & s.black, 0, "colour overlap");
            assert_eq!(s.kings & !s.occ(), 0, "king on empty square");
            // Piece count drops by exactly the captured count.
            assert_eq!(s.occ().count_ones() + ncaptured, before, "piece-count drift");
        }
    }
}
