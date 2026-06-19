//! Leaf evaluation. `Evaluator` is the seam the ML model will later plug into:
//! same search, swap the scoring of leaf states.

use crate::moves::distance_to_goal;
use crate::state::{Side, State};

/// Score a state from the perspective of `side` (negamax convention: higher is
/// better for `side`). Units are centi-steps; squash to 0..100 via `win_prob`.
pub trait Evaluator {
    fn eval(&self, state: &State, side: Side) -> i32;
}

/// Hand-tuned heuristic.
///
/// Terms (all from the scored side's point of view):
///   - `w_path` : shortest-path race in half-steps,
///     `2*(dist_opp - dist_me) ± 1` for the side-to-move tie-break. Doubling
///     keeps the tempo nudge below a full path step so it only breaks ties
///     between equidistant positions instead of distorting the race margin.
///   - `w_wall` : wall-stock difference. Set HIGH (~2 path-steps per wall). The
///     `w_path` race term credits "delay opponent by 1" exactly as much as
///     "advance myself by 1", but advancing *wins* while a wall only delays and
///     spends a finite resource. Without a strong stock cost, deeper search
///     discovers multi-tempo wall combos that inflate the opponent's distance,
///     stops racing, burns every wall, and then LOSES — measured as deeper
///     search losing to shallower (d4 lost 0-8 to d2 at w_wall=4). Pricing the
///     stock at ~100 restores depth monotonicity (d4 beats d2 16-0) by making
///     the engine spend a wall only when the delay it buys clearly exceeds the
///     tempo and resource it costs.
///
/// Every term negates under swapping `side`/`opp`, so the eval is exactly
/// antisymmetric (zero-sum): `eval(s, South) == -eval(s, North)` on all
/// non-terminal states. Negamax relies on this — a non-antisymmetric term gets
/// mis-signed by ply parity and corrupts odd-depth search.
#[derive(Clone, Copy, Debug)]
pub struct Heuristic {
    pub w_path: i32,
    pub w_wall: i32,
    // ── Gen-2 terms; every field defaults to a no-op so `Default` reproduces the
    // previously-deployed eval byte-for-byte (the clean A/B baseline). ──
    /// Parity sign bonus: rewards being on the winning side of the race without
    /// re-pricing each marginal step (so it never tempts wall-trading). Default 0.
    pub w_lead_quant: i32,
    /// Exact endgame race resolution: when the outcome is *provably* decided
    /// (pure-race loser has zero walls left), return a decisive [`ENDGAME_WIN`]
    /// instead of a fuzzy race margin. The "better endgame". Default off.
    pub exact_endgame: bool,
    /// Corridor-jump safety band for the resolver: 0 = strict (mover wins ties),
    /// 1 = conservative (require a one-step cushion). Default 0.
    pub endgame_margin: i32,
}

impl Default for Heuristic {
    /// Reproduces the previously-deployed eval exactly: race + wall stock only,
    /// no endgame resolution. Gen-2 terms are inert (weight 0 / flag off).
    fn default() -> Self {
        Heuristic {
            w_path: 50,
            w_wall: 100,
            w_lead_quant: 0,
            exact_endgame: false,
            endgame_margin: 0,
        }
    }
}

impl Heuristic {
    /// Build a config from `WC_EVAL_*` environment variables. An absent var
    /// leaves the `Default` (== legacy) value, so a stripped environment is
    /// exactly the previously-deployed eval. Used by the native arena/xmatch/
    /// tune binaries so a match can flip eval terms without recompiling.
    pub fn from_env() -> Self {
        use crate::search::{env_flag, env_num};
        let mut h = Heuristic::default();
        env_num("WC_EVAL_W_PATH", &mut h.w_path);
        env_num("WC_EVAL_W_WALL", &mut h.w_wall);
        env_num("WC_EVAL_W_LEAD_QUANT", &mut h.w_lead_quant);
        env_flag("WC_EVAL_EXACT_ENDGAME", &mut h.exact_endgame);
        env_num("WC_EVAL_ENDGAME_MARGIN", &mut h.endgame_margin);
        h.endgame_margin = h.endgame_margin.clamp(0, 4); // never negative / absurd
        h
    }

    /// Explicit constructor for the WASM deploy path (WASM has no env vars).
    /// `w_lead_quant`/`endgame_margin` keep their inert defaults.
    pub fn new(w_path: i32, w_wall: i32, exact_endgame: bool) -> Self {
        Heuristic {
            w_path,
            w_wall,
            exact_endgame,
            ..Heuristic::default()
        }
    }
}

/// Large finite magnitude for terminal states (kept below i32 saturation so
/// alpha-beta windows never overflow).
pub const WIN_SCORE: i32 = 1_000_000;
const UNREACHABLE: u16 = 1_000; // path blocked (shouldn't happen under rules)

/// Decisive magnitude for a *provably resolved* (but not yet terminal) race.
/// STRICTLY below `WIN_SCORE` so a real terminal win is always preferred (the
/// engine still races to actually finish) and so the search's `score.abs() >=
/// WIN_SCORE` iterative-deepening break and `mate_zone` (`beta >= WIN_SCORE`)
/// pruning guards are NOT tripped by a resolved-but-non-terminal node. Far above
/// the heuristic range (a few thousand centi-steps) so a resolved race always
/// dominates positional scoring and the engine commits to the won/lost line.
pub const ENDGAME_WIN: i32 = WIN_SCORE - 1000; // 999_000

/// Exact pure-race endgame resolver. Returns `Some(definite winner)` iff the
/// outcome is *provably* decided, else `None` (fall through to the heuristic).
///
/// `dist_side`/`dist_opp` are the BFS distances already computed in [`eval`], so
/// no path is recomputed. **Side-invariant by construction**: the winner depends
/// only on `state.turn`, the two distances, and `walls_left` — never on the
/// querying `side`. This is the load-bearing antisymmetry property; the `side`
/// argument is used solely to pair the two distances into the mover's frame.
///
/// Parity: with side-to-move `M`, `M` reaches goal at absolute ply
/// `2*dist_M - 1` and the other at `2*dist_O`, so `M` wins iff `dist_M <=
/// dist_O` (mover wins ties). Soundness boundary: the result is definite only
/// when the racing *loser* has zero walls left — with no walls it can only move
/// its pawn, which can neither lengthen the winner's BFS path (walls are the
/// only edge-remover) nor permanently block it (jump rules + alternate routes).
/// `margin` shrinks the claimable band to absorb the rare width-1 corridor
/// jump-tempo (0 = strict, 1 = conservative).
#[inline]
fn race_outcome(
    state: &State,
    side: Side,
    dist_side: i32,
    dist_opp: i32,
    margin: i32,
) -> Option<Side> {
    // Malformed / blocked board never asserts a decision.
    if dist_side >= UNREACHABLE as i32 || dist_opp >= UNREACHABLE as i32 {
        return None;
    }
    // Re-pair distances into the mover's frame so the verdict is side-invariant.
    let mover = state.turn;
    let (dist_mover, dist_other) = if mover == side {
        (dist_side, dist_opp)
    } else {
        (dist_opp, dist_side)
    };
    let winner = if dist_mover + margin <= dist_other {
        mover
    } else if dist_other + margin < dist_mover {
        mover.other()
    } else {
        return None; // inside the margin band: undecided
    };
    // Definite only if the LOSER cannot alter any distance (zero walls left).
    if state.walls_left[winner.other().idx()] == 0 {
        Some(winner)
    } else {
        None
    }
}

impl Evaluator for Heuristic {
    fn eval(&self, state: &State, side: Side) -> i32 {
        if let Some(w) = state.winner {
            return if w == side { WIN_SCORE } else { -WIN_SCORE };
        }
        let opp = side.other();
        let dist_me = distance_to_goal(state, state.pawn(side), side.goal_row())
            .unwrap_or(UNREACHABLE) as i32;
        let dist_opp =
            distance_to_goal(state, state.pawn(opp), opp.goal_row()).unwrap_or(UNREACHABLE) as i32;

        // Exact endgame: a provably-decided race scores decisively, so the engine
        // plays a perfect won/lost endgame and never burns walls in a won race.
        // The score is GRADED by the winner's remaining distance — a flat
        // decisive value loses the progress gradient and stalls a won race into a
        // ply-cap draw (measured: flat → ~8% draws, graded → ~0%). `dist` is tiny
        // (≤ board diameter) vs `ENDGAME_WIN`, so the result stays decisive and
        // strictly inside (heuristic range, WIN_SCORE). Antisymmetric: from the
        // loser's POV `dist_opp` is the same winner distance, so the two POVs negate.
        if self.exact_endgame {
            if let Some(w) = race_outcome(state, side, dist_me, dist_opp, self.endgame_margin) {
                return if w == side {
                    ENDGAME_WIN - dist_me
                } else {
                    -ENDGAME_WIN + dist_opp
                };
            }
        }

        // Race in half-steps; the ±1 side-to-move tie-break stays below a full
        // step. Antisymmetric: every factor negates when side and opp swap.
        let path = 2 * (dist_opp - dist_me) + if state.turn == side { 1 } else { -1 };
        let wall = state.walls_left[side.idx()] as i32 - state.walls_left[opp.idx()] as i32;

        // Parity sign bonus: `signum` is odd, `path` is the (antisymmetric) race
        // quantity, so the term negates under side↔opp swap. Inert at weight 0.
        let lead_bonus = path.signum();

        self.w_path * path + self.w_wall * wall + self.w_lead_quant * lead_bonus
    }
}

/// Map a side-relative eval to that side's win probability in 0..=100.
///
/// Display-only: a logistic squash, not a trained calibration. `k` is the score
/// gap worth ~73% (one e-fold). The eval scores one full board-step of race lead
/// at `w_path * 2 = 100` units, so `k = 200` reads as "two clear steps ahead ≈
/// 73%": 1 step → 62%, 2 → 73%, 3 → 78%. Pick `k` in those step-units; a real
/// calibration would fit it to self-play outcomes (see the training loop).
pub fn win_prob(eval: i32, k: f64) -> u8 {
    if eval >= WIN_SCORE {
        return 100;
    }
    if eval <= -WIN_SCORE {
        return 0;
    }
    let p = 1.0 / (1.0 + (-(eval as f64) / k).exp());
    (p * 100.0).round().clamp(0.0, 100.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::{distance_to_goal, legal_moves};
    use crate::state::{Cell, State};

    /// A Heuristic with every Gen-2 term enabled — used to exercise the new
    /// code paths (lead bonus + exact endgame) in the antisymmetry guards.
    fn full_cfg() -> Heuristic {
        Heuristic {
            w_path: 50,
            w_wall: 100,
            w_lead_quant: 30,
            exact_endgame: true,
            endgame_margin: 0,
        }
    }

    fn xorshift(seed: u64) -> impl FnMut() -> u64 {
        let mut rng = seed;
        move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        }
    }

    /// `eval(s, South) == -eval(s, North)` on every non-terminal state reached
    /// by pseudo-random play. Negamax assumes this; a broken term would only
    /// surface at odd search depths, so guard it directly.
    #[test]
    fn eval_is_antisymmetric() {
        let h = Heuristic::default();
        let mut rng: u64 = 0x9e3779b97f4a7c15;
        let mut next = || {
            // xorshift64 — deterministic, no std rng dependency.
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..200 {
            let mut state = State::initial();
            for _ in 0..60 {
                if state.winner.is_some() {
                    break;
                }
                assert_eq!(
                    h.eval(&state, Side::South),
                    -h.eval(&state, Side::North),
                    "non-antisymmetric eval at {state:?}"
                );
                let moves = legal_moves(&state);
                if moves.is_empty() {
                    break;
                }
                let mv = moves[(next() as usize) % moves.len()];
                state = state.apply(mv);
            }
        }
    }

    /// Antisymmetry must still hold with EVERY Gen-2 term on (lead bonus +
    /// exact-endgame resolver). A sign slip in the new terms only surfaces at
    /// odd depth, so guard it directly over random play.
    #[test]
    fn eval_antisymmetric_with_all_gen2_terms() {
        let h = full_cfg();
        let mut next = xorshift(0xd1b54a32d192ed03);
        for _ in 0..200 {
            let mut state = State::initial();
            for _ in 0..60 {
                if state.winner.is_some() {
                    break;
                }
                assert_eq!(
                    h.eval(&state, Side::South),
                    -h.eval(&state, Side::North),
                    "non-antisymmetric (gen2) at {state:?}"
                );
                let moves = legal_moves(&state);
                if moves.is_empty() {
                    break;
                }
                state = state.apply(moves[(next() as usize) % moves.len()]);
            }
        }
    }

    /// `race_outcome` must not depend on the querying `side`: for a fixed state
    /// the South-POV and North-POV queries (with their distance args swapped)
    /// must agree on the winner. Catches any accidental POV dependence.
    #[test]
    fn race_outcome_is_side_invariant() {
        let mut next = xorshift(0x517cc1b727220a95);
        for _ in 0..200 {
            let mut state = State::initial();
            for _ in 0..60 {
                if state.winner.is_some() {
                    break;
                }
                let ds = distance_to_goal(&state, state.pawn(Side::South), Side::South.goal_row())
                    .unwrap_or(UNREACHABLE) as i32;
                let dn = distance_to_goal(&state, state.pawn(Side::North), Side::North.goal_row())
                    .unwrap_or(UNREACHABLE) as i32;
                for m in 0..=2 {
                    // South POV: dist_side=ds, dist_opp=dn. North POV: swapped.
                    assert_eq!(
                        race_outcome(&state, Side::South, ds, dn, m),
                        race_outcome(&state, Side::North, dn, ds, m),
                        "race_outcome POV-dependent (margin={m}) at {state:?}"
                    );
                }
                let moves = legal_moves(&state);
                if moves.is_empty() {
                    break;
                }
                state = state.apply(moves[(next() as usize) % moves.len()]);
            }
        }
    }

    /// Guard 0: an unreachable (sentinel) distance never asserts a decision.
    #[test]
    fn race_outcome_bails_on_unreachable() {
        let s = State::initial();
        let u = UNREACHABLE as i32;
        assert_eq!(race_outcome(&s, Side::South, u, 3, 0), None);
        assert_eq!(race_outcome(&s, Side::South, 3, u, 0), None);
    }

    /// A provably-won race (mover strictly closer, loser has zero walls) scores
    /// exactly `ENDGAME_WIN` from the winner's POV and `-ENDGAME_WIN` from the
    /// loser's — and the rule is symmetric across both POVs.
    #[test]
    fn race_resolution_scores_decided_win() {
        // South to move at (7,5): dist 2 to row 9. North at (8,5): dist 7 to row 1.
        // North (the racing loser) has 0 walls -> South definitely wins.
        let state = State {
            pawns: [Cell::new(7, 5), Cell::new(8, 5)],
            h_walls: 0,
            v_walls: 0,
            walls_left: [5, 0],
            turn: Side::South,
            winner: None,
        };
        // South dist 2 from (7,5) -> graded score ENDGAME_WIN - 2; loser POV negates.
        let h = full_cfg();
        assert_eq!(h.eval(&state, Side::South), ENDGAME_WIN - 2);
        assert_eq!(h.eval(&state, Side::North), -(ENDGAME_WIN - 2));
        // Still firmly in the decisive band (race-to-finish gradient is tiny).
        assert!(h.eval(&state, Side::South) > ENDGAME_WIN - 100);
    }

    /// A pure-race tie is won by the mover at margin 0, but the conservative
    /// margin (1) refuses to claim it (falls through to the heuristic).
    #[test]
    fn race_resolution_tie_obeys_margin() {
        // South (7,5) dist 2; North (3,5) dist 2 -> exact tie. North 0 walls.
        let state = State {
            pawns: [Cell::new(7, 5), Cell::new(3, 5)],
            h_walls: 0,
            v_walls: 0,
            walls_left: [5, 0],
            turn: Side::South,
            winner: None,
        };
        let strict = full_cfg(); // margin 0: mover wins the tie
        assert_eq!(strict.eval(&state, Side::South), ENDGAME_WIN - 2); // South dist 2

        let conservative = Heuristic {
            endgame_margin: 1,
            ..full_cfg()
        };
        // Tie no longer claimed -> heuristic score, strictly below ENDGAME_WIN.
        assert!(conservative.eval(&state, Side::South).abs() < ENDGAME_WIN);
    }

    /// The resolver must NOT fire while the racing loser still has walls — a
    /// well-placed wall could still flip the race, so it is not provably decided.
    #[test]
    fn race_resolution_silent_when_loser_has_walls() {
        let state = State {
            pawns: [Cell::new(7, 5), Cell::new(8, 5)],
            h_walls: 0,
            v_walls: 0,
            walls_left: [5, 1], // North (loser) keeps a wall
            turn: Side::South,
            winner: None,
        };
        let h = full_cfg();
        assert!(h.eval(&state, Side::South).abs() < ENDGAME_WIN);
    }

    /// `Heuristic::default()` must reproduce the legacy eval byte-for-byte
    /// (race + wall stock only) on every randomly reached state, for both POVs.
    #[test]
    fn default_eval_matches_legacy_formula() {
        let h = Heuristic::default();
        let mut next = xorshift(0x9e3779b97f4a7c15);
        for _ in 0..200 {
            let mut state = State::initial();
            for _ in 0..60 {
                if state.winner.is_some() {
                    break;
                }
                for side in [Side::South, Side::North] {
                    let opp = side.other();
                    let dm = distance_to_goal(&state, state.pawn(side), side.goal_row())
                        .unwrap_or(UNREACHABLE) as i32;
                    let dop = distance_to_goal(&state, state.pawn(opp), opp.goal_row())
                        .unwrap_or(UNREACHABLE) as i32;
                    let path = 2 * (dop - dm) + if state.turn == side { 1 } else { -1 };
                    let wall =
                        state.walls_left[side.idx()] as i32 - state.walls_left[opp.idx()] as i32;
                    let legacy = 50 * path + 100 * wall;
                    assert_eq!(h.eval(&state, side), legacy, "default != legacy at {state:?}");
                }
                let moves = legal_moves(&state);
                if moves.is_empty() {
                    break;
                }
                state = state.apply(moves[(next() as usize) % moves.len()]);
            }
        }
    }
}
