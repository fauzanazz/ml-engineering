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
///   - `w_wall` : wall-stock difference. Kept low — a wall's real worth shows
///     up through `w_path` when it lengthens the opponent's path; over-valuing
///     the *stock* makes the engine hoard/spam walls.
///
/// Every term negates under swapping `side`/`opp`, so the eval is exactly
/// antisymmetric (zero-sum): `eval(s, South) == -eval(s, North)` on all
/// non-terminal states. Negamax relies on this — a non-antisymmetric term gets
/// mis-signed by ply parity and corrupts odd-depth search.
#[derive(Clone, Copy, Debug)]
pub struct Heuristic {
    pub w_path: i32,
    pub w_wall: i32,
}

impl Default for Heuristic {
    fn default() -> Self {
        Heuristic {
            w_path: 50,
            w_wall: 4,
        }
    }
}

/// Large finite magnitude for terminal states (kept below i32 saturation so
/// alpha-beta windows never overflow).
pub const WIN_SCORE: i32 = 1_000_000;
const UNREACHABLE: u16 = 1_000; // path blocked (shouldn't happen under rules)

impl Evaluator for Heuristic {
    fn eval(&self, state: &State, side: Side) -> i32 {
        if let Some(w) = state.winner {
            return if w == side { WIN_SCORE } else { -WIN_SCORE };
        }
        let opp = side.other();
        let dist_me =
            distance_to_goal(state, state.pawn(side), side.goal_row()).unwrap_or(UNREACHABLE) as i32;
        let dist_opp = distance_to_goal(state, state.pawn(opp), opp.goal_row())
            .unwrap_or(UNREACHABLE) as i32;

        // Race in half-steps; the ±1 side-to-move tie-break stays below a full
        // step. Antisymmetric: every factor negates when side and opp swap.
        let path = 2 * (dist_opp - dist_me) + if state.turn == side { 1 } else { -1 };
        let wall = state.walls_left[side.idx()] as i32 - state.walls_left[opp.idx()] as i32;

        self.w_path * path + self.w_wall * wall
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
    use crate::moves::legal_moves;
    use crate::state::State;

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
}
