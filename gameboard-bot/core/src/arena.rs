//! Arena: play one configured bot against another and report the result.
//! Foundation for eval tuning — pit weight/depth configs head-to-head.
//!
//! Generic over the game via the evaluator's `E::G` (one type parameter, like
//! [`crate::search::Search`]): Wall Chess and International Draughts share this
//! exact match loop. Players are the game-neutral [`Player::P0`] (moves first)
//! and [`Player::P1`]; each game maps its own colour onto them.

use crate::game::{Evaluator, Game, Player};
use crate::search::Search;

type Gm<E> = <E as Evaluator>::G;
type St<E> = <Gm<E> as Game>::State;
type Mv<E> = <Gm<E> as Game>::Move;

/// A bot configuration: leaf evaluator weights + search depth.
#[derive(Clone, Copy, Debug)]
pub struct BotConfig<E: Evaluator> {
    pub eval: E,
    pub depth: u8,
}

impl<E: Evaluator> BotConfig<E> {
    pub fn new(eval: E, depth: u8) -> Self {
        BotConfig { eval, depth }
    }

    /// Best move for `state` under this config.
    pub fn choose(&self, state: &St<E>) -> Option<Mv<E>> {
        let mut s = Search::new(&self.eval);
        s.search(state, self.depth).best
    }
}

/// Game result from the seating perspective: P0 is the side that moved first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    P0Win,
    P1Win,
    Draw, // ply cap reached without a winner
}

/// Play one game: `p0` config moves first ([`Player::P0`]), `p1` second.
/// `max_plies` caps the game (declares a draw on overrun).
pub fn play_game<E: Evaluator>(
    p0: &BotConfig<E>,
    p1: &BotConfig<E>,
    max_plies: u32,
) -> (Outcome, u32) {
    let mut state = <Gm<E> as Game>::initial();
    let mut ply = 0u32;
    while !<Gm<E> as Game>::is_terminal(&state) && ply < max_plies {
        let cfg = match <Gm<E> as Game>::turn(&state) {
            Player::P0 => p0,
            Player::P1 => p1,
        };
        let mv = match cfg.choose(&state) {
            Some(m) => m,
            None => break, // no legal move (shouldn't happen under rules)
        };
        debug_assert!(
            <Gm<E> as Game>::is_legal(&state, mv),
            "config produced illegal move"
        );
        state = <Gm<E> as Game>::apply(&state, mv);
        ply += 1;
    }
    let outcome = match <Gm<E> as Game>::winner(&state) {
        Some(Player::P0) => Outcome::P0Win,
        Some(Player::P1) => Outcome::P1Win,
        None => Outcome::Draw,
    };
    (outcome, ply)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MatchResult {
    pub a_wins: u32,
    pub b_wins: u32,
    pub draws: u32,
}

/// Play a match of config A vs config B, alternating who starts (P0 moves
/// first, so swapping seats each game removes the first-move advantage bias).
/// `games` games total; results are tracked from A's perspective.
pub fn play_match<E: Evaluator>(
    a: &BotConfig<E>,
    b: &BotConfig<E>,
    games: u32,
    max_plies: u32,
) -> MatchResult {
    let mut res = MatchResult::default();
    for g in 0..games {
        let a_is_p0 = g % 2 == 0;
        let (p0, p1) = if a_is_p0 { (a, b) } else { (b, a) };
        let (outcome, _) = play_game(p0, p1, max_plies);
        let a_won = match outcome {
            Outcome::P0Win => a_is_p0,
            Outcome::P1Win => !a_is_p0,
            Outcome::Draw => {
                res.draws += 1;
                continue;
            }
        };
        if a_won {
            res.a_wins += 1;
        } else {
            res.b_wins += 1;
        }
    }
    res
}
