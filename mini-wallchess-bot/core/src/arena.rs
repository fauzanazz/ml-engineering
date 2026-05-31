//! Arena: play one configured bot against another and report the result.
//! Foundation for eval tuning — pit weight/depth configs head-to-head.

use crate::eval::Heuristic;
use crate::moves::is_legal;
use crate::search::Search;
use crate::state::{Move, Side, State};

/// A bot configuration: leaf evaluator weights + search depth.
#[derive(Clone, Copy, Debug)]
pub struct BotConfig {
    pub eval: Heuristic,
    pub depth: u8,
}

impl BotConfig {
    pub fn new(eval: Heuristic, depth: u8) -> Self {
        BotConfig { eval, depth }
    }

    /// Best move for `state` under this config.
    pub fn choose(&self, state: &State) -> Option<Move> {
        let mut s = Search::new(&self.eval);
        s.search(state, self.depth).best
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    SouthWin,
    NorthWin,
    Draw, // ply cap reached without a winner
}

/// Play one game: `south` config moves first, `north` second.
/// `max_plies` caps the game (declares a draw on overrun).
pub fn play_game(south: &BotConfig, north: &BotConfig, max_plies: u32) -> (Outcome, u32) {
    let mut state = State::initial();
    let mut ply = 0u32;
    while state.winner.is_none() && ply < max_plies {
        let cfg = if state.turn == Side::South { south } else { north };
        let mv = match cfg.choose(&state) {
            Some(m) => m,
            None => break, // no legal move (shouldn't happen under rules)
        };
        debug_assert!(is_legal(&state, mv), "config produced illegal move");
        state = state.apply(mv);
        ply += 1;
    }
    let outcome = match state.winner {
        Some(Side::South) => Outcome::SouthWin,
        Some(Side::North) => Outcome::NorthWin,
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

/// Play a match of config A vs config B, alternating who starts (South moves
/// first, so swapping sides each game removes the first-move advantage bias).
/// `games` games total; results are tracked from A's perspective.
pub fn play_match(a: &BotConfig, b: &BotConfig, games: u32, max_plies: u32) -> MatchResult {
    let mut res = MatchResult::default();
    for g in 0..games {
        let a_is_south = g % 2 == 0;
        let (south, north) = if a_is_south { (a, b) } else { (b, a) };
        let (outcome, _) = play_game(south, north, max_plies);
        let a_won = match outcome {
            Outcome::SouthWin => a_is_south,
            Outcome::NorthWin => !a_is_south,
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
