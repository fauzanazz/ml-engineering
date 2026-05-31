//! Compact precomputed books/caches consumed by search.

use std::collections::HashMap;

use crate::action::index_to_move;
use crate::features::mirror_move;
use crate::moves::is_legal;
use crate::state::{Move, State};
use crate::state_key;

/// Compressed endgame hints keyed by [`state_key`].
///
/// Actions are stored in the side-to-move "me-frame" used by the policy head.
/// Lookup mirrors the action back into the state's absolute board frame.
#[derive(Clone, Debug, Default)]
pub struct EndgameBook {
    actions: HashMap<String, usize>,
}

impl EndgameBook {
    pub fn from_jsonl(text: &str) -> Self {
        let actions = text
            .lines()
            .filter(|line| line.contains("\"type\":\"hint\""))
            .filter_map(|line| Some((field_str(line, "key")?.to_string(), field_usize(line, "a")?)))
            .collect();
        EndgameBook { actions }
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    pub fn best_move(&self, state: &State) -> Option<Move> {
        let action = *self.actions.get(&state_key(state))?;
        let mv = mirror_move(state.turn, index_to_move(action));
        is_legal(state, mv).then_some(mv)
    }
}

fn field_str<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!("\"{name}\":\"");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}

fn field_usize(line: &str, name: &str) -> Option<usize> {
    let needle = format!("\"{name}\":");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Cell, Side};

    #[test]
    fn reads_hint_and_returns_legal_absolute_move() {
        let text = concat!(
            "{\"type\":\"meta\",\"format\":\"wallchess-endgame-hints-v1\"}\n",
            "{\"type\":\"hint\",\"key\":\"s|11|21|00|\",\"a\":18,\"winner\":\"north\",\"score\":-1000000,\"dist_me\":8,\"dist_opp\":1}\n",
        );
        let book = EndgameBook::from_jsonl(text);
        let state = State {
            pawns: [Cell::new(1, 1), Cell::new(2, 1)],
            h_walls: 0,
            v_walls: 0,
            walls_left: [0, 0],
            turn: Side::South,
            winner: None,
        };

        assert_eq!(book.len(), 1);
        assert!(book.best_move(&state).is_some());
    }

    #[test]
    fn ignores_missing_or_illegal_hint() {
        let book = EndgameBook::from_jsonl(
            "{\"type\":\"hint\",\"key\":\"s|11|21|00|\",\"a\":80,\"winner\":\"north\",\"score\":1}\n",
        );
        let state = State {
            pawns: [Cell::new(1, 1), Cell::new(2, 1)],
            h_walls: 0,
            v_walls: 0,
            walls_left: [0, 0],
            turn: Side::South,
            winner: None,
        };

        assert_eq!(book.best_move(&State::initial()), None);
        assert_eq!(book.best_move(&state), None);
    }
}
