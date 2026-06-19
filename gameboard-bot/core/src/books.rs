//! Compact precomputed books/caches consumed by search.

use std::collections::HashMap;

use crate::action::index_to_move;
use crate::features::mirror_move;
use crate::moves::{is_legal, legal_moves};
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

/// Generic move book keyed by [`state_key`].
///
/// Records use `{"type":"book","key":"...","a":123}` with action indices in
/// the side-to-move me-frame.
#[derive(Clone, Debug, Default)]
pub struct MoveBook {
    actions: HashMap<String, usize>,
}

impl EndgameBook {
    pub fn from_jsonl(text: &str) -> Self {
        let actions = read_actions(text, "hint");
        EndgameBook { actions }
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    pub fn best_move(&self, state: &State) -> Option<Move> {
        action_to_move(&self.actions, state)
    }
}

impl MoveBook {
    pub fn from_jsonl(text: &str) -> Self {
        MoveBook {
            actions: read_actions(text, "book"),
        }
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    pub fn best_move(&self, state: &State) -> Option<Move> {
        action_to_move(&self.actions, state)
    }
}

fn read_actions(text: &str, record_type: &str) -> HashMap<String, usize> {
    let type_tag = format!("\"type\":\"{record_type}\"");
    text.lines()
        .filter(|line| line.contains(&type_tag))
        .filter_map(|line| Some((field_str(line, "key")?.to_string(), field_usize(line, "a")?)))
        .collect()
}

fn action_to_move(actions: &HashMap<String, usize>, state: &State) -> Option<Move> {
    let action = *actions.get(&state_key(state))?;
    let mv = mirror_move(state.turn, index_to_move(action));
    is_legal(state, mv).then_some(mv)
}

/// Exact one-ply tactical endgame: if the side to move can step onto its goal
/// row now, do it before spending any search budget.
pub fn immediate_winning_move(state: &State) -> Option<Move> {
    legal_moves(state).into_iter().find(|mv| match mv {
        Move::Pawn(cell) => cell.r == state.turn.goal_row(),
        Move::Wall(_) => false,
    })
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

    #[test]
    fn immediate_winning_move_finds_goal_step() {
        let state = State {
            pawns: [Cell::new(8, 3), Cell::new(9, 5)],
            h_walls: 0,
            v_walls: 0,
            walls_left: [10, 10],
            turn: Side::South,
            winner: None,
        };

        assert_eq!(
            immediate_winning_move(&state),
            Some(Move::Pawn(Cell::new(9, 3)))
        );
    }

    #[test]
    fn reads_generic_move_book() {
        let text = concat!(
            "{\"type\":\"meta\",\"format\":\"wallchess-move-book-v1\"}\n",
            "{\"type\":\"book\",\"key\":\"s|15|95|1010|\",\"a\":13,\"score\":50}\n",
        );
        let book = MoveBook::from_jsonl(text);
        assert_eq!(book.len(), 1);
        assert_eq!(
            book.best_move(&State::initial()),
            Some(Move::Pawn(Cell::new(2, 5)))
        );
    }
}
