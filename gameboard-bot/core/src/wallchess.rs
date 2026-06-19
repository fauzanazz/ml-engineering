//! Wall Chess (Quoridor) as a [`Game`] implementation.
//!
//! This is the glue that binds the existing Quoridor rules engine
//! ([`crate::state`], [`crate::moves`], [`crate::action`], [`crate::features`])
//! to the generic search/arena/NN pipeline. The numerics live in those modules
//! unchanged — this file only wires them through the [`Game`] / [`Evaluator`] /
//! [`Encoder`] contracts. (Phase-3 of the multi-game refactor relocates the
//! rules modules under `games/wallchess/`; today they sit at the crate root and
//! this module re-binds them.)

use crate::action::{action_index, index_to_move, ACTION_COUNT};
use crate::features::{encode, mirror_move, FEATURE_LEN};
use crate::game::{Encoder, Game, Player};
use crate::moves::{is_legal, legal_moves, pawn_moves, search_moves, search_moves_wide};
use crate::state::{Move, Orientation, Side, State};

/// Zero-size marker type implementing [`Game`] for Wall Chess.
#[derive(Clone, Copy, Debug)]
pub struct WallChess;

#[inline]
pub fn side_to_player(s: Side) -> Player {
    match s {
        Side::South => Player::P0,
        Side::North => Player::P1,
    }
}

#[inline]
pub fn player_to_side(p: Player) -> Side {
    match p {
        Player::P0 => Side::South,
        Player::P1 => Side::North,
    }
}

/// History/killer table size. Indices:
///   Pawn(to)    → 0..=153   (to.r * 16 + to.c)
///   Wall(r,c,H) → 256..=319
///   Wall(r,c,V) → 320..=383
const HISTORY_SIZE: usize = 384;

impl Game for WallChess {
    type State = State;
    type Move = Move;

    const ID: &'static str = "wallchess";
    const ACTION_COUNT: usize = ACTION_COUNT; // 209
    const FEATURE_LEN: usize = FEATURE_LEN; // 300
    const MOVE_INDEX_SPACE: usize = HISTORY_SIZE; // 384

    #[inline]
    fn initial() -> State {
        State::initial()
    }

    #[inline]
    fn turn(s: &State) -> Player {
        side_to_player(s.turn)
    }

    #[inline]
    fn winner(s: &State) -> Option<Player> {
        s.winner.map(side_to_player)
    }

    #[inline]
    fn is_terminal(s: &State) -> bool {
        s.winner.is_some()
    }

    #[inline]
    fn apply(s: &State, mv: Move) -> State {
        s.apply(mv)
    }

    #[inline]
    fn null_move(s: &State) -> State {
        s.null_move()
    }

    #[inline]
    fn is_legal(s: &State, mv: Move) -> bool {
        is_legal(s, mv)
    }

    #[inline]
    fn legal_moves(s: &State) -> Vec<Move> {
        legal_moves(s)
    }

    #[inline]
    fn search_moves(s: &State) -> Vec<Move> {
        search_moves(s)
    }

    #[inline]
    fn search_moves_wide(s: &State) -> Vec<Move> {
        search_moves_wide(s)
    }

    /// Wall placements are the quiet moves; pawn advances are tactical and must
    /// never be reduced/pruned. (Matches the legacy `is_wall` predicate exactly.)
    #[inline]
    fn is_quiet(mv: Move) -> bool {
        matches!(mv, Move::Wall(_))
    }

    /// Fast 64-bit state hash for array-TT indexing. Mixes walls, pawns, wall
    /// counts, and turn; never returns 0 (the empty-slot sentinel). Reproduces
    /// the previously-deployed `search::state_hash` byte-for-byte.
    #[inline]
    fn hash(s: &State) -> u64 {
        let mut h = s.h_walls;
        h ^= s.v_walls.wrapping_mul(0x9e3779b97f4a7c15u64);
        h ^= (s.pawns[0].r as u64 * 10 + s.pawns[0].c as u64)
            .wrapping_mul(0x517cc1b727220a95u64)
            << 32;
        h ^= (s.pawns[1].r as u64 * 10 + s.pawns[1].c as u64)
            .wrapping_mul(0xbf58476d1ce4e5b9u64)
            << 16;
        h ^= (s.walls_left[0] as u64) << 56;
        h ^= (s.walls_left[1] as u64) << 48;
        h ^= (s.turn as u64) << 63;
        h ^= h >> 30;
        h = h.wrapping_mul(0xbf58476d1ce4e5b9u64);
        h ^= h >> 27;
        if h == 0 {
            1
        } else {
            h
        }
    }

    #[inline]
    fn move_order_index(mv: Move) -> usize {
        match mv {
            Move::Pawn(c) => c.r as usize * 16 + c.c as usize,
            Move::Wall(w) => match w.o {
                Orientation::H => 256 + (w.r as usize - 1) * 8 + (w.c as usize - 1),
                Orientation::V => 320 + (w.r as usize - 1) * 8 + (w.c as usize - 1),
            },
        }
    }

    #[inline]
    fn immediate_winning_move(s: &State) -> Option<Move> {
        pawn_moves(s, s.turn)
            .into_iter()
            .find(|to| to.r == s.turn.goal_row())
            .map(Move::Pawn)
    }

    #[inline]
    fn state_key(s: &State) -> String {
        crate::state_key(s)
    }

    #[inline]
    fn parse_state_key(k: &str) -> Option<State> {
        crate::parse_state_key(k)
    }
}

impl Encoder for WallChess {
    type G = WallChess;

    #[inline]
    fn encode(state: &State) -> Vec<f32> {
        encode(state)
    }

    #[inline]
    fn action_index(mv: Move) -> usize {
        action_index(mv)
    }

    #[inline]
    fn index_to_move(i: usize) -> Move {
        index_to_move(i)
    }

    #[inline]
    fn mirror_move(p: Player, mv: Move) -> Move {
        mirror_move(player_to_side(p), mv)
    }
}
