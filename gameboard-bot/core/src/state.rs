//! Board state: bit-packed, cheap to copy, Zobrist-hashable.
//!
//! Rules mirror the reference TS engine (webui/src/game/engine.ts).
//! 9x9 board, cells (r, c) with r, c in 1..=9.
//! SOUTH starts row 1 -> reaches row 9; NORTH starts row 9 -> reaches row 1.
//!
//! Walls sit on the 8x8 grid of internal intersections, anchor (r, c) in 1..=8:
//!   - horizontal wall at (r, c) spans columns c, c+1 between rows r and r+1
//!     (blocks vertical movement),
//!   - vertical wall at (r, c) spans rows r, r+1 between cols c and c+1
//!     (blocks horizontal movement).

pub const SIZE: u8 = 9;
pub const START_WALLS: u8 = 10;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Side {
    South = 0,
    North = 1,
}

impl Side {
    #[inline]
    pub fn other(self) -> Side {
        match self {
            Side::South => Side::North,
            Side::North => Side::South,
        }
    }

    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }

    /// Goal row a pawn of this side must reach.
    #[inline]
    pub fn goal_row(self) -> u8 {
        match self {
            Side::South => SIZE,
            Side::North => 1,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Orientation {
    H,
    V,
}

/// A cell on the board, 1-based. Stored compact but exposed as (r, c).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Cell {
    pub r: u8,
    pub c: u8,
}

impl Cell {
    #[inline]
    pub fn new(r: u8, c: u8) -> Self {
        Cell { r, c }
    }
    #[inline]
    pub fn in_bounds(self) -> bool {
        self.r >= 1 && self.r <= SIZE && self.c >= 1 && self.c <= SIZE
    }
}

/// A wall placement, anchor 1-based in 1..=8.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Wall {
    pub r: u8,
    pub c: u8,
    pub o: Orientation,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Move {
    Pawn(Cell),
    Wall(Wall),
}

/// Bit-packed game state. h_walls / v_walls index anchor (r,c) as bit
/// (r-1)*8 + (c-1), r,c in 1..=8 -> 0..=63.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct State {
    pub pawns: [Cell; 2],
    pub h_walls: u64,
    pub v_walls: u64,
    pub walls_left: [u8; 2],
    pub turn: Side,
    pub winner: Option<Side>,
}

#[inline]
fn anchor_bit(r: u8, c: u8) -> u64 {
    1u64 << (((r - 1) * 8 + (c - 1)) as u64)
}

#[inline]
fn anchor_in_range(r: u8, c: u8) -> bool {
    r >= 1 && r <= SIZE - 1 && c >= 1 && c <= SIZE - 1
}

impl State {
    pub fn initial() -> Self {
        let mid = SIZE.div_ceil(2); // 5
        State {
            pawns: [Cell::new(1, mid), Cell::new(SIZE, mid)],
            h_walls: 0,
            v_walls: 0,
            walls_left: [START_WALLS, START_WALLS],
            turn: Side::South,
            winner: None,
        }
    }

    #[inline]
    pub fn pawn(&self, side: Side) -> Cell {
        self.pawns[side.idx()]
    }

    #[inline]
    pub fn has_h_wall(&self, r: u8, c: u8) -> bool {
        anchor_in_range(r, c) && (self.h_walls & anchor_bit(r, c)) != 0
    }

    #[inline]
    pub fn has_v_wall(&self, r: u8, c: u8) -> bool {
        anchor_in_range(r, c) && (self.v_walls & anchor_bit(r, c)) != 0
    }

    /// Is a step between orthogonally-adjacent `from` and `to` blocked by a wall?
    /// Non-adjacent inputs are treated as blocked (defensive).
    pub fn is_blocked(&self, from: Cell, to: Cell) -> bool {
        let dr = to.r as i16 - from.r as i16;
        let dc = to.c as i16 - from.c as i16;
        if dr.abs() + dc.abs() != 1 {
            return true;
        }
        if dc == 0 {
            // vertical move -> horizontal wall on the crossed line
            let line_r = from.r.min(to.r);
            self.has_h_wall(line_r, from.c) || self.has_h_wall(line_r, from.c.wrapping_sub(1))
        } else {
            // horizontal move -> vertical wall on the crossed line
            let line_c = from.c.min(to.c);
            self.has_v_wall(from.r, line_c) || self.has_v_wall(from.r.wrapping_sub(1), line_c)
        }
    }

    /// Skip the current side's turn (null move). Used for null-move pruning only.
    #[inline]
    pub fn null_move(&self) -> State {
        let mut next = *self;
        next.turn = self.turn.other();
        next
    }

    /// Apply a move assumed legal. Flips turn, sets winner on goal.
    /// Callers must validate with `moves::is_legal` first.
    pub fn apply(&self, mv: Move) -> State {
        let side = self.turn;
        let mut next = *self;
        next.turn = side.other();
        next.winner = None;
        match mv {
            Move::Pawn(to) => {
                next.pawns[side.idx()] = to;
                if to.r == side.goal_row() {
                    next.winner = Some(side);
                }
            }
            Move::Wall(w) => {
                let bit = anchor_bit(w.r, w.c);
                match w.o {
                    Orientation::H => next.h_walls |= bit,
                    Orientation::V => next.v_walls |= bit,
                }
                next.walls_left[side.idx()] = next.walls_left[side.idx()].saturating_sub(1);
            }
        }
        next
    }
}
