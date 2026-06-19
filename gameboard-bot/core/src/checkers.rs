//! International draughts (10×10 checkers) as a [`Game`] implementation.
//!
//! Second game on the shared engine. Everything above the `State`/`Move` seam —
//! negamax/alpha-beta + the transposition table + the arena match driver — is
//! reused unchanged from Wall Chess (see [`crate::game`]); this module is the
//! per-game rules + evaluation + encoding.
//!
//! ## Ruleset (FMJD International)
//! - 10×10 board, 50 dark squares, PDN-numbered 1..50 (internal indices 0..49).
//! - 20 men per side. White (`P0`) moves first, up the board; Black (`P1`) down.
//! - Men step diagonally forward; **capture in all four directions**.
//! - Kings are *flying*: slide/capture any distance along a diagonal.
//! - Capture is **mandatory** and must take the **maximum number** of pieces
//!   (count only — no material/majority tiebreak).
//! - A man promotes **only if it stops on the back rank** (jumping *through* the
//!   back rank mid-capture does not promote — the "passing"/Turkish rule).
//! - Captured pieces stay on the board as blockers until the whole move
//!   completes, and may not be jumped twice.
//! - No legal move ⇒ side to move loses. 25 king-only non-capturing moves per
//!   side ⇒ draw ([`DRAW_PLIES`]).
//!
//! See `docs/checkers-rules-and-encoding.md` for the design rationale (bitboard
//! packing, the multi-jump-as-one-`Move` representation, the dedup/max-count
//! filter that is the difference between correct and inflated perft, and the
//! provisional action/feature encoding).

use crate::game::{Encoder, Evaluator, Game, Player, WIN_SCORE};

// ── Board geometry ──────────────────────────────────────────────────────────

/// Number of playable (dark) squares.
pub const N: usize = 50;

/// All 50 dark squares set.
const BOARD: u64 = (1u64 << N) - 1;
/// White's 20 starting men: indices 30..=49 (rows 6..9, the bottom half).
const WHITE_INIT: u64 = BOARD ^ ((1u64 << 30) - 1);
/// Black's 20 starting men: indices 0..=19 (rows 0..3, the top half).
const BLACK_INIT: u64 = (1u64 << 20) - 1;
/// White promotes on row 0 (indices 0..=4); Black on row 9 (indices 45..=49).
const WHITE_PROMO: u64 = 0b11111;
const BLACK_PROMO: u64 = 0b11111u64 << 45;

/// Plies of king-only, non-capturing play before a draw (25 moves per side).
pub const DRAW_PLIES: u16 = 50;

// Diagonal directions, indexed 0..4: up-left, up-right, down-left, down-right.
// "Up" = decreasing row (toward White's promotion edge).
const UP_LEFT: usize = 0;
const UP_RIGHT: usize = 1;
const DOWN_LEFT: usize = 2;
const DOWN_RIGHT: usize = 3;
const WHITE_FWD: [usize; 2] = [UP_LEFT, UP_RIGHT];
const BLACK_FWD: [usize; 2] = [DOWN_LEFT, DOWN_RIGHT];

/// (row, col) on the 10×10 grid for dark-square index `i` (0..49).
/// Row 0 is the top; even rows have dark squares on odd columns and vice-versa,
/// matching PDN numbering (square 1 = top row, leftmost dark square).
const fn idx_to_rc(i: usize) -> (i32, i32) {
    let r = (i / 5) as i32;
    let within = (i % 5) as i32;
    let c = if r % 2 == 0 { within * 2 + 1 } else { within * 2 };
    (r, c)
}

/// Inverse of [`idx_to_rc`] for an in-bounds dark square.
const fn rc_to_idx(r: i32, c: i32) -> i32 {
    let within = if r % 2 == 0 { (c - 1) / 2 } else { c / 2 };
    r * 5 + within
}

/// `NEIGHBOR[i][dir]` = the dark-square index one diagonal step from `i` in
/// `dir`, or `-1` if that step leaves the board. Walking a full diagonal ray is
/// just repeated application in the same `dir` (directions stay consistent).
const fn build_neighbors() -> [[i8; 4]; N] {
    let mut t = [[-1i8; 4]; N];
    let dirs = [(-1i32, -1i32), (-1, 1), (1, -1), (1, 1)];
    let mut i = 0;
    while i < N {
        let (r, c) = idx_to_rc(i);
        let mut d = 0;
        while d < 4 {
            let nr = r + dirs[d].0;
            let nc = c + dirs[d].1;
            if nr >= 0 && nr < 10 && nc >= 0 && nc < 10 {
                t[i][d] = rc_to_idx(nr, nc) as i8;
            }
            d += 1;
        }
        i += 1;
    }
    t
}

static NEIGHBOR: [[i8; 4]; N] = build_neighbors();

#[inline]
fn nb(sq: i32, dir: usize) -> i32 {
    NEIGHBOR[sq as usize][dir] as i32
}

/// Iterate set-bit indices of a 50-bit board, low to high.
#[inline]
fn bits(mut b: u64) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if b == 0 {
            None
        } else {
            let i = b.trailing_zeros() as u8;
            b &= b - 1;
            Some(i)
        }
    })
}

// ── Color ───────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Color {
    White,
    Black,
}

impl Color {
    #[inline]
    pub fn other(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[inline]
pub fn color_to_player(c: Color) -> Player {
    match c {
        Color::White => Player::P0,
        Color::Black => Player::P1,
    }
}

#[inline]
pub fn player_to_color(p: Player) -> Color {
    match p {
        Player::P0 => Color::White,
        Player::P1 => Color::Black,
    }
}

// ── State ───────────────────────────────────────────────────────────────────

/// Bit-packed board. `white`/`black` mark piece occupancy; `kings` is the subset
/// of *either* colour that has been crowned (a man of colour `c` is
/// `color_bb & !kings`). `idle` counts plies since the last capture or man move.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct State {
    pub white: u64,
    pub black: u64,
    pub kings: u64,
    pub stm: Color,
    pub idle: u16,
}

impl State {
    #[inline]
    pub fn initial() -> State {
        State {
            white: WHITE_INIT,
            black: BLACK_INIT,
            kings: 0,
            stm: Color::White,
            idle: 0,
        }
    }

    #[inline]
    pub fn occ(&self) -> u64 {
        self.white | self.black
    }

    #[inline]
    fn own(&self) -> u64 {
        match self.stm {
            Color::White => self.white,
            Color::Black => self.black,
        }
    }

    #[inline]
    fn enemy(&self) -> u64 {
        match self.stm {
            Color::White => self.black,
            Color::Black => self.white,
        }
    }

    /// Apply a move assumed legal: relocate the piece, remove captured pieces,
    /// crown on promotion-by-stopping, flip the side to move, update `idle`.
    pub fn apply(&self, mv: Move) -> State {
        let fbit = 1u64 << mv.from;
        let tbit = 1u64 << mv.to;
        let mut white = self.white;
        let mut black = self.black;
        let mut kings = self.kings;

        let was_king = kings & fbit != 0;
        let is_white = white & fbit != 0;

        if is_white {
            white = (white & !fbit) | tbit;
        } else {
            black = (black & !fbit) | tbit;
        }
        if was_king {
            kings = (kings & !fbit) | tbit;
        }

        // Remove the whole captured set in one shot (they blocked until now).
        if mv.captured != 0 {
            white &= !mv.captured;
            black &= !mv.captured;
            kings &= !mv.captured;
        }

        // Promotion: only a man that *stops* on its far rank is crowned.
        if !was_king {
            let promo = if is_white { WHITE_PROMO } else { BLACK_PROMO };
            if tbit & promo != 0 {
                kings |= tbit;
            }
        }

        // A capture or any man move resets the draw counter; a quiet king move
        // (the only remaining case) advances it.
        let idle = if mv.captured != 0 || !was_king {
            0
        } else {
            self.idle + 1
        };

        State {
            white,
            black,
            kings,
            stm: self.stm.other(),
            idle,
        }
    }

    /// Flip the side to move without touching the board (null-move pruning only).
    #[inline]
    pub fn null_move(&self) -> State {
        State {
            stm: self.stm.other(),
            ..*self
        }
    }
}

// ── Move ────────────────────────────────────────────────────────────────────

/// A single ply. A multi-jump is ONE move: `captured` is the union of every
/// jumped square, `to` the final landing. A quiet move has `captured == 0`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub captured: u64,
}

// ── Move generation ─────────────────────────────────────────────────────────

/// Recursively extend a *man* capture from `cur` (the piece's live square),
/// pushing a completed [`Move`] when no further jump is possible. `blockers` is
/// the original occupancy minus the origin square; captured pieces stay in it
/// (they block landings until the move resolves) and are tracked in `captured`
/// (so they cannot be jumped twice).
fn extend_man(
    from: u8,
    cur: u8,
    enemy: u64,
    blockers: u64,
    captured: u64,
    out: &mut Vec<Move>,
) {
    let mut extended = false;
    for dir in 0..4 {
        let over = nb(cur as i32, dir);
        if over < 0 {
            continue;
        }
        let overbit = 1u64 << over;
        // Must jump an enemy piece that has not already been captured.
        if enemy & overbit == 0 || captured & overbit != 0 {
            continue;
        }
        let land = nb(over, dir);
        if land < 0 {
            continue;
        }
        let landbit = 1u64 << land;
        // Landing square must be empty. `from` was removed from `blockers`, so
        // re-landing on the origin is allowed; captured pieces remain blockers.
        if blockers & landbit != 0 {
            continue;
        }
        extended = true;
        extend_man(from, land as u8, enemy, blockers, captured | overbit, out);
    }
    if !extended && captured != 0 {
        out.push(Move {
            from,
            to: cur,
            captured,
        });
    }
}

/// Recursively extend a *flying king* capture. Along each diagonal it scans past
/// empties to the first occupied square; if that is an un-captured enemy with at
/// least one empty square beyond, every such landing square branches a capture.
fn extend_king(
    from: u8,
    cur: u8,
    enemy: u64,
    blockers: u64,
    captured: u64,
    out: &mut Vec<Move>,
) {
    let mut extended = false;
    for dir in 0..4 {
        // Scan to the first occupied square along the ray.
        let mut sq = nb(cur as i32, dir);
        while sq >= 0 && blockers & (1u64 << sq) == 0 {
            sq = nb(sq, dir);
        }
        if sq < 0 {
            continue; // ran off the board with no piece to jump
        }
        let victimbit = 1u64 << sq;
        // Own piece, or an already-captured blocker → cannot jump here.
        if enemy & victimbit == 0 || captured & victimbit != 0 {
            continue;
        }
        // Every empty square beyond the victim is a legal landing.
        let mut land = nb(sq, dir);
        while land >= 0 && blockers & (1u64 << land) == 0 {
            extended = true;
            extend_king(from, land as u8, enemy, blockers, captured | victimbit, out);
            land = nb(land, dir);
        }
    }
    if !extended && captured != 0 {
        out.push(Move {
            from,
            to: cur,
            captured,
        });
    }
}

/// All legal *capture* moves: every maximum-length sequence, deduplicated by
/// `(from, to, captured)`. Empty when no capture exists. International rules use
/// the maximum **count** with no further tiebreak.
pub fn captures(s: &State) -> Vec<Move> {
    let own = s.own();
    let enemy = s.enemy();
    let occ = s.occ();
    let mut raw: Vec<Move> = Vec::new();
    for sq in bits(own) {
        let blockers = occ & !(1u64 << sq);
        if s.kings & (1u64 << sq) != 0 {
            extend_king(sq, sq, enemy, blockers, 0, &mut raw);
        } else {
            extend_man(sq, sq, enemy, blockers, 0, &mut raw);
        }
    }
    if raw.is_empty() {
        return raw;
    }
    let maxn = raw.iter().map(|m| m.captured.count_ones()).max().unwrap();
    raw.retain(|m| m.captured.count_ones() == maxn);
    // Dedup by (from, to, captured): distinct jump *orders* that take the same
    // set via the same landing collapse to one legal move. Sort for a stable,
    // deterministic ordering, then drop adjacent duplicates.
    raw.sort_unstable_by_key(|m| (m.from, m.to, m.captured));
    raw.dedup();
    raw
}

/// All legal *quiet* (non-capturing) moves: man single steps forward, flying
/// king slides any distance. Used only when [`captures`] is empty.
pub fn quiets(s: &State) -> Vec<Move> {
    let own = s.own();
    let occ = s.occ();
    let mut out = Vec::new();
    let fwd = match s.stm {
        Color::White => WHITE_FWD,
        Color::Black => BLACK_FWD,
    };
    for sq in bits(own) {
        if s.kings & (1u64 << sq) != 0 {
            for dir in 0..4 {
                let mut t = nb(sq as i32, dir);
                while t >= 0 && occ & (1u64 << t) == 0 {
                    out.push(Move {
                        from: sq,
                        to: t as u8,
                        captured: 0,
                    });
                    t = nb(t, dir);
                }
            }
        } else {
            for dir in fwd {
                let t = nb(sq as i32, dir);
                if t >= 0 && occ & (1u64 << t) == 0 {
                    out.push(Move {
                        from: sq,
                        to: t as u8,
                        captured: 0,
                    });
                }
            }
        }
    }
    out
}

/// Captures if any exist (mandatory), otherwise quiet moves.
pub fn legal_moves(s: &State) -> Vec<Move> {
    let caps = captures(s);
    if caps.is_empty() {
        quiets(s)
    } else {
        caps
    }
}

/// Cheap "does at least one capture exist?" — a single-jump probe (if a capture
/// sequence exists, so does its first jump). Used by terminal detection.
fn has_capture(s: &State) -> bool {
    let own = s.own();
    let enemy = s.enemy();
    let occ = s.occ();
    for sq in bits(own) {
        let blockers = occ & !(1u64 << sq);
        if s.kings & (1u64 << sq) != 0 {
            for dir in 0..4 {
                let mut t = nb(sq as i32, dir);
                while t >= 0 && blockers & (1u64 << t) == 0 {
                    t = nb(t, dir);
                }
                if t >= 0 && enemy & (1u64 << t) != 0 {
                    let land = nb(t, dir);
                    if land >= 0 && blockers & (1u64 << land) == 0 {
                        return true;
                    }
                }
            }
        } else {
            for dir in 0..4 {
                let over = nb(sq as i32, dir);
                if over < 0 || enemy & (1u64 << over) == 0 {
                    continue;
                }
                let land = nb(over, dir);
                if land >= 0 && occ & (1u64 << land) == 0 {
                    return true;
                }
            }
        }
    }
    false
}

/// Cheap "does at least one quiet move exist?".
fn has_quiet(s: &State) -> bool {
    let own = s.own();
    let occ = s.occ();
    let fwd = match s.stm {
        Color::White => WHITE_FWD,
        Color::Black => BLACK_FWD,
    };
    for sq in bits(own) {
        if s.kings & (1u64 << sq) != 0 {
            for dir in 0..4 {
                let t = nb(sq as i32, dir);
                if t >= 0 && occ & (1u64 << t) == 0 {
                    return true;
                }
            }
        } else {
            for dir in fwd {
                let t = nb(sq as i32, dir);
                if t >= 0 && occ & (1u64 << t) == 0 {
                    return true;
                }
            }
        }
    }
    false
}

/// At least one legal move (capture or quiet) exists for the side to move.
pub fn any_legal_move(s: &State) -> bool {
    has_capture(s) || has_quiet(s)
}

/// Move-tree node count to `depth` from `s` (fidelity oracle; no draw/terminal
/// pruning — a pure enumeration of legal sequences).
pub fn perft(s: &State, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = legal_moves(s);
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut total = 0;
    for mv in moves {
        total += perft(&s.apply(mv), depth - 1);
    }
    total
}

// ── Game impl ───────────────────────────────────────────────────────────────

/// Zero-size marker type implementing [`Game`] for International draughts.
#[derive(Clone, Copy, Debug)]
pub struct Checkers;

impl Game for Checkers {
    type State = State;
    type Move = Move;

    const ID: &'static str = "checkers";
    // Provisional NN dims (revisited when training starts): from×to = 50×50.
    const ACTION_COUNT: usize = N * N; // 2500
    const FEATURE_LEN: usize = 6 * N + 8; // 308: 6 planes × 50 dark squares + scalars
    const MOVE_INDEX_SPACE: usize = N * N; // 2500 (from*50 + to)

    #[inline]
    fn initial() -> State {
        State::initial()
    }

    #[inline]
    fn turn(s: &State) -> Player {
        color_to_player(s.stm)
    }

    #[inline]
    fn winner(s: &State) -> Option<Player> {
        if s.idle >= DRAW_PLIES {
            return None; // draw
        }
        if any_legal_move(s) {
            return None; // ongoing
        }
        // Side to move has no move → it loses.
        Some(color_to_player(s.stm.other()))
    }

    #[inline]
    fn is_terminal(s: &State) -> bool {
        s.idle >= DRAW_PLIES || !any_legal_move(s)
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
        legal_moves(s).contains(&mv)
    }

    #[inline]
    fn legal_moves(s: &State) -> Vec<Move> {
        legal_moves(s)
    }

    // No interior-node move pruning yet (alpha-beta does the heavy lifting); the
    // search sees the full legal set at every node, root or interior.
    #[inline]
    fn search_moves(s: &State) -> Vec<Move> {
        legal_moves(s)
    }

    #[inline]
    fn search_moves_wide(s: &State) -> Vec<Move> {
        legal_moves(s)
    }

    /// Mandatory captures, for quiescence: empty when the position is quiet so
    /// qsearch returns the static eval immediately.
    #[inline]
    fn capture_moves(s: &State) -> Vec<Move> {
        captures(s)
    }

    /// A quiet move is reducible (LMR/LMP); captures must never be reduced.
    #[inline]
    fn is_quiet(mv: Move) -> bool {
        mv.captured == 0
    }

    #[inline]
    fn hash(s: &State) -> u64 {
        let mix = |mut x: u64| {
            x ^= x >> 33;
            x = x.wrapping_mul(0xff51afd7ed558ccd);
            x ^= x >> 33;
            x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
            x ^= x >> 33;
            x
        };
        let mut h = mix(s.white)
            ^ mix(s.black).rotate_left(21)
            ^ mix(s.kings).rotate_left(42)
            ^ ((s.stm as u64).wrapping_mul(0x9e3779b97f4a7c15));
        h ^= (s.idle as u64).wrapping_mul(0x2545f4914f6cdd1d);
        if h == 0 {
            1
        } else {
            h
        }
    }

    /// `from*50 + to` (0..2499). Distinct capture *targets* from the same square
    /// share a slot — acceptable for move-ordering history.
    #[inline]
    fn move_order_index(mv: Move) -> usize {
        mv.from as usize * N + mv.to as usize
    }

    /// One-ply forced win detection is left to the search (returns `None`).
    #[inline]
    fn immediate_winning_move(_s: &State) -> Option<Move> {
        None
    }

    fn state_key(s: &State) -> String {
        let stm = match s.stm {
            Color::White => 'w',
            Color::Black => 'b',
        };
        format!(
            "{:x}.{:x}.{:x}.{}.{}",
            s.white, s.black, s.kings, stm, s.idle
        )
    }

    fn parse_state_key(k: &str) -> Option<State> {
        let mut it = k.split('.');
        let white = u64::from_str_radix(it.next()?, 16).ok()?;
        let black = u64::from_str_radix(it.next()?, 16).ok()?;
        let kings = u64::from_str_radix(it.next()?, 16).ok()?;
        let stm = match it.next()? {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return None,
        };
        let idle = it.next()?.parse().ok()?;
        if it.next().is_some() {
            return None;
        }
        Some(State {
            white,
            black,
            kings,
            stm,
            idle,
        })
    }
}

// ── Evaluation ──────────────────────────────────────────────────────────────

/// Hand evaluator for draughts. Antisymmetric (`eval(s,P0) == -eval(s,P1)` on
/// non-terminal states), terminal positions score `±WIN_SCORE`. Weights tunable
/// via `GB_CK_*` env vars (parity with Wall Chess's `WC_*` knobs).
#[derive(Clone, Copy, Debug)]
pub struct CheckersHeuristic {
    pub man: i32,
    pub king: i32,
    pub advance: i32,
    pub back_rank: i32,
}

impl Default for CheckersHeuristic {
    fn default() -> Self {
        CheckersHeuristic {
            man: 100,
            king: 300,
            advance: 4,
            back_rank: 6,
        }
    }
}

impl CheckersHeuristic {
    pub fn new(man: i32, king: i32, advance: i32, back_rank: i32) -> Self {
        CheckersHeuristic {
            man,
            king,
            advance,
            back_rank,
        }
    }

    /// Read weight overrides from `GB_CK_MAN` / `GB_CK_KING` / `GB_CK_ADV` /
    /// `GB_CK_BACK`; unset knobs keep their defaults.
    pub fn from_env() -> Self {
        let mut h = Self::default();
        let get = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<i32>().ok());
        if let Some(v) = get("GB_CK_MAN") {
            h.man = v;
        }
        if let Some(v) = get("GB_CK_KING") {
            h.king = v;
        }
        if let Some(v) = get("GB_CK_ADV") {
            h.advance = v;
        }
        if let Some(v) = get("GB_CK_BACK") {
            h.back_rank = v;
        }
        h
    }

    /// Static score from White's point of view (positive = good for White).
    fn white_score(&self, s: &State) -> i32 {
        let wk = s.white & s.kings;
        let bk = s.black & s.kings;
        let wm = s.white & !s.kings;
        let bm = s.black & !s.kings;

        let mut score = 0;
        score += self.man * wm.count_ones() as i32;
        score -= self.man * bm.count_ones() as i32;
        score += self.king * wk.count_ones() as i32;
        score -= self.king * bk.count_ones() as i32;

        // Advancement: men closer to their promotion rank are worth more.
        for sq in bits(wm) {
            let r = sq as i32 / 5; // White advances toward row 0
            score += self.advance * (9 - r);
        }
        for sq in bits(bm) {
            let r = sq as i32 / 5; // Black advances toward row 9
            score -= self.advance * r;
        }

        // Back-rank guard: men held on the home rank deny opponent promotions.
        score += self.back_rank * (wm & BLACK_PROMO).count_ones() as i32; // White home = row 9
        score -= self.back_rank * (bm & WHITE_PROMO).count_ones() as i32; // Black home = row 0

        score
    }
}

impl Evaluator for CheckersHeuristic {
    type G = Checkers;

    fn eval(&self, state: &State, p: Player) -> i32 {
        // Terminal: a draw is 0; otherwise the side to move is the loser.
        if state.idle >= DRAW_PLIES {
            return 0;
        }
        if !any_legal_move(state) {
            // `state.stm` to move has no move → lost. Report from `p`'s view.
            return if p == color_to_player(state.stm) {
                -WIN_SCORE
            } else {
                WIN_SCORE
            };
        }
        let white = self.white_score(state);
        match p {
            Player::P0 => white,
            Player::P1 => -white,
        }
    }
}

// ── NN encoding (provisional; revisited when training starts) ────────────────

impl Encoder for Checkers {
    type G = Checkers;

    /// Me-frame feature vector, length [`Game::FEATURE_LEN`]. Six 50-wide planes
    /// (my men / my kings / opp men / opp kings / all men / all kings) plus a
    /// short scalar tail (side bias, normalized idle counter, piece counts).
    fn encode(state: &State) -> Vec<f32> {
        let (me, opp) = match state.stm {
            Color::White => (state.white, state.black),
            Color::Black => (state.black, state.white),
        };
        // Me-frame: rotate the board 180° (index i → 49-i) so the side to move
        // always views from the same orientation.
        let m = |bb: u64| -> u64 {
            let mut out = 0u64;
            for sq in bits(bb) {
                out |= 1u64 << (49 - sq);
            }
            out
        };
        let (me, opp, kings) = (m(me), m(opp), m(state.kings));
        let me_men = me & !kings;
        let me_kings = me & kings;
        let opp_men = opp & !kings;
        let opp_kings = opp & kings;

        let mut f = Vec::with_capacity(Checkers::FEATURE_LEN);
        for plane in [
            me_men,
            me_kings,
            opp_men,
            opp_kings,
            me_men | opp_men,
            me_kings | opp_kings,
        ] {
            for i in 0..N {
                f.push(((plane >> i) & 1) as f32);
            }
        }
        f.push(1.0); // side-to-move bias
        f.push(state.idle as f32 / DRAW_PLIES as f32);
        f.push(me_men.count_ones() as f32 / 20.0);
        f.push(me_kings.count_ones() as f32 / 20.0);
        f.push(opp_men.count_ones() as f32 / 20.0);
        f.push(opp_kings.count_ones() as f32 / 20.0);
        f.push(0.0);
        f.push(0.0);
        f
    }

    #[inline]
    fn action_index(mv: Move) -> usize {
        mv.from as usize * N + mv.to as usize
    }

    /// Inverse of [`Encoder::action_index`]; `captured` is unknown from the index
    /// alone, so callers must intersect with the legal set to recover it.
    #[inline]
    fn index_to_move(i: usize) -> Move {
        Move {
            from: (i / N) as u8,
            to: (i % N) as u8,
            captured: 0,
        }
    }

    /// 180° board rotation maps the absolute frame to the me-frame and back
    /// (involution): square i ↔ 49-i, for both endpoints of the move.
    #[inline]
    fn mirror_move(_p: Player, mv: Move) -> Move {
        Move {
            from: 49 - mv.from,
            to: 49 - mv.to,
            captured: {
                let mut out = 0u64;
                for sq in bits(mv.captured) {
                    out |= 1u64 << (49 - sq);
                }
                out
            },
        }
    }
}

#[cfg(test)]
mod tests;
