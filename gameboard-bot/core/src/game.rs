//! The game-agnostic seam.
//!
//! `gameboard-core` is a multi-game engine: one shared search/eval/match/NN
//! pipeline that any two-player, perfect-information, zero-sum board game plugs
//! into by implementing [`Game`]. Wall Chess (Quoridor) lives in
//! [`crate::games::wallchess`]; International Draughts in
//! [`crate::games::checkers`].
//!
//! The boundary runs exactly at the concrete `State`/`Move` types: everything
//! above them (negamax/alpha-beta, the transposition table, killers/history,
//! LMR/NMP/PVS, the arena match driver) is generic over [`Game`]; everything
//! that names a wall, a pawn jump, a flying king, a goal row, or a board
//! dimension is per-game logic behind this trait.
//!
//! Dispatch is **monomorphized**, not dynamic: `Search<E: Evaluator>` resolves
//! its game through `E::G`, so each concrete game compiles to its own fully
//! optimized engine with no boxing and no `Move`-erasure (the `Copy` value-type
//! `Move` and const-sized tables the deployed bot relies on are preserved). Game
//! selection happens once at a binary boundary (`match game_id { .. }`).

/// The two players, in turn order. The generic, game-neutral replacement for
/// any per-game `Side`/`Color`: P0 is the side that moves first from the initial
/// position. Each game maps its own colour onto this (Wall Chess: South=P0,
/// North=P1; Draughts: White=P0, Black=P1).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Player {
    P0 = 0,
    P1 = 1,
}

impl Player {
    #[inline]
    pub fn other(self) -> Player {
        match self {
            Player::P0 => Player::P1,
            Player::P1 => Player::P0,
        }
    }

    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }
}

// ── Shared score scale (negamax, side-to-move POV) ──────────────────────────

/// Large finite magnitude for terminal states (kept below i32 saturation so
/// alpha-beta windows never overflow). Shared across every game so the search's
/// `score.abs() >= WIN_SCORE` mate guards mean the same thing everywhere.
pub const WIN_SCORE: i32 = 1_000_000;

/// Decisive magnitude for a *provably resolved* but not-yet-terminal position
/// (e.g. a won race / won material endgame). STRICTLY below [`WIN_SCORE`] so a
/// real terminal win is always preferred and the search's mate-zone guards are
/// not tripped by a resolved-but-non-terminal node.
pub const ENDGAME_WIN: i32 = WIN_SCORE - 1000; // 999_000

// ── The game contract ───────────────────────────────────────────────────────

/// A two-player, perfect-information, zero-sum board game.
///
/// Implementors are zero-size marker types (e.g. `struct WallChess;`) that tie
/// together a `State`, a `Move`, and the rules/encoding functions. All methods
/// are associated (no `&self`) — the game has no instance state, only its rules.
pub trait Game: Sized + 'static {
    /// Bit-packed, cheap-to-copy board state.
    type State: Clone + Copy + PartialEq + Eq + std::hash::Hash + std::fmt::Debug;
    /// The action type. `Copy` so the search keeps it in const-sized tables.
    type Move: Clone + Copy + PartialEq + Eq + std::hash::Hash + std::fmt::Debug;

    // --- identity / dims ---
    /// Stable game id used at CLI/wasm dispatch boundaries and in data files.
    const ID: &'static str;
    /// Dense policy-head width (NN action space). Provisional games may revise.
    const ACTION_COUNT: usize;
    /// NN feature-vector length produced by the [`Encoder`].
    const FEATURE_LEN: usize;
    /// Size of the search's history / move-ordering table — must be `> ` every
    /// value [`Game::move_order_index`] can return.
    const MOVE_INDEX_SPACE: usize;

    // --- rules ---
    fn initial() -> Self::State;
    fn turn(s: &Self::State) -> Player;
    /// The winner if the game is decided, else `None` (ongoing OR drawn).
    fn winner(s: &Self::State) -> Option<Player>;
    /// True if the position is terminal (won, lost, or drawn) — the search stops
    /// here. Kept separate from [`Game::winner`] so games with O(1) terminal
    /// markers (Wall Chess: a stored winner) stay cheap, while games whose
    /// terminality is "side to move has no move" can answer with an early-exit
    /// probe instead of full move generation.
    fn is_terminal(s: &Self::State) -> bool;
    /// Apply a move assumed legal (flip turn, update board). Callers validate
    /// with [`Game::is_legal`] / membership in [`Game::legal_moves`] first.
    fn apply(s: &Self::State, mv: Self::Move) -> Self::State;
    /// Flip the side to move without changing the board (null-move pruning only).
    fn null_move(s: &Self::State) -> Self::State;
    fn is_legal(s: &Self::State, mv: Self::Move) -> bool;

    // --- move generation ---
    /// All legal moves for the side to move.
    fn legal_moves(s: &Self::State) -> Vec<Self::Move>;
    /// Pruned candidate set for interior search nodes (e.g. drop walls that cut
    /// no shortest path). May equal [`Game::legal_moves`].
    fn search_moves(s: &Self::State) -> Vec<Self::Move>;
    /// Wider candidate set used at the root only.
    fn search_moves_wide(s: &Self::State) -> Vec<Self::Move>;
    /// Capture / tactical moves only, for quiescence search. Default: none
    /// (games without a capture notion never enter quiescence).
    fn capture_moves(_s: &Self::State) -> Vec<Self::Move> {
        Vec::new()
    }
    /// Is `mv` a quiet move (eligible for LMR / LMP / futility reduction)?
    /// Tactical moves (Wall Chess pawn advances, draughts captures) return
    /// `false` so the search never reduces or skips them.
    fn is_quiet(mv: Self::Move) -> bool;

    // --- search hooks ---
    /// 64-bit state hash for the transposition table. Never returns 0 (the
    /// empty-slot sentinel).
    fn hash(s: &Self::State) -> u64;
    /// Compact, collision-free index into the history/killer tables.
    /// Must be `< ` [`Game::MOVE_INDEX_SPACE`].
    fn move_order_index(mv: Self::Move) -> usize;
    /// A one-ply forced win for the side to move, if any (lets the search
    /// short-circuit at the root without deepening).
    fn immediate_winning_move(s: &Self::State) -> Option<Self::Move>;

    // --- serialization (graph dedup / UI parity / books) ---
    fn state_key(s: &Self::State) -> String;
    fn parse_state_key(k: &str) -> Option<Self::State>;
}

/// Leaf evaluator: scores a position from `p`'s point of view in the negamax
/// convention (higher is better for `p`), antisymmetric (`eval(s, P0) ==
/// -eval(s, P1)` on non-terminal states). Terminal positions score `±WIN_SCORE`.
///
/// Carries its game as an associated type so `Search<E: Evaluator>` keeps a
/// single type parameter — construction (`Search::with_config(&heuristic, cfg)`)
/// infers the game from the evaluator with no turbofish at the call site.
pub trait Evaluator {
    type G: Game;
    fn eval(&self, state: &<Self::G as Game>::State, p: Player) -> i32;
}

// NOTE: the AlphaZero-style policy/value MCTS seam currently lives in
// [`crate::mcts`] (`PolicyValue`), bound to Wall Chess. Genericizing it over
// `Game` is deferred to when checkers NN training begins (see the multi-game
// architecture plan). The [`Encoder`] contract below is already game-generic.

/// NN feature-encoding contract for a game: the me-frame tensor layout that must
/// stay byte-identical to the Python trainer's encoder.
pub trait Encoder {
    type G: Game;
    /// Flat feature vector of length [`Game::FEATURE_LEN`], from the side-to-move
    /// ("me") frame so the net sees a consistent view regardless of who moves.
    fn encode(state: &<Self::G as Game>::State) -> Vec<f32>;
    /// Dense action index for a move (`< ` [`Game::ACTION_COUNT`]).
    fn action_index(mv: <Self::G as Game>::Move) -> usize;
    /// Inverse of [`Encoder::action_index`] (no legality check; intersect with
    /// the legal set).
    fn index_to_move(i: usize) -> <Self::G as Game>::Move;
    /// Map an absolute-frame move into the me-frame of `p` (involution).
    fn mirror_move(p: Player, mv: <Self::G as Game>::Move) -> <Self::G as Game>::Move;
}
