//! Negamax + alpha-beta with iterative deepening, a transposition table, and a
//! configurable bundle of pruning/reduction heuristics. The transposition table
//! is a fixed-size array indexed by a cheap multiplicative hash — much faster
//! than HashMap for the tight inner loop (no pointer-chasing, no SipHash).
//!
//! Every aggressive heuristic is gated behind a [`SearchConfig`] flag so
//! strength/speed tradeoffs can be A/B-tested without recompiling. The native
//! `bestmove` / `search_bench` binaries build their config with
//! [`SearchConfig::from_env`], so a match can be driven by `WC_*` env vars
//! (e.g. `WC_PRESET=aggressive`, `WC_NULLMOVE=1 WC_NMP_R=3`).
//!
//! `SearchConfig::default()` reproduces the previously-deployed behavior
//! exactly — LMR + TT on, every newer pruning off — so `Search::new` and the
//! wasm `analyze` path are unchanged until flags are flipped.
//!
//! Heuristics implemented (all flagged):
//!   - PVS         : null-window scout searches, full re-search only on the PV.
//!   - null-move   : skip a turn at reduced depth; cut if it still beats beta.
//!   - RFP         : reverse futility / static null move — prune when the static
//!                   eval already clears beta by a depth-scaled margin.
//!   - razoring    : drop nodes whose static eval sits far below alpha.
//!   - futility    : skip frontier wall moves that cannot lift alpha.
//!   - LMP         : after enough quiet (wall) moves at low depth, skip the rest.
//!   - LMR         : late quiet moves searched shallow first, re-searched if they
//!                   beat alpha.
//!   - aspiration  : root iterative deepening searches a narrow window first.
//!
//! Move ordering (priority, highest first): TT move → killer[0] → killer[1] →
//! history heuristic.

use crate::game::{Evaluator, Game, WIN_SCORE};

// Projections through the evaluator's game, so `Search` keeps a single type
// parameter (`E`) while operating on that game's concrete `State`/`Move`.
type Gm<E> = <E as Evaluator>::G;
type St<E> = <Gm<E> as Game>::State;
type Mv<E> = <Gm<E> as Game>::Move;

// ── constants ─────────────────────────────────────────────────────────────────

const MAX_PLY: usize = 64; // search tree depth ceiling

// TT: 2^18 = 262 144 slots × ~24 bytes ≈ 6 MB per Search instance.
const TT_BITS: usize = 18;
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: usize = TT_SIZE - 1;

// ── Search configuration ────────────────────────────────────────────────────

/// Toggles + margins for every search heuristic. `Copy` so a single config is
/// cheaply cloned into a fresh [`Search`] per query. `Default` == the
/// previously-deployed engine (LMR + TT only).
#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    /// Transposition table probe/store (exact — affects speed, not strength).
    pub tt: bool,

    /// Principal Variation Search: scout non-PV moves with a null window.
    pub pvs: bool,

    /// Aspiration windows at the root iterative-deepening loop.
    pub aspiration: bool,
    pub asp_min_depth: u8,
    pub asp_window: i32,

    /// Null-move pruning.
    pub null_move: bool,
    pub nmp_min_depth: u8,
    pub nmp_reduction: u8,

    /// Reverse futility / static null move pruning.
    pub rfp: bool,
    pub rfp_max_depth: u8,
    pub rfp_margin: i32,

    /// Razoring: trust the static eval when it sits far below alpha.
    pub razoring: bool,
    pub razor_max_depth: u8,
    pub razor_margin: i32,

    /// Futility pruning of frontier quiet (wall) moves.
    pub futility: bool,
    pub futility_max_depth: u8,
    pub futility_margin: i32,

    /// Late move pruning (move-count based, quiet moves only).
    pub lmp: bool,
    pub lmp_max_depth: u8,
    pub lmp_base: u32,
    pub lmp_factor: u32,

    /// Late move reductions.
    pub lmr: bool,
    pub lmr_min_depth: u8,
    pub lmr_full_moves: usize,
    pub lmr_reduction: u8,

    /// Quiescence search at the leaf: extend through forced tactical exchanges
    /// (captures) so the static eval is never taken mid-trade. Off by default —
    /// Wall Chess has no captures, so this is inert there and the deployed engine
    /// is unchanged. Draughts turns it on (see [`SearchConfig::draughts`]).
    pub quiescence: bool,
}

impl Default for SearchConfig {
    /// Reproduces the previously-deployed search exactly: LMR + TT on, every
    /// newer pruning off. Margins are the defaults the new prunings use *when*
    /// their flag is flipped on (they have no effect while disabled).
    fn default() -> Self {
        SearchConfig {
            tt: true,
            pvs: false,
            aspiration: false,
            asp_min_depth: 4,
            asp_window: 75,
            null_move: false,
            nmp_min_depth: 3,
            nmp_reduction: 2,
            rfp: false,
            rfp_max_depth: 4,
            rfp_margin: 120,
            razoring: false,
            razor_max_depth: 2,
            razor_margin: 300,
            futility: false,
            futility_max_depth: 2,
            futility_margin: 150,
            lmp: false,
            lmp_max_depth: 3,
            lmp_base: 4,
            lmp_factor: 2,
            lmr: true,
            lmr_min_depth: 4,
            lmr_full_moves: 4,
            lmr_reduction: 1,
            quiescence: false,
        }
    }
}

impl SearchConfig {
    /// Every heuristic on, default margins. The "throw everything at it" preset.
    pub fn aggressive() -> Self {
        SearchConfig {
            pvs: true,
            aspiration: true,
            null_move: true,
            rfp: true,
            razoring: true,
            futility: true,
            lmp: true,
            lmr: true,
            ..SearchConfig::default()
        }
    }

    /// Conservative Stockfish-style bundle: the heuristics that are usually safe
    /// (PVS, null-move, RFP, LMR, aspiration) without the lossy quiet-move
    /// skippers (razoring, futility, LMP).
    pub fn safe() -> Self {
        SearchConfig {
            pvs: true,
            aspiration: true,
            null_move: true,
            rfp: true,
            lmr: true,
            ..SearchConfig::default()
        }
    }

    /// Pure alpha-beta + TT — no reductions or prunings. Baseline for A/B speed
    /// and exactness comparisons.
    pub fn plain() -> Self {
        SearchConfig {
            lmr: false,
            ..SearchConfig::default()
        }
    }

    /// Tuned defaults for International Draughts (forced-capture games).
    /// Quiescence ON (capture chains are the dominant horizon-effect source) and
    /// **null-move OFF** — with mandatory captures a "pass" is illegal and the
    /// null-move soundness assumption (a free move can only help) breaks in
    /// zugzwang-prone positions. PVS + LMR + aspiration are kept (LMR never
    /// reduces captures because they are not quiet — see [`Game::is_quiet`]).
    pub fn draughts() -> Self {
        SearchConfig {
            pvs: true,
            aspiration: true,
            null_move: false,
            lmr: true,
            quiescence: true,
            ..SearchConfig::default()
        }
    }

    /// Build a config from `WC_*` environment variables.
    ///
    /// `WC_PRESET` (`aggressive` | `safe` | `plain` | `default`) picks a base;
    /// any individual `WC_*` var then overrides a single knob. Flags accept
    /// `1/0/true/false/on/off/yes/no`; numbers parse directly.
    pub fn from_env() -> Self {
        let mut c = match std::env::var("WC_PRESET").ok().as_deref() {
            Some("aggressive") => Self::aggressive(),
            Some("safe") | Some("lite") => Self::safe(),
            Some("plain") | Some("baseline") => Self::plain(),
            _ => Self::default(),
        };
        env_flag("WC_TT", &mut c.tt);
        env_flag("WC_PVS", &mut c.pvs);
        env_flag("WC_ASP", &mut c.aspiration);
        env_num("WC_ASP_MIN", &mut c.asp_min_depth);
        env_num("WC_ASP_WINDOW", &mut c.asp_window);
        env_flag("WC_NULLMOVE", &mut c.null_move);
        env_num("WC_NMP_MIN", &mut c.nmp_min_depth);
        env_num("WC_NMP_R", &mut c.nmp_reduction);
        env_flag("WC_RFP", &mut c.rfp);
        env_num("WC_RFP_MAXD", &mut c.rfp_max_depth);
        env_num("WC_RFP_MARGIN", &mut c.rfp_margin);
        env_flag("WC_RAZOR", &mut c.razoring);
        env_num("WC_RAZOR_MAXD", &mut c.razor_max_depth);
        env_num("WC_RAZOR_MARGIN", &mut c.razor_margin);
        env_flag("WC_FUTILITY", &mut c.futility);
        env_num("WC_FUT_MAXD", &mut c.futility_max_depth);
        env_num("WC_FUT_MARGIN", &mut c.futility_margin);
        env_flag("WC_LMP", &mut c.lmp);
        env_num("WC_LMP_MAXD", &mut c.lmp_max_depth);
        env_num("WC_LMP_BASE", &mut c.lmp_base);
        env_num("WC_LMP_FACTOR", &mut c.lmp_factor);
        env_flag("WC_LMR", &mut c.lmr);
        env_num("WC_LMR_MIN", &mut c.lmr_min_depth);
        env_num("WC_LMR_FULL", &mut c.lmr_full_moves);
        env_num("WC_LMR_R", &mut c.lmr_reduction);
        env_flag("WC_QUIESCENCE", &mut c.quiescence);
        c
    }

    /// One-line dump of the active config (for logging which knobs ran).
    pub fn summary(&self) -> String {
        format!("{self:?}")
    }
}

pub(crate) fn env_flag(key: &str, slot: &mut bool) {
    if let Ok(v) = std::env::var(key) {
        match v.trim() {
            "1" | "true" | "on" | "yes" => *slot = true,
            "0" | "false" | "off" | "no" => *slot = false,
            _ => {}
        }
    }
}

pub(crate) fn env_num<T: std::str::FromStr>(key: &str, slot: &mut T) {
    if let Ok(v) = std::env::var(key) {
        if let Ok(n) = v.trim().parse() {
            *slot = n;
        }
    }
}

// ── TT ────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Default)]
enum Bound {
    Exact,
    Lower,
    #[default]
    Upper,
}

/// One slot in the array-indexed transposition table, generic over the game's
/// move type. `depth == 0` is the empty/unoccupied sentinel.
///
/// The state hash and the history/move-ordering index are now game methods
/// ([`Game::hash`] / [`Game::move_order_index`]); for Wall Chess they reproduce
/// the previously-deployed `state_hash` / `history_idx` byte-for-byte.
#[derive(Clone, Copy)]
struct TtSlot<M: Copy> {
    key: u64,  // state hash; 0 means empty
    depth: u8, // 0 means empty
    score: i32,
    bound: Bound,
    best: Option<M>,
}

impl<M: Copy> Default for TtSlot<M> {
    fn default() -> Self {
        TtSlot {
            key: 0,
            depth: 0,
            score: 0,
            bound: Bound::default(),
            best: None,
        }
    }
}

// ── Search ────────────────────────────────────────────────────────────────────

pub struct Search<'a, E: Evaluator> {
    eval: &'a E,
    config: SearchConfig,
    tt: Vec<TtSlot<Mv<E>>>,
    killers: [[Option<Mv<E>>; 2]; MAX_PLY],
    // Heap-allocated (was a fixed stack array) because its size is the game's
    // `MOVE_INDEX_SPACE` associated const — not nameable as an array length in a
    // generic context. Values are identical, so search results are unchanged.
    history: Vec<i32>,
    node_limit: Option<u64>,
    stopped: bool,
    pub nodes: u64,
}

pub struct SearchResult<M> {
    pub best: Option<M>,
    /// Score from the side-to-move perspective at the root.
    pub score: i32,
    pub depth: u8,
    pub nodes: u64,
    pub stopped: bool,
}

impl<'a, E: Evaluator> Search<'a, E> {
    /// Construct with the default (previously-deployed) config.
    pub fn new(eval: &'a E) -> Self {
        Self::with_config(eval, SearchConfig::default())
    }

    /// Construct with an explicit heuristic config.
    pub fn with_config(eval: &'a E, config: SearchConfig) -> Self {
        Search {
            eval,
            config,
            tt: vec![TtSlot::default(); TT_SIZE],
            killers: [[None; 2]; MAX_PLY],
            history: vec![0; <Gm<E> as Game>::MOVE_INDEX_SPACE],
            node_limit: None,
            stopped: false,
            nodes: 0,
        }
    }

    pub fn config(&self) -> SearchConfig {
        self.config
    }

    /// Iterative deepening to `max_depth`. Returns the best root move and its
    /// score (side-to-move POV).
    pub fn search(&mut self, state: &St<E>, max_depth: u8) -> SearchResult<Mv<E>> {
        self.stopped = false;
        if max_depth == 0 {
            return SearchResult {
                best: None,
                score: self.eval.eval(state, <Gm<E> as Game>::turn(state)),
                depth: 0,
                nodes: self.nodes,
                stopped: false,
            };
        }
        if let Some(mv) = <Gm<E> as Game>::immediate_winning_move(state) {
            return SearchResult {
                best: Some(mv),
                score: WIN_SCORE,
                depth: 1,
                nodes: self.nodes,
                stopped: false,
            };
        }
        let mut best: Option<Mv<E>> = None;
        let mut score = 0;
        let mut reached = 0;
        for depth in 1..=max_depth {
            let (s, m) = self.search_root(state, depth, best, score);
            if self.stopped {
                break;
            }
            score = s;
            best = m.or(best);
            reached = depth;
            if score.abs() >= WIN_SCORE {
                break; // forced result found, deeper search is wasted
            }
        }
        SearchResult {
            best,
            score,
            depth: reached,
            nodes: self.nodes,
            stopped: self.stopped,
        }
    }

    /// Iterative deepening with a relative node budget. If the budget trips in
    /// the middle of a depth, the previous completed depth is returned.
    pub fn search_with_node_limit(
        &mut self,
        state: &St<E>,
        max_depth: u8,
        node_limit: u64,
    ) -> SearchResult<Mv<E>> {
        let old_limit = self.node_limit;
        self.node_limit = Some(self.nodes.saturating_add(node_limit));
        let result = self.search(state, max_depth);
        self.node_limit = old_limit;
        result
    }

    /// Root search for one iterative-deepening depth, optionally wrapped in an
    /// aspiration window centered on the previous iteration's score.
    fn search_root(
        &mut self,
        state: &St<E>,
        depth: u8,
        hint: Option<Mv<E>>,
        prev_score: i32,
    ) -> (i32, Option<Mv<E>>) {
        const FULL: i32 = WIN_SCORE * 2;
        if !self.config.aspiration || depth < self.config.asp_min_depth {
            return self.root(state, depth, hint, -FULL, FULL);
        }
        let mut window = self.config.asp_window.max(1);
        loop {
            let alpha = (prev_score - window).max(-FULL);
            let beta = (prev_score + window).min(FULL);
            let (s, m) = self.root(state, depth, hint, alpha, beta);
            if self.stopped {
                return (s, m);
            }
            if s <= alpha {
                if alpha <= -FULL {
                    return (s, m); // already widest — accept
                }
                window = window.saturating_mul(3);
            } else if s >= beta {
                if beta >= FULL {
                    return (s, m);
                }
                window = window.saturating_mul(3);
            } else {
                return (s, m);
            }
        }
    }

    fn root(
        &mut self,
        state: &St<E>,
        depth: u8,
        hint: Option<Mv<E>>,
        mut alpha: i32,
        beta: i32,
    ) -> (i32, Option<Mv<E>>) {
        let mut best_move = None;
        let mut best_score = -WIN_SCORE * 2;
        // Prefer the prior iteration's best move, then any TT move for this state.
        let hash = <Gm<E> as Game>::hash(state);
        let tt_move = hint.or_else(|| {
            if self.config.tt {
                let slot = self.tt[hash as usize & TT_MASK]; // Copy
                if slot.key == hash && slot.depth > 0 {
                    return slot.best;
                }
            }
            None
        });
        for (i, mv) in self
            .order_moves(state, tt_move, 0, true)
            .into_iter()
            .enumerate()
        {
            let child = <Gm<E> as Game>::apply(state, mv);
            let s = if i == 0 || !self.config.pvs {
                -self.negamax(&child, depth - 1, -beta, -alpha, 1, true, true)
            } else {
                // PVS scout with a null window, full re-search if it beats alpha.
                let mut s = -self.negamax(&child, depth - 1, -alpha - 1, -alpha, 1, false, true);
                if !self.stopped && s > alpha && s < beta {
                    s = -self.negamax(&child, depth - 1, -beta, -alpha, 1, true, true);
                }
                s
            };
            if self.stopped {
                break;
            }
            if s > best_score {
                best_score = s;
                best_move = Some(mv);
            }
            if s > alpha {
                alpha = s;
            }
            if alpha >= beta {
                break; // only reachable under a finite (aspiration) beta
            }
        }
        (best_score, best_move)
    }

    /// Score every legal move at full width (no alpha-beta at the root so each
    /// move gets an exact comparable score), sorted best-first from the
    /// side-to-move's perspective. Used by the graph to prune to the strong
    /// moves only. Shares one transposition table across all children.
    ///
    /// Note: children are searched as PV nodes; with pruning flags enabled their
    /// scores become approximate (the default config keeps them exact).
    pub fn ranked(&mut self, state: &St<E>, depth: u8) -> Vec<(Mv<E>, i32)> {
        let alpha = -WIN_SCORE * 2;
        let beta = WIN_SCORE * 2;
        let mut out: Vec<(Mv<E>, i32)> = <Gm<E> as Game>::legal_moves(state)
            .into_iter()
            .map(|mv| {
                let child = <Gm<E> as Game>::apply(state, mv);
                let s = -self.negamax(&child, depth.saturating_sub(1), -beta, -alpha, 1, true, true);
                (mv, s)
            })
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1));
        out
    }

    /// Negamax with alpha-beta plus the configured pruning bundle.
    ///
    /// `is_pv` marks principal-variation nodes — the lossy prunings (RFP,
    /// razoring, null-move, futility, LMP) only fire on non-PV nodes. `null_ok`
    /// guards against two null moves in a row.
    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        state: &St<E>,
        depth: u8,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        is_pv: bool,
        null_ok: bool,
    ) -> i32 {
        if self.over_budget() {
            self.stopped = true;
            return self.eval.eval(state, <Gm<E> as Game>::turn(state));
        }
        self.nodes += 1;
        let alpha_orig = alpha;
        let hash = <Gm<E> as Game>::hash(state);
        let idx = hash as usize & TT_MASK;

        let mut tt_move = None;
        if self.config.tt {
            let slot = self.tt[idx]; // Copy — no live borrow held into recursive calls
            if slot.key == hash && slot.depth > 0 {
                tt_move = slot.best;
                if slot.depth >= depth {
                    match slot.bound {
                        Bound::Exact => return slot.score,
                        Bound::Lower if slot.score >= beta => return slot.score,
                        Bound::Upper if slot.score <= alpha => return slot.score,
                        _ => {}
                    }
                }
            }
        }

        if <Gm<E> as Game>::is_terminal(state) {
            // eval is from the side *to move* at this node — negamax convention.
            return self.eval.eval(state, <Gm<E> as Game>::turn(state));
        }
        if depth == 0 {
            // Leaf: quiescence (when enabled) resolves pending forced captures so
            // the static eval is never read mid-exchange; else the static eval.
            return if self.config.quiescence {
                self.qsearch(state, alpha, beta, ply)
            } else {
                self.eval.eval(state, <Gm<E> as Game>::turn(state))
            };
        }

        // Static eval feeds the margin-based prunings; compute once, lazily.
        let need_static = !is_pv
            && (self.config.rfp
                || self.config.razoring
                || self.config.null_move
                || self.config.futility);
        let static_eval = if need_static {
            self.eval.eval(state, <Gm<E> as Game>::turn(state))
        } else {
            0
        };
        // Eval magnitudes are bounded well below WIN_SCORE, so a window bound at
        // mate magnitude means a forced result is in play — never prune there.
        let mate_zone = beta >= WIN_SCORE || alpha <= -WIN_SCORE;

        // ── node-level pruning (non-PV, away from mate scores) ──
        if !is_pv && !mate_zone {
            // Reverse futility / static null move pruning. Only `value >= beta`
            // is proven (not `>= static_eval`), so return the proven bound — a
            // larger fail-soft score would be a wrong lower bound to the parent.
            if self.config.rfp
                && depth <= self.config.rfp_max_depth
                && static_eval - self.config.rfp_margin * depth as i32 >= beta
            {
                return beta;
            }
            // Razoring: static eval sits far below alpha — trust it (no qsearch).
            // Only `value <= alpha` is asserted, so return alpha, not the lower
            // static_eval (which would over-claim the upper bound).
            if self.config.razoring
                && depth <= self.config.razor_max_depth
                && static_eval + self.config.razor_margin <= alpha
            {
                return alpha;
            }
            // Null-move pruning: hand the opponent a free move; if we still beat
            // beta, this node is too good to be worth a full search.
            if self.config.null_move
                && null_ok
                && depth >= self.config.nmp_min_depth
                && static_eval >= beta
            {
                let r = self.config.nmp_reduction;
                let reduced = depth.saturating_sub(1 + r);
                let child = <Gm<E> as Game>::null_move(state);
                let s = -self.negamax(&child, reduced, -beta, -beta + 1, ply + 1, false, false);
                if self.stopped {
                    return alpha; // discarded by caller's stop check
                }
                if s >= beta {
                    // Never propagate an unverified mate score out of a null search.
                    return if s >= WIN_SCORE { beta } else { s };
                }
            }
        }

        let mut best = -WIN_SCORE * 2;
        let mut best_move = None;
        let safe_ply = ply.min(MAX_PLY - 1);
        let mut quiets_seen = 0u32;

        for (move_idx, mv) in self
            .order_moves(state, tt_move, safe_ply, false)
            .into_iter()
            .enumerate()
        {
            let is_quiet = <Gm<E> as Game>::is_quiet(mv);
            let is_killer = self.is_killer(mv, safe_ply);
            if is_quiet {
                quiets_seen += 1;
            }
            let have_fallback = best > -WIN_SCORE;

            // Late move pruning: after enough quiet (wall) moves at low depth,
            // skip the remaining ones outright.
            if self.config.lmp
                && !is_pv
                && have_fallback
                && is_quiet
                && !is_killer
                && depth <= self.config.lmp_max_depth
                && quiets_seen > self.config.lmp_base + self.config.lmp_factor * depth as u32
            {
                continue;
            }

            // Futility pruning: frontier quiet move whose static eval cannot
            // realistically lift alpha.
            if self.config.futility
                && !is_pv
                && have_fallback
                && is_quiet
                && !is_killer
                && move_idx > 0
                && depth <= self.config.futility_max_depth
                && static_eval + self.config.futility_margin * depth as i32 <= alpha
            {
                continue;
            }

            let child = <Gm<E> as Game>::apply(state, mv);
            let child_pv = is_pv && move_idx == 0;
            let new_depth = depth - 1;

            // Late-move reduction target depth for late quiet moves.
            let reduced_depth = if self.config.lmr
                && depth >= self.config.lmr_min_depth
                && move_idx >= self.config.lmr_full_moves
                && is_quiet
                && !is_killer
            {
                new_depth.saturating_sub(self.config.lmr_reduction)
            } else {
                new_depth
            };

            let score = if move_idx == 0 || !self.config.pvs {
                // Full-window search (PVS off, or the principal move).
                if reduced_depth < new_depth {
                    let sr =
                        -self.negamax(&child, reduced_depth, -beta, -alpha, ply + 1, false, true);
                    if !self.stopped && sr > alpha {
                        -self.negamax(&child, new_depth, -beta, -alpha, ply + 1, child_pv, true)
                    } else {
                        sr
                    }
                } else {
                    -self.negamax(&child, new_depth, -beta, -alpha, ply + 1, child_pv, true)
                }
            } else {
                // PVS: scout with a null window, widening only when it pays off.
                let mut s =
                    -self.negamax(&child, reduced_depth, -alpha - 1, -alpha, ply + 1, false, true);
                if !self.stopped && reduced_depth < new_depth && s > alpha {
                    // Reduced scout beat alpha: re-search full depth, null window.
                    s = -self.negamax(&child, new_depth, -alpha - 1, -alpha, ply + 1, false, true);
                }
                if !self.stopped && s > alpha && s < beta {
                    // Looks like a new PV: full-window, full-depth re-search.
                    s = -self.negamax(&child, new_depth, -beta, -alpha, ply + 1, true, true);
                }
                s
            };

            if self.stopped {
                break;
            }

            if score > best {
                best = score;
                best_move = Some(mv);
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                // Beta cutoff: update killers + history.
                self.update_killers(mv, safe_ply);
                self.history[<Gm<E> as Game>::move_order_index(mv)] =
                    self.history[<Gm<E> as Game>::move_order_index(mv)].saturating_add((depth as i32) * (depth as i32));
                break;
            }
        }

        if self.stopped {
            return best;
        }

        if self.config.tt {
            let bound = if best <= alpha_orig {
                Bound::Upper
            } else if best >= beta {
                Bound::Lower
            } else {
                Bound::Exact
            };
            // Depth-preferred replacement: keep the deeper (more expensive) analysis.
            let cur = self.tt[idx]; // Copy
            if cur.key != hash || depth >= cur.depth {
                self.tt[idx] = TtSlot {
                    key: hash,
                    depth,
                    score: best,
                    bound,
                    best: best_move,
                };
            }
        }
        best
    }

    // ── Killer helpers ────────────────────────────────────────────────────────

    #[inline]
    fn is_killer(&self, mv: Mv<E>, ply: usize) -> bool {
        self.killers[ply][0] == Some(mv) || self.killers[ply][1] == Some(mv)
    }

    fn update_killers(&mut self, mv: Mv<E>, ply: usize) {
        if self.killers[ply][0] != Some(mv) {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = Some(mv);
        }
    }

    // ── Move ordering ─────────────────────────────────────────────────────────

    /// Return legal moves in priority order:
    ///   TT move → killer[0] → killer[1] → rest sorted by history (descending).
    /// Sorts in-place to avoid a second Vec allocation.
    fn order_moves(
        &self,
        state: &St<E>,
        tt_move: Option<Mv<E>>,
        ply: usize,
        wide: bool,
    ) -> Vec<Mv<E>> {
        let mut moves = if wide {
            <Gm<E> as Game>::search_moves_wide(state)
        } else {
            <Gm<E> as Game>::search_moves(state)
        };
        let k = &self.killers[ply];
        moves.sort_by_cached_key(|&mv| {
            std::cmp::Reverse(self.move_order_score(state, mv, tt_move, k))
        });
        moves
    }

    #[inline]
    fn over_budget(&self) -> bool {
        self.node_limit.is_some_and(|limit| self.nodes >= limit)
    }

    fn move_order_score(
        &self,
        _state: &St<E>,
        mv: Mv<E>,
        tt_move: Option<Mv<E>>,
        killers: &[Option<Mv<E>>; 2],
    ) -> i64 {
        if Some(mv) == tt_move {
            return 8_000_000_000;
        } else if killers[0] == Some(mv) {
            return 7_000_000_000;
        } else if killers[1] == Some(mv) {
            return 6_000_000_000;
        }

        self.history[<Gm<E> as Game>::move_order_index(mv)] as i64
    }

    /// Quiescence search: resolve pending forced captures before scoring a leaf,
    /// so the static eval is never read in the middle of a capture exchange (the
    /// horizon effect). Only entered when [`SearchConfig::quiescence`] is on.
    ///
    /// For mandatory-capture games (draughts) a capture cannot be declined, so
    /// there is no "stand-pat" option while captures exist: a node with captures
    /// must search them all; a node with none is quiet and returns its static
    /// eval. Games with no captures ([`Game::capture_moves`] empty) return the
    /// static eval immediately, so this is inert for Wall Chess.
    fn qsearch(&mut self, state: &St<E>, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        if self.over_budget() {
            self.stopped = true;
            return self.eval.eval(state, <Gm<E> as Game>::turn(state));
        }
        self.nodes += 1;
        if <Gm<E> as Game>::is_terminal(state) {
            return self.eval.eval(state, <Gm<E> as Game>::turn(state));
        }
        let stand_pat = self.eval.eval(state, <Gm<E> as Game>::turn(state));
        let caps = <Gm<E> as Game>::capture_moves(state);
        if caps.is_empty() || ply >= MAX_PLY - 1 {
            return stand_pat;
        }
        let mut best = -WIN_SCORE * 2;
        for mv in caps {
            let child = <Gm<E> as Game>::apply(state, mv);
            let s = -self.qsearch(&child, -beta, -alpha, ply + 1);
            if self.stopped {
                return best.max(alpha);
            }
            if s > best {
                best = s;
            }
            if s > alpha {
                alpha = s;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Heuristic;
    use crate::moves::{is_legal, legal_moves};
    use crate::state::{Cell, Move, Side, State};

    /// Pseudo-random balanced-ish positions for cross-config comparisons.
    fn sample_states(count: usize) -> Vec<State> {
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut out = vec![State::initial()];
        while out.len() < count {
            let mut state = State::initial();
            let plies = 4 + (next() as usize % 16);
            for _ in 0..plies {
                if state.winner.is_some() {
                    break;
                }
                let moves = legal_moves(&state);
                if moves.is_empty() {
                    break;
                }
                state = state.apply(moves[(next() as usize) % moves.len()]);
            }
            if state.winner.is_none() {
                out.push(state);
            }
        }
        out
    }

    #[test]
    fn immediate_goal_move_short_circuits_search() {
        let state = State {
            pawns: [Cell::new(8, 5), Cell::new(9, 1)],
            h_walls: 0,
            v_walls: 0,
            walls_left: [0, 0],
            turn: Side::South,
            winner: None,
        };
        let eval = Heuristic::default();
        let mut search = Search::new(&eval);
        let result = search.search(&state, 10);

        assert_eq!(result.best, Some(Move::Pawn(Cell::new(9, 5))));
        assert_eq!(result.score, WIN_SCORE);
        assert_eq!(result.depth, 1);
        assert_eq!(result.nodes, 0);
        assert!(!result.stopped);
    }

    #[test]
    fn node_limit_keeps_last_completed_depth() {
        let state = State::initial();
        let eval = Heuristic::default();
        let mut search = Search::new(&eval);
        let result = search.search_with_node_limit(&state, 10, 1_000);

        assert!(result.stopped);
        assert!(result.depth > 0);
        assert!(result.nodes <= 1_000);
        assert!(result.best.is_some_and(|mv| is_legal(&state, mv)));
    }

    /// PVS + aspiration are exact transforms: on top of plain alpha-beta (no
    /// lossy reductions) they must return the identical minimax score. A
    /// mismatch means a window/re-search bug.
    #[test]
    fn pvs_and_aspiration_preserve_exact_score() {
        let eval = Heuristic::default();
        let plain = SearchConfig::plain();
        let pvs_asp = SearchConfig {
            pvs: true,
            aspiration: true,
            ..SearchConfig::plain()
        };
        for state in sample_states(12) {
            for depth in 3..=6u8 {
                let sa = Search::with_config(&eval, plain).search(&state, depth).score;
                let sb = Search::with_config(&eval, pvs_asp)
                    .search(&state, depth)
                    .score;
                assert_eq!(
                    sa, sb,
                    "PVS/aspiration changed exact score at depth {depth} on {state:?}"
                );
            }
        }
    }

    /// Default config must reproduce the previously-deployed engine exactly,
    /// i.e. LMR + TT and nothing else — guards against an accidental flag flip.
    #[test]
    fn default_config_is_lmr_plus_tt_only() {
        let c = SearchConfig::default();
        assert!(c.tt && c.lmr);
        assert!(
            !(c.pvs
                || c.aspiration
                || c.null_move
                || c.rfp
                || c.razoring
                || c.futility
                || c.lmp)
        );
        assert_eq!(c.lmr_min_depth, 4);
        assert_eq!(c.lmr_full_moves, 4);
        assert_eq!(c.lmr_reduction, 1);
    }

    /// Every aggressive heuristic on: search must still complete and return a
    /// legal move on a range of positions (no panic, no illegal output).
    #[test]
    fn aggressive_config_returns_legal_moves() {
        let eval = Heuristic::default();
        let cfg = SearchConfig::aggressive();
        for state in sample_states(16) {
            let result = Search::with_config(&eval, cfg).search(&state, 6);
            assert!(!result.stopped);
            assert!(
                result.best.is_some_and(|mv| is_legal(&state, mv)),
                "aggressive search returned illegal/no move on {state:?}"
            );
        }
    }

    /// Null-move pruning must not blind the engine to a forced win.
    #[test]
    fn null_move_still_finds_forced_win() {
        // South to move, one step from the goal row with the path open.
        let state = State {
            pawns: [Cell::new(8, 5), Cell::new(2, 1)],
            h_walls: 0,
            v_walls: 0,
            walls_left: [5, 5],
            turn: Side::South,
            winner: None,
        };
        let eval = Heuristic::default();
        let cfg = SearchConfig {
            null_move: true,
            ..SearchConfig::aggressive()
        };
        let result = Search::with_config(&eval, cfg).search(&state, 6);
        assert_eq!(result.score, WIN_SCORE);
        assert_eq!(result.best, Some(Move::Pawn(Cell::new(9, 5))));
    }
}
