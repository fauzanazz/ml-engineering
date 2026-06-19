//! Gen-1 validation harness for the International Draughts heuristic bot.
//!
//! The shared `arena::play_match` always starts from `State::initial()`, and the
//! search is deterministic — so it would replay one identical game N times. This
//! harness instead seeds every game with a short **random opening** (distinct,
//! reproducible per seed) so a match is a real sample, and it adds a
//! **random-move opponent** (which is not an `Evaluator`, so it can't ride the
//! generic match loop). Games are independent pure functions of their seed, so
//! they run across threads and the aggregate is reproducible regardless of
//! timing.
//!
//! Usage:
//!   checkers_arena vsrandom [depth=8] [games=200] [seed=1] [open=6]
//!   checkers_arena ladder   [games=160] [seed=1] [open=6]      # d8/d6/d4/d2
//!   checkers_arena tune     [depth=6] [games=160] [seed=1] [open=6]
//!   checkers_arena duel  manA kingA advA backA dA  manB kingB advB backB dB \
//!                        [games=160] [seed=1] [open=6]
//!
//! Conventions (from memory): run the FULL game count — no early stop; a
//! favourable partial read is sampling bias. Never `cargo build` mid-match.

use std::thread;

use gameboard_core::checkers::{legal_moves, CheckersHeuristic, Color, State, DRAW_PLIES};
use gameboard_core::game::Game;
use gameboard_core::search::{Search, SearchConfig};
use gameboard_core::Checkers;

// ── seeded RNG (xorshift64; same family as the golden corpus generator) ──────
struct Rng(u64);
impl Rng {
    #[inline]
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[inline]
fn game_seed(base: u64, g: u32) -> u64 {
    // SplitMix-style spread so adjacent games are uncorrelated; never zero.
    let s = base
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((g as u64).wrapping_mul(0xD1B54A32D192ED03));
    let mut z = s ^ (s >> 31);
    z = z.wrapping_mul(0xBF58476D1CE4E5B9);
    z ^= z >> 27;
    if z == 0 {
        1
    } else {
        z
    }
}

// ── agents ──────────────────────────────────────────────────────────────────
#[derive(Clone, Copy)]
enum Agent {
    Search(CheckersHeuristic, u8),
    Random,
}

#[inline]
fn choose(a: &Agent, s: &State, rng: &mut Rng) -> Option<gameboard_core::checkers::Move> {
    match a {
        Agent::Search(h, d) => Search::with_config(h, SearchConfig::draughts()).search(s, *d).best,
        Agent::Random => {
            let m = legal_moves(s);
            if m.is_empty() {
                None
            } else {
                Some(m[rng.below(m.len())])
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Out {
    P0,
    P1,
    Draw,
}

#[inline]
fn material(bb_men: u64, bb_kings: u64) -> i32 {
    bb_men.count_ones() as i32 + 3 * bb_kings.count_ones() as i32
}

/// Final result of a position. Idle ⇒ draw; no-move ⇒ mover loses; a ply-cap
/// overrun is broken by weighted material (king≈3 men, matching the eval ratio)
/// so the sample isn't drowned in cap-draws — equal material stays a draw.
fn result(s: &State) -> Out {
    if s.idle >= DRAW_PLIES {
        return Out::Draw;
    }
    if !gameboard_core::checkers::any_legal_move(s) {
        // side to move (s.stm) has no move and loses
        return if s.stm == Color::White { Out::P1 } else { Out::P0 };
    }
    let wm = material(s.white & !s.kings, s.white & s.kings);
    let bm = material(s.black & !s.kings, s.black & s.kings);
    if wm > bm {
        Out::P0
    } else if bm > wm {
        Out::P1
    } else {
        Out::Draw
    }
}

/// One game: random opening, then `p0` (White) and `p1` (Black) alternate.
fn play_one(seed: u64, open: u32, p0: &Agent, p1: &Agent, max_plies: u32) -> Out {
    let mut rng = Rng(seed);
    let mut s = State::initial();
    for _ in 0..open {
        if Checkers::is_terminal(&s) {
            break;
        }
        let m = legal_moves(&s);
        s = s.apply(m[rng.below(m.len())]);
    }
    let mut ply = 0u32;
    while !Checkers::is_terminal(&s) && ply < max_plies {
        let agent = if s.stm == Color::White { p0 } else { p1 };
        let mv = match choose(agent, &s, &mut rng) {
            Some(m) => m,
            None => break,
        };
        s = s.apply(mv);
        ply += 1;
    }
    result(&s)
}

#[derive(Clone, Copy, Default)]
struct Tally {
    a: u32,
    b: u32,
    d: u32,
}
impl Tally {
    fn add(&mut self, o: &Tally) {
        self.a += o.a;
        self.b += o.b;
        self.d += o.d;
    }
    fn games(&self) -> u32 {
        self.a + self.b + self.d
    }
    /// Win rate for A with draws counted as half a point (score%).
    fn score(&self) -> f64 {
        let g = self.games().max(1) as f64;
        (self.a as f64 + 0.5 * self.d as f64) / g * 100.0
    }
}

/// A vs B over `games`, alternating seats each game, parallel across threads.
/// Each game is a pure function of its seed → reproducible regardless of timing.
fn run_match(a: Agent, b: Agent, games: u32, max_plies: u32, base: u64, open: u32) -> Tally {
    let nthreads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4) as u32;
    let nthreads = nthreads.clamp(1, games.max(1));
    let total = thread::scope(|sc| {
        let mut handles = Vec::new();
        for t in 0..nthreads {
            handles.push(sc.spawn(move || {
                let mut local = Tally::default();
                let mut g = t;
                while g < games {
                    let a_is_p0 = g % 2 == 0;
                    let (p0, p1) = if a_is_p0 { (&a, &b) } else { (&b, &a) };
                    let o = play_one(game_seed(base, g), open, p0, p1, max_plies);
                    match o {
                        Out::Draw => local.d += 1,
                        Out::P0 if a_is_p0 => local.a += 1,
                        Out::P1 if !a_is_p0 => local.a += 1,
                        _ => local.b += 1,
                    }
                    g += nthreads;
                }
                local
            }));
        }
        let mut sum = Tally::default();
        for h in handles {
            sum.add(&h.join().unwrap());
        }
        sum
    });
    total
}

fn report(label_a: &str, label_b: &str, t: Tally) {
    println!(
        "{label_a:>22}  vs  {label_b:<22}  {:5.1}%   {:>3}W-{:>3}L-{:>3}D / {}",
        t.score(),
        t.a,
        t.b,
        t.d,
        t.games()
    );
}

fn parse<T: std::str::FromStr>(s: Option<&String>, d: T) -> T {
    s.and_then(|v| v.parse().ok()).unwrap_or(d)
}

const MAX_PLIES: u32 = 400;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("vsrandom");
    let base = CheckersHeuristic::default();

    match cmd {
        // Sanity floor: the bot must crush a uniform-random mover.
        "vsrandom" => {
            let depth: u8 = parse(args.get(2), 8);
            let games: u32 = parse(args.get(3), 200);
            let seed: u64 = parse(args.get(4), 1);
            let open: u32 = parse(args.get(5), 6);
            println!(
                "# vsrandom  depth={depth} games={games} seed={seed} open={open}  (score% = win + ½draw)"
            );
            let t = run_match(
                Agent::Search(base, depth),
                Agent::Random,
                games,
                MAX_PLIES,
                seed,
                open,
            );
            report(&format!("bot d{depth}"), "random", t);
        }
        // Strength monotonicity: deeper search must beat shallower with the same
        // eval. Confirms the search itself is sound on draughts.
        "ladder" => {
            let games: u32 = parse(args.get(2), 160);
            let seed: u64 = parse(args.get(3), 1);
            let open: u32 = parse(args.get(4), 6);
            println!("# ladder  games={games} seed={seed} open={open}  (deeper vs shallower)");
            for (hi, lo) in [(8u8, 6u8), (6, 4), (4, 2)] {
                let t = run_match(
                    Agent::Search(base, hi),
                    Agent::Search(base, lo),
                    games,
                    MAX_PLIES,
                    seed,
                    open,
                );
                report(&format!("d{hi}"), &format!("d{lo}"), t);
            }
        }
        // Weight sweep: each candidate vs the default baseline at fixed depth.
        // Pick the Gen-1 weights from the winners. man fixed at 100 (unit).
        "tune" => {
            let depth: u8 = parse(args.get(2), 6);
            let games: u32 = parse(args.get(3), 160);
            let seed: u64 = parse(args.get(4), 1);
            let open: u32 = parse(args.get(5), 6);
            println!(
                "# tune  depth={depth} games={games} seed={seed} open={open}  (candidate vs default 100/300/4/6)"
            );
            // (label, man, king, advance, back_rank)
            let cands: &[(&str, i32, i32, i32, i32)] = &[
                ("king250", 100, 250, 4, 6),
                ("king350", 100, 350, 4, 6),
                ("king400", 100, 400, 4, 6),
                ("adv2", 100, 300, 2, 6),
                ("adv8", 100, 300, 8, 6),
                ("adv12", 100, 300, 12, 6),
                ("back0", 100, 300, 4, 0),
                ("back12", 100, 300, 4, 12),
                ("back20", 100, 300, 4, 20),
                ("k350_adv8", 100, 350, 8, 6),
                ("k350_adv8_b12", 100, 350, 8, 12),
                ("k400_adv8_b12", 100, 400, 8, 12),
            ];
            for (lbl, m, k, a, b) in cands {
                let cand = CheckersHeuristic::new(*m, *k, *a, *b);
                let t = run_match(
                    Agent::Search(cand, depth),
                    Agent::Search(base, depth),
                    games,
                    MAX_PLIES,
                    seed,
                    open,
                );
                report(lbl, "default", t);
            }
        }
        // Head-to-head of two fully specified configs.
        "duel" => {
            let a = CheckersHeuristic::new(
                parse(args.get(2), 100),
                parse(args.get(3), 300),
                parse(args.get(4), 4),
                parse(args.get(5), 6),
            );
            let da: u8 = parse(args.get(6), 6);
            let b = CheckersHeuristic::new(
                parse(args.get(7), 100),
                parse(args.get(8), 300),
                parse(args.get(9), 4),
                parse(args.get(10), 6),
            );
            let db: u8 = parse(args.get(11), 6);
            let games: u32 = parse(args.get(12), 160);
            let seed: u64 = parse(args.get(13), 1);
            let open: u32 = parse(args.get(14), 6);
            println!("# duel  games={games} seed={seed} open={open}");
            let t = run_match(
                Agent::Search(a, da),
                Agent::Search(b, db),
                games,
                MAX_PLIES,
                seed,
                open,
            );
            report(
                &format!("A {}/{}/{}/{} d{}", a.man, a.king, a.advance, a.back_rank, da),
                &format!("B {}/{}/{}/{} d{}", b.man, b.king, b.advance, b.back_rank, db),
                t,
            );
        }
        other => {
            eprintln!("unknown command {other:?}");
            eprintln!("usage: checkers_arena [vsrandom|ladder|tune|duel] ...");
            std::process::exit(2);
        }
    }
}
