//! Cross-binary match referee. Pits two external `bestmove`-protocol engine
//! binaries head-to-head over randomized balanced openings, alternating sides.
//! The referee owns all game logic (move application, legality, win detection),
//! so the two engines can be different builds (e.g. new search vs old search).
//!
//! Usage: xmatch <engineA> <engineB> [games [open_plies [seed]]]
//!   engineA / engineB: shell command for each engine, e.g. "/tmp/bm_new 12 600000".
//!     The whole command goes in one arg (quote it); first token = binary, rest = args.
//!
//! Result is reported from A's perspective.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use gameboard_core::distance_to_goal;
use gameboard_core::legal_moves;
use gameboard_core::state::{Cell, Move, Orientation, Side, State, Wall};

fn xorshift(rng: &mut u64) -> u64 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    *rng
}

struct Engine {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Engine {
    fn spawn(cmd: &str) -> Engine {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        let mut c = Command::new(parts[0]);
        c.args(&parts[1..]);
        c.stdin(Stdio::piped()).stdout(Stdio::piped());
        let mut child = c.spawn().expect("spawn engine");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Engine {
            _child: child,
            stdin,
            stdout,
        }
    }

    fn ask(&mut self, state: &State) -> Option<Move> {
        let turn = if state.turn == Side::South { 0 } else { 1 };
        writeln!(
            self.stdin,
            "{} {} {} {} {} {} {} {} {}",
            state.h_walls,
            state.v_walls,
            state.pawns[0].r,
            state.pawns[0].c,
            state.pawns[1].r,
            state.pawns[1].c,
            state.walls_left[0],
            state.walls_left[1],
            turn
        )
        .ok()?;
        self.stdin.flush().ok()?;
        let mut line = String::new();
        self.stdout.read_line(&mut line).ok()?;
        parse_move(line.trim())
    }
}

fn parse_move(s: &str) -> Option<Move> {
    let f: Vec<&str> = s.split_whitespace().collect();
    match f.as_slice() {
        ["P", r, c] => Some(Move::Pawn(Cell::new(r.parse().ok()?, c.parse().ok()?))),
        ["H", r, c] => Some(Move::Wall(Wall {
            r: r.parse().ok()?,
            c: c.parse().ok()?,
            o: Orientation::H,
        })),
        ["V", r, c] => Some(Move::Wall(Wall {
            r: r.parse().ok()?,
            c: c.parse().ok()?,
            o: Orientation::V,
        })),
        _ => None,
    }
}

fn random_opening(open_plies: u32, rng: &mut u64) -> State {
    for _ in 0..200 {
        let mut state = State::initial();
        let mut ok = true;
        for _ in 0..open_plies {
            if state.winner.is_some() {
                ok = false;
                break;
            }
            let moves: Vec<Move> = legal_moves(&state)
                .into_iter()
                .filter(|m| matches!(m, Move::Pawn(_)))
                .collect();
            if moves.is_empty() {
                ok = false;
                break;
            }
            let mv = moves[(xorshift(rng) as usize) % moves.len()];
            state = state.apply(mv);
        }
        if !ok {
            continue;
        }
        let ds = distance_to_goal(&state, state.pawn(Side::South), Side::South.goal_row())
            .unwrap_or(999) as i32;
        let dn = distance_to_goal(&state, state.pawn(Side::North), Side::North.goal_row())
            .unwrap_or(999) as i32;
        if (ds - dn).abs() <= 1 {
            return state;
        }
    }
    State::initial()
}

#[derive(PartialEq)]
enum Res {
    A,
    B,
    Draw,
}

fn play(a: &mut Engine, b: &mut Engine, a_is_south: bool, start: State) -> Res {
    let mut state = start;
    let mut ply = 0u32;
    while state.winner.is_none() && ply < 300 {
        let a_turn = (state.turn == Side::South) == a_is_south;
        let eng = if a_turn { &mut *a } else { &mut *b };
        let mv = match eng.ask(&state) {
            Some(m) => m,
            None => break,
        };
        // Referee trusts engine legality but guards against protocol drift.
        if !gameboard_core::moves::is_legal(&state, mv) {
            eprintln!("illegal move from {} engine: {:?}", if a_turn { "A" } else { "B" }, mv);
            break;
        }
        state = state.apply(mv);
        ply += 1;
    }
    match state.winner {
        Some(Side::South) => {
            if a_is_south {
                Res::A
            } else {
                Res::B
            }
        }
        Some(Side::North) => {
            if a_is_south {
                Res::B
            } else {
                Res::A
            }
        }
        None => Res::Draw,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd_a = args.get(1).expect("engineA command").clone();
    let cmd_b = args.get(2).expect("engineB command").clone();
    let games: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let open_plies: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6);
    let seed: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

    let mut a = Engine::spawn(&cmd_a);
    let mut b = Engine::spawn(&cmd_b);

    let (mut aw, mut bw, mut draws) = (0u32, 0u32, 0u32);
    let mut rng = seed;
    for g in 0..games {
        let start = random_opening(open_plies, &mut rng);
        let a_is_south = g % 2 == 0;
        let res = play(&mut a, &mut b, a_is_south, start);
        let tag = match res {
            Res::A => {
                aw += 1;
                "A"
            }
            Res::B => {
                bw += 1;
                "B"
            }
            Res::Draw => {
                draws += 1;
                "draw"
            }
        };
        println!("game {g:>3}: A_south={a_is_south} winner={tag}");
    }
    println!(
        "TOTAL A [{cmd_a}] {aw} - {bw} B [{cmd_b}]  draws={draws}  (games={games} open={open_plies} seed={seed})"
    );
}
