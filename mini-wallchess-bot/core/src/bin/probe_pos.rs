//! Diagnostic: load the u4kytv stuck position and print the engine's ranked
//! moves (with scores) for North at depths 1..=4, to see why it oscillates.

use wallchess_core::eval::Heuristic;
use wallchess_core::search::Search;
use wallchess_core::state::{Cell, Orientation::*, Side, State, Wall};

fn main() {
    let walls = [
        (5, 4, H), (5, 6, H), (4, 3, V), (2, 3, V), (5, 8, H), (1, 4, H), (1, 6, H),
        (2, 6, V), (3, 5, H), (3, 4, V), (6, 5, V), (7, 6, H), (7, 7, V), (8, 7, H),
        (4, 8, H), (4, 6, H), (7, 4, V), (2, 7, H), (6, 3, V),
    ];
    let mut st = State::initial();
    st.h_walls = 0;
    st.v_walls = 0;
    for (r, c, o) in walls {
        let bit = 1u64 << (((r - 1) * 8 + (c - 1)) as u64);
        match o { H => st.h_walls |= bit, V => st.v_walls |= bit }
    }
    st.walls_left = [1, 0];
    st.pawns = [Cell::new(3, 7), Cell::new(6, 5)]; // [south, north]

    let h = Heuristic { w_path: 50, w_wall: 100 };
    for &(label, np) in &[("at (6,5)", Cell::new(6, 5)), ("at (6,4)", Cell::new(6, 4))] {
        st.pawns[Side::North.idx()] = np;
        st.turn = Side::North;
        println!("\n=== North {label}, North to move ===");
        for depth in 1..=4u8 {
            let mut s = Search::new(&h);
            let ranked = s.ranked(&st, depth);
            let top: Vec<String> = ranked.iter().take(4).map(|(m, sc)| format!("{m:?}={sc}")).collect();
            println!("  d{depth}: {}", top.join("  "));
        }
    }
}
