//! Parity probe: print feature vectors + net (value, top policy) for a handful
//! of positions, as JSONL. Feed the SAME feature vectors to the Python teacher
//! (trainer/parity_check.py) and compare values to verify the Rust ResT forward
//! matches PyTorch.
//!
//! Usage: rest_probe <weights.safetensors>   (needs --features net)

use gameboard_core::{encode, legal_moves, net::NetEvaluator, PolicyValue, State};

fn main() {
    let weights = std::env::args()
        .nth(1)
        .expect("usage: rest_probe <weights>");
    let net = NetEvaluator::load(&weights).expect("load net");

    // A few distinct positions: initial, then walk a deterministic move sequence.
    let mut state = State::initial();
    let mut states = vec![state.clone()];
    for k in 0..6 {
        let moves = legal_moves(&state);
        if moves.is_empty() {
            break;
        }
        // deterministic pick: spread across the move list
        state = state.apply(moves[(k * 7 + 3) % moves.len()]);
        states.push(state.clone());
    }

    for st in &states {
        let feats = encode(st);
        let (value, priors) = net.evaluate(st);
        // top-3 absolute-frame actions by prior
        let mut idx: Vec<usize> = (0..priors.len()).collect();
        idx.sort_by(|&a, &b| priors[b].partial_cmp(&priors[a]).unwrap());
        let top: Vec<String> = idx
            .iter()
            .take(3)
            .map(|&i| format!("[{i},{:.6}]", priors[i]))
            .collect();
        let feat_str: Vec<String> = feats.iter().map(|f| format!("{f}")).collect();
        println!(
            "{{\"value\":{value:.6},\"top\":[{}],\"f\":[{}]}}",
            top.join(","),
            feat_str.join(",")
        );
    }
}
