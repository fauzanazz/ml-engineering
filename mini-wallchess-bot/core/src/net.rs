//! Trained value/policy net as a [`PolicyValue`] — the same MCTS seam the hand
//! heuristic plugs into, now backed by candle. Loads the safetensors exported by
//! `trainer/train.py`; arch and tensor names must match `trainer/model.py`.
//!
//! Gated behind the `net` feature so the default engine and the wasm graph build
//! stay free of the candle dependency. Enable with `--features net`.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::action::{action_index, index_to_move, ACTION_COUNT};
use crate::features::{encode, mirror_move, FEATURE_LEN};
use crate::mcts::PolicyValue;
use crate::state::State;

/// Hidden width — must equal `HIDDEN` in `trainer/model.py`.
const HIDDEN: usize = 256;

pub struct NetEvaluator {
    l1: Linear,
    l2: Linear,
    policy: Linear,
    value: Linear,
    device: Device,
}

impl NetEvaluator {
    /// Load weights from a safetensors file produced by the trainer.
    pub fn load(path: &str) -> candle_core::Result<Self> {
        let device = Device::Cpu;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)? };
        Ok(NetEvaluator {
            l1: candle_nn::linear(FEATURE_LEN, HIDDEN, vb.pp("l1"))?,
            l2: candle_nn::linear(HIDDEN, HIDDEN, vb.pp("l2"))?,
            policy: candle_nn::linear(HIDDEN, ACTION_COUNT, vb.pp("policy"))?,
            value: candle_nn::linear(HIDDEN, 1, vb.pp("value"))?,
            device,
        })
    }

    /// Construct directly from already-loaded bytes (for wasm, where there is no
    /// filesystem — the browser fetches the safetensors and hands over the buffer).
    pub fn from_buffer(bytes: &[u8]) -> candle_core::Result<Self> {
        let device = Device::Cpu;
        let vb = VarBuilder::from_slice_safetensors(bytes, DType::F32, &device)?;
        Ok(NetEvaluator {
            l1: candle_nn::linear(FEATURE_LEN, HIDDEN, vb.pp("l1"))?,
            l2: candle_nn::linear(HIDDEN, HIDDEN, vb.pp("l2"))?,
            policy: candle_nn::linear(HIDDEN, ACTION_COUNT, vb.pp("policy"))?,
            value: candle_nn::linear(HIDDEN, 1, vb.pp("value"))?,
            device,
        })
    }

    fn forward(&self, state: &State) -> candle_core::Result<(f32, Vec<f32>)> {
        let feats = encode(state); // me-frame, length FEATURE_LEN
        let x = Tensor::from_vec(feats, (1, FEATURE_LEN), &self.device)?;
        let h = self.l1.forward(&x)?.relu()?;
        let h = self.l2.forward(&h)?.relu()?;
        let logits = self.policy.forward(&h)?; // (1, ACTION_COUNT), me-frame
        let v = self.value.forward(&h)?.tanh()?; // (1, 1)

        let me_probs = candle_nn::ops::softmax(&logits, 1)?.to_vec2::<f32>()?;
        let me_probs = &me_probs[0];
        let value = v.flatten_all()?.to_vec1::<f32>()?[0];

        // Scatter me-frame priors back to the absolute frame MCTS works in.
        // `mirror_move` is an involution, so mirroring a me-frame move with the
        // same side yields the absolute move.
        let mut priors = vec![0.0f32; ACTION_COUNT];
        for (m, &p) in me_probs.iter().enumerate() {
            let abs = mirror_move(state.turn, index_to_move(m));
            priors[action_index(abs)] = p;
        }
        Ok((value, priors))
    }
}

impl PolicyValue for NetEvaluator {
    fn evaluate(&self, state: &State) -> (f32, Vec<f32>) {
        if state.winner.is_some() {
            return (-1.0, vec![0.0; ACTION_COUNT]);
        }
        self.forward(state).expect("net forward pass failed")
    }
}
