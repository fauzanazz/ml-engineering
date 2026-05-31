//! Trained value/policy net as a [`PolicyValue`] — the same MCTS seam the hand
//! heuristic plugs into, now backed by candle. Loads the safetensors exported by
//! `trainer/train.py`; arch and tensor names must match `trainer/model.py`.
//!
//! Gated behind the `net` feature so the default engine and the wasm graph build
//! stay free of the candle dependency. Enable with `--features net`.

use std::collections::HashMap;

use candle_core::{bail, safetensors, Device, Tensor};
use candle_nn::{Linear, Module};

use crate::action::{action_index, index_to_move, ACTION_COUNT};
use crate::features::{encode, mirror_move, FEATURE_LEN};
use crate::mcts::PolicyValue;
use crate::state::State;

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
        let tensors = safetensors::load(path, &device)?;
        Self::from_tensors(tensors, device)
    }

    /// Construct directly from already-loaded bytes (for wasm, where there is no
    /// filesystem — the browser fetches the safetensors and hands over the buffer).
    pub fn from_buffer(bytes: &[u8]) -> candle_core::Result<Self> {
        let device = Device::Cpu;
        let tensors = safetensors::load_buffer(bytes, &device)?;
        Self::from_tensors(tensors, device)
    }

    fn from_tensors(
        mut tensors: HashMap<String, Tensor>,
        device: Device,
    ) -> candle_core::Result<Self> {
        let hidden = hidden_width(&tensors)?;
        check_weight(&tensors, "l1.weight", &[hidden, FEATURE_LEN])?;
        check_bias(&tensors, "l1.bias", hidden)?;
        check_weight(&tensors, "l2.weight", &[hidden, hidden])?;
        check_bias(&tensors, "l2.bias", hidden)?;
        check_weight(&tensors, "policy.weight", &[ACTION_COUNT, hidden])?;
        check_bias(&tensors, "policy.bias", ACTION_COUNT)?;
        check_weight(&tensors, "value.weight", &[1, hidden])?;
        check_bias(&tensors, "value.bias", 1)?;

        Ok(NetEvaluator {
            l1: take_linear(&mut tensors, "l1")?,
            l2: take_linear(&mut tensors, "l2")?,
            policy: take_linear(&mut tensors, "policy")?,
            value: take_linear(&mut tensors, "value")?,
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

fn hidden_width(tensors: &HashMap<String, Tensor>) -> candle_core::Result<usize> {
    let dims = tensor(tensors, "l1.weight")?.dims();
    match dims {
        [hidden, FEATURE_LEN] => Ok(*hidden),
        other => bail!("l1.weight has invalid shape {other:?}; expected [hidden, {FEATURE_LEN}]"),
    }
}

fn take_linear(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> candle_core::Result<Linear> {
    let weight = take_tensor(tensors, &format!("{prefix}.weight"))?;
    let bias = take_tensor(tensors, &format!("{prefix}.bias"))?;
    Ok(Linear::new(weight, Some(bias)))
}

fn take_tensor(tensors: &mut HashMap<String, Tensor>, name: &str) -> candle_core::Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor {name}")).bt())
}

fn tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> candle_core::Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor {name}")).bt())
}

fn check_weight(
    tensors: &HashMap<String, Tensor>,
    name: &str,
    expected: &[usize],
) -> candle_core::Result<()> {
    let dims = tensor(tensors, name)?.dims();
    if dims != expected {
        bail!("{name} has invalid shape {dims:?}; expected {expected:?}");
    }
    Ok(())
}

fn check_bias(
    tensors: &HashMap<String, Tensor>,
    name: &str,
    expected: usize,
) -> candle_core::Result<()> {
    check_weight(tensors, name, &[expected])
}

impl PolicyValue for NetEvaluator {
    fn evaluate(&self, state: &State) -> (f32, Vec<f32>) {
        if state.winner.is_some() {
            return (-1.0, vec![0.0; ACTION_COUNT]);
        }
        self.forward(state).expect("net forward pass failed")
    }
}
