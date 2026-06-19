//! Trained value/policy net as a [`PolicyValue`] — the same MCTS seam the hand
//! heuristic plugs into, now backed by candle. Loads the safetensors exported by
//! `trainer/train.py`; arch and tensor names must match `trainer/model.py`.
//!
//! Two architectures are supported; detection is automatic:
//!   MLP: conv1.weight absent → WallNet (l1, l2, policy, value)
//!   CNN: conv1.weight present → WallNetCNN (conv1..conv_out, fc_scalar, fc_head,
//!                                            policy, value)
//!
//! Gated behind the `net` feature so the default engine and the wasm graph build
//! stay free of the candle dependency. Enable with `--features net`.

use std::collections::HashMap;

use candle_core::{bail, safetensors, Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Linear, Module};

use crate::action::{action_index, index_to_move, ACTION_COUNT};
use crate::features::{encode, mirror_move, FEATURE_LEN};
use crate::mcts::PolicyValue;
use crate::state::State;

// ---------------------------------------------------------------------------
// Public loader: auto-detects MLP vs CNN from the safetensors keys
// ---------------------------------------------------------------------------

pub struct NetEvaluator {
    inner: NetArch,
    device: Device,
}

enum NetArch {
    Mlp(MlpNet),
    Cnn(CnnNet),
    ResT(ResTNet),
}

impl NetEvaluator {
    pub fn load(path: &str) -> candle_core::Result<Self> {
        let device = Device::Cpu;
        let tensors = safetensors::load(path, &device)?;
        Self::from_tensors(tensors, device)
    }

    pub fn from_buffer(bytes: &[u8]) -> candle_core::Result<Self> {
        let device = Device::Cpu;
        let tensors = safetensors::load_buffer(bytes, &device)?;
        Self::from_tensors(tensors, device)
    }

    fn from_tensors(
        mut tensors: HashMap<String, Tensor>,
        device: Device,
    ) -> candle_core::Result<Self> {
        let inner = if tensors.contains_key("stem.0.weight") {
            NetArch::ResT(ResTNet::from_tensors(&mut tensors)?)
        } else if tensors.contains_key("conv1.weight") {
            NetArch::Cnn(CnnNet::from_tensors(&mut tensors)?)
        } else {
            NetArch::Mlp(MlpNet::from_tensors(&mut tensors)?)
        };
        Ok(NetEvaluator { inner, device })
    }
}

impl PolicyValue for NetEvaluator {
    fn evaluate(&self, state: &State) -> (f32, Vec<f32>) {
        if state.winner.is_some() {
            return (-1.0, vec![0.0; ACTION_COUNT]);
        }
        let feats = encode(state);
        let result = match &self.inner {
            NetArch::Mlp(m) => m.forward(&feats, state, &self.device),
            NetArch::Cnn(c) => c.forward(&feats, state, &self.device),
            NetArch::ResT(r) => r.forward(&feats, state, &self.device),
        };
        result.expect("net forward pass failed")
    }
}

// ---------------------------------------------------------------------------
// Shared output: me-frame logits → absolute-frame priors + value
// ---------------------------------------------------------------------------

fn to_output(state: &State, logits: Tensor, v: Tensor) -> candle_core::Result<(f32, Vec<f32>)> {
    let me_probs = candle_nn::ops::softmax(&logits, 1)?.to_vec2::<f32>()?;
    let value = v.flatten_all()?.to_vec1::<f32>()?[0];
    let mut priors = vec![0.0f32; ACTION_COUNT];
    for (m, &p) in me_probs[0].iter().enumerate() {
        let abs = mirror_move(state.turn, index_to_move(m));
        priors[action_index(abs)] = p;
    }
    Ok((value, priors))
}

// ---------------------------------------------------------------------------
// MLP architecture — WallNet (l1, l2, policy, value)
// ---------------------------------------------------------------------------

struct MlpNet {
    l1: Linear,
    l2: Linear,
    policy: Linear,
    value: Linear,
}

impl MlpNet {
    fn from_tensors(tensors: &mut HashMap<String, Tensor>) -> candle_core::Result<Self> {
        let hidden = {
            let dims = get_tensor(tensors, "l1.weight")?.dims();
            match dims {
                [h, fl] if *fl == FEATURE_LEN => *h,
                other => bail!("l1.weight shape {other:?}; expected [hidden, {FEATURE_LEN}]"),
            }
        };
        check_shape(tensors, "l2.weight", &[hidden, hidden])?;
        check_shape(tensors, "policy.weight", &[ACTION_COUNT, hidden])?;
        check_shape(tensors, "value.weight", &[1, hidden])?;
        Ok(MlpNet {
            l1: take_linear(tensors, "l1")?,
            l2: take_linear(tensors, "l2")?,
            policy: take_linear(tensors, "policy")?,
            value: take_linear(tensors, "value")?,
        })
    }

    fn forward(
        &self,
        feats: &[f32],
        state: &State,
        device: &Device,
    ) -> candle_core::Result<(f32, Vec<f32>)> {
        let x = Tensor::from_slice(feats, (1, FEATURE_LEN), device)?;
        let h = self.l1.forward(&x)?.relu()?;
        let h = self.l2.forward(&h)?.relu()?;
        to_output(
            state,
            self.policy.forward(&h)?,
            self.value.forward(&h)?.tanh()?,
        )
    }
}

// ---------------------------------------------------------------------------
// CNN architecture — WallNetCNN
//
// Board tensor channels (me-frame, 4 × 9×9):
//   ch0  my pawn one-hot      feats[0..81]    row-major
//   ch1  opp pawn one-hot     feats[81..162]
//   ch2  h-walls 8×8          feats[162..226] placed in top-left 8×8 of 9×9
//   ch3  v-walls 8×8          feats[226..290]
//
// Scalar features (7):
//   feats[290] my_walls/10,  feats[291] opp_walls/10
//   feats[295] race_margin/16
//   feats[296..300] 4 progress flags
// ---------------------------------------------------------------------------

struct CnnNet {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv_out: Conv2d,
    fc_scalar: Linear,
    fc_head: Linear,
    policy: Linear,
    value: Linear,
}

const BOARD_SIZE: usize = 9;
const CNN_IN_CH: usize = 4;
const SCALAR_LEN: usize = 7;
const HEAD_HIDDEN: usize = 256;
const SCALAR_HIDDEN: usize = 32;

impl CnnNet {
    fn from_tensors(tensors: &mut HashMap<String, Tensor>) -> candle_core::Result<Self> {
        let ch = {
            let dims = get_tensor(tensors, "conv1.weight")?.dims();
            match dims {
                [c, _, 3, 3] => *c,
                other => bail!("conv1.weight shape {other:?}; expected [ch, 4, 3, 3]"),
            }
        };
        let board_flat = (ch / 2) * BOARD_SIZE * BOARD_SIZE;
        check_shape(tensors, "conv2.weight", &[ch, ch, 3, 3])?;
        check_shape(tensors, "conv3.weight", &[ch, ch, 3, 3])?;
        check_shape(tensors, "conv_out.weight", &[ch / 2, ch, 1, 1])?;
        check_shape(tensors, "fc_scalar.weight", &[SCALAR_HIDDEN, SCALAR_LEN])?;
        check_shape(
            tensors,
            "fc_head.weight",
            &[HEAD_HIDDEN, board_flat + SCALAR_HIDDEN],
        )?;
        check_shape(tensors, "policy.weight", &[ACTION_COUNT, HEAD_HIDDEN])?;
        check_shape(tensors, "value.weight", &[1, HEAD_HIDDEN])?;

        let p1 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let p0 = Conv2dConfig::default();
        Ok(CnnNet {
            conv1: take_conv2d(tensors, "conv1", p1)?,
            conv2: take_conv2d(tensors, "conv2", p1)?,
            conv3: take_conv2d(tensors, "conv3", p1)?,
            conv_out: take_conv2d(tensors, "conv_out", p0)?,
            fc_scalar: take_linear(tensors, "fc_scalar")?,
            fc_head: take_linear(tensors, "fc_head")?,
            policy: take_linear(tensors, "policy")?,
            value: take_linear(tensors, "value")?,
        })
    }

    fn forward(
        &self,
        feats: &[f32],
        state: &State,
        device: &Device,
    ) -> candle_core::Result<(f32, Vec<f32>)> {
        let board = build_board_tensor(feats, device)?;
        let scalars = build_scalar_tensor(feats, device)?;

        let h = self.conv1.forward(&board)?.relu()?;
        let h = self.conv2.forward(&h)?.relu()?;
        let h = self.conv3.forward(&h)?.relu()?;
        let h = self.conv_out.forward(&h)?.relu()?;
        let h_flat = h.flatten_from(1)?; // (1, ch/2 * 81)

        let s = self.fc_scalar.forward(&scalars)?.relu()?; // (1, 32)

        let combined = Tensor::cat(&[h_flat, s], 1)?; // (1, board_flat+32)
        let h = self.fc_head.forward(&combined)?.relu()?; // (1, 256)

        to_output(
            state,
            self.policy.forward(&h)?,
            self.value.forward(&h)?.tanh()?,
        )
    }
}

fn build_board_tensor(feats: &[f32], device: &Device) -> candle_core::Result<Tensor> {
    let mut board = vec![0.0f32; CNN_IN_CH * BOARD_SIZE * BOARD_SIZE];
    // ch0: my pawn
    board[..81].copy_from_slice(&feats[0..81]);
    // ch1: opp pawn
    board[81..162].copy_from_slice(&feats[81..162]);
    // ch2: h-walls (8×8 anchor grid into top-left 8 rows of ch2 9×9)
    for r in 0..8usize {
        for c in 0..8usize {
            board[2 * 81 + r * BOARD_SIZE + c] = feats[162 + r * 8 + c];
        }
    }
    // ch3: v-walls
    for r in 0..8usize {
        for c in 0..8usize {
            board[3 * 81 + r * BOARD_SIZE + c] = feats[226 + r * 8 + c];
        }
    }
    Tensor::from_vec(board, (1, CNN_IN_CH, BOARD_SIZE, BOARD_SIZE), device)
}

fn build_scalar_tensor(feats: &[f32], device: &Device) -> candle_core::Result<Tensor> {
    let s = vec![
        feats[290], feats[291], // my_walls/10, opp_walls/10
        feats[295], // race_margin/16
        feats[296], feats[297], feats[298], feats[299], // 4 progress flags
    ];
    Tensor::from_vec(s, (1, SCALAR_LEN), device)
}

// ---------------------------------------------------------------------------
// ResTNet architecture — interleaved Residual + Transformer tower
//
// Mirrors trainer/model.py WallNetResT. Shares the CNN board/scalar layout.
// Tower blocks auto-detected per index from tensor keys:
//   tower.{i}.conv1.weight  → ResBlock (conv-bn-conv-bn + skip)
//   tower.{i}.attn.in_proj_weight → TBlock (LN + MHA(4 heads) + LN + FFN)
// Detection key for the whole arch: stem.0.weight.
// ---------------------------------------------------------------------------

const REST_HEADS: usize = 4;
const REST_EPS: f64 = 1e-5;
const LN_EPS: f64 = 1e-5;

struct BatchNorm2dW {
    weight: Tensor, // (C,)
    bias: Tensor,
    mean: Tensor,
    var: Tensor,
}

impl BatchNorm2dW {
    fn load(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> candle_core::Result<Self> {
        Ok(Self {
            weight: take_tensor(tensors, &format!("{prefix}.weight"))?,
            bias: take_tensor(tensors, &format!("{prefix}.bias"))?,
            mean: take_tensor(tensors, &format!("{prefix}.running_mean"))?,
            var: take_tensor(tensors, &format!("{prefix}.running_var"))?,
        })
    }

    // x: (1, C, H, W). Normalize per-channel with running stats (eval mode).
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let c = self.weight.dims1()?;
        let shp = (1, c, 1, 1);
        let mean = self.mean.reshape(shp)?;
        let var = self.var.reshape(shp)?;
        let w = self.weight.reshape(shp)?;
        let b = self.bias.reshape(shp)?;
        let xc = x.broadcast_sub(&mean)?;
        let denom = (var + REST_EPS)?.sqrt()?;
        let xn = xc.broadcast_div(&denom)?;
        xn.broadcast_mul(&w)?.broadcast_add(&b)
    }
}

struct LayerNormW {
    weight: Tensor, // (C,)
    bias: Tensor,
}

impl LayerNormW {
    fn load(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> candle_core::Result<Self> {
        Ok(Self {
            weight: take_tensor(tensors, &format!("{prefix}.weight"))?,
            bias: take_tensor(tensors, &format!("{prefix}.bias"))?,
        })
    }

    // x: (..., C). Normalize over the last dim.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let last = x.rank() - 1;
        let mean = x.mean_keepdim(last)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(last)?;
        let xn = xc.broadcast_div(&(var + LN_EPS)?.sqrt()?)?;
        xn.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

enum ResTBlock {
    Res {
        conv1: Conv2d,
        bn1: BatchNorm2dW,
        conv2: Conv2d,
        bn2: BatchNorm2dW,
    },
    T {
        pos: Tensor, // (1, 81, C)
        norm1: LayerNormW,
        in_proj: Linear,  // (3C, C)
        out_proj: Linear, // (C, C)
        norm2: LayerNormW,
        ff0: Linear, // (2C, C)
        ff2: Linear, // (C, 2C)
    },
}

impl ResTBlock {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            ResTBlock::Res {
                conv1,
                bn1,
                conv2,
                bn2,
            } => {
                let h = bn1.forward(&conv1.forward(x)?)?.relu()?;
                let h = bn2.forward(&conv2.forward(&h)?)?;
                (x + h)?.relu()
            }
            ResTBlock::T {
                pos,
                norm1,
                in_proj,
                out_proj,
                norm2,
                ff0,
                ff2,
            } => {
                // x: (1, C, 9, 9) → tokens (1, 81, C), row-major
                let (b, c, hh, ww) = x.dims4()?;
                let n = hh * ww;
                let mut t = x.flatten_from(2)?.transpose(1, 2)?.contiguous()?; // (1,81,C)
                t = t.broadcast_add(pos)?;

                // self-attention (pre-norm)
                let normed = norm1.forward(&t)?;
                let attn = self_attention(&normed, in_proj, out_proj, b, n, c)?;
                t = (t + attn)?;

                // FFN (pre-norm)
                let ff = ff2.forward(&ff0.forward(&norm2.forward(&t)?)?.gelu_erf()?)?;
                t = (t + ff)?;

                // back to (1, C, 9, 9)
                t.transpose(1, 2)?.contiguous()?.reshape((b, c, hh, ww))
            }
        }
    }
}

// Multi-head self-attention matching torch.nn.MultiheadAttention(batch_first).
fn self_attention(
    x: &Tensor, // (B, N, C)
    in_proj: &Linear,
    out_proj: &Linear,
    b: usize,
    n: usize,
    c: usize,
) -> candle_core::Result<Tensor> {
    let h = REST_HEADS;
    let d = c / h;
    let qkv = in_proj.forward(x)?; // (B, N, 3C)
    let q = qkv.narrow(2, 0, c)?;
    let k = qkv.narrow(2, c, c)?;
    let v = qkv.narrow(2, 2 * c, c)?;
    // (B,N,C) → (B,h,N,d)
    let split = |t: &Tensor| -> candle_core::Result<Tensor> {
        t.reshape((b, n, h, d))?.transpose(1, 2)?.contiguous()
    };
    let q = split(&q)?;
    let k = split(&k)?;
    let v = split(&v)?;
    let scale = 1.0 / (d as f64).sqrt();
    let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?; // (B,h,N,N)
    let attn = candle_nn::ops::softmax(&scores, 3)?;
    let out = attn.matmul(&v)?; // (B,h,N,d)
    let out = out.transpose(1, 2)?.contiguous()?.reshape((b, n, c))?; // (B,N,C)
    out_proj.forward(&out)
}

struct ResTNet {
    stem_conv: Conv2d,
    stem_bn: BatchNorm2dW,
    tower: Vec<ResTBlock>,
    conv_out: Conv2d,
    fc_scalar: Linear,
    fc_head: Linear,
    policy: Linear,
    value: Linear,
}

impl ResTNet {
    fn from_tensors(tensors: &mut HashMap<String, Tensor>) -> candle_core::Result<Self> {
        let p1 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let p0 = Conv2dConfig::default();

        let stem_conv = Conv2d::new(take_tensor(tensors, "stem.0.weight")?, None, p1);
        let stem_bn = BatchNorm2dW::load(tensors, "stem.1")?;

        let mut tower = Vec::new();
        let mut i = 0usize;
        loop {
            let res_key = format!("tower.{i}.conv1.weight");
            let t_key = format!("tower.{i}.attn.in_proj_weight");
            if tensors.contains_key(&res_key) {
                let conv1 = Conv2d::new(take_tensor(tensors, &res_key)?, None, p1);
                let bn1 = BatchNorm2dW::load(tensors, &format!("tower.{i}.bn1"))?;
                let conv2 = Conv2d::new(
                    take_tensor(tensors, &format!("tower.{i}.conv2.weight"))?,
                    None,
                    p1,
                );
                let bn2 = BatchNorm2dW::load(tensors, &format!("tower.{i}.bn2"))?;
                tower.push(ResTBlock::Res {
                    conv1,
                    bn1,
                    conv2,
                    bn2,
                });
            } else if tensors.contains_key(&t_key) {
                let pos = take_tensor(tensors, &format!("tower.{i}.pos"))?;
                let norm1 = LayerNormW::load(tensors, &format!("tower.{i}.norm1"))?;
                let in_proj = Linear::new(
                    take_tensor(tensors, &t_key)?,
                    Some(take_tensor(
                        tensors,
                        &format!("tower.{i}.attn.in_proj_bias"),
                    )?),
                );
                let out_proj = take_linear(tensors, &format!("tower.{i}.attn.out_proj"))?;
                let norm2 = LayerNormW::load(tensors, &format!("tower.{i}.norm2"))?;
                let ff0 = take_linear(tensors, &format!("tower.{i}.ff.0"))?;
                let ff2 = take_linear(tensors, &format!("tower.{i}.ff.2"))?;
                tower.push(ResTBlock::T {
                    pos,
                    norm1,
                    in_proj,
                    out_proj,
                    norm2,
                    ff0,
                    ff2,
                });
            } else {
                break;
            }
            i += 1;
        }
        if tower.is_empty() {
            bail!("ResTNet: no tower blocks found");
        }

        Ok(ResTNet {
            stem_conv,
            stem_bn,
            tower,
            conv_out: take_conv2d(tensors, "conv_out", p0)?,
            fc_scalar: take_linear(tensors, "fc_scalar")?,
            fc_head: take_linear(tensors, "fc_head")?,
            policy: take_linear(tensors, "policy")?,
            value: take_linear(tensors, "value")?,
        })
    }

    fn forward(
        &self,
        feats: &[f32],
        state: &State,
        device: &Device,
    ) -> candle_core::Result<(f32, Vec<f32>)> {
        let board = build_board_tensor(feats, device)?;
        let scalars = build_scalar_tensor(feats, device)?;

        let mut h = self
            .stem_bn
            .forward(&self.stem_conv.forward(&board)?)?
            .relu()?;
        for blk in &self.tower {
            h = blk.forward(&h)?;
        }
        let h = self.conv_out.forward(&h)?.relu()?;
        let h_flat = h.flatten_from(1)?;

        let s = self.fc_scalar.forward(&scalars)?.relu()?;
        let combined = Tensor::cat(&[h_flat, s], 1)?;
        let h = self.fc_head.forward(&combined)?.relu()?;

        to_output(
            state,
            self.policy.forward(&h)?,
            self.value.forward(&h)?.tanh()?,
        )
    }
}

// ---------------------------------------------------------------------------
// Tensor utilities
// ---------------------------------------------------------------------------

fn take_conv2d(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    cfg: Conv2dConfig,
) -> candle_core::Result<Conv2d> {
    let w = take_tensor(tensors, &format!("{prefix}.weight"))?;
    let b = take_tensor(tensors, &format!("{prefix}.bias"))?;
    Ok(Conv2d::new(w, Some(b), cfg))
}

fn take_linear(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> candle_core::Result<Linear> {
    let w = take_tensor(tensors, &format!("{prefix}.weight"))?;
    let b = take_tensor(tensors, &format!("{prefix}.bias"))?;
    Ok(Linear::new(w, Some(b)))
}

fn take_tensor(tensors: &mut HashMap<String, Tensor>, name: &str) -> candle_core::Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor {name}")).bt())
}

fn get_tensor<'a>(
    tensors: &'a HashMap<String, Tensor>,
    name: &str,
) -> candle_core::Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor {name}")).bt())
}

fn check_shape(
    tensors: &HashMap<String, Tensor>,
    name: &str,
    expected: &[usize],
) -> candle_core::Result<()> {
    let dims = get_tensor(tensors, name)?.dims();
    if dims != expected {
        bail!("{name} shape {dims:?}; expected {expected:?}");
    }
    Ok(())
}
