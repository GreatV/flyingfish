//! TRELLIS.2 shared-modulation flows and clean-sample CFG rescaling.
use crate::{
    slat_ops::{AttentionPlan, Ops, Rotary3d, layer_norm},
    sparse::Grid,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use ff_core::weights::ModelWeights;
use serde_json::Value;

pub struct Flow2 {
    ops: Ops,
    rows: usize,
    input: usize,
    output: usize,
    width: usize,
    heads: usize,
    layers: usize,
    condition_width: usize,
    attention: AttentionPlan,
    rotary: Rotary3d,
    qk_norm: bool,
    qk_cross: bool,
}
impl Flow2 {
    pub fn new(
        config: &Value,
        weights: ModelWeights,
        grid: &Grid,
        device: &Device,
        query_chunk: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            config["pe_mode"] == "rope" && config["share_mod"] == true,
            "TRELLIS.2 flow requires shared modulation and 3D RoPE"
        );
        let n = |key: &str| -> Result<usize> {
            let v = usize::try_from(
                config[key]
                    .as_u64()
                    .with_context(|| format!("missing flow {key}"))?,
            )?;
            anyhow::ensure!(v > 0, "flow {key} must be positive");
            Ok(v)
        };
        let width = n("model_channels")?;
        let heads = n("num_heads")?;
        anyhow::ensure!(
            width.is_multiple_of(heads) && query_chunk > 0 && grid.coords.iter().all(|c| c[0] == 0),
            "invalid batch-one TRELLIS.2 flow geometry"
        );
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            match config["dtype"].as_str() {
                Some("bfloat16") => DType::BF16,
                Some("float32") => DType::F32,
                Some("float16") => DType::F16,
                _ => anyhow::bail!("unsupported TRELLIS.2 flow dtype"),
            }
        };
        Ok(Self {
            rows: grid.coords.len(),
            input: n("in_channels")?,
            output: n("out_channels")?,
            width,
            heads,
            layers: n("num_blocks")?,
            condition_width: n("cond_channels")?,
            attention: AttentionPlan::new(grid, None, device)?,
            rotary: Rotary3d::new(grid, width / heads, device)?,
            qk_norm: config["qk_rms_norm"]
                .as_bool()
                .context("missing QK norm flag")?,
            qk_cross: config["qk_rms_norm_cross"]
                .as_bool()
                .context("missing cross QK norm flag")?,
            ops: Ops {
                weights,
                device: device.clone(),
                dtype,
                query_chunk,
                voxel_chunk: 256,
            },
        })
    }
    pub fn forward(
        &self,
        x: &Tensor,
        t: f32,
        condition: &Tensor,
        concat: Option<&Tensor>,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            x.dims() == [self.rows, self.output]
                && condition.dims().len() == 3
                && condition.dims()[0] == 1
                && condition.dims()[2] == self.condition_width,
            "TRELLIS.2 flow input shape mismatch"
        );
        let input = match concat {
            Some(c) => Tensor::cat(&[x, c], 1)?,
            None => x.clone(),
        };
        anyhow::ensure!(
            input.dim(1)? == self.input,
            "TRELLIS.2 concatenated conditioning width mismatch"
        );
        let time = Tensor::new(&[t], &self.ops.device)?;
        let emb = self.ops.timestep_fp32(&time, self.width)?;
        let global = self
            .ops
            .linear(&candle_nn::ops::silu(&emb)?, "adaLN_modulation.1")?
            .to_dtype(self.ops.dtype)?;
        let condition = condition.to_dtype(self.ops.dtype)?;
        let mut h = self
            .ops
            .linear(&input, "input_layer")?
            .to_dtype(self.ops.dtype)?;
        for layer in 0..self.layers {
            let p = format!("blocks.{layer}");
            let modulation = self
                .ops
                .get(&format!("{p}.modulation"), DType::F32)?
                .broadcast_add(&global.to_dtype(DType::F32)?)?
                .to_dtype(self.ops.dtype)?;
            let m = |i| modulation.narrow(1, i * self.width, self.width);
            let norm = self
                .ops
                .norm(&h, None, 1e-6)?
                .broadcast_mul(&(m(1)? + 1.)?)?
                .broadcast_add(&m(0)?)?;
            let attn = self.ops.self_attention_rotary(
                &norm,
                &format!("{p}.self_attn"),
                self.heads,
                self.qk_norm,
                &self.attention,
                Some(&self.rotary),
            )?;
            h = (&h + attn.broadcast_mul(&m(2)?)?)?;
            let norm = self.ops.norm(&h, Some(&format!("{p}.norm2")), 1e-6)?;
            h = (&h
                + self.ops.cross_attention(
                    &norm,
                    &condition,
                    &format!("{p}.cross_attn"),
                    self.heads,
                    self.qk_cross,
                )?)?;
            let norm = self
                .ops
                .norm(&h, None, 1e-6)?
                .broadcast_mul(&(m(4)? + 1.)?)?
                .broadcast_add(&m(3)?)?;
            h = (&h
                + self
                    .ops
                    .ff(&norm, &format!("{p}.mlp"))?
                    .broadcast_mul(&m(5)?)?)?;
        }
        self.ops
            .linear(&layer_norm(&h.to_dtype(x.dtype())?, 1e-5)?, "out_layer")
    }
}

pub struct Sampling2 {
    pub steps: usize,
    pub strength: f64,
    pub rescale: f64,
    pub interval: (f64, f64),
    pub time_scale: f64,
    pub sigma_min: f64,
}
impl Sampling2 {
    pub fn read(value: &Value, steps: Option<usize>) -> Result<Self> {
        anyhow::ensure!(
            value["name"] == "FlowEulerGuidanceIntervalSampler",
            "unsupported TRELLIS.2 sampler"
        );
        let params = &value["params"];
        let number = |key: &str| {
            params[key]
                .as_f64()
                .with_context(|| format!("missing TRELLIS.2 sampler {key}"))
        };
        let steps = steps.unwrap_or(usize::try_from(
            params["steps"].as_u64().context("invalid sampling steps")?,
        )?);
        let result = Self {
            steps,
            strength: number("guidance_strength")?,
            rescale: number("guidance_rescale")?,
            interval: (
                params["guidance_interval"][0]
                    .as_f64()
                    .context("missing interval start")?,
                params["guidance_interval"][1]
                    .as_f64()
                    .context("missing interval end")?,
            ),
            time_scale: number("rescale_t")?,
            sigma_min: value["args"]["sigma_min"]
                .as_f64()
                .context("missing sigma_min")?,
        };
        anyhow::ensure!(
            result.steps > 0
                && result.time_scale.is_finite()
                && result.time_scale > 0.
                && result.strength.is_finite()
                && result.rescale.is_finite()
                && (0. ..=1.).contains(&result.rescale)
                && result.interval.0.is_finite()
                && result.interval.1.is_finite()
                && result.interval.0 <= result.interval.1
                && result.sigma_min.is_finite()
                && (0. ..1.).contains(&result.sigma_min),
            "invalid TRELLIS.2 sampling policy"
        );
        Ok(result)
    }
    pub fn sample(
        &self,
        flow: &Flow2,
        noise: &Tensor,
        cond: &Tensor,
        concat: Option<&Tensor>,
        sparse: bool,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Tensor> {
        let negative = cond.zeros_like()?;
        let mut x = noise.clone();
        let times = (0..=self.steps)
            .map(|i| {
                let t = 1. - i as f64 / self.steps as f64;
                self.time_scale * t / (1. + (self.time_scale - 1.) * t)
            })
            .collect::<Vec<_>>();
        for (step, pair) in times.windows(2).enumerate() {
            let t = pair[0];
            let strength = if self.interval.0 <= t && t <= self.interval.1 {
                self.strength
            } else {
                1.
            };
            let positive = flow.forward(
                &x,
                (t * 1000.) as f32,
                if strength == 0. { &negative } else { cond },
                concat,
            )?;
            let mut velocity = if strength == 1. || strength == 0. {
                positive.clone()
            } else {
                (positive.affine(strength, 0.)?
                    + flow
                        .forward(&x, (t * 1000.) as f32, &negative, concat)?
                        .affine(1. - strength, 0.)?)?
            };
            if strength != 0. && strength != 1. && self.rescale > 0. {
                let factor = self.sigma_min + (1. - self.sigma_min) * t;
                let common = (&x * (1. - self.sigma_min))?;
                let clean_positive = (&common - (positive * factor)?)?;
                let clean_cfg = (&common - (&velocity * factor)?)?;
                let std = |v: &Tensor| -> Result<Tensor> {
                    let mean = v.mean_all()?;
                    let variance = if sparse {
                        (v.sqr()?.mean_all()? - mean.sqr()?)?
                    } else {
                        (v.broadcast_sub(&mean)?.sqr()?.sum_all()? / (v.elem_count() - 1) as f64)?
                    };
                    Ok(variance.sqrt()?)
                };
                let denominator = std(&clean_cfg)?;
                let std_value = denominator.to_scalar::<f32>()?;
                anyhow::ensure!(
                    std_value.is_finite() && std_value > 0.,
                    "zero/non-finite CFG rescaling standard deviation"
                );
                let rescaled = clean_cfg.broadcast_mul(&(std(&clean_positive)? / denominator)?)?;
                let clean = ((rescaled * self.rescale)? + (clean_cfg * (1. - self.rescale))?)?;
                velocity = ((common - clean)? / factor)?;
            }
            x = (x - (velocity * (pair[0] - pair[1]))?)?;
            progress(step + 1, self.steps);
        }
        Ok(x)
    }
}
