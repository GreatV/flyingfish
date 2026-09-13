use crate::{
    Component,
    math::{attention, heads, rope},
};
use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use rand::Rng;

pub(crate) struct Qwen<M: std::borrow::Borrow<Component>> {
    model: M,
    cache: Vec<Option<(Tensor, Tensor)>>,
    position: usize,
    chunk: usize,
}

impl<M: std::borrow::Borrow<Component>> Qwen<M> {
    pub(crate) fn new(model: M, chunk: usize) -> Result<Self> {
        let component = model.borrow();
        component.expect("model_type", "qwen3".into())?;
        component.expect("hidden_act", "silu".into())?;
        component.expect("use_sliding_window", false.into())?;
        component.expect("attention_bias", false.into())?;
        anyhow::ensure!(
            component.config["rope_parameters"]["rope_type"] == "default",
            "unsupported Music3 Qwen RoPE"
        );
        let layers = component.n("num_hidden_layers")?;
        anyhow::ensure!(
            component.config["layer_types"]
                .as_array()
                .is_some_and(|v| v.len() == layers && v.iter().all(|x| x == "full_attention")),
            "unsupported Qwen attention schedule"
        );
        Ok(Self {
            model,
            cache: vec![None; layers],
            position: 0,
            chunk,
        })
    }
    pub(crate) fn forward(&mut self, embeds: &Tensor) -> Result<Tensor> {
        let (batch, length, width) = embeds.dims3()?;
        let m = self.model.borrow();
        anyhow::ensure!(
            batch == 2 && width == m.n("hidden_size")? && length > 0,
            "invalid Music3 global-LM input"
        );
        anyhow::ensure!(
            self.position + length <= m.n("max_position_embeddings")?,
            "Music3 language model context exceeded"
        );
        let h = m.n("num_attention_heads")?;
        let kv = m.n("num_key_value_heads")?;
        let dim = m.n("head_dim")?;
        let eps = m.config["rms_norm_eps"]
            .as_f64()
            .context("missing Qwen RMSNorm epsilon")?;
        let theta = m.config["rope_parameters"]["rope_theta"]
            .as_f64()
            .context("missing Qwen RoPE theta")?;
        let mut x = embeds.clone();
        for layer in 0..self.cache.len() {
            let p = format!("model.layers.{layer}");
            let norm = m.rms(&x, &format!("{p}.input_layernorm"), eps)?;
            let q = heads(&m.linear(&norm, &format!("{p}.self_attn.q_proj"))?, h)?;
            let k = heads(&m.linear(&norm, &format!("{p}.self_attn.k_proj"))?, kv)?;
            let v = heads(&m.linear(&norm, &format!("{p}.self_attn.v_proj"))?, kv)?;
            let q = rope(
                &m.rms(&q, &format!("{p}.self_attn.q_norm"), eps)?,
                dim,
                theta,
                self.position,
            )?;
            let k = rope(
                &m.rms(&k, &format!("{p}.self_attn.k_norm"), eps)?,
                dim,
                theta,
                self.position,
            )?;
            let (k, v) = match &self.cache[layer] {
                None => (k, v),
                Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
            };
            let out = attention(&q, &k, &v, Some(self.position), self.chunk)?;
            self.cache[layer] = Some((k, v));
            x = (&x + m.linear(&out, &format!("{p}.self_attn.o_proj"))?)?;
            let norm = m.rms(&x, &format!("{p}.post_attention_layernorm"), eps)?;
            let gate = candle_nn::ops::silu(&m.linear(&norm, &format!("{p}.mlp.gate_proj"))?)?;
            let up = m.linear(&norm, &format!("{p}.mlp.up_proj"))?;
            x = (&x + m.linear(&(&gate * up)?, &format!("{p}.mlp.down_proj"))?)?;
        }
        self.position += length;
        Ok(m.rms(&x.narrow(1, length - 1, 1)?, "model.norm", eps)?
            .squeeze(1)?)
    }
}

pub(crate) fn depth_forward(model: &Component, input: &Tensor, chunk: usize) -> Result<Tensor> {
    let length = input.dim(1)?;
    anyhow::ensure!(
        length <= model.n("max_position_embeddings")?,
        "RVQ depth sequence too long"
    );
    let position = model
        .embedding("pos_embedding", &(0..length as u32).collect::<Vec<_>>())?
        .unsqueeze(0)?;
    let mut x = input.broadcast_add(&position)?;
    for layer in 0..model.n("num_layers")? {
        let p = format!("layers.{layer}");
        let norm = model.rms(&x, &format!("{p}.input_layernorm"), 1e-6)?;
        let h = model.n("num_attention_heads")?;
        let q = heads(&model.linear(&norm, &format!("{p}.attn.to_q"))?, h)?;
        let k = heads(&model.linear(&norm, &format!("{p}.attn.to_k"))?, h)?;
        let v = heads(&model.linear(&norm, &format!("{p}.attn.to_v"))?, h)?;
        x = (&x
            + model.linear(
                &attention(&q, &k, &v, Some(0), chunk)?,
                &format!("{p}.attn.to_out"),
            )?)?;
        let norm = model.rms(&x, &format!("{p}.post_attention_layernorm"), 1e-6)?;
        let gate = candle_nn::ops::silu(&model.linear(&norm, &format!("{p}.gate_proj"))?)?;
        let up = model.linear(&norm, &format!("{p}.up_proj"))?;
        x = (&x + model.linear(&(&gate * up)?, &format!("{p}.down_proj"))?)?;
    }
    Ok(model
        .rms(&x.narrow(1, length - 1, 1)?, "norm", 1e-6)?
        .squeeze(1)?)
}

fn sample(values: &[(u32, f32)], rng: &mut impl Rng) -> Result<u32> {
    let mut values = values
        .iter()
        .copied()
        .filter(|(_, v)| v.is_finite())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        !values.is_empty(),
        "Music3 sampling has no finite candidates"
    );
    values.sort_by(|a, b| b.1.total_cmp(&a.1));
    let threshold = values[49.min(values.len() - 1)].1;
    values.retain(|(_, v)| *v >= threshold);
    let max = values[0].1;
    let probabilities = values
        .iter()
        .map(|(_, v)| ((*v - max) as f64).exp())
        .collect::<Vec<_>>();
    let mut draw = rng.random::<f64>() * probabilities.iter().sum::<f64>();
    for ((token, _), weight) in values.iter().zip(probabilities) {
        draw -= weight;
        if draw <= 0. {
            return Ok(*token);
        }
    }
    Ok(values.last().context("empty sampling support")?.0)
}

fn guided_sample(logits: &Tensor, semantic: bool, rng: &mut impl Rng) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    anyhow::ensure!(
        logits.len() == 2,
        "Music3 guidance expects a conditional/unconditional pair"
    );
    let ids: Vec<usize> = if semantic {
        (151675..151675 + 16384).chain([151670]).collect()
    } else {
        (0..logits[0].len()).collect()
    };
    let mut conditional = ids.iter().map(|&i| logits[0][i]).collect::<Vec<_>>();
    conditional.sort_by(|a, b| b.total_cmp(a));
    let threshold = if semantic {
        conditional[49.min(conditional.len() - 1)]
    } else {
        f32::NEG_INFINITY
    };
    let values = ids
        .into_iter()
        .filter(|&i| logits[0][i] >= threshold)
        .map(|i| (i as u32, logits[1][i] + (logits[0][i] - logits[1][i]) * 1.5))
        .collect::<Vec<_>>();
    sample(&values, rng)
}

pub(crate) fn generate(
    language: &Component,
    depth: &Component,
    ids: &[u32],
    frames: usize,
    chunk: usize,
    rng: &mut impl Rng,
    progress: &mut impl FnMut(&str, usize, usize) -> Result<()>,
) -> Result<Tensor> {
    anyhow::ensure!(
        language.n("vocab_size")? >= 151675 + 16384,
        "Music3 language model is missing its semantic vocabulary"
    );
    anyhow::ensure!(
        depth.n("hidden_size")? == language.n("hidden_size")?
            && depth.n("num_codebooks")? == 8
            && depth.n("audio_vocab_size")? == 1024,
        "unsupported Music3 depth/global-LM pairing"
    );
    let prompt_length = ids.len() / 2;
    let mut lm = Qwen::new(language, chunk)?;
    let input = lm
        .model
        .embedding("model.embed_tokens", ids)?
        .reshape((2, prompt_length, ()))?;
    let mut hidden = lm.forward(&input)?;
    let mut frame_hiddens = Vec::new();
    for frame in 0..=frames {
        let semantic = guided_sample(&lm.model.linear(&hidden, "lm_head")?, true, rng)?;
        if semantic == 151670 {
            break;
        }
        let semantic_embed = lm.model.embedding("model.embed_tokens", &[semantic; 2])?;
        let mut sequence = vec![
            depth.linear(&hidden, "projection")?.unsqueeze(1)?,
            depth.linear(&semantic_embed, "projection")?.unsqueeze(1)?,
        ];
        let mut parts = vec![hidden.narrow(0, 0, 1)?];
        let mut feedback = semantic_embed;
        for codebook in 1..8 {
            let local = depth_forward(depth, &Tensor::cat(&sequence, 1)?, chunk)?;
            parts.push(local.narrow(0, 0, 1)?);
            let code = guided_sample(
                &depth.linear(&local, &format!("audio_heads.{}", codebook - 1))?,
                false,
                rng,
            )?;
            let embedding =
                depth.embedding("audio_embeddings", &[code + (codebook - 1) * 1024; 2])?;
            feedback = (&feedback + &embedding)?;
            if codebook < 7 {
                sequence.push(depth.linear(&embedding, "projection")?.unsqueeze(1)?);
            }
        }
        if frame > 0 {
            frame_hiddens.push(
                Tensor::cat(&parts, 1)?
                    .to_dtype(DType::F32)?
                    .to_device(&candle_core::Device::Cpu)?,
            );
            progress("autoregressive", frame, frames)?;
            if frame_hiddens.len() >= frames {
                break;
            }
        }
        hidden = lm.forward(&(feedback / 8f64.sqrt())?.unsqueeze(1)?)?;
    }
    anyhow::ensure!(
        !frame_hiddens.is_empty(),
        "Music3 generated zero audio frames"
    );
    Ok(Tensor::stack(&frame_hiddens, 1)?)
}
