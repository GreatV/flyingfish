use ff_edge0::config::Edge0Config;
use ff_edge0::model::Edge0Text;
use std::path::Path;
use std::time::Instant;

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn main() -> anyhow::Result<()> {
    let model_dir = Path::new("models/Edge0/Edge0-35B-A3B-preview");
    let config = Edge0Config::from_model_dir(model_dir)?;
    println!(
        "edge0: {} layers, {} experts, top_k={}",
        config.text_config.num_hidden_layers,
        config.text_config.num_experts,
        config.text_config.effective_top_k()
    );
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let mut model = Edge0Text::load(model_dir, config)?;
    #[cfg(feature = "cuda")]
    if std::env::var_os("EDGE0_GPU").is_some() {
        let experts = std::env::var("EDGE0_GPU")
            .map(|v| v == "full")
            .unwrap_or(false);
        println!(
            "enabling GPU ({})...",
            if experts {
                "full resident"
            } else {
                "static projections"
            }
        );
        let started = std::time::Instant::now();
        model.enable_gpu(experts)?;
        #[cfg(feature = "cuda")]
        if let Some(rt) = model.gpu.as_ref() {
            println!(
                "moE path: {}",
                if rt.use_moe_mega {
                    "moe_mega"
                } else {
                    "moe_closed"
                }
            );
        }
        println!("GPU ready in {:.1}s", started.elapsed().as_secs_f32());
    }

    let prompt = "Explain paging to a systems programmer.";
    let templated =
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n");
    let encoding = tokenizer
        .encode(templated.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let ids = encoding.get_ids();
    println!("prompt tokens: {} {:?}", ids.len(), ids);

    let started = Instant::now();
    let mut hidden = None;
    for &id in ids {
        hidden = Some(model.forward(id)?);
    }
    println!("prefill: {:.2}s", started.elapsed().as_secs_f32());

    let mut generated = Vec::new();
    let mut prev = 0u32;
    for step in 0..8 {
        let tick = Instant::now();
        #[cfg(feature = "cuda")]
        let best = if model.is_resident() && std::env::var_os("EDGE0_VEC_DECODE").is_none() {
            if step == 0 {
                model.first_token()?
            } else {
                model.forward_token(prev)?
            }
        } else {
            // A/B: the Vec path (forward + lm + host argmax).
            let h = hidden.take().unwrap();
            let logits = model.logits(&h)?;
            argmax(&logits)
        };
        #[cfg(not(feature = "cuda"))]
        let best = {
            let logits = model.logits(&hidden.take().unwrap())?;
            argmax(&logits)
        };
        generated.push(best);
        println!(
            "token {step}: id {best} in {:.2}s  [gdn-proj {:.0} gdn-recur {:.0} attn {:.0} moe {:.0}(load {:.0} compute {:.0}) logits {:.0} | fwd {:.0}ms]",
            tick.elapsed().as_secs_f32(),
            model.timing.gdn_proj_ms,
            model.timing.gdn_recur_ms,
            model.timing.attn_proj_ms,
            model.timing.moe_ms,
            model.timing.expert_load_ms,
            model.timing.moe_ms - model.timing.expert_load_ms,
            model.timing.logits_ms,
            model.timing.forward_ms
        );
        println!(
            "  syncs {} | layer-inner {:.0}ms | outer(embed) {:.1}ms",
            model.timing.gpu_syncs, model.timing.layer_inner_ms, model.timing.outer_ms
        );
        prev = best;
        #[cfg(feature = "cuda")]
        if std::env::var_os("EDGE0_VEC_DECODE").is_some() || !model.is_resident() {
            hidden = Some(model.forward(best)?);
        }
    }
    println!("output: {:?}", tokenizer.decode(&generated, false));
    Ok(())
}
