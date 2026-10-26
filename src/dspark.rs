use crate::config::Config;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct DraftConfig {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub dtype: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub draft_vocab_size: usize,
    pub rms_norm_eps: f32,
    pub block_size: usize,
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
    pub num_target_layers: usize,
    pub markov_rank: usize,
    pub markov_head_type: String,
    pub projector_type: String,
    pub rope_parameters: Rope,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rope {
    pub rope_theta: f32,
    pub rope_type: String,
    pub rope_dim: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct Rules {
    pub h_mean: f64,
    pub h_rel: f64,
    pub b_mean: f64,
    pub s_mean: f64,
    pub kv_rel: f64,
    pub margin: f64,
}

impl Rules {
    pub fn read(path: &Path) -> Result<Self> {
        let table: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let gates = table.get("gates").context("DSpark gates missing")?;
        let number = |name: &str| -> Result<f64> {
            gates
                .get(name)
                .and_then(serde_json::Value::as_f64)
                .with_context(|| format!("DSpark {name} numeric gate missing"))
        };
        let embedded = |name: &str, marker: &str| -> Result<f64> {
            let text = gates
                .get(name)
                .and_then(serde_json::Value::as_str)
                .with_context(|| format!("DSpark {name} gate description missing"))?;
            text.split_once(marker)
                .with_context(|| format!("DSpark {name} lacks {marker}"))?
                .1
                .split([' ', ';', ')', ','])
                .next()
                .context("DSpark gate number missing")?
                .parse()
                .with_context(|| format!("DSpark {name} invalid numeric gate"))
        };
        let rules = Self {
            h_mean: number("H_mean_abs")?,
            h_rel: number("H_rel_rmse")?,
            b_mean: number("B_mean_abs")?,
            s_mean: number("s_mean_abs")?,
            kv_rel: embedded("kv_inject", "rel_rmse <= ")?,
            margin: embedded("proposals", "top1-top2 < ")?,
        };
        ensure!(
            [
                rules.h_mean,
                rules.h_rel,
                rules.b_mean,
                rules.s_mean,
                rules.kv_rel,
                rules.margin
            ]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0),
            "invalid DSpark numeric gate"
        );
        Ok(rules)
    }
}

pub const MIN_MOE_VERIFY_ROWS: usize = 2;
pub const MAX_MOE_VERIFY_ROWS: usize = 8;
pub const MOE_VERIFY_CONFIDENCE_TAU: f32 = 0.80;

/// Per-request speculative verification width cap and cumulative survival floor.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerifyCostSchedule {
    pub max_rows: usize,
    pub step_tau: f32,
    pub min_survival: [f32; MAX_MOE_VERIFY_ROWS],
}

impl VerifyCostSchedule {
    pub const UNCONSTRAINED: Self = Self {
        max_rows: MAX_MOE_VERIFY_ROWS,
        step_tau: MOE_VERIFY_CONFIDENCE_TAU,
        min_survival: [0.0; MAX_MOE_VERIFY_ROWS],
    };

    pub fn derive(
        draft: &DraftConfig,
        target: &crate::model_spec::ModelSpec,
        residency: (
            &crate::experts::plan::Capacity,
            crate::experts::plan::Admission,
            &crate::experts::plan::ServiceTable,
            u32,
        ),
        hardware: ([f64; 4], [u64; 2]),
        peer_transfers: &[(u64, f64, f64)],
        frequency: &[Vec<f64>],
    ) -> Result<Self> {
        let max_verify_rows = draft.target_rows(target)?;
        let geometry = crate::experts::catalogue::ExpertGeometry::from_model(target)?;
        let (capacity, admission, service, copy_engines) = residency;
        let ([gpu0_mem_bps, peer_mem_bps, h2d_bps, d2h_bps], [gpu0_l2_bytes, peer_l2_bytes]) =
            hardware;
        ensure!(
            gpu0_mem_bps.is_finite()
                && gpu0_mem_bps > 0.0
                && h2d_bps.is_finite()
                && h2d_bps > 0.0
                && d2h_bps.is_finite()
                && d2h_bps > 0.0
                && copy_engines > 0,
            "invalid verify cost bandwidths={:?} copy_engines={copy_engines}",
            hardware.0
        );
        ensure!(
            capacity.gpu0_layers <= geometry.layers
                && capacity.slots_per_layer.len() == geometry.layers as usize,
            "verify schedule capacity layers={} gpu0_layers={} expected={}",
            capacity.slots_per_layer.len(),
            capacity.gpu0_layers,
            geometry.layers
        );
        if capacity.gpu0_layers < geometry.layers {
            ensure!(
                peer_mem_bps.is_finite() && peer_mem_bps > 0.0,
                "invalid peer memory bandwidth={peer_mem_bps}"
            );
        }
        let resident = crate::experts::plan::initial_residency(
            frequency,
            &capacity.slots_per_layer,
            &geometry,
        )?;
        let experts = geometry.experts as usize;
        let top_k = f64::from(geometry.top_k);
        let uniform_p = top_k / f64::from(geometry.experts);
        let mut probs = Vec::with_capacity(geometry.layers as usize);
        let mut is_resident = Vec::with_capacity(geometry.layers as usize);
        for layer in 0..geometry.layers as usize {
            let mut flags = vec![false; experts];
            for &expert in &resident[layer] {
                flags[expert as usize] = true;
            }
            let total: f64 = frequency[layer].iter().sum();
            let row = if total > 0.0 {
                frequency[layer]
                    .iter()
                    .map(|&count| (count * top_k / total).min(1.0))
                    .collect::<Vec<_>>()
            } else {
                vec![uniform_p; experts]
            };
            probs.push(row);
            is_resident.push(flags);
        }
        let d_est = draft.round_weight_bytes()? as f64 / gpu0_mem_bps;
        let delta_i_rej = draft.rejection_replay_bytes(target)? as f64 / gpu0_mem_bps;
        let t_non_expert = target_decode_non_expert_bytes(target)? as f64 / gpu0_mem_bps;
        let slot_bytes = capacity.slot_bytes();
        let workers = service.worker_count()?;
        let max_service_rows = f64::from(service.max_rows());
        let hard_cap = admission.hard_cap(slot_bytes)?;
        let backed = capacity.host_backed(&geometry);
        let t_h2d = slot_bytes as f64 / h2d_bps;
        let t_d2h = slot_bytes as f64 / d2h_bps;
        // Unbacked slot replacement waits on victim D2H before H2D into the same slot.
        let t_swap_one = if backed { t_h2d } else { t_h2d + t_d2h };
        let t_cpu1 = service.estimate_seconds(1.0, 1.0, workers)?;
        let t_gpu1 = slot_bytes as f64 / gpu0_mem_bps;
        let gpu0_layers = capacity.gpu0_layers as usize;
        let mut non_resident_probs = Vec::new();
        let mut m1_total = 0.0;
        for layer in 0..gpu0_layers {
            for expert in 0..experts {
                if !is_resident[layer][expert] {
                    let p = probs[layer][expert];
                    non_resident_probs.push(p);
                    m1_total += p;
                }
            }
        }
        non_resident_probs.sort_by(|a, b| b.total_cmp(a));
        let queue_wait = if hard_cap > 0 {
            (m1_total / f64::from(hard_cap)).clamp(1.0, crate::experts::plan::F_SEED_SCALE)
        } else {
            1.0
        };
        let per_miss_deferral =
            (t_cpu1 - t_gpu1).max(0.0) + if backed { 0.0 } else { queue_wait * t_d2h };
        let mut v_est = vec![0.0; max_verify_rows + 1];
        for (k, v_slot) in v_est.iter_mut().enumerate().skip(1) {
            let k_i32 = k as i32;
            let k_f64 = k as f64;
            let mut c_moe_gpu0 = 0.0;
            let mut m_total = 0.0;
            for layer in 0..gpu0_layers {
                let mut u_l = 0.0;
                let mut m_l = 0.0;
                let mut m_l_1 = 0.0;
                for expert in 0..experts {
                    let p = probs[layer][expert];
                    let active = 1.0 - (1.0 - p).powi(k_i32);
                    u_l += active;
                    if !is_resident[layer][expert] {
                        m_l += active;
                        m_l_1 += p;
                    }
                }
                let h_l = (u_l - m_l).max(0.0);
                let h_choices = k_f64 * (top_k - m_l_1).max(0.0);
                let t_gpu =
                    moe_dram_bytes(h_l, h_choices, slot_bytes, gpu0_l2_bytes) / gpu0_mem_bps;
                m_total += m_l;
                let t_cpu = if m_l > 0.0 {
                    let r_bar = (k_f64 * m_l_1 / m_l).clamp(1.0, k_f64.min(max_service_rows));
                    service.estimate_seconds(r_bar, m_l, workers)?
                } else {
                    0.0
                };
                c_moe_gpu0 += t_gpu.max(t_cpu);
            }
            let mut c_peer = 0.0;
            if gpu0_layers < geometry.layers as usize {
                let send_bytes =
                    (k as u64) * (u64::from(geometry.hidden) * 2 + u64::from(geometry.top_k) * 6);
                let recv_bytes = (k as u64) * u64::from(geometry.hidden) * 2;
                let bounce = crate::experts::plan::peer_bounce_seconds(
                    peer_transfers,
                    send_bytes,
                    recv_bytes,
                )?;
                for probs_layer in &probs[gpu0_layers..] {
                    let u_l: f64 = probs_layer
                        .iter()
                        .map(|&p| 1.0 - (1.0 - p).powi(k_i32))
                        .sum();
                    let peer_bytes = moe_dram_bytes(u_l, k_f64 * top_k, slot_bytes, peer_l2_bytes);
                    c_peer += peer_bytes / peer_mem_bps + bounce;
                }
            }
            let overlap = if k == 1 { 0.0 } else { d_est };
            let c_swap = (m_total.min(f64::from(hard_cap)) * t_swap_one - overlap).max(0.0);
            let mut delta_c_cap = 0.0;
            if k >= 2 && hard_cap > 0 && !non_resident_probs.is_empty() {
                let cap_1 = f64::from(hard_cap);
                for j in 2..=k {
                    let j_i32 = j as i32;
                    let cap_j = (j as f64) * cap_1;
                    let mut cum = 0.0;
                    for &p in &non_resident_probs {
                        let next = cum + (1.0 - (1.0 - p).powi(j_i32));
                        let truncated = (next.min(cap_j) - cum.max(cap_1)).max(0.0);
                        delta_c_cap += truncated * p * per_miss_deferral;
                        cum = next;
                        if cum >= cap_j {
                            break;
                        }
                    }
                }
            }
            *v_slot = t_non_expert + c_moe_gpu0 + c_peer + c_swap + delta_c_cap;
        }
        let mut k_max = MIN_MOE_VERIFY_ROWS;
        let mut best_full_accept =
            (d_est + v_est[MIN_MOE_VERIFY_ROWS]) / (MIN_MOE_VERIFY_ROWS as f64);
        for (k, &v_k) in v_est.iter().enumerate().skip(MIN_MOE_VERIFY_ROWS + 1) {
            let full_accept = (d_est + v_k) / (k as f64);
            if full_accept < best_full_accept {
                best_full_accept = full_accept;
                k_max = k;
            }
        }
        let r_ref = v_est[1];
        ensure!(
            r_ref.is_finite() && r_ref > 0.0,
            "invalid verify reference token time={r_ref}"
        );
        let mut min_survival = [0.0f32; MAX_MOE_VERIFY_ROWS];
        for k in (MIN_MOE_VERIFY_ROWS + 1)..=k_max {
            let marginal = (v_est[k] - v_est[k - 1]).max(0.0) + delta_i_rej;
            min_survival[k - 1] = (marginal / (r_ref + delta_i_rej)).clamp(0.0, 1.0) as f32;
        }
        for slot in &mut min_survival[k_max..MAX_MOE_VERIFY_ROWS] {
            *slot = 1.0;
        }
        Ok(Self {
            max_rows: k_max,
            step_tau: MOE_VERIFY_CONFIDENCE_TAU,
            min_survival,
        })
    }
}

fn moe_dram_bytes(distinct: f64, total_choices: f64, slot_bytes: u64, l2_bytes: u64) -> f64 {
    if distinct <= 0.0 {
        return 0.0;
    }
    let repeats = (total_choices - distinct).max(0.0);
    let reuse_experts = 0.5 * (distinct - 1.0).max(0.0);
    let l2 = l2_bytes as f64;
    let gate_up_bytes = slot_bytes as f64 * (2.0 / 3.0);
    let down_bytes = slot_bytes as f64 * (1.0 / 3.0);
    let hit = |proj_bytes: f64| -> f64 {
        let reuse_bytes = reuse_experts * proj_bytes;
        if reuse_bytes <= l2 {
            1.0
        } else if l2 > 0.0 {
            l2 / reuse_bytes
        } else {
            0.0
        }
    };
    (distinct + repeats * (1.0 - hit(gate_up_bytes))) * gate_up_bytes
        + (distinct + repeats * (1.0 - hit(down_bytes))) * down_bytes
}

fn linear_weight_bytes(format: crate::model_spec::LinearFormat, n: u32, k: u32) -> Result<u64> {
    let elems = u64::from(n)
        .checked_mul(u64::from(k))
        .context("linear weight element overflow")?;
    match format {
        crate::model_spec::LinearFormat::Bf16 => {
            elems.checked_mul(2).context("linear BF16 byte overflow")
        }
        crate::model_spec::LinearFormat::Fp8E4m3Block { block, scale_dtype } => {
            let scale = u64::try_from(
                crate::fp8::Fp8BlockScale::new(block, scale_dtype)?.scale_bytes(n, k)?,
            )?;
            elems
                .checked_add(scale)
                .context("linear FP8 block byte overflow")
        }
        crate::model_spec::LinearFormat::Fp8E4m3Channel => u64::from(n)
            .checked_mul(4)
            .and_then(|scale| elems.checked_add(scale))
            .context("linear FP8 channel byte overflow"),
    }
}

pub fn target_decode_non_expert_bytes(spec: &crate::model_spec::ModelSpec) -> Result<u64> {
    use crate::model_spec::{Architecture, FfnKind, Mixer};
    let hidden = spec.hidden;
    let mut total = u64::from(spec.vocab)
        .checked_mul(u64::from(hidden))
        .and_then(|elems| elems.checked_mul(2))
        .context("target lm_head byte overflow")?;
    let gdn = if spec.layers.iter().any(|layer| layer.mixer == Mixer::Gdn) {
        Some(crate::op_contract::GdnSpec::from_model(spec)?)
    } else {
        None
    };
    let q_proj_rows = if matches!(
        spec.architecture,
        Architecture::Qwen35Text | Architecture::MiniCpmV47
    ) {
        spec.q_width
            .checked_mul(2)
            .context("gated Q projection width overflow")?
    } else {
        spec.q_width
    };
    let kv_rows = spec
        .kv_heads
        .checked_mul(spec.head_dim)
        .context("KV projection width overflow")?;
    let qkv_rows = kv_rows
        .checked_mul(2)
        .and_then(|kv2| q_proj_rows.checked_add(kv2))
        .context("QKV projection width overflow")?;
    for layer in &spec.layers {
        let mixer_bytes = match layer.mixer {
            Mixer::FullAttention => linear_weight_bytes(spec.linear_format, qkv_rows, hidden)?
                .checked_add(linear_weight_bytes(
                    spec.linear_format,
                    hidden,
                    spec.q_width,
                )?)
                .context("attention projection byte overflow")?,
            Mixer::Gdn => {
                let gdn = gdn.as_ref().context("GDN spec missing")?;
                let v_dim = gdn
                    .hv
                    .checked_mul(gdn.dv)
                    .context("GDN value width overflow")?;
                let qkvz_rows = gdn
                    .conv_dim()?
                    .checked_add(v_dim)
                    .context("GDN qkvz width overflow")?;
                let ab_bytes = u64::from(gdn.hv)
                    .checked_mul(2)
                    .and_then(|ab| ab.checked_mul(u64::from(hidden)))
                    .and_then(|elems| elems.checked_mul(2))
                    .context("GDN gate projection byte overflow")?;
                let state_rw = gdn
                    .image_bytes()?
                    .checked_mul(2)
                    .context("GDN state read/write overflow")?;
                linear_weight_bytes(spec.linear_format, qkvz_rows, hidden)?
                    .checked_add(linear_weight_bytes(spec.linear_format, hidden, v_dim)?)
                    .and_then(|n| n.checked_add(ab_bytes))
                    .and_then(|n| n.checked_add(state_rw))
                    .context("GDN layer byte overflow")?
            }
        };
        let ffn_bytes = match layer.ffn {
            FfnKind::Dense => {
                let gate_up = spec
                    .intermediate
                    .checked_mul(2)
                    .context("dense gate_up width overflow")?;
                linear_weight_bytes(spec.linear_format, gate_up, hidden)?
                    .checked_add(linear_weight_bytes(
                        spec.linear_format,
                        hidden,
                        spec.intermediate,
                    )?)
                    .context("dense FFN byte overflow")?
            }
            FfnKind::Moe => {
                let moe = spec.moe.context("MoE config missing")?;
                let router = u64::from(moe.experts)
                    .checked_mul(u64::from(hidden))
                    .and_then(|elems| elems.checked_mul(2))
                    .context("MoE router byte overflow")?;
                let shared_gate_up = moe
                    .shared_intermediate
                    .checked_mul(2)
                    .context("MoE shared gate_up overflow")?;
                let shared = linear_weight_bytes(spec.linear_format, shared_gate_up, hidden)?
                    .checked_add(linear_weight_bytes(
                        spec.linear_format,
                        hidden,
                        moe.shared_intermediate,
                    )?)
                    .context("MoE shared FFN byte overflow")?;
                let gate_scalar = if moe.shared_gate {
                    u64::from(hidden) * 2
                } else {
                    0
                };
                router
                    .checked_add(shared)
                    .and_then(|n| n.checked_add(gate_scalar))
                    .context("MoE non-expert FFN byte overflow")?
            }
        };
        total = total
            .checked_add(mixer_bytes)
            .and_then(|n| n.checked_add(ffn_bytes))
            .context("target decode non-expert byte overflow")?;
    }
    Ok(total)
}

impl DraftConfig {
    pub fn verify_rows(&self) -> Result<usize> {
        self.block_size
            .checked_add(1)
            .context("DSpark block_size+1 overflow")
    }
    pub fn target_rows(&self, spec: &crate::model_spec::ModelSpec) -> Result<usize> {
        let rows = self.verify_rows()?;
        ensure!(
            spec.moe.is_some()
                && self.head_dim == 256
                && self.num_attention_heads == 8
                && self.num_key_value_heads == 2,
            "DSpark target architecture={:?} draft attention={}/{}/{} requires MoE with 8/2/256 draft heads",
            spec.architecture,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim
        );
        ensure!(
            (MIN_MOE_VERIFY_ROWS..=crate::linear_plan::MAX_SKINNY_ROWS).contains(&rows),
            "DSpark block_size={} verify_rows={rows} exceeds skinny and routed arithmetic bound={}",
            self.block_size,
            crate::linear_plan::MAX_SKINNY_ROWS
        );
        Ok(rows.min(MAX_MOE_VERIFY_ROWS))
    }
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn target_state_bytes(&self, spec: &crate::model_spec::ModelSpec) -> Result<usize> {
        let rows = self.target_rows(spec)?;
        let layers = spec
            .layers
            .iter()
            .filter(|layer| layer.mixer == crate::model_spec::Mixer::Gdn)
            .count();
        let replay_inputs =
            usize::try_from(crate::op_contract::GdnSpec::from_model(spec)?.input_row_bytes()?)?
                .checked_mul(layers)
                .and_then(|n| n.checked_mul(rows - 1))
                .context("DSpark GDN replay input overflow")?;
        crate::mtp::checkpoint_bytes(spec, rows)?
            .checked_add(replay_inputs)
            .context("DSpark target transaction byte overflow")
    }
    pub fn capture_width(&self) -> usize {
        self.target_layer_ids.len() * self.hidden_size
    }
    pub fn workspace_row_bytes(&self) -> Result<usize> {
        let kv = self
            .num_key_value_heads
            .checked_mul(self.head_dim)
            .and_then(|n| n.checked_mul(5))
            .context("DSpark KV workspace row overflow")?;
        self.hidden_size
            .checked_mul(2)
            .and_then(|n| n.checked_add(kv))
            .and_then(|n| n.checked_mul(size_of::<half::bf16>()))
            .context("DSpark workspace row bytes overflow")
    }
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    pub fn qkv_dim(&self) -> usize {
        (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim
    }
    pub fn round_weight_bytes(&self) -> Result<u64> {
        let hidden = u64::try_from(self.hidden_size)?;
        let inter = u64::try_from(self.intermediate_size)?;
        let layers = u64::try_from(self.num_hidden_layers)?;
        let q_dim = u64::try_from(
            self.num_attention_heads
                .checked_mul(self.head_dim)
                .context("DSpark query width overflow")?,
        )?;
        let kv_dim = u64::try_from(self.kv_dim())?;
        let head_dim = u64::try_from(self.head_dim)?;
        let vocab = u64::try_from(self.vocab_size)?;
        let rank = u64::try_from(self.markov_rank)?;
        let capture = u64::try_from(self.capture_width())?;
        let block_rows = u64::try_from(self.block_size)?;
        let inject_elems = capture
            .checked_mul(hidden)
            .and_then(|n| n.checked_add(hidden))
            .and_then(|n| {
                let per_layer = kv_dim
                    .checked_mul(2)
                    .and_then(|kv2| kv2.checked_mul(hidden))
                    .and_then(|kv_proj| kv_proj.checked_add(head_dim))?;
                layers
                    .checked_mul(per_layer)
                    .and_then(|all| n.checked_add(all))
            })
            .context("DSpark inject weight element overflow")?;
        let layer_elems = u64::try_from(self.qkv_dim())?
            .checked_mul(hidden)
            .and_then(|qkv| q_dim.checked_mul(hidden).and_then(|o| qkv.checked_add(o)))
            .and_then(|attn| {
                inter
                    .checked_mul(3)
                    .and_then(|ffn_w| ffn_w.checked_mul(hidden))
                    .and_then(|ffn| attn.checked_add(ffn))
            })
            .and_then(|body| {
                hidden
                    .checked_mul(2)
                    .and_then(|hn| head_dim.checked_mul(2).and_then(|qn| hn.checked_add(qn)))
                    .and_then(|norms| body.checked_add(norms))
            })
            .context("DSpark block layer element overflow")?;
        let head_elems = vocab
            .checked_mul(hidden)
            .and_then(|lm| lm.checked_add(hidden))
            .context("DSpark lm_head element overflow")?;
        let markov_step = vocab
            .checked_mul(rank)
            .and_then(|w2| w2.checked_add(rank))
            .context("DSpark Markov step element overflow")?;
        let markov_elems = block_rows
            .checked_mul(markov_step)
            .and_then(|all| {
                hidden
                    .checked_add(rank)
                    .and_then(|conf| all.checked_add(conf))
            })
            .context("DSpark Markov total element overflow")?;
        layers
            .checked_mul(layer_elems)
            .and_then(|body| body.checked_add(inject_elems))
            .and_then(|n| n.checked_add(head_elems))
            .and_then(|n| n.checked_add(markov_elems))
            .and_then(|elems| elems.checked_mul(2))
            .context("DSpark round weight byte overflow")
    }
    pub fn rejection_replay_bytes(&self, spec: &crate::model_spec::ModelSpec) -> Result<u64> {
        let gdn_layers = u64::try_from(
            spec.layers
                .iter()
                .filter(|layer| layer.mixer == crate::model_spec::Mixer::Gdn)
                .count(),
        )?;
        let gdn_bytes = if gdn_layers > 0 {
            let gdn = crate::op_contract::GdnSpec::from_model(spec)?;
            let input_row = gdn.input_row_bytes()?;
            gdn.image_bytes()?
                .checked_mul(4)
                .and_then(|state| state.checked_add(input_row))
                .and_then(|per_layer| per_layer.checked_mul(gdn_layers))
                .context("DSpark GDN rejection replay overflow")?
        } else {
            0
        };
        let attn_layers = u64::try_from(
            spec.layers
                .iter()
                .filter(|layer| layer.mixer == crate::model_spec::Mixer::FullAttention)
                .count(),
        )?;
        let target_kv_row = u64::from(spec.kv_heads)
            .checked_mul(u64::from(spec.head_dim))
            .and_then(|elems| elems.checked_mul(4))
            .and_then(|row| row.checked_mul(attn_layers))
            .context("DSpark target KV rollback byte overflow")?;
        let draft_layers = u64::try_from(self.num_hidden_layers)?;
        let draft_kv_row = u64::try_from(self.kv_dim())?
            .checked_mul(4)
            .and_then(|row| row.checked_mul(draft_layers))
            .context("DSpark draft KV rollback byte overflow")?;
        let target_kv_x2 = target_kv_row
            .checked_mul(2)
            .context("target KV rollback x2")?;
        let draft_kv_x2 = draft_kv_row
            .checked_mul(2)
            .context("draft KV rollback x2")?;
        gdn_bytes
            .checked_add(target_kv_x2)
            .and_then(|n| n.checked_add(draft_kv_x2))
            .context("DSpark rejection replay byte overflow")
    }
    pub fn read(path: &Path, target: &Config) -> Result<Self> {
        let c: Self = serde_json::from_value(read_config(path)?)?;
        ensure!(
            c.model_type == "qwen3" && c.architectures == ["Qwen3DSparkModel"],
            "unsupported DSpark architecture"
        );
        ensure!(c.dtype == "bfloat16", "DSpark dtype must be bfloat16");
        ensure!(
            c.hidden_size == target.hidden_size,
            "DSpark/target hidden_size mismatch"
        );
        ensure!(
            matches!(
                (c.num_attention_heads, c.num_key_value_heads, c.head_dim),
                (16, 2, 128) | (8, 2, 256)
            ) && c.num_attention_heads.checked_mul(c.head_dim) == Some(c.hidden_size),
            "unsupported DSpark attention geometry"
        );
        ensure!(
            c.intermediate_size > 0 && c.intermediate_size.is_multiple_of(256),
            "unsupported DSpark intermediate_size"
        );
        ensure!(
            c.vocab_size == target.vocab_size,
            "DSpark shared vocabulary mismatch"
        );
        ensure!(
            c.draft_vocab_size == c.vocab_size,
            "DSpark draft_vocab_size differs from vocab_size"
        );
        ensure!(
            c.num_hidden_layers == 5 && c.markov_rank == 256,
            "DSpark requires five layers and Markov rank256"
        );
        let row_limit = if c.head_dim == 128 { 8 } else { 64 };
        ensure!(
            (1..=row_limit).contains(&c.block_size),
            "DSpark block_size exceeds its attention geometry row limit {row_limit}"
        );
        ensure!(
            c.block_size.checked_mul(c.vocab_size).is_some()
                && c.vocab_size <= i32::MAX as usize
                && c.capture_width().checked_mul(c.hidden_size).is_some(),
            "DSpark tensor extent overflow"
        );
        ensure!(
            c.num_target_layers == target.num_hidden_layers,
            "invalid target hidden capture layers"
        );
        check_capture_layers(&c.target_layer_ids, target.num_hidden_layers)?;
        ensure!(
            (c.mask_token_id as usize) < target.vocab_size
                && c.rms_norm_eps.is_finite()
                && c.rms_norm_eps > 0.0,
            "invalid DSpark mask token or RMSNorm epsilon"
        );
        ensure!(
            c.projector_type == "dspark" && c.markov_head_type == "vanilla",
            "unsupported DSpark projector or Markov head"
        );
        ensure!(
            c.rope_parameters.rope_type == "default"
                && c.rope_parameters.rope_theta.is_finite()
                && c.rope_parameters.rope_theta > 0.0
                && c.rope_parameters.rope_dim.unwrap_or(c.head_dim) == c.head_dim,
            "DSpark requires default full-head rotary geometry"
        );
        if c.head_dim == 256 {
            crate::op_contract::FullAttentionSpec {
                hq: c.num_attention_heads as u32,
                hkv: c.num_key_value_heads as u32,
                d: c.head_dim as u32,
                scale: (c.head_dim as f32).sqrt().recip(),
                rms_eps: c.rms_norm_eps,
                rope_theta: c.rope_parameters.rope_theta,
                rotary_dim: c.head_dim as u32,
            }
            .validate()
            .context("unsupported DSpark full-attention parameters")?;
        }
        Ok(c)
    }
}

fn read_config(path: &Path) -> Result<serde_json::Value> {
    let path = path.join("config.json");
    let mut value: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&path).with_context(|| format!("read draft config {}", path.display()))?,
    )?;
    if let Some(nested) = value.get("dflash_config").cloned() {
        let fields = nested.as_object().context("invalid DSpark dflash_config")?;
        for name in fields.keys() {
            ensure!(
                matches!(
                    name.as_str(),
                    "target_layer_ids"
                        | "mask_token_id"
                        | "markov_rank"
                        | "markov_head_type"
                        | "projector_type"
                        | "attention_mode"
                        | "enable_confidence_head"
                        | "confidence_head_with_markov"
                        | "confidence_head_alpha"
                ),
                "unknown DSpark dflash_config field {name}"
            );
        }
        let root = value.as_object_mut().context("invalid DSpark config")?;
        for name in [
            "target_layer_ids",
            "mask_token_id",
            "markov_rank",
            "markov_head_type",
            "projector_type",
            "attention_mode",
            "enable_confidence_head",
            "confidence_head_with_markov",
            "confidence_head_alpha",
        ] {
            if let Some(field) = fields.get(name) {
                if let Some(existing) = root.get(name) {
                    ensure!(
                        name != "target_layer_ids" && existing == field,
                        "ambiguous DSpark config field {name}"
                    );
                }
                root.insert(name.into(), field.clone());
            }
        }
    }
    for (name, expected) in [
        ("attention_mode", serde_json::json!("gqa")),
        ("enable_confidence_head", serde_json::json!(true)),
        ("confidence_head_with_markov", serde_json::json!(true)),
        ("attention_bias", serde_json::json!(false)),
        ("mlp_bias", serde_json::json!(false)),
        ("hidden_act", serde_json::json!("silu")),
        ("sliding_window", serde_json::Value::Null),
        ("use_sliding_window", serde_json::json!(false)),
        ("tie_word_embeddings", serde_json::json!(false)),
    ] {
        if let Some(actual) = value.get(name) {
            ensure!(*actual == expected, "unsupported DSpark {name}");
        }
    }
    if let Some(alpha) = value.get("confidence_head_alpha") {
        ensure!(
            alpha.as_f64() == Some(1.0),
            "unsupported DSpark confidence_head_alpha"
        );
    }
    if let Some(types) = value.get("layer_types") {
        ensure!(
            types.as_array().is_some_and(|types| {
                value["num_hidden_layers"].as_u64() == Some(types.len() as u64)
                    && types.iter().all(|kind| kind == "full_attention")
            }),
            "unsupported DSpark layer_types"
        );
    }
    if let Some(rope) = value.get("rope_parameters") {
        serde_json::from_value::<Rope>(rope.clone()).context("invalid DSpark rope_parameters")?;
    }
    Ok(value)
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn read_capture_layers(path: &Path, target: &Config) -> Result<Vec<usize>> {
    let value = read_config(path)?;
    ensure!(
        value["architectures"] == serde_json::json!(["Qwen3DSparkModel"]),
        "unsupported capture draft architecture"
    );
    for (name, expected) in [
        ("hidden_size", target.hidden_size),
        ("vocab_size", target.vocab_size),
        ("num_target_layers", target.num_hidden_layers),
    ] {
        ensure!(
            value[name].as_u64() == Some(u64::try_from(expected)?),
            "capture draft/target {name} mismatch"
        );
    }
    let ids: Vec<usize> = serde_json::from_value(
        value
            .get("target_layer_ids")
            .context("DSpark target_layer_ids missing")?
            .clone(),
    )?;
    check_capture_layers(&ids, target.num_hidden_layers)?;
    Ok(ids)
}

fn check_capture_layers(ids: &[usize], target_layers: usize) -> Result<()> {
    ensure!(
        ids.len() == 5 && ids.iter().all(|&i| i < target_layers),
        "invalid target hidden capture layers"
    );
    for (index, &id) in ids.iter().enumerate() {
        ensure!(
            !ids[..index].contains(&id),
            "duplicate target hidden capture layer {id}"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accept {
    pub proposals: usize,
    pub bonus: u32,
}

pub fn select_verify_rows(
    confidences: &[f32],
    min_rows: usize,
    schedule: &VerifyCostSchedule,
) -> Result<usize> {
    ensure!(
        min_rows >= MIN_MOE_VERIFY_ROWS
            && min_rows <= schedule.max_rows
            && schedule.max_rows <= MAX_MOE_VERIFY_ROWS
            && schedule.max_rows <= confidences.len() + 1
            && schedule.step_tau.is_finite()
            && (0.0..=1.0).contains(&schedule.step_tau)
            && schedule
                .min_survival
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            && confidences.iter().all(|v| v.is_finite()),
        "invalid DSpark verify row selection min_rows={min_rows} max_rows={} confidences={} step_tau={}",
        schedule.max_rows,
        confidences.len(),
        schedule.step_tau
    );
    let mut rows = 1usize;
    let mut survival = 1.0f32;
    for (idx, &c) in confidences.iter().take(schedule.max_rows - 1).enumerate() {
        let next_rows = idx + 2;
        survival *= c;
        if c < schedule.step_tau || survival < schedule.min_survival[next_rows - 1] {
            break;
        }
        rows = next_rows;
    }
    Ok(rows.max(min_rows))
}

pub fn accept(proposals: &[u32], predictions: &[u32]) -> Result<Accept> {
    ensure!(
        predictions.len() == proposals.len() + 1,
        "verification must contain anchor plus proposals"
    );
    let matched = proposals
        .iter()
        .zip(predictions)
        .take_while(|(p, t)| p == t)
        .count();
    Ok(Accept {
        proposals: matched,
        bonus: predictions[matched],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejection_commits_only_matching_prefix_and_target_bonus() {
        assert_eq!(
            accept(&[3, 4, 5], &[3, 7, 5, 8]).unwrap(),
            Accept {
                proposals: 1,
                bonus: 7
            }
        );
        assert_eq!(
            accept(&[3, 4, 5], &[9, 4, 5, 8]).unwrap(),
            Accept {
                proposals: 0,
                bonus: 9
            }
        );
        assert_eq!(
            accept(&[3, 4, 5], &[3, 4, 5, 8]).unwrap(),
            Accept {
                proposals: 3,
                bonus: 8
            }
        );
        assert!(accept(&[3, 4, 5], &[3, 4, 5]).is_err());
        let unconstrained = VerifyCostSchedule::UNCONSTRAINED;
        assert_eq!(
            select_verify_rows(&[0.5; 15], MIN_MOE_VERIFY_ROWS, &unconstrained).unwrap(),
            2
        );
        assert_eq!(
            select_verify_rows(
                &[
                    0.85, 0.79, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99,
                    0.99, 0.99
                ],
                MIN_MOE_VERIFY_ROWS,
                &unconstrained,
            )
            .unwrap(),
            2
        );
        assert_eq!(
            select_verify_rows(
                &[
                    0.95, 0.85, 0.80, 0.79, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99,
                    0.99, 0.99
                ],
                MIN_MOE_VERIFY_ROWS,
                &unconstrained,
            )
            .unwrap(),
            4
        );
        assert_eq!(
            select_verify_rows(&[0.95; 15], MIN_MOE_VERIFY_ROWS, &unconstrained).unwrap(),
            MAX_MOE_VERIFY_ROWS
        );
        assert!(select_verify_rows(&[f32::NAN; 15], 2, &unconstrained).is_err());
        assert!(select_verify_rows(&[0.9; 15], 1, &unconstrained).is_err());
        assert!(select_verify_rows(&[0.9; 6], 2, &unconstrained).is_err());
    }

    #[test]
    fn fetch_cost_schedule_adapts_across_residency_regimes() -> Result<()> {
        use crate::experts::{
            catalogue::{ExpertGeometry, ExpertLayout},
            plan::{Admission, Capacity, ServiceCell, ServiceTable, admission_budget, equal_slots},
        };
        use crate::model_spec::{ExpertFormat, LinearFormat, ModelSpec};
        let bf16_model = ModelSpec::minicpm(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/minicpm-config.json"
            )))?,
            &serde_json::json!({"eos_token_id":[248044,248046]}),
        )?;
        let mut fp8_model = bf16_model.clone();
        fp8_model.linear_format = LinearFormat::Fp8E4m3Channel;
        fp8_model.expert_format = Some(ExpertFormat::Fp8W8a16);
        let geometry = ExpertGeometry::from_model(&fp8_model)?;
        let fp8_layout = ExpertLayout::new(ExpertFormat::Fp8W8a16, &geometry)?;
        let bf16_layout = ExpertLayout::new(ExpertFormat::Bf16, &geometry)?;
        let draft = DraftConfig {
            model_type: "qwen3".into(),
            architectures: vec!["Qwen3DSparkModel".into()],
            dtype: "bfloat16".into(),
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 5,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 256,
            vocab_size: 248144,
            draft_vocab_size: 248144,
            rms_norm_eps: 1e-6,
            block_size: 15,
            mask_token_id: 248076,
            target_layer_ids: vec![3, 11, 19, 27, 35],
            num_target_layers: 40,
            markov_rank: 256,
            markov_head_type: "vanilla".into(),
            projector_type: "dspark".into(),
            rope_parameters: Rope {
                rope_theta: 1e7,
                rope_type: "default".into(),
                rope_dim: Some(256),
            },
        };
        let cold_freq = vec![vec![0.0; geometry.experts as usize]; geometry.layers as usize];
        let skewed_freq: Vec<Vec<f64>> = (0..geometry.layers as usize)
            .map(|layer| {
                (0..geometry.experts as usize)
                    .map(|expert| {
                        let rank = (expert * 173 + layer * 37) % (geometry.experts as usize);
                        1.0 / ((rank as f64 + 2.5).powf(1.05))
                    })
                    .collect()
            })
            .collect();
        let make_table = |grid: &[(u32, u32, f64)]| -> Result<ServiceTable> {
            ServiceTable::new(
                "test",
                grid.iter()
                    .map(|&(rows, experts, seconds)| ServiceCell {
                        workers: 24,
                        rows,
                        experts,
                        seconds,
                    })
                    .collect(),
            )
        };
        let fp8_4090_service = make_table(&[
            (1, 1, 0.0001205),
            (1, 2, 0.0002298),
            (1, 4, 0.0004531),
            (1, 8, 0.0009117),
            (2, 1, 0.0002290),
            (2, 2, 0.0004567),
            (2, 4, 0.0008996),
            (2, 8, 0.0017913),
            (5, 1, 0.0005646),
            (5, 2, 0.0011233),
            (5, 4, 0.0022498),
            (5, 8, 0.0044884),
            (8, 1, 0.0008990),
            (8, 2, 0.0017870),
            (8, 4, 0.0035705),
            (8, 8, 0.0071404),
        ])?;
        let rates_4090 = [1_008.1e9, 0.0, 19.66e9, 18.68e9];
        let fp8_4090_admission = Admission {
            budget_bytes_per_token: admission_budget(
                &geometry,
                &fp8_4090_service,
                rates_4090[2],
                rates_4090[3],
                2,
            )?,
        };
        let fp8_slot_bytes = fp8_layout.slot_bytes();
        let full_capacity = Capacity {
            layout: fp8_layout.clone(),
            span_rows: 1024,
            wave_rows: 1024,
            device_expert_bytes: 10240 * fp8_slot_bytes,
            peer_expert_bytes: 0,
            host_expert_bytes: 10240 * fp8_slot_bytes,
            gpu0_layers: 40,
            slots_per_layer: vec![256; 40],
            host_blocks: 10240,
            spare_blocks: 28,
            disk_bytes: 0,
            worker_count: 24,
            disk_workers: 0,
            prefill_staging_slots: 2,
            pinned: true,
        };
        let l2_4090 = [75_497_472, 0];
        let full = VerifyCostSchedule::derive(
            &draft,
            &fp8_model,
            (&full_capacity, fp8_4090_admission, &fp8_4090_service, 2),
            (rates_4090, l2_4090),
            &[],
            &skewed_freq,
        )?;
        ensure!(
            full.max_rows == MAX_MOE_VERIFY_ROWS
                && select_verify_rows(&[0.80; 15], MIN_MOE_VERIFY_ROWS, &full)?
                    == MAX_MOE_VERIFY_ROWS,
            "full-resident schedule regressed: {full:?}"
        );

        let mut fp8_4090_capacity = full_capacity.clone();
        fp8_4090_capacity.slots_per_layer = equal_slots(4634, 40)?;
        fp8_4090_capacity.device_expert_bytes = 4634 * fp8_slot_bytes;
        let fp8_4090 = VerifyCostSchedule::derive(
            &draft,
            &fp8_model,
            (&fp8_4090_capacity, fp8_4090_admission, &fp8_4090_service, 2),
            (rates_4090, l2_4090),
            &[],
            &skewed_freq,
        )?;
        ensure!(
            fp8_4090.max_rows == MAX_MOE_VERIFY_ROWS
                && select_verify_rows(&[0.80; 15], MIN_MOE_VERIFY_ROWS, &fp8_4090)? < 7
                && select_verify_rows(&[0.97; 15], MIN_MOE_VERIFY_ROWS, &fp8_4090)?
                    == MAX_MOE_VERIFY_ROWS,
            "4090 FP8 host-backed schedule failed discriminator: {fp8_4090:?}"
        );

        let a4000_service = make_table(&[
            (1, 1, 0.0001500),
            (1, 2, 0.0002870),
            (1, 4, 0.0005660),
            (1, 8, 0.0011390),
            (2, 1, 0.0002860),
            (2, 2, 0.0005710),
            (2, 4, 0.0011240),
            (2, 8, 0.0022390),
            (5, 1, 0.0007050),
            (5, 2, 0.0014040),
            (5, 4, 0.0028120),
            (5, 8, 0.0056100),
            (8, 1, 0.0011240),
            (8, 2, 0.0022340),
            (8, 4, 0.0044630),
            (8, 8, 0.0089250),
        ])?;
        let rates_a4000 = [448.1e9, 448.1e9, 13.09e9, 13.09e9];
        let a4000_admission = Admission {
            budget_bytes_per_token: admission_budget(
                &geometry,
                &a4000_service,
                rates_a4000[2],
                rates_a4000[3],
                2,
            )?,
        };
        let mut peer_slots = equal_slots(1390, 21)?;
        peer_slots.extend(std::iter::repeat_n(256, 19));
        let peer_capacity = Capacity {
            layout: fp8_layout,
            span_rows: 1024,
            wave_rows: 1024,
            device_expert_bytes: 1390 * fp8_slot_bytes,
            peer_expert_bytes: 19 * 256 * fp8_slot_bytes,
            host_expert_bytes: 21 * 256 * fp8_slot_bytes,
            gpu0_layers: 21,
            slots_per_layer: peer_slots,
            host_blocks: 21 * 256,
            spare_blocks: 16,
            disk_bytes: 0,
            worker_count: 24,
            disk_workers: 0,
            prefill_staging_slots: 2,
            pinned: true,
        };
        let peer_transfers = [
            (4096u64, 200.0e6, 200.0e6),
            (8 << 20, 8.2e9, 8.2e9),
            (68 << 20, 10.6e9, 10.6e9),
        ];
        let fp8_peer = VerifyCostSchedule::derive(
            &draft,
            &fp8_model,
            (&peer_capacity, a4000_admission, &a4000_service, 2),
            (rates_a4000, [4_194_304, 4_194_304]),
            &peer_transfers,
            &skewed_freq,
        )?;
        ensure!(
            fp8_peer.max_rows == MAX_MOE_VERIFY_ROWS
                && select_verify_rows(&[0.80; 15], MIN_MOE_VERIFY_ROWS, &fp8_peer)? < 7
                && select_verify_rows(&[0.97; 15], MIN_MOE_VERIFY_ROWS, &fp8_peer)?
                    == MAX_MOE_VERIFY_ROWS,
            "2xA4000 peer-backed schedule failed discriminator: {fp8_peer:?}"
        );

        let bf16_service = make_table(&[
            (1, 1, 0.0001227),
            (1, 2, 0.0002309),
            (1, 4, 0.0004059),
            (1, 8, 0.0008052),
            (2, 1, 0.0001024),
            (2, 2, 0.0002542),
            (2, 4, 0.0005042),
            (2, 8, 0.0010606),
            (5, 1, 0.0001511),
            (5, 2, 0.0004135),
            (5, 4, 0.0008455),
            (5, 8, 0.0019198),
            (8, 1, 0.0001580),
            (8, 2, 0.0003348),
            (8, 4, 0.0006500),
            (8, 8, 0.0013023),
        ])?;
        let rates_4090_bf16 = [1_008.1e9, 0.0, 19.71e9, 18.77e9];
        let bf16_admission = Admission {
            budget_bytes_per_token: admission_budget(
                &geometry,
                &bf16_service,
                rates_4090_bf16[2],
                rates_4090_bf16[3],
                2,
            )?,
        };
        let bf16_slot_bytes = bf16_layout.slot_bytes();
        let bf16_unbacked_capacity = Capacity {
            layout: bf16_layout,
            span_rows: 1024,
            wave_rows: 1024,
            device_expert_bytes: 2283 * bf16_slot_bytes,
            peer_expert_bytes: 0,
            host_expert_bytes: 7966 * bf16_slot_bytes,
            gpu0_layers: 40,
            slots_per_layer: equal_slots(2283, 40)?,
            host_blocks: 7966,
            spare_blocks: 14,
            disk_bytes: 0,
            worker_count: 24,
            disk_workers: 0,
            prefill_staging_slots: 2,
            pinned: true,
        };
        for freq in [&cold_freq, &skewed_freq] {
            let bf16_4090 = VerifyCostSchedule::derive(
                &draft,
                &bf16_model,
                (&bf16_unbacked_capacity, bf16_admission, &bf16_service, 2),
                (rates_4090_bf16, l2_4090),
                &[],
                freq,
            )?;
            ensure!(
                (MIN_MOE_VERIFY_ROWS..=6).contains(&bf16_4090.max_rows)
                    && select_verify_rows(&[0.99; 15], MIN_MOE_VERIFY_ROWS, &bf16_4090)? <= 6,
                "4090 BF16 unbacked schedule failed cap discriminator: {bf16_4090:?}"
            );
        }
        let bf16_unbacked = VerifyCostSchedule::derive(
            &draft,
            &bf16_model,
            (&bf16_unbacked_capacity, bf16_admission, &bf16_service, 2),
            (rates_4090_bf16, l2_4090),
            &[],
            &skewed_freq,
        )?;
        let mut bf16_backed_control = bf16_unbacked_capacity;
        bf16_backed_control.host_blocks = 10240;
        let bf16_backed = VerifyCostSchedule::derive(
            &draft,
            &bf16_model,
            (&bf16_backed_control, bf16_admission, &bf16_service, 2),
            (rates_4090_bf16, l2_4090),
            &[],
            &skewed_freq,
        )?;
        ensure!(
            bf16_backed.max_rows > bf16_unbacked.max_rows,
            "negative control failed: host-backed perturbation did not relax BF16 max_rows ({:?} vs {:?})",
            bf16_backed.max_rows,
            bf16_unbacked.max_rows
        );
        Ok(())
    }

    #[test]
    fn target_capture_metadata_preserves_slots_and_rejects_bad_extents() -> Result<()> {
        let target: Config = serde_json::from_value(serde_json::json!({
            "model_type":"llama", "hidden_size":2048, "intermediate_size":6144,
            "num_hidden_layers":42, "num_attention_heads":16, "num_key_value_heads":2,
            "head_dim":128, "vocab_size":130560, "max_position_embeddings":32768,
            "rms_norm_eps":1e-6, "rope_theta":5000000.0, "rope_scaling":null,
            "hidden_act":"silu", "tie_word_embeddings":false,
            "torch_dtype":"bfloat16", "eos_token_id":[2]
        }))?;
        let ids = [39usize, 1, 30, 10, 20];
        let flat = serde_json::json!({
            "architectures":["Qwen3DSparkModel"], "hidden_size":2048,
            "vocab_size":130560, "num_target_layers":42, "target_layer_ids":ids
        });
        let mut nested = flat.clone();
        nested
            .as_object_mut()
            .context("capture config object")?
            .remove("target_layer_ids");
        nested["dflash_config"] = serde_json::json!({"target_layer_ids":ids});
        let root = std::env::temp_dir().join(format!(
            "flyingfish-capture-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir(&root)?;
        let result = (|| -> Result<()> {
            let write = |value: &serde_json::Value| -> Result<()> {
                Ok(std::fs::write(
                    root.join("config.json"),
                    serde_json::to_vec(value)?,
                )?)
            };
            for value in [&flat, &nested] {
                write(value)?;
                ensure!(
                    read_capture_layers(&root, &target)? == ids,
                    "capture slot order changed"
                );
            }
            ensure!(
                serde_json::from_value::<Vec<usize>>(nested["target_layer_ids"].clone()).is_err(),
                "flat-only capture control accepted nested metadata"
            );
            let mut invalid = vec![flat.clone(); 7];
            invalid[0]["dflash_config"] = serde_json::json!({"target_layer_ids":ids});
            invalid[1]["target_layer_ids"] = serde_json::json!([1, 10, 20, 30, 42]);
            invalid[2]["target_layer_ids"] = serde_json::json!([1, 10, 20, 30, 30]);
            invalid[3]["target_layer_ids"] = serde_json::json!([1, 10, 20, 30]);
            invalid[4]["hidden_size"] = serde_json::json!(5120);
            invalid[5]["vocab_size"] = serde_json::json!(248144);
            invalid[6]["num_target_layers"] = serde_json::json!(40);
            for value in invalid {
                write(&value)?;
                ensure!(
                    read_capture_layers(&root, &target).is_err(),
                    "invalid capture extent admitted"
                );
            }
            let mut full = flat.clone();
            for (name, value) in serde_json::json!({
                "model_type":"qwen3", "dtype":"bfloat16", "intermediate_size":6144,
                "num_hidden_layers":5, "num_attention_heads":16, "num_key_value_heads":2,
                "head_dim":128, "draft_vocab_size":130560, "rms_norm_eps":1e-6,
                "block_size":7, "mask_token_id":75982, "markov_rank":256,
                "markov_head_type":"vanilla", "projector_type":"dspark",
                "rope_parameters":{"rope_theta":5000000.0,"rope_type":"default"}
            })
            .as_object()
            .context("draft fixture object")?
            {
                full[name] = value.clone();
            }
            write(&full)?;
            ensure!(
                DraftConfig::read(&root, &target)?.block_size == 7,
                "dense draft rejected"
            );
            let flat = full.clone();
            let mut hybrid_target = target.clone();
            hybrid_target.intermediate_size = 512;
            hybrid_target.num_attention_heads = 16;
            hybrid_target.head_dim = 256;
            hybrid_target.vocab_size = 248144;
            hybrid_target.num_hidden_layers = 40;
            hybrid_target.rope_theta = 1e7;
            for (name, value) in serde_json::json!({
                "num_attention_heads":8, "head_dim":256, "vocab_size":248144,
                "draft_vocab_size":248144, "num_target_layers":40, "block_size":15,
                "rope_parameters":{"rope_theta":1e7,"rope_type":"default","rope_dim":256}
            })
            .as_object()
            .context("hybrid draft fixture object")?
            {
                full[name] = value.clone();
            }
            let mut nested = serde_json::Map::new();
            for name in [
                "target_layer_ids",
                "mask_token_id",
                "markov_rank",
                "markov_head_type",
                "projector_type",
            ] {
                nested.insert(
                    name.into(),
                    full.as_object_mut()
                        .context("draft object")?
                        .remove(name)
                        .context("draft field")?,
                );
            }
            nested.insert(
                "target_layer_ids".into(),
                serde_json::json!([3, 11, 19, 27, 35]),
            );
            nested.insert("mask_token_id".into(), serde_json::json!(248076));
            nested.insert("enable_confidence_head".into(), serde_json::json!(true));
            full["dflash_config"] = nested.into();
            write(&full)?;
            let draft = DraftConfig::read(&root, &hybrid_target)?;
            ensure!(
                draft.block_size == 15 && draft.kv_dim() == 512 && draft.qkv_dim() == 3072,
                "draft geometry borrowed target dimensions"
            );
            let model = crate::model_spec::ModelSpec::minicpm(
                &serde_json::from_str(include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/data/minicpm-config.json"
                )))?,
                &serde_json::json!({"eos_token_id":[248044,248046]}),
            )?;
            ensure!(
                draft.verify_rows()? == 16 && draft.target_rows(&model)? == MAX_MOE_VERIFY_ROWS,
                "draft verify extent changed"
            );
            ensure!(
                draft.workspace_row_bytes()? == 13_312,
                "draft row workspace must include projected/context and five KV widths"
            );
            let total = 24usize << 30;
            let mut row_request = crate::prefill::WorkspaceRequest {
                capacity: 16,
                requested: Some(16),
                free_after_resident: total / 16
                    + 16 * model.hidden as usize * size_of::<half::bf16>(),
                total,
                alignment: 1,
                row_bytes: model.hidden as usize * size_of::<half::bf16>(),
                fixed_transient_bytes: 0,
            };
            ensure!(
                crate::prefill::WorkspaceBudget::new(row_request, |_| Ok(0), Ok).is_ok(),
                "omitted draft-row reserve control did not admit"
            );
            row_request.row_bytes += draft.workspace_row_bytes()?;
            ensure!(
                crate::prefill::WorkspaceBudget::new(row_request, |_| Ok(0), Ok).is_err(),
                "draft rows escaped the workspace budget"
            );
            let verify_rows = draft.target_rows(&model)?;
            let gdn_layers = model
                .layers
                .iter()
                .filter(|layer| layer.mixer == crate::model_spec::Mixer::Gdn)
                .count();
            let gdn = crate::op_contract::GdnSpec::from_model(&model)?;
            let checkpoint = crate::mtp::checkpoint_bytes(&model, verify_rows)?;
            let replay_inputs =
                usize::try_from(gdn.input_row_bytes()?)? * gdn_layers * (verify_rows - 1);
            let snapshot_images =
                usize::try_from(gdn.image_bytes()?)? * gdn_layers * (verify_rows - 1);
            let expected_state_bytes = checkpoint + replay_inputs;
            let legacy_state_bytes = checkpoint + snapshot_images;
            let state_bytes = draft.target_state_bytes(&model)?;
            ensure!(
                state_bytes == expected_state_bytes && state_bytes < legacy_state_bytes,
                "block15 transaction must retain base GDN checkpoint, 15 replay input rows and 16 KV write slots"
            );
            let geometry = crate::experts::catalogue::ExpertGeometry::from_model(&model)?;
            let slot_bytes = crate::experts::catalogue::ExpertLayout::new(
                crate::model_spec::ExpertFormat::Bf16,
                &geometry,
            )?
            .slot_bytes();
            let base_reserve = usize::try_from(u64::from(geometry.layers + 2) * slot_bytes)?;
            let request = crate::prefill::ExpertWorkspaceRequest {
                mixer: crate::prefill::WorkspaceRequest {
                    capacity: 16,
                    requested: Some(16),
                    free_after_resident: base_reserve + state_bytes / 2,
                    total: 24 << 30,
                    alignment: 1,
                    row_bytes: model.hidden as usize * size_of::<half::bf16>(),
                    fixed_transient_bytes: 0,
                },
                span_rows: 16,
                wave_rows: 16,
                wave_workspace_bytes: 0,
                device_headroom_bytes: 0,
                host_fixed_bytes: 0,
                prefill_staging_slots: 1,
                disk_ring_slots: 0,
            };
            let budget = |request| {
                crate::prefill::ExpertWorkspaceBudget::new(
                    request,
                    geometry,
                    slot_bytes,
                    |_| Ok(0),
                    Ok,
                )
            };
            ensure!(
                budget(request).is_ok(),
                "omitted-reserve control did not admit"
            );
            let mut reserved = request;
            reserved.mixer.fixed_transient_bytes = state_bytes;
            ensure!(
                budget(reserved).is_err(),
                "snapshot reserve escaped expert admission"
            );
            let service = crate::experts::plan::ServiceTable::new(
                "test",
                vec![
                    crate::experts::plan::ServiceCell {
                        workers: 8,
                        rows: 1,
                        experts: 1,
                        seconds: 0.001,
                    },
                    crate::experts::plan::ServiceCell {
                        workers: 8,
                        rows: 1,
                        experts: geometry.experts,
                        seconds: 0.02,
                    },
                ],
            )?;
            for format in [
                crate::model_spec::ExpertFormat::Bf16,
                crate::model_spec::ExpertFormat::Fp8W8a16,
            ] {
                let layout = crate::experts::catalogue::ExpertLayout::new(format, &geometry)?;
                let format_slot_bytes = layout.slot_bytes();
                let free_bytes =
                    usize::try_from(u64::from(geometry.layers + 2) * format_slot_bytes)?
                        + legacy_state_bytes * 2;
                let mut replay_req = request;
                replay_req.mixer.free_after_resident = free_bytes;
                replay_req.mixer.fixed_transient_bytes = state_bytes;
                let mut legacy_req = replay_req;
                legacy_req.mixer.fixed_transient_bytes = legacy_state_bytes;
                let replay_budget = crate::prefill::ExpertWorkspaceBudget::new(
                    replay_req,
                    geometry,
                    format_slot_bytes,
                    |_| Ok(0),
                    Ok,
                )?;
                let legacy_budget = crate::prefill::ExpertWorkspaceBudget::new(
                    legacy_req,
                    geometry,
                    format_slot_bytes,
                    |_| Ok(0),
                    Ok,
                )?;
                let select_slots = |b: crate::prefill::ExpertWorkspaceBudget| -> Result<u32> {
                    let reserves = b.candidate.reserves;
                    let span_rows = b.candidate.span_rows;
                    let wave_rows = b.candidate.wave_rows;
                    Ok(crate::prefill::ExpertWorkspaceBudget::select(
                        &crate::experts::plan::PlanInputs {
                            geometry,
                            layout: layout.clone(),
                            device_free_bytes: free_bytes as u64,
                            host_available_bytes: 64 << 30,
                            h2d_bytes_per_second: 24e9,
                            d2h_bytes_per_second: 24e9,
                            copy_engines: 2,
                            disk_readers: 2,
                            host_registration_bytes: 64 << 30,
                            service: &service,
                            reserves: &reserves,
                            span_rows,
                            wave_rows,
                        },
                        16,
                        &[b],
                    )?
                    .1
                    .slots_total())
                };
                let replay_persistent = replay_budget
                    .candidate
                    .reserves
                    .persistent_bytes(format_slot_bytes)?;
                let legacy_persistent = legacy_budget
                    .candidate
                    .reserves
                    .persistent_bytes(format_slot_bytes)?;
                let replay_slots = select_slots(replay_budget)?;
                let legacy_slots = select_slots(legacy_budget)?;
                let expected_replay_slots =
                    u32::try_from((free_bytes as u64 - replay_persistent) / format_slot_bytes)?;
                let expected_legacy_slots =
                    u32::try_from((free_bytes as u64 - legacy_persistent) / format_slot_bytes)?;
                let delta_bytes = (legacy_state_bytes - state_bytes) as u64;
                let min_extra = u32::try_from(delta_bytes / format_slot_bytes)?;
                let max_extra = u32::try_from(delta_bytes.div_ceil(format_slot_bytes))?;
                ensure!(
                    replay_slots == expected_replay_slots
                        && legacy_slots == expected_legacy_slots
                        && (min_extra..=max_extra).contains(&(replay_slots - legacy_slots))
                        && min_extra > 0,
                    "planner slot gain mismatch for {format:?}: replay={replay_slots} legacy={legacy_slots}"
                );
            }
            let mut invalid_rows = DraftConfig::read(&root, &hybrid_target)?;
            for block in [0, crate::linear_plan::MAX_SKINNY_ROWS, usize::MAX] {
                invalid_rows.block_size = block;
                ensure!(
                    invalid_rows.target_rows(&model).is_err(),
                    "invalid verify extent admitted block={block}"
                );
            }
            invalid_rows.hidden_size = usize::MAX;
            ensure!(
                invalid_rows.workspace_row_bytes().is_err(),
                "overflowed draft hidden workspace admitted"
            );
            invalid_rows.hidden_size = draft.hidden_size;
            invalid_rows.num_key_value_heads = usize::MAX;
            ensure!(
                invalid_rows.workspace_row_bytes().is_err(),
                "overflowed draft KV workspace admitted"
            );
            println!(
                "PASS draft_workspace_row_bytes={} omitted_reserve=admitted charged_reserve=rejected overflow_controls=2",
                draft.workspace_row_bytes()?
            );
            let mut admitted = Vec::new();
            for (schema, original, target) in
                [("flat", &flat, &target), ("nested", &full, &hybrid_target)]
            {
                for (field, value) in [
                    ("attention_mode", serde_json::json!("mha")),
                    ("enable_confidence_head", serde_json::json!(false)),
                    ("confidence_head_with_markov", serde_json::json!(false)),
                    ("confidence_head_alpha", serde_json::json!(0.5)),
                    ("attention_bias", serde_json::json!(true)),
                    ("mlp_bias", serde_json::json!(true)),
                    ("hidden_act", serde_json::json!("gelu")),
                    ("sliding_window", serde_json::json!(4096)),
                    ("use_sliding_window", serde_json::json!(true)),
                    (
                        "layer_types",
                        serde_json::json!(vec!["sliding_attention"; 5]),
                    ),
                    ("tie_word_embeddings", serde_json::json!(true)),
                ] {
                    let mut invalid = original.clone();
                    if schema == "nested"
                        && [
                            "attention_mode",
                            "enable_confidence_head",
                            "confidence_head_with_markov",
                            "confidence_head_alpha",
                        ]
                        .contains(&field)
                    {
                        invalid["dflash_config"][field] = value;
                    } else {
                        invalid[field] = value;
                    }
                    write(&invalid)?;
                    let rejected = DraftConfig::read(&root, target).is_err();
                    println!(
                        "DSPARK_CONFIG_REJECTION schema={schema} field={field} rejected={rejected}"
                    );
                    if !rejected {
                        admitted.push(format!("{schema}.{field}"));
                    }
                }
                let mut invalid = original.clone();
                invalid["rope_parameters"]["unknown_rope"] = serde_json::json!(true);
                write(&invalid)?;
                let rejected = DraftConfig::read(&root, target).is_err();
                println!(
                    "DSPARK_CONFIG_REJECTION schema={schema} field=rope_parameters.unknown_rope rejected={rejected}"
                );
                if !rejected {
                    admitted.push(format!("{schema}.rope_parameters.unknown_rope"));
                }
            }
            let mut invalid = full.clone();
            invalid["dflash_config"]["unknown_draft"] = serde_json::json!(true);
            write(&invalid)?;
            let rejected = DraftConfig::read(&root, &hybrid_target).is_err();
            println!(
                "DSPARK_CONFIG_REJECTION schema=nested field=dflash_config.unknown_draft rejected={rejected}"
            );
            if !rejected {
                admitted.push("nested.dflash_config.unknown_draft".into());
            }
            ensure!(
                admitted.is_empty(),
                "unsupported draft fields admitted: {admitted:?}"
            );
            for (field, value) in [
                ("draft_vocab_size", serde_json::json!(130560)),
                ("block_size", serde_json::json!(65)),
                ("num_attention_heads", serde_json::json!(16)),
                ("enable_confidence_head", serde_json::json!(false)),
                (
                    "rope_parameters",
                    serde_json::json!({"rope_theta":1e7,"rope_type":"default","rope_dim":64}),
                ),
            ] {
                let mut invalid = full.clone();
                invalid[field] = value;
                write(&invalid)?;
                ensure!(
                    DraftConfig::read(&root, &hybrid_target).is_err(),
                    "invalid draft geometry admitted"
                );
            }
            Ok(())
        })();
        let cleanup = std::fs::remove_dir_all(root);
        result?;
        cleanup?;
        Ok(())
    }
}

#[derive(Debug, serde::Serialize)]
pub struct Round {
    pub logits_rows: Vec<usize>,
    pub proposals: Vec<u32>,
    pub predictions: Vec<u32>,
    pub accepted: usize,
    pub committed: usize,
    pub output: Vec<u32>,
    pub position: usize,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub inject_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tree: Option<TreeTiming>,
}

#[derive(Debug, serde::Serialize)]
pub struct TreeTiming {
    pub builder: crate::backend::TreeBuilder,
    pub base_ms: f64,
    pub build_ms: f64,
    pub waves: usize,
    pub requests: usize,
    pub used_requests: usize,
    pub batch_sizes: Vec<usize>,
    pub compared_serial: bool,
}
