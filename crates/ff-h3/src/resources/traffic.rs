use super::{
    ComputeTrafficEstimate, ResourceAssumptions, SequenceRows, TransformerShape,
    WeightMemoryEstimate, as_u64, checked_product, checked_sum,
};
use anyhow::{Context, Result};

pub(super) fn estimate_compute_and_traffic(
    model: TransformerShape,
    rows: SequenceRows,
    weights: WeightMemoryEstimate,
    assumptions: ResourceAssumptions,
) -> Result<ComputeTrafficEstimate> {
    let sequence = rows.total;
    let layers = as_u64(model.num_layers, "transformer layer count")?;
    let heads = as_u64(model.num_attention_heads, "attention head count")?;
    let head_dim = as_u64(model.attention_head_dim, "attention head dimension")?;
    let hidden = as_u64(model.hidden_size, "hidden size")?;
    let attention_width = checked_product("attention width", &[heads, head_dim])?;
    let ffn = as_u64(model.ffn_dim, "FFN dimension")?;

    let attention_qk_flops_per_evaluation = checked_product(
        "attention QK FLOPs per evaluation",
        &[2, layers, heads, sequence, sequence, head_dim],
    )?;
    let attention_pv_flops_per_evaluation = checked_product(
        "attention PV FLOPs per evaluation",
        &[2, layers, heads, sequence, sequence, head_dim],
    )?;
    let attention_projection_flops_per_evaluation = checked_product(
        "attention projection FLOPs per evaluation",
        &[2, layers, sequence, hidden, attention_width, 4],
    )?;
    let ffn_flops_per_evaluation = checked_product(
        "FFN FLOPs per evaluation",
        &[2, layers, sequence, hidden, ffn, 3],
    )?;
    let total_flops_per_evaluation = checked_sum(
        "total FLOPs per evaluation",
        &[
            attention_qk_flops_per_evaluation,
            attention_pv_flops_per_evaluation,
            attention_projection_flops_per_evaluation,
            ffn_flops_per_evaluation,
        ],
    )?;
    let total_schedule_flops = checked_product(
        "total schedule FLOPs",
        &[total_flops_per_evaluation, assumptions.evaluation_count],
    )?;

    let attention_and_ffn_weight_bytes_per_layer = checked_sum(
        "attention and FFN weight bytes per layer",
        &[
            weights.attention_stage_bytes,
            weights.feed_forward_stage_bytes,
        ],
    )?;
    let transformer_weight_bytes_per_evaluation_with_adaln_precompute = checked_product(
        "transformer weight bytes per precomputed evaluation",
        &[layers, attention_and_ffn_weight_bytes_per_layer],
    )?;
    let all_block_weight_bytes_per_layer = checked_sum(
        "all transformer block weight bytes per layer",
        &[
            attention_and_ffn_weight_bytes_per_layer,
            weights.adaln_stage_bytes,
        ],
    )?;
    let transformer_weight_bytes_per_evaluation_without_adaln_precompute = checked_product(
        "transformer weight bytes per dynamic evaluation",
        &[layers, all_block_weight_bytes_per_layer],
    )?;
    let transformer_weight_materialization_bytes_without_adaln_precompute = checked_product(
        "transformer weight materialization without AdaLN precompute",
        &[
            transformer_weight_bytes_per_evaluation_without_adaln_precompute,
            assumptions.evaluation_count,
        ],
    )?;
    let adaln_weight_bytes_per_pass = checked_product(
        "AdaLN weight bytes per pass",
        &[layers, weights.adaln_stage_bytes],
    )?;
    let repeated_attention_and_ffn_weight_bytes = checked_product(
        "scheduled attention and FFN weight materialization",
        &[
            transformer_weight_bytes_per_evaluation_with_adaln_precompute,
            assumptions.evaluation_count,
        ],
    )?;
    let transformer_weight_materialization_bytes_with_adaln_precompute = checked_sum(
        "transformer weight materialization with AdaLN precompute",
        &[
            repeated_attention_and_ffn_weight_bytes,
            adaln_weight_bytes_per_pass,
        ],
    )?;
    let configured_precomputed_evaluation_count = assumptions
        .precompute_adaln_steps
        .min(assumptions.evaluation_count);
    let configured_adaln_passes = if configured_precomputed_evaluation_count == 0 {
        assumptions.evaluation_count
    } else {
        checked_sum(
            "configured AdaLN materialization passes",
            &[
                1,
                assumptions.evaluation_count - configured_precomputed_evaluation_count,
            ],
        )?
    };
    let configured_transformer_weight_materialization_bytes = checked_sum(
        "configured transformer weight materialization",
        &[
            repeated_attention_and_ffn_weight_bytes,
            checked_product(
                "configured AdaLN weight materialization",
                &[adaln_weight_bytes_per_pass, configured_adaln_passes],
            )?,
        ],
    )?;
    let adaln_precompute_saved_weight_materialization_bytes =
        transformer_weight_materialization_bytes_without_adaln_precompute
            .checked_sub(transformer_weight_materialization_bytes_with_adaln_precompute)
            .context("AdaLN precompute weight saving underflow")?;

    Ok(ComputeTrafficEstimate {
        evaluation_count: assumptions.evaluation_count,
        attention_qk_flops_per_evaluation,
        attention_pv_flops_per_evaluation,
        attention_projection_flops_per_evaluation,
        ffn_flops_per_evaluation,
        total_flops_per_evaluation,
        total_schedule_flops,
        transformer_weight_bytes_per_evaluation_without_adaln_precompute,
        transformer_weight_bytes_per_evaluation_with_adaln_precompute,
        transformer_weight_materialization_bytes_without_adaln_precompute,
        transformer_weight_materialization_bytes_with_adaln_precompute,
        configured_transformer_weight_materialization_bytes,
        adaln_precompute_saved_weight_materialization_bytes,
        configured_precomputed_evaluation_count,
    })
}
