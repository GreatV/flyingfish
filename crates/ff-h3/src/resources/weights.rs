use super::{
    H3_MODALITY_COUNT, ResourceAssumptions, TransformerShape, WeightMemoryEstimate, as_u64, bytes,
    checked_product, checked_sum,
};
use anyhow::Result;

pub(super) fn estimate_weights(
    model: TransformerShape,
    assumptions: ResourceAssumptions,
) -> Result<WeightMemoryEstimate> {
    let hidden = as_u64(model.hidden_size, "hidden size")?;
    let heads = as_u64(model.num_attention_heads, "attention head count")?;
    let head_dim = as_u64(model.attention_head_dim, "attention head dimension")?;
    let attention_width = checked_product("attention width", &[heads, head_dim])?;
    let ffn = as_u64(model.ffn_dim, "FFN dimension")?;
    let time_dim = as_u64(model.time_embed_dim, "time embedding dimension")?;

    let attention_elements = checked_sum(
        "attention stage elements",
        &[
            hidden,
            checked_product("four attention matrices", &[4, hidden, attention_width])?,
            checked_product("Q/K norm elements", &[2, head_dim])?,
        ],
    )?;
    let attention_stage_bytes = bytes(
        "attention stage weights",
        attention_elements,
        assumptions.weight_element_bytes,
    )?;

    let feed_forward_elements = checked_sum(
        "feed-forward stage elements",
        &[
            hidden,
            checked_product("SwiGLU matrices", &[3, hidden, ffn])?,
        ],
    )?;
    let feed_forward_stage_bytes = bytes(
        "feed-forward stage weights",
        feed_forward_elements,
        assumptions.weight_element_bytes,
    )?;

    let adaln_outputs = checked_product("AdaLN output width", &[6, H3_MODALITY_COUNT, hidden])?;
    let adaln_elements = checked_sum(
        "AdaLN stage elements",
        &[
            checked_product("AdaLN matrix", &[adaln_outputs, time_dim])?,
            adaln_outputs,
        ],
    )?;
    let adaln_stage_bytes = bytes(
        "AdaLN stage weights",
        adaln_elements,
        assumptions.weight_element_bytes,
    )?;

    let patch_volume = checked_product(
        "patch volume",
        &[
            as_u64(model.patch_size[0], "temporal patch")?,
            as_u64(model.patch_size[1], "height patch")?,
            as_u64(model.patch_size[2], "width patch")?,
        ],
    )?;
    let video_input = checked_product(
        "video input width",
        &[
            as_u64(model.in_channels, "video input channels")?,
            patch_volume,
        ],
    )?;
    let audio_input = as_u64(model.audio_in_channels, "audio input channels")?;
    let time_hidden = as_u64(model.time_embed_hidden_dim, "time MLP hidden dimension")?;
    let freq = as_u64(model.freq_dim, "frequency dimension")?;
    let latent_input_elements = checked_sum(
        "latent input stage elements",
        &[
            checked_product("video input projection", &[hidden, video_input])?,
            hidden,
            checked_product("audio input projection", &[hidden, audio_input])?,
            hidden,
        ],
    )?;
    let time_input_elements = checked_sum(
        "time input stage elements",
        &[
            checked_product("time MLP input", &[time_hidden, freq])?,
            time_hidden,
            checked_product("time MLP output", &[time_dim, time_hidden])?,
            time_dim,
        ],
    )?;
    let latent_input_stage_bytes = bytes(
        "latent input stage weights",
        latent_input_elements,
        assumptions.io_weight_element_bytes,
    )?;
    let time_input_stage_bytes = bytes(
        "time input stage weights",
        time_input_elements,
        assumptions.io_weight_element_bytes,
    )?;
    let context_elements = checked_sum(
        "context stage elements",
        &[
            checked_product(
                "context projection",
                &[hidden, as_u64(model.text_dim, "text dimension")?],
            )?,
            hidden,
        ],
    )?;
    let context_projection_stage_bytes = bytes(
        "context stage weights",
        context_elements,
        assumptions.weight_element_bytes,
    )?;

    let video_output = video_input;
    let audio_output = audio_input;
    let output_bf16_elements = checked_sum(
        "output BF16 elements",
        &[
            hidden,
            checked_product("output modulation matrix", &[2, hidden, time_dim])?,
            checked_product("output modulation bias", &[2, hidden])?,
        ],
    )?;
    let output_io_elements = checked_sum(
        "output F32 elements",
        &[
            checked_product("video output projection", &[video_output, hidden])?,
            video_output,
            checked_product("audio output projection", &[audio_output, hidden])?,
            audio_output,
        ],
    )?;
    let output_stage_bytes = checked_sum(
        "output stage weights",
        &[
            bytes(
                "output BF16 weights",
                output_bf16_elements,
                assumptions.weight_element_bytes,
            )?,
            bytes(
                "output F32 weights",
                output_io_elements,
                assumptions.io_weight_element_bytes,
            )?,
        ],
    )?;

    let derived_peak_materialized_bytes = [
        context_projection_stage_bytes,
        time_input_stage_bytes,
        latent_input_stage_bytes,
        attention_stage_bytes,
        feed_forward_stage_bytes,
        adaln_stage_bytes,
        output_stage_bytes,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let peak_materialized_bytes = assumptions
        .peak_materialized_weight_bytes_override
        .map_or(derived_peak_materialized_bytes, |observed| {
            observed.max(derived_peak_materialized_bytes)
        });
    Ok(WeightMemoryEstimate {
        checkpoint_bytes: assumptions.checkpoint_weight_bytes,
        context_projection_stage_bytes,
        time_input_stage_bytes,
        latent_input_stage_bytes,
        attention_stage_bytes,
        feed_forward_stage_bytes,
        adaln_stage_bytes,
        output_stage_bytes,
        derived_peak_materialized_bytes,
        peak_materialized_bytes,
    })
}
