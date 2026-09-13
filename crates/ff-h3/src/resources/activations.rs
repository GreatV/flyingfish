use super::{
    ActivationMemoryEstimate, H3_MODALITY_COUNT, ResourceAssumptions, SequenceRows, T2vaGeometry,
    TransformerShape, as_u64, bytes, checked_product, checked_sum,
};
use anyhow::Result;

#[derive(Clone, Copy)]
struct ActivationDimensions {
    hidden: u64,
    attention_heads: u64,
    attention_head_dim: u64,
    attention_width: u64,
    ffn: u64,
    activation_bytes: u64,
    accumulator_bytes: u64,
    hidden_elements: u64,
    hidden_state_bytes: u64,
}

impl ActivationDimensions {
    fn new(
        model: TransformerShape,
        rows: SequenceRows,
        assumptions: ResourceAssumptions,
    ) -> Result<Self> {
        let hidden = as_u64(model.hidden_size, "hidden size")?;
        let attention_heads = as_u64(model.num_attention_heads, "attention head count")?;
        let attention_head_dim = as_u64(model.attention_head_dim, "attention head dimension")?;
        let attention_width =
            checked_product("attention width", &[attention_heads, attention_head_dim])?;
        let ffn = as_u64(model.ffn_dim, "FFN dimension")?;
        let activation_bytes = assumptions.activation_element_bytes;
        let accumulator_bytes = assumptions.accumulator_element_bytes;
        let hidden_elements = checked_product("hidden state elements", &[rows.total, hidden])?;
        let hidden_state_bytes = bytes("hidden state", hidden_elements, activation_bytes)?;

        Ok(Self {
            hidden,
            attention_heads,
            attention_head_dim,
            attention_width,
            ffn,
            activation_bytes,
            accumulator_bytes,
            hidden_elements,
            hidden_state_bytes,
        })
    }
}

#[derive(Clone, Copy)]
struct AttentionActivationEstimate {
    normalization_f32_workspace_bytes: u64,
    modulation_workspace_bytes: u64,
    qkv_projected_bytes: u64,
    qkv_transposed_bytes: u64,
    score_chunk_bytes: u64,
    softmax_workspace_bytes: u64,
    flash_backend_workspace_bytes: u64,
    output_bytes: u64,
    working_set_bytes: u64,
}

#[derive(Clone, Copy)]
struct AttentionBackendBuffers {
    qkv_projected_bytes: u64,
    qkv_transposed_bytes: u64,
    score_chunk_bytes: u64,
    softmax_workspace_bytes: u64,
    flash_backend_workspace_bytes: u64,
}

fn estimate_attention(
    dimensions: ActivationDimensions,
    geometry: T2vaGeometry,
    rows: SequenceRows,
    assumptions: ResourceAssumptions,
) -> Result<AttentionActivationEstimate> {
    let projection_rows = as_u64(
        geometry.attention_projection_chunk_size,
        "attention projection chunk size",
    )?
    .min(rows.total);
    let query_rows = if assumptions.use_flash_attention {
        projection_rows
    } else {
        as_u64(
            geometry.attention_query_chunk_size,
            "attention query chunk size",
        )?
        .min(rows.total)
        .min(projection_rows)
    };
    let key_rows = geometry
        .attention_key_chunk_policy
        .configured_chunk_size()
        .map(|chunk| as_u64(chunk.get(), "attention key chunk size"))
        .transpose()?
        .unwrap_or(rows.total)
        .min(rows.total);
    let projection_hidden_elements = checked_product(
        "attention projection chunk hidden elements",
        &[projection_rows, dimensions.hidden],
    )?;
    let projected_query_elements = checked_product(
        "projected query chunk elements",
        &[projection_rows, dimensions.attention_width],
    )?;
    let reduction_elements = checked_product(
        "attention normalization reduction elements",
        &[projection_rows, 2],
    )?;

    let normalization_f32_workspace_bytes = checked_sum(
        "attention chunk normalization workspace",
        &[
            bytes(
                "attention normalization buffers",
                checked_product(
                    "attention normalization buffers",
                    &[
                        projection_hidden_elements,
                        assumptions.normalization_f32_buffer_count,
                    ],
                )?,
                dimensions.accumulator_bytes,
            )?,
            bytes(
                "attention normalization reductions",
                reduction_elements,
                dimensions.accumulator_bytes,
            )?,
            bytes(
                "Q/K normalization buffers",
                checked_product(
                    "Q/K normalization buffers",
                    &[
                        projected_query_elements,
                        assumptions.normalization_f32_buffer_count,
                    ],
                )?,
                dimensions.accumulator_bytes,
            )?,
            bytes(
                "Q/K normalization reductions",
                reduction_elements,
                dimensions.accumulator_bytes,
            )?,
        ],
    )?;
    let modulation_workspace_bytes = bytes(
        "attention chunk modulation workspace",
        checked_product(
            "attention modulation buffers",
            &[
                projection_hidden_elements,
                assumptions.modulation_buffer_count,
            ],
        )?,
        dimensions.activation_bytes,
    )?;

    let complete_kv_elements = checked_product(
        "complete K/V elements",
        &[2, rows.total, dimensions.attention_width],
    )?;
    let backend = if assumptions.use_flash_attention {
        estimate_flash_attention_buffers(
            dimensions,
            complete_kv_elements,
            projected_query_elements,
            query_rows,
        )?
    } else {
        estimate_exact_attention_buffers(
            dimensions,
            assumptions,
            complete_kv_elements,
            projected_query_elements,
            query_rows,
            key_rows,
            rows.total,
        )?
    };

    let output_bytes = bytes(
        "attention implementation output residency",
        checked_sum(
            "attention implementation output elements",
            &[
                dimensions.hidden_elements,
                dimensions.hidden_elements,
                projected_query_elements,
                projection_hidden_elements,
            ],
        )?,
        dimensions.activation_bytes,
    )?;
    let working_set_bytes = checked_sum(
        "attention working set",
        &[
            dimensions.hidden_state_bytes,
            normalization_f32_workspace_bytes,
            modulation_workspace_bytes,
            backend.qkv_projected_bytes,
            backend.qkv_transposed_bytes,
            backend.score_chunk_bytes,
            backend.softmax_workspace_bytes,
            backend.flash_backend_workspace_bytes,
            output_bytes,
        ],
    )?;

    Ok(AttentionActivationEstimate {
        normalization_f32_workspace_bytes,
        modulation_workspace_bytes,
        qkv_projected_bytes: backend.qkv_projected_bytes,
        qkv_transposed_bytes: backend.qkv_transposed_bytes,
        score_chunk_bytes: backend.score_chunk_bytes,
        softmax_workspace_bytes: backend.softmax_workspace_bytes,
        flash_backend_workspace_bytes: backend.flash_backend_workspace_bytes,
        output_bytes,
        working_set_bytes,
    })
}

fn estimate_flash_attention_buffers(
    dimensions: ActivationDimensions,
    complete_kv_elements: u64,
    projected_query_elements: u64,
    query_rows: u64,
) -> Result<AttentionBackendBuffers> {
    let resident_kv_and_query_elements = checked_sum(
        "resident FlashAttention K/V and query elements",
        &[complete_kv_elements, projected_query_elements],
    )?;
    let qkv_projected_bytes = bytes(
        "complete FlashAttention K/V and query chunk",
        resident_kv_and_query_elements,
        dimensions.activation_bytes,
    )?;
    let backend_workspace_bytes = bytes(
        "FlashAttention log-sum-exp workspace",
        checked_product(
            "FlashAttention log-sum-exp elements",
            &[query_rows, dimensions.attention_heads],
        )?,
        dimensions.accumulator_bytes,
    )?;
    Ok(AttentionBackendBuffers {
        qkv_projected_bytes,
        qkv_transposed_bytes: 0,
        score_chunk_bytes: 0,
        softmax_workspace_bytes: 0,
        flash_backend_workspace_bytes: backend_workspace_bytes,
    })
}

fn estimate_exact_attention_buffers(
    dimensions: ActivationDimensions,
    assumptions: ResourceAssumptions,
    complete_kv_elements: u64,
    projected_query_elements: u64,
    query_rows: u64,
    key_rows: u64,
    total_key_rows: u64,
) -> Result<AttentionBackendBuffers> {
    let qkv_projected_bytes = bytes(
        "K/V/Q projection chunks",
        checked_product("three projection chunks", &[3, projected_query_elements])?,
        dimensions.activation_bytes,
    )?;
    let qkv_transposed_bytes = bytes(
        "prepared K/V and query chunk",
        checked_sum(
            "prepared K/V and query elements",
            &[complete_kv_elements, projected_query_elements],
        )?,
        dimensions.activation_bytes,
    )?;
    let score_elements = checked_product(
        "attention score chunk elements",
        &[dimensions.attention_heads, query_rows, key_rows],
    )?;
    let score_chunk_bytes = bytes(
        "attention score chunk",
        score_elements,
        dimensions.activation_bytes,
    )?;
    let softmax_workspace_bytes = if key_rows == total_key_rows {
        estimate_materialized_softmax_workspace(
            dimensions,
            assumptions,
            score_elements,
            query_rows,
        )?
    } else {
        estimate_online_softmax_workspace(dimensions, score_elements, query_rows, key_rows)?
    };
    Ok(AttentionBackendBuffers {
        qkv_projected_bytes,
        qkv_transposed_bytes,
        score_chunk_bytes,
        softmax_workspace_bytes,
        flash_backend_workspace_bytes: 0,
    })
}

fn estimate_materialized_softmax_workspace(
    dimensions: ActivationDimensions,
    assumptions: ResourceAssumptions,
    score_elements: u64,
    query_rows: u64,
) -> Result<u64> {
    let reduction_elements = checked_product(
        "softmax reduction elements",
        &[dimensions.attention_heads, query_rows],
    )?;
    checked_sum(
        "softmax workspace",
        &[
            bytes(
                "full F32 softmax buffers",
                checked_product(
                    "softmax F32 buffers",
                    &[score_elements, assumptions.softmax_f32_buffer_count],
                )?,
                dimensions.accumulator_bytes,
            )?,
            bytes(
                "softmax reductions",
                checked_product("two softmax reductions", &[2, reduction_elements])?,
                dimensions.accumulator_bytes,
            )?,
            bytes(
                "native softmax result",
                score_elements,
                dimensions.activation_bytes,
            )?,
        ],
    )
}

fn estimate_online_softmax_workspace(
    dimensions: ActivationDimensions,
    score_elements: u64,
    query_rows: u64,
    key_rows: u64,
) -> Result<u64> {
    let key_or_value_tile_elements = checked_product(
        "online attention key/value tile elements",
        &[
            dimensions.attention_heads,
            key_rows,
            dimensions.attention_head_dim,
        ],
    )?;
    let reduction_elements = checked_product(
        "online attention reduction elements",
        &[dimensions.attention_heads, query_rows],
    )?;
    let output_state_elements = checked_product(
        "online attention output state elements",
        &[query_rows, dimensions.attention_width],
    )?;

    checked_sum(
        "online softmax workspace",
        &[
            bytes(
                "online contiguous native key/value tiles",
                checked_product(
                    "two online native key/value tiles",
                    &[2, key_or_value_tile_elements],
                )?,
                dimensions.activation_bytes,
            )?,
            bytes(
                "online F32 value tile",
                key_or_value_tile_elements,
                dimensions.accumulator_bytes,
            )?,
            bytes(
                "online F32 score buffers",
                checked_product("three online F32 score buffers", &[3, score_elements])?,
                dimensions.accumulator_bytes,
            )?,
            bytes(
                "online running and merged reductions",
                checked_product("eight online reduction buffers", &[8, reduction_elements])?,
                dimensions.accumulator_bytes,
            )?,
            bytes(
                "online running and merged outputs",
                checked_product("four online output buffers", &[4, output_state_elements])?,
                dimensions.accumulator_bytes,
            )?,
        ],
    )
}

#[derive(Clone, Copy)]
struct FeedForwardActivationEstimate {
    projected_bytes: u64,
    activated_bytes: u64,
    output_bytes: u64,
    normalization_f32_workspace_bytes: u64,
    modulation_workspace_bytes: u64,
    working_set_bytes: u64,
}

fn estimate_feed_forward(
    dimensions: ActivationDimensions,
    geometry: T2vaGeometry,
    rows: SequenceRows,
    assumptions: ResourceAssumptions,
) -> Result<FeedForwardActivationEstimate> {
    let ffn_rows = as_u64(geometry.ffn_token_chunk_size, "FFN token chunk size")?.min(rows.total);
    let hidden_elements =
        checked_product("FFN chunk hidden elements", &[ffn_rows, dimensions.hidden])?;
    let normalization_f32_workspace_bytes = checked_sum(
        "FFN chunk normalization workspace",
        &[
            bytes(
                "FFN full normalization buffers",
                checked_product(
                    "FFN normalization buffers",
                    &[hidden_elements, assumptions.normalization_f32_buffer_count],
                )?,
                dimensions.accumulator_bytes,
            )?,
            bytes(
                "FFN normalization reductions",
                checked_product("FFN normalization reductions", &[ffn_rows, 2])?,
                dimensions.accumulator_bytes,
            )?,
        ],
    )?;
    let modulation_workspace_bytes = bytes(
        "FFN modulation workspace",
        checked_product(
            "FFN modulation buffers",
            &[hidden_elements, assumptions.modulation_buffer_count],
        )?,
        dimensions.activation_bytes,
    )?;
    let projected_bytes = bytes(
        "FFN projected activation",
        checked_product("FFN projected elements", &[ffn_rows, 2, dimensions.ffn])?,
        dimensions.activation_bytes,
    )?;
    let activated_bytes = bytes(
        "FFN activated values",
        checked_product("FFN activated elements", &[ffn_rows, dimensions.ffn])?,
        dimensions.activation_bytes,
    )?;
    let output_bytes = bytes(
        "FFN chunk output",
        hidden_elements,
        dimensions.activation_bytes,
    )?;
    let working_set_bytes = checked_sum(
        "feed-forward working set",
        &[
            dimensions.hidden_state_bytes,
            normalization_f32_workspace_bytes,
            modulation_workspace_bytes,
            projected_bytes,
            activated_bytes,
            output_bytes,
        ],
    )?;
    Ok(FeedForwardActivationEstimate {
        projected_bytes,
        activated_bytes,
        output_bytes,
        normalization_f32_workspace_bytes,
        modulation_workspace_bytes,
        working_set_bytes,
    })
}

#[derive(Clone, Copy)]
struct OutputHeadChunkEstimate {
    rows: u64,
    hidden_chunk_bytes: u64,
    normalization_f32_bytes: u64,
    modulation_f32_bytes: u64,
}

fn estimate_output_head_chunk(
    rows: u64,
    output_width: u64,
    hidden: u64,
    assumptions: ResourceAssumptions,
) -> Result<OutputHeadChunkEstimate> {
    let hidden_elements = checked_product("output head hidden chunk elements", &[rows, hidden])?;
    let hidden_chunk_bytes = bytes(
        "output head selected hidden chunk",
        hidden_elements,
        assumptions.activation_element_bytes,
    )?;
    let normalization_f32_bytes = checked_sum(
        "output head normalization workspace",
        &[
            bytes(
                "output head normalization buffers",
                checked_product(
                    "output head normalization buffers",
                    &[hidden_elements, assumptions.normalization_f32_buffer_count],
                )?,
                assumptions.accumulator_element_bytes,
            )?,
            bytes(
                "output head normalization reductions",
                checked_product("output head normalization reductions", &[rows, 2])?,
                assumptions.accumulator_element_bytes,
            )?,
        ],
    )?;
    let modulation_f32_bytes = bytes(
        "output head modulation workspace",
        checked_product(
            "output head modulation buffers",
            &[hidden_elements, assumptions.modulation_buffer_count],
        )?,
        assumptions.accumulator_element_bytes,
    )?;
    checked_product("output head projected elements", &[rows, output_width])?;
    Ok(OutputHeadChunkEstimate {
        rows,
        hidden_chunk_bytes,
        normalization_f32_bytes,
        modulation_f32_bytes,
    })
}

#[derive(Clone, Copy)]
struct OutputHeadActivationEstimate {
    chunk_rows: u64,
    hidden_chunk_bytes: u64,
    normalization_f32_workspace_bytes: u64,
    modulation_f32_workspace_bytes: u64,
    modulation_table_bytes: u64,
    projected_bytes: u64,
    working_set_bytes: u64,
}

fn estimate_output_head(
    dimensions: ActivationDimensions,
    model: TransformerShape,
    geometry: T2vaGeometry,
    rows: SequenceRows,
    assumptions: ResourceAssumptions,
) -> Result<OutputHeadActivationEstimate> {
    let modulation_table_bytes = bytes(
        "output head modulation table",
        checked_product(
            "output head modulation table elements",
            &[assumptions.timestep_rows, 2, dimensions.hidden],
        )?,
        dimensions.accumulator_bytes,
    )?;
    let patch_volume = checked_product(
        "output patch volume",
        &[
            as_u64(model.patch_size[0], "temporal patch")?,
            as_u64(model.patch_size[1], "height patch")?,
            as_u64(model.patch_size[2], "width patch")?,
        ],
    )?;
    let video_output_width = checked_product(
        "video output width",
        &[
            as_u64(model.in_channels, "video output channels")?,
            patch_volume,
        ],
    )?;
    let audio_output_width = as_u64(model.audio_in_channels, "audio output width")?;
    let configured_rows = as_u64(geometry.output_token_chunk_size, "output token chunk size")?;
    let chunk_rows = configured_rows.min(rows.total);
    let combined_output_width = checked_sum(
        "combined video/audio output width",
        &[video_output_width, audio_output_width],
    )?;
    let chunk = estimate_output_head_chunk(
        chunk_rows,
        combined_output_width,
        dimensions.hidden,
        assumptions,
    )?;
    let full_packed_projected_bytes = bytes(
        "full-packed output head projections",
        checked_product(
            "full-packed output head projection elements",
            &[rows.total, combined_output_width],
        )?,
        assumptions.io_weight_element_bytes,
    )?;
    let selected_projected_bytes = bytes(
        "selected modality output head projections",
        checked_sum(
            "selected modality output head projection elements",
            &[
                checked_product(
                    "selected video output elements",
                    &[rows.video, video_output_width],
                )?,
                checked_product(
                    "selected audio output elements",
                    &[rows.audio, audio_output_width],
                )?,
            ],
        )?,
        assumptions.io_weight_element_bytes,
    )?;
    let projected_bytes = checked_sum(
        "output head projected residency",
        &[
            checked_product(
                "retained and concatenated full-packed output projections",
                &[2, full_packed_projected_bytes],
            )?,
            selected_projected_bytes,
        ],
    )?;
    let working_set_bytes = checked_sum(
        "output head working set",
        &[
            dimensions.hidden_state_bytes,
            chunk.hidden_chunk_bytes,
            chunk.normalization_f32_bytes,
            chunk.modulation_f32_bytes,
            modulation_table_bytes,
            projected_bytes,
        ],
    )?;
    Ok(OutputHeadActivationEstimate {
        chunk_rows: chunk.rows,
        hidden_chunk_bytes: chunk.hidden_chunk_bytes,
        normalization_f32_workspace_bytes: chunk.normalization_f32_bytes,
        modulation_f32_workspace_bytes: chunk.modulation_f32_bytes,
        modulation_table_bytes,
        projected_bytes,
        working_set_bytes,
    })
}

#[derive(Clone, Copy)]
struct AdalnActivationEstimate {
    modulation_bytes: u64,
    schedule_cache_bytes: u64,
    working_set_bytes: u64,
}

fn estimate_adaln(
    dimensions: ActivationDimensions,
    model: TransformerShape,
    assumptions: ResourceAssumptions,
) -> Result<AdalnActivationEstimate> {
    let modulation_bytes = bytes(
        "AdaLN modulation cache",
        checked_product(
            "AdaLN modulation elements",
            &[
                assumptions.timestep_rows,
                H3_MODALITY_COUNT,
                6,
                dimensions.hidden,
            ],
        )?,
        dimensions.activation_bytes,
    )?;
    let schedule_cache_bytes = checked_product(
        "AdaLN schedule cache",
        &[
            modulation_bytes,
            as_u64(model.num_layers, "transformer layer count")?,
            assumptions.precompute_adaln_steps,
        ],
    )?;
    let working_set_bytes = checked_sum(
        "AdaLN working set",
        &[dimensions.hidden_state_bytes, modulation_bytes],
    )?;
    Ok(AdalnActivationEstimate {
        modulation_bytes,
        schedule_cache_bytes,
        working_set_bytes,
    })
}

#[derive(Clone, Copy)]
struct PersistentActivationEstimate {
    prompt_embedding_bytes: u64,
    refined_text_cache_bytes: u64,
    rotary_cache_bytes: u64,
    packed_layout_bytes: u64,
    latent_state_bytes: u64,
    pipeline_latent_working_set_bytes: u64,
    static_context_bytes: u64,
    persistent_pipeline_bytes: u64,
}

fn estimate_persistent_pipeline(
    dimensions: ActivationDimensions,
    model: TransformerShape,
    geometry: T2vaGeometry,
    rows: SequenceRows,
    assumptions: ResourceAssumptions,
) -> Result<PersistentActivationEstimate> {
    let prompt_embedding_bytes = bytes(
        "prompt embeddings",
        checked_product(
            "prompt embedding elements",
            &[
                rows.text,
                as_u64(model.text_dim, "text embedding dimension")?,
            ],
        )?,
        dimensions.activation_bytes,
    )?;
    let refined_text_cache_bytes = bytes(
        "refined text cache",
        checked_product("refined text elements", &[rows.text, dimensions.hidden])?,
        dimensions.activation_bytes,
    )?;
    let rotary_cache_bytes = bytes(
        "rotary cache",
        checked_product(
            "rotary cache elements",
            &[
                2,
                rows.total,
                6,
                as_u64(model.rope_freq_dim, "RoPE frequency dimension")?,
            ],
        )?,
        dimensions.accumulator_bytes,
    )?;
    let packed_layout_bytes = checked_sum(
        "packed layout tensors",
        &[
            bytes("position IDs", rows.total, 3 * 8)?,
            bytes("token tags", rows.total, 4)?,
            bytes("modality indices", rows.total, 4)?,
            bytes("timestep indices", rows.total, 4)?,
        ],
    )?;
    let video_latent_elements = checked_product(
        "video latent elements",
        &[
            as_u64(model.in_channels, "video latent channels")?,
            as_u64(geometry.latent_frames, "latent frames")?,
            as_u64(geometry.latent_height, "latent height")?,
            as_u64(geometry.latent_width, "latent width")?,
        ],
    )?;
    let audio_latent_elements = checked_product(
        "audio latent elements",
        &[
            as_u64(geometry.audio_channels, "audio channels")?,
            as_u64(model.audio_in_channels, "audio latent channels")?,
            as_u64(geometry.audio_frames, "audio frames")?,
        ],
    )?;
    let latent_state_bytes = bytes(
        "F32 latent state",
        checked_sum(
            "latent state elements",
            &[video_latent_elements, audio_latent_elements],
        )?,
        4,
    )?;
    let pipeline_latent_working_set_bytes = checked_product(
        "pipeline latent working set",
        &[latent_state_bytes, assumptions.pipeline_latent_buffer_count],
    )?;
    let static_context_bytes = checked_sum(
        "static context",
        &[
            prompt_embedding_bytes,
            refined_text_cache_bytes,
            rotary_cache_bytes,
            packed_layout_bytes,
        ],
    )?;
    let persistent_pipeline_bytes = checked_sum(
        "persistent pipeline tensors",
        &[static_context_bytes, pipeline_latent_working_set_bytes],
    )?;
    Ok(PersistentActivationEstimate {
        prompt_embedding_bytes,
        refined_text_cache_bytes,
        rotary_cache_bytes,
        packed_layout_bytes,
        latent_state_bytes,
        pipeline_latent_working_set_bytes,
        static_context_bytes,
        persistent_pipeline_bytes,
    })
}

pub(super) fn estimate_activations(
    model: TransformerShape,
    geometry: T2vaGeometry,
    rows: SequenceRows,
    assumptions: ResourceAssumptions,
) -> Result<ActivationMemoryEstimate> {
    let dimensions = ActivationDimensions::new(model, rows, assumptions)?;
    let attention = estimate_attention(dimensions, geometry, rows, assumptions)?;
    let feed_forward = estimate_feed_forward(dimensions, geometry, rows, assumptions)?;
    let output_head = estimate_output_head(dimensions, model, geometry, rows, assumptions)?;
    let adaln = estimate_adaln(dimensions, model, assumptions)?;
    let persistent = estimate_persistent_pipeline(dimensions, model, geometry, rows, assumptions)?;

    Ok(ActivationMemoryEstimate {
        hidden_state_bytes: dimensions.hidden_state_bytes,
        normalization_f32_workspace_bytes: attention.normalization_f32_workspace_bytes,
        modulation_workspace_bytes: attention.modulation_workspace_bytes,
        qkv_projected_bytes: attention.qkv_projected_bytes,
        qkv_transposed_bytes: attention.qkv_transposed_bytes,
        attention_score_chunk_bytes: attention.score_chunk_bytes,
        attention_softmax_workspace_bytes: attention.softmax_workspace_bytes,
        flash_attention_backend_workspace_bytes: attention.flash_backend_workspace_bytes,
        attention_output_bytes: attention.output_bytes,
        ffn_projected_bytes: feed_forward.projected_bytes,
        ffn_activated_bytes: feed_forward.activated_bytes,
        ffn_output_bytes: feed_forward.output_bytes,
        ffn_normalization_f32_workspace_bytes: feed_forward.normalization_f32_workspace_bytes,
        ffn_modulation_workspace_bytes: feed_forward.modulation_workspace_bytes,
        adaln_modulation_bytes: adaln.modulation_bytes,
        output_head_chunk_rows: output_head.chunk_rows,
        output_head_hidden_chunk_bytes: output_head.hidden_chunk_bytes,
        output_head_normalization_f32_workspace_bytes: output_head
            .normalization_f32_workspace_bytes,
        output_head_modulation_f32_workspace_bytes: output_head.modulation_f32_workspace_bytes,
        output_head_modulation_table_bytes: output_head.modulation_table_bytes,
        output_head_projected_bytes: output_head.projected_bytes,
        output_head_working_set_bytes: output_head.working_set_bytes,
        adaln_schedule_cache_bytes: adaln.schedule_cache_bytes,
        prompt_embedding_bytes: persistent.prompt_embedding_bytes,
        refined_text_cache_bytes: persistent.refined_text_cache_bytes,
        rotary_cache_bytes: persistent.rotary_cache_bytes,
        packed_layout_bytes: persistent.packed_layout_bytes,
        latent_state_bytes: persistent.latent_state_bytes,
        pipeline_latent_working_set_bytes: persistent.pipeline_latent_working_set_bytes,
        static_context_bytes: persistent.static_context_bytes,
        persistent_pipeline_bytes: persistent.persistent_pipeline_bytes,
        attention_working_set_bytes: attention.working_set_bytes,
        feed_forward_working_set_bytes: feed_forward.working_set_bytes,
        adaln_working_set_bytes: adaln.working_set_bytes,
    })
}
