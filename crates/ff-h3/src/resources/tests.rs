use super::*;

fn default_estimate(text_rows: usize) -> ResourceEstimate {
    ResourceEstimate::for_shape(
        TransformerShape::h3_base(),
        T2vaGeometry::h3_default(text_rows),
        ResourceAssumptions::h3_bf16_mmap(),
    )
    .unwrap()
}

fn tiny_shape() -> TransformerShape {
    TransformerShape {
        num_layers: 2,
        num_attention_heads: 2,
        attention_head_dim: 3,
        hidden_size: 4,
        ffn_dim: 5,
        in_channels: 1,
        audio_in_channels: 1,
        patch_size: [1, 1, 1],
        text_dim: 1,
        freq_dim: 1,
        time_embed_hidden_dim: 1,
        time_embed_dim: 2,
        rope_freq_dim: 1,
    }
}

fn tiny_geometry() -> T2vaGeometry {
    T2vaGeometry {
        text_rows: 1,
        latent_frames: 1,
        latent_height: 1,
        latent_width: 1,
        audio_frames: 1,
        audio_channels: 1,
        attention_projection_chunk_size: 1,
        attention_query_chunk_size: 1,
        attention_key_chunk_policy: AttentionKeyChunkPolicy::chunked(1).unwrap(),
        ffn_token_chunk_size: 1,
        output_token_chunk_size: 1,
    }
}

#[test]
fn retained_weights_preserve_stage_headroom_and_cpu_host_accounting() {
    for cpu in [false, true] {
        let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
        assumptions.device_memory_is_host = cpu;
        let baseline =
            ResourceEstimate::for_shape(tiny_shape(), tiny_geometry(), assumptions).unwrap();
        assumptions.device_weight_cache_bytes = 4096;
        let retained =
            ResourceEstimate::for_shape(tiny_shape(), tiny_geometry(), assumptions).unwrap();
        assert_eq!(
            retained.peak_device_bytes,
            baseline.peak_device_bytes + 4096
        );
        assert_eq!(
            retained.peak_host_bytes,
            baseline.peak_host_bytes + if cpu { 4096 } else { 0 }
        );
        let budget = ResourceBudget {
            max_host_bytes: None,
            max_device_bytes: Some(baseline.peak_device_bytes),
        };
        assert!(budget.check(&baseline).within_budget);
        assert!(!budget.check(&retained).within_budget);
        assumptions.device_weight_cache_bytes = u64::MAX;
        assert!(ResourceEstimate::for_shape(tiny_shape(), tiny_geometry(), assumptions).is_err());
    }
}

#[test]
fn h3_default_geometry_matches_packed_layout_math() {
    let rows = T2vaGeometry::h3_default(123)
        .sequence_rows([1, 2, 2])
        .unwrap();
    assert_eq!(rows.video_rows_per_frame, 24 * 42);
    assert_eq!(rows.patched_video_frames, 37);
    assert_eq!(rows.video, 37_296);
    assert_eq!(rows.audio, 414);
    assert_eq!(rows.text, 123);
    assert_eq!(rows.total, 37_833);
}

#[test]
fn geometry_json_requires_every_current_chunk_field() {
    let mut value = serde_json::to_value(T2vaGeometry::h3_default(7)).unwrap();
    let object = value.as_object_mut().unwrap();
    object.remove("attention_projection_chunk_size");
    assert!(serde_json::from_value::<T2vaGeometry>(value).is_err());
}

#[test]
fn arbitrary_temporal_and_spatial_patches_are_supported() {
    let geometry = T2vaGeometry {
        text_rows: 5,
        latent_frames: 8,
        latent_height: 12,
        latent_width: 20,
        audio_frames: 7,
        audio_channels: 2,
        attention_projection_chunk_size: 64,
        attention_query_chunk_size: 64,
        attention_key_chunk_policy: AttentionKeyChunkPolicy::chunked(64).unwrap(),
        ffn_token_chunk_size: 16,
        output_token_chunk_size: 16,
    };
    let rows = geometry.sequence_rows([2, 3, 4]).unwrap();
    assert_eq!(rows.patched_video_frames, 4);
    assert_eq!(rows.video_rows_per_frame, 20);
    assert_eq!(rows.video, 80);
    assert_eq!(rows.audio, 14);
    assert_eq!(rows.total, 99);
}

#[test]
fn rejects_zero_and_non_divisible_geometry() {
    let mut geometry = T2vaGeometry::h3_default(1);
    geometry.audio_frames = 0;
    assert!(
        geometry
            .sequence_rows(TransformerShape::h3_base().patch_size)
            .unwrap_err()
            .to_string()
            .contains("audio_frames")
    );

    let mut geometry = T2vaGeometry::h3_default(1);
    geometry.latent_width = 83;
    assert!(
        geometry
            .sequence_rows(TransformerShape::h3_base().patch_size)
            .unwrap_err()
            .to_string()
            .contains("not divisible")
    );

    let mut geometry = T2vaGeometry::h3_default(1);
    geometry.attention_projection_chunk_size = 0;
    assert!(
        geometry
            .sequence_rows(TransformerShape::h3_base().patch_size)
            .unwrap_err()
            .to_string()
            .contains("attention_projection_chunk_size")
    );

    let mut geometry = T2vaGeometry::h3_default(1);
    geometry.output_token_chunk_size = 0;
    assert!(
        geometry
            .sequence_rows(TransformerShape::h3_base().patch_size)
            .unwrap_err()
            .to_string()
            .contains("output_token_chunk_size")
    );
}

#[test]
fn default_stage_weights_match_the_public_checkpoint() {
    let estimate = default_estimate(0);
    assert_eq!(estimate.weights.attention_stage_bytes, 308_292_608);
    assert_eq!(estimate.weights.feed_forward_stage_bytes, 462_432_768);
    assert_eq!(estimate.weights.adaln_stage_bytes, 520_418_304);
    assert_eq!(estimate.weights.time_input_stage_bytes, 63_340_032);
    assert_eq!(estimate.weights.latent_input_stage_bytes, 2_795_520);
    assert_eq!(estimate.weights.output_stage_bytes, 60_588_032);
    assert_eq!(
        estimate.weights.derived_peak_materialized_bytes,
        520_418_304
    );
    assert_eq!(estimate.weights.peak_materialized_bytes, 520_418_304);
    assert_eq!(
        estimate.weights.checkpoint_bytes,
        Some(H3_BASE_CHECKPOINT_BYTES)
    );
}

#[test]
fn compute_and_weight_traffic_follow_the_documented_formulas() {
    let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
    assumptions.evaluation_count = 3;
    assumptions.precompute_adaln_steps = 2;
    let estimate = ResourceEstimate::for_shape(tiny_shape(), tiny_geometry(), assumptions).unwrap();
    let traffic = estimate.compute_and_traffic;

    assert_eq!(traffic.attention_qk_flops_per_evaluation, 216);
    assert_eq!(traffic.attention_pv_flops_per_evaluation, 216);
    assert_eq!(traffic.attention_projection_flops_per_evaluation, 1_152);
    assert_eq!(traffic.ffn_flops_per_evaluation, 720);
    assert_eq!(traffic.total_flops_per_evaluation, 2_304);
    assert_eq!(traffic.total_schedule_flops, 6_912);

    assert_eq!(
        traffic.transformer_weight_bytes_per_evaluation_without_adaln_precompute,
        1_544
    );
    assert_eq!(
        traffic.transformer_weight_bytes_per_evaluation_with_adaln_precompute,
        680
    );
    assert_eq!(
        traffic.transformer_weight_materialization_bytes_without_adaln_precompute,
        4_632
    );
    assert_eq!(
        traffic.transformer_weight_materialization_bytes_with_adaln_precompute,
        2_904
    );
    assert_eq!(
        traffic.configured_transformer_weight_materialization_bytes,
        3_768
    );
    assert_eq!(
        traffic.adaln_precompute_saved_weight_materialization_bytes,
        1_728
    );
    assert_eq!(traffic.configured_precomputed_evaluation_count, 2);
}

#[test]
fn output_head_workspace_retains_both_full_packed_heads_before_selection() {
    let estimate = ResourceEstimate::for_shape(
        tiny_shape(),
        tiny_geometry(),
        ResourceAssumptions::h3_bf16_mmap(),
    )
    .unwrap();
    let activations = estimate.activations;

    assert_eq!(activations.output_head_chunk_rows, 1);
    assert_eq!(activations.output_head_hidden_chunk_bytes, 8);
    assert_eq!(
        activations.output_head_normalization_f32_workspace_bytes,
        56
    );
    assert_eq!(activations.output_head_modulation_f32_workspace_bytes, 80);
    assert_eq!(activations.output_head_modulation_table_bytes, 64);
    assert_eq!(activations.output_head_projected_bytes, 56);
    assert_eq!(activations.output_head_working_set_bytes, 288);

    let mut output_heavy = tiny_shape();
    output_heavy.in_channels = 128;
    let output_peak = ResourceEstimate::for_shape(
        output_heavy,
        tiny_geometry(),
        ResourceAssumptions::h3_bf16_mmap(),
    )
    .unwrap();
    assert_eq!(
        output_peak.peak_device_bytes,
        output_peak.output_stage_peak_device_bytes
    );
    assert!(
        output_peak.output_stage_peak_device_bytes > output_peak.attention_stage_peak_device_bytes
    );
}

#[test]
fn flash_attention_small_geometry_streams_query_and_output_chunks() {
    let assumptions = ResourceAssumptions::h3_bf16_mmap();
    let online = ResourceEstimate::for_shape(tiny_shape(), tiny_geometry(), assumptions).unwrap();
    let mut flash_assumptions = assumptions;
    flash_assumptions.use_flash_attention = true;
    let flash =
        ResourceEstimate::for_shape(tiny_shape(), tiny_geometry(), flash_assumptions).unwrap();

    assert_eq!(flash.activations.qkv_projected_bytes, 84);
    assert_eq!(flash.activations.qkv_transposed_bytes, 0);
    assert_eq!(flash.activations.attention_score_chunk_bytes, 0);
    assert_eq!(flash.activations.attention_softmax_workspace_bytes, 0);
    assert_eq!(flash.activations.flash_attention_backend_workspace_bytes, 8);
    assert_eq!(
        flash.activations.attention_output_bytes,
        (2 * 3 * 4 + 6 + 4) * 2
    );
    assert_eq!(
        online.activations.attention_output_bytes,
        flash.activations.attention_output_bytes
    );
    assert_eq!(flash.activations.attention_working_set_bytes, 360);
    assert_eq!(online.activations.attention_working_set_bytes, 624);
    assert_eq!(flash.compute_and_traffic, online.compute_and_traffic);
}

#[test]
fn attention_output_concat_bound_applies_to_both_backends() {
    let mut geometry = tiny_geometry();
    geometry.attention_projection_chunk_size = 2;
    geometry.attention_query_chunk_size = 1;
    geometry.attention_key_chunk_policy = AttentionKeyChunkPolicy::Full;
    let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
    let full = ResourceEstimate::for_shape(tiny_shape(), geometry, assumptions).unwrap();
    assumptions.use_flash_attention = true;
    let flash = ResourceEstimate::for_shape(tiny_shape(), geometry, assumptions).unwrap();

    let concrete_bytes = (2 * 3 * 4 + 2 * 6 + 2 * 4) * 2;
    assert_eq!(full.activations.attention_output_bytes, concrete_bytes);
    assert_eq!(flash.activations.attention_output_bytes, concrete_bytes);
}

#[test]
fn default_compute_scale_is_reported_for_all_49_evaluations() {
    let estimate = default_estimate(0);
    let traffic = estimate.compute_and_traffic;
    assert_eq!(traffic.evaluation_count, H3_DEFAULT_EVALUATION_COUNT);
    assert_eq!(
        traffic.total_schedule_flops,
        traffic.total_flops_per_evaluation * H3_DEFAULT_EVALUATION_COUNT
    );
    assert_eq!(
        traffic.configured_transformer_weight_materialization_bytes,
        traffic.transformer_weight_materialization_bytes_with_adaln_precompute
    );
    assert!(traffic.attention_qk_flops_per_evaluation > 0);
    assert_eq!(
        traffic.attention_qk_flops_per_evaluation,
        traffic.attention_pv_flops_per_evaluation
    );
    assert!(traffic.total_schedule_flops > traffic.total_flops_per_evaluation);
    assert!(
        traffic.transformer_weight_materialization_bytes_without_adaln_precompute
            > traffic.transformer_weight_materialization_bytes_with_adaln_precompute
    );
}

#[test]
fn reports_required_default_activation_components() {
    let estimate = default_estimate(0);
    assert_eq!(estimate.sequence_rows.total, 37_710);
    assert_eq!(
        estimate.activations.normalization_f32_workspace_bytes,
        4_817_408
    );
    assert_eq!(estimate.activations.modulation_workspace_bytes, 1_720_320);
    assert_eq!(estimate.activations.qkv_projected_bytes, 1_376_256);
    assert_eq!(estimate.activations.qkv_transposed_bytes, 1_081_679_872);
    assert_eq!(
        estimate.activations.attention_score_chunk_bytes,
        135_152_640
    );
    assert_eq!(
        estimate.activations.flash_attention_backend_workspace_bytes,
        0
    );
    assert_eq!(estimate.activations.ffn_projected_bytes, 14_680_064);
    assert_eq!(estimate.activations.ffn_activated_bytes, 7_340_032);
    assert_eq!(estimate.activations.ffn_output_bytes, 2_752_512);
    assert_eq!(estimate.activations.output_head_chunk_rows, 256);
    assert_eq!(
        estimate.activations.output_head_hidden_chunk_bytes,
        2_752_512
    );
    assert_eq!(
        estimate
            .activations
            .output_head_normalization_f32_workspace_bytes,
        16_517_120
    );
    assert_eq!(
        estimate
            .activations
            .output_head_modulation_f32_workspace_bytes,
        27_525_120
    );
    assert_eq!(
        estimate.activations.output_head_modulation_table_bytes,
        86_016
    );
    assert_eq!(estimate.activations.output_head_projected_bytes, 52_989_696);
    assert_eq!(
        estimate.activations.output_head_working_set_bytes,
        505_328_384
    );
    assert_eq!(estimate.activations.adaln_schedule_cache_bytes, 948_326_400);
    assert_eq!(estimate.activations.attention_output_bytes, 811_718_656);
    assert_eq!(
        estimate.activations.attention_working_set_bytes,
        3_658_311_168
    );
    assert_eq!(estimate.attention_stage_peak_device_bytes, 4_040_046_584);
    assert_eq!(estimate.output_stage_peak_device_bytes, 639_359_224);
    assert_eq!(estimate.peak_device_bytes, 4_040_046_584);
    assert!(
        estimate.activations.attention_softmax_workspace_bytes
            > estimate.activations.attention_score_chunk_bytes
    );
    assert!(estimate.peak_device_bytes > estimate.weights.peak_materialized_bytes);
}

#[test]
fn flash_attention_default_geometry_replaces_full_softmax_workspace() {
    let full = default_estimate(0);
    let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
    assumptions.use_flash_attention = true;
    let flash = ResourceEstimate::for_shape(
        TransformerShape::h3_base(),
        T2vaGeometry::h3_default(0),
        assumptions,
    )
    .unwrap();
    let rows = full.sequence_rows.total;
    let attention_width = 56 * 128;

    assert!(!full.assumptions.use_flash_attention);
    assert_eq!(flash.activations.attention_score_chunk_bytes, 0);
    assert_eq!(flash.activations.attention_softmax_workspace_bytes, 0);
    assert_eq!(
        flash.activations.qkv_projected_bytes,
        (2 * rows * attention_width + 32 * attention_width) * 2
    );
    assert_eq!(
        flash.activations.flash_attention_backend_workspace_bytes,
        32 * 56 * 4
    );
    assert_eq!(
        flash.activations.attention_output_bytes,
        (2 * rows * 5_376 + 32 * (attention_width + 5_376)) * 2
    );
    assert_eq!(flash.activations.attention_working_set_bytes, 2_305_401_344);
    assert_eq!(flash.attention_stage_peak_device_bytes, 2_687_136_760);
    assert_eq!(flash.peak_device_bytes, 2_687_136_760);
    assert_eq!(flash.compute_and_traffic, full.compute_and_traffic);
    assert!(
        flash.activations.attention_working_set_bytes
            < full.activations.attention_working_set_bytes
    );
    assert!(flash.peak_device_bytes < full.peak_device_bytes);
    assert!(
        flash
            .to_string()
            .contains("Attention strategy: streamed FlashAttention")
    );
    assert!(
        flash
            .to_string()
            .contains("FlashAttention backend statistics workspace")
    );
    assert!(flash.to_string().contains("output-side residency"));
}

#[test]
fn all_token_chunks_are_clamped_to_their_available_rows() {
    let mut geometry = T2vaGeometry {
        text_rows: 1,
        latent_frames: 1,
        latent_height: 1,
        latent_width: 1,
        audio_frames: 1,
        audio_channels: 1,
        attention_projection_chunk_size: 10_000,
        attention_query_chunk_size: 10_000,
        attention_key_chunk_policy: AttentionKeyChunkPolicy::chunked(10_000).unwrap(),
        ffn_token_chunk_size: 10_000,
        output_token_chunk_size: 10_000,
    };
    let mut shape = TransformerShape::h3_base();
    shape.patch_size = [1, 1, 1];
    let large_chunk =
        ResourceEstimate::for_shape(shape, geometry, ResourceAssumptions::h3_bf16_mmap()).unwrap();
    geometry.attention_query_chunk_size = 3;
    geometry.attention_projection_chunk_size = 3;
    geometry.ffn_token_chunk_size = 3;
    geometry.output_token_chunk_size = 3;
    let exact_chunk =
        ResourceEstimate::for_shape(shape, geometry, ResourceAssumptions::h3_bf16_mmap()).unwrap();
    assert_eq!(large_chunk.sequence_rows.total, 3);
    assert_eq!(
        large_chunk.activations.attention_score_chunk_bytes,
        exact_chunk.activations.attention_score_chunk_bytes
    );
    assert_eq!(
        large_chunk.activations.ffn_projected_bytes,
        exact_chunk.activations.ffn_projected_bytes
    );
    assert_eq!(
        large_chunk.activations.normalization_f32_workspace_bytes,
        exact_chunk.activations.normalization_f32_workspace_bytes
    );
    assert_eq!(
        large_chunk.activations.output_head_working_set_bytes,
        exact_chunk.activations.output_head_working_set_bytes
    );
}

#[test]
fn attention_query_chunk_only_changes_score_and_softmax_workspaces() {
    let chunk_32 = default_estimate(0);
    let mut geometry = T2vaGeometry::h3_default(0);
    geometry.attention_query_chunk_size = 16;
    let chunk_16 = ResourceEstimate::for_shape(
        TransformerShape::h3_base(),
        geometry,
        ResourceAssumptions::h3_bf16_mmap(),
    )
    .unwrap();
    assert_eq!(
        chunk_32.activations.qkv_projected_bytes,
        chunk_16.activations.qkv_projected_bytes
    );
    assert_eq!(
        chunk_32.activations.qkv_transposed_bytes,
        chunk_16.activations.qkv_transposed_bytes
    );
    assert_eq!(
        chunk_32.activations.normalization_f32_workspace_bytes,
        chunk_16.activations.normalization_f32_workspace_bytes
    );
    assert_eq!(
        chunk_32.activations.modulation_workspace_bytes,
        chunk_16.activations.modulation_workspace_bytes
    );
    assert_eq!(
        chunk_32.activations.attention_score_chunk_bytes,
        2 * chunk_16.activations.attention_score_chunk_bytes
    );
    assert!(
        chunk_32.activations.attention_softmax_workspace_bytes
            > chunk_16.activations.attention_softmax_workspace_bytes
    );
    assert_eq!(
        chunk_32.compute_and_traffic, chunk_16.compute_and_traffic,
        "query chunking bounds score memory but must not change attention FLOPs"
    );
}

#[test]
fn attention_projection_chunk_only_changes_projection_workspaces() {
    let mut geometry = T2vaGeometry::h3_default(0);
    geometry.attention_projection_chunk_size = 1_024;
    let projection_1024 = ResourceEstimate::for_shape(
        TransformerShape::h3_base(),
        geometry,
        ResourceAssumptions::h3_bf16_mmap(),
    )
    .unwrap();
    geometry.attention_projection_chunk_size = 512;
    let projection_512 = ResourceEstimate::for_shape(
        TransformerShape::h3_base(),
        geometry,
        ResourceAssumptions::h3_bf16_mmap(),
    )
    .unwrap();
    assert_eq!(
        projection_1024.activations.qkv_projected_bytes
            - projection_512.activations.qkv_projected_bytes,
        3 * 512 * 7_168 * 2
    );
    assert_eq!(
        projection_1024.activations.qkv_transposed_bytes
            - projection_512.activations.qkv_transposed_bytes,
        512 * 7_168 * 2
    );
    assert_eq!(
        projection_1024
            .activations
            .normalization_f32_workspace_bytes,
        2 * projection_512.activations.normalization_f32_workspace_bytes
    );
    assert_eq!(
        projection_1024.activations.modulation_workspace_bytes,
        2 * projection_512.activations.modulation_workspace_bytes
    );
    assert_eq!(
        projection_1024.activations.attention_score_chunk_bytes,
        projection_512.activations.attention_score_chunk_bytes
    );
    assert_eq!(
        projection_1024
            .activations
            .attention_softmax_workspace_bytes,
        projection_512.activations.attention_softmax_workspace_bytes
    );
    assert_eq!(
        projection_1024.compute_and_traffic, projection_512.compute_and_traffic,
        "projection chunking changes residency but must not change FLOPs"
    );
}

#[test]
fn output_chunk_only_changes_output_head_workspaces() {
    let output_256 = default_estimate(0);
    let mut geometry = T2vaGeometry::h3_default(0);
    geometry.output_token_chunk_size = 128;
    let output_128 = ResourceEstimate::for_shape(
        TransformerShape::h3_base(),
        geometry,
        ResourceAssumptions::h3_bf16_mmap(),
    )
    .unwrap();
    assert_eq!(
        output_256.activations.output_head_hidden_chunk_bytes,
        2 * output_128.activations.output_head_hidden_chunk_bytes
    );
    assert_eq!(
        output_256
            .activations
            .output_head_normalization_f32_workspace_bytes,
        2 * output_128
            .activations
            .output_head_normalization_f32_workspace_bytes
    );
    assert_eq!(
        output_256
            .activations
            .output_head_modulation_f32_workspace_bytes,
        2 * output_128
            .activations
            .output_head_modulation_f32_workspace_bytes
    );
    assert_eq!(
        output_256.activations.output_head_projected_bytes,
        output_128.activations.output_head_projected_bytes
    );
    assert_eq!(
        output_256.activations.output_head_modulation_table_bytes,
        output_128.activations.output_head_modulation_table_bytes
    );
    assert_eq!(
        output_256.activations.attention_working_set_bytes,
        output_128.activations.attention_working_set_bytes
    );
    assert_eq!(
        output_256.activations.feed_forward_working_set_bytes,
        output_128.activations.feed_forward_working_set_bytes
    );
    assert!(output_256.output_stage_peak_device_bytes > output_128.output_stage_peak_device_bytes);
    assert_eq!(
        output_256.compute_and_traffic,
        output_128.compute_and_traffic
    );
}

#[test]
fn attention_key_chunk_only_changes_online_softmax_workspace() {
    let mut geometry = T2vaGeometry::h3_default(0);
    geometry.attention_key_chunk_policy = AttentionKeyChunkPolicy::chunked(4096).unwrap();
    let key_4096 = ResourceEstimate::for_shape(
        TransformerShape::h3_base(),
        geometry,
        ResourceAssumptions::h3_bf16_mmap(),
    )
    .unwrap();
    geometry.attention_key_chunk_policy = AttentionKeyChunkPolicy::chunked(2048).unwrap();
    let key_2048 = ResourceEstimate::for_shape(
        TransformerShape::h3_base(),
        geometry,
        ResourceAssumptions::h3_bf16_mmap(),
    )
    .unwrap();
    assert_eq!(
        key_4096.activations.attention_score_chunk_bytes,
        2 * key_2048.activations.attention_score_chunk_bytes
    );
    assert!(
        key_4096.activations.attention_softmax_workspace_bytes
            > key_2048.activations.attention_softmax_workspace_bytes
    );
    assert_eq!(
        key_4096.activations.qkv_transposed_bytes,
        key_2048.activations.qkv_transposed_bytes
    );
    assert_eq!(
        key_4096.activations.normalization_f32_workspace_bytes,
        key_2048.activations.normalization_f32_workspace_bytes
    );
    assert_eq!(
        key_4096.compute_and_traffic, key_2048.compute_and_traffic,
        "key chunking tiles exact attention but must not change FLOPs"
    );
}

#[test]
fn online_softmax_accounts_for_tiles_and_running_merge_state() {
    let mut geometry = T2vaGeometry::h3_default(0);
    geometry.attention_projection_chunk_size = 1;
    geometry.attention_query_chunk_size = 1;
    geometry.attention_key_chunk_policy = AttentionKeyChunkPolicy::chunked(1_024).unwrap();
    let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
    assumptions.backend_workspace_bytes = 1_536 << 20;
    let estimate =
        ResourceEstimate::for_shape(TransformerShape::h3_base(), geometry, assumptions).unwrap();

    assert_eq!(
        estimate.activations.attention_softmax_workspace_bytes,
        59_524_864
    );
    assert_eq!(
        estimate.activations.attention_score_chunk_bytes,
        56 * 1_024 * 2
    );
    let four_gib = 4 << 30;
    assert!(estimate.peak_device_bytes > four_gib);
    assert!(
        !ResourceBudget {
            max_host_bytes: None,
            max_device_bytes: Some(four_gib),
        }
        .check(&estimate)
        .within_budget,
        "the previously omitted online buffers must not produce a false-positive 4 GiB plan"
    );
}

#[test]
fn online_softmax_workspace_is_monotonic_in_query_and_key_chunks() {
    let mut geometry = T2vaGeometry::h3_default(0);
    geometry.attention_projection_chunk_size = 2;
    geometry.attention_query_chunk_size = 1;
    geometry.attention_key_chunk_policy = AttentionKeyChunkPolicy::chunked(1_024).unwrap();
    let assumptions = ResourceAssumptions::h3_bf16_mmap();
    let query_1_key_1024 =
        ResourceEstimate::for_shape(TransformerShape::h3_base(), geometry, assumptions).unwrap();

    geometry.attention_query_chunk_size = 2;
    let query_2_key_1024 =
        ResourceEstimate::for_shape(TransformerShape::h3_base(), geometry, assumptions).unwrap();
    assert!(
        query_2_key_1024
            .activations
            .attention_softmax_workspace_bytes
            > query_1_key_1024
                .activations
                .attention_softmax_workspace_bytes
    );

    geometry.attention_key_chunk_policy = AttentionKeyChunkPolicy::chunked(2_048).unwrap();
    let query_2_key_2048 =
        ResourceEstimate::for_shape(TransformerShape::h3_base(), geometry, assumptions).unwrap();
    assert!(
        query_2_key_2048
            .activations
            .attention_softmax_workspace_bytes
            > query_2_key_1024
                .activations
                .attention_softmax_workspace_bytes
    );
}

#[test]
fn online_softmax_workspace_overflow_is_rejected() {
    let rows = 1_500_000_000usize;
    let shape = TransformerShape {
        num_layers: 1,
        num_attention_heads: 1,
        attention_head_dim: 1,
        hidden_size: 1,
        ffn_dim: 1,
        in_channels: 1,
        audio_in_channels: 1,
        patch_size: [1, 1, 1],
        text_dim: 1,
        freq_dim: 1,
        time_embed_hidden_dim: 1,
        time_embed_dim: 1,
        rope_freq_dim: 1,
    };
    let geometry = T2vaGeometry {
        text_rows: rows,
        latent_frames: 1,
        latent_height: 1,
        latent_width: 1,
        audio_frames: 1,
        audio_channels: 1,
        attention_projection_chunk_size: rows,
        attention_query_chunk_size: rows,
        attention_key_chunk_policy: AttentionKeyChunkPolicy::chunked(rows - 1).unwrap(),
        ffn_token_chunk_size: 1,
        output_token_chunk_size: 1,
    };
    let error = ResourceEstimate::for_shape(shape, geometry, ResourceAssumptions::h3_bf16_mmap())
        .unwrap_err();
    assert!(
        error.to_string().contains("online F32 score buffers")
            && error.to_string().contains("overflow"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn disabling_adaln_precompute_removes_host_schedule_cache() {
    let enabled = default_estimate(1);
    let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
    assumptions.precompute_adaln_steps = 0;
    let disabled = ResourceEstimate::for_shape(
        TransformerShape::h3_base(),
        T2vaGeometry::h3_default(1),
        assumptions,
    )
    .unwrap();
    assert_eq!(disabled.activations.adaln_schedule_cache_bytes, 0);
    assert_eq!(
        enabled.peak_host_bytes - disabled.peak_host_bytes,
        enabled.activations.adaln_schedule_cache_bytes
    );
    assert_eq!(
        disabled
            .compute_and_traffic
            .configured_transformer_weight_materialization_bytes,
        disabled
            .compute_and_traffic
            .transformer_weight_materialization_bytes_without_adaln_precompute
    );
    assert_eq!(
        enabled
            .compute_and_traffic
            .transformer_weight_materialization_bytes_with_adaln_precompute,
        disabled
            .compute_and_traffic
            .transformer_weight_materialization_bytes_with_adaln_precompute
    );
}

#[test]
fn execution_plan_peak_can_raise_but_not_lower_derived_peak() {
    let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
    assumptions.peak_materialized_weight_bytes_override = Some(600_000_000);
    let raised = ResourceEstimate::for_shape(
        TransformerShape::h3_base(),
        T2vaGeometry::h3_default(1),
        assumptions,
    )
    .unwrap();
    assert_eq!(raised.weights.peak_materialized_bytes, 600_000_000);

    assumptions.peak_materialized_weight_bytes_override = Some(1);
    let not_lowered = ResourceEstimate::for_shape(
        TransformerShape::h3_base(),
        T2vaGeometry::h3_default(1),
        assumptions,
    )
    .unwrap();
    assert_eq!(
        not_lowered.weights.peak_materialized_bytes,
        not_lowered.weights.derived_peak_materialized_bytes
    );
}

#[test]
fn checks_both_optional_budgets() {
    let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
    assumptions.host_weight_cache_bytes = 1_000_000;
    let estimate = ResourceEstimate::for_shape(
        TransformerShape::h3_base(),
        T2vaGeometry::h3_default(4),
        assumptions,
    )
    .unwrap();
    let exact = ResourceBudget {
        max_host_bytes: Some(estimate.peak_host_bytes),
        max_device_bytes: Some(estimate.peak_device_bytes),
    };
    assert!(exact.check(&estimate).within_budget);
    exact.validate(&estimate).unwrap();

    let too_small = ResourceBudget {
        max_host_bytes: Some(estimate.peak_host_bytes - 7),
        max_device_bytes: Some(estimate.peak_device_bytes - 11),
    };
    let report = too_small.check(&estimate);
    assert!(!report.within_budget);
    assert_eq!(report.violations.len(), 2);
    assert_eq!(report.violations[0].domain, ResourceDomain::Host);
    assert_eq!(report.violations[0].excess_bytes, 7);
    assert_eq!(report.violations[1].domain, ResourceDomain::Device);
    assert_eq!(report.violations[1].excess_bytes, 11);
    assert!(
        too_small
            .validate(&estimate)
            .unwrap_err()
            .to_string()
            .contains("device")
    );
}

#[test]
fn absent_limits_always_pass() {
    let estimate = default_estimate(1);
    assert!(ResourceBudget::default().check(&estimate).within_budget);
}

#[test]
fn cpu_execution_memory_is_charged_to_the_host_budget() {
    let gpu = default_estimate(1);
    let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
    assumptions.device_memory_is_host = true;
    let cpu = ResourceEstimate::for_shape(
        TransformerShape::h3_base(),
        T2vaGeometry::h3_default(1),
        assumptions,
    )
    .unwrap();
    assert_eq!(cpu.peak_device_bytes, gpu.peak_device_bytes);
    assert_eq!(
        cpu.peak_host_bytes,
        gpu.peak_host_bytes + gpu.peak_device_bytes
    );
    assert!(cpu.to_string().contains("CPU host peak"));
}

#[test]
fn estimate_round_trips_as_json() {
    let estimate = default_estimate(9);
    let json = estimate.to_pretty_json().unwrap();
    assert_eq!(estimate.schema_version, RESOURCE_ESTIMATE_SCHEMA_VERSION);
    assert!(json.contains("\"schema_version\": 1"));
    assert!(json.contains("\"attention_projection_chunk_size\": 32"));
    assert!(json.contains("\"attention_key_chunk_policy\": \"full\""));
    assert!(json.contains("\"output_token_chunk_size\": 256"));
    assert!(json.contains("\"qkv_projected_bytes\""));
    assert!(json.contains("\"output_head_working_set_bytes\""));
    assert!(json.contains("\"output_stage_peak_device_bytes\""));
    assert!(json.contains("\"peak_device_bytes\""));
    assert!(json.contains("\"total_schedule_flops\""));
    assert!(json.contains("\"use_flash_attention\": false"));
    assert!(json.contains("\"mapped_weight_residency_bytes\": 0"));
    assert!(json.contains("\"flash_attention_backend_workspace_bytes\""));
    assert!(json.contains("\"transformer_weight_materialization_bytes_with_adaln_precompute\""));
    let decoded = ResourceEstimate::from_json(json.as_bytes()).unwrap();
    assert_eq!(decoded, estimate);

    let mut stale = serde_json::to_value(&estimate).unwrap();
    stale["schema_version"] = serde_json::json!(2);
    assert!(ResourceEstimate::from_json(&serde_json::to_vec(&stale).unwrap()).is_err());

    let mut unknown = serde_json::to_value(&estimate).unwrap();
    unknown["geometry"]["unexpected"] = serde_json::json!(true);
    assert!(ResourceEstimate::from_json(&serde_json::to_vec(&unknown).unwrap()).is_err());
}

#[test]
fn human_report_names_online_softmax() {
    let mut geometry = T2vaGeometry::h3_default(2);
    geometry.attention_key_chunk_policy = AttentionKeyChunkPolicy::chunked(64).unwrap();
    let estimate = ResourceEstimate::for_shape(
        TransformerShape::h3_base(),
        geometry,
        ResourceAssumptions::h3_bf16_mmap(),
    )
    .unwrap();
    assert!(
        estimate
            .to_string()
            .contains("Attention strategy: online key-chunked softmax")
    );
}

#[test]
fn human_report_contains_decision_fields() {
    let report = default_estimate(2).to_string();
    assert!(report.contains("T2VA rows"));
    assert!(report.contains("Attention projection/query chunks"));
    assert!(report.contains("key chunk: full"));
    assert!(report.contains("Attention strategy: materialized full softmax"));
    assert!(report.contains("Chunked K/V/Q projections"));
    assert!(report.contains("FFN projected/activated"));
    assert!(report.contains("Output head configured/effective chunk"));
    assert!(report.contains("Modeled FLOPs/evaluation"));
    assert!(report.contains("full-attention schedule"));
    assert!(report.contains("weight materialization"));
    assert!(report.contains("Conservative peak host/device"));
    assert!(report.contains("GiB"));
}

#[test]
fn detects_arithmetic_overflow() {
    let geometry = T2vaGeometry {
        text_rows: usize::MAX,
        ..T2vaGeometry::h3_default(1)
    };
    assert!(
        ResourceEstimate::for_shape(
            TransformerShape::h3_base(),
            geometry,
            ResourceAssumptions::h3_bf16_mmap(),
        )
        .unwrap_err()
        .to_string()
        .contains("overflow")
    );

    let baseline = default_estimate(1);
    let rows = SequenceRows {
        text: u64::MAX,
        audio: 0,
        video: 0,
        total: u64::MAX,
        patched_video_frames: 0,
        video_rows_per_frame: 0,
    };
    assert!(
        estimate_compute_and_traffic(baseline.model, rows, baseline.weights, baseline.assumptions,)
            .unwrap_err()
            .to_string()
            .contains("FLOPs")
    );
}

#[test]
fn validates_conservative_assumptions() {
    let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
    assumptions.accumulator_element_bytes = 1;
    assert!(
        ResourceEstimate::for_shape(
            TransformerShape::h3_base(),
            T2vaGeometry::h3_default(1),
            assumptions,
        )
        .unwrap_err()
        .to_string()
        .contains("narrower")
    );

    let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
    assumptions.evaluation_count = 0;
    assert!(
        ResourceEstimate::for_shape(
            TransformerShape::h3_base(),
            T2vaGeometry::h3_default(1),
            assumptions,
        )
        .unwrap_err()
        .to_string()
        .contains("evaluation_count")
    );
}

#[test]
fn byte_formatter_uses_binary_units() {
    assert_eq!(format_bytes(17), "17 B");
    assert_eq!(format_bytes(1_536), "1.50 KiB");
    assert_eq!(format_bytes(1 << 30), "1.00 GiB");
}

#[test]
fn flop_formatter_uses_decimal_units() {
    assert_eq!(format_flops(17), "17 FLOPs");
    assert_eq!(format_flops(1_500), "1.50 KFLOPs");
    assert_eq!(format_flops(1_000_000_000_000_000), "1.00 PFLOPs");
}

#[test]
fn conditioned_rows_and_timestep_tables_are_charged_without_changing_t2va_defaults() {
    let model = tiny_shape();
    let geometry = tiny_geometry();
    let baseline = ResourceAssumptions::h3_bf16_mmap();
    let original = ResourceEstimate::for_shape(model, geometry, baseline).unwrap();
    let encoded = serde_json::to_value(baseline).unwrap();
    assert!(encoded.get("timestep_rows").is_none());
    assert_eq!(
        serde_json::from_value::<ResourceAssumptions>(encoded)
            .unwrap()
            .timestep_rows,
        2
    );
    let mut rows = geometry.sequence_rows(model.patch_size).unwrap();
    rows.video += 10;
    rows.audio += 4;
    rows.total += 14;
    let mut conditioned = baseline;
    conditioned.timestep_rows = 4;
    let estimate = ResourceEstimate::for_shape_rows(model, geometry, conditioned, rows).unwrap();
    assert_eq!(
        estimate.sequence_rows.total,
        original.sequence_rows.total + 14
    );
    assert_eq!(
        estimate.activations.adaln_schedule_cache_bytes,
        original.activations.adaln_schedule_cache_bytes * 2
    );
    assert!(estimate.peak_device_bytes > original.peak_device_bytes);
    rows.total += 1;
    assert!(ResourceEstimate::for_shape_rows(model, geometry, conditioned, rows).is_err());
}

#[test]
fn t2va_requirement_exposes_the_deriver_fields() {
    let estimate = default_estimate(1);
    let requirement =
        H3T2vaRequirement::from_estimate(&estimate, 63_000_000_000, true, 0, None).unwrap();
    use ff_core::configure::ModelRequirement as _;
    let traffic = &estimate.compute_and_traffic;
    assert_eq!(
        requirement.steady_weight_bytes().unwrap(),
        traffic.transformer_weight_bytes_per_evaluation_with_adaln_precompute
    );
    assert_eq!(
        requirement.single_pass_weight_bytes().unwrap(),
        traffic.transformer_weight_bytes_per_evaluation_without_adaln_precompute
            - traffic.transformer_weight_bytes_per_evaluation_with_adaln_precompute
            + 63_000_000_000
    );
    assert!(
        requirement.activation_peak_bytes().unwrap() > 0,
        "the flash-mode activation peak must be positive"
    );
    assert_eq!(
        requirement.flops_per_evaluation().unwrap(),
        traffic.total_flops_per_evaluation
    );
}

fn machine_profile(host_bytes: u64, device_bytes: u64) -> ff_core::topology::TopologyProfile {
    use ff_core::probe::{DeviceBackend, HardwareFingerprint};
    use ff_core::topology::{InterconnectLevel, TopologyDevice};
    ff_core::topology::TopologyProfile {
        schema_version: ff_core::topology::TOPOLOGY_PROFILE_SCHEMA_VERSION,
        fingerprint: HardwareFingerprint::collect(&candle_core::Device::Cpu),
        host_memory_total_bytes: Some(host_bytes),
        cgroup_memory_limit_bytes: None,
        devices: vec![TopologyDevice {
            ordinal: 0,
            backend: DeviceBackend::Cuda,
            name: Some("fixture".to_owned()),
            total_memory_bytes: Some(device_bytes),
            compute_capability: None,
        }],
        interconnect: InterconnectLevel::SingleDevice,
        storage_bytes_per_second: None,
    }
}

#[test]
fn derivation_reproduces_the_measured_machine_configurations() {
    use ff_core::configure::{self, WeightSourceChoice};
    let gib = 1u64 << 30;
    let estimate = default_estimate(357);
    let requirement = H3T2vaRequirement::from_estimate(&estimate, 0, true, 0, None).unwrap();

    let rtx4090 =
        configure::derive(0, &machine_profile(67_200_000_000, 24 * gib), &requirement).unwrap();
    assert_eq!(rtx4090.weight_source, WeightSourceChoice::Memory);
    assert!(
        rtx4090.host_cache_ceiling_bytes.unwrap() > 60 * gib,
        "the cache ceiling must cover the full transformer materialization"
    );
    assert_eq!(
        rtx4090.chunks,
        Some(configure::ChunkPlan {
            attention_projection: 4096,
            feed_forward: 1024,
            output: 1024
        })
    );

    let blackwell =
        configure::derive(0, &machine_profile(1007 * gib, 96 * gib), &requirement).unwrap();
    assert_eq!(blackwell.weight_source, WeightSourceChoice::Memory);
    assert_eq!(blackwell.chunks, rtx4090.chunks);

    let dual_a4000_host =
        configure::derive(0, &machine_profile(30_000_000_000, 16 * gib), &requirement).unwrap();
    assert_eq!(dual_a4000_host.weight_source, WeightSourceChoice::Mmap);
    assert_eq!(dual_a4000_host.host_cache_ceiling_bytes, None);

    let knife_edge_host =
        configure::derive(0, &machine_profile(62 * gib, 24 * gib), &requirement).unwrap();
    assert_eq!(knife_edge_host.weight_source, WeightSourceChoice::Mmap);
    assert_eq!(dual_a4000_host.chunks, rtx4090.chunks);
}

#[test]
fn scratch_host_overhead_numbers() {
    let estimate = default_estimate(357);
    println!(
        "peak_host={:.3} GiB workspace={:.3} GiB static_ctx={:.3} persistent={:.3}",
        estimate.peak_host_bytes as f64 / 2f64.powi(30),
        estimate.host_runtime_workspace_bytes as f64 / 2f64.powi(30),
        estimate.activations.static_context_bytes as f64 / 2f64.powi(30),
        estimate.activations.persistent_pipeline_bytes as f64 / 2f64.powi(30)
    );
}
