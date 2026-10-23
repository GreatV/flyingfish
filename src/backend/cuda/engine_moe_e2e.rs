//! Published-checkpoint text, media and teacher-forced decode acceptance gate.
use super::*;
use crate::vision_input::{ImageGrid, MultimodalInput};
use safetensors::{Dtype, SafeTensors};

fn placement_compare(
    label: &str,
    a: &[f32],
    b: &[f32],
    bf16_ulp_limit: Option<u16>,
) -> Result<usize> {
    ensure!(
        !a.is_empty() && a.len() == b.len(),
        "Runner FAIL {label} extent"
    );
    ensure!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "Runner FAIL {label} nonfinite"
    );
    let mismatches = a
        .iter()
        .zip(b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let first = a
        .iter()
        .zip(b)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let ordered = |value: f32| {
        let bits = bf16::from_f32(value).to_bits();
        if bits & 0x8000 == 0 {
            bits | 0x8000
        } else {
            !bits
        }
    };
    let max_bf16_ulp = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| ordered(a).abs_diff(ordered(b)))
        .max()
        .context("empty comparison")?;
    let passed = match bf16_ulp_limit {
        Some(limit) => max_bf16_ulp <= limit,
        None => mismatches == 0,
    };
    println!(
        "{} placement_comparison {}",
        if passed { "PASS" } else { "DEFECT" },
        serde_json::json!({
            "label":label,"elements":a.len(),"bit_mismatches":mismatches,"first":first,
            "first_a":first.map(|i|a[i]),"first_b":first.map(|i|b[i]),"max_abs":max_abs,"max_bf16_ulp":max_bf16_ulp,
            "strict_bits":bf16_ulp_limit.is_none(),"bf16_ulp_limit":bf16_ulp_limit,"passed":passed
        })
    );
    Ok(usize::from(!passed))
}

fn placement_graph(
    stream: &Arc<CudaStream>,
    launch: impl FnOnce() -> Result<()>,
) -> Result<CudaGraph> {
    stream.synchronize()?;
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let launched = launch();
    let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
    launched?;
    let graph = captured?.context("Runner FAIL empty placement graph")?;
    graph.upload()?;
    stream.synchronize()?;
    Ok(graph)
}

fn placement_components(engine: &Engine, model: &Path) -> Result<usize> {
    use crate::experts::{
        catalogue::ExpertCatalogue,
        disk::{DiskExperts, DiskReader},
    };
    let device = &engine.device;
    let stream = &device.stream;
    let geometry = ExpertGeometry::from_model(&engine.model_spec)?;
    ensure!(
        geometry.hidden == 2048
            && geometry.intermediate == 512
            && geometry.top_k == 8
            && geometry.experts == 256,
        "Runner FAIL placement geometry {geometry:?}"
    );
    let format = engine
        .model_spec
        .expert_format
        .context("Runner FAIL placement expert format missing")?;
    let manifest = crate::checkpoint::MappedCheckpoint::open(model)?
        .manifest(&engine.model_spec, crate::checkpoint::LoadPolicy::TextOnly)?;
    let catalogue = Arc::new(ExpertCatalogue::from_manifest(&manifest, geometry, format)?);
    let reader = DiskReader::open(&catalogue, model, 1)?;
    let elements = usize::try_from(catalogue.slot_bytes() / 2)?;
    let host = HostBlocks::new(1, elements)?;
    let mut gpu_weights = Vec::new();
    for expert in 0..8 {
        let index = reader.read_expert(0, expert, &catalogue)?;
        let lease = reader.values(index)?;
        if expert == 0 {
            host.claim(0)?.slice()?.copy_from_slice(&lease);
        }
        gpu_weights.push(stream.clone_htod(lease.values())?);
    }
    let mut gate_ptrs = vec![0u64; 256];
    let mut down_ptrs = vec![0u64; 256];
    let mut valid = vec![0u32; 256];
    let mut generations = vec![0u64; 256];
    for (expert, weights) in gpu_weights.iter().enumerate() {
        gate_ptrs[expert] = weights.device_ptr(stream).0;
        down_ptrs[expert] = gate_ptrs[expert] + catalogue.layout().down_offset();
        valid[expert] = 1;
        generations[expert] = 1;
    }
    let gates = stream.clone_htod(&gate_ptrs)?;
    let downs = stream.clone_htod(&down_ptrs)?;
    let valid = stream.clone_htod(&valid)?;
    let generations = stream.clone_htod(&generations)?;
    let table = SlotBindings {
        gate_up: view(&gates, stream, 0, [256], [1])?,
        down: view(&downs, stream, 0, [256], [1])?,
        valid: view(&valid, stream, 0, [256], [1])?,
        generation: view(&generations, stream, 0, [256], [1])?,
    };
    let input = (0..2048)
        .map(|index| bf16::from_f32(((index * 37 % 131) as f32 - 65.0) / 128.0))
        .collect::<Vec<_>>();
    let x_buffer = stream.clone_htod(&input)?;
    let y = stream.alloc_zeros::<bf16>(2048)?;
    let x = qwen::matrix(&x_buffer, stream, 0, 1, 2048)?;
    let mut y_view = TensorMut(qwen::matrix(&y, stream, 0, 1, 2048)?);
    let ffn = engine.moe_ffn_binding(0)?;
    let mut decode = Decode::new(device, geometry, 1, 1, (1, 8), format)?;
