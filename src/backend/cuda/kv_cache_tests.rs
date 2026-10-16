use super::state::{CacheView, StateCheckpoint, StateStore};
use super::{
    Device, cubin,
    flash::{Flash, Shape},
    ops::flat,
};
use crate::{kv::KvFormat, op_contract::AttentionWorkspace, prefix_state::sha256};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg};
use half::bf16;
use serde_json::{Value, json};
use std::path::PathBuf;

fn input() -> Result<(PathBuf, PathBuf, Value)> {
    let root = PathBuf::from(
        std::env::var_os("FF_Q38_MTP_REF_ROOT").context("FF_Q38_MTP_REF_ROOT missing")?,
    );
    let out =
        PathBuf::from(std::env::var_os("FF_Q38_MTP_OUTPUT").context("FF_Q38_MTP_OUTPUT missing")?);
    let manifest: Value = serde_json::from_slice(&std::fs::read(root.join("FIXTURES.json"))?)?;
    ensure!(
        manifest["capacity"] == 16640 && manifest["shape"] == json!([4, 16640, 256]),
        "KV fixture geometry"
    );
    std::fs::create_dir_all(&out)?;
    Ok((root, out, manifest))
}
#[test]
fn compact_binding_rejects_wrong_geometry_and_alignment() -> Result<()> {
    assert_eq!(std::mem::size_of::<CacheView>(), 64);
    let view = CacheView {
        k_code: 256,
        v_code: 512,
        k_stats: 768,
        v_stats: 1024,
        k_ring: 1280,
        v_ring: 1536,
        cursor: 1792,
        capacity: 16640,
        ring: 2243,
    };
    for rows in 1..=5 {
        view.validate(16640, rows)?;
    }
    assert!(CacheView { ring: 2242, ..view }.validate(16640, 4).is_err());
    assert!(
        CacheView {
            k_stats: 769,
            ..view
        }
        .validate(16640, 4)
        .is_err()
    );
    assert!(view.validate(16640, 0).is_err());
    assert!(view.validate(16640, 6).is_err());
    Ok(())
}
fn bf16_file(path: PathBuf, elements: usize) -> Result<Vec<bf16>> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() == elements * 2, "KV fixture extent");
    let data = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]])))
        .collect::<Vec<_>>();
    ensure!(data.iter().all(|x| x.is_finite()), "nonfinite KV fixture");
    Ok(data)
}
fn digest<T: DeviceRepr>(device: &Device, tensor: &CudaSlice<T>) -> Result<String> {
    let host = device.stream.clone_dtoh(tensor)?;
    let bytes = unsafe {
        std::slice::from_raw_parts(
            host.as_ptr().cast::<u8>(),
            std::mem::size_of_val(host.as_slice()),
        )
    };
    Ok(sha256(bytes))
}
fn host_digest(values: &[bf16]) -> String {
    sha256(
        &values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn cache_hashes(device: &Device, state: &StateStore) -> Result<Value> {
    let c = state.kv[0]
        .compact
        .as_ref()
        .context("compact cache missing")?;
    Ok(
        json!({"k_code":digest(device,&c.k_code)?,"v_code":digest(device,&c.v_code)?,"k_stats":digest(device,&c.k_stats)?,"v_stats":digest(device,&c.v_stats)?}),
    )
}
struct Core<'a> {
    device: &'a Device,
    q: &'a CudaSlice<bf16>,
    k: &'a CudaSlice<bf16>,
    v: &'a CudaSlice<bf16>,
    compact: Option<CacheView>,
}
impl Core<'_> {
    fn run(&self, prefix: usize, rows: u32, qrow: usize, graph: bool) -> Result<Vec<bf16>> {
        let cap = 16640u32;
        let ws = AttentionWorkspace::q38(rows, cap, 256)?;
        let s = &self.device.stream;
        let meta = s.clone_htod(&[
            prefix as i32,
            rows as i32,
            prefix as i32 + rows as i32,
            cap as i32,
            1,
            0,
            0,
            0,
        ])?;
        let scratch = s.alloc_zeros::<u8>(ws.bytes as usize)?;
        let output = s.alloc_zeros::<bf16>(rows as usize * 6144)?;
        let mask = s.clone_htod(
            &(0..64)
                .map(|i| {
                    if i == 63 {
                        u64::MAX
                    } else {
                        (1u64 << (i + 1)) - 1
                    }
                })
                .collect::<Vec<_>>(),
        )?;
        let module = cubin::module(&self.device.ctx, "tcmqa_d256")?;
        let p1 = module.load_function(if self.compact.is_some() {
            "tcmqa_d256_phase1_compact"
        } else {
            "tcmqa_d256_phase1"
        })?;
        p1.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,68096)?;
        let p2 = module.load_function("tcmqa_d256_phase2")?;
        let q = self.q.device_ptr(s).0 + (qrow * 6144 * 2) as u64;
        let (k, v) = (self.k.device_ptr(s).0, self.v.device_ptr(s).0);
        let base = scratch.device_ptr(s).0;
        let (max, sum, out) = (base + ws.p_max, base + ws.p_sum, base + ws.p_out);
        let (meta, mask, output_ptr) = (
            meta.device_ptr(s).0,
            mask.device_ptr(s).0,
            output.device_ptr(s).0,
        );
        let (qt, head, pos, chunk, parts, seg, row0) = (
            6144i32,
            cap as i32 * 256,
            256i32,
            256i32,
            cap.div_ceil(256) as i32,
            rows as i32,
            0i32,
        );
        let label = if self.compact.is_some() {
            "compact TCMQA"
        } else {
            "BF16 TCMQA"
        };
        if graph {
            s.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .with_context(|| format!("{label} graph begin"))?;
        }
        let launches = (|| -> Result<()> {
            unsafe {
                let mut launch = s.launch_builder(&p1);
                launch
                    .arg(&q)
                    .arg(&k)
                    .arg(&v)
                    .arg(&max)
                    .arg(&sum)
                    .arg(&out)
                    .arg(&mask)
                    .arg(&meta)
                    .arg(&qt)
                    .arg(&head)
                    .arg(&pos)
                    .arg(&head)
                    .arg(&pos)
                    .arg(&chunk)
                    .arg(&parts)
                    .arg(&seg)
                    .arg(&row0);
                if let Some(cache) = self.compact.as_ref() {
                    launch.arg(cache);
                }
                launch.launch(LaunchConfig {
                    grid_dim: (cap.div_ceil(256), 4, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 68096,
                })?;
                s.launch_builder(&p2)
                    .arg(&max)
                    .arg(&sum)
                    .arg(&out)
                    .arg(&output_ptr)
                    .arg(&meta)
                    .arg(&qt)
                    .arg(&chunk)
                    .arg(&parts)
                    .arg(&seg)
                    .arg(&row0)
                    .launch(LaunchConfig {
                        grid_dim: (rows * 24, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: (2 * cap.div_ceil(256) + 256) * 4,
                    })?;
            }
            Ok(())
        })();
        let captured = if graph {
            Some(s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0)))
        } else {
            None
        };
        launches.with_context(|| format!("{label} eager/graph launches"))?;
        let graph = captured
            .map(|g| {
                g.with_context(|| format!("{label} graph end"))?
                    .context("empty TCMQA graph")
            })
            .transpose()?;
        if let Some(g) = graph.as_ref() {
            g.launch()?;
        }
        s.synchronize()?;
        let result = s.clone_dtoh(&output)?;
        ensure!(
            result.iter().all(|x| x.is_finite()),
            "nonfinite attention output"
        );
        Ok(result)
    }
}
fn same(a: &[bf16], b: &[bf16], label: &str) -> Result<()> {
    ensure!(a.len() == b.len(), "{label}: extent");
    if let Some(i) = a
        .iter()
        .zip(b)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        anyhow::bail!(
            "{label}: word{i} got{} expected{}",
            a[i].to_bits(),
            b[i].to_bits()
        );
    }
    Ok(())
}

#[test]
#[ignore]
fn kv_cache_kernel_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "coordinator GPU window required"
    );
    let (root, out, manifest) = input()?;
    let device = Device::new(0)?;
    device.finish_upload()?;
    ensure!(
        !device.ctx.is_event_tracking(),
        "compact gate requires event tracking disabled before allocation"
    );
    let s = &device.stream;
    let cap = 16640;
    let spec = crate::model_spec::ModelSpec::qwen(
        &serde_json::from_str(include_str!("../../../tests/data/q38-config.json"))?,
        &Value::Null,
    )?;
    let spec = crate::mtp::component_spec(&spec);
    let smoke = match manifest["run_mode"].as_str() {
        Some("smoke") => true,
        Some("formal") => false,
        _ => anyhow::bail!("fixture run_mode must be smoke/formal"),
    };
    let mut reports = Vec::new();
    let mut defects = Vec::new();
    macro_rules! score {($condition:expr,$label:expr)=>{if !$condition {defects.push(json!({"stage":$label,"class":"DEFECT"}));}};}
    macro_rules! bits {($a:expr,$b:expr,$label:expr)=>{if let Err(error)=same($a,$b,$label) {defects.push(json!({"stage":$label,"class":"DEFECT","difference":error.to_string()}));}};}
    for family in if smoke {
        vec!["synthetic"]
    } else {
        vec!["synthetic", "real"]
    } {
        let k_src = s.clone_htod(&bf16_file(
            root.join(format!("{family}-k.bf16")),
            4 * cap * 256,
        )?)?;
        let v_src = s.clone_htod(&bf16_file(
            root.join(format!("{family}-v.bf16")),
            4 * cap * 256,
        )?)?;
        let q = s.clone_htod(&bf16_file(
            root.join(format!("{family}-q.bf16")),
            2048 * 6144,
        )?)?;
        let k_proj = s.clone_htod(&bf16_file(root.join("writer-k.bf16"), 4 * 1024)?)?;
        let v_proj = s.clone_htod(&bf16_file(root.join("writer-v.bf16"), 4 * 1024)?)?;
        let format = KvFormat::Affine8;
        let codec = "affine8";
        for prefix in if smoke {
            vec![2176usize]
        } else {
            vec![
                2047, 2048, 2049, 2175, 2176, 2177, 2242, 2243, 2244, 4095, 4096, 16320,
            ]
        } {
            let defect_start = defects.len();
            let mut state = StateStore::new_with_format(&spec, cap, format, s)?;
            state.set_sequence(s, 0, prefix as u32, prefix as u32)?;
            let kv = &state.kv[0];
            let c = kv.compact.as_ref().context("compact missing")?;
            c.finish(
                c.view(&kv.k, &kv.v, s),
                state.seq_ref(s)?,
                Some((&k_src, &v_src, prefix)),
                true,
                s,
            )?;
            s.synchronize()?;
            let expected = &manifest["expected"][family][codec][prefix.to_string()];
            let cache_initial = cache_hashes(&device, &state)?;
            score!(
                cache_initial == expected["cache"],
                format!("CPU codec oracle {family}/{codec}/{prefix}")
            );
            state.set_sequence(s, prefix as u32, 4, 4)?;
            let k_full = s.alloc_zeros::<bf16>(4 * cap * 256)?;
            let v_full = s.alloc_zeros::<bf16>(4 * cap * 256)?;
            let kv = &state.kv[0];
            let c = kv.compact.as_ref().context("compact missing")?;
            c.dequant_prefix(
                c.view(&kv.k, &kv.v, s),
                state.seq_ref(s)?,
                &k_full,
                &v_full,
                prefix,
                s,
            )?;
            let dequant_k = digest(&device, &k_full)?;
            let dequant_v = digest(&device, &v_full)?;
            score!(
                dequant_k == expected["dequant_k"] && dequant_v == expected["dequant_v"],
                format!("CPU dequant oracle {family}/{codec}/{prefix}")
            );
            let mut token = s.alloc_zeros::<u32>(1)?;
            let mut checkpoint = StateCheckpoint::new(&state, &spec, 4, s)?;
            let before = [digest(&device, &kv.k)?, digest(&device, &kv.v)?];
            checkpoint.save(&state, &token, prefix, s, None)?;
            let meta = state.seq_ref(s)?.ptr;
            let cache = c.view(&kv.k, &kv.v, s);
            let module = cubin::module(&device.ctx, "tcmqa_d256")?;
            let write = module.load_function("tcmqa_d256_write_ring")?;
            let k_scatter = module.load_function("tcmqa_d256_rope_k_scatter")?;
            let v_scatter = module.load_function("tcmqa_d256_v_scatter")?;
            let (kp, vp) = (k_proj.device_ptr(s).0, v_proj.device_ptr(s).0);
            let (kf, vf) = (k_full.device_ptr(s).0, v_full.device_ptr(s).0);
            let (stride, head, dim, axes) = (1024i32, (cap * 256) as i32, 256i32, 0u64);
            unsafe {
                s.launch_builder(&write)
                    .arg(&kp)
                    .arg(&vp)
                    .arg(&meta)
                    .arg(&axes)
                    .arg(&stride)
                    .arg(&stride)
                    .arg(&cache)
                    .launch(flat(4 * 1024))?;
                s.launch_builder(&k_scatter)
                    .arg(&kp)
                    .arg(&kf)
                    .arg(&meta)
                    .arg(&stride)
                    .arg(&head)
                    .arg(&dim)
                    .launch(flat(4 * 1024))?;
                s.launch_builder(&v_scatter)
                    .arg(&vp)
                    .arg(&vf)
                    .arg(&meta)
                    .arg(&stride)
                    .arg(&head)
                    .arg(&dim)
                    .launch(flat(4 * 1024))?;
            }
            for h in 0..4 {
                for r in 0..4 {
                    let src = (h * cap + prefix + r) * 256;
                    let dst = (h * cache.ring as usize
                        + crate::kv::ring_slot(prefix + r, cache.ring as usize)?)
                        * 256;
                    bits!(
                        &s.clone_dtoh(&kv.k.slice(dst..dst + 256))?,
                        &s.clone_dtoh(&k_full.slice(src..src + 256))?,
                        "K ring writer"
                    );
                    bits!(
                        &s.clone_dtoh(&kv.v.slice(dst..dst + 256))?,
                        &s.clone_dtoh(&v_full.slice(src..src + 256))?,
                        "V ring writer"
                    );
                }
            }
            let linear_k = digest(&device, &k_full)?;
            let equal_axes = s.clone_htod(
                &(0..4)
                    .flat_map(|r| [prefix as i32 + r; 3])
                    .collect::<Vec<_>>(),
            )?;
            let distinct_axes = s.clone_htod(
                &(0..4)
                    .flat_map(|r| {
                        [
                            prefix as i32 + r + 3,
                            prefix as i32 + r + 7,
                            prefix as i32 + r + 13,
                        ]
                    })
                    .collect::<Vec<_>>(),
            )?;
            let graph_axes = equal_axes.device_ptr(s).0;
            let scatter_axes = module.load_function("tcmqa_d256_rope_k_scatter_axes")?;
            for (coordinates, equal) in [(&equal_axes, true), (&distinct_axes, false)] {
                let axes_ptr = coordinates.device_ptr(s).0;
                unsafe {
                    s.launch_builder(&write)
                        .arg(&kp)
                        .arg(&vp)
                        .arg(&meta)
                        .arg(&axes_ptr)
                        .arg(&stride)
                        .arg(&stride)
                        .arg(&cache)
                        .launch(flat(4 * 1024))?;
                    s.launch_builder(&scatter_axes)
                        .arg(&kp)
                        .arg(&kf)
                        .arg(&meta)
                        .arg(&axes_ptr)
                        .arg(&stride)
                        .arg(&head)
                        .arg(&dim)
                        .launch(flat(4 * 1024))?;
                }
                ensure!(
                    (digest(&device, &k_full)? == linear_k) == equal,
                    "axes reference negative control invalid"
                );
                for h in 0..4 {
                    for r in 0..4 {
                        let src = (h * cap + prefix + r) * 256;
                        let dst = (h * cache.ring as usize
                            + crate::kv::ring_slot(prefix + r, cache.ring as usize)?)
                            * 256;
                        bits!(
                            &s.clone_dtoh(&kv.k.slice(dst..dst + 256))?,
                            &s.clone_dtoh(&k_full.slice(src..src + 256))?,
                            "axes K ring writer"
                        );
                        bits!(
                            &s.clone_dtoh(&kv.v.slice(dst..dst + 256))?,
                            &s.clone_dtoh(&v_full.slice(src..src + 256))?,
                            "axes V ring writer"
                        );
                    }
                }
            }
            unsafe {
                s.launch_builder(&write)
                    .arg(&kp)
                    .arg(&vp)
                    .arg(&meta)
                    .arg(&axes)
                    .arg(&stride)
                    .arg(&stride)
                    .arg(&cache)
                    .launch(flat(4 * 1024))?;
                s.launch_builder(&k_scatter)
                    .arg(&kp)
                    .arg(&kf)
                    .arg(&meta)
                    .arg(&stride)
                    .arg(&head)
                    .arg(&dim)
                    .launch(flat(4 * 1024))?;
            }
            let mut row_checks = Vec::new();
            for rows in 1..=4 {
                let compact = Core {
                    device: &device,
                    q: &q,
                    k: &kv.k,
                    v: &kv.v,
                    compact: Some(cache),
                }
                .run(prefix, rows, 0, false)?;
                let reference = Core {
                    device: &device,
                    q: &q,
                    k: &k_full,
                    v: &v_full,
                    compact: None,
                }
                .run(prefix, rows, 0, false)?;
                bits!(&compact, &reference, "TCMQA dequant vs reconstructed BF16");
                row_checks.push(json!({"rows":rows,"candidate":host_digest(&compact),"reference":host_digest(&reference)}));
                for row in 0..rows as usize {
                    let one = Core {
                        device: &device,
                        q: &q,
                        k: &kv.k,
                        v: &kv.v,
                        compact: Some(cache),
                    }
                    .run(prefix + row, 1, row, false)?;
                    bits!(
                        &compact[row * 6144..(row + 1) * 6144],
                        &one,
                        "TCMQA row invariance"
                    );
                    let control = Core {
                        device: &device,
                        q: &q,
                        k: &k_full,
                        v: &v_full,
                        compact: None,
                    }
                    .run(prefix + row, 1, row, false)?;
                    same(
                        &reference[row * 6144..(row + 1) * 6144],
                        &control,
                        "BF16 oracle row invariance validity",
                    )?;
                }
            }
            let cg = Core {
                device: &device,
                q: &q,
                k: &kv.k,
                v: &kv.v,
                compact: Some(cache),
            }
            .run(prefix, 4, 0, true)?;
            let rg = Core {
                device: &device,
                q: &q,
                k: &k_full,
                v: &v_full,
                compact: None,
            }
            .run(prefix, 4, 0, true)?;
            bits!(&cg, &rg, "TCMQA graph bits");
            s.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .context("KV writer/deferred commit graph begin")?;
            let launches = (|| -> Result<()> {
                unsafe {
                    s.launch_builder(&write)
                        .arg(&kp)
                        .arg(&vp)
                        .arg(&meta)
                        .arg(&graph_axes)
                        .arg(&stride)
                        .arg(&stride)
                        .arg(&cache)
                        .launch(flat(4 * 1024))?;
                }
                state.compact_commit(s, false)
            })();
            let captured = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
            launches.context("KV writer/deferred commit graph launches")?;
            let graph = captured
                .context("KV writer/deferred commit graph end")?
                .context("empty KV writer/deferred commit graph")?;
            graph.launch()?;
            s.synchronize()?;
            state.set_sequence(s, prefix as u32 + 4, 1, 4)?;
            state.compact_commit(s, false)?;
            score!(
                s.clone_dtoh(
                    &state.kv[0]
                        .compact
                        .as_ref()
                        .context("compact missing")?
                        .cursor
                )? == [prefix as i32, crate::kv::sealed_until(prefix) as i32, 1, 0],
                "tentative rows sealed"
            );
            checkpoint.restore(&mut state, &mut token, s)?;
            score!(
                before
                    == [
                        digest(&device, &state.kv[0].k)?,
                        digest(&device, &state.kv[0].v)?
                    ],
                "rollback ring bits"
            );
            unsafe {
                s.launch_builder(&write)
                    .arg(&kp)
                    .arg(&vp)
                    .arg(&meta)
                    .arg(&axes)
                    .arg(&stride)
                    .arg(&stride)
                    .arg(&cache)
                    .launch(flat(4 * 1024))?;
            }
            let mut committed_hashes = Vec::new();
            for accepted in 0..=4 {
                state.set_sequence(s, prefix as u32 + accepted, 1, 4)?;
                state.compact_commit(s, true)?;
                let expected = &manifest["expected"][family][codec]
                    [(prefix + accepted as usize).to_string()]["cache"];
                let actual = cache_hashes(&device, &state)?;
                score!(actual == *expected, "committed sealing oracle");
                committed_hashes.push(json!({"prefix":prefix+accepted as usize,"cache":actual}));
            }
            reports.push(json!({"family":family,"codec":codec,"prefix":prefix,"rows":[1,2,3,4],"writer_coordinates":["linear","axes_equal","axes_distinct"],"graph_writer_coordinates":"axes_equal","cache_initial":cache_initial,"dequant_k":dequant_k,"dequant_v":dequant_v,"committed":committed_hashes,"row_checks":row_checks,"graph_candidate":host_digest(&cg),"graph_reference":host_digest(&rg),"checks":["writer_bits","TCMQA_bits","CPU_dequant_bits","commit0..4","rollback_bits"],"defect_indices":(defect_start..defects.len()).collect::<Vec<_>>()}));
        }
        let defect_start = defects.len();
        let prefix = 14336usize;
        let rows = 2048usize;
        let end = prefix + rows;
        let mut state = StateStore::new_with_format(&spec, cap, format, s)?;
        state.set_sequence(s, 0, prefix as u32, prefix as u32)?;
        let kv = &state.kv[0];
        let c = kv.compact.as_ref().context("compact missing")?;
        c.finish(
            c.view(&kv.k, &kv.v, s),
            state.seq_ref(s)?,
            Some((&k_src, &v_src, prefix)),
            true,
            s,
        )?;
        state.set_sequence(s, prefix as u32, rows as u32, rows as u32)?;
        let kv = &state.kv[0];
        let c = kv.compact.as_ref().context("compact missing")?;
        let mut k_full = s.alloc_zeros::<bf16>(4 * cap * 256)?;
        let mut v_full = s.alloc_zeros::<bf16>(4 * cap * 256)?;
        c.dequant_prefix(
            c.view(&kv.k, &kv.v, s),
            state.seq_ref(s)?,
            &k_full,
            &v_full,
            prefix,
            s,
        )?;
        for h in 0..4 {
            let start = (h * cap + prefix) * 256;
            let stop = (h * cap + end) * 256;
            s.memcpy_dtod(
                &k_src.slice(start..stop),
                &mut k_full.slice_mut(start..stop),
            )?;
            s.memcpy_dtod(
                &v_src.slice(start..stop),
                &mut v_full.slice_mut(start..stop),
            )?;
        }
        let oracle_k = s.clone_htod(&bf16_file(
            root.join(format!("{family}-{codec}-prefill-k.bf16")),
            4 * cap * 256,
        )?)?;
        let oracle_v = s.clone_htod(&bf16_file(
            root.join(format!("{family}-{codec}-prefill-v.bf16")),
            4 * cap * 256,
        )?)?;
        let input_k = digest(&device, &k_full)?;
        let input_v = digest(&device, &v_full)?;
        let reference_k = digest(&device, &oracle_k)?;
        let reference_v = digest(&device, &oracle_v)?;
        ensure!(
            reference_k == manifest["files"][format!("{family}-{codec}-prefill-k.bf16")]["sha256"]
                && reference_v
                    == manifest["files"][format!("{family}-{codec}-prefill-v.bf16")]["sha256"],
            "FA2 uploaded oracle invalid"
        );
        score!(
            input_k == reference_k && input_v == reference_v,
            "FA2 input oracle"
        );
        let mut flash = Flash::new(s.clone(), rows, 24)?;
        let mut result = s.alloc_zeros::<bf16>(rows * 6144)?;
        let mut reference = s.alloc_zeros::<bf16>(rows * 6144)?;
        let shape = Shape {
            rows,
            start: prefix,
            q_heads: 24,
            kv_heads: 4,
            capacity: cap,
            head_dim: 256,
            causal: true,
        };
        flash.run(&q, 6144, &k_full, &v_full, &mut result, shape)?;
        flash.run(&q, 6144, &oracle_k, &oracle_v, &mut reference, shape)?;
        let fa_candidate = s.clone_dtoh(&result)?;
        let fa_reference = s.clone_dtoh(&reference)?;
        ensure!(
            fa_candidate
                .iter()
                .chain(&fa_reference)
                .all(|v| v.is_finite()),
            "nonfinite FA2 output"
        );
        bits!(&fa_candidate, &fa_reference, "FA2 reconstructed prefix");
        c.finish(
            c.view(&kv.k, &kv.v, s),
            state.seq_ref(s)?,
            Some((&k_full, &v_full, rows)),
            true,
            s,
        )?;
        let cache_final = cache_hashes(&device, &state)?;
        score!(
            cache_final == manifest["expected"][family][codec][end.to_string()]["cache"],
            "prefill sealing oracle"
        );
        reports.push(json!({"family":family,"codec":codec,"prefill_rows":rows,"prefix":prefix,"input_k":input_k,"input_v":input_v,"reference_k":reference_k,"reference_v":reference_v,"candidate":host_digest(&fa_candidate),"reference":host_digest(&fa_reference),"cache_final":cache_final,"checks":["FA2_bits","CPU_sealed_once_bits"],"defect_indices":(defect_start..defects.len()).collect::<Vec<_>>()}));
    }
    ensure!(
        reports.len() == if smoke { 2 } else { 26 },
        "kernel case count"
    );
    std::fs::write(
        out.join("KERNEL.json"),
        serde_json::to_vec_pretty(
            &json!({"status":"KERNEL_VALIDITY_PASS","run_mode":manifest["run_mode"],"cases":reports,"defects":defects,"candidate_pass":defects.is_empty(),"GPU_executed":true}),
        )?,
    )?;
    println!(
        "KV8_KERNEL_VALIDITY_PASS cases={} defects={}",
        reports.len(),
        defects.len()
    );
    Ok(())
}

#[test]
#[ignore = "authorized component GPU window required"]
fn kv_cache_prefix_request_history_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "coordinator GPU window required"
    );
    let (root, out, _) = input()?;
    let device = Device::new(0)?;
    device.finish_upload()?;
    ensure!(
        !device.ctx.is_event_tracking(),
        "prefix history test requires event tracking disabled"
    );
    let s = &device.stream;
    let spec = crate::model_spec::ModelSpec::qwen(
        &serde_json::from_str(include_str!("../../../tests/data/q38-config.json"))?,
        &Value::Null,
    )?;
    let spec = crate::mtp::component_spec(&spec);
    let cap = 16640;
    let k = s.clone_htod(&bf16_file(root.join("synthetic-k.bf16"), 4 * cap * 256)?)?;
    let v = s.clone_htod(&bf16_file(root.join("synthetic-v.bf16"), 4 * cap * 256)?)?;
    let mut ring = unsafe { device.ctx.alloc_pinned_with_flags::<u8>(1 << 20, 0)? };
    let mut receipts = Vec::new();
    let format = KvFormat::Affine8;
    let mut stores = [
        StateStore::new_with_format(&spec, cap, format, s)?,
        StateStore::new_with_format(&spec, cap, format, s)?,
    ];
    let mut history = Vec::new();
    let mut payloads = Vec::new();
    for (index, state) in stores.iter_mut().enumerate() {
        state.set_sequence(s, 0, 4096, 4096)?;
        let kv = &state.kv[0];
        let c = kv.compact.as_ref().context("compact cache missing")?;
        let (old_k, old_v) = if index == 0 { (&k, &v) } else { (&v, &k) };
        c.finish(
            c.view(&kv.k, &kv.v, s),
            state.seq_ref(s)?,
            Some((old_k, old_v, 4096)),
            true,
            s,
        )?;
        history.push(cache_hashes(&device, state)?);
        state.reset(s)?;
        state.set_sequence(s, 0, 2177, 2177)?;
        let kv = &state.kv[0];
        let c = kv.compact.as_ref().context("compact cache missing")?;
        c.finish(
            c.view(&kv.k, &kv.v, s),
            state.seq_ref(s)?,
            Some((&k, &v, 2177)),
            true,
            s,
        )?;
        let mut bytes = Vec::new();
        for start in (0..2177).step_by(256) {
            bytes.extend(state.kv[0].compact_prefix_block(
                start..(start + 256).min(2177),
                2177,
                s,
                &mut ring,
                None,
            )?);
        }
        payloads.push(bytes);
    }
    ensure!(
        history[0] != history[1],
        "request-history negative control did not change prior cache"
    );
    let identical = payloads[0] == payloads[1];
    receipts.push(json!({"codec":format.tag(),"prefix":2177,"previous_prefix":4096,"history_control_distinct":true,"payload_sha256":sha256(&payloads[0]),"payload_bytes":payloads[0].len(),"two_state_stores_identical":identical}));
    std::fs::write(
        out.join("PREFIX_HISTORY.json"),
        serde_json::to_vec_pretty(
            &json!({"status":"PREFIX_HISTORY_VALIDITY_PASS","candidate_pass":receipts.iter().all(|r|r["two_state_stores_identical"]==true),"GPU_executed":true,"cases":receipts}),
        )?,
    )?;
    println!(
        "PREFIX_HISTORY_VALIDITY_PASS: affine8; two engine-owned state stores; different previous requests; prefix payload comparison recorded"
    );
    Ok(())
}
