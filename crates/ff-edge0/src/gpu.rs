//! CUDA device wrapper for the fused int4/int8 GEMV kernels (cudarc 0.19
//! stream-centric API: CudaContext -> module -> CudaStream launches).

use crate::int4::GroupQuant;
use anyhow::{Context, Result, ensure};
use cudarc::driver::safe::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

pub struct GpuContext {
    pub context: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    gemv4: cudarc::driver::safe::CudaFunction,
    gemv8: cudarc::driver::safe::CudaFunction,
    gemv4_r4: cudarc::driver::safe::CudaFunction,
    gemv8_r4: cudarc::driver::safe::CudaFunction,
    silu_mul: cudarc::driver::safe::CudaFunction,
    batched4: cudarc::driver::safe::CudaFunction,
    batched4s: cudarc::driver::safe::CudaFunction,
    batched4silu: cudarc::driver::safe::CudaFunction,
    k_gemv1_silu: cudarc::driver::safe::CudaFunction,
    lora_add: cudarc::driver::safe::CudaFunction,
    gdn_conv: cudarc::driver::safe::CudaFunction,
    gdn_heads: cudarc::driver::safe::CudaFunction,
    k_embed_row: cudarc::driver::safe::CudaFunction,
    k_rmsnorm: cudarc::driver::safe::CudaFunction,
    k_final_norm: cudarc::driver::safe::CudaFunction,
    k_add: cudarc::driver::safe::CudaFunction,
    k_attn_qk: cudarc::driver::safe::CudaFunction,
    k_attn_scores: cudarc::driver::safe::CudaFunction,
    k_group4: cudarc::driver::safe::CudaFunction,
    k_moe_mega: cudarc::driver::safe::CudaFunction,
    /// Cached co-residency capacity for the mega grid (per-SM blocks x SMs);
    /// queried once — this path is sync-latency-bound.
    mega_coresident: std::sync::OnceLock<u32>,
    k_read_scatter: cudarc::driver::safe::CudaFunction,
    k_add_rmsnorm: cudarc::driver::safe::CudaFunction,
    k_router_topk: cudarc::driver::safe::CudaFunction,
    k_moe_combine: cudarc::driver::safe::CudaFunction,
    k_argmax: cudarc::driver::safe::CudaFunction,
    k_argmax_part: cudarc::driver::safe::CudaFunction,
    k_argmax_final: cudarc::driver::safe::CudaFunction,
    k_rmsnorm_zc: cudarc::driver::safe::CudaFunction,
    k_add_rmsnorm_zc: cudarc::driver::safe::CudaFunction,
    k_attn_qk_zc: cudarc::driver::safe::CudaFunction,
    k_attn_qk_zc_mrope: cudarc::driver::safe::CudaFunction,
    k_inc: cudarc::driver::safe::CudaFunction,
    k_inc3: cudarc::driver::safe::CudaFunction,
    /// Counted per kernel launch — read BEFORE reading time (today's four
    /// bogus per-sync constants all came from reading time first).
    pub launch_count: std::sync::atomic::AtomicU64,
    /// Counted per synchronize() — the "number it before dividing" fix:
    /// every per-round-trip constant comes from THIS counter plus wall
    /// time, never from a guessed denominator.
    pub sync_count: std::sync::atomic::AtomicU64,
    /// Two-pass argmax partials (128 blocks).
    argmax_scratch: (CudaSlice<f32>, CudaSlice<i32>),
}

pub struct LoraGpu {
    a: CudaSlice<f32>,
    b: CudaSlice<f32>,
    rank: i32,
}

pub struct GpuQuant {
    packed: CudaSlice<u32>,
    scales: CudaSlice<f32>,
    biases: CudaSlice<f32>,
    y: CudaSlice<f32>,
    lora: Option<LoraGpu>,
    pub out_dim: usize,
    pub in_dim: usize,
    bits: u32,
}

impl GpuContext {
    pub fn new() -> Result<Self> {
        let context = CudaContext::new(0).context("failed to init CUDA context")?;
        // cudarc's per-launch safety events turn on once a second stream
        // exists (the capture stream) and each launch then waits on events
        // recorded before capture — CUDA_ERROR_STREAM_CAPTURE_ISOLATION.
        // Safe here: every slice is allocated, used, and dropped on this
        // one stream, the only cross-stream rule the events enforce.
        unsafe { context.disable_event_tracking() };
        let ptx =
            cudarc::nvrtc::Ptx::from_src(include_str!(concat!(env!("OUT_DIR"), "/edge0_gemv.ptx")));
        let module = context
            .load_module(ptx)
            .context("failed to load edge0 PTX module")?;
        let silu_ptx = cudarc::nvrtc::Ptx::from_src(include_str!(concat!(
            env!("OUT_DIR"),
            "/edge0_silu_mul.ptx"
        )));
        let silu_module = context
            .load_module(silu_ptx)
            .context("failed to load silu PTX module")?;
        let batched_ptx = cudarc::nvrtc::Ptx::from_src(include_str!(concat!(
            env!("OUT_DIR"),
            "/edge0_batched_gemv.ptx"
        )));
        let batched_module = context
            .load_module(batched_ptx)
            .context("failed to load batched PTX module")?;
        let lora_ptx =
            cudarc::nvrtc::Ptx::from_src(include_str!(concat!(env!("OUT_DIR"), "/lora_add.ptx")));
        let lora_module = context
            .load_module(lora_ptx)
            .context("failed to load lora PTX module")?;
        let gdn_ptx =
            cudarc::nvrtc::Ptx::from_src(include_str!(concat!(env!("OUT_DIR"), "/edge0_gdn.ptx")));
        let gdn_module = context
            .load_module(gdn_ptx)
            .context("failed to load gdn PTX module")?;
        let glue_ptx =
            cudarc::nvrtc::Ptx::from_src(include_str!(concat!(env!("OUT_DIR"), "/edge0_glue.ptx")));
        let glue_module = context
            .load_module(glue_ptx)
            .context("failed to load glue PTX module")?;
        let gemv4 = module
            .load_function("edge0_gemv4")
            .context("edge0_gemv4 missing")?;
        let gemv8 = module
            .load_function("edge0_gemv8")
            .context("edge0_gemv8 missing")?;
        // Small-out variants: 4 rows/block → 4x the blocks, more loads in
        // flight (router [256,2048] ran latency-bound at 16 blocks).
        let gemv4_r4 = module
            .load_function("edge0_gemv4r4")
            .context("edge0_gemv4r4 missing")?;
        let gemv8_r4 = module
            .load_function("edge0_gemv8r4")
            .context("edge0_gemv8r4 missing")?;
        let silu_mul = silu_module
            .load_function("edge0_silu_mul")
            .context("edge0_silu_mul missing")?;
        let batched4 = batched_module
            .load_function("edge0_batched_gemv4")
            .context("edge0_batched_gemv4 missing")?;
        let batched4s = batched_module
            .load_function("edge0_batched_gemv4_slotx")
            .context("edge0_batched_gemv4_slotx missing")?;
        let batched4silu = batched_module
            .load_function("edge0_batched_gemv4_slotx_silu")
            .context("edge0_batched_gemv4_slotx_silu missing")?;
        let k_gemv1_silu = glue_module
            .load_function("edge0_gemv1_silu_lora")
            .context("edge0_gemv1_silu_lora missing")?;
        let lora_add = lora_module
            .load_function("edge0_lora_add")
            .context("edge0_lora_add missing")?;
        let gdn_conv = gdn_module
            .load_function("edge0_gdn_conv")
            .context("edge0_gdn_conv missing")?;
        let gdn_heads = gdn_module
            .load_function("edge0_gdn_heads")
            .context("edge0_gdn_heads missing")?;
        let k_embed_row = glue_module
            .load_function("edge0_embed_row")
            .context("edge0_embed_row missing")?;
        let k_rmsnorm = glue_module
            .load_function("edge0_rmsnorm")
            .context("edge0_rmsnorm missing")?;
        let k_final_norm = glue_module
            .load_function("edge0_final_norm")
            .context("edge0_final_norm missing")?;
        let k_add = glue_module
            .load_function("edge0_add_inplace")
            .context("edge0_add_inplace missing")?;
        let k_attn_qk = glue_module
            .load_function("edge0_attn_qk")
            .context("edge0_attn_qk missing")?;
        let k_attn_scores = glue_module
            .load_function("edge0_attn_scores")
            .context("edge0_attn_scores missing")?;
        let k_group4 = glue_module
            .load_function("edge0_gemv_group4_lora")
            .context("edge0_gemv_group4_lora missing")?;
        let mega_ptx =
            cudarc::nvrtc::Ptx::from_src(include_str!(concat!(env!("OUT_DIR"), "/edge0_mega.ptx")));
        let mega_module = context
            .load_module(mega_ptx)
            .context("failed to load mega PTX module")?;
        let k_moe_mega = mega_module
            .load_function("edge0_moe_mega")
            .context("edge0_moe_mega missing")?;
        let k_read_scatter = mega_module
            .load_function("edge0_read_scatter")
            .context("edge0_read_scatter missing")?;
        let k_add_rmsnorm = glue_module
            .load_function("edge0_add_rmsnorm")
            .context("edge0_add_rmsnorm missing")?;
        let k_router_topk = glue_module
            .load_function("edge0_router_topk")
            .context("edge0_router_topk missing")?;
        let k_moe_combine = glue_module
            .load_function("edge0_moe_combine")
            .context("edge0_moe_combine missing")?;
        let k_argmax = glue_module
            .load_function("edge0_argmax")
            .context("edge0_argmax missing")?;
        let k_argmax_part = glue_module
            .load_function("edge0_argmax_part")
            .context("edge0_argmax_part missing")?;
        let k_argmax_final = glue_module
            .load_function("edge0_argmax_final")
            .context("edge0_argmax_final missing")?;
        // qwen3_5 dense variants: zero-centered norms.
        let k_rmsnorm_zc = glue_module
            .load_function("edge0_rmsnorm_zc")
            .context("edge0_rmsnorm_zc missing")?;
        let k_add_rmsnorm_zc = glue_module
            .load_function("edge0_add_rmsnorm_zc")
            .context("edge0_add_rmsnorm_zc missing")?;
        let k_attn_qk_zc = glue_module
            .load_function("edge0_attn_qk_zc")
            .context("edge0_attn_qk_zc missing")?;
        let k_attn_qk_zc_mrope = glue_module
            .load_function("edge0_attn_qk_zc_mrope")
            .context("edge0_attn_qk_zc_mrope missing")?;
        let k_inc3 = glue_module
            .load_function("edge0_inc3")
            .context("edge0_inc3 missing")?;
        let k_inc = glue_module
            .load_function("edge0_inc")
            .context("edge0_inc missing")?;
        // A created stream: stream capture is only legal off the legacy
        // default stream (graph replay replaces the CPU enqueue bottleneck).
        let stream = context.new_stream().context("decode stream")?;
        let argmax_scratch = (
            stream.alloc_zeros::<f32>(128).context("argmax pv")?,
            stream.alloc_zeros::<i32>(128).context("argmax pi")?,
        );
        Ok(Self {
            context,
            stream,
            gemv4,
            gemv8,
            gemv4_r4,
            gemv8_r4,
            silu_mul,
            batched4,
            batched4s,
            batched4silu,
            k_gemv1_silu,
            lora_add,
            gdn_conv,
            gdn_heads,
            k_embed_row,
            k_rmsnorm,
            k_final_norm,
            k_add,
            k_attn_qk,
            k_attn_scores,
            k_group4,
            k_moe_mega,
            mega_coresident: std::sync::OnceLock::new(),
            k_read_scatter,
            k_add_rmsnorm,
            k_router_topk,
            k_moe_combine,
            k_argmax,
            k_argmax_part,
            k_argmax_final,
            k_rmsnorm_zc,
            k_add_rmsnorm_zc,
            k_attn_qk_zc,
            k_attn_qk_zc_mrope,
            k_inc,
            k_inc3,
            sync_count: std::sync::atomic::AtomicU64::new(0),
            launch_count: std::sync::atomic::AtomicU64::new(0),
            argmax_scratch,
        })
    }

    pub fn upload_slice(&self, values: &[u32]) -> Result<CudaSlice<u32>> {
        let out = self
            .stream
            .clone_htod(values)
            .context("u32 upload failed")?;
        Ok(out)
    }

    pub fn ctx_u64(&self, values: &[u64]) -> Result<CudaSlice<u64>> {
        self.stream.clone_htod(values).context("u64 upload failed")
    }

    pub fn upload_i32(&self, values: &[i32]) -> Result<CudaSlice<i32>> {
        let out = self
            .stream
            .clone_htod(values)
            .context("i32 upload failed")?;
        Ok(out)
    }

    pub fn dtoh(&self, values: &CudaSlice<f32>) -> Result<Vec<f32>> {
        let mut host = vec![0f32; values.len()];
        self.stream.memcpy_dtoh(values, &mut host)?;
        self.stream.synchronize()?;
        Ok(host)
    }

    pub fn upload_f32(&self, values: &[f32]) -> Result<CudaSlice<f32>> {
        let out = self
            .stream
            .clone_htod(values)
            .context("f32 upload failed")?;
        Ok(out)
    }

    pub fn upload(
        &self,
        quant: &GroupQuant,
        lora: Option<(&[f32], &[f32], usize)>,
    ) -> Result<GpuQuant> {
        let lora_gpu = match lora {
            Some((a, b, rank)) => {
                // The lora kernels stage A-side dots in __shared__ ax[16].
                ensure!(
                    rank <= 16,
                    "lora rank {rank} exceeds the kernels' shared-memory cap 16"
                );
                Some(LoraGpu {
                    a: self.stream.clone_htod(a).context("lora A upload failed")?,
                    b: self.stream.clone_htod(b).context("lora B upload failed")?,
                    rank: rank as i32,
                })
            }
            None => None,
        };
        let packed = self
            .stream
            .clone_htod(&quant.packed)
            .context("packed upload failed")?;
        let scales = self
            .stream
            .clone_htod(&quant.scales)
            .context("scales upload failed")?;
        let biases = self
            .stream
            .clone_htod(&quant.biases)
            .context("biases upload failed")?;
        self.stream.synchronize().context("upload sync failed")?;
        let y = self
            .stream
            .alloc_zeros::<f32>(quant.out_dim)
            .context("y alloc failed")?;
        Ok(GpuQuant {
            packed,
            scales,
            biases,
            y,
            lora: lora_gpu,
            out_dim: quant.out_dim,
            in_dim: quant.in_dim,
            bits: quant.bits,
        })
    }
}

impl GpuContext {
    /// Batched expert GEMV: one launch covers all 4 routed experts for
    /// one projection of one layer (gather_qmm shape — the stacked
    /// tensor's first dim IS the expert index). Kernel-count first: the
    /// expected reading is 120 launches/token after this replaces the
    /// per-expert dispatch.
    pub fn batched_expert_gemv_slotx(
        &self,
        experts: &GpuExperts,
        layer: usize,
        part: usize,
        expert_ids: &CudaSlice<i32>,
        x: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
        slots: usize,
    ) -> Result<()> {
        let rows = experts.rows[part] as i32;
        let in_dim = experts.in_dim[part] as i32;
        let slots_i = slots as i32;
        self.launch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            self.stream
                .launch_builder(&self.batched4s)
                .arg(&experts.stacked[layer][part])
                .arg(&experts.stacked_scales[layer][part])
                .arg(&experts.stacked_biases[layer][part])
                .arg(expert_ids)
                .arg(x)
                .arg(y)
                .arg(&rows)
                .arg(&in_dim)
                .arg(&slots_i)
                .launch(LaunchConfig {
                    grid_dim: ((rows as u32).div_ceil(4), slots as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("slotx launch failed: {e}"))?;
        Ok(())
    }

    pub fn batched_expert_gemv(
        &self,
        experts: &GpuExperts,
        layer: usize,
        part: usize,
        expert_ids: &CudaSlice<i32>,
        x: &CudaSlice<f32>,
        y: &CudaSlice<f32>, // [slots, rows]
        slots: usize,
    ) -> Result<()> {
        let rows = experts.rows[part] as i32;
        let in_dim = experts.in_dim[part] as i32;
        let slots_i = slots as i32;
        self.launch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            self.stream
                .launch_builder(&self.batched4)
                .arg(&experts.stacked[layer][part])
                .arg(&experts.stacked_scales[layer][part])
                .arg(&experts.stacked_biases[layer][part])
                .arg(expert_ids)
                .arg(x)
                .arg(y)
                .arg(&rows)
                .arg(&in_dim)
                .arg(&slots_i)
                .launch(LaunchConfig {
                    grid_dim: ((rows as u32).div_ceil(4), slots as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("batched launch failed: {e}"))?;
        Ok(())
    }

    pub fn lora_add(
        &self,
        lora: &LoraGpu,
        x: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<()> {
        let in_i = in_dim as i32;
        let out_i = out_dim as i32;
        unsafe {
            self.stream
                .launch_builder(&self.lora_add)
                .arg(&lora.a)
                .arg(&lora.b)
                .arg(x)
                .arg(y)
                .arg(&lora.rank)
                .arg(&in_i)
                .arg(&out_i)
                .launch(LaunchConfig {
                    grid_dim: (out_dim.div_ceil(256).min(64) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("lora launch failed: {e}"))?;
        Ok(())
    }

    pub fn lora_add_test(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
        rank: usize,
        in_dim: usize,
        out_dim: usize,
        blocks: usize,
    ) -> Result<()> {
        let rank_i = rank as i32;
        let in_i = in_dim as i32;
        let out_i = out_dim as i32;
        unsafe {
            self.stream
                .launch_builder(&self.lora_add)
                .arg(a)
                .arg(b)
                .arg(x)
                .arg(y)
                .arg(&rank_i)
                .arg(&in_i)
                .arg(&out_i)
                .launch(LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("lora launch failed: {e}"))?;
        Ok(())
    }

    /// slotx down with silu + router-weight fold inline (§3c(3)).
    #[allow(clippy::too_many_arguments)]
    pub fn batched_expert_gemv_slotx_silu(
        &self,
        experts: &GpuExperts,
        layer: usize,
        expert_ids: &CudaSlice<i32>,
        g: &CudaSlice<f32>,
        u: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
        slots: usize,
    ) -> Result<()> {
        let rows = experts.rows[2] as i32;
        let in_dim = experts.in_dim[2] as i32;
        let slots_i = slots as i32;
        self.launch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            self.stream
                .launch_builder(&self.batched4silu)
                .arg(&experts.stacked[layer][2])
                .arg(&experts.stacked_scales[layer][2])
                .arg(&experts.stacked_biases[layer][2])
                .arg(expert_ids)
                .arg(g)
                .arg(u)
                .arg(w)
                .arg(y)
                .arg(&rows)
                .arg(&in_dim)
                .arg(&slots_i)
                .launch(LaunchConfig {
                    grid_dim: ((rows as u32).div_ceil(4), slots as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("slotx+silu launch failed: {e}"))?;
        Ok(())
    }

    /// Single int4 projection over silu(g)*u with LoRA folded in.
    pub fn gemv1_silu(
        &self,
        q: &GpuQuant,
        g: &CudaSlice<f32>,
        u: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
        rank: usize,
    ) -> Result<()> {
        let rows_i = q.out_dim as i32;
        let in_i = q.in_dim as i32;
        let rank_i = rank as i32;
        // Dummy LoRA pointers when the projection has none.
        let (la, lb) = match &q.lora {
            Some(l) => (&l.a, &l.b),
            None => (&q.scales, &q.biases),
        };
        self.launch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            self.stream
                .launch_builder(&self.k_gemv1_silu)
                .arg(&q.packed)
                .arg(&q.scales)
                .arg(&q.biases)
                .arg(la)
                .arg(lb)
                .arg(y)
                .arg(&rows_i)
                .arg(g)
                .arg(u)
                .arg(&in_i)
                .arg(&rank_i)
                .launch(LaunchConfig {
                    grid_dim: ((q.out_dim as u32).div_ceil(8), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("gemv1+silu launch failed: {e}"))?;
        Ok(())
    }

    pub fn counted_sync(&self) -> Result<()> {
        self.sync_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stream.synchronize().context("kernel sync failed")
    }

    /// GPU silu(g)*u into y; n = lane count. Eliminates the phase2->3
    /// host round trip (the inner never touches the host).
    pub fn silu_mul(
        &self,
        g: &CudaSlice<f32>,
        u: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
        n: usize,
    ) -> Result<()> {
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(&self.silu_mul)
                .arg(g)
                .arg(u)
                .arg(y)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: ((n as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("silu launch failed: {e}"))?;
        Ok(())
    }

    pub fn gdn_conv_launch(&self, g: &GpuGdn, qkv_y: &CudaSlice<f32>) -> Result<()> {
        let conv_dim = g.conv_dim as i32;
        let kernel = g.kernel as i32;
        self.launch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            self.stream
                .launch_builder(&self.gdn_conv)
                .arg(qkv_y)
                .arg(&g.conv1d_w)
                .arg(&g.conv_state)
                .arg(&g.conv_out)
                .arg(&conv_dim)
                .arg(&kernel)
                .launch(LaunchConfig {
                    grid_dim: ((g.conv_dim as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("gdn conv launch failed: {e}"))?;
        Ok(())
    }

    /// conv + heads in one call, chained in-stream (the cross-crate form
    /// — GpuGdn's scratch fields are private to this module).
    pub fn gdn_conv_heads(
        &self,
        g: &GpuGdn,
        qkv_y: &CudaSlice<f32>,
        z: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        a: &CudaSlice<f32>,
    ) -> Result<()> {
        self.gdn_conv_launch(g, qkv_y)?;
        self.gdn_heads_launch(g, &g.conv_out, z, b, a)
    }

    /// The heads kernel's output scratch (out_proj's input).
    pub fn gdn_out_buf<'a>(&self, g: &'a GpuGdn) -> &'a CudaSlice<f32> {
        &g.out
    }

    /// The conv kernel's output scratch (heads' input).
    pub fn gdn_conv_buf<'a>(&self, g: &'a GpuGdn) -> &'a CudaSlice<f32> {
        &g.conv_out
    }

    /// Mutable state accessors for the reject-restore (dtod scratch copy).
    pub fn gdn_state_mut<'a>(
        &self,
        g: &'a mut GpuGdn,
    ) -> (&'a mut CudaSlice<f32>, &'a mut CudaSlice<f32>) {
        (&mut g.conv_state, &mut g.recurrent)
    }

    /// Speculative-verify access to a GDN layer's tensors:
    /// (conv_w, a_log, dt_bias, norm_w, conv_state, recurrent).
    /// Kernels write the state buffers through these shared refs, as
    /// everywhere else in this module.
    #[allow(clippy::type_complexity)]
    pub fn gdn_parts<'a>(
        &self,
        g: &'a GpuGdn,
    ) -> (
        &'a CudaSlice<f32>,
        &'a CudaSlice<f32>,
        &'a CudaSlice<f32>,
        &'a CudaSlice<f32>,
        &'a CudaSlice<f32>,
        &'a CudaSlice<f32>,
    ) {
        (
            &g.conv1d_w,
            &g.a_log,
            &g.dt_bias,
            &g.norm_w,
            &g.conv_state,
            &g.recurrent,
        )
    }

    pub fn gdn_heads_launch(
        &self,
        g: &GpuGdn,
        conv_out: &CudaSlice<f32>,
        z: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        a: &CudaSlice<f32>,
    ) -> Result<()> {
        let num_v = g.num_v as i32;
        let num_k = g.num_k as i32;
        let dk = g.dk as i32;
        let dv = g.dv as i32;
        self.launch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            self.stream
                .launch_builder(&self.gdn_heads)
                .arg(conv_out)
                .arg(z)
                .arg(b)
                .arg(a)
                .arg(&g.a_log)
                .arg(&g.dt_bias)
                .arg(&g.norm_w)
                .arg(&g.recurrent)
                .arg(&g.out)
                .arg(&num_v)
                .arg(&num_k)
                .arg(&dk)
                .arg(&dv)
                .arg(&g.scale)
                .arg(&g.eps)
                .launch(LaunchConfig {
                    grid_dim: (g.num_v as u32, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("gdn heads launch failed: {e}"))?;
        Ok(())
    }

    /// Harness getters for kernel-level parity checks.
    pub fn gdn_conv_out(&self, g: &GpuGdn) -> Result<Vec<f32>> {
        let mut host = vec![0f32; g.conv_out.len()];
        self.stream.memcpy_dtoh(&g.conv_out, &mut host)?;
        self.stream.synchronize()?;
        Ok(host)
    }

    pub fn gdn_out(&self, g: &GpuGdn) -> Result<Vec<f32>> {
        let mut host = vec![0f32; g.out.len()];
        self.stream.memcpy_dtoh(&g.out, &mut host)?;
        self.stream.synchronize()?;
        Ok(host)
    }
}

impl GpuQuant {
    /// Raw weight buffers for the batch-2 verify GEMV (spec.rs drives its
    /// own kernel; launch() stays the single-column path).
    #[allow(clippy::type_complexity)]
    pub fn gemv_parts(&self) -> (&CudaSlice<u32>, &CudaSlice<f32>, &CudaSlice<f32>) {
        (&self.packed, &self.scales, &self.biases)
    }

    /// Synchronous GEMV: uploads x, launches, copies y back.
    pub fn matvec_sync(&self, ctx: &GpuContext, x: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            x.len() == self.in_dim,
            "matvec x length {} != {}",
            x.len(),
            self.in_dim
        );
        let dx = ctx.stream.clone_htod(x).context("x upload failed")?;
        let dy = ctx
            .stream
            .alloc_zeros::<f32>(self.out_dim)
            .context("y alloc failed")?;
        self.launch(ctx, &dx, &dy)?;
        ctx.counted_sync()?;
        let mut host = vec![0f32; self.out_dim];
        ctx.stream
            .memcpy_dtoh(&dy, &mut host)
            .context("y download failed")?;
        Ok(host)
    }

    pub fn y_ref(&self) -> &CudaSlice<f32> {
        &self.y
    }

    /// Device buffers for external wide-shape kernels (additive).
    pub fn tensors(&self) -> (&CudaSlice<u32>, &CudaSlice<f32>, &CudaSlice<f32>) {
        (&self.packed, &self.scales, &self.biases)
    }

    /// This projection as a grouped-GEMV segment.
    pub fn group_seg(&self) -> GroupSeg<'_> {
        GroupSeg {
            packed: &self.packed,
            scales: &self.scales,
            biases: &self.biases,
            y: &self.y,
            rows: self.out_dim,
            lora: self.lora.as_ref().map(|l| (&l.a, &l.b)),
        }
    }

    /// A zero-row segment reusing another projection's pointers (slots the
    /// group does not use still need valid pointers).
    pub fn empty_seg_like(&self) -> GroupSeg<'_> {
        GroupSeg {
            packed: &self.packed,
            scales: &self.scales,
            biases: &self.biases,
            y: &self.y,
            rows: 0,
            lora: None,
        }
    }

    pub fn launch(&self, ctx: &GpuContext, x: &CudaSlice<f32>, y: &CudaSlice<f32>) -> Result<()> {
        // Small out_dims get the 4-row variant: 4x blocks, more loads in
        // flight. Same per-row arithmetic — only the grid tiling differs.
        let r4 = self.out_dim <= 512;
        let function = match (self.bits, r4) {
            (4, false) => &ctx.gemv4,
            (4, true) => &ctx.gemv4_r4,
            (_, false) => &ctx.gemv8,
            (_, true) => &ctx.gemv8_r4,
        };
        let out_dim = self.out_dim as i32;
        let in_dim = self.in_dim as i32;
        let rpb = if r4 { 4 } else { 16 };
        unsafe {
            ctx.stream
                .launch_builder(function)
                .arg(&self.packed)
                .arg(&self.scales)
                .arg(&self.biases)
                .arg(x)
                .arg(y)
                .arg(&out_dim)
                .arg(&in_dim)
                .launch(LaunchConfig {
                    // ROWS_PER_BLOCK = 16 in edge0_gemv.cu / batched_gemv.cu.
                    grid_dim: ((self.out_dim as u32).div_ceil(rpb), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("gemv launch failed: {e}"))?;
        if let Some(lora) = &self.lora {
            ctx.lora_add(lora, x, y, self.in_dim, self.out_dim)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::Edge0Weights;
    use std::path::Path;

    #[test]
    fn gpu_gemv_matches_cpu_matvec_on_a_real_projection() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("no CUDA device ({e}); skipping gpu test");
                return;
            }
        };
        let weights = Edge0Weights::open(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/Edge0/Edge0-35B-A3B-preview"
        )))
        .unwrap();
        let name = "language_model.model.layers.3.self_attn.q_proj";
        let quant = weights.quant_projection(name).unwrap();
        let gpu = ctx.upload(&quant, None).unwrap();
        let x: Vec<f32> = (0..quant.in_dim)
            .map(|i| ((i as f32) * 0.037).sin() * 0.5)
            .collect();
        let cpu = quant.matvec(&x, None);
        let gpu_y = gpu.matvec_sync(&ctx, &x).unwrap();
        assert_eq!(cpu.len(), gpu_y.len());
        let max_rel = cpu
            .iter()
            .zip(&gpu_y)
            .map(|(&c, &g)| (c - g).abs() / c.abs().max(1.0))
            .fold(0.0_f32, f32::max);
        eprintln!(
            "gpu[:4] {:?} cpu[:4] {:?} max_rel {max_rel}",
            &gpu_y[..4.min(gpu_y.len())],
            &cpu[..4.min(cpu.len())]
        );
        assert!(max_rel < 1e-3, "gpu vs cpu max rel diff {max_rel}");
    }
}

/// Resident static projections keyed by checkpoint name, plus persistent
/// per-width x uploads and a host scratch for results — the decode loop
/// runs zero device allocations per token.
#[derive(Default, Clone)]
pub struct BatchTrace {
    pub htod_us: u64,
    pub launch_us: u64,
    pub sync_us: u64,
    pub dtoh_us: u64,
    pub n_batch: u64,
    pub prepared_us: u64,
    pub n_prepared: u64,
}

#[derive(Default, Clone, Copy)]
pub struct MoeTrace {
    pub router_us: u64,
    pub enq_us: u64,
    pub silu_us: u64,
    pub down_us: u64,
    pub shared_us: u64,
    pub tail_us: u64,
    pub calls: u64,
}

/// Send wrapper: the decode graph is created, launched, and read from the
/// single decode thread only — cudarc marks CudaGraph !thread-safe because
/// concurrent graph API use is UB, and this runtime has no second thread.
pub struct DecodeGraph(pub cudarc::driver::safe::CudaGraph);
unsafe impl Send for DecodeGraph {}

pub struct GpuRuntime {
    pub ctx: GpuContext,
    pub proj: std::collections::HashMap<String, GpuQuant>,
    /// Uniform LoRA rank across adapters (0 = none resident).
    pub lora_rank: usize,
    /// GridBarrier state for moe_mega (count, sense — returns to 0/0 after
    /// an even number of barriers).
    pub mega_bar: CudaSlice<i32>,
    /// moe_mega dispatch, read once at init (EDGE0_MEGA).
    pub use_moe_mega: bool,
    /// Device-resident GDN layers, indexed by GDN-layer order.
    pub gdn: Vec<GpuGdn>,
    pub res: Option<ResidentState>,
    moe_trace: std::sync::Mutex<Option<(usize, MoeTrace)>>,
    pub batch_trace: std::sync::Mutex<BatchTrace>,
    x_bufs: std::sync::Mutex<std::collections::HashMap<usize, CudaSlice<f32>>>,
    // Per-SLOT inner buffers: same-width live inputs (4 expert inners +
    // shared inner, all 512) cannot share a per-width buffer — see the
    // aliasing trap in docs/edge0-design.md §3c(2).
    inner_slots: std::sync::Mutex<Vec<Option<CudaSlice<f32>>>>,
    expert_ids: std::sync::Mutex<CudaSlice<i32>>,
    batched_gate_y: std::sync::Mutex<CudaSlice<f32>>,
    batched_up_y: std::sync::Mutex<CudaSlice<f32>>,
    batched_down_y: std::sync::Mutex<CudaSlice<f32>>,
    batched_inner_y: std::sync::Mutex<CudaSlice<f32>>,
}

impl GpuRuntime {
    pub fn new(
        ctx: GpuContext,
        proj: std::collections::HashMap<String, GpuQuant>,
        gdn: Vec<GpuGdn>,
        res: Option<ResidentState>,
        lora_rank: usize,
    ) -> Self {
        let expert_ids = ctx.stream.alloc_zeros::<i32>(4).expect("expert ids");
        let batched_gate_y = ctx.stream.alloc_zeros::<f32>(4 * 512).expect("gate y");
        let batched_up_y = ctx.stream.alloc_zeros::<f32>(4 * 512).expect("up y");
        let batched_down_y = ctx.stream.alloc_zeros::<f32>(4 * 2048).expect("down y");
        let batched_inner_y = ctx.stream.alloc_zeros::<f32>(4 * 512).expect("inner y");
        let mega_bar = ctx.upload_i32(&[0i32, 0]).expect("mega barrier");
        let use_moe_mega = std::env::var_os("EDGE0_MEGA").is_some();
        Self {
            ctx,
            proj,
            gdn,
            res,
            lora_rank,
            mega_bar,
            use_moe_mega,
            x_bufs: std::sync::Mutex::new(Default::default()),
            inner_slots: std::sync::Mutex::new((0..8).map(|_| None).collect()),
            expert_ids: std::sync::Mutex::new(expert_ids),
            batched_gate_y: std::sync::Mutex::new(batched_gate_y),
            batched_up_y: std::sync::Mutex::new(batched_up_y),
            batched_down_y: std::sync::Mutex::new(batched_down_y),
            batched_inner_y: std::sync::Mutex::new(batched_inner_y),
            batch_trace: std::sync::Mutex::new(Default::default()),
            moe_trace: std::sync::Mutex::new(
                std::env::var("EDGE0_TRACE_LAYER")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .map(|layer| (layer, MoeTrace::default())),
            ),
        }
    }

    fn trace_seg(&self, layer: usize, seg: u8, since: std::time::Instant) {
        if let Ok(mut tr) = self.moe_trace.lock() {
            if let Some((l, t)) = tr.as_mut() {
                if *l != layer {
                    return;
                }
                let us = since.elapsed().as_micros() as u64;
                match seg {
                    0 => {
                        t.router_us += us;
                        t.calls += 1;
                    }
                    1 => t.enq_us += us,
                    2 => t.silu_us += us,
                    3 => t.down_us += us,
                    4 => t.shared_us += us,
                    _ => t.tail_us += us,
                }
            }
        }
    }

    pub fn moe_trace_summary(&self) -> Option<(usize, MoeTrace)> {
        self.moe_trace.lock().ok().and_then(|t| *t)
    }

    pub fn batch_trace_summary(&self) -> BatchTrace {
        self.batch_trace
            .lock()
            .map(|t| t.clone())
            .unwrap_or_default()
    }

    /// MoE block, three phases with THREE syncs per layer (router must be
    /// read before routing is known): (1) router; (2) gate+up for routed
    /// experts AND shared gate/up/gate-scalar on one x upload; (3) downs
    /// into per-slot inner buffers. Host does silu-mul and the weighted
    /// combine. Returns (router_logits, expert_outputs, shared_inner_ready)
    /// with shared down output read separately via `read_shared`.
    #[allow(clippy::type_complexity)]
    pub fn moe_router_dx(
        &self,
        layer: usize,
        router: &GpuQuant,
        dx: &CudaSlice<f32>,
    ) -> Result<Vec<f32>> {
        let t0 = std::time::Instant::now();
        router.launch(&self.ctx, dx, router.y_ref())?;
        self.ctx.counted_sync()?;
        let mut logits = vec![0f32; router.out_dim];
        self.ctx.stream.memcpy_dtoh(router.y_ref(), &mut logits)?;
        self.trace_seg(layer, 0, t0);
        Ok(logits)
    }

    /// Whole GDN layer, one x upload + one sync: the in_proj GEMVs feed the
    /// conv and heads kernels directly (qkv/z/b/a never touch the host) and
    /// out_proj consumes the heads output in place.
    /// Resident variant: consumes the normed input from device memory and
    /// leaves the layer output in out_proj's y — no sync, no host round
    /// trip; the caller adds it into `hidden`.
    pub fn gdn_layer_dx(
        &self,
        gdn_index: usize,
        qkv: &GpuQuant,
        z: &GpuQuant,
        b: &GpuQuant,
        a: &GpuQuant,
        out_proj: &GpuQuant,
        dx: &CudaSlice<f32>,
    ) -> Result<()> {
        let g = &self.gdn[gdn_index];
        // One launch for qkv/z/b/a: same x, all int4, LoRA folded in the
        // epilogue (was 4 GEMVs + 4 lora_adds).
        let segs = [qkv.group_seg(), z.group_seg(), b.group_seg(), a.group_seg()];
        let rank = if segs.iter().any(|s| s.lora.is_some()) {
            self.lora_rank
        } else {
            0
        };
        self.ctx.glue_group4(&segs, dx, qkv.in_dim, rank)?;
        self.ctx.gdn_conv_launch(g, qkv.y_ref())?;
        self.ctx
            .gdn_heads_launch(g, &g.conv_out, z.y_ref(), b.y_ref(), a.y_ref())?;
        let segs = [
            out_proj.group_seg(),
            out_proj.empty_seg_like(),
            out_proj.empty_seg_like(),
            out_proj.empty_seg_like(),
        ];
        let rank = if segs[0].lora.is_some() {
            self.lora_rank
        } else {
            0
        };
        self.ctx.glue_group4(&segs, &g.out, out_proj.in_dim, rank)?;
        Ok(())
    }

    pub fn matvec(&self, name: &str, x: &[f32]) -> Option<Result<Vec<f32>>> {
        self.proj.get(name).map(|q| q.matvec_sync(&self.ctx, x))
    }

    pub fn ctx_ref(&self) -> &GpuContext {
        &self.ctx
    }

    /// Two-phase MoE (router needs one sync for host top-k; everything
    /// else — routed gate/up/down for 4 experts AND shared gate/up/down —
    /// runs with GPU silu and ONE further sync). Returns weighted combine
    /// inputs: (down outputs for routed experts, shared down output).
    #[allow(clippy::type_complexity)]
    pub fn moe_fused_dx(
        &self,
        experts: &GpuExperts,
        layer: usize,
        chosen: &[usize],
        shared: (&GpuQuant, &GpuQuant, &GpuQuant, &GpuQuant),
        dx: &CudaSlice<f32>,
    ) -> Result<(f32, Vec<Vec<f32>>)> {
        let t0 = std::time::Instant::now();
        // Batched gate/up: ONE launch per projection covers all routed
        // experts (per-expert dispatch measured 54 µs/kernel × 8/layer).
        let slots = chosen.len();
        let _rows = experts.rows[0];
        {
            let mut ids = self.expert_ids.lock().expect("expert ids");
            if ids.len() < slots {
                *ids = self
                    .ctx
                    .stream
                    .alloc_zeros::<i32>(slots)
                    .context("ids alloc")?;
            }
            let ids_host: Vec<i32> = chosen.iter().map(|&e| e as i32).collect();
            self.ctx.stream.memcpy_htod(&ids_host, &mut *ids)?;
            self.ctx.batched_expert_gemv(
                experts,
                layer,
                0,
                &ids,
                dx,
                &self.batched_gate_y.lock().expect("gy"),
                slots,
            )?;
            self.ctx.batched_expert_gemv(
                experts,
                layer,
                1,
                &ids,
                dx,
                &self.batched_up_y.lock().expect("uy"),
                slots,
            )?;
        }
        shared.0.launch(&self.ctx, dx, shared.0.y_ref())?;
        shared.1.launch(&self.ctx, dx, shared.1.y_ref())?;
        shared.2.launch(&self.ctx, dx, shared.2.y_ref())?;
        self.trace_seg(layer, 1, t0);
        let t1 = std::time::Instant::now();
        let width = experts.rows[0].max(1);
        let lanes = chosen.len() * width;
        {
            let g = self.batched_gate_y.lock().expect("gy");
            let u = self.batched_up_y.lock().expect("uy");
            let mut inner = self.batched_inner_y.lock().expect("iy");
            if inner.len() < lanes {
                *inner = self
                    .ctx
                    .stream
                    .alloc_zeros::<f32>(lanes)
                    .context("inner scratch alloc")?;
            }
            self.ctx.silu_mul(&g, &u, &inner, lanes)?;
        }
        self.trace_seg(layer, 2, t1);
        let t2 = std::time::Instant::now();
        self.ctx.batched_expert_gemv_slotx(
            experts,
            layer,
            2,
            &self.expert_ids.lock().expect("expert ids"),
            &self.batched_inner_y.lock().expect("iy"),
            &self.batched_down_y.lock().expect("dy"),
            chosen.len(),
        )?;
        self.trace_seg(layer, 3, t2);
        let t3 = std::time::Instant::now();
        let shared_w = shared.0.out_dim;
        // Shared expert: silu on device (its gate/up sit in their own
        // GpuQuant y buffers), down on the shared slot buffer.
        {
            let mut slots = self.inner_slots.lock().expect("inner slots");
            while slots.len() < chosen.len() + 1 {
                slots.push(None);
            }
            let stale = slots
                .get(chosen.len())
                .and_then(|s| s.as_ref().map(|b| b.len() != shared_w))
                .unwrap_or(true);
            if stale {
                slots[chosen.len()] = Some(
                    self.ctx
                        .stream
                        .alloc_zeros::<f32>(shared_w)
                        .context("shared slot alloc")?,
                );
            }
            let sg = shared.0.y_ref();
            let su = shared.1.y_ref();
            self.ctx.silu_mul(
                sg,
                su,
                slots[chosen.len()].as_ref().expect("shared slot"),
                shared_w,
            )?;
            shared.3.launch(
                &self.ctx,
                slots[chosen.len()].as_ref().expect("shared slot"),
                shared.3.y_ref(),
            )?;
        }
        self.trace_seg(layer, 4, t3);
        let t4 = std::time::Instant::now();
        self.ctx.counted_sync()?;
        // Batched down results from the scratch: [slots, rows].
        let down_rows = experts.rows[2];
        let mut down_host = vec![0f32; chosen.len() * down_rows];
        {
            let dy = self.batched_down_y.lock().expect("dy");
            self.ctx.stream.memcpy_dtoh(&*dy, &mut down_host)?;
        }
        let mut outs = Vec::with_capacity(chosen.len());
        for s in 0..chosen.len() {
            outs.push(down_host[s * down_rows..(s + 1) * down_rows].to_vec());
        }
        let mut sv = vec![0f32; shared.3.out_dim];
        self.ctx.stream.memcpy_dtoh(shared.3.y_ref(), &mut sv)?;
        let mut scalar = 0f32;
        self.ctx
            .stream
            .memcpy_dtoh(shared.2.y_ref(), std::slice::from_mut(&mut scalar))?;

        self.trace_seg(layer, 5, t4);
        let mut all_outs = outs;
        all_outs.push(sv);
        Ok((scalar, all_outs))
    }

    pub fn read_y(&self, q: &GpuQuant) -> Result<Vec<f32>> {
        let mut host = vec![0f32; q.out_dim];
        self.ctx
            .stream
            .memcpy_dtoh(q.y_ref(), &mut host)
            .context("y download failed")?;
        Ok(host)
    }

    /// N projections over the SAME x on one upload + one sync; each
    /// projection keeps its own persistent y (outputs never alias).
    pub fn batch_matvec(&self, names: &[&str], x: &[f32]) -> Option<Result<Vec<Vec<f32>>>> {
        let quants: Vec<&GpuQuant> = names
            .iter()
            .map(|n| self.proj.get(*n))
            .collect::<Option<Vec<_>>>()?;
        let t0 = std::time::Instant::now();
        let mut guard = self.x_bufs.lock().ok()?;
        let entry = guard.entry(x.len()).or_insert_with(|| {
            self.ctx
                .stream
                .alloc_zeros::<f32>(x.len())
                .expect("x alloc")
        });
        self.ctx.stream.memcpy_htod(x, entry).ok()?;
        let t1 = std::time::Instant::now();
        for q in &quants {
            if q.launch(&self.ctx, entry, q.y_ref()).is_err() {
                return None;
            }
        }
        drop(guard);
        let t2 = std::time::Instant::now();
        self.ctx.counted_sync().ok()?;
        let t3 = std::time::Instant::now();
        let mut outs = Vec::with_capacity(quants.len());
        for q in &quants {
            let mut host = vec![0f32; q.out_dim];
            if self.ctx.stream.memcpy_dtoh(q.y_ref(), &mut host).is_err() {
                return None;
            }
            outs.push(host);
        }
        let t4 = std::time::Instant::now();
        if let Ok(mut tr) = self.batch_trace.lock() {
            tr.htod_us += t1.duration_since(t0).as_micros() as u64;
            tr.launch_us += t2.duration_since(t1).as_micros() as u64;
            tr.sync_us += t3.duration_since(t2).as_micros() as u64;
            tr.dtoh_us += t4.duration_since(t3).as_micros() as u64;
            tr.n_batch += 1;
        }
        Some(Ok(outs))
    }

    pub fn prepared(&self, name: &str, x: &[f32]) -> Option<Result<Vec<f32>>> {
        let tp = std::time::Instant::now();
        let q = self.proj.get(name)?;
        let mut guard = self.x_bufs.lock().expect("x buffers");
        let entry = guard.entry(x.len()).or_insert_with(|| {
            self.ctx
                .stream
                .alloc_zeros::<f32>(x.len())
                .expect("persistent x alloc")
        });
        if self.ctx.stream.memcpy_htod(x, entry).is_err() {
            return None;
        }
        if q.launch(&self.ctx, entry, &q.y).is_err() {
            return None;
        }
        drop(guard);
        // counted, not raw: ~41 uncounted syncs/token hid here (out_proj/o_proj/lm_head).
        if self.ctx.counted_sync().is_err() {
            return None;
        }
        let r = self.read_y(q);
        if let Ok(mut tr) = self.batch_trace.lock() {
            tr.prepared_us += tp.elapsed().as_micros() as u64;
            tr.n_prepared += 1;
        }
        Some(r)
    }
}

/// Device-resident GDN layer: gating statics plus conv and recurrent state
/// (recurrent stored transposed per head — [num_v, dv, dk] — so the
/// per-column sweeps in edge0_gdn_heads are contiguous).
pub struct GpuGdn {
    conv1d_w: CudaSlice<f32>,
    a_log: CudaSlice<f32>,
    dt_bias: CudaSlice<f32>,
    norm_w: CudaSlice<f32>,
    conv_state: CudaSlice<f32>,
    recurrent: CudaSlice<f32>,
    conv_out: CudaSlice<f32>,
    out: CudaSlice<f32>,
    conv_dim: usize,
    kernel: usize,
    num_v: usize,
    num_k: usize,
    dk: usize,
    dv: usize,
    scale: f32,
    eps: f32,
}

impl GpuGdn {
    pub fn upload(
        ctx: &GpuContext,
        conv1d: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        norm: &[f32],
        conv_dim: usize,
        kernel: usize,
        num_v: usize,
        num_k: usize,
        dk: usize,
        dv: usize,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            conv1d_w: ctx.upload_f32(conv1d)?,
            a_log: ctx.upload_f32(a_log)?,
            dt_bias: ctx.upload_f32(dt_bias)?,
            norm_w: ctx.upload_f32(norm)?,
            conv_state: ctx
                .stream
                .alloc_zeros::<f32>(conv_dim * (kernel - 1))
                .context("conv state")?,
            recurrent: ctx
                .stream
                .alloc_zeros::<f32>(num_v * dk * dv)
                .context("recurrent state")?,
            conv_out: ctx
                .stream
                .alloc_zeros::<f32>(conv_dim)
                .context("conv scratch")?,
            out: ctx
                .stream
                .alloc_zeros::<f32>(num_v * dv)
                .context("heads scratch")?,
            conv_dim,
            kernel,
            num_v,
            num_k,
            dk,
            dv,
            scale: 1.0 / (dk as f32).sqrt(),
            eps,
        })
    }
}

/// All experts of all layers resident as whole stacked tensors
/// ([256 experts, rows, in/8] per projection per layer) — the batched
/// kernels address experts by first-dim index (gather_qmm shape).
pub struct GpuExperts {
    pub stacked: Vec<[CudaSlice<u32>; 3]>,
    pub stacked_scales: Vec<[CudaSlice<f32>; 3]>,
    pub stacked_biases: Vec<[CudaSlice<f32>; 3]>,
    /// Projection geometry: rows (out) and in_dim per projection part.
    pub rows: [usize; 3],
    pub in_dim: [usize; 3],
}

/// Buffers for the device-resident decode path: hidden state, the normed
/// layer input, attention scratch, per-layer norm weights and fixed-capacity
/// KV caches (fixed capacity so step 3 can graph-capture the decode step).
pub struct ResidentState {
    pub hidden: CudaSlice<f32>,
    pub x1: CudaSlice<f32>,
    q_out: CudaSlice<f32>,
    gate_out: CudaSlice<f32>,
    attn_scratch: CudaSlice<f32>,
    moe_stage: std::sync::Mutex<CudaSlice<f32>>,
    /// [layer][input, post] norm weights.
    ln: Vec<[CudaSlice<f32>; 2]>,
    /// [attn layer][q, k] norm weights, indexed like `kv`.
    attn_norms: Vec<[CudaSlice<f32>; 2]>,
    final_norm: CudaSlice<f32>,
    /// [attn layer][max_ctx, kv_stride].
    kv_keys: Vec<CudaSlice<f32>>,
    kv_values: Vec<CudaSlice<f32>>,
    pub max_ctx: usize,
    kv_stride: usize,
    /// Device-side routing state, written by edge0_router_topk.
    pub topk_ids: CudaSlice<i32>,
    pub topk_w: CudaSlice<f32>,
    /// Argmax output / embed row selector (closed decode loop).
    pub next_token: std::sync::Mutex<CudaSlice<i32>>,
    /// Device-side position counter (rope/KV length).
    pub pos_buf: std::sync::Mutex<CudaSlice<i32>>,
}

impl ResidentState {
    pub fn upload(
        ctx: &GpuContext,
        hidden_size: usize,
        layer_norms: &[(Vec<f32>, Vec<f32>)],
        attn_norms: &[(Vec<f32>, Vec<f32>)],
        final_norm: &[f32],
        q_total: usize,
        kv_stride: usize,
        num_attn_layers: usize,
    ) -> Result<Self> {
        let max_ctx = std::env::var("EDGE0_MAX_CTX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n >= 1)
            .unwrap_or(4096);
        // edge0_attn_scores keeps the step scores in shared memory.
        ensure!(
            max_ctx <= 8192,
            "EDGE0_MAX_CTX {max_ctx} exceeds the kernel cap 8192"
        );
        let mut kv_keys = Vec::with_capacity(num_attn_layers);
        let mut kv_values = Vec::with_capacity(num_attn_layers);
        for _ in 0..num_attn_layers {
            kv_keys.push(
                ctx.stream
                    .alloc_zeros::<f32>(max_ctx * kv_stride)
                    .context("kv keys")?,
            );
            kv_values.push(
                ctx.stream
                    .alloc_zeros::<f32>(max_ctx * kv_stride)
                    .context("kv values")?,
            );
        }
        Ok(Self {
            hidden: ctx
                .stream
                .alloc_zeros::<f32>(hidden_size)
                .context("hidden")?,
            x1: ctx.stream.alloc_zeros::<f32>(hidden_size).context("x1")?,
            q_out: ctx.stream.alloc_zeros::<f32>(q_total).context("q out")?,
            gate_out: ctx.stream.alloc_zeros::<f32>(q_total).context("gate out")?,
            attn_scratch: ctx
                .stream
                .alloc_zeros::<f32>(q_total)
                .context("attn scratch")?,
            moe_stage: std::sync::Mutex::new(
                ctx.stream
                    .alloc_zeros::<f32>(hidden_size)
                    .context("moe stage")?,
            ),
            ln: layer_norms
                .iter()
                .map(|(i, p)| Ok([ctx.upload_f32(i)?, ctx.upload_f32(p)?]))
                .collect::<Result<Vec<_>>>()?,
            attn_norms: attn_norms
                .iter()
                .map(|(q, k)| Ok([ctx.upload_f32(q)?, ctx.upload_f32(k)?]))
                .collect::<Result<Vec<_>>>()?,
            final_norm: ctx.upload_f32(final_norm)?,
            kv_keys,
            kv_values,
            max_ctx,
            kv_stride,
            topk_ids: ctx.upload_i32(&[0; 4])?,
            topk_w: ctx.upload_f32(&[0f32; 4])?,
            next_token: std::sync::Mutex::new(ctx.upload_i32(&[0])?),
            pos_buf: std::sync::Mutex::new(ctx.upload_i32(&[0])?),
        })
    }
}

impl GpuContext {
    pub fn glue_embed_row(
        &self,
        embed: &GpuQuant,
        token: &CudaSlice<i32>,
        out: &CudaSlice<f32>,
    ) -> Result<()> {
        let in_dim = embed.in_dim as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k_embed_row)
                .arg(&embed.packed)
                .arg(&embed.scales)
                .arg(&embed.biases)
                .arg(token)
                .arg(out)
                .arg(&in_dim)
                .launch(LaunchConfig {
                    grid_dim: ((in_dim as u32 / 8).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("embed row launch failed: {e}"))?;
        Ok(())
    }

    pub fn glue_rmsnorm(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k_rmsnorm)
                .arg(x)
                .arg(w)
                .arg(out)
                .arg(&n_i)
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("rmsnorm launch failed: {e}"))?;
        Ok(())
    }

    pub fn glue_final_norm(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k_final_norm)
                .arg(x)
                .arg(w)
                .arg(out)
                .arg(&n_i)
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("final norm launch failed: {e}"))?;
        Ok(())
    }

    pub fn glue_add_inplace(
        &self,
        acc: &CudaSlice<f32>,
        delta: &CudaSlice<f32>,
        n: usize,
    ) -> Result<()> {
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k_add)
                .arg(acc)
                .arg(delta)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: ((n as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("add launch failed: {e}"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn glue_attn_qk(
        &self,
        res: &ResidentState,
        kv_index: usize,
        q_raw: &CudaSlice<f32>,
        q_norm_w: &CudaSlice<f32>,
        k_raw: &CudaSlice<f32>,
        k_norm_w: &CudaSlice<f32>,
        v_raw: &CudaSlice<f32>,
        position: &CudaSlice<i32>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f64,
    ) -> Result<()> {
        let kv_stride_i = res.kv_stride as i32;
        let heads_i = heads as i32;
        let kv_heads_i = kv_heads as i32;
        let hd_i = head_dim as i32;
        let rot_i = rotary_dim as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k_attn_qk)
                .arg(q_raw)
                .arg(q_norm_w)
                .arg(k_raw)
                .arg(k_norm_w)
                .arg(v_raw)
                .arg(&res.q_out)
                .arg(&res.gate_out)
                .arg(&res.kv_keys[kv_index])
                .arg(&res.kv_values[kv_index])
                .arg(position)
                .arg(&kv_stride_i)
                .arg(&heads_i)
                .arg(&kv_heads_i)
                .arg(&hd_i)
                .arg(&rot_i)
                .arg(&theta)
                .launch(LaunchConfig {
                    grid_dim: ((heads + 2 * kv_heads) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("attn qk launch failed: {e}"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn glue_attn_scores(
        &self,
        res: &ResidentState,
        kv_index: usize,
        q: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        position: &CudaSlice<i32>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        let kv_stride_i = res.kv_stride as i32;
        let heads_i = heads as i32;
        let kv_heads_i = kv_heads as i32;
        let hd_i = head_dim as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k_attn_scores)
                .arg(q)
                .arg(gate)
                .arg(&res.kv_keys[kv_index])
                .arg(&res.kv_values[kv_index])
                .arg(out)
                .arg(position)
                .arg(&kv_stride_i)
                .arg(&heads_i)
                .arg(&kv_heads_i)
                .arg(&hd_i)
                .arg(&scale)
                .launch(LaunchConfig {
                    grid_dim: (heads as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("attn scores launch failed: {e}"))?;
        Ok(())
    }
}

impl GpuRuntime {
    /// Resident decode-step primitives. All same-stream, no syncs — the
    /// layer body stays sequential launches until the router's host top-k.
    pub fn embed_into_hidden(&self, token: u32) -> Result<()> {
        self.set_next_token(token)?;
        self.embed_from_device_token()
    }

    /// hidden += delta; x1 = rmsnorm(hidden, ln[layer][which]) — fused.
    pub fn add_norm_x1(&self, layer: usize, which: usize, delta: &CudaSlice<f32>) -> Result<()> {
        let res = self.res.as_ref().expect("resident state");
        let n = res.hidden.len();
        self.ctx
            .glue_add_rmsnorm(&res.hidden, delta, &res.ln[layer][which], &res.x1, n, 1e-6)
    }

    /// x1 = rmsnorm(hidden, ln[layer][which]); which: 0 = input, 1 = post.
    pub fn rmsnorm_x1(&self, layer: usize, which: usize) -> Result<()> {
        let res = self.res.as_ref().expect("resident state");
        let n = res.hidden.len();
        let eps = 1e-6;
        self.ctx
            .glue_rmsnorm(&res.hidden, &res.ln[layer][which], &res.x1, n, eps)
    }

    pub fn add_residual(&self, delta: &CudaSlice<f32>) -> Result<()> {
        let res = self.res.as_ref().expect("resident state");
        let n = res.hidden.len();
        self.ctx.glue_add_inplace(&res.hidden, delta, n)
    }

    /// Attention layer on device: q/k/v GEMVs on x1, norm+rope+KV append,
    /// scores+softmax+gate, o_proj — output lands in o_proj's y.
    pub fn attn_layer(
        &self,
        kv_index: usize,
        q: &GpuQuant,
        k: &GpuQuant,
        v: &GpuQuant,
        o_proj: &GpuQuant,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f64,
    ) -> Result<()> {
        let res = self.res.as_ref().expect("resident state");
        let pos = res.pos_buf.lock().expect("pos");
        let segs = [
            q.group_seg(),
            k.group_seg(),
            v.group_seg(),
            q.empty_seg_like(),
        ];
        let rank = if segs.iter().any(|s| s.lora.is_some()) {
            self.lora_rank
        } else {
            0
        };
        self.ctx.glue_group4(&segs, &res.x1, q.in_dim, rank)?;
        let norms = &res.attn_norms[kv_index];
        self.ctx.glue_attn_qk(
            res,
            kv_index,
            q.y_ref(),
            &norms[0],
            k.y_ref(),
            &norms[1],
            v.y_ref(),
            &pos,
            heads,
            kv_heads,
            head_dim,
            rotary_dim,
            theta,
        )?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        self.ctx.glue_attn_scores(
            res,
            kv_index,
            &res.q_out,
            &res.gate_out,
            &res.attn_scratch,
            &pos,
            heads,
            kv_heads,
            head_dim,
            scale,
        )?;
        let segs = [
            o_proj.group_seg(),
            o_proj.empty_seg_like(),
            o_proj.empty_seg_like(),
            o_proj.empty_seg_like(),
        ];
        let rank = if segs[0].lora.is_some() {
            self.lora_rank
        } else {
            0
        };
        self.ctx
            .glue_group4(&segs, &res.attn_scratch, o_proj.in_dim, rank)?;
        Ok(())
    }

    /// Final zero-centered norm applied to hidden in place (x1 reuse).
    pub fn final_norm(&self) -> Result<()> {
        let res = self.res.as_ref().expect("resident state");
        let n = res.hidden.len();
        self.ctx
            .glue_final_norm(&res.hidden, &res.final_norm, &res.x1, n, 1e-6)
    }

    pub fn read_hidden_x1(&self) -> Result<Vec<f32>> {
        let res = self.res.as_ref().expect("resident state");
        let mut host = vec![0f32; res.x1.len()];
        self.ctx.stream.memcpy_dtoh(&res.x1, &mut host)?;
        self.ctx.counted_sync()?;
        Ok(host)
    }

    pub fn hidden_ref(&self) -> &CudaSlice<f32> {
        &self.res.as_ref().expect("resident state").hidden
    }

    /// lm_head GEMV over the device hidden — no htod of the hidden state.
    pub fn lm_logits(&self) -> Result<Vec<f32>> {
        let lm = self
            .proj
            .get("language_model.lm_head")
            .expect("lm_head resident");
        lm.launch(&self.ctx, self.hidden_ref(), lm.y_ref())?;
        self.ctx.counted_sync()?;
        self.read_y(lm)
    }

    /// Upload the host-combined MoE output into the y of a scratch GpuQuant
    /// and add it into hidden. Reuses moe router's y as staging (the router
    /// output was already consumed by host top-k).
    pub fn add_moe_residual(&self, y: &[f32]) -> Result<()> {
        let res = self.res.as_ref().expect("resident state");
        let n = res.hidden.len();
        ensure!(y.len() == n, "moe output len {} != hidden {n}", y.len());
        let mut stage = res.moe_stage.lock().expect("moe stage");
        self.ctx.stream.memcpy_htod(y, &mut *stage)?;
        self.ctx.glue_add_inplace(&res.hidden, &stage, n)
    }
}

impl GpuRuntime {
    pub fn hidden_x1(&self) -> &CudaSlice<f32> {
        &self.res.as_ref().expect("resident state").x1
    }

    /// Harness path for block_parts: host x in, synced layer output out.
    pub fn gdn_layer_host(
        &self,
        gdn_index: usize,
        qkv: &GpuQuant,
        z: &GpuQuant,
        b: &GpuQuant,
        a: &GpuQuant,
        out_proj: &GpuQuant,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let dx = self.ctx.stream.clone_htod(x).context("harness x upload")?;
        self.gdn_layer_dx(gdn_index, qkv, z, b, a, out_proj, &dx)?;
        self.ctx.counted_sync()?;
        self.read_y(out_proj)
    }
}

impl GpuRuntime {
    pub fn debug_hidden(&self) -> Result<Vec<f32>> {
        let res = self.res.as_ref().expect("resident state");
        let mut host = vec![0f32; res.hidden.len()];
        self.ctx.stream.memcpy_dtoh(&res.hidden, &mut host)?;
        self.ctx.counted_sync()?;
        Ok(host)
    }
}

impl GpuContext {
    pub fn glue_router_topk(
        &self,
        logits: &CudaSlice<f32>,
        ids: &CudaSlice<i32>,
        w: &CudaSlice<f32>,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let n_i = n as i32;
        let k_i = k as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k_router_topk)
                .arg(logits)
                .arg(ids)
                .arg(w)
                .arg(&n_i)
                .arg(&k_i)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("router topk launch failed: {e}"))?;
        Ok(())
    }

    pub fn glue_moe_combine(
        &self,
        hidden: &CudaSlice<f32>,
        down_y: &CudaSlice<f32>,
        shared_y: &CudaSlice<f32>,
        gate_logit: &CudaSlice<f32>,
        rows: usize,
        slots: usize,
    ) -> Result<()> {
        let rows_i = rows as i32;
        let slots_i = slots as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k_moe_combine)
                .arg(hidden)
                .arg(down_y)
                .arg(shared_y)
                .arg(gate_logit)
                .arg(&rows_i)
                .arg(&slots_i)
                .launch(LaunchConfig {
                    grid_dim: ((rows as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("moe combine launch failed: {e}"))?;
        Ok(())
    }

    pub fn glue_argmax(
        &self,
        logits: &CudaSlice<f32>,
        out: &mut CudaSlice<i32>,
        n: usize,
    ) -> Result<()> {
        let n_i = n as i32;
        // Two-pass above this size: one 1024-thread block scans n serially
        // and runs latency-bound (~58 us at vocab 248320).
        if n > 16384 {
            let (pv, pi) = &self.argmax_scratch;
            let nparts = 128i32;
            unsafe {
                self.stream
                    .launch_builder(&self.k_argmax_part)
                    .arg(logits)
                    .arg(pv)
                    .arg(pi)
                    .arg(&n_i)
                    .launch(LaunchConfig {
                        grid_dim: (nparts as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .map(|_| ())
                    .map_err(|e| anyhow::anyhow!("argmax part launch failed: {e}"))?;
                self.stream
                    .launch_builder(&self.k_argmax_final)
                    .arg(pv)
                    .arg(pi)
                    .arg(&mut *out)
                    .arg(&nparts)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .map(|_| ())
                    .map_err(|e| anyhow::anyhow!("argmax final launch failed: {e}"))?;
            }
            return Ok(());
        }
        unsafe {
            self.stream
                .launch_builder(&self.k_argmax)
                .arg(logits)
                .arg(out)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map(|_| ())
        }
        .map_err(|e| anyhow::anyhow!("argmax launch failed: {e}"))?;
        Ok(())
    }

    /// Zero-centered rmsnorm (qwen3_5 dense): out = rms(x) * (1 + w).
    pub fn glue_rmsnorm_zc(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k_rmsnorm_zc)
                .arg(x)
                .arg(w)
                .arg(out)
                .arg(&n_i)
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map(|_| ())
        }
        .map_err(|e| anyhow::anyhow!("rmsnorm_zc launch failed: {e}"))?;
        Ok(())
    }

    /// Fused residual add + zero-centered rmsnorm (qwen3_5 dense).
    pub fn glue_add_rmsnorm_zc(
        &self,
        acc: &CudaSlice<f32>,
        delta: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k_add_rmsnorm_zc)
                .arg(acc)
                .arg(delta)
                .arg(w)
                .arg(out)
                .arg(&n_i)
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map(|_| ())
        }
        .map_err(|e| anyhow::anyhow!("add_rmsnorm_zc launch failed: {e}"))?;
        Ok(())
    }

    /// Buffer-level attention prephase (qwen3_5: zc=true for the
    /// zero-centered q/k norms). Everything edge0's glue_attn_qk does, but
    /// with explicit buffers instead of ResidentState.
    #[allow(clippy::too_many_arguments)]
    pub fn glue_attn_qk_raw(
        &self,
        q_raw: &CudaSlice<f32>,
        q_norm_w: &CudaSlice<f32>,
        k_raw: &CudaSlice<f32>,
        k_norm_w: &CudaSlice<f32>,
        v_raw: &CudaSlice<f32>,
        q_out: &CudaSlice<f32>,
        gate_out: &CudaSlice<f32>,
        kv_keys: &CudaSlice<f32>,
        kv_values: &CudaSlice<f32>,
        position: &CudaSlice<i32>,
        kv_stride: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f64,
        zc: bool,
    ) -> Result<()> {
        let kv_stride_i = kv_stride as i32;
        let heads_i = heads as i32;
        let kv_heads_i = kv_heads as i32;
        let hd_i = head_dim as i32;
        let rot_i = rotary_dim as i32;
        let k = if zc {
            &self.k_attn_qk_zc
        } else {
            &self.k_attn_qk
        };
        unsafe {
            self.stream
                .launch_builder(k)
                .arg(q_raw)
                .arg(q_norm_w)
                .arg(k_raw)
                .arg(k_norm_w)
                .arg(v_raw)
                .arg(q_out)
                .arg(gate_out)
                .arg(kv_keys)
                .arg(kv_values)
                .arg(position)
                .arg(&kv_stride_i)
                .arg(&heads_i)
                .arg(&kv_heads_i)
                .arg(&hd_i)
                .arg(&rot_i)
                .arg(&theta)
                .launch(LaunchConfig {
                    grid_dim: ((heads + 2 * kv_heads) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map(|_| ())
        }
        .map_err(|e| anyhow::anyhow!("attn qk raw launch failed: {e}"))?;
        Ok(())
    }

    /// Buffer-level scores/softmax/gated-V (see glue_attn_qk_raw).
    #[allow(clippy::too_many_arguments)]
    pub fn glue_attn_scores_raw(
        &self,
        q: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        kv_keys: &CudaSlice<f32>,
        kv_values: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        position: &CudaSlice<i32>,
        kv_stride: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        let kv_stride_i = kv_stride as i32;
        let heads_i = heads as i32;
        let kv_heads_i = kv_heads as i32;
        let hd_i = head_dim as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k_attn_scores)
                .arg(q)
                .arg(gate)
                .arg(kv_keys)
                .arg(kv_values)
                .arg(out)
                .arg(position)
                .arg(&kv_stride_i)
                .arg(&heads_i)
                .arg(&kv_heads_i)
                .arg(&hd_i)
                .arg(&scale)
                .launch(LaunchConfig {
                    grid_dim: (heads as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map(|_| ())
        }
        .map_err(|e| anyhow::anyhow!("attn scores raw launch failed: {e}"))?;
        Ok(())
    }

    pub fn glue_inc(&self, counter: &mut CudaSlice<i32>) -> Result<()> {
        unsafe {
            self.stream
                .launch_builder(&self.k_inc)
                .arg(counter)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("inc launch failed: {e}"))?;
        Ok(())
    }

    /// Increment a 3-int mrope position counter (t,h,w together).
    pub fn glue_inc3(&self, counter: &mut CudaSlice<i32>) -> Result<()> {
        unsafe {
            self.stream
                .launch_builder(&self.k_inc3)
                .arg(counter)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("inc3 launch failed: {e}"))?;
        Ok(())
    }

    /// glue_attn_qk_raw with 3-axis mrope (rope from rope_pos[3]; KV length
    /// from `position`). Bit-identical at rope_pos == [pos, pos, pos].
    #[allow(clippy::too_many_arguments)]
    pub fn glue_attn_qk_zc_mrope(
        &self,
        q_raw: &CudaSlice<f32>,
        q_norm_w: &CudaSlice<f32>,
        k_raw: &CudaSlice<f32>,
        k_norm_w: &CudaSlice<f32>,
        v_raw: &CudaSlice<f32>,
        q_out: &CudaSlice<f32>,
        gate_out: &CudaSlice<f32>,
        kv_keys: &CudaSlice<f32>,
        kv_values: &CudaSlice<f32>,
        position: &CudaSlice<i32>,
        rope_pos: &CudaSlice<i32>,
        kv_stride: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f64,
        sec_h: usize,
        sec_w: usize,
    ) -> Result<()> {
        let kv_stride_i = kv_stride as i32;
        let heads_i = heads as i32;
        let kv_heads_i = kv_heads as i32;
        let hd_i = head_dim as i32;
        let rot_i = rotary_dim as i32;
        let sec_h_i = sec_h as i32;
        let sec_w_i = sec_w as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k_attn_qk_zc_mrope)
                .arg(q_raw)
                .arg(q_norm_w)
                .arg(k_raw)
                .arg(k_norm_w)
                .arg(v_raw)
                .arg(q_out)
                .arg(gate_out)
                .arg(kv_keys)
                .arg(kv_values)
                .arg(position)
                .arg(rope_pos)
                .arg(&kv_stride_i)
                .arg(&heads_i)
                .arg(&kv_heads_i)
                .arg(&hd_i)
                .arg(&rot_i)
                .arg(&theta)
                .arg(&sec_h_i)
                .arg(&sec_w_i)
                .launch(LaunchConfig {
                    grid_dim: ((heads + 2 * kv_heads) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map(|_| ())
        }
        .map_err(|e| anyhow::anyhow!("attn_qk_zc_mrope launch failed: {e}"))?;
        Ok(())
    }
}

impl GpuRuntime {
    /// Closed MoE block: router GEMV -> device top-k -> batched experts +
    /// shared -> device combine into hidden. No sync, no host data.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_closed(
        &self,
        layer: usize,
        router: &GpuQuant,
        experts: &GpuExperts,
        shared: (&GpuQuant, &GpuQuant, &GpuQuant, &GpuQuant),
        dx: &CudaSlice<f32>,
        top_k: usize,
    ) -> Result<()> {
        let res = self.res.as_ref().expect("resident state");
        let rows = experts.rows;
        if self.use_moe_mega {
            let ids = self.expert_ids.lock().expect("expert ids");
            let gy = self.batched_gate_y.lock().expect("gy");
            let uy = self.batched_up_y.lock().expect("uy");
            let dy = self.batched_down_y.lock().expect("dy");
            return self.ctx.moe_mega_launch(
                router,
                shared.2,
                shared.0,
                shared.1,
                shared.3,
                experts,
                layer,
                dx,
                &ids,
                &res.topk_w,
                &gy,
                &uy,
                &dy,
                &res.hidden,
                &self.mega_bar,
                if shared.0.lora.is_some() {
                    self.lora_rank
                } else {
                    0
                },
                top_k,
            );
        }
        let kt = std::env::var_os("EDGE0_KERNEL_TIMES").is_some();
        let mut kt_t = std::time::Instant::now();
        router.launch(&self.ctx, dx, router.y_ref())?;
        if kt {
            self.ctx.stream.synchronize().ok();
            eprintln!("KT router {:.0}us", kt_t.elapsed().as_micros());
            kt_t = std::time::Instant::now();
        }
        {
            // top-k writes the batched-kernel id buffer directly — no d2d
            // copy node in the per-token graph.
            let mut ids = self.expert_ids.lock().expect("expert ids");
            self.ctx.glue_router_topk(
                router.y_ref(),
                &mut ids,
                &res.topk_w,
                router.out_dim,
                top_k,
            )?;
            self.ctx.batched_expert_gemv(
                experts,
                layer,
                0,
                &ids,
                dx,
                &self.batched_gate_y.lock().expect("gy"),
                top_k,
            )?;
            if kt {
                self.ctx.stream.synchronize().ok();
                eprintln!("KT gate {:.0}us", kt_t.elapsed().as_micros());
                kt_t = std::time::Instant::now();
            }
            self.ctx.batched_expert_gemv(
                experts,
                layer,
                1,
                &ids,
                dx,
                &self.batched_up_y.lock().expect("uy"),
                top_k,
            )?;
            if kt {
                self.ctx.stream.synchronize().ok();
                eprintln!("KT up {:.0}us", kt_t.elapsed().as_micros());
                kt_t = std::time::Instant::now();
            }
        }
        // shared.2 (the gate scalar) is int8 — the group kernel is int4-only.
        let segs = [
            shared.0.group_seg(),
            shared.1.group_seg(),
            shared.0.empty_seg_like(),
            shared.0.empty_seg_like(),
        ];
        let rank = if segs.iter().any(|s| s.lora.is_some()) {
            self.lora_rank
        } else {
            0
        };
        self.ctx.glue_group4(&segs, dx, shared.0.in_dim, rank)?;
        shared.2.launch(&self.ctx, dx, shared.2.y_ref())?;

        // Down pass: silu + router-weight fold inline (§3c(3) rules 2-3).
        {
            let g = self.batched_gate_y.lock().expect("gy");
            let u = self.batched_up_y.lock().expect("uy");
            let ids2 = self.expert_ids.lock().expect("expert ids");
            self.ctx.batched_expert_gemv_slotx_silu(
                experts,
                layer,
                &ids2,
                &g,
                &u,
                &res.topk_w,
                &self.batched_down_y.lock().expect("dy"),
                top_k,
            )?;
        }
        if kt {
            self.ctx.stream.synchronize().ok();
            eprintln!("KT slotx_silu {:.0}us", kt_t.elapsed().as_micros());
            kt_t = std::time::Instant::now();
        }
        // Shared expert down over silu(g)*u with its LoRA folded in.
        let rank = if shared.3.lora.is_some() {
            self.lora_rank
        } else {
            0
        };
        self.ctx.gemv1_silu(
            shared.3,
            shared.0.y_ref(),
            shared.1.y_ref(),
            shared.3.y_ref(),
            rank,
        )?;
        if kt {
            self.ctx.stream.synchronize().ok();
            eprintln!("KT shared_down {:.0}us", kt_t.elapsed().as_micros());
            kt_t = std::time::Instant::now();
        }
        self.ctx.glue_moe_combine(
            &res.hidden,
            &self.batched_down_y.lock().expect("dy"),
            shared.3.y_ref(),
            shared.2.y_ref(),
            rows[2],
            top_k,
        )?;
        if kt {
            self.ctx.stream.synchronize().ok();
            eprintln!("KT combine {:.0}us", kt_t.elapsed().as_micros());
        }
        Ok(())
    }

    /// Set the embed row selector for prefill.
    pub fn set_next_token(&self, token: u32) -> Result<()> {
        let res = self.res.as_ref().expect("resident state");
        let mut nt = res.next_token.lock().expect("next token");
        self.ctx.stream.memcpy_htod(&[token as i32], &mut *nt)?;
        Ok(())
    }

    /// embed_row from the device token selector into hidden.
    pub fn embed_from_device_token(&self) -> Result<()> {
        let res = self.res.as_ref().expect("resident state");
        let embed = self
            .proj
            .get("language_model.model.embed_tokens")
            .expect("embed resident");
        let nt = res.next_token.lock().expect("next token");
        self.ctx.glue_embed_row(embed, &nt, &res.hidden)
    }

    /// lm_head over the final-normed x1; returns the logits (host argmax).
    pub fn lm_logits_x1(&self) -> Result<Vec<f32>> {
        let res = self.res.as_ref().expect("resident state");
        let lm = self
            .proj
            .get("language_model.lm_head")
            .expect("lm_head resident");
        lm.launch(&self.ctx, &res.x1, lm.y_ref())?;
        self.ctx.counted_sync()?;
        self.read_y(lm)
    }

    /// lm_head over the already-final-normed x1 + device argmax (the first
    /// decode token after prefill: no forward runs before it).
    pub fn argmax_x1(&self) -> Result<u32> {
        let res = self.res.as_ref().expect("resident state");
        let lm = self
            .proj
            .get("language_model.lm_head")
            .expect("lm_head resident");
        lm.launch(&self.ctx, &res.x1, lm.y_ref())?;
        let mut nt = res.next_token.lock().expect("next token");
        self.ctx.glue_argmax(lm.y_ref(), &mut nt, lm.out_dim)?;
        let mut id = [0i32];
        self.ctx.stream.memcpy_dtoh(&*nt, &mut id)?;
        self.ctx.counted_sync()?;
        Ok(id[0] as u32)
    }

    /// lm_head + device argmax; returns the token id (one dtoh of an int).
    pub fn argmax_token(&self) -> Result<u32> {
        let res = self.res.as_ref().expect("resident state");
        let lm = self
            .proj
            .get("language_model.lm_head")
            .expect("lm_head resident");
        // x1 holds the final-normed hidden (final_norm writes x1).
        lm.launch(&self.ctx, &res.x1, lm.y_ref())?;
        let mut nt = res.next_token.lock().expect("next token");
        self.ctx.glue_argmax(lm.y_ref(), &mut nt, lm.out_dim)?;
        let mut id = [0i32];
        self.ctx.stream.memcpy_dtoh(&*nt, &mut id)?;
        self.ctx.counted_sync()?;
        Ok(id[0] as u32)
    }

    pub fn bump_position(&self) -> Result<()> {
        let res = self.res.as_ref().expect("resident state");
        let mut pos = res.pos_buf.lock().expect("pos");
        self.ctx.glue_inc(&mut pos)
    }
}

impl GpuRuntime {
    /// Finalize a captured decode step: lm_head + argmax into next_token +
    /// position bump (all inside the graph), then read the token (the one
    /// dtoh + sync per replay).
    pub fn finalize_token(&self) -> Result<()> {
        let res = self.res.as_ref().expect("resident state");
        let lm = self
            .proj
            .get("language_model.lm_head")
            .expect("lm_head resident");
        lm.launch(&self.ctx, &res.x1, lm.y_ref())?;
        let mut nt = res.next_token.lock().expect("next token");
        self.ctx.glue_argmax(lm.y_ref(), &mut nt, lm.out_dim)?;
        let mut pos = res.pos_buf.lock().expect("pos");
        self.ctx.glue_inc(&mut pos)
    }

    pub fn read_next_token(&self) -> Result<u32> {
        let res = self.res.as_ref().expect("resident state");
        let nt = res.next_token.lock().expect("next token");
        let mut id = [0i32];
        self.ctx.stream.memcpy_dtoh(&*nt, &mut id)?;
        self.ctx.counted_sync()?;
        Ok(id[0] as u32)
    }
}

/// One grouped-GEMV segment: a projection's device tensors plus row count.
pub struct GroupSeg<'a> {
    pub packed: &'a CudaSlice<u32>,
    pub scales: &'a CudaSlice<f32>,
    pub biases: &'a CudaSlice<f32>,
    pub y: &'a CudaSlice<f32>,
    pub rows: usize,
    /// LoRA pair for this segment, if any.
    pub lora: Option<(&'a CudaSlice<f32>, &'a CudaSlice<f32>)>,
}

impl GpuContext {
    /// Grouped int4 GEMV over up to four same-x projections with LoRA
    /// folded in. `dummy` supplies valid pointers for empty slots.
    pub fn glue_group4(
        &self,
        segs: &[GroupSeg; 4],
        x: &CudaSlice<f32>,
        in_dim: usize,
        rank: usize,
    ) -> Result<()> {
        let in_i = in_dim as i32;
        let rank_i = rank as i32;
        let mut rows = [0i32; 4];
        let mut flags = [0i32; 4];
        let mut total_blocks = 0u32;
        for (i, seg) in segs.iter().enumerate() {
            rows[i] = seg.rows as i32;
            flags[i] = seg.lora.is_some() as i32;
            total_blocks += (seg.rows as u32).div_ceil(16);
        }
        // Dummy pointers for slots without a LoRA pair: reuse slot 0's A/B
        // (never dereferenced when the flag is 0).
        let null_a: &CudaSlice<f32> = segs
            .iter()
            .find_map(|s| s.lora.as_ref().map(|(a, _)| *a))
            .unwrap_or(&x as &CudaSlice<f32>);
        let null_b: &CudaSlice<f32> = segs
            .iter()
            .find_map(|s| s.lora.as_ref().map(|(_, b)| *b))
            .unwrap_or(&x as &CudaSlice<f32>);
        unsafe {
            self.stream
                .launch_builder(&self.k_group4)
                .arg(segs[0].packed)
                .arg(segs[0].scales)
                .arg(segs[0].biases)
                .arg(segs[0].lora.map(|(a, _)| a).unwrap_or(null_a))
                .arg(segs[0].lora.map(|(_, b)| b).unwrap_or(null_b))
                .arg(segs[0].y)
                .arg(&rows[0])
                .arg(segs[1].packed)
                .arg(segs[1].scales)
                .arg(segs[1].biases)
                .arg(segs[1].lora.map(|(a, _)| a).unwrap_or(null_a))
                .arg(segs[1].lora.map(|(_, b)| b).unwrap_or(null_b))
                .arg(segs[1].y)
                .arg(&rows[1])
                .arg(segs[2].packed)
                .arg(segs[2].scales)
                .arg(segs[2].biases)
                .arg(segs[2].lora.map(|(a, _)| a).unwrap_or(null_a))
                .arg(segs[2].lora.map(|(_, b)| b).unwrap_or(null_b))
                .arg(segs[2].y)
                .arg(&rows[2])
                .arg(segs[3].packed)
                .arg(segs[3].scales)
                .arg(segs[3].biases)
                .arg(segs[3].lora.map(|(a, _)| a).unwrap_or(null_a))
                .arg(segs[3].lora.map(|(_, b)| b).unwrap_or(null_b))
                .arg(segs[3].y)
                .arg(&rows[3])
                .arg(x)
                .arg(&in_i)
                .arg(&rank_i)
                .arg(&flags[0])
                .arg(&flags[1])
                .arg(&flags[2])
                .arg(&flags[3])
                .launch(LaunchConfig {
                    grid_dim: (total_blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("group4 launch failed: {e}"))?;
        Ok(())
    }

    pub fn glue_add_rmsnorm(
        &self,
        acc: &CudaSlice<f32>,
        delta: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k_add_rmsnorm)
                .arg(acc)
                .arg(delta)
                .arg(w)
                .arg(out)
                .arg(&n_i)
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("add+rmsnorm launch failed: {e}"))?;
        Ok(())
    }
}

impl GpuContext {
    /// moe_mega: the whole MoE block in one launch (4 grid barriers).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_mega_launch(
        &self,
        router: &GpuQuant,
        ss: &GpuQuant,
        sg: &GpuQuant,
        su: &GpuQuant,
        sd: &GpuQuant,
        experts: &GpuExperts,
        layer: usize,
        x: &CudaSlice<f32>,
        ids: &CudaSlice<i32>,
        w: &CudaSlice<f32>,
        gate_y: &CudaSlice<f32>,
        up_y: &CudaSlice<f32>,
        down_y: &CudaSlice<f32>,
        hidden: &CudaSlice<f32>,
        bar: &CudaSlice<i32>,
        rank: usize,
        top_k: usize,
    ) -> Result<()> {
        self.launch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // The kernel's expert slots are structurally 4 — check, don't assume.
        ensure!(
            top_k == 4,
            "moe_mega kernel hardcodes 4 expert slots, config top_k={top_k} — use moe_closed's batched path"
        );
        // Co-residency: a non-resident grid deadlocks the grid barrier
        // (or __traps at the spin cap). Occupancy-checked once, cached.
        const MEGA_GRID: u32 = 256;
        const MEGA_BLOCK: u32 = 256;
        let cap = match self.mega_coresident.get() {
            Some(&c) => c,
            None => {
                let per_sm = self
                    .k_moe_mega
                    .occupancy_max_active_blocks_per_multiprocessor(MEGA_BLOCK, 0, None)
                    .context("moe_mega occupancy query")?;
                let sms = self
                    .context
                    .attribute(
                        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    )
                    .context("SM count query")? as u32;
                let _ = self.mega_coresident.set(per_sm * sms); // benign race
                per_sm * sms
            }
        };
        ensure!(
            cap >= MEGA_GRID,
            "moe_mega needs {MEGA_GRID} co-resident blocks; this device fits \
             {cap} — run without EDGE0_MEGA"
        );
        let r_rows = router.out_dim as i32;
        let sg_rows = sg.out_dim as i32;
        let in_dim = sg.in_dim as i32;
        let rank_i = rank as i32;
        let ex_rows = experts.rows[0] as i32;
        let down_rows = experts.rows[2] as i32;
        let cap = if std::env::var_os("EDGE0_MEGA_NOCAP").is_some() {
            0
        } else {
            50_000_000
        };
        let dummy = &router.scales;
        let (ra, rb) = match &router.lora {
            Some(l) => (&l.a, &l.b),
            None => (dummy, dummy),
        };
        let (ssa, ssb) = match &ss.lora {
            Some(l) => (&l.a, &l.b),
            None => (dummy, dummy),
        };
        let (sga, sgb) = match &sg.lora {
            Some(l) => (&l.a, &l.b),
            None => (dummy, dummy),
        };
        let (sua, sub) = match &su.lora {
            Some(l) => (&l.a, &l.b),
            None => (dummy, dummy),
        };
        let (sda, sdb) = match &sd.lora {
            Some(l) => (&l.a, &l.b),
            None => (dummy, dummy),
        };
        unsafe {
            self.stream
                .launch_builder(&self.k_moe_mega)
                .arg(bar)
                .arg(&cap)
                .arg(&router.packed)
                .arg(&router.scales)
                .arg(&router.biases)
                .arg(ra)
                .arg(rb)
                .arg(&r_rows)
                .arg(&ss.packed)
                .arg(&ss.scales)
                .arg(&ss.biases)
                .arg(ssa)
                .arg(ssb)
                .arg(ss.y_ref())
                .arg(&sg.packed)
                .arg(&sg.scales)
                .arg(&sg.biases)
                .arg(sga)
                .arg(sgb)
                .arg(&sg_rows)
                .arg(sg.y_ref())
                .arg(&su.packed)
                .arg(&su.scales)
                .arg(&su.biases)
                .arg(sua)
                .arg(sub)
                .arg(su.y_ref())
                .arg(x)
                .arg(&in_dim)
                .arg(&rank_i)
                .arg(ids)
                .arg(w)
                .arg(&experts.stacked[layer][0])
                .arg(&experts.stacked_scales[layer][0])
                .arg(&experts.stacked_biases[layer][0])
                .arg(&experts.stacked[layer][1])
                .arg(&experts.stacked_scales[layer][1])
                .arg(&experts.stacked_biases[layer][1])
                .arg(gate_y)
                .arg(up_y)
                .arg(&ex_rows)
                .arg(&experts.stacked[layer][2])
                .arg(&experts.stacked_scales[layer][2])
                .arg(&experts.stacked_biases[layer][2])
                .arg(down_y)
                .arg(&down_rows)
                .arg(&sd.packed)
                .arg(&sd.scales)
                .arg(&sd.biases)
                .arg(sda)
                .arg(sdb)
                .arg(sd.y_ref())
                .arg(hidden)
                .arg(router.y_ref())
                .launch(LaunchConfig {
                    grid_dim: (MEGA_GRID, 1, 1),
                    block_dim: (MEGA_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("moe_mega launch failed: {e}"))?;
        Ok(())
    }
}

impl GpuContext {
    /// Cold-bench helper: stream `bytes` of device memory to evict L2.
    pub fn flush_l2(&self, buf: &CudaSlice<f32>) -> Result<()> {
        use cudarc::driver::safe::DevicePtr;
        let ptrs: Vec<u64> = vec![buf.device_ptr(&self.stream).0 as u64];
        let tbl = self.ctx_u64(&ptrs)?;
        let scratch = self.upload_f32(&[0f32; 4])?;
        let n = (buf.len() / 4) as i64;
        unsafe {
            self.stream
                .launch_builder(&self.k_read_scatter)
                .arg(&tbl)
                .arg(&1i32)
                .arg(&n)
                .arg(&scratch)
                .launch(LaunchConfig {
                    grid_dim: (256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| anyhow::anyhow!("flush launch failed: {e}"))?;
        Ok(())
    }
}
