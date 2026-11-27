//! Pure-Rust StableHLO MLIR text emitter for TPU execution via PJRT.

use anyhow::{Result, ensure};
use std::fmt::Write;

/// Geometry and loop-chunking parameters of one MiniMax-H3 DiT main block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DitBlockHloSpec {
    pub rows: usize,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub rotary_dim: usize,
    pub head_chunk: usize,
    pub adaln_pairs: usize,
    pub norm_eps: f32,
    pub qk_norm_eps: f32,
}

/// Geometry of one plain pre-norm MiniMax-H3 token refiner block (no AdaLN, no RoPE).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DitRefinerBlockHloSpec {
    pub rows: usize,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub head_chunk: usize,
    pub norm_eps: f32,
    pub qk_norm_eps: f32,
}

/// Geometry of one MiniMax-H3-Image ViT VAE decoder block (BF16 projections, F32 norms/scales/residual).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VaeBlockHloSpec {
    pub batch: usize,
    pub tokens: usize,
    pub dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub rotary_dim: usize,
    pub norm_eps: f32,
}

/// Geometry of one Qwen3-VL text decoder layer (causal GQA, QK-RMSNorm, interleaved mRoPE, SwiGLU).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen3VlLayerHloSpec {
    pub rows: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub head_chunk: usize,
    pub rms_norm_eps: f32,
}

/// Geometry of BF16 `norm_out` (RMSNorm + AdaLN shift/scale + image-row slice).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DitNormOutSpec {
    pub rows: usize,
    pub text_rows: usize,
    pub hidden: usize,
    pub timesteps: usize,
    pub final_norm_eps: f32,
}

/// Geometry of Image VAE pre-projection (`post_quant_conv` + `proj_in` + register/cls append).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VaePreHloSpec {
    pub batch: usize,
    pub patches: usize,
    pub latent_channels: usize,
    pub dim: usize,
    pub registers: usize,
}

/// Geometry of Image VAE post-projection (`norm_out` F32 LayerNorm -> BF16 `proj_out`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VaePostHloSpec {
    pub batch: usize,
    pub tokens: usize,
    pub num_patches: usize,
    pub dim: usize,
    pub patch_width: usize,
    pub norm_eps: f32,
}

/// Emits `Y[M, N] = bf16(f32_acc(X[M, K] @ W[N, K]^T))` using BF16 MXU `dot_general`.
pub fn emit_bf16_matmul(m: usize, k: usize, n: usize) -> Result<String> {
    ensure!(
        m > 0 && k > 0 && n > 0,
        "matmul dims must be positive: ({m}, {k}, {n})"
    );
    Ok(format!(
        "\
module @ff_bf16_matmul {{
  func.func public @main(%x: tensor<{m}x{k}xbf16>, %w: tensor<{n}x{k}xbf16>) -> tensor<{m}x{n}xbf16> {{
    %acc = stablehlo.dot_general %x, %w, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{m}x{k}xbf16>, tensor<{n}x{k}xbf16>) -> tensor<{m}x{n}xf32>
    %out = stablehlo.convert %acc : (tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xbf16>
    func.return %out : tensor<{m}x{n}xbf16>
  }}
}}
"
    ))
}

/// Emits `Y[T, N] = bf16(f32_acc(X[T, K] @ W[N, K]^T) + f32(bias[N]))` for setup-time AdaLN projections.
pub fn emit_bf16_adaln_proj(timesteps: usize, time_dim: usize, out_dim: usize) -> Result<String> {
    ensure!(
        timesteps > 0 && time_dim > 0 && out_dim > 0,
        "adaln_proj dims must be positive: ({timesteps}, {time_dim}, {out_dim})"
    );
    Ok(format!(
        "\
module @ff_bf16_adaln_proj {{
  func.func public @main(%x: tensor<{timesteps}x{time_dim}xbf16>, %w: tensor<{out_dim}x{time_dim}xbf16>, %b: tensor<{out_dim}xbf16>) -> tensor<{timesteps}x{out_dim}xbf16> {{
    %acc = stablehlo.dot_general %x, %w, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{timesteps}x{time_dim}xbf16>, tensor<{out_dim}x{time_dim}xbf16>) -> tensor<{timesteps}x{out_dim}xf32>
    %b_f32 = stablehlo.convert %b : (tensor<{out_dim}xbf16>) -> tensor<{out_dim}xf32>
    %b_b = stablehlo.broadcast_in_dim %b_f32, dims = [1] : (tensor<{out_dim}xf32>) -> tensor<{timesteps}x{out_dim}xf32>
    %sum = stablehlo.add %acc, %b_b : tensor<{timesteps}x{out_dim}xf32>
    %out = stablehlo.convert %sum : (tensor<{timesteps}x{out_dim}xf32>) -> tensor<{timesteps}x{out_dim}xbf16>
    func.return %out : tensor<{timesteps}x{out_dim}xbf16>
  }}
}}
"
    ))
}

/// Emits bidirectional BF16 self-attention on packed `[R, H, D]` tensors with F32 softmax
/// and BF16 probability-plane `P @ V` accumulation.
pub fn emit_bf16_attention(
    rows: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<String> {
    ensure!(
        rows > 0 && heads > 0 && head_dim > 0,
        "attention dims must be positive: ({rows}, {heads}, {head_dim})"
    );
    ensure!(
        scale.is_finite() && scale > 0.0,
        "attention scale must be finite positive: {scale}"
    );
    let scale_hex = f32_hex(scale);
    Ok(format!(
        "\
module @ff_bf16_attention {{
  func.func public @main(%q: tensor<{rows}x{heads}x{head_dim}xbf16>, %k: tensor<{rows}x{heads}x{head_dim}xbf16>, %v: tensor<{rows}x{heads}x{head_dim}xbf16>) -> tensor<{rows}x{heads}x{head_dim}xbf16> {{
    %qk = stablehlo.dot_general %q, %k, batching_dims = [1] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<{rows}x{heads}x{head_dim}xbf16>, tensor<{rows}x{heads}x{head_dim}xbf16>) -> tensor<{heads}x{rows}x{rows}xf32>
    %scale = stablehlo.constant dense<{scale_hex}> : tensor<{heads}x{rows}x{rows}xf32>
    %scores = stablehlo.multiply %qk, %scale : tensor<{heads}x{rows}x{rows}xf32>
    %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %max = \"stablehlo.reduce\"(%scores, %neg_inf) ({{
      ^bb0(%a: tensor<f32>, %b: tensor<f32>):
        %m = stablehlo.maximum %a, %b : tensor<f32>
        stablehlo.return %m : tensor<f32>
    }}) {{dimensions = array<i64: 2>}} : (tensor<{heads}x{rows}x{rows}xf32>, tensor<f32>) -> tensor<{heads}x{rows}xf32>
    %max_b = stablehlo.broadcast_in_dim %max, dims = [0, 1] : (tensor<{heads}x{rows}xf32>) -> tensor<{heads}x{rows}x{rows}xf32>
    %shifted = stablehlo.subtract %scores, %max_b : tensor<{heads}x{rows}x{rows}xf32>
    %exp = stablehlo.exponential %shifted : tensor<{heads}x{rows}x{rows}xf32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sum = \"stablehlo.reduce\"(%exp, %zero) ({{
      ^bb0(%a: tensor<f32>, %b: tensor<f32>):
        %s = stablehlo.add %a, %b : tensor<f32>
        stablehlo.return %s : tensor<f32>
    }}) {{dimensions = array<i64: 2>}} : (tensor<{heads}x{rows}x{rows}xf32>, tensor<f32>) -> tensor<{heads}x{rows}xf32>
    %sum_b = stablehlo.broadcast_in_dim %sum, dims = [0, 1] : (tensor<{heads}x{rows}xf32>) -> tensor<{heads}x{rows}x{rows}xf32>
    %probs_f32 = stablehlo.divide %exp, %sum_b : tensor<{heads}x{rows}x{rows}xf32>
    %probs = stablehlo.convert %probs_f32 : (tensor<{heads}x{rows}x{rows}xf32>) -> tensor<{heads}x{rows}x{rows}xbf16>
    %pv = stablehlo.dot_general %probs, %v, batching_dims = [0] x [1], contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<{heads}x{rows}x{rows}xbf16>, tensor<{rows}x{heads}x{head_dim}xbf16>) -> tensor<{heads}x{rows}x{head_dim}xf32>
    %trans = stablehlo.transpose %pv, dims = [1, 0, 2] : (tensor<{heads}x{rows}x{head_dim}xf32>) -> tensor<{rows}x{heads}x{head_dim}xf32>
    %out = stablehlo.convert %trans : (tensor<{rows}x{heads}x{head_dim}xf32>) -> tensor<{rows}x{heads}x{head_dim}xbf16>
    func.return %out : tensor<{rows}x{heads}x{head_dim}xbf16>
  }}
}}
"
    ))
}

/// Emits one MiniMax-H3 DiT main block in published BF16 precision:
/// RMSNorm1 -> AdaLN modulate -> Q/K/V -> QK-RMSNorm -> 3D RoPE -> head-chunked bidirectional SDPA
/// -> to_out -> AdaLN gated residual -> RMSNorm2 -> AdaLN modulate -> SwiGLU FFN -> AdaLN gated residual.
pub fn emit_h3_dit_block(spec: &DitBlockHloSpec) -> Result<String> {
    let r = spec.rows;
    let d = spec.hidden;
    let h = spec.heads;
    let hd = spec.head_dim;
    let f = spec.ffn;
    let rot = spec.rotary_dim;
    let hc = spec.head_chunk;
    let p = spec.adaln_pairs;
    ensure!(
        r > 0 && d > 0 && h > 0 && hd > 0 && f > 0 && p > 0,
        "DiT block dims must be positive: {spec:?}"
    );
    ensure!(
        rot > 0 && rot <= hd && rot.is_multiple_of(2),
        "rotary_dim {rot} must be even and within head_dim {hd}"
    );
    ensure!(
        hc > 0 && h.is_multiple_of(hc),
        "head_chunk {hc} must divide heads {h}"
    );
    ensure!(
        spec.norm_eps.is_finite()
            && spec.norm_eps > 0.0
            && spec.qk_norm_eps.is_finite()
            && spec.qk_norm_eps > 0.0,
        "RMSNorm eps must be finite positive: norm={} qk={}",
        spec.norm_eps,
        spec.qk_norm_eps
    );

    let width = h * hd;
    let chunks = h / hc;
    let half_rot = rot / 2;
    let ffn2 = 2 * f;
    let scale_hex = f32_hex(1.0f32 / (hd as f32).sqrt());

    let mut out = String::with_capacity(16_384);
    writeln!(
        out,
        "module @ff_h3_dit_block {{
  func.func public @main(
    %x: tensor<{r}x{d}xbf16>,
    %adaln: tensor<{p}x6x{d}xbf16>,
    %row_pairs: tensor<{r}xi32>,
    %rotary: tensor<{r}x{rot}xbf16>,
    %norm1_w: tensor<{d}xbf16>,
    %wq: tensor<{width}x{d}xbf16>,
    %wk: tensor<{width}x{d}xbf16>,
    %wv: tensor<{width}x{d}xbf16>,
    %norm_q_w: tensor<{hd}xbf16>,
    %norm_k_w: tensor<{hd}xbf16>,
    %wout: tensor<{d}x{width}xbf16>,
    %norm2_w: tensor<{d}xbf16>,
    %wff0: tensor<{ffn2}x{d}xbf16>,
    %wff2: tensor<{d}x{f}xbf16>
  ) -> tensor<{r}x{d}xbf16> {{"
    )?;

    writeln!(
        out,
        "    %f32_zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>"
    )?;
    writeln!(
        out,
        "    %f32_one = stablehlo.constant dense<1.000000e+00> : tensor<f32>"
    )?;
    writeln!(
        out,
        "    %bf16_zero_s = stablehlo.convert %f32_zero : (tensor<f32>) -> tensor<bf16>"
    )?;
    writeln!(
        out,
        "    %bf16_one_s = stablehlo.convert %f32_one : (tensor<f32>) -> tensor<bf16>"
    )?;
    writeln!(
        out,
        "    %bf16_one = stablehlo.broadcast_in_dim %bf16_one_s, dims = [] : (tensor<bf16>) -> tensor<{r}x{d}xbf16>"
    )?;

    // 1. Pre-attention RMSNorm + AdaLN modulate
    push_rms_norm_2d(&mut out, "n1", "%x", "%norm1_w", spec)?;
    push_adaln_gather(&mut out, "shift_msa", 0, spec)?;
    push_adaln_gather(&mut out, "scale_msa", 1, spec)?;
    writeln!(
        out,
        "    %msa_scale = stablehlo.add %bf16_one, %scale_msa : tensor<{r}x{d}xbf16>"
    )?;
    writeln!(
        out,
        "    %msa_mul = stablehlo.multiply %n1, %msa_scale : tensor<{r}x{d}xbf16>"
    )?;
    writeln!(
        out,
        "    %h1 = stablehlo.add %msa_mul, %shift_msa : tensor<{r}x{d}xbf16>"
    )?;

    // 2. Q, K, V projections
    writeln!(
        out,
        "    %q_f32 = stablehlo.dot_general %h1, %wq, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{d}xbf16>, tensor<{width}x{d}xbf16>) -> tensor<{r}x{width}xf32>"
    )?;
    writeln!(
        out,
        "    %q_2d = stablehlo.convert %q_f32 : (tensor<{r}x{width}xf32>) -> tensor<{r}x{width}xbf16>"
    )?;
    writeln!(
        out,
        "    %q_3d = stablehlo.reshape %q_2d : (tensor<{r}x{width}xbf16>) -> tensor<{r}x{h}x{hd}xbf16>"
    )?;

    writeln!(
        out,
        "    %k_f32 = stablehlo.dot_general %h1, %wk, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{d}xbf16>, tensor<{width}x{d}xbf16>) -> tensor<{r}x{width}xf32>"
    )?;
    writeln!(
        out,
        "    %k_2d = stablehlo.convert %k_f32 : (tensor<{r}x{width}xf32>) -> tensor<{r}x{width}xbf16>"
    )?;
    writeln!(
        out,
        "    %k_3d = stablehlo.reshape %k_2d : (tensor<{r}x{width}xbf16>) -> tensor<{r}x{h}x{hd}xbf16>"
    )?;

    writeln!(
        out,
        "    %v_f32 = stablehlo.dot_general %h1, %wv, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{d}xbf16>, tensor<{width}x{d}xbf16>) -> tensor<{r}x{width}xf32>"
    )?;
    writeln!(
        out,
        "    %v_2d = stablehlo.convert %v_f32 : (tensor<{r}x{width}xf32>) -> tensor<{r}x{width}xbf16>"
    )?;
    writeln!(
        out,
        "    %v_3d = stablehlo.reshape %v_2d : (tensor<{r}x{width}xbf16>) -> tensor<{r}x{h}x{hd}xbf16>"
    )?;

    // 3. QK-RMSNorm
    push_rms_norm_3d(&mut out, "qn", "%q_3d", "%norm_q_w", spec)?;
    push_rms_norm_3d(&mut out, "kn", "%k_3d", "%norm_k_w", spec)?;

    // 4. 3D RoPE on leading rotary_dim channels, passing trailing channels unchanged
    writeln!(
        out,
        "    %cos_2d = \"stablehlo.slice\"(%rotary) {{start_indices = array<i64: 0, 0>, limit_indices = array<i64: {r}, {half_rot}>, strides = array<i64: 1, 1>}} : (tensor<{r}x{rot}xbf16>) -> tensor<{r}x{half_rot}xbf16>"
    )?;
    writeln!(
        out,
        "    %sin_2d = \"stablehlo.slice\"(%rotary) {{start_indices = array<i64: 0, {half_rot}>, limit_indices = array<i64: {r}, {rot}>, strides = array<i64: 1, 1>}} : (tensor<{r}x{rot}xbf16>) -> tensor<{r}x{half_rot}xbf16>"
    )?;
    writeln!(
        out,
        "    %cos_b = stablehlo.broadcast_in_dim %cos_2d, dims = [0, 2] : (tensor<{r}x{half_rot}xbf16>) -> tensor<{r}x{h}x{half_rot}xbf16>"
    )?;
    writeln!(
        out,
        "    %sin_b = stablehlo.broadcast_in_dim %sin_2d, dims = [0, 2] : (tensor<{r}x{half_rot}xbf16>) -> tensor<{r}x{h}x{half_rot}xbf16>"
    )?;
    push_rope_3d(&mut out, "q_rot", "%qn", spec)?;
    push_rope_3d(&mut out, "k_rot", "%kn", spec)?;

    // 5. Head-chunked bidirectional SDPA via self-contained stablehlo.while
    let chunk_ty = format!("tensor<{chunks}x{hc}x{r}x{hd}xbf16>");
    writeln!(
        out,
        "    %q_4d = stablehlo.reshape %q_rot : (tensor<{r}x{h}x{hd}xbf16>) -> tensor<{r}x{chunks}x{hc}x{hd}xbf16>"
    )?;
    writeln!(
        out,
        "    %qt = stablehlo.transpose %q_4d, dims = [1, 2, 0, 3] : (tensor<{r}x{chunks}x{hc}x{hd}xbf16>) -> {chunk_ty}"
    )?;
    writeln!(
        out,
        "    %k_4d = stablehlo.reshape %k_rot : (tensor<{r}x{h}x{hd}xbf16>) -> tensor<{r}x{chunks}x{hc}x{hd}xbf16>"
    )?;
    writeln!(
        out,
        "    %kt = stablehlo.transpose %k_4d, dims = [1, 2, 0, 3] : (tensor<{r}x{chunks}x{hc}x{hd}xbf16>) -> {chunk_ty}"
    )?;
    writeln!(
        out,
        "    %v_4d = stablehlo.reshape %v_3d : (tensor<{r}x{h}x{hd}xbf16>) -> tensor<{r}x{chunks}x{hc}x{hd}xbf16>"
    )?;
    writeln!(
        out,
        "    %vt = stablehlo.transpose %v_4d, dims = [1, 2, 0, 3] : (tensor<{r}x{chunks}x{hc}x{hd}xbf16>) -> {chunk_ty}"
    )?;
    writeln!(
        out,
        "    %i_init = stablehlo.constant dense<0> : tensor<i32>"
    )?;
    writeln!(
        out,
        "    %acc_init = stablehlo.broadcast_in_dim %bf16_zero_s, dims = [] : (tensor<bf16>) -> {chunk_ty}"
    )?;
    writeln!(
        out,
        "    %attn_while:5 = \"stablehlo.while\"(%i_init, %acc_init, %qt, %kt, %vt) ({{
      ^bb0(%wi: tensor<i32>, %wacc: {chunk_ty}, %wqt: {chunk_ty}, %wkt: {chunk_ty}, %wvt: {chunk_ty}):
        %wlim = stablehlo.constant dense<{chunks}> : tensor<i32>
        %wcond = stablehlo.compare LT, %wi, %wlim, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %wcond : tensor<i1>
    }}, {{
      ^bb0(%wi: tensor<i32>, %wacc: {chunk_ty}, %wqt: {chunk_ty}, %wkt: {chunk_ty}, %wvt: {chunk_ty}):
        %wc0 = stablehlo.constant dense<0> : tensor<i32>
        %wc1 = stablehlo.constant dense<1> : tensor<i32>
        %wqc_4d = \"stablehlo.dynamic_slice\"(%wqt, %wi, %wc0, %wc0, %wc0) {{slice_sizes = array<i64: 1, {hc}, {r}, {hd}>}} : ({chunk_ty}, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x{hc}x{r}x{hd}xbf16>
        %wqc = stablehlo.reshape %wqc_4d : (tensor<1x{hc}x{r}x{hd}xbf16>) -> tensor<{hc}x{r}x{hd}xbf16>
        %wkc_4d = \"stablehlo.dynamic_slice\"(%wkt, %wi, %wc0, %wc0, %wc0) {{slice_sizes = array<i64: 1, {hc}, {r}, {hd}>}} : ({chunk_ty}, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x{hc}x{r}x{hd}xbf16>
        %wkc = stablehlo.reshape %wkc_4d : (tensor<1x{hc}x{r}x{hd}xbf16>) -> tensor<{hc}x{r}x{hd}xbf16>
        %wvc_4d = \"stablehlo.dynamic_slice\"(%wvt, %wi, %wc0, %wc0, %wc0) {{slice_sizes = array<i64: 1, {hc}, {r}, {hd}>}} : ({chunk_ty}, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x{hc}x{r}x{hd}xbf16>
        %wvc = stablehlo.reshape %wvc_4d : (tensor<1x{hc}x{r}x{hd}xbf16>) -> tensor<{hc}x{r}x{hd}xbf16>
        %wqk = stablehlo.dot_general %wqc, %wkc, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<{hc}x{r}x{hd}xbf16>, tensor<{hc}x{r}x{hd}xbf16>) -> tensor<{hc}x{r}x{r}xf32>
        %wscale_s = stablehlo.constant dense<{scale_hex}> : tensor<f32>
        %wscale = stablehlo.broadcast_in_dim %wscale_s, dims = [] : (tensor<f32>) -> tensor<{hc}x{r}x{r}xf32>
        %wscores = stablehlo.multiply %wqk, %wscale : tensor<{hc}x{r}x{r}xf32>
        %wneg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
        %wmax = \"stablehlo.reduce\"(%wscores, %wneg_inf) ({{
          ^bb0(%a: tensor<f32>, %b: tensor<f32>):
            %m = stablehlo.maximum %a, %b : tensor<f32>
            stablehlo.return %m : tensor<f32>
        }}) {{dimensions = array<i64: 2>}} : (tensor<{hc}x{r}x{r}xf32>, tensor<f32>) -> tensor<{hc}x{r}xf32>
        %wmax_b = stablehlo.broadcast_in_dim %wmax, dims = [0, 1] : (tensor<{hc}x{r}xf32>) -> tensor<{hc}x{r}x{r}xf32>
        %wshifted = stablehlo.subtract %wscores, %wmax_b : tensor<{hc}x{r}x{r}xf32>
        %wexp = stablehlo.exponential %wshifted : tensor<{hc}x{r}x{r}xf32>
        %wzero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %wsum = \"stablehlo.reduce\"(%wexp, %wzero) ({{
          ^bb0(%a: tensor<f32>, %b: tensor<f32>):
            %s = stablehlo.add %a, %b : tensor<f32>
            stablehlo.return %s : tensor<f32>
        }}) {{dimensions = array<i64: 2>}} : (tensor<{hc}x{r}x{r}xf32>, tensor<f32>) -> tensor<{hc}x{r}xf32>
        %wsum_b = stablehlo.broadcast_in_dim %wsum, dims = [0, 1] : (tensor<{hc}x{r}xf32>) -> tensor<{hc}x{r}x{r}xf32>
        %wprobs_f32 = stablehlo.divide %wexp, %wsum_b : tensor<{hc}x{r}x{r}xf32>
        %wprobs = stablehlo.convert %wprobs_f32 : (tensor<{hc}x{r}x{r}xf32>) -> tensor<{hc}x{r}x{r}xbf16>
        %wpv_f32 = stablehlo.dot_general %wprobs, %wvc, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{hc}x{r}x{r}xbf16>, tensor<{hc}x{r}x{hd}xbf16>) -> tensor<{hc}x{r}x{hd}xf32>
        %wpv_bf16 = stablehlo.convert %wpv_f32 : (tensor<{hc}x{r}x{hd}xf32>) -> tensor<{hc}x{r}x{hd}xbf16>
        %wpv_4d = stablehlo.reshape %wpv_bf16 : (tensor<{hc}x{r}x{hd}xbf16>) -> tensor<1x{hc}x{r}x{hd}xbf16>
        %wacc_next = \"stablehlo.dynamic_update_slice\"(%wacc, %wpv_4d, %wi, %wc0, %wc0, %wc0) : ({chunk_ty}, tensor<1x{hc}x{r}x{hd}xbf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> {chunk_ty}
        %wi_next = stablehlo.add %wi, %wc1 : tensor<i32>
        stablehlo.return %wi_next, %wacc_next, %wqt, %wkt, %wvt : tensor<i32>, {chunk_ty}, {chunk_ty}, {chunk_ty}, {chunk_ty}
    }}) : (tensor<i32>, {chunk_ty}, {chunk_ty}, {chunk_ty}, {chunk_ty}) -> (tensor<i32>, {chunk_ty}, {chunk_ty}, {chunk_ty}, {chunk_ty})"
    )?;
    writeln!(
        out,
        "    %attn_t = stablehlo.transpose %attn_while#1, dims = [2, 0, 1, 3] : ({chunk_ty}) -> tensor<{r}x{chunks}x{hc}x{hd}xbf16>"
    )?;
    writeln!(
        out,
        "    %attn_flat = stablehlo.reshape %attn_t : (tensor<{r}x{chunks}x{hc}x{hd}xbf16>) -> tensor<{r}x{width}xbf16>"
    )?;

    // 6. Attention output projection + AdaLN gated residual
    writeln!(
        out,
        "    %aout_f32 = stablehlo.dot_general %attn_flat, %wout, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{width}xbf16>, tensor<{d}x{width}xbf16>) -> tensor<{r}x{d}xf32>"
    )?;
    writeln!(
        out,
        "    %aout = stablehlo.convert %aout_f32 : (tensor<{r}x{d}xf32>) -> tensor<{r}x{d}xbf16>"
    )?;
    push_adaln_gather(&mut out, "gate_msa", 2, spec)?;
    writeln!(
        out,
        "    %gated_attn = stablehlo.multiply %gate_msa, %aout : tensor<{r}x{d}xbf16>"
    )?;
    writeln!(
        out,
        "    %x_mid = stablehlo.add %x, %gated_attn : tensor<{r}x{d}xbf16>"
    )?;

    // 7. Pre-FFN RMSNorm + AdaLN modulate
    push_rms_norm_2d(&mut out, "n2", "%x_mid", "%norm2_w", spec)?;
    push_adaln_gather(&mut out, "shift_mlp", 3, spec)?;
    push_adaln_gather(&mut out, "scale_mlp", 4, spec)?;
    writeln!(
        out,
        "    %mlp_scale = stablehlo.add %bf16_one, %scale_mlp : tensor<{r}x{d}xbf16>"
    )?;
    writeln!(
        out,
        "    %mlp_mul = stablehlo.multiply %n2, %mlp_scale : tensor<{r}x{d}xbf16>"
    )?;
    writeln!(
        out,
        "    %h2 = stablehlo.add %mlp_mul, %shift_mlp : tensor<{r}x{d}xbf16>"
    )?;

    // 8. SwiGLU FFN: lower_half * SiLU(upper_half), matching diffusers SwiGLU
    writeln!(
        out,
        "    %up_f32 = stablehlo.dot_general %h2, %wff0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{d}xbf16>, tensor<{ffn2}x{d}xbf16>) -> tensor<{r}x{ffn2}xf32>"
    )?;
    writeln!(
        out,
        "    %up = stablehlo.convert %up_f32 : (tensor<{r}x{ffn2}xf32>) -> tensor<{r}x{ffn2}xbf16>"
    )?;
    writeln!(
        out,
        "    %ff_val = \"stablehlo.slice\"(%up) {{start_indices = array<i64: 0, 0>, limit_indices = array<i64: {r}, {f}>, strides = array<i64: 1, 1>}} : (tensor<{r}x{ffn2}xbf16>) -> tensor<{r}x{f}xbf16>"
    )?;
    writeln!(
        out,
        "    %ff_gate = \"stablehlo.slice\"(%up) {{start_indices = array<i64: 0, {f}>, limit_indices = array<i64: {r}, {ffn2}>, strides = array<i64: 1, 1>}} : (tensor<{r}x{ffn2}xbf16>) -> tensor<{r}x{f}xbf16>"
    )?;
    writeln!(
        out,
        "    %ff_gate_f32 = stablehlo.convert %ff_gate : (tensor<{r}x{f}xbf16>) -> tensor<{r}x{f}xf32>"
    )?;
    writeln!(
        out,
        "    %ff_sig_f32 = stablehlo.logistic %ff_gate_f32 : tensor<{r}x{f}xf32>"
    )?;
    writeln!(
        out,
        "    %ff_silu_f32 = stablehlo.multiply %ff_gate_f32, %ff_sig_f32 : tensor<{r}x{f}xf32>"
    )?;
    writeln!(
        out,
        "    %ff_silu = stablehlo.convert %ff_silu_f32 : (tensor<{r}x{f}xf32>) -> tensor<{r}x{f}xbf16>"
    )?;
    writeln!(
        out,
        "    %ff_act = stablehlo.multiply %ff_val, %ff_silu : tensor<{r}x{f}xbf16>"
    )?;
    writeln!(
        out,
        "    %ff_out_f32 = stablehlo.dot_general %ff_act, %wff2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{f}xbf16>, tensor<{d}x{f}xbf16>) -> tensor<{r}x{d}xf32>"
    )?;
    writeln!(
        out,
        "    %ff_out = stablehlo.convert %ff_out_f32 : (tensor<{r}x{d}xf32>) -> tensor<{r}x{d}xbf16>"
    )?;

    // 9. FFN AdaLN gated residual
    push_adaln_gather(&mut out, "gate_mlp", 5, spec)?;
    writeln!(
        out,
        "    %gated_ff = stablehlo.multiply %gate_mlp, %ff_out : tensor<{r}x{d}xbf16>"
    )?;
    writeln!(
        out,
        "    %out = stablehlo.add %x_mid, %gated_ff : tensor<{r}x{d}xbf16>"
    )?;
    writeln!(out, "    func.return %out : tensor<{r}x{d}xbf16>")?;
    writeln!(out, "  }}\n}}")?;
    Ok(out)
}

fn f32_hex(v: f32) -> String {
    format!("0x{:08X}", v.to_bits())
}

fn push_adaln_gather(
    out: &mut String,
    name: &str,
    slot: usize,
    spec: &DitBlockHloSpec,
) -> Result<()> {
    let pairs = spec.adaln_pairs;
    let rows = spec.rows;
    let hidden = spec.hidden;
    let next_slot = slot + 1;
    writeln!(
        out,
        "    %{name}_3d = \"stablehlo.slice\"(%adaln) {{start_indices = array<i64: 0, {slot}, 0>, limit_indices = array<i64: {pairs}, {next_slot}, {hidden}>, strides = array<i64: 1, 1, 1>}} : (tensor<{pairs}x6x{hidden}xbf16>) -> tensor<{pairs}x1x{hidden}xbf16>"
    )?;
    writeln!(
        out,
        "    %{name}_tbl = stablehlo.reshape %{name}_3d : (tensor<{pairs}x1x{hidden}xbf16>) -> tensor<{pairs}x{hidden}xbf16>"
    )?;
    writeln!(
        out,
        "    %{name} = \"stablehlo.gather\"(%{name}_tbl, %row_pairs) {{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, {hidden}>}} : (tensor<{pairs}x{hidden}xbf16>, tensor<{rows}xi32>) -> tensor<{rows}x{hidden}xbf16>"
    )?;
    Ok(())
}

fn push_rms_norm_2d(
    out: &mut String,
    prefix: &str,
    input: &str,
    weight: &str,
    spec: &DitBlockHloSpec,
) -> Result<()> {
    let rows = spec.rows;
    let dim = spec.hidden;
    let inv_d = f32_hex(1.0f32 / dim as f32);
    let eps_hex = f32_hex(spec.norm_eps);
    writeln!(
        out,
        "    %{prefix}_f32 = stablehlo.convert {input} : (tensor<{rows}x{dim}xbf16>) -> tensor<{rows}x{dim}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_sq = stablehlo.multiply %{prefix}_f32, %{prefix}_f32 : tensor<{rows}x{dim}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_sum = \"stablehlo.reduce\"(%{prefix}_sq, %f32_zero) ({{
      ^bb0(%a: tensor<f32>, %b: tensor<f32>):
        %s = stablehlo.add %a, %b : tensor<f32>
        stablehlo.return %s : tensor<f32>
    }}) {{dimensions = array<i64: 1>}} : (tensor<{rows}x{dim}xf32>, tensor<f32>) -> tensor<{rows}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_inv_s = stablehlo.constant dense<{inv_d}> : tensor<f32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_inv = stablehlo.broadcast_in_dim %{prefix}_inv_s, dims = [] : (tensor<f32>) -> tensor<{rows}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_mean = stablehlo.multiply %{prefix}_sum, %{prefix}_inv : tensor<{rows}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_eps_s = stablehlo.constant dense<{eps_hex}> : tensor<f32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_eps = stablehlo.broadcast_in_dim %{prefix}_eps_s, dims = [] : (tensor<f32>) -> tensor<{rows}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_var = stablehlo.add %{prefix}_mean, %{prefix}_eps : tensor<{rows}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_rstd = stablehlo.rsqrt %{prefix}_var : tensor<{rows}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_rstd_b = stablehlo.broadcast_in_dim %{prefix}_rstd, dims = [0] : (tensor<{rows}xf32>) -> tensor<{rows}x{dim}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_norm = stablehlo.multiply %{prefix}_f32, %{prefix}_rstd_b : tensor<{rows}x{dim}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_wf32 = stablehlo.convert {weight} : (tensor<{dim}xbf16>) -> tensor<{dim}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_wb = stablehlo.broadcast_in_dim %{prefix}_wf32, dims = [1] : (tensor<{dim}xf32>) -> tensor<{rows}x{dim}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_scaled = stablehlo.multiply %{prefix}_norm, %{prefix}_wb : tensor<{rows}x{dim}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix} = stablehlo.convert %{prefix}_scaled : (tensor<{rows}x{dim}xf32>) -> tensor<{rows}x{dim}xbf16>"
    )?;
    Ok(())
}

fn push_rms_norm_3d(
    out: &mut String,
    prefix: &str,
    input: &str,
    weight: &str,
    spec: &DitBlockHloSpec,
) -> Result<()> {
    let rows = spec.rows;
    let heads = spec.heads;
    let dim = spec.head_dim;
    let inv_d = f32_hex(1.0f32 / dim as f32);
    let eps_hex = f32_hex(spec.qk_norm_eps);
    writeln!(
        out,
        "    %{prefix}_f32 = stablehlo.convert {input} : (tensor<{rows}x{heads}x{dim}xbf16>) -> tensor<{rows}x{heads}x{dim}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_sq = stablehlo.multiply %{prefix}_f32, %{prefix}_f32 : tensor<{rows}x{heads}x{dim}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_sum = \"stablehlo.reduce\"(%{prefix}_sq, %f32_zero) ({{
      ^bb0(%a: tensor<f32>, %b: tensor<f32>):
        %s = stablehlo.add %a, %b : tensor<f32>
        stablehlo.return %s : tensor<f32>
    }}) {{dimensions = array<i64: 2>}} : (tensor<{rows}x{heads}x{dim}xf32>, tensor<f32>) -> tensor<{rows}x{heads}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_inv_s = stablehlo.constant dense<{inv_d}> : tensor<f32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_inv = stablehlo.broadcast_in_dim %{prefix}_inv_s, dims = [] : (tensor<f32>) -> tensor<{rows}x{heads}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_mean = stablehlo.multiply %{prefix}_sum, %{prefix}_inv : tensor<{rows}x{heads}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_eps_s = stablehlo.constant dense<{eps_hex}> : tensor<f32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_eps = stablehlo.broadcast_in_dim %{prefix}_eps_s, dims = [] : (tensor<f32>) -> tensor<{rows}x{heads}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_var = stablehlo.add %{prefix}_mean, %{prefix}_eps : tensor<{rows}x{heads}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_rstd = stablehlo.rsqrt %{prefix}_var : tensor<{rows}x{heads}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_rstd_b = stablehlo.broadcast_in_dim %{prefix}_rstd, dims = [0, 1] : (tensor<{rows}x{heads}xf32>) -> tensor<{rows}x{heads}x{dim}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_norm = stablehlo.multiply %{prefix}_f32, %{prefix}_rstd_b : tensor<{rows}x{heads}x{dim}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_wf32 = stablehlo.convert {weight} : (tensor<{dim}xbf16>) -> tensor<{dim}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_wb = stablehlo.broadcast_in_dim %{prefix}_wf32, dims = [2] : (tensor<{dim}xf32>) -> tensor<{rows}x{heads}x{dim}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix}_scaled = stablehlo.multiply %{prefix}_norm, %{prefix}_wb : tensor<{rows}x{heads}x{dim}xf32>"
    )?;
    writeln!(
        out,
        "    %{prefix} = stablehlo.convert %{prefix}_scaled : (tensor<{rows}x{heads}x{dim}xf32>) -> tensor<{rows}x{heads}x{dim}xbf16>"
    )?;
    Ok(())
}

fn push_rope_3d(out: &mut String, prefix: &str, input: &str, spec: &DitBlockHloSpec) -> Result<()> {
    let rows = spec.rows;
    let heads = spec.heads;
    let head_dim = spec.head_dim;
    let rotary_dim = spec.rotary_dim;
    let half = rotary_dim / 2;
    let pass = head_dim - rotary_dim;
    writeln!(
        out,
        "    %{prefix}_x1 = \"stablehlo.slice\"({input}) {{start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: {rows}, {heads}, {half}>, strides = array<i64: 1, 1, 1>}} : (tensor<{rows}x{heads}x{head_dim}xbf16>) -> tensor<{rows}x{heads}x{half}xbf16>"
    )?;
    writeln!(
        out,
        "    %{prefix}_x2 = \"stablehlo.slice\"({input}) {{start_indices = array<i64: 0, 0, {half}>, limit_indices = array<i64: {rows}, {heads}, {rotary_dim}>, strides = array<i64: 1, 1, 1>}} : (tensor<{rows}x{heads}x{head_dim}xbf16>) -> tensor<{rows}x{heads}x{half}xbf16>"
    )?;
    writeln!(
        out,
        "    %{prefix}_t1 = stablehlo.multiply %{prefix}_x1, %cos_b : tensor<{rows}x{heads}x{half}xbf16>"
    )?;
    writeln!(
        out,
        "    %{prefix}_t2 = stablehlo.multiply %{prefix}_x2, %sin_b : tensor<{rows}x{heads}x{half}xbf16>"
    )?;
    writeln!(
        out,
        "    %{prefix}_lo = stablehlo.subtract %{prefix}_t1, %{prefix}_t2 : tensor<{rows}x{heads}x{half}xbf16>"
    )?;
    writeln!(
        out,
        "    %{prefix}_t3 = stablehlo.multiply %{prefix}_x2, %cos_b : tensor<{rows}x{heads}x{half}xbf16>"
    )?;
    writeln!(
        out,
        "    %{prefix}_t4 = stablehlo.multiply %{prefix}_x1, %sin_b : tensor<{rows}x{heads}x{half}xbf16>"
    )?;
    writeln!(
        out,
        "    %{prefix}_hi = stablehlo.add %{prefix}_t3, %{prefix}_t4 : tensor<{rows}x{heads}x{half}xbf16>"
    )?;
    if pass > 0 {
        writeln!(
            out,
            "    %{prefix}_xp = \"stablehlo.slice\"({input}) {{start_indices = array<i64: 0, 0, {rotary_dim}>, limit_indices = array<i64: {rows}, {heads}, {head_dim}>, strides = array<i64: 1, 1, 1>}} : (tensor<{rows}x{heads}x{head_dim}xbf16>) -> tensor<{rows}x{heads}x{pass}xbf16>"
        )?;
        writeln!(
            out,
            "    %{prefix} = \"stablehlo.concatenate\"(%{prefix}_lo, %{prefix}_hi, %{prefix}_xp) {{dimension = 2 : i64}} : (tensor<{rows}x{heads}x{half}xbf16>, tensor<{rows}x{heads}x{half}xbf16>, tensor<{rows}x{heads}x{pass}xbf16>) -> tensor<{rows}x{heads}x{head_dim}xbf16>"
        )?;
    } else {
        writeln!(
            out,
            "    %{prefix} = \"stablehlo.concatenate\"(%{prefix}_lo, %{prefix}_hi) {{dimension = 2 : i64}} : (tensor<{rows}x{heads}x{half}xbf16>, tensor<{rows}x{heads}x{half}xbf16>) -> tensor<{rows}x{heads}x{head_dim}xbf16>"
        )?;
    }
    Ok(())
}

/// Emits F32 `proj_in` (`precision = [HIGHEST, HIGHEST]`) + BF16 cast + `[text | image]` concat.
pub fn emit_h3_dit_proj_in_pack(
    text_rows: usize,
    img_rows: usize,
    patch_dim: usize,
    hidden: usize,
) -> Result<String> {
    ensure!(
        text_rows > 0 && img_rows > 0 && patch_dim > 0 && hidden > 0,
        "proj_in dims must be positive: text={text_rows} img={img_rows} patch={patch_dim} hidden={hidden}"
    );
    let total = text_rows + img_rows;
    Ok(format!(
        "\
module @ff_h3_dit_proj_in_pack {{
  func.func public @main(
    %text: tensor<{text_rows}x{hidden}xbf16>,
    %latents: tensor<{img_rows}x{patch_dim}xf32>,
    %w: tensor<{hidden}x{patch_dim}xf32>,
    %b: tensor<{hidden}xf32>
  ) -> tensor<{total}x{hidden}xbf16> {{
    %dot = stablehlo.dot_general %latents, %w, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<{img_rows}x{patch_dim}xf32>, tensor<{hidden}x{patch_dim}xf32>) -> tensor<{img_rows}x{hidden}xf32>
    %b_b = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<{hidden}xf32>) -> tensor<{img_rows}x{hidden}xf32>
    %img_f32 = stablehlo.add %dot, %b_b : tensor<{img_rows}x{hidden}xf32>
    %img = stablehlo.convert %img_f32 : (tensor<{img_rows}x{hidden}xf32>) -> tensor<{img_rows}x{hidden}xbf16>
    %out = \"stablehlo.concatenate\"(%text, %img) {{dimension = 0 : i64}} : (tensor<{text_rows}x{hidden}xbf16>, tensor<{img_rows}x{hidden}xbf16>) -> tensor<{total}x{hidden}xbf16>
    func.return %out : tensor<{total}x{hidden}xbf16>
  }}
}}
"
    ))
}

/// Emits BF16 `context_embedder`: `bf16(f32_acc(X @ W^T) + f32(bias))`.
pub fn emit_h3_dit_context_embedder(rows: usize, text_dim: usize, hidden: usize) -> Result<String> {
    ensure!(
        rows > 0 && text_dim > 0 && hidden > 0,
        "context_embedder dims must be positive: ({rows}, {text_dim}, {hidden})"
    );
    Ok(format!(
        "\
module @ff_h3_dit_context_embedder {{
  func.func public @main(
    %x: tensor<{rows}x{text_dim}xbf16>,
    %w: tensor<{hidden}x{text_dim}xbf16>,
    %b: tensor<{hidden}xbf16>
  ) -> tensor<{rows}x{hidden}xbf16> {{
    %acc = stablehlo.dot_general %x, %w, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{rows}x{text_dim}xbf16>, tensor<{hidden}x{text_dim}xbf16>) -> tensor<{rows}x{hidden}xf32>
    %b_f32 = stablehlo.convert %b : (tensor<{hidden}xbf16>) -> tensor<{hidden}xf32>
    %b_b = stablehlo.broadcast_in_dim %b_f32, dims = [1] : (tensor<{hidden}xf32>) -> tensor<{rows}x{hidden}xf32>
    %sum = stablehlo.add %acc, %b_b : tensor<{rows}x{hidden}xf32>
    %out = stablehlo.convert %sum : (tensor<{rows}x{hidden}xf32>) -> tensor<{rows}x{hidden}xbf16>
    func.return %out : tensor<{rows}x{hidden}xbf16>
  }}
}}
"
    ))
}

/// Emits BF16 RMSNorm with F32 variance (`token_refiner.final_norm`).
pub fn emit_bf16_rms_norm(rows: usize, dim: usize, eps: f32) -> Result<String> {
    ensure!(rows > 0 && dim > 0, "rms_norm dims must be positive");
    ensure!(eps.is_finite() && eps > 0.0, "rms_norm eps invalid: {eps}");
    let mut out = String::with_capacity(2048);
    writeln!(
        out,
        "module @ff_bf16_rms_norm {{
  func.func public @main(%x: tensor<{rows}x{dim}xbf16>, %w: tensor<{dim}xbf16>) -> tensor<{rows}x{dim}xbf16> {{
    %f32_zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>"
    )?;
    push_rms_norm_2d_raw(&mut out, "n", "%x", "%w", (rows, dim, eps))?;
    writeln!(
        out,
        "    func.return %n : tensor<{rows}x{dim}xbf16>\n  }}\n}}"
    )?;
    Ok(out)
}

/// Emits one plain pre-norm BF16 `token_refiner` block (no AdaLN, no RoPE, bidirectional SDPA, SwiGLU).
pub fn emit_h3_dit_refiner_block(spec: &DitRefinerBlockHloSpec) -> Result<String> {
    let r = spec.rows;
    let d = spec.hidden;
    let h = spec.heads;
    let hd = spec.head_dim;
    let f = spec.ffn;
    let hc = spec.head_chunk;
    ensure!(
        r > 0 && d > 0 && h > 0 && hd > 0 && f > 0,
        "refiner block dims must be positive: {spec:?}"
    );
    ensure!(
        hc > 0 && h.is_multiple_of(hc),
        "head_chunk {hc} must divide heads {h}"
    );
    ensure!(
        spec.norm_eps.is_finite()
            && spec.norm_eps > 0.0
            && spec.qk_norm_eps.is_finite()
            && spec.qk_norm_eps > 0.0,
        "refiner RMSNorm eps invalid"
    );
    let width = h * hd;
    let chunks = h / hc;
    let ffn2 = 2 * f;
    let scale_hex = f32_hex(1.0f32 / (hd as f32).sqrt());
    let chunk_ty = format!("tensor<{chunks}x{hc}x{r}x{hd}xbf16>");

    let mut out = String::with_capacity(12_288);
    writeln!(
        out,
        "module @ff_h3_dit_refiner_block {{
  func.func public @main(
    %x: tensor<{r}x{d}xbf16>,
    %norm1_w: tensor<{d}xbf16>,
    %wq: tensor<{width}x{d}xbf16>,
    %wk: tensor<{width}x{d}xbf16>,
    %wv: tensor<{width}x{d}xbf16>,
    %norm_q_w: tensor<{hd}xbf16>,
    %norm_k_w: tensor<{hd}xbf16>,
    %wout: tensor<{d}x{width}xbf16>,
    %norm2_w: tensor<{d}xbf16>,
    %wff0: tensor<{ffn2}x{d}xbf16>,
    %wff2: tensor<{d}x{f}xbf16>
  ) -> tensor<{r}x{d}xbf16> {{
    %f32_zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %bf16_zero_s = stablehlo.convert %f32_zero : (tensor<f32>) -> tensor<bf16>"
    )?;
    push_rms_norm_2d_raw(&mut out, "h1", "%x", "%norm1_w", (r, d, spec.norm_eps))?;
    writeln!(
        out,
        "    %q_f32 = stablehlo.dot_general %h1, %wq, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{d}xbf16>, tensor<{width}x{d}xbf16>) -> tensor<{r}x{width}xf32>
    %q_2d = stablehlo.convert %q_f32 : (tensor<{r}x{width}xf32>) -> tensor<{r}x{width}xbf16>
    %q_3d = stablehlo.reshape %q_2d : (tensor<{r}x{width}xbf16>) -> tensor<{r}x{h}x{hd}xbf16>
    %k_f32 = stablehlo.dot_general %h1, %wk, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{d}xbf16>, tensor<{width}x{d}xbf16>) -> tensor<{r}x{width}xf32>
    %k_2d = stablehlo.convert %k_f32 : (tensor<{r}x{width}xf32>) -> tensor<{r}x{width}xbf16>
    %k_3d = stablehlo.reshape %k_2d : (tensor<{r}x{width}xbf16>) -> tensor<{r}x{h}x{hd}xbf16>
    %v_f32 = stablehlo.dot_general %h1, %wv, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{d}xbf16>, tensor<{width}x{d}xbf16>) -> tensor<{r}x{width}xf32>
    %v_2d = stablehlo.convert %v_f32 : (tensor<{r}x{width}xf32>) -> tensor<{r}x{width}xbf16>
    %v_3d = stablehlo.reshape %v_2d : (tensor<{r}x{width}xbf16>) -> tensor<{r}x{h}x{hd}xbf16>"
    )?;
    push_rms_norm_3d_raw(
        &mut out,
        "qn",
        "%q_3d",
        "%norm_q_w",
        (r, h, hd, spec.qk_norm_eps),
    )?;
    push_rms_norm_3d_raw(
        &mut out,
        "kn",
        "%k_3d",
        "%norm_k_w",
        (r, h, hd, spec.qk_norm_eps),
    )?;
    writeln!(
        out,
        "    %q_4d = stablehlo.reshape %qn : (tensor<{r}x{h}x{hd}xbf16>) -> tensor<{r}x{chunks}x{hc}x{hd}xbf16>
    %qt = stablehlo.transpose %q_4d, dims = [1, 2, 0, 3] : (tensor<{r}x{chunks}x{hc}x{hd}xbf16>) -> {chunk_ty}
    %k_4d = stablehlo.reshape %kn : (tensor<{r}x{h}x{hd}xbf16>) -> tensor<{r}x{chunks}x{hc}x{hd}xbf16>
    %kt = stablehlo.transpose %k_4d, dims = [1, 2, 0, 3] : (tensor<{r}x{chunks}x{hc}x{hd}xbf16>) -> {chunk_ty}
    %v_4d = stablehlo.reshape %v_3d : (tensor<{r}x{h}x{hd}xbf16>) -> tensor<{r}x{chunks}x{hc}x{hd}xbf16>
    %vt = stablehlo.transpose %v_4d, dims = [1, 2, 0, 3] : (tensor<{r}x{chunks}x{hc}x{hd}xbf16>) -> {chunk_ty}
    %i_init = stablehlo.constant dense<0> : tensor<i32>
    %acc_init = stablehlo.broadcast_in_dim %bf16_zero_s, dims = [] : (tensor<bf16>) -> {chunk_ty}
    %attn_while:5 = \"stablehlo.while\"(%i_init, %acc_init, %qt, %kt, %vt) ({{
      ^bb0(%wi: tensor<i32>, %wacc: {chunk_ty}, %wqt: {chunk_ty}, %wkt: {chunk_ty}, %wvt: {chunk_ty}):
        %wlim = stablehlo.constant dense<{chunks}> : tensor<i32>
        %wcond = stablehlo.compare LT, %wi, %wlim, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %wcond : tensor<i1>
    }}, {{
      ^bb0(%wi: tensor<i32>, %wacc: {chunk_ty}, %wqt: {chunk_ty}, %wkt: {chunk_ty}, %wvt: {chunk_ty}):
        %wc0 = stablehlo.constant dense<0> : tensor<i32>
        %wc1 = stablehlo.constant dense<1> : tensor<i32>
        %wqc_4d = \"stablehlo.dynamic_slice\"(%wqt, %wi, %wc0, %wc0, %wc0) {{slice_sizes = array<i64: 1, {hc}, {r}, {hd}>}} : ({chunk_ty}, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x{hc}x{r}x{hd}xbf16>
        %wqc = stablehlo.reshape %wqc_4d : (tensor<1x{hc}x{r}x{hd}xbf16>) -> tensor<{hc}x{r}x{hd}xbf16>
        %wkc_4d = \"stablehlo.dynamic_slice\"(%wkt, %wi, %wc0, %wc0, %wc0) {{slice_sizes = array<i64: 1, {hc}, {r}, {hd}>}} : ({chunk_ty}, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x{hc}x{r}x{hd}xbf16>
        %wkc = stablehlo.reshape %wkc_4d : (tensor<1x{hc}x{r}x{hd}xbf16>) -> tensor<{hc}x{r}x{hd}xbf16>
        %wvc_4d = \"stablehlo.dynamic_slice\"(%wvt, %wi, %wc0, %wc0, %wc0) {{slice_sizes = array<i64: 1, {hc}, {r}, {hd}>}} : ({chunk_ty}, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x{hc}x{r}x{hd}xbf16>
        %wvc = stablehlo.reshape %wvc_4d : (tensor<1x{hc}x{r}x{hd}xbf16>) -> tensor<{hc}x{r}x{hd}xbf16>
        %wqk = stablehlo.dot_general %wqc, %wkc, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<{hc}x{r}x{hd}xbf16>, tensor<{hc}x{r}x{hd}xbf16>) -> tensor<{hc}x{r}x{r}xf32>
        %wscale_s = stablehlo.constant dense<{scale_hex}> : tensor<f32>
        %wscale = stablehlo.broadcast_in_dim %wscale_s, dims = [] : (tensor<f32>) -> tensor<{hc}x{r}x{r}xf32>
        %wscores = stablehlo.multiply %wqk, %wscale : tensor<{hc}x{r}x{r}xf32>
        %wneg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
        %wmax = \"stablehlo.reduce\"(%wscores, %wneg_inf) ({{
          ^bb0(%a: tensor<f32>, %b: tensor<f32>):
            %m = stablehlo.maximum %a, %b : tensor<f32>
            stablehlo.return %m : tensor<f32>
        }}) {{dimensions = array<i64: 2>}} : (tensor<{hc}x{r}x{r}xf32>, tensor<f32>) -> tensor<{hc}x{r}xf32>
        %wmax_b = stablehlo.broadcast_in_dim %wmax, dims = [0, 1] : (tensor<{hc}x{r}xf32>) -> tensor<{hc}x{r}x{r}xf32>
        %wshifted = stablehlo.subtract %wscores, %wmax_b : tensor<{hc}x{r}x{r}xf32>
        %wexp = stablehlo.exponential %wshifted : tensor<{hc}x{r}x{r}xf32>
        %wzero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %wsum = \"stablehlo.reduce\"(%wexp, %wzero) ({{
          ^bb0(%a: tensor<f32>, %b: tensor<f32>):
            %s = stablehlo.add %a, %b : tensor<f32>
            stablehlo.return %s : tensor<f32>
        }}) {{dimensions = array<i64: 2>}} : (tensor<{hc}x{r}x{r}xf32>, tensor<f32>) -> tensor<{hc}x{r}xf32>
        %wsum_b = stablehlo.broadcast_in_dim %wsum, dims = [0, 1] : (tensor<{hc}x{r}xf32>) -> tensor<{hc}x{r}x{r}xf32>
        %wprobs_f32 = stablehlo.divide %wexp, %wsum_b : tensor<{hc}x{r}x{r}xf32>
        %wprobs = stablehlo.convert %wprobs_f32 : (tensor<{hc}x{r}x{r}xf32>) -> tensor<{hc}x{r}x{r}xbf16>
        %wpv_f32 = stablehlo.dot_general %wprobs, %wvc, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{hc}x{r}x{r}xbf16>, tensor<{hc}x{r}x{hd}xbf16>) -> tensor<{hc}x{r}x{hd}xf32>
        %wpv_bf16 = stablehlo.convert %wpv_f32 : (tensor<{hc}x{r}x{hd}xf32>) -> tensor<{hc}x{r}x{hd}xbf16>
        %wpv_4d = stablehlo.reshape %wpv_bf16 : (tensor<{hc}x{r}x{hd}xbf16>) -> tensor<1x{hc}x{r}x{hd}xbf16>
        %wacc_next = \"stablehlo.dynamic_update_slice\"(%wacc, %wpv_4d, %wi, %wc0, %wc0, %wc0) : ({chunk_ty}, tensor<1x{hc}x{r}x{hd}xbf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> {chunk_ty}
        %wi_next = stablehlo.add %wi, %wc1 : tensor<i32>
        stablehlo.return %wi_next, %wacc_next, %wqt, %wkt, %wvt : tensor<i32>, {chunk_ty}, {chunk_ty}, {chunk_ty}, {chunk_ty}
    }}) : (tensor<i32>, {chunk_ty}, {chunk_ty}, {chunk_ty}, {chunk_ty}) -> (tensor<i32>, {chunk_ty}, {chunk_ty}, {chunk_ty}, {chunk_ty})
    %attn_t = stablehlo.transpose %attn_while#1, dims = [2, 0, 1, 3] : ({chunk_ty}) -> tensor<{r}x{chunks}x{hc}x{hd}xbf16>
    %attn_flat = stablehlo.reshape %attn_t : (tensor<{r}x{chunks}x{hc}x{hd}xbf16>) -> tensor<{r}x{width}xbf16>
    %aout_f32 = stablehlo.dot_general %attn_flat, %wout, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{width}xbf16>, tensor<{d}x{width}xbf16>) -> tensor<{r}x{d}xf32>
    %aout = stablehlo.convert %aout_f32 : (tensor<{r}x{d}xf32>) -> tensor<{r}x{d}xbf16>
    %x_mid = stablehlo.add %x, %aout : tensor<{r}x{d}xbf16>"
    )?;
    push_rms_norm_2d_raw(&mut out, "h2", "%x_mid", "%norm2_w", (r, d, spec.norm_eps))?;
    writeln!(
        out,
        "    %up_f32 = stablehlo.dot_general %h2, %wff0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{d}xbf16>, tensor<{ffn2}x{d}xbf16>) -> tensor<{r}x{ffn2}xf32>
    %up = stablehlo.convert %up_f32 : (tensor<{r}x{ffn2}xf32>) -> tensor<{r}x{ffn2}xbf16>
    %ff_val = \"stablehlo.slice\"(%up) {{start_indices = array<i64: 0, 0>, limit_indices = array<i64: {r}, {f}>, strides = array<i64: 1, 1>}} : (tensor<{r}x{ffn2}xbf16>) -> tensor<{r}x{f}xbf16>
    %ff_gate = \"stablehlo.slice\"(%up) {{start_indices = array<i64: 0, {f}>, limit_indices = array<i64: {r}, {ffn2}>, strides = array<i64: 1, 1>}} : (tensor<{r}x{ffn2}xbf16>) -> tensor<{r}x{f}xbf16>
    %ff_gate_f32 = stablehlo.convert %ff_gate : (tensor<{r}x{f}xbf16>) -> tensor<{r}x{f}xf32>
    %ff_sig_f32 = stablehlo.logistic %ff_gate_f32 : tensor<{r}x{f}xf32>
    %ff_silu_f32 = stablehlo.multiply %ff_gate_f32, %ff_sig_f32 : tensor<{r}x{f}xf32>
    %ff_silu = stablehlo.convert %ff_silu_f32 : (tensor<{r}x{f}xf32>) -> tensor<{r}x{f}xbf16>
    %ff_act = stablehlo.multiply %ff_val, %ff_silu : tensor<{r}x{f}xbf16>
    %ff_out_f32 = stablehlo.dot_general %ff_act, %wff2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{f}xbf16>, tensor<{d}x{f}xbf16>) -> tensor<{r}x{d}xf32>
    %ff_out = stablehlo.convert %ff_out_f32 : (tensor<{r}x{d}xf32>) -> tensor<{r}x{d}xbf16>
    %out = stablehlo.add %x_mid, %ff_out : tensor<{r}x{d}xbf16>
    func.return %out : tensor<{r}x{d}xbf16>
  }}
}}"
    )?;
    Ok(out)
}

/// Emits BF16 `norm_out` (RMSNorm + per-row-timestep AdaLN shift/scale) and slices image rows `[text_rows..rows]`.
pub fn emit_h3_dit_norm_out(spec: &DitNormOutSpec) -> Result<String> {
    let r = spec.rows;
    let t = spec.text_rows;
    let d = spec.hidden;
    let ts = spec.timesteps;
    ensure!(
        t > 0 && r > t && d > 0 && ts > 0,
        "norm_out dims invalid: {spec:?}"
    );
    ensure!(
        spec.final_norm_eps.is_finite() && spec.final_norm_eps > 0.0,
        "final_norm_eps invalid: {}",
        spec.final_norm_eps
    );
    let img = r - t;
    let mut out = String::with_capacity(4096);
    writeln!(
        out,
        "module @ff_h3_dit_norm_out {{
  func.func public @main(
    %x: tensor<{r}x{d}xbf16>,
    %norm_w: tensor<{d}xbf16>,
    %norm_adaln: tensor<{ts}x2x{d}xbf16>,
    %row_timesteps: tensor<{r}xi32>
  ) -> tensor<{img}x{d}xbf16> {{
    %f32_zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %f32_one = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %bf16_one_s = stablehlo.convert %f32_one : (tensor<f32>) -> tensor<bf16>
    %bf16_one = stablehlo.broadcast_in_dim %bf16_one_s, dims = [] : (tensor<bf16>) -> tensor<{r}x{d}xbf16>"
    )?;
    push_rms_norm_2d_raw(&mut out, "n", "%x", "%norm_w", (r, d, spec.final_norm_eps))?;
    writeln!(
        out,
        "    %shift_3d = \"stablehlo.slice\"(%norm_adaln) {{start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: {ts}, 1, {d}>, strides = array<i64: 1, 1, 1>}} : (tensor<{ts}x2x{d}xbf16>) -> tensor<{ts}x1x{d}xbf16>
    %shift_tbl = stablehlo.reshape %shift_3d : (tensor<{ts}x1x{d}xbf16>) -> tensor<{ts}x{d}xbf16>
    %shift = \"stablehlo.gather\"(%shift_tbl, %row_timesteps) {{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, {d}>}} : (tensor<{ts}x{d}xbf16>, tensor<{r}xi32>) -> tensor<{r}x{d}xbf16>
    %scale_3d = \"stablehlo.slice\"(%norm_adaln) {{start_indices = array<i64: 0, 1, 0>, limit_indices = array<i64: {ts}, 2, {d}>, strides = array<i64: 1, 1, 1>}} : (tensor<{ts}x2x{d}xbf16>) -> tensor<{ts}x1x{d}xbf16>
    %scale_tbl = stablehlo.reshape %scale_3d : (tensor<{ts}x1x{d}xbf16>) -> tensor<{ts}x{d}xbf16>
    %scale = \"stablehlo.gather\"(%scale_tbl, %row_timesteps) {{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, {d}>}} : (tensor<{ts}x{d}xbf16>, tensor<{r}xi32>) -> tensor<{r}x{d}xbf16>
    %scale1 = stablehlo.add %bf16_one, %scale : tensor<{r}x{d}xbf16>
    %mod_mul = stablehlo.multiply %n, %scale1 : tensor<{r}x{d}xbf16>
    %mod = stablehlo.add %mod_mul, %shift : tensor<{r}x{d}xbf16>
    %img_bf16 = \"stablehlo.slice\"(%mod) {{start_indices = array<i64: {t}, 0>, limit_indices = array<i64: {r}, {d}>, strides = array<i64: 1, 1>}} : (tensor<{r}x{d}xbf16>) -> tensor<{img}x{d}xbf16>
    func.return %img_bf16 : tensor<{img}x{d}xbf16>
  }}
}}"
    )?;
    Ok(out)
}

/// Emits F32 `pdd_proj_out` (`precision = [HIGHEST, HIGHEST]`) from BF16 `norm_out` image rows.
pub fn emit_h3_dit_pdd_proj_out(
    img_rows: usize,
    hidden: usize,
    patch_dim: usize,
) -> Result<String> {
    ensure!(
        img_rows > 0 && hidden > 0 && patch_dim > 0,
        "pdd_proj_out dims invalid: img_rows={img_rows} hidden={hidden} patch_dim={patch_dim}"
    );
    Ok(format!(
        "\
module @ff_h3_dit_pdd_proj_out {{
  func.func public @main(
    %img_bf16: tensor<{img_rows}x{hidden}xbf16>,
    %w_out: tensor<{patch_dim}x{hidden}xf32>,
    %b_out: tensor<{patch_dim}xf32>
  ) -> tensor<{img_rows}x{patch_dim}xf32> {{
    %img_f32 = stablehlo.convert %img_bf16 : (tensor<{img_rows}x{hidden}xbf16>) -> tensor<{img_rows}x{hidden}xf32>
    %vel_dot = stablehlo.dot_general %img_f32, %w_out, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<{img_rows}x{hidden}xf32>, tensor<{patch_dim}x{hidden}xf32>) -> tensor<{img_rows}x{patch_dim}xf32>
    %b_b = stablehlo.broadcast_in_dim %b_out, dims = [1] : (tensor<{patch_dim}xf32>) -> tensor<{img_rows}x{patch_dim}xf32>
    %vel = stablehlo.add %vel_dot, %b_b : tensor<{img_rows}x{patch_dim}xf32>
    func.return %vel : tensor<{img_rows}x{patch_dim}xf32>
  }}
}}
"
    ))
}

/// Emits Image VAE pre-projection: BF16 `post_quant_conv` -> BF16 `proj_in` -> append `register_tokens` + zero cls -> F32 stream.
pub fn emit_h3_image_vae_pre(spec: &VaePreHloSpec) -> Result<String> {
    let b = spec.batch;
    let p = spec.patches;
    let c = spec.latent_channels;
    let d = spec.dim;
    let reg = spec.registers;
    ensure!(
        b > 0 && p > 0 && c > 0 && d > 0 && reg > 0,
        "vae_pre dims must be positive: {spec:?}"
    );
    let tok = p + reg + 1;
    Ok(format!(
        "\
module @ff_h3_image_vae_pre {{
  func.func public @main(
    %z: tensor<{b}x{p}x{c}xbf16>,
    %pq_w: tensor<{c}x{c}xbf16>,
    %pq_b: tensor<{c}xbf16>,
    %pin_w: tensor<{d}x{c}xbf16>,
    %pin_b: tensor<{d}xbf16>,
    %regs: tensor<{reg}x{d}xbf16>
  ) -> tensor<{b}x{tok}x{d}xf32> {{
    %pq_dot = stablehlo.dot_general %z, %pq_w, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{b}x{p}x{c}xbf16>, tensor<{c}x{c}xbf16>) -> tensor<{b}x{p}x{c}xf32>
    %pq_bf32 = stablehlo.convert %pq_b : (tensor<{c}xbf16>) -> tensor<{c}xf32>
    %pq_bb = stablehlo.broadcast_in_dim %pq_bf32, dims = [2] : (tensor<{c}xf32>) -> tensor<{b}x{p}x{c}xf32>
    %pq_sum = stablehlo.add %pq_dot, %pq_bb : tensor<{b}x{p}x{c}xf32>
    %pq = stablehlo.convert %pq_sum : (tensor<{b}x{p}x{c}xf32>) -> tensor<{b}x{p}x{c}xbf16>
    %in_dot = stablehlo.dot_general %pq, %pin_w, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{b}x{p}x{c}xbf16>, tensor<{d}x{c}xbf16>) -> tensor<{b}x{p}x{d}xf32>
    %in_bf32 = stablehlo.convert %pin_b : (tensor<{d}xbf16>) -> tensor<{d}xf32>
    %in_bb = stablehlo.broadcast_in_dim %in_bf32, dims = [2] : (tensor<{d}xf32>) -> tensor<{b}x{p}x{d}xf32>
    %in_sum = stablehlo.add %in_dot, %in_bb : tensor<{b}x{p}x{d}xf32>
    %patches = stablehlo.convert %in_sum : (tensor<{b}x{p}x{d}xf32>) -> tensor<{b}x{p}x{d}xbf16>
    %regs_b = stablehlo.broadcast_in_dim %regs, dims = [1, 2] : (tensor<{reg}x{d}xbf16>) -> tensor<{b}x{reg}x{d}xbf16>
    %z0_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %z0_bf16 = stablehlo.convert %z0_f32 : (tensor<f32>) -> tensor<bf16>
    %cls = stablehlo.broadcast_in_dim %z0_bf16, dims = [] : (tensor<bf16>) -> tensor<{b}x1x{d}xbf16>
    %cat = \"stablehlo.concatenate\"(%patches, %regs_b, %cls) {{dimension = 1 : i64}} : (tensor<{b}x{p}x{d}xbf16>, tensor<{b}x{reg}x{d}xbf16>, tensor<{b}x1x{d}xbf16>) -> tensor<{b}x{tok}x{d}xbf16>
    %out = stablehlo.convert %cat : (tensor<{b}x{tok}x{d}xbf16>) -> tensor<{b}x{tok}x{d}xf32>
    func.return %out : tensor<{b}x{tok}x{d}xf32>
  }}
}}
"
    ))
}

/// Emits one Image VAE 2D ViT decoder block in publisher mixed precision:
/// F32 residual/norms/scales + BF16 biased linears/SDPA + F32 affine-free QK-RMSNorm + 2D split-half RoPE.
pub fn emit_h3_image_vae_block(spec: &VaeBlockHloSpec) -> Result<String> {
    let b = spec.batch;
    let tok = spec.tokens;
    let d = spec.dim;
    let h = spec.heads;
    let hd = spec.head_dim;
    let f = spec.ffn;
    let rot = spec.rotary_dim;
    ensure!(
        b > 0 && tok > 0 && d > 0 && h > 0 && hd > 0 && f > 0 && d == h * hd,
        "vae_block dims invalid: {spec:?}"
    );
    ensure!(
        rot > 0 && rot <= hd && rot.is_multiple_of(2),
        "vae_block rotary_dim {rot} invalid for head_dim {hd}"
    );
    ensure!(
        spec.norm_eps.is_finite() && spec.norm_eps > 0.0,
        "vae_block norm_eps invalid: {}",
        spec.norm_eps
    );
    let f2 = 2 * f;
    let half = rot / 2;
    let pass = hd - rot;
    let scale_hex = f32_hex(1.0f32 / (hd as f32).sqrt());
    let mut out = String::with_capacity(14_336);
    writeln!(
        out,
        "module @ff_h3_image_vae_block {{
  func.func public @main(
    %x: tensor<{b}x{tok}x{d}xf32>,
    %rotary: tensor<{tok}x{rot}xbf16>,
    %norm1_w: tensor<{d}xf32>,
    %wq: tensor<{d}x{d}xbf16>,
    %bq: tensor<{d}xbf16>,
    %wk: tensor<{d}x{d}xbf16>,
    %bk: tensor<{d}xbf16>,
    %wv: tensor<{d}x{d}xbf16>,
    %bv: tensor<{d}xbf16>,
    %wout: tensor<{d}x{d}xbf16>,
    %bout: tensor<{d}xbf16>,
    %scale1: tensor<{d}xf32>,
    %norm2_w: tensor<{d}xf32>,
    %wff0: tensor<{f2}x{d}xbf16>,
    %bff0: tensor<{f2}xbf16>,
    %wff2: tensor<{d}x{f}xbf16>,
    %bff2: tensor<{d}xbf16>,
    %scale2: tensor<{d}xf32>
  ) -> tensor<{b}x{tok}x{d}xf32> {{
    %f32_zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>"
    )?;
    push_vae_rms_norm_f32_to_bf16(&mut out, "h1", "%x", "%norm1_w", (b, tok, d, spec.norm_eps))?;
    push_batched_biased_bf16_linear(&mut out, "q_2d", "%h1", ("%wq", "%bq"), (b, tok, d, d))?;
    push_batched_biased_bf16_linear(&mut out, "k_2d", "%h1", ("%wk", "%bk"), (b, tok, d, d))?;
    push_batched_biased_bf16_linear(&mut out, "v_2d", "%h1", ("%wv", "%bv"), (b, tok, d, d))?;
    writeln!(
        out,
        "    %q_4d = stablehlo.reshape %q_2d : (tensor<{b}x{tok}x{d}xbf16>) -> tensor<{b}x{tok}x{h}x{hd}xbf16>
    %k_4d = stablehlo.reshape %k_2d : (tensor<{b}x{tok}x{d}xbf16>) -> tensor<{b}x{tok}x{h}x{hd}xbf16>
    %v_4d = stablehlo.reshape %v_2d : (tensor<{b}x{tok}x{d}xbf16>) -> tensor<{b}x{tok}x{h}x{hd}xbf16>"
    )?;
    push_vae_qk_rms_norm_4d(&mut out, "qn", "%q_4d", (b, tok, h, hd, spec.norm_eps))?;
    push_vae_qk_rms_norm_4d(&mut out, "kn", "%k_4d", (b, tok, h, hd, spec.norm_eps))?;
    writeln!(
        out,
        "    %cos_2d = \"stablehlo.slice\"(%rotary) {{start_indices = array<i64: 0, 0>, limit_indices = array<i64: {tok}, {half}>, strides = array<i64: 1, 1>}} : (tensor<{tok}x{rot}xbf16>) -> tensor<{tok}x{half}xbf16>
    %sin_2d = \"stablehlo.slice\"(%rotary) {{start_indices = array<i64: 0, {half}>, limit_indices = array<i64: {tok}, {rot}>, strides = array<i64: 1, 1>}} : (tensor<{tok}x{rot}xbf16>) -> tensor<{tok}x{half}xbf16>
    %cos_b = stablehlo.broadcast_in_dim %cos_2d, dims = [1, 3] : (tensor<{tok}x{half}xbf16>) -> tensor<{b}x{tok}x{h}x{half}xbf16>
    %sin_b = stablehlo.broadcast_in_dim %sin_2d, dims = [1, 3] : (tensor<{tok}x{half}xbf16>) -> tensor<{b}x{tok}x{h}x{half}xbf16>"
    )?;
    push_rope_4d(&mut out, "q_rot", "%qn", (b, tok, h, hd), (rot, pass))?;
    push_rope_4d(&mut out, "k_rot", "%kn", (b, tok, h, hd), (rot, pass))?;
    writeln!(
        out,
        "    %qt = stablehlo.transpose %q_rot, dims = [0, 2, 1, 3] : (tensor<{b}x{tok}x{h}x{hd}xbf16>) -> tensor<{b}x{h}x{tok}x{hd}xbf16>
    %kt = stablehlo.transpose %k_rot, dims = [0, 2, 1, 3] : (tensor<{b}x{tok}x{h}x{hd}xbf16>) -> tensor<{b}x{h}x{tok}x{hd}xbf16>
    %vt = stablehlo.transpose %v_4d, dims = [0, 2, 1, 3] : (tensor<{b}x{tok}x{h}x{hd}xbf16>) -> tensor<{b}x{h}x{tok}x{hd}xbf16>
    %qk = stablehlo.dot_general %qt, %kt, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<{b}x{h}x{tok}x{hd}xbf16>, tensor<{b}x{h}x{tok}x{hd}xbf16>) -> tensor<{b}x{h}x{tok}x{tok}xf32>
    %scale_s = stablehlo.constant dense<{scale_hex}> : tensor<f32>
    %scale_t = stablehlo.broadcast_in_dim %scale_s, dims = [] : (tensor<f32>) -> tensor<{b}x{h}x{tok}x{tok}xf32>
    %scores = stablehlo.multiply %qk, %scale_t : tensor<{b}x{h}x{tok}x{tok}xf32>
    %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %smax = \"stablehlo.reduce\"(%scores, %neg_inf) ({{
      ^bb0(%a: tensor<f32>, %c: tensor<f32>):
        %m = stablehlo.maximum %a, %c : tensor<f32>
        stablehlo.return %m : tensor<f32>
    }}) {{dimensions = array<i64: 3>}} : (tensor<{b}x{h}x{tok}x{tok}xf32>, tensor<f32>) -> tensor<{b}x{h}x{tok}xf32>
    %smax_b = stablehlo.broadcast_in_dim %smax, dims = [0, 1, 2] : (tensor<{b}x{h}x{tok}xf32>) -> tensor<{b}x{h}x{tok}x{tok}xf32>
    %shifted = stablehlo.subtract %scores, %smax_b : tensor<{b}x{h}x{tok}x{tok}xf32>
    %exp = stablehlo.exponential %shifted : tensor<{b}x{h}x{tok}x{tok}xf32>
    %esum = \"stablehlo.reduce\"(%exp, %f32_zero) ({{
      ^bb0(%a: tensor<f32>, %c: tensor<f32>):
        %s = stablehlo.add %a, %c : tensor<f32>
        stablehlo.return %s : tensor<f32>
    }}) {{dimensions = array<i64: 3>}} : (tensor<{b}x{h}x{tok}x{tok}xf32>, tensor<f32>) -> tensor<{b}x{h}x{tok}xf32>
    %esum_b = stablehlo.broadcast_in_dim %esum, dims = [0, 1, 2] : (tensor<{b}x{h}x{tok}xf32>) -> tensor<{b}x{h}x{tok}x{tok}xf32>
    %probs_f32 = stablehlo.divide %exp, %esum_b : tensor<{b}x{h}x{tok}x{tok}xf32>
    %probs = stablehlo.convert %probs_f32 : (tensor<{b}x{h}x{tok}x{tok}xf32>) -> tensor<{b}x{h}x{tok}x{tok}xbf16>
    %pv_f32 = stablehlo.dot_general %probs, %vt, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<{b}x{h}x{tok}x{tok}xbf16>, tensor<{b}x{h}x{tok}x{hd}xbf16>) -> tensor<{b}x{h}x{tok}x{hd}xf32>
    %pv_t = stablehlo.transpose %pv_f32, dims = [0, 2, 1, 3] : (tensor<{b}x{h}x{tok}x{hd}xf32>) -> tensor<{b}x{tok}x{h}x{hd}xf32>
    %pv_bf16 = stablehlo.convert %pv_t : (tensor<{b}x{tok}x{h}x{hd}xf32>) -> tensor<{b}x{tok}x{h}x{hd}xbf16>
    %attn_flat = stablehlo.reshape %pv_bf16 : (tensor<{b}x{tok}x{h}x{hd}xbf16>) -> tensor<{b}x{tok}x{d}xbf16>"
    )?;
    push_batched_biased_bf16_linear(
        &mut out,
        "aout",
        "%attn_flat",
        ("%wout", "%bout"),
        (b, tok, d, d),
    )?;
    writeln!(
        out,
        "    %aout_f32 = stablehlo.convert %aout : (tensor<{b}x{tok}x{d}xbf16>) -> tensor<{b}x{tok}x{d}xf32>
    %s1_b = stablehlo.broadcast_in_dim %scale1, dims = [2] : (tensor<{d}xf32>) -> tensor<{b}x{tok}x{d}xf32>
    %aout_s = stablehlo.multiply %aout_f32, %s1_b : tensor<{b}x{tok}x{d}xf32>
    %x_mid = stablehlo.add %x, %aout_s : tensor<{b}x{tok}x{d}xf32>"
    )?;
    push_vae_rms_norm_f32_to_bf16(
        &mut out,
        "h2",
        "%x_mid",
        "%norm2_w",
        (b, tok, d, spec.norm_eps),
    )?;
    push_batched_biased_bf16_linear(&mut out, "up", "%h2", ("%wff0", "%bff0"), (b, tok, d, f2))?;
    writeln!(
        out,
        "    %ff_val = \"stablehlo.slice\"(%up) {{start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: {b}, {tok}, {f}>, strides = array<i64: 1, 1, 1>}} : (tensor<{b}x{tok}x{f2}xbf16>) -> tensor<{b}x{tok}x{f}xbf16>
    %ff_gate = \"stablehlo.slice\"(%up) {{start_indices = array<i64: 0, 0, {f}>, limit_indices = array<i64: {b}, {tok}, {f2}>, strides = array<i64: 1, 1, 1>}} : (tensor<{b}x{tok}x{f2}xbf16>) -> tensor<{b}x{tok}x{f}xbf16>
    %ff_gate_f32 = stablehlo.convert %ff_gate : (tensor<{b}x{tok}x{f}xbf16>) -> tensor<{b}x{tok}x{f}xf32>
    %ff_sig_f32 = stablehlo.logistic %ff_gate_f32 : tensor<{b}x{tok}x{f}xf32>
    %ff_silu_f32 = stablehlo.multiply %ff_gate_f32, %ff_sig_f32 : tensor<{b}x{tok}x{f}xf32>
    %ff_silu = stablehlo.convert %ff_silu_f32 : (tensor<{b}x{tok}x{f}xf32>) -> tensor<{b}x{tok}x{f}xbf16>
    %ff_act = stablehlo.multiply %ff_val, %ff_silu : tensor<{b}x{tok}x{f}xbf16>"
    )?;
    push_batched_biased_bf16_linear(
        &mut out,
        "ff_out",
        "%ff_act",
        ("%wff2", "%bff2"),
        (b, tok, f, d),
    )?;
    writeln!(
        out,
        "    %ff_out_f32 = stablehlo.convert %ff_out : (tensor<{b}x{tok}x{d}xbf16>) -> tensor<{b}x{tok}x{d}xf32>
    %s2_b = stablehlo.broadcast_in_dim %scale2, dims = [2] : (tensor<{d}xf32>) -> tensor<{b}x{tok}x{d}xf32>
    %ff_out_s = stablehlo.multiply %ff_out_f32, %s2_b : tensor<{b}x{tok}x{d}xf32>
    %out = stablehlo.add %x_mid, %ff_out_s : tensor<{b}x{tok}x{d}xf32>
    func.return %out : tensor<{b}x{tok}x{d}xf32>
  }}
}}"
    )?;
    Ok(out)
}

/// Emits Image VAE post-projection: F32 `LayerNorm` (`norm_out`) -> BF16 -> slice `:num_patches` -> BF16 `proj_out`.
pub fn emit_h3_image_vae_post(spec: &VaePostHloSpec) -> Result<String> {
    let b = spec.batch;
    let tok = spec.tokens;
    let p = spec.num_patches;
    let d = spec.dim;
    let pw = spec.patch_width;
    ensure!(
        b > 0 && p > 0 && p <= tok && d > 0 && pw > 0,
        "vae_post dims invalid: {spec:?}"
    );
    ensure!(
        spec.norm_eps.is_finite() && spec.norm_eps > 0.0,
        "vae_post norm_eps invalid: {}",
        spec.norm_eps
    );
    let inv_d = f32_hex(1.0f32 / d as f32);
    let eps_hex = f32_hex(spec.norm_eps);
    Ok(format!(
        "\
module @ff_h3_image_vae_post {{
  func.func public @main(
    %x: tensor<{b}x{tok}x{d}xf32>,
    %norm_w: tensor<{d}xf32>,
    %norm_b: tensor<{d}xf32>,
    %w_out: tensor<{pw}x{d}xbf16>,
    %b_out: tensor<{pw}xbf16>
  ) -> tensor<{b}x{p}x{pw}xbf16> {{
    %f32_zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %x_patches = \"stablehlo.slice\"(%x) {{start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: {b}, {p}, {d}>, strides = array<i64: 1, 1, 1>}} : (tensor<{b}x{tok}x{d}xf32>) -> tensor<{b}x{p}x{d}xf32>
    %sum = \"stablehlo.reduce\"(%x_patches, %f32_zero) ({{
      ^bb0(%a: tensor<f32>, %c: tensor<f32>):
        %s = stablehlo.add %a, %c : tensor<f32>
        stablehlo.return %s : tensor<f32>
    }}) {{dimensions = array<i64: 2>}} : (tensor<{b}x{p}x{d}xf32>, tensor<f32>) -> tensor<{b}x{p}xf32>
    %inv_s = stablehlo.constant dense<{inv_d}> : tensor<f32>
    %inv_t = stablehlo.broadcast_in_dim %inv_s, dims = [] : (tensor<f32>) -> tensor<{b}x{p}xf32>
    %mean = stablehlo.multiply %sum, %inv_t : tensor<{b}x{p}xf32>
    %mean_b = stablehlo.broadcast_in_dim %mean, dims = [0, 1] : (tensor<{b}x{p}xf32>) -> tensor<{b}x{p}x{d}xf32>
    %centered = stablehlo.subtract %x_patches, %mean_b : tensor<{b}x{p}x{d}xf32>
    %sq = stablehlo.multiply %centered, %centered : tensor<{b}x{p}x{d}xf32>
    %vsum = \"stablehlo.reduce\"(%sq, %f32_zero) ({{
      ^bb0(%a: tensor<f32>, %c: tensor<f32>):
        %s = stablehlo.add %a, %c : tensor<f32>
        stablehlo.return %s : tensor<f32>
    }}) {{dimensions = array<i64: 2>}} : (tensor<{b}x{p}x{d}xf32>, tensor<f32>) -> tensor<{b}x{p}xf32>
    %var = stablehlo.multiply %vsum, %inv_t : tensor<{b}x{p}xf32>
    %eps_s = stablehlo.constant dense<{eps_hex}> : tensor<f32>
    %eps_t = stablehlo.broadcast_in_dim %eps_s, dims = [] : (tensor<f32>) -> tensor<{b}x{p}xf32>
    %veps = stablehlo.add %var, %eps_t : tensor<{b}x{p}xf32>
    %rstd = stablehlo.rsqrt %veps : tensor<{b}x{p}xf32>
    %rstd_b = stablehlo.broadcast_in_dim %rstd, dims = [0, 1] : (tensor<{b}x{p}xf32>) -> tensor<{b}x{p}x{d}xf32>
    %normed = stablehlo.multiply %centered, %rstd_b : tensor<{b}x{p}x{d}xf32>
    %wb = stablehlo.broadcast_in_dim %norm_w, dims = [2] : (tensor<{d}xf32>) -> tensor<{b}x{p}x{d}xf32>
    %bb = stablehlo.broadcast_in_dim %norm_b, dims = [2] : (tensor<{d}xf32>) -> tensor<{b}x{p}x{d}xf32>
    %scaled = stablehlo.multiply %normed, %wb : tensor<{b}x{p}x{d}xf32>
    %ln_f32 = stablehlo.add %scaled, %bb : tensor<{b}x{p}x{d}xf32>
    %ln = stablehlo.convert %ln_f32 : (tensor<{b}x{p}x{d}xf32>) -> tensor<{b}x{p}x{d}xbf16>
    %dot = stablehlo.dot_general %ln, %w_out, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{b}x{p}x{d}xbf16>, tensor<{pw}x{d}xbf16>) -> tensor<{b}x{p}x{pw}xf32>
    %bo_f32 = stablehlo.convert %b_out : (tensor<{pw}xbf16>) -> tensor<{pw}xf32>
    %bo_b = stablehlo.broadcast_in_dim %bo_f32, dims = [2] : (tensor<{pw}xf32>) -> tensor<{b}x{p}x{pw}xf32>
    %sum_out = stablehlo.add %dot, %bo_b : tensor<{b}x{p}x{pw}xf32>
    %out = stablehlo.convert %sum_out : (tensor<{b}x{p}x{pw}xf32>) -> tensor<{b}x{p}x{pw}xbf16>
    func.return %out : tensor<{b}x{p}x{pw}xbf16>
  }}
}}
"
    ))
}

/// Emits one Qwen3-VL text decoder layer in published BF16 precision (causal GQA, QK-RMSNorm, split-half mRoPE, SwiGLU).
pub fn emit_qwen3_vl_text_layer(spec: &Qwen3VlLayerHloSpec) -> Result<String> {
    let r = spec.rows;
    let d = spec.hidden;
    let h = spec.heads;
    let kvh = spec.kv_heads;
    let hd = spec.head_dim;
    let inter = spec.intermediate;
    let hc = spec.head_chunk;
    ensure!(
        r > 0 && d > 0 && h > 0 && kvh > 0 && hd > 0 && inter > 0,
        "qwen3_vl layer dims must be positive: {spec:?}"
    );
    ensure!(
        h.is_multiple_of(kvh) && hc > 0 && h.is_multiple_of(hc) && hd.is_multiple_of(2),
        "qwen3_vl head geometry invalid: {spec:?}"
    );
    ensure!(
        spec.rms_norm_eps.is_finite() && spec.rms_norm_eps > 0.0,
        "qwen3_vl rms_norm_eps invalid: {}",
        spec.rms_norm_eps
    );
    let q_width = h * hd;
    let kv_width = kvh * hd;
    let group = h / kvh;
    let chunks = h / hc;
    let half = hd / 2;
    let scale_hex = f32_hex(1.0f32 / (hd as f32).sqrt());
    let chunk_ty = format!("tensor<{chunks}x{hc}x{r}x{hd}xbf16>");
    let mask_ty = format!("tensor<{hc}x{r}x{r}xi1>");

    let mut out = String::with_capacity(14_336);
    writeln!(
        out,
        "module @ff_qwen3_vl_text_layer {{
  func.func public @main(
    %x: tensor<{r}x{d}xbf16>,
    %rope: tensor<{r}x{hd}xbf16>,
    %norm1_w: tensor<{d}xbf16>,
    %wq: tensor<{q_width}x{d}xbf16>,
    %wk: tensor<{kv_width}x{d}xbf16>,
    %wv: tensor<{kv_width}x{d}xbf16>,
    %qn_w: tensor<{hd}xbf16>,
    %kn_w: tensor<{hd}xbf16>,
    %wo: tensor<{d}x{q_width}xbf16>,
    %norm2_w: tensor<{d}xbf16>,
    %wgate: tensor<{inter}x{d}xbf16>,
    %wup: tensor<{inter}x{d}xbf16>,
    %wdown: tensor<{d}x{inter}xbf16>
  ) -> tensor<{r}x{d}xbf16> {{
    %f32_zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %bf16_zero_s = stablehlo.convert %f32_zero : (tensor<f32>) -> tensor<bf16>"
    )?;
    push_rms_norm_2d_raw(&mut out, "h1", "%x", "%norm1_w", (r, d, spec.rms_norm_eps))?;
    writeln!(
        out,
        "    %q_f32 = stablehlo.dot_general %h1, %wq, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{d}xbf16>, tensor<{q_width}x{d}xbf16>) -> tensor<{r}x{q_width}xf32>
    %q_2d = stablehlo.convert %q_f32 : (tensor<{r}x{q_width}xf32>) -> tensor<{r}x{q_width}xbf16>
    %q_3d = stablehlo.reshape %q_2d : (tensor<{r}x{q_width}xbf16>) -> tensor<{r}x{h}x{hd}xbf16>
    %k_f32 = stablehlo.dot_general %h1, %wk, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{d}xbf16>, tensor<{kv_width}x{d}xbf16>) -> tensor<{r}x{kv_width}xf32>
    %k_2d = stablehlo.convert %k_f32 : (tensor<{r}x{kv_width}xf32>) -> tensor<{r}x{kv_width}xbf16>
    %k_3d = stablehlo.reshape %k_2d : (tensor<{r}x{kv_width}xbf16>) -> tensor<{r}x{kvh}x{hd}xbf16>
    %v_f32 = stablehlo.dot_general %h1, %wv, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{d}xbf16>, tensor<{kv_width}x{d}xbf16>) -> tensor<{r}x{kv_width}xf32>
    %v_2d = stablehlo.convert %v_f32 : (tensor<{r}x{kv_width}xf32>) -> tensor<{r}x{kv_width}xbf16>
    %v_3d = stablehlo.reshape %v_2d : (tensor<{r}x{kv_width}xbf16>) -> tensor<{r}x{kvh}x{hd}xbf16>"
    )?;
    push_rms_norm_3d_raw(
        &mut out,
        "qn",
        "%q_3d",
        "%qn_w",
        (r, h, hd, spec.rms_norm_eps),
    )?;
    push_rms_norm_3d_raw(
        &mut out,
        "kn",
        "%k_3d",
        "%kn_w",
        (r, kvh, hd, spec.rms_norm_eps),
    )?;
    writeln!(
        out,
        "    %cos_2d = \"stablehlo.slice\"(%rope) {{start_indices = array<i64: 0, 0>, limit_indices = array<i64: {r}, {half}>, strides = array<i64: 1, 1>}} : (tensor<{r}x{hd}xbf16>) -> tensor<{r}x{half}xbf16>
    %sin_2d = \"stablehlo.slice\"(%rope) {{start_indices = array<i64: 0, {half}>, limit_indices = array<i64: {r}, {hd}>, strides = array<i64: 1, 1>}} : (tensor<{r}x{hd}xbf16>) -> tensor<{r}x{half}xbf16>
    %cos_q = stablehlo.broadcast_in_dim %cos_2d, dims = [0, 2] : (tensor<{r}x{half}xbf16>) -> tensor<{r}x{h}x{half}xbf16>
    %sin_q = stablehlo.broadcast_in_dim %sin_2d, dims = [0, 2] : (tensor<{r}x{half}xbf16>) -> tensor<{r}x{h}x{half}xbf16>
    %cos_k = stablehlo.broadcast_in_dim %cos_2d, dims = [0, 2] : (tensor<{r}x{half}xbf16>) -> tensor<{r}x{kvh}x{half}xbf16>
    %sin_k = stablehlo.broadcast_in_dim %sin_2d, dims = [0, 2] : (tensor<{r}x{half}xbf16>) -> tensor<{r}x{kvh}x{half}xbf16>"
    )?;
    push_full_rope_3d(&mut out, "q_rot", "%qn", ("%cos_q", "%sin_q"), (r, h, hd))?;
    push_full_rope_3d(&mut out, "k_rot", "%kn", ("%cos_k", "%sin_k"), (r, kvh, hd))?;
    writeln!(
        out,
        "    %k_rep = stablehlo.broadcast_in_dim %k_rot, dims = [0, 1, 3] : (tensor<{r}x{kvh}x{hd}xbf16>) -> tensor<{r}x{kvh}x{group}x{hd}xbf16>
    %k_full = stablehlo.reshape %k_rep : (tensor<{r}x{kvh}x{group}x{hd}xbf16>) -> tensor<{r}x{h}x{hd}xbf16>
    %v_rep = stablehlo.broadcast_in_dim %v_3d, dims = [0, 1, 3] : (tensor<{r}x{kvh}x{hd}xbf16>) -> tensor<{r}x{kvh}x{group}x{hd}xbf16>
    %v_full = stablehlo.reshape %v_rep : (tensor<{r}x{kvh}x{group}x{hd}xbf16>) -> tensor<{r}x{h}x{hd}xbf16>
    %q_4d = stablehlo.reshape %q_rot : (tensor<{r}x{h}x{hd}xbf16>) -> tensor<{r}x{chunks}x{hc}x{hd}xbf16>
    %qt = stablehlo.transpose %q_4d, dims = [1, 2, 0, 3] : (tensor<{r}x{chunks}x{hc}x{hd}xbf16>) -> {chunk_ty}
    %k_4d = stablehlo.reshape %k_full : (tensor<{r}x{h}x{hd}xbf16>) -> tensor<{r}x{chunks}x{hc}x{hd}xbf16>
    %kt = stablehlo.transpose %k_4d, dims = [1, 2, 0, 3] : (tensor<{r}x{chunks}x{hc}x{hd}xbf16>) -> {chunk_ty}
    %v_4d = stablehlo.reshape %v_full : (tensor<{r}x{h}x{hd}xbf16>) -> tensor<{r}x{chunks}x{hc}x{hd}xbf16>
    %vt = stablehlo.transpose %v_4d, dims = [1, 2, 0, 3] : (tensor<{r}x{chunks}x{hc}x{hd}xbf16>) -> {chunk_ty}
    %idx = \"stablehlo.iota\"() {{iota_dimension = 0 : i64}} : () -> tensor<{r}xi32>
    %q_pos = stablehlo.broadcast_in_dim %idx, dims = [0] : (tensor<{r}xi32>) -> tensor<{r}x{r}xi32>
    %k_pos = stablehlo.broadcast_in_dim %idx, dims = [1] : (tensor<{r}xi32>) -> tensor<{r}x{r}xi32>
    %mask_2d = stablehlo.compare GE, %q_pos, %k_pos, SIGNED : (tensor<{r}x{r}xi32>, tensor<{r}x{r}xi32>) -> tensor<{r}x{r}xi1>
    %causal_mask = stablehlo.broadcast_in_dim %mask_2d, dims = [1, 2] : (tensor<{r}x{r}xi1>) -> {mask_ty}
    %i_init = stablehlo.constant dense<0> : tensor<i32>
    %acc_init = stablehlo.broadcast_in_dim %bf16_zero_s, dims = [] : (tensor<bf16>) -> {chunk_ty}
    %attn_while:6 = \"stablehlo.while\"(%i_init, %acc_init, %qt, %kt, %vt, %causal_mask) ({{
      ^bb0(%wi: tensor<i32>, %wacc: {chunk_ty}, %wqt: {chunk_ty}, %wkt: {chunk_ty}, %wvt: {chunk_ty}, %wmask: {mask_ty}):
        %wlim = stablehlo.constant dense<{chunks}> : tensor<i32>
        %wcond = stablehlo.compare LT, %wi, %wlim, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %wcond : tensor<i1>
    }}, {{
      ^bb0(%wi: tensor<i32>, %wacc: {chunk_ty}, %wqt: {chunk_ty}, %wkt: {chunk_ty}, %wvt: {chunk_ty}, %wmask: {mask_ty}):
        %wc0 = stablehlo.constant dense<0> : tensor<i32>
        %wc1 = stablehlo.constant dense<1> : tensor<i32>
        %wqc_4d = \"stablehlo.dynamic_slice\"(%wqt, %wi, %wc0, %wc0, %wc0) {{slice_sizes = array<i64: 1, {hc}, {r}, {hd}>}} : ({chunk_ty}, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x{hc}x{r}x{hd}xbf16>
        %wqc = stablehlo.reshape %wqc_4d : (tensor<1x{hc}x{r}x{hd}xbf16>) -> tensor<{hc}x{r}x{hd}xbf16>
        %wkc_4d = \"stablehlo.dynamic_slice\"(%wkt, %wi, %wc0, %wc0, %wc0) {{slice_sizes = array<i64: 1, {hc}, {r}, {hd}>}} : ({chunk_ty}, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x{hc}x{r}x{hd}xbf16>
        %wkc = stablehlo.reshape %wkc_4d : (tensor<1x{hc}x{r}x{hd}xbf16>) -> tensor<{hc}x{r}x{hd}xbf16>
        %wvc_4d = \"stablehlo.dynamic_slice\"(%wvt, %wi, %wc0, %wc0, %wc0) {{slice_sizes = array<i64: 1, {hc}, {r}, {hd}>}} : ({chunk_ty}, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x{hc}x{r}x{hd}xbf16>
        %wvc = stablehlo.reshape %wvc_4d : (tensor<1x{hc}x{r}x{hd}xbf16>) -> tensor<{hc}x{r}x{hd}xbf16>
        %wqk = stablehlo.dot_general %wqc, %wkc, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<{hc}x{r}x{hd}xbf16>, tensor<{hc}x{r}x{hd}xbf16>) -> tensor<{hc}x{r}x{r}xf32>
        %wscale_s = stablehlo.constant dense<{scale_hex}> : tensor<f32>
        %wscale = stablehlo.broadcast_in_dim %wscale_s, dims = [] : (tensor<f32>) -> tensor<{hc}x{r}x{r}xf32>
        %wscores_raw = stablehlo.multiply %wqk, %wscale : tensor<{hc}x{r}x{r}xf32>
        %wneg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
        %wneg_inf_t = stablehlo.broadcast_in_dim %wneg_inf, dims = [] : (tensor<f32>) -> tensor<{hc}x{r}x{r}xf32>
        %wscores = \"stablehlo.select\"(%wmask, %wscores_raw, %wneg_inf_t) : ({mask_ty}, tensor<{hc}x{r}x{r}xf32>, tensor<{hc}x{r}x{r}xf32>) -> tensor<{hc}x{r}x{r}xf32>
        %wmax = \"stablehlo.reduce\"(%wscores, %wneg_inf) ({{
          ^bb0(%a: tensor<f32>, %b: tensor<f32>):
            %m = stablehlo.maximum %a, %b : tensor<f32>
            stablehlo.return %m : tensor<f32>
        }}) {{dimensions = array<i64: 2>}} : (tensor<{hc}x{r}x{r}xf32>, tensor<f32>) -> tensor<{hc}x{r}xf32>
        %wmax_b = stablehlo.broadcast_in_dim %wmax, dims = [0, 1] : (tensor<{hc}x{r}xf32>) -> tensor<{hc}x{r}x{r}xf32>
        %wshifted = stablehlo.subtract %wscores, %wmax_b : tensor<{hc}x{r}x{r}xf32>
        %wexp = stablehlo.exponential %wshifted : tensor<{hc}x{r}x{r}xf32>
        %wzero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %wsum = \"stablehlo.reduce\"(%wexp, %wzero) ({{
          ^bb0(%a: tensor<f32>, %b: tensor<f32>):
            %s = stablehlo.add %a, %b : tensor<f32>
            stablehlo.return %s : tensor<f32>
        }}) {{dimensions = array<i64: 2>}} : (tensor<{hc}x{r}x{r}xf32>, tensor<f32>) -> tensor<{hc}x{r}xf32>
        %wsum_b = stablehlo.broadcast_in_dim %wsum, dims = [0, 1] : (tensor<{hc}x{r}xf32>) -> tensor<{hc}x{r}x{r}xf32>
        %wprobs_f32 = stablehlo.divide %wexp, %wsum_b : tensor<{hc}x{r}x{r}xf32>
        %wprobs = stablehlo.convert %wprobs_f32 : (tensor<{hc}x{r}x{r}xf32>) -> tensor<{hc}x{r}x{r}xbf16>
        %wpv_f32 = stablehlo.dot_general %wprobs, %wvc, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{hc}x{r}x{r}xbf16>, tensor<{hc}x{r}x{hd}xbf16>) -> tensor<{hc}x{r}x{hd}xf32>
        %wpv_bf16 = stablehlo.convert %wpv_f32 : (tensor<{hc}x{r}x{hd}xf32>) -> tensor<{hc}x{r}x{hd}xbf16>
        %wpv_4d = stablehlo.reshape %wpv_bf16 : (tensor<{hc}x{r}x{hd}xbf16>) -> tensor<1x{hc}x{r}x{hd}xbf16>
        %wacc_next = \"stablehlo.dynamic_update_slice\"(%wacc, %wpv_4d, %wi, %wc0, %wc0, %wc0) : ({chunk_ty}, tensor<1x{hc}x{r}x{hd}xbf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> {chunk_ty}
        %wi_next = stablehlo.add %wi, %wc1 : tensor<i32>
        stablehlo.return %wi_next, %wacc_next, %wqt, %wkt, %wvt, %wmask : tensor<i32>, {chunk_ty}, {chunk_ty}, {chunk_ty}, {chunk_ty}, {mask_ty}
    }}) : (tensor<i32>, {chunk_ty}, {chunk_ty}, {chunk_ty}, {chunk_ty}, {mask_ty}) -> (tensor<i32>, {chunk_ty}, {chunk_ty}, {chunk_ty}, {chunk_ty}, {mask_ty})
    %attn_t = stablehlo.transpose %attn_while#1, dims = [2, 0, 1, 3] : ({chunk_ty}) -> tensor<{r}x{chunks}x{hc}x{hd}xbf16>
    %attn_flat = stablehlo.reshape %attn_t : (tensor<{r}x{chunks}x{hc}x{hd}xbf16>) -> tensor<{r}x{q_width}xbf16>
    %aout_f32 = stablehlo.dot_general %attn_flat, %wo, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{q_width}xbf16>, tensor<{d}x{q_width}xbf16>) -> tensor<{r}x{d}xf32>
    %aout = stablehlo.convert %aout_f32 : (tensor<{r}x{d}xf32>) -> tensor<{r}x{d}xbf16>
    %x_mid = stablehlo.add %x, %aout : tensor<{r}x{d}xbf16>"
    )?;
    push_rms_norm_2d_raw(
        &mut out,
        "h2",
        "%x_mid",
        "%norm2_w",
        (r, d, spec.rms_norm_eps),
    )?;
    writeln!(
        out,
        "    %gate_f32 = stablehlo.dot_general %h2, %wgate, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{d}xbf16>, tensor<{inter}x{d}xbf16>) -> tensor<{r}x{inter}xf32>
    %gate = stablehlo.convert %gate_f32 : (tensor<{r}x{inter}xf32>) -> tensor<{r}x{inter}xbf16>
    %up_f32 = stablehlo.dot_general %h2, %wup, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{d}xbf16>, tensor<{inter}x{d}xbf16>) -> tensor<{r}x{inter}xf32>
    %up = stablehlo.convert %up_f32 : (tensor<{r}x{inter}xf32>) -> tensor<{r}x{inter}xbf16>
    %g_f32 = stablehlo.convert %gate : (tensor<{r}x{inter}xbf16>) -> tensor<{r}x{inter}xf32>
    %g_sig = stablehlo.logistic %g_f32 : tensor<{r}x{inter}xf32>
    %g_silu_f32 = stablehlo.multiply %g_f32, %g_sig : tensor<{r}x{inter}xf32>
    %g_silu = stablehlo.convert %g_silu_f32 : (tensor<{r}x{inter}xf32>) -> tensor<{r}x{inter}xbf16>
    %mlp_act = stablehlo.multiply %g_silu, %up : tensor<{r}x{inter}xbf16>
    %down_f32 = stablehlo.dot_general %mlp_act, %wdown, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{r}x{inter}xbf16>, tensor<{d}x{inter}xbf16>) -> tensor<{r}x{d}xf32>
    %down = stablehlo.convert %down_f32 : (tensor<{r}x{d}xf32>) -> tensor<{r}x{d}xbf16>
    %out = stablehlo.add %x_mid, %down : tensor<{r}x{d}xbf16>
    func.return %out : tensor<{r}x{d}xbf16>
  }}
}}"
    )?;
    Ok(out)
}

fn push_rms_norm_2d_raw(
    out: &mut String,
    prefix: &str,
    input: &str,
    weight: &str,
    (rows, dim, eps): (usize, usize, f32),
) -> Result<()> {
    let inv_d = f32_hex(1.0f32 / dim as f32);
    let eps_hex = f32_hex(eps);
    writeln!(
        out,
        "    %{prefix}_f32 = stablehlo.convert {input} : (tensor<{rows}x{dim}xbf16>) -> tensor<{rows}x{dim}xf32>
    %{prefix}_sq = stablehlo.multiply %{prefix}_f32, %{prefix}_f32 : tensor<{rows}x{dim}xf32>
    %{prefix}_sum = \"stablehlo.reduce\"(%{prefix}_sq, %f32_zero) ({{
      ^bb0(%a: tensor<f32>, %b: tensor<f32>):
        %s = stablehlo.add %a, %b : tensor<f32>
        stablehlo.return %s : tensor<f32>
    }}) {{dimensions = array<i64: 1>}} : (tensor<{rows}x{dim}xf32>, tensor<f32>) -> tensor<{rows}xf32>
    %{prefix}_inv_s = stablehlo.constant dense<{inv_d}> : tensor<f32>
    %{prefix}_inv = stablehlo.broadcast_in_dim %{prefix}_inv_s, dims = [] : (tensor<f32>) -> tensor<{rows}xf32>
    %{prefix}_mean = stablehlo.multiply %{prefix}_sum, %{prefix}_inv : tensor<{rows}xf32>
    %{prefix}_eps_s = stablehlo.constant dense<{eps_hex}> : tensor<f32>
    %{prefix}_eps = stablehlo.broadcast_in_dim %{prefix}_eps_s, dims = [] : (tensor<f32>) -> tensor<{rows}xf32>
    %{prefix}_var = stablehlo.add %{prefix}_mean, %{prefix}_eps : tensor<{rows}xf32>
    %{prefix}_rstd = stablehlo.rsqrt %{prefix}_var : tensor<{rows}xf32>
    %{prefix}_rstd_b = stablehlo.broadcast_in_dim %{prefix}_rstd, dims = [0] : (tensor<{rows}xf32>) -> tensor<{rows}x{dim}xf32>
    %{prefix}_norm = stablehlo.multiply %{prefix}_f32, %{prefix}_rstd_b : tensor<{rows}x{dim}xf32>
    %{prefix}_wf32 = stablehlo.convert {weight} : (tensor<{dim}xbf16>) -> tensor<{dim}xf32>
    %{prefix}_wb = stablehlo.broadcast_in_dim %{prefix}_wf32, dims = [1] : (tensor<{dim}xf32>) -> tensor<{rows}x{dim}xf32>
    %{prefix}_scaled = stablehlo.multiply %{prefix}_norm, %{prefix}_wb : tensor<{rows}x{dim}xf32>
    %{prefix} = stablehlo.convert %{prefix}_scaled : (tensor<{rows}x{dim}xf32>) -> tensor<{rows}x{dim}xbf16>"
    )?;
    Ok(())
}

fn push_rms_norm_3d_raw(
    out: &mut String,
    prefix: &str,
    input: &str,
    weight: &str,
    (rows, heads, dim, eps): (usize, usize, usize, f32),
) -> Result<()> {
    let inv_d = f32_hex(1.0f32 / dim as f32);
    let eps_hex = f32_hex(eps);
    writeln!(
        out,
        "    %{prefix}_f32 = stablehlo.convert {input} : (tensor<{rows}x{heads}x{dim}xbf16>) -> tensor<{rows}x{heads}x{dim}xf32>
    %{prefix}_sq = stablehlo.multiply %{prefix}_f32, %{prefix}_f32 : tensor<{rows}x{heads}x{dim}xf32>
    %{prefix}_sum = \"stablehlo.reduce\"(%{prefix}_sq, %f32_zero) ({{
      ^bb0(%a: tensor<f32>, %b: tensor<f32>):
        %s = stablehlo.add %a, %b : tensor<f32>
        stablehlo.return %s : tensor<f32>
    }}) {{dimensions = array<i64: 2>}} : (tensor<{rows}x{heads}x{dim}xf32>, tensor<f32>) -> tensor<{rows}x{heads}xf32>
    %{prefix}_inv_s = stablehlo.constant dense<{inv_d}> : tensor<f32>
    %{prefix}_inv = stablehlo.broadcast_in_dim %{prefix}_inv_s, dims = [] : (tensor<f32>) -> tensor<{rows}x{heads}xf32>
    %{prefix}_mean = stablehlo.multiply %{prefix}_sum, %{prefix}_inv : tensor<{rows}x{heads}xf32>
    %{prefix}_eps_s = stablehlo.constant dense<{eps_hex}> : tensor<f32>
    %{prefix}_eps = stablehlo.broadcast_in_dim %{prefix}_eps_s, dims = [] : (tensor<f32>) -> tensor<{rows}x{heads}xf32>
    %{prefix}_var = stablehlo.add %{prefix}_mean, %{prefix}_eps : tensor<{rows}x{heads}xf32>
    %{prefix}_rstd = stablehlo.rsqrt %{prefix}_var : tensor<{rows}x{heads}xf32>
    %{prefix}_rstd_b = stablehlo.broadcast_in_dim %{prefix}_rstd, dims = [0, 1] : (tensor<{rows}x{heads}xf32>) -> tensor<{rows}x{heads}x{dim}xf32>
    %{prefix}_norm = stablehlo.multiply %{prefix}_f32, %{prefix}_rstd_b : tensor<{rows}x{heads}x{dim}xf32>
    %{prefix}_wf32 = stablehlo.convert {weight} : (tensor<{dim}xbf16>) -> tensor<{dim}xf32>
    %{prefix}_wb = stablehlo.broadcast_in_dim %{prefix}_wf32, dims = [2] : (tensor<{dim}xf32>) -> tensor<{rows}x{heads}x{dim}xf32>
    %{prefix}_scaled = stablehlo.multiply %{prefix}_norm, %{prefix}_wb : tensor<{rows}x{heads}x{dim}xf32>
    %{prefix} = stablehlo.convert %{prefix}_scaled : (tensor<{rows}x{heads}x{dim}xf32>) -> tensor<{rows}x{heads}x{dim}xbf16>"
    )?;
    Ok(())
}

fn push_full_rope_3d(
    out: &mut String,
    prefix: &str,
    input: &str,
    (cos_name, sin_name): (&str, &str),
    (rows, heads, head_dim): (usize, usize, usize),
) -> Result<()> {
    let half = head_dim / 2;
    writeln!(
        out,
        "    %{prefix}_x1 = \"stablehlo.slice\"({input}) {{start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: {rows}, {heads}, {half}>, strides = array<i64: 1, 1, 1>}} : (tensor<{rows}x{heads}x{head_dim}xbf16>) -> tensor<{rows}x{heads}x{half}xbf16>
    %{prefix}_x2 = \"stablehlo.slice\"({input}) {{start_indices = array<i64: 0, 0, {half}>, limit_indices = array<i64: {rows}, {heads}, {head_dim}>, strides = array<i64: 1, 1, 1>}} : (tensor<{rows}x{heads}x{head_dim}xbf16>) -> tensor<{rows}x{heads}x{half}xbf16>
    %{prefix}_t1 = stablehlo.multiply %{prefix}_x1, {cos_name} : tensor<{rows}x{heads}x{half}xbf16>
    %{prefix}_t2 = stablehlo.multiply %{prefix}_x2, {sin_name} : tensor<{rows}x{heads}x{half}xbf16>
    %{prefix}_lo = stablehlo.subtract %{prefix}_t1, %{prefix}_t2 : tensor<{rows}x{heads}x{half}xbf16>
    %{prefix}_t3 = stablehlo.multiply %{prefix}_x2, {cos_name} : tensor<{rows}x{heads}x{half}xbf16>
    %{prefix}_t4 = stablehlo.multiply %{prefix}_x1, {sin_name} : tensor<{rows}x{heads}x{half}xbf16>
    %{prefix}_hi = stablehlo.add %{prefix}_t3, %{prefix}_t4 : tensor<{rows}x{heads}x{half}xbf16>
    %{prefix} = \"stablehlo.concatenate\"(%{prefix}_lo, %{prefix}_hi) {{dimension = 2 : i64}} : (tensor<{rows}x{heads}x{half}xbf16>, tensor<{rows}x{heads}x{half}xbf16>) -> tensor<{rows}x{heads}x{head_dim}xbf16>"
    )?;
    Ok(())
}

fn push_vae_rms_norm_f32_to_bf16(
    out: &mut String,
    prefix: &str,
    input: &str,
    weight_f32: &str,
    (batch, tokens, dim, eps): (usize, usize, usize, f32),
) -> Result<()> {
    let inv_d = f32_hex(1.0f32 / dim as f32);
    let eps_hex = f32_hex(eps);
    writeln!(
        out,
        "    %{prefix}_sq = stablehlo.multiply {input}, {input} : tensor<{batch}x{tokens}x{dim}xf32>
    %{prefix}_sum = \"stablehlo.reduce\"(%{prefix}_sq, %f32_zero) ({{
      ^bb0(%a: tensor<f32>, %c: tensor<f32>):
        %s = stablehlo.add %a, %c : tensor<f32>
        stablehlo.return %s : tensor<f32>
    }}) {{dimensions = array<i64: 2>}} : (tensor<{batch}x{tokens}x{dim}xf32>, tensor<f32>) -> tensor<{batch}x{tokens}xf32>
    %{prefix}_inv_s = stablehlo.constant dense<{inv_d}> : tensor<f32>
    %{prefix}_inv = stablehlo.broadcast_in_dim %{prefix}_inv_s, dims = [] : (tensor<f32>) -> tensor<{batch}x{tokens}xf32>
    %{prefix}_mean = stablehlo.multiply %{prefix}_sum, %{prefix}_inv : tensor<{batch}x{tokens}xf32>
    %{prefix}_eps_s = stablehlo.constant dense<{eps_hex}> : tensor<f32>
    %{prefix}_eps = stablehlo.broadcast_in_dim %{prefix}_eps_s, dims = [] : (tensor<f32>) -> tensor<{batch}x{tokens}xf32>
    %{prefix}_var = stablehlo.add %{prefix}_mean, %{prefix}_eps : tensor<{batch}x{tokens}xf32>
    %{prefix}_rstd = stablehlo.rsqrt %{prefix}_var : tensor<{batch}x{tokens}xf32>
    %{prefix}_rstd_b = stablehlo.broadcast_in_dim %{prefix}_rstd, dims = [0, 1] : (tensor<{batch}x{tokens}xf32>) -> tensor<{batch}x{tokens}x{dim}xf32>
    %{prefix}_norm = stablehlo.multiply {input}, %{prefix}_rstd_b : tensor<{batch}x{tokens}x{dim}xf32>
    %{prefix}_wb = stablehlo.broadcast_in_dim {weight_f32}, dims = [2] : (tensor<{dim}xf32>) -> tensor<{batch}x{tokens}x{dim}xf32>
    %{prefix}_scaled = stablehlo.multiply %{prefix}_norm, %{prefix}_wb : tensor<{batch}x{tokens}x{dim}xf32>
    %{prefix} = stablehlo.convert %{prefix}_scaled : (tensor<{batch}x{tokens}x{dim}xf32>) -> tensor<{batch}x{tokens}x{dim}xbf16>"
    )?;
    Ok(())
}

fn push_batched_biased_bf16_linear(
    out: &mut String,
    prefix: &str,
    input: &str,
    (weight, bias): (&str, &str),
    (batch, tokens, in_dim, out_dim): (usize, usize, usize, usize),
) -> Result<()> {
    writeln!(
        out,
        "    %{prefix}_dot = stablehlo.dot_general {input}, {weight}, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<{batch}x{tokens}x{in_dim}xbf16>, tensor<{out_dim}x{in_dim}xbf16>) -> tensor<{batch}x{tokens}x{out_dim}xf32>
    %{prefix}_bf32 = stablehlo.convert {bias} : (tensor<{out_dim}xbf16>) -> tensor<{out_dim}xf32>
    %{prefix}_bb = stablehlo.broadcast_in_dim %{prefix}_bf32, dims = [2] : (tensor<{out_dim}xf32>) -> tensor<{batch}x{tokens}x{out_dim}xf32>
    %{prefix}_sum = stablehlo.add %{prefix}_dot, %{prefix}_bb : tensor<{batch}x{tokens}x{out_dim}xf32>
    %{prefix} = stablehlo.convert %{prefix}_sum : (tensor<{batch}x{tokens}x{out_dim}xf32>) -> tensor<{batch}x{tokens}x{out_dim}xbf16>"
    )?;
    Ok(())
}

fn push_vae_qk_rms_norm_4d(
    out: &mut String,
    prefix: &str,
    input: &str,
    (batch, tokens, heads, head_dim, eps): (usize, usize, usize, usize, f32),
) -> Result<()> {
    let inv_d = f32_hex(1.0f32 / head_dim as f32);
    let eps_hex = f32_hex(eps);
    writeln!(
        out,
        "    %{prefix}_f32 = stablehlo.convert {input} : (tensor<{batch}x{tokens}x{heads}x{head_dim}xbf16>) -> tensor<{batch}x{tokens}x{heads}x{head_dim}xf32>
    %{prefix}_sq = stablehlo.multiply %{prefix}_f32, %{prefix}_f32 : tensor<{batch}x{tokens}x{heads}x{head_dim}xf32>
    %{prefix}_sum = \"stablehlo.reduce\"(%{prefix}_sq, %f32_zero) ({{
      ^bb0(%a: tensor<f32>, %c: tensor<f32>):
        %s = stablehlo.add %a, %c : tensor<f32>
        stablehlo.return %s : tensor<f32>
    }}) {{dimensions = array<i64: 3>}} : (tensor<{batch}x{tokens}x{heads}x{head_dim}xf32>, tensor<f32>) -> tensor<{batch}x{tokens}x{heads}xf32>
    %{prefix}_inv_s = stablehlo.constant dense<{inv_d}> : tensor<f32>
    %{prefix}_inv = stablehlo.broadcast_in_dim %{prefix}_inv_s, dims = [] : (tensor<f32>) -> tensor<{batch}x{tokens}x{heads}xf32>
    %{prefix}_mean = stablehlo.multiply %{prefix}_sum, %{prefix}_inv : tensor<{batch}x{tokens}x{heads}xf32>
    %{prefix}_eps_s = stablehlo.constant dense<{eps_hex}> : tensor<f32>
    %{prefix}_eps = stablehlo.broadcast_in_dim %{prefix}_eps_s, dims = [] : (tensor<f32>) -> tensor<{batch}x{tokens}x{heads}xf32>
    %{prefix}_var = stablehlo.add %{prefix}_mean, %{prefix}_eps : tensor<{batch}x{tokens}x{heads}xf32>
    %{prefix}_rstd = stablehlo.rsqrt %{prefix}_var : tensor<{batch}x{tokens}x{heads}xf32>
    %{prefix}_rstd_b = stablehlo.broadcast_in_dim %{prefix}_rstd, dims = [0, 1, 2] : (tensor<{batch}x{tokens}x{heads}xf32>) -> tensor<{batch}x{tokens}x{heads}x{head_dim}xf32>
    %{prefix}_norm = stablehlo.multiply %{prefix}_f32, %{prefix}_rstd_b : tensor<{batch}x{tokens}x{heads}x{head_dim}xf32>
    %{prefix} = stablehlo.convert %{prefix}_norm : (tensor<{batch}x{tokens}x{heads}x{head_dim}xf32>) -> tensor<{batch}x{tokens}x{heads}x{head_dim}xbf16>"
    )?;
    Ok(())
}

fn push_rope_4d(
    out: &mut String,
    prefix: &str,
    input: &str,
    (batch, tokens, heads, head_dim): (usize, usize, usize, usize),
    (rotary_dim, pass): (usize, usize),
) -> Result<()> {
    let half = rotary_dim / 2;
    writeln!(
        out,
        "    %{prefix}_x1 = \"stablehlo.slice\"({input}) {{start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: {batch}, {tokens}, {heads}, {half}>, strides = array<i64: 1, 1, 1, 1>}} : (tensor<{batch}x{tokens}x{heads}x{head_dim}xbf16>) -> tensor<{batch}x{tokens}x{heads}x{half}xbf16>
    %{prefix}_x2 = \"stablehlo.slice\"({input}) {{start_indices = array<i64: 0, 0, 0, {half}>, limit_indices = array<i64: {batch}, {tokens}, {heads}, {rotary_dim}>, strides = array<i64: 1, 1, 1, 1>}} : (tensor<{batch}x{tokens}x{heads}x{head_dim}xbf16>) -> tensor<{batch}x{tokens}x{heads}x{half}xbf16>
    %{prefix}_t1 = stablehlo.multiply %{prefix}_x1, %cos_b : tensor<{batch}x{tokens}x{heads}x{half}xbf16>
    %{prefix}_t2 = stablehlo.multiply %{prefix}_x2, %sin_b : tensor<{batch}x{tokens}x{heads}x{half}xbf16>
    %{prefix}_lo = stablehlo.subtract %{prefix}_t1, %{prefix}_t2 : tensor<{batch}x{tokens}x{heads}x{half}xbf16>
    %{prefix}_t3 = stablehlo.multiply %{prefix}_x2, %cos_b : tensor<{batch}x{tokens}x{heads}x{half}xbf16>
    %{prefix}_t4 = stablehlo.multiply %{prefix}_x1, %sin_b : tensor<{batch}x{tokens}x{heads}x{half}xbf16>
    %{prefix}_hi = stablehlo.add %{prefix}_t3, %{prefix}_t4 : tensor<{batch}x{tokens}x{heads}x{half}xbf16>"
    )?;
    if pass > 0 {
        writeln!(
            out,
            "    %{prefix}_xp = \"stablehlo.slice\"({input}) {{start_indices = array<i64: 0, 0, 0, {rotary_dim}>, limit_indices = array<i64: {batch}, {tokens}, {heads}, {head_dim}>, strides = array<i64: 1, 1, 1, 1>}} : (tensor<{batch}x{tokens}x{heads}x{head_dim}xbf16>) -> tensor<{batch}x{tokens}x{heads}x{pass}xbf16>
    %{prefix} = \"stablehlo.concatenate\"(%{prefix}_lo, %{prefix}_hi, %{prefix}_xp) {{dimension = 3 : i64}} : (tensor<{batch}x{tokens}x{heads}x{half}xbf16>, tensor<{batch}x{tokens}x{heads}x{half}xbf16>, tensor<{batch}x{tokens}x{heads}x{pass}xbf16>) -> tensor<{batch}x{tokens}x{heads}x{head_dim}xbf16>"
        )?;
    } else {
        writeln!(
            out,
            "    %{prefix} = \"stablehlo.concatenate\"(%{prefix}_lo, %{prefix}_hi) {{dimension = 3 : i64}} : (tensor<{batch}x{tokens}x{heads}x{half}xbf16>, tensor<{batch}x{tokens}x{heads}x{half}xbf16>) -> tensor<{batch}x{tokens}x{heads}x{head_dim}xbf16>"
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_dot_precisions(
        mlir: &str,
        expected_f32_highest: usize,
        expected_bf16_default: usize,
    ) {
        let mut f32_highest = 0usize;
        let mut bf16_default = 0usize;
        for line in mlir.lines() {
            if !line.contains("stablehlo.dot_general") {
                continue;
            }
            if line.contains("xf32>, tensor<") {
                assert!(
                    line.contains("precision = [HIGHEST, HIGHEST]"),
                    "F32 dot_general missing HIGHEST precision: {line}"
                );
                f32_highest += 1;
            } else if line.contains("xbf16>, tensor<") {
                assert!(
                    line.contains("precision = [DEFAULT, DEFAULT]")
                        && line.contains("-> tensor<")
                        && line.trim_end().ends_with("xf32>"),
                    "BF16 dot_general must use DEFAULT precision with f32 result: {line}"
                );
                bf16_default += 1;
            } else {
                panic!("unexpected dot_general operand dtype: {line}");
            }
        }
        assert_eq!(f32_highest, expected_f32_highest, "F32 HIGHEST dot count");
        assert_eq!(
            bf16_default, expected_bf16_default,
            "BF16 DEFAULT dot count"
        );
    }

    #[test]
    fn emits_h3_image_ops_and_dit_block_stablehlo() -> Result<()> {
        let mm = emit_bf16_matmul(4103, 5376, 7168)?;
        assert!(mm.contains("tensor<4103x5376xbf16>"));
        assert!(mm.contains("tensor<7168x5376xbf16>"));
        assert!(mm.contains("contracting_dims = [1] x [1]"));
        assert_dot_precisions(&mm, 0, 1);

        let adaln = emit_bf16_adaln_proj(6, 2688, 64512)?;
        assert!(adaln.contains("tensor<6x2688xbf16>"));
        assert!(adaln.contains("tensor<64512x2688xbf16>"));
        assert!(adaln.contains("-> tensor<6x64512xbf16>"));
        assert_dot_precisions(&adaln, 0, 1);

        let scale = 1.0f32 / 128.0f32.sqrt();
        let attn = emit_bf16_attention(1031, 56, 128, scale)?;
        assert!(attn.contains("tensor<56x1031x1031xf32>"));
        assert!(attn.contains("dense<0x3DB504F3>"));
        assert_dot_precisions(&attn, 0, 2);

        let block_spec = DitBlockHloSpec {
            rows: 4103,
            hidden: 5376,
            heads: 56,
            head_dim: 128,
            ffn: 14336,
            rotary_dim: 96,
            head_chunk: 2,
            adaln_pairs: 12,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
        };
        let block = emit_h3_dit_block(&block_spec)?;
        assert!(block.contains("%x: tensor<4103x5376xbf16>"));
        assert!(block.contains("%adaln: tensor<12x6x5376xbf16>"));
        assert!(block.contains("%row_pairs: tensor<4103xi32>"));
        assert!(block.contains("%rotary: tensor<4103x96xbf16>"));
        assert!(block.contains("tensor<28x2x4103x128xbf16>"));
        assert!(block.contains("tensor<2x4103x4103xf32>"));
        assert!(block.contains("stablehlo.logistic"));
        assert!(block.contains("stablehlo.rsqrt"));
        assert!(block.contains("\"stablehlo.gather\""));
        assert!(block.contains("\"stablehlo.while\""));
        assert_dot_precisions(&block, 0, 8);

        let proj_in = emit_h3_dit_proj_in_pack(7, 512, 96, 5376)?;
        assert_dot_precisions(&proj_in, 1, 0);
        assert!(proj_in.contains("tensor<519x5376xbf16>"));

        let ctx = emit_h3_dit_context_embedder(7, 5120, 5376)?;
        assert_dot_precisions(&ctx, 0, 1);

        let refiner = emit_h3_dit_refiner_block(&DitRefinerBlockHloSpec {
            rows: 7,
            hidden: 5376,
            heads: 56,
            head_dim: 128,
            ffn: 14336,
            head_chunk: 7,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
        })?;
        assert_dot_precisions(&refiner, 0, 8);

        let norm_out = emit_h3_dit_norm_out(&DitNormOutSpec {
            rows: 519,
            text_rows: 7,
            hidden: 5376,
            timesteps: 6,
            final_norm_eps: 1e-5,
        })?;
        assert_dot_precisions(&norm_out, 0, 0);
        assert!(norm_out.contains("-> tensor<512x5376xbf16>"));

        let pdd = emit_h3_dit_pdd_proj_out(512, 5376, 96)?;
        assert_dot_precisions(&pdd, 1, 0);
        assert!(pdd.contains("-> tensor<512x96xf32>"));

        let vae_pre = emit_h3_image_vae_pre(&VaePreHloSpec {
            batch: 15,
            patches: 256,
            latent_channels: 24,
            dim: 1024,
            registers: 4,
        })?;
        assert_dot_precisions(&vae_pre, 0, 2);
        assert!(vae_pre.contains("-> tensor<15x261x1024xf32>"));

        let vae_blk = emit_h3_image_vae_block(&VaeBlockHloSpec {
            batch: 15,
            tokens: 261,
            dim: 1024,
            heads: 16,
            head_dim: 64,
            ffn: 4096,
            rotary_dim: 32,
            norm_eps: 1e-5,
        })?;
        assert_dot_precisions(&vae_blk, 0, 8);

        let vae_post = emit_h3_image_vae_post(&VaePostHloSpec {
            batch: 15,
            tokens: 261,
            num_patches: 256,
            dim: 1024,
            patch_width: 768,
            norm_eps: 1e-5,
        })?;
        assert_dot_precisions(&vae_post, 0, 1);

        let qwen = emit_qwen3_vl_text_layer(&Qwen3VlLayerHloSpec {
            rows: 7,
            hidden: 5120,
            heads: 64,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 25600,
            head_chunk: 8,
            rms_norm_eps: 1e-6,
        })?;
        assert_dot_precisions(&qwen, 0, 9);
        assert!(qwen.contains("\"stablehlo.iota\""));
        assert!(qwen.contains("\"stablehlo.select\""));

        let mut bad = block_spec;
        bad.head_chunk = 3;
        assert!(emit_h3_dit_block(&bad).is_err());
        assert!(emit_bf16_matmul(0, 5376, 7168).is_err());
        assert!(emit_bf16_attention(1031, 56, 128, f32::NAN).is_err());
        Ok(())
    }
}
