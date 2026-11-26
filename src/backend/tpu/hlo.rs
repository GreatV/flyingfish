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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_h3_image_ops_and_dit_block_stablehlo() -> Result<()> {
        let mm = emit_bf16_matmul(4103, 5376, 7168)?;
        assert!(mm.contains("tensor<4103x5376xbf16>"));
        assert!(mm.contains("tensor<7168x5376xbf16>"));
        assert!(mm.contains("contracting_dims = [1] x [1]"));

        let adaln = emit_bf16_adaln_proj(6, 2688, 64512)?;
        assert!(adaln.contains("tensor<6x2688xbf16>"));
        assert!(adaln.contains("tensor<64512x2688xbf16>"));
        assert!(adaln.contains("-> tensor<6x64512xbf16>"));

        let scale = 1.0f32 / 128.0f32.sqrt();
        let attn = emit_bf16_attention(1031, 56, 128, scale)?;
        assert!(attn.contains("tensor<56x1031x1031xf32>"));
        assert!(attn.contains("dense<0x3DB504F3>"));

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

        let mut bad = block_spec;
        bad.head_chunk = 3;
        assert!(emit_h3_dit_block(&bad).is_err());
        assert!(emit_bf16_matmul(0, 5376, 7168).is_err());
        assert!(emit_bf16_attention(1031, 56, 128, f32::NAN).is_err());
        Ok(())
    }
}
