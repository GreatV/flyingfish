//! Pure-Rust StableHLO MLIR text emitter for TPU execution via PJRT.

use anyhow::{Result, ensure};

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
    let scale_hex = format!("0x{:08X}", scale.to_bits());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_h3_image_to_q_matmul_and_attention_stablehlo() -> Result<()> {
        let mm = emit_bf16_matmul(4103, 5376, 7168)?;
        assert!(mm.contains("tensor<4103x5376xbf16>"));
        assert!(mm.contains("tensor<7168x5376xbf16>"));
        assert!(mm.contains("contracting_dims = [1] x [1]"));
        assert!(mm.contains("-> tensor<4103x7168xf32>"));
        assert!(mm.contains("-> tensor<4103x7168xbf16>"));

        let scale = 1.0f32 / 128.0f32.sqrt();
        let attn = emit_bf16_attention(1031, 56, 128, scale)?;
        assert!(attn.contains("tensor<1031x56x128xbf16>"));
        assert!(attn.contains("tensor<56x1031x1031xf32>"));
        assert!(attn.contains("dense<0x3DB504F3>"));
        assert!(attn.contains("dense<0xFF800000>"));
        assert!(attn.contains("dims = [1, 0, 2]"));
        assert!(emit_bf16_matmul(0, 5376, 7168).is_err());
        assert!(emit_bf16_attention(1031, 56, 128, f32::NAN).is_err());
        Ok(())
    }
}
