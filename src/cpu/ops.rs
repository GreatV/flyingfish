//! Definition-pinned CPU F32 oracles for the Music3 synthesis operators.
//! Literals in the tests come from the pinned model definition (diffusers
//! dafe3733) run on CPU in F32; each pin carries the discriminating
//! wrong-form control that must fail on it.

/// Vocoder activation: y = x + (alpha + 1e-9)^-1 * sin(alpha * x)^2 with a
/// linear per-channel alpha (MiniMaxMusic3Snake1d).
pub fn snake_f32(x: f32, alpha: f32) -> f32 {
    let s = (alpha * x).sin();
    x + (alpha + 1e-9).recip() * s * s
}

/// Fourier time features: angle = (2*pi*t) * w, out = (cos(angle), sin(angle))
/// with the checkpoint weight w (MiniMaxMusic3FourierEmbedding; the angle is
/// a k=1 product, cos half first).
pub fn fourier_time_features(t: f32, weight: f32) -> (f32, f32) {
    let angle = (2.0 * std::f32::consts::PI * t) * weight;
    (angle.cos(), angle.sin())
}

/// inv_freq[i] = theta^-(2i/rotary_dim) (MiniMaxMusic3RotaryEmbedding).
pub fn rotary_inv_freq(theta: f32, rotary_dim: usize, index: usize) -> f32 {
    1.0 / theta.powf((2 * index) as f32 / rotary_dim as f32)
}

/// Split-half partial rotary apply: the leading cos.len() dims rotate as
/// rotated * cos + cat(-h2, h1) * sin, the tail passes through
/// (_apply_partial_rotary_emb).
pub fn partial_rotary_apply_f32(head: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
    let rotary_dim = cos.len();
    let half = rotary_dim / 2;
    let mut out = vec![0.0f32; head.len()];
    for d in 0..rotary_dim {
        let rotate_half = if d < half {
            -head[half + d]
        } else {
            head[d - half]
        };
        out[d] = head[d] * cos[d] + rotate_half * sin[d];
    }
    out[rotary_dim..].copy_from_slice(&head[rotary_dim..]);
    out
}

/// Centered-variance LayerNorm with affine (nn.LayerNorm; the same math as
/// the q38_layer_norm CUDA family in F32).
pub fn layer_norm_f32(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let variance = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = (variance + eps).sqrt().recip();
    x.iter()
        .zip(weight)
        .zip(bias)
        .map(|((v, w), b)| (v - mean) * inv * w + b)
        .collect()
}

/// Nearest-neighbor resample index map: out[j] = in[floor(j * in_len /
/// out_len)] (F.interpolate mode="nearest", verified against the definition
/// environment).
pub fn nearest_interpolate_map(in_len: usize, out_len: usize) -> Vec<usize> {
    (0..out_len).map(|j| j * in_len / out_len).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn snake_pin_matches_definition_f32() {
        // torch F32, definition form, per-channel alpha {0.5, 1.0, 2.0}.
        let cases: &[(f32, f32, f32)] = &[
            (-1.2, 0.5, -0.5623578),
            (-1.2, 1.0, -0.33130318),
            (-1.2, 2.0, -0.97187483),
            (-0.3, 0.5, -0.2553365),
            (-0.3, 1.0, -0.21266782),
            (-0.3, 2.0, -0.14058945),
            (0.0, 0.5, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.7, 0.5, 0.9351578),
            (0.7, 1.0, 1.1150165),
            (0.7, 2.0, 1.1855556),
            (2.1, 0.5, 3.604846),
            (2.1, 1.0, 2.8451304),
            (2.1, 2.0, 2.479822),
            (10.0, 0.5, 11.839071),
            (10.0, 1.0, 10.295959),
            (10.0, 2.0, 10.416735),
        ];
        for &(x, alpha, expected) in cases {
            let got = snake_f32(x, alpha);
            assert!(
                (got - expected).abs() <= 5e-6,
                "snake({x}, {alpha}): {got} vs {expected}"
            );
        }
        // Wrong form (log-scale alpha applied to the same parameter value)
        // differs by orders of magnitude.
        let wrong = 2.1 + (0.5f32.exp() + 1e-9).recip() * (0.5f32.exp() * 2.1).sin().powi(2);
        assert!((wrong - 3.604846).abs() > 1e-3);
        // Wrong form (epsilon outside the reciprocal): at alpha ~ epsilon the
        // two disagree by sin^2 * eps / (alpha * (alpha + eps)) in the added
        // term, which is large only when sin^2(alpha * x) is; x = 1e5 makes
        // the term difference order-10.
        let small = 2e-9f32;
        let right = snake_f32(1.0e5, small);
        let outside = 1.0e5 + (small * 1.0e5).sin().powi(2) / small + 1e-9;
        assert!((right - outside).abs() > 5.0, "{right} vs {outside}");
    }

    #[test]
    fn fourier_time_pin_matches_definition_f32() {
        let weights = [0.3f32, -1.7, 2.4, 0.05];
        let expected: &[(f32, [f32; 8])] = &[
            (0.0, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            (
                0.033333335,
                [
                    0.9980267,
                    0.93728197,
                    0.87630665,
                    0.99994516,
                    0.06279053,
                    -0.3485721,
                    0.48175377,
                    0.0104717845,
                ],
            ),
            (
                0.5,
                [
                    0.5877852, 0.5877855, 0.30901635, 0.98768836, 0.80901706, 0.8090168, 0.9510567,
                    0.15643448,
                ],
            ),
            (
                1.0,
                [
                    -0.30901715,
                    -0.3090165,
                    -0.8090178,
                    0.95105654,
                    0.9510565,
                    0.95105666,
                    0.5877842,
                    0.309017,
                ],
            ),
        ];
        for &(t, exp) in expected {
            // Definition layout: cat(cos(w) for all w, sin(w) for all w).
            let mut got = Vec::with_capacity(8);
            got.extend(weights.iter().map(|w| fourier_time_features(t, *w).0));
            got.extend(weights.iter().map(|w| fourier_time_features(t, *w).1));
            assert!(max_abs(&got, &exp) <= 1e-6, "t={t}: {got:?} vs {exp:?}");
        }
        // Wrong form (sin half first) is a permutation the definition never
        // produces at non-trivial angles.
        let (c, s) = fourier_time_features(0.5, 0.3);
        assert!((c - 0.5877852).abs() <= 1e-6 && (s - 0.80901706).abs() <= 1e-6);
        assert!((c - s).abs() > 1e-2);
    }

    #[test]
    fn partial_rotary_pin_matches_definition_f32() {
        // theta=10000, rotary_dim=8 of head_dim=16, positions 0..=4.
        let inv: Vec<f32> = (0..4).map(|i| rotary_inv_freq(10000.0, 8, i)).collect();
        assert!(max_abs(&inv, &[1.0, 0.1, 0.01, 1.0e-03]) <= 1e-7, "{inv:?}");
        let pos = 2.0f32;
        let freqs: Vec<f32> = inv.iter().map(|f| pos * f).collect();
        let cos: Vec<f32> = freqs.iter().chain(freqs.iter()).map(|a| a.cos()).collect();
        let sin: Vec<f32> = freqs.iter().chain(freqs.iter()).map(|a| a.sin()).collect();
        assert!(
            max_abs(
                &cos,
                &[
                    -0.41614684,
                    0.9800666,
                    0.9998,
                    0.99999803,
                    -0.41614684,
                    0.9800666,
                    0.9998,
                    0.99999803
                ]
            ) <= 1e-6
        );
        assert!(
            max_abs(
                &sin,
                &[
                    0.9092974,
                    0.19866933,
                    0.019998666,
                    0.0019999987,
                    0.9092974,
                    0.19866933,
                    0.019998666,
                    0.0019999987
                ]
            ) <= 1e-6
        );
        let head: Vec<f32> = [0.4, -1.3, 2.2, 0.1, -0.6, 1.5, -2.0, 0.8]
            .iter()
            .chain([0.3f32; 8].iter())
            .copied()
            .collect();
        let out = partial_rotary_apply_f32(&head, &cos, &sin);
        assert!(
            max_abs(
                &out,
                &[
                    0.37911975, -1.5720905, 2.2395575, 0.09839981, 0.6134071, 1.2118298, -1.955603,
                    0.80019844, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3
                ]
            ) <= 1e-6,
            "{out:?}"
        );
        // Wrong pairing (interleaved i / i + half on the rotated segment)
        // differs from the definition's split-half form.
        let mut wrong = head.clone();
        for i in 0..4 {
            let (a, b) = (head[i], head[i + 4]);
            wrong[i] = a * cos[i] + b * sin[i];
            wrong[i + 4] = -b * cos[i + 4] + a * sin[i + 4];
        }
        assert!(max_abs(&wrong, &out) > 1e-2);
    }

    #[test]
    fn layer_norm_pin_matches_definition_f32() {
        let x = [0.2, -1.1, 0.7, 2.3, -0.4, 0.9, -2.1, 1.4];
        let w = [1.1, 0.8, 1.0, 1.3, 0.9, 1.05, 0.95, 1.2];
        let b = [0.01, -0.02, 0.03, 0.0, 0.02, -0.01, 0.0, 0.04];
        let y = layer_norm_f32(&x, &w, &b, 1e-5);
        assert!(
            max_abs(
                &y,
                &[
                    -0.021272829,
                    -0.831199,
                    0.3806351,
                    2.0327356,
                    -0.414977,
                    0.51737416,
                    -1.683522,
                    1.0975913
                ]
            ) <= 1e-6,
            "{y:?}"
        );
        // Wrong form (RMS, no centering) and unbiased variance both differ.
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let _ = mean;
        let mut rms = y.clone();
        let n = x.len() as f32;
        let ms = x.iter().map(|v| v * v).sum::<f32>() / n;
        let inv = (ms + 1e-5).sqrt().recip();
        for i in 0..8 {
            rms[i] = x[i] * inv * w[i] + b[i];
        }
        assert!(max_abs(&rms, &y) > 1e-2);
    }

    #[test]
    fn nearest_interpolate_pin_matches_definition() {
        assert_eq!(nearest_interpolate_map(5, 8), vec![0, 0, 1, 1, 2, 3, 3, 4],);
        let map = nearest_interpolate_map(26, 86);
        assert_eq!(&map[..14], &[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]);
        // Wrong form (floor with half-pixel offset) differs at j=3.
        let half: Vec<usize> = (0..8)
            .map(|j| ((j as f32 + 0.5) * 5.0 / 8.0) as usize)
            .collect();
        assert_ne!(half, nearest_interpolate_map(5, 8));
    }
}
