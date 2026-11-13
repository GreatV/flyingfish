use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Sampling {
    pub fn new(temperature: f32, top_k: usize, top_p: f32) -> Result<Self> {
        ensure!(
            temperature.is_finite() && temperature > 0.0,
            "invalid sampling temperature {temperature}"
        );
        ensure!(top_k > 0, "sampling top-k is zero");
        ensure!(
            top_p.is_finite() && (0.0..=1.0).contains(&top_p) && top_p > 0.0,
            "invalid sampling top-p {top_p}"
        );
        Ok(Self {
            temperature,
            top_k,
            top_p,
        })
    }

    pub fn probabilities(
        self,
        logits: &[f32],
        reserved: (u32, u32),
        allow: Option<u32>,
    ) -> Result<Vec<f32>> {
        ensure!(
            !logits.is_empty() && logits.len() <= 4096,
            "sampling vocabulary"
        );
        let mut ranked = Vec::with_capacity(logits.len());
        for (id, &logit) in logits.iter().enumerate() {
            let id = u32::try_from(id)?;
            let masked = id >= reserved.0 && id <= reserved.1 && allow != Some(id);
            if masked {
                continue;
            }
            ensure!(logit.is_finite(), "non-finite sampling logit at {id}");
            ranked.push((id, logit / self.temperature));
        }
        ensure!(!ranked.is_empty(), "sampling vocabulary is fully masked");
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(self.top_k.min(ranked.len()));
        let max = ranked[0].1;
        let mut weights: Vec<(u32, f64)> = ranked
            .into_iter()
            .map(|(id, logit)| (id, f64::from(logit - max).exp()))
            .collect();
        let total: f64 = weights.iter().map(|item| item.1).sum();
        ensure!(total.is_finite() && total > 0.0, "sampling normalization");
        let mut cumulative = 0.0;
        let mut keep = 0usize;
        for &(_, weight) in &weights {
            cumulative += weight / total;
            keep += 1;
            if cumulative > f64::from(self.top_p) {
                break;
            }
        }
        weights.truncate(keep);
        let retained: f64 = weights.iter().map(|item| item.1).sum();
        ensure!(
            retained.is_finite() && retained > 0.0,
            "sampling top-p mass"
        );
        let mut probabilities = vec![0.0f32; logits.len()];
        for (id, weight) in weights {
            probabilities[id as usize] = (weight / retained) as f32;
        }
        Ok(probabilities)
    }
}

pub fn philox_words(seed: u64, stream: u64, counter: u64) -> [u32; 4] {
    let mut key = [seed as u32, (seed >> 32) as u32];
    let mut state = [
        stream as u32,
        (stream >> 32) as u32,
        counter as u32,
        (counter >> 32) as u32,
    ];
    for round in 0..10 {
        if round > 0 {
            key[0] = key[0].wrapping_add(0x9E37_79B9);
            key[1] = key[1].wrapping_add(0xBB67_AE85);
        }
        let mul = |a: u32, b: u32| {
            let product = u64::from(a) * u64::from(b);
            ((product >> 32) as u32, product as u32)
        };
        let (hi0, lo0) = mul(0xD251_1F53, state[0]);
        let (hi1, lo1) = mul(0xCD9E_8D57, state[2]);
        state = [hi1 ^ state[1] ^ key[0], lo1, hi0 ^ state[3] ^ key[1], lo0];
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_masks_scales_and_seeds_are_deterministic() -> Result<()> {
        let sampling = Sampling::new(2.0, 3, 0.8)?;
        let logits = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let probabilities = sampling.probabilities(&logits, (6, 6), None)?;
        assert!(probabilities[6] == 0.0 && probabilities[4] == 0.0);
        assert!(probabilities[7] > 0.0 && probabilities[5] > 0.0);
        assert!(probabilities[..5].iter().all(|value| *value == 0.0));
        let sum: f32 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        let ratio = probabilities[7] / probabilities[5];
        assert!((ratio - std::f32::consts::E).abs() < 1e-5, "{ratio}");
        assert_eq!(
            probabilities,
            sampling.probabilities(&logits, (6, 6), None)?
        );
        let a = philox_words(7, 1, 3);
        assert_eq!(a, philox_words(7, 1, 3));
        assert_ne!(a, philox_words(8, 1, 3));
        assert_ne!(a, philox_words(7, 0, 3));
        assert!(Sampling::new(0.0, 50, 1.0).is_err());
        assert!(Sampling::new(0.9, 0, 1.0).is_err());
        assert!(Sampling::new(0.9, 50, 0.0).is_err());
        Ok(())
    }
}
