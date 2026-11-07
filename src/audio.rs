//! Shared radix-2 FFT and log-mel feature extraction.
use anyhow::{Context, Result, ensure};

pub struct Radix2Fft {
    size: usize,
    reverse: Vec<usize>,
    twiddles: Vec<[f64; 2]>,
}

impl Radix2Fft {
    pub fn new(size: usize) -> Result<Self> {
        ensure!(
            size >= 2 && size.is_power_of_two(),
            "FFT size must be a power of two of at least two"
        );
        let bits = size.trailing_zeros();
        let reverse = (0..size)
            .map(|index| index.reverse_bits() >> (usize::BITS - bits))
            .collect();
        let twiddles = (0..size / 2)
            .map(|index| {
                let angle = -2.0 * std::f64::consts::PI * index as f64 / size as f64;
                [angle.cos(), angle.sin()]
            })
            .collect();
        Ok(Self {
            size,
            reverse,
            twiddles,
        })
    }

    /// Input is `[size]` real samples; output is interleaved `[size,2]` complex values.
    pub fn run(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        ensure!(
            input.len() == self.size
                && output.len() == 2 * self.size
                && input.iter().all(|value| value.is_finite()),
            "FFT input/output shape or finite-value mismatch"
        );
        let mut spectrum = vec![[0.0f64; 2]; self.size];
        for (index, &source) in self.reverse.iter().enumerate() {
            spectrum[index] = [f64::from(input[source]), 0.0];
        }
        let mut span = 2;
        while span <= self.size {
            let half = span / 2;
            let stride = self.size / span;
            for base in (0..self.size).step_by(span) {
                for offset in 0..half {
                    let even = base + offset;
                    let odd = even + half;
                    let [wr, wi] = self.twiddles[offset * stride];
                    let [xr, xi] = spectrum[odd];
                    let vr = wr.mul_add(xr, -wi * xi);
                    let vi = wr.mul_add(xi, wi * xr);
                    let [ur, ui] = spectrum[even];
                    spectrum[even] = [ur + vr, ui + vi];
                    spectrum[odd] = [ur - vr, ui - vi];
                }
            }
            span *= 2;
        }
        for (index, value) in spectrum.iter().enumerate() {
            output[2 * index] = value[0] as f32;
            output[2 * index + 1] = value[1] as f32;
        }
        ensure!(
            output.iter().all(|value| value.is_finite()),
            "FFT produced a non-finite value"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LogMelConfig {
    pub sampling_rate: u32,
    pub n_fft: u32,
    pub win_length: u32,
    pub hop_length: u32,
    pub mel_bins: u32,
    pub preemphasis: f32,
    pub log_guard: f32,
    pub norm_eps: f32,
}

pub struct LogMel {
    config: LogMelConfig,
    fft: Radix2Fft,
    window: Vec<f32>,
    filters: Vec<f32>,
}

pub struct LogMelBatch {
    pub frames: usize,
    pub mel_bins: usize,
    /// `[batch, frames, mel_bins]` F32 values.
    pub features: Vec<f32>,
    /// `[batch, frames]` valid-frame flags.
    pub mask: Vec<bool>,
}

impl LogMel {
    pub fn new(config: LogMelConfig) -> Result<Self> {
        let n_fft = config.n_fft as usize;
        let win_length = config.win_length as usize;
        ensure!(
            config.sampling_rate > 0
                && config.hop_length > 0
                && config.mel_bins > 0
                && win_length > 1
                && win_length <= n_fft
                && config.preemphasis.is_finite()
                && config.log_guard.is_finite()
                && config.log_guard > 0.0
                && config.norm_eps.is_finite()
                && config.norm_eps > 0.0,
            "invalid log-mel geometry or constants"
        );
        let fft = Radix2Fft::new(n_fft)?;
        let mut window = vec![0.0; n_fft];
        let first = (n_fft - win_length) / 2;
        let step = ((std::f64::consts::PI * 2.0) / (win_length - 1) as f64) as f32;
        for index in 0..win_length {
            let angle = index as f32 * step;
            window[first + index] = -0.5 * angle.cos() + 0.5;
        }
        let bins = n_fft / 2 + 1;
        let mel_bins = config.mel_bins as usize;
        let hz_to_mel = |hz: f64| {
            if hz < 1000.0 {
                hz / (200.0 / 3.0)
            } else {
                15.0 + (hz / 1000.0).ln() / (6.4f64.ln() / 27.0)
            }
        };
        let low = hz_to_mel(0.0);
        let high = hz_to_mel(f64::from(config.sampling_rate) / 2.0);
        let mel_to_hz = |mel: f64| {
            if mel < 15.0 {
                mel * (200.0 / 3.0)
            } else {
                1000.0 * ((mel - 15.0) * (6.4f64.ln() / 27.0)).exp()
            }
        };
        let mel: Vec<f64> = (0..mel_bins + 2)
            .map(|index| mel_to_hz(low + (high - low) * index as f64 / (mel_bins + 1) as f64))
            .collect();
        let mut filters = vec![0.0; mel_bins * bins];
        for band in 0..mel_bins {
            let lower_width = mel[band + 1] - mel[band];
            let upper_width = mel[band + 2] - mel[band + 1];
            let normalization = (2.0 / (mel[band + 2] - mel[band])) as f32;
            for bin in 0..bins {
                let hz = bin as f64 * f64::from(config.sampling_rate) / n_fft as f64;
                let lower = (hz - mel[band]) / lower_width;
                let upper = (mel[band + 2] - hz) / upper_width;
                let triangle = lower.min(upper).max(0.0) as f32;
                filters[band * bins + bin] = triangle * normalization;
            }
        }
        Ok(Self {
            config,
            fft,
            window,
            filters,
        })
    }

    pub fn extract(
        &self,
        waveforms: &[&[f32]],
        normalize: bool,
        center: bool,
        last: bool,
    ) -> Result<LogMelBatch> {
        ensure!(!last || !center, "last-chunk pad requires center false");
        if !center {
            return self.extract_edge(waveforms, normalize, last);
        }
        ensure!(!waveforms.is_empty(), "empty audio batch");
        for waveform in waveforms {
            ensure!(
                waveform.len() >= 2 * self.config.hop_length as usize
                    && waveform.iter().all(|value| value.is_finite()),
                "audio needs at least two valid frames and finite samples"
            );
        }
        let longest = waveforms
            .iter()
            .map(|waveform| waveform.len())
            .max()
            .context("audio batch has no maximum length")?;
        let frames = longest / self.config.hop_length as usize + 1;
        let mel_bins = self.config.mel_bins as usize;
        let bins = self.config.n_fft as usize / 2 + 1;
        let feature_count = waveforms
            .len()
            .checked_mul(frames)
            .and_then(|count| count.checked_mul(mel_bins))
            .context("log-mel output size overflow")?;
        let mut features = vec![0.0; feature_count];
        let mut mask = vec![false; waveforms.len() * frames];
        let mut frame = vec![0.0; self.config.n_fft as usize];
        let mut spectrum = vec![0.0; self.config.n_fft as usize * 2];
        let mut power = vec![0.0; bins];
        for (batch, waveform) in waveforms.iter().enumerate() {
            let valid = waveform.len() / self.config.hop_length as usize;
            for time in 0..=valid {
                for (index, value) in frame.iter_mut().enumerate() {
                    let center = time * self.config.hop_length as usize + index;
                    let sample = center.checked_sub(self.config.n_fft as usize / 2);
                    let emphasized = match sample {
                        Some(position) if position < waveform.len() => {
                            if position == 0 {
                                waveform[0]
                            } else {
                                waveform[position]
                                    - self.config.preemphasis * waveform[position - 1]
                            }
                        }
                        _ => 0.0,
                    };
                    *value = emphasized * self.window[index];
                }
                self.fft.run(&frame, &mut spectrum)?;
                for (bin, power) in power.iter_mut().enumerate() {
                    let re = spectrum[2 * bin];
                    let im = spectrum[2 * bin + 1];
                    let magnitude = (re * re + im * im).sqrt();
                    *power = magnitude * magnitude;
                }
                for mel in 0..mel_bins {
                    let mut value = 0.0f32;
                    for (&weight, &power) in self.filters[mel * bins..(mel + 1) * bins]
                        .iter()
                        .zip(power.iter())
                    {
                        value = weight.mul_add(power, value);
                    }
                    features[(batch * frames + time) * mel_bins + mel] =
                        (value + self.config.log_guard).ln();
                }
            }
            for mel in 0..mel_bins {
                if normalize {
                    let mut mean = 0.0;
                    for time in 0..valid {
                        mean += features[(batch * frames + time) * mel_bins + mel];
                    }
                    mean /= valid as f32;
                    let mut variance = 0.0;
                    for time in 0..valid {
                        let delta = features[(batch * frames + time) * mel_bins + mel] - mean;
                        variance = delta.mul_add(delta, variance);
                    }
                    let std = (variance / (valid - 1) as f32).sqrt();
                    for time in 0..valid {
                        let index = (batch * frames + time) * mel_bins + mel;
                        features[index] = (features[index] - mean) / (std + self.config.norm_eps);
                    }
                }
                for time in valid..frames {
                    features[(batch * frames + time) * mel_bins + mel] = 0.0;
                }
            }
            for time in 0..valid {
                mask[batch * frames + time] = true;
            }
        }
        ensure!(
            features.iter().all(|value| value.is_finite()),
            "log-mel produced a non-finite value"
        );
        Ok(LogMelBatch {
            frames,
            mel_bins,
            features,
            mask,
        })
    }

    fn extract_edge(
        &self,
        waveforms: &[&[f32]],
        normalize: bool,
        last: bool,
    ) -> Result<LogMelBatch> {
        ensure!(!waveforms.is_empty(), "empty audio batch");
        let n_fft = self.config.n_fft as usize;
        let hop = self.config.hop_length as usize;
        ensure!(hop < n_fft / 2, "last-chunk pad is empty");
        let mut prepared = Vec::with_capacity(waveforms.len());
        for waveform in waveforms {
            ensure!(
                waveform.len() >= n_fft && waveform.iter().all(|value| value.is_finite()),
                "center-false waveform length {} is shorter than n_fft {n_fft}",
                waveform.len()
            );
            let mut samples = Vec::with_capacity(waveform.len() + n_fft / 2);
            samples.push(waveform[0]);
            for index in 1..waveform.len() {
                samples.push(waveform[index] - self.config.preemphasis * waveform[index - 1]);
            }
            if last {
                samples.resize(samples.len() + n_fft / 2 - hop, 0.0);
            }
            let valid = (samples.len() - n_fft) / hop + 1;
            ensure!(valid > 0, "center-false waveform produced no frames");
            prepared.push((samples, valid));
        }
        let frames = prepared
            .iter()
            .map(|(_, valid)| *valid)
            .max()
            .context("audio batch has no maximum length")?;
        let mel_bins = self.config.mel_bins as usize;
        let bins = n_fft / 2 + 1;
        let feature_count = waveforms
            .len()
            .checked_mul(frames)
            .and_then(|count| count.checked_mul(mel_bins))
            .context("log-mel output size overflow")?;
        let mut features = vec![0.0; feature_count];
        let mut mask = vec![false; waveforms.len() * frames];
        let mut frame = vec![0.0; n_fft];
        let mut spectrum = vec![0.0; n_fft * 2];
        let mut power = vec![0.0; bins];
        for (batch, (samples, valid)) in prepared.iter().enumerate() {
            for time in 0..*valid {
                for (index, value) in frame.iter_mut().enumerate() {
                    let sample = time * hop + index;
                    let emphasized = samples.get(sample).copied().unwrap_or(0.0);
                    *value = emphasized * self.window[index];
                }
                self.fft.run(&frame, &mut spectrum)?;
                for (bin, power) in power.iter_mut().enumerate() {
                    let re = spectrum[2 * bin];
                    let im = spectrum[2 * bin + 1];
                    let magnitude = (re * re + im * im).sqrt();
                    *power = magnitude * magnitude;
                }
                for mel in 0..mel_bins {
                    let mut value = 0.0f32;
                    for (&weight, &power) in self.filters[mel * bins..(mel + 1) * bins]
                        .iter()
                        .zip(power.iter())
                    {
                        value = weight.mul_add(power, value);
                    }
                    features[(batch * frames + time) * mel_bins + mel] =
                        (value + self.config.log_guard).ln();
                }
            }
            for mel in 0..mel_bins {
                if normalize {
                    ensure!(
                        *valid >= 2,
                        "normalized center-false audio needs two frames"
                    );
                    let mut mean = 0.0;
                    for time in 0..*valid {
                        mean += features[(batch * frames + time) * mel_bins + mel];
                    }
                    mean /= *valid as f32;
                    let mut variance = 0.0;
                    for time in 0..*valid {
                        let delta = features[(batch * frames + time) * mel_bins + mel] - mean;
                        variance = delta.mul_add(delta, variance);
                    }
                    let std = (variance / (*valid - 1) as f32).sqrt();
                    for time in 0..*valid {
                        let index = (batch * frames + time) * mel_bins + mel;
                        features[index] = (features[index] - mean) / (std + self.config.norm_eps);
                    }
                }
                for time in *valid..frames {
                    features[(batch * frames + time) * mel_bins + mel] = 0.0;
                }
            }
            for time in 0..*valid {
                mask[batch * frames + time] = true;
            }
        }
        ensure!(
            features.iter().all(|value| value.is_finite()),
            "log-mel produced a non-finite value"
        );
        Ok(LogMelBatch {
            frames,
            mel_bins,
            features,
            mask,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix2_fft_matches_direct_transform() -> Result<()> {
        let fft = Radix2Fft::new(16)?;
        let input: Vec<f32> = (0..16)
            .map(|i| ((i * 37 + 11) % 101) as f32 / 64.0)
            .collect();
        let mut output = vec![0.0; 32];
        fft.run(&input, &mut output)?;
        for k in 0..16 {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (n, &value) in input.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * (k * n) as f64 / 16.0;
                re += f64::from(value) * angle.cos();
                im += f64::from(value) * angle.sin();
            }
            ensure!(
                (f64::from(output[2 * k]) - re).abs() < 2e-5
                    && (f64::from(output[2 * k + 1]) - im).abs() < 2e-5,
                "FFT bin {k} differs from direct transform"
            );
        }
        ensure!(
            Radix2Fft::new(1).is_err() && Radix2Fft::new(15).is_err(),
            "invalid FFT size accepted"
        );
        Ok(())
    }

    #[test]
    fn unnormalized_log_mel_keeps_the_frame_mean() -> Result<()> {
        let mel = LogMel::new(LogMelConfig {
            sampling_rate: 16000,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            mel_bins: 128,
            preemphasis: 0.97,
            log_guard: 2.0f32.powi(-24),
            norm_eps: 1e-5,
        })?;
        let wave: Vec<f32> = (0..2560)
            .map(|index| {
                let centered = ((index * 73 + 19) % 257) as f32;
                let integer = (centered - 128.0) / 128.0;
                let tone = (2.0 * std::f64::consts::PI * index as f64 / 251.0).sin() as f32 * 0.25;
                integer + tone
            })
            .collect();
        let normalized = mel.extract(&[&wave], true, true, false)?;
        let logged = mel.extract(&[&wave], false, true, false)?;
        ensure!(
            normalized.frames == 17 && logged.frames == 17,
            "frame count"
        );
        ensure!(
            logged.mask.iter().filter(|flag| **flag).count() == 16,
            "valid frame count"
        );
        let mut logged_mean = 0.0f32;
        let mut normalized_mean = 0.0f32;
        for time in 0..16 {
            logged_mean += logged.features[time * 128];
            normalized_mean += normalized.features[time * 128];
            ensure!(
                logged.features[time * 128].is_finite(),
                "non-finite log mel"
            );
        }
        logged_mean /= 16.0;
        normalized_mean /= 16.0;
        let edge = mel.extract(&[&vec![0.1; 5520]], false, false, false)?;
        let tail = mel.extract(&[&vec![0.1; 2000]], false, false, true)?;
        let first = mel.extract(&[&vec![0.1; 4040]], false, true, false)?;
        ensure!(
            edge.frames == 32
                && edge.mask.iter().all(|value| *value)
                && tail.frames == 10
                && tail.mask.iter().all(|value| *value)
                && first.frames == 26
                && first.mask.iter().filter(|value| **value).count() == 25
                && mel
                    .extract(&[&vec![0.1; 100]], false, false, false)
                    .is_err()
                && mel.extract(&[&vec![0.1; 4040]], false, true, true).is_err(),
            "streaming log-mel frame counts changed"
        );
        ensure!(
            logged_mean.abs() > 1e-3 && normalized_mean.abs() < 1e-5,
            "normalize flag did not change the frame mean: log={logged_mean} normalized={normalized_mean}"
        );
        ensure!(
            logged.features[16 * 128..]
                .iter()
                .all(|value| *value == 0.0),
            "masked log-mel frame was not zeroed"
        );
        Ok(())
    }
}
