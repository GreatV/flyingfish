//! Diagnostic statistics for calibration outputs. Equal summaries do not imply
//! equal tensors; output verification belongs to direct output comparisons.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

/// The leading and trailing values kept verbatim, so a difference confined to
/// one end of a latent is visible rather than only summarised.
const RETAINED_EDGE_VALUES: usize = 8;
const SUMMARY_CHUNK_ELEMENTS: usize = 64 * 1024;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSummary {
    pub dims: Vec<usize>,
    pub elements: u64,
    /// Bit patterns of the statistics, not a representation of all tensor values.
    pub sum_bits: u64,
    pub sum_of_squares_bits: u64,
    pub minimum_bits: u32,
    pub maximum_bits: u32,
    pub leading_bits: Vec<u32>,
    pub trailing_bits: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct T2vaLatentSummary {
    pub video: TensorSummary,
    pub audio: TensorSummary,
}

impl T2vaLatentSummary {
    pub fn collect(video: &Tensor, audio: &Tensor) -> Result<Self> {
        Ok(Self {
            video: TensorSummary::collect("video", video)?,
            audio: TensorSummary::collect("audio", audio)?,
        })
    }

    /// The first modality with different statistics. `None` says nothing
    /// about unsampled values or their positions within the tensor.
    pub fn first_statistics_difference(&self, other: &Self) -> Option<&'static str> {
        self.video
            .first_statistics_difference(&other.video, "video")
            .or_else(|| {
                self.audio
                    .first_statistics_difference(&other.audio, "audio")
            })
    }

    pub fn validate(&self) -> Result<()> {
        self.video.validate("video")?;
        self.audio.validate("audio")
    }
}

impl TensorSummary {
    fn collect(name: &str, tensor: &Tensor) -> Result<Self> {
        let dims = tensor.dims().to_vec();
        let mut elements = 1u64;
        for &dimension in &dims {
            let dimension = u64::try_from(dimension).context("latent dimension exceeds u64")?;
            elements = elements
                .checked_mul(dimension)
                .context("latent element count overflow")?;
        }
        let expected = usize::try_from(elements).context("latent element count exceeds usize")?;
        anyhow::ensure!(
            expected > 0,
            "the {name} latent summary needs at least one element"
        );
        let flat = tensor.flatten_all()?;
        let edge = RETAINED_EDGE_VALUES.min(expected);
        let mut leading_bits = Vec::with_capacity(edge);
        let mut trailing_bits = Vec::with_capacity(2 * edge);
        let mut sum = 0f64;
        let mut sum_of_squares = 0f64;
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        for start in (0..expected).step_by(SUMMARY_CHUNK_ELEMENTS) {
            let count = SUMMARY_CHUNK_ELEMENTS.min(expected - start);
            let values = flat
                .narrow(0, start, count)?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            for &value in &values {
                anyhow::ensure!(
                    value.is_finite(),
                    "cannot summarise a non-finite {name} latent"
                );
                sum += f64::from(value);
                sum_of_squares += f64::from(value) * f64::from(value);
                minimum = minimum.min(value);
                maximum = maximum.max(value);
            }
            if start == 0 {
                leading_bits.extend(values.iter().take(edge).map(|v| v.to_bits()));
            }
            trailing_bits.extend(
                values[values.len().saturating_sub(edge)..]
                    .iter()
                    .map(|v| v.to_bits()),
            );
            if trailing_bits.len() > edge {
                trailing_bits.drain(..trailing_bits.len() - edge);
            }
        }
        Ok(Self {
            dims,
            elements,
            sum_bits: sum.to_bits(),
            sum_of_squares_bits: sum_of_squares.to_bits(),
            minimum_bits: minimum.to_bits(),
            maximum_bits: maximum.to_bits(),
            leading_bits,
            trailing_bits,
        })
    }

    fn first_statistics_difference(
        &self,
        other: &Self,
        name: &'static str,
    ) -> Option<&'static str> {
        if self.dims != other.dims || self.elements != other.elements {
            return Some(name);
        }
        if self.sum_bits != other.sum_bits
            || self.sum_of_squares_bits != other.sum_of_squares_bits
            || self.minimum_bits != other.minimum_bits
            || self.maximum_bits != other.maximum_bits
            || self.leading_bits != other.leading_bits
            || self.trailing_bits != other.trailing_bits
        {
            return Some(name);
        }
        None
    }

    fn validate(&self, name: &str) -> Result<()> {
        anyhow::ensure!(
            !self.dims.is_empty() && self.elements > 0,
            "the {name} latent summary records no elements"
        );
        let edge = RETAINED_EDGE_VALUES
            .min(usize::try_from(self.elements).context("latent element count exceeds usize")?);
        anyhow::ensure!(
            self.leading_bits.len() == edge && self.trailing_bits.len() == edge,
            "the {name} latent summary retains the wrong number of edge values"
        );
        anyhow::ensure!(
            f64::from_bits(self.sum_bits).is_finite()
                && f64::from_bits(self.sum_of_squares_bits).is_finite()
                && f32::from_bits(self.minimum_bits).is_finite()
                && f32::from_bits(self.maximum_bits).is_finite(),
            "the {name} latent summary records a non-finite statistic"
        );
        Ok(())
    }
}
