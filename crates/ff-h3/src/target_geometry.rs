//! The official MiniMax-H3 request geometry.
//!
//! MiniMax's own API takes a target as `short_edge` + `aspect_ratio` +
//! `duration_seconds`:
//!
//! ```json
//! "target": { "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 10 }
//! ```
//!
//! Everything downstream of the request is expressed in latents and aligned
//! frame counts instead. This module is the single conversion between the two,
//! so the released canvases and durations are reproduced exactly rather than
//! restated by hand at each call site.

use crate::fl2va::{
    MINIMAX_H3_FPS, MINIMAX_H3_MAX_SECONDS, MINIMAX_H3_MIN_SECONDS, MINIMAX_H3_PATCH_SIZE,
    resolve_fl2va_frame_geometry,
};
use anyhow::{Context, Result};
use std::{fmt, str::FromStr};

/// The released video VAE compresses each spatial axis by 16.
pub const MINIMAX_H3_VAE_SPATIAL_COMPRESSION_RATIO: usize = 16;

/// The released VAE clip geometry that `17 * n + 5` frame alignment comes from.
pub const MINIMAX_H3_VAE_CLIP_LENGTH: usize = 17;
pub const MINIMAX_H3_VAE_TEMPORAL_COMPRESSION_RATIO: usize = 4;
pub const MINIMAX_H3_VAE_TOKEN_DROP: usize = 3;

/// The short edge MiniMax publishes 768p assets at.
pub const MINIMAX_H3_DEFAULT_SHORT_EDGE: usize = 768;

/// A `width:height` target ratio, as the official request spells it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct H3AspectRatio {
    pub width: usize,
    pub height: usize,
}

impl H3AspectRatio {
    /// The ratio the released 768p assets use: `1344x768`.
    pub const WIDESCREEN: Self = Self {
        width: 16,
        height: 9,
    };

    pub fn new(width: usize, height: usize) -> Result<Self> {
        anyhow::ensure!(
            width > 0 && height > 0,
            "aspect ratio terms must be positive; got {width}:{height}"
        );
        Ok(Self { width, height })
    }

    /// The ratio of an existing canvas, for the official `auto` target.
    pub fn of_canvas(width: usize, height: usize) -> Result<Self> {
        Self::new(width, height)
    }

    const fn is_landscape(self) -> bool {
        self.width >= self.height
    }
}

impl FromStr for H3AspectRatio {
    type Err = anyhow::Error;

    fn from_str(text: &str) -> Result<Self> {
        let (width, height) = text.split_once(':').with_context(|| {
            format!("aspect ratio must be written `width:height`, such as `16:9`; got `{text}`")
        })?;
        let parse = |term: &str, axis: &str| -> Result<usize> {
            term.parse::<usize>()
                .with_context(|| format!("aspect ratio {axis} term `{term}` is not a number"))
        };
        Self::new(parse(width, "width")?, parse(height, "height")?)
    }
}

impl fmt::Display for H3AspectRatio {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.width, self.height)
    }
}

/// An official target resolved into the geometry the pipeline actually runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct H3TargetGeometry {
    pub short_edge: usize,
    pub aspect_ratio: H3AspectRatio,
    pub duration_seconds: usize,
    pub canvas_height: usize,
    pub canvas_width: usize,
    pub latent_height: usize,
    pub latent_width: usize,
    /// `duration_seconds * 24`, before `17 * n + 5` alignment.
    pub requested_num_frames: usize,
    /// The aligned pixel-frame count the run actually produces.
    pub num_frames: usize,
    pub latent_frames: usize,
    pub audio_frames: usize,
}

impl H3TargetGeometry {
    /// The realised duration, which alignment can push past the request.
    pub fn aligned_duration_seconds(&self) -> f64 {
        self.num_frames as f64 / MINIMAX_H3_FPS as f64
    }
}

/// The pixel multiple each canvas axis must land on.
///
/// The VAE compresses by 16 and the transformer patchifies the latent by 2, so
/// a canvas axis is only representable when it is a multiple of both.
pub fn minimax_h3_canvas_alignment() -> (usize, usize) {
    (
        MINIMAX_H3_VAE_SPATIAL_COMPRESSION_RATIO * MINIMAX_H3_PATCH_SIZE[1],
        MINIMAX_H3_VAE_SPATIAL_COMPRESSION_RATIO * MINIMAX_H3_PATCH_SIZE[2],
    )
}

/// Convert an official `short_edge` / `aspect_ratio` / `duration_seconds`
/// target into the canvas, latent, and frame geometry the pipeline takes.
///
/// The long edge is floored to the canvas alignment rather than rounded, which
/// is what reproduces the released `1344x768`: `768 * 16 / 9` is `1365.33`, and
/// only flooring lands on `1344`.
pub fn resolve_h3_target_geometry(
    short_edge: usize,
    aspect_ratio: H3AspectRatio,
    duration_seconds: usize,
) -> Result<H3TargetGeometry> {
    anyhow::ensure!(
        (MINIMAX_H3_MIN_SECONDS..=MINIMAX_H3_MAX_SECONDS).contains(&duration_seconds),
        "MiniMax-H3 generates {MINIMAX_H3_MIN_SECONDS} through {MINIMAX_H3_MAX_SECONDS} seconds; requested {duration_seconds}"
    );
    let requested_num_frames = duration_seconds
        .checked_mul(MINIMAX_H3_FPS)
        .context("requested frame count overflow")?;
    let frames = resolve_fl2va_frame_geometry(
        requested_num_frames,
        MINIMAX_H3_VAE_CLIP_LENGTH,
        MINIMAX_H3_VAE_TEMPORAL_COMPRESSION_RATIO,
        MINIMAX_H3_VAE_TOKEN_DROP,
    )?;

    let (height_alignment, width_alignment) = minimax_h3_canvas_alignment();
    let (short_alignment, long_alignment) = if aspect_ratio.is_landscape() {
        (height_alignment, width_alignment)
    } else {
        (width_alignment, height_alignment)
    };
    anyhow::ensure!(short_edge > 0, "short_edge must be positive");
    anyhow::ensure!(
        short_edge.is_multiple_of(short_alignment),
        "short_edge must be a multiple of {short_alignment} for the {}x VAE and patch {} geometry; {short_edge} is not, try {} or {}",
        MINIMAX_H3_VAE_SPATIAL_COMPRESSION_RATIO,
        short_alignment / MINIMAX_H3_VAE_SPATIAL_COMPRESSION_RATIO,
        short_edge / short_alignment * short_alignment,
        short_edge.div_ceil(short_alignment) * short_alignment
    );
    let (long_term, short_term) = if aspect_ratio.is_landscape() {
        (aspect_ratio.width, aspect_ratio.height)
    } else {
        (aspect_ratio.height, aspect_ratio.width)
    };
    let long_edge = short_edge
        .checked_mul(long_term)
        .context("long edge overflow")?
        / short_term
        / long_alignment
        * long_alignment;
    anyhow::ensure!(
        long_edge > 0,
        "aspect ratio {aspect_ratio} at short_edge {short_edge} has no representable long edge"
    );

    let (canvas_height, canvas_width) = if aspect_ratio.is_landscape() {
        (short_edge, long_edge)
    } else {
        (long_edge, short_edge)
    };
    Ok(H3TargetGeometry {
        short_edge,
        aspect_ratio,
        duration_seconds,
        canvas_height,
        canvas_width,
        latent_height: canvas_height / MINIMAX_H3_VAE_SPATIAL_COMPRESSION_RATIO,
        latent_width: canvas_width / MINIMAX_H3_VAE_SPATIAL_COMPRESSION_RATIO,
        requested_num_frames,
        num_frames: frames.num_frames,
        latent_frames: frames.num_latent_frames,
        audio_frames: frames.num_audio_latents,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_768p_widescreen_target_reproduces_the_published_canvas() {
        let target = resolve_h3_target_geometry(
            MINIMAX_H3_DEFAULT_SHORT_EDGE,
            H3AspectRatio::WIDESCREEN,
            10,
        )
        .expect("released 768p target");
        assert_eq!((target.canvas_width, target.canvas_height), (1344, 768));
        assert_eq!((target.latent_width, target.latent_height), (84, 48));
    }

    #[test]
    fn released_durations_reproduce_the_published_frame_counts() {
        for (duration_seconds, num_frames, latent_frames, audio_frames) in
            [(10, 243, 72, 405), (8, 192, 57, 320), (5, 124, 37, 207)]
        {
            let target = resolve_h3_target_geometry(
                MINIMAX_H3_DEFAULT_SHORT_EDGE,
                H3AspectRatio::WIDESCREEN,
                duration_seconds,
            )
            .expect("released duration");
            assert_eq!(target.num_frames, num_frames, "{duration_seconds}s frames");
            assert_eq!(
                target.latent_frames, latent_frames,
                "{duration_seconds}s latent frames"
            );
            assert_eq!(
                target.audio_frames, audio_frames,
                "{duration_seconds}s audio frames"
            );
            assert_eq!(
                target.requested_num_frames,
                duration_seconds * MINIMAX_H3_FPS
            );
        }
    }

    #[test]
    fn the_five_second_widescreen_target_matches_the_command_line_defaults() {
        let target =
            resolve_h3_target_geometry(MINIMAX_H3_DEFAULT_SHORT_EDGE, H3AspectRatio::WIDESCREEN, 5)
                .expect("default target");
        assert_eq!(
            (
                target.latent_frames,
                target.latent_height,
                target.latent_width,
                target.audio_frames
            ),
            (37, 48, 84, 207)
        );
    }

    #[test]
    fn documented_aspect_ratios_stay_on_the_canvas_alignment() {
        let (height_alignment, width_alignment) = minimax_h3_canvas_alignment();
        for (ratio, width, height) in [
            ("21:9", 1792, 768),
            ("16:9", 1344, 768),
            ("4:3", 1024, 768),
            ("1:1", 768, 768),
            ("3:4", 768, 1024),
            ("9:16", 768, 1344),
        ] {
            let target = resolve_h3_target_geometry(
                MINIMAX_H3_DEFAULT_SHORT_EDGE,
                ratio.parse().expect("documented ratio"),
                10,
            )
            .expect("documented ratio resolves");
            assert_eq!(
                (target.canvas_width, target.canvas_height),
                (width, height),
                "{ratio}"
            );
            assert!(
                target.canvas_height.is_multiple_of(height_alignment),
                "{ratio}"
            );
            assert!(
                target.canvas_width.is_multiple_of(width_alignment),
                "{ratio}"
            );
            assert_eq!(target.short_edge, MINIMAX_H3_DEFAULT_SHORT_EDGE, "{ratio}");
        }
    }

    #[test]
    fn the_two_thousand_pixel_regenerate_target_is_exact() {
        let target =
            resolve_h3_target_geometry(1440, H3AspectRatio::WIDESCREEN, 10).expect("2K target");
        assert_eq!((target.canvas_width, target.canvas_height), (2560, 1440));
    }

    #[test]
    fn a_short_edge_off_the_canvas_alignment_is_refused_with_the_neighbours() {
        let error = resolve_h3_target_geometry(720, H3AspectRatio::WIDESCREEN, 10)
            .expect_err("720 is not a multiple of 32");
        let message = format!("{error}");
        assert!(
            message.contains("704") && message.contains("736"),
            "{message}"
        );
    }

    #[test]
    fn durations_outside_the_released_range_are_refused() {
        for duration_seconds in [0, MINIMAX_H3_MIN_SECONDS - 1, MINIMAX_H3_MAX_SECONDS + 1] {
            assert!(
                resolve_h3_target_geometry(
                    MINIMAX_H3_DEFAULT_SHORT_EDGE,
                    H3AspectRatio::WIDESCREEN,
                    duration_seconds
                )
                .is_err(),
                "{duration_seconds}s must be refused"
            );
        }
    }

    #[test]
    fn aspect_ratios_round_trip_through_their_official_spelling() {
        for text in ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"] {
            let ratio: H3AspectRatio = text.parse().expect("documented ratio");
            assert_eq!(format!("{ratio}"), text);
        }
        for text in ["16", "16:0", "0:9", "16:9:1", "sixteen:nine", ":", ""] {
            assert!(text.parse::<H3AspectRatio>().is_err(), "{text}");
        }
    }

    #[test]
    fn a_canvas_derived_ratio_reproduces_its_own_canvas() {
        let ratio = H3AspectRatio::of_canvas(1344, 768).expect("canvas ratio");
        let target = resolve_h3_target_geometry(768, ratio, 10).expect("auto target");
        assert_eq!((target.canvas_width, target.canvas_height), (1344, 768));
    }
}
