//! Bounded PNG inspection.
//!
//! Frames are decoded only far enough to confirm RGB8 and their dimensions,
//! under an explicit work-memory ceiling, so a hostile or corrupt file cannot
//! turn manifest validation into an unbounded allocation.

use super::*;

pub(super) const PNG_DECODER_WORK_BYTES: usize = 64 * 1024 * 1024;

pub(super) const PNG_IEND: [u8; 12] = [0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xae, 0x42, 0x60, 0x82];

pub(super) fn validate_dimensions(width: u32, height: u32) -> Result<u64> {
    anyhow::ensure!(
        (1..=MAX_PNG_FRAME_DIMENSION).contains(&width)
            && (1..=MAX_PNG_FRAME_DIMENSION).contains(&height),
        "PNG frame dimensions must each be in 1..={MAX_PNG_FRAME_DIMENSION}"
    );
    let decoded_bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(3))
        .context("PNG decoded frame size overflow")?;
    anyhow::ensure!(
        decoded_bytes <= MAX_PNG_DECODED_FRAME_BYTES,
        "PNG decoded frame size exceeds {MAX_PNG_DECODED_FRAME_BYTES} bytes"
    );
    Ok(decoded_bytes)
}

pub(super) fn decode_png_rgb8(file_name: &str, bytes: &[u8]) -> Result<(u32, u32)> {
    anyhow::ensure!(
        bytes.ends_with(&PNG_IEND),
        "PNG frame {file_name} does not end at a valid IEND marker"
    );
    let decoder = png::Decoder::new_with_limits(
        Cursor::new(bytes),
        png::Limits {
            bytes: PNG_DECODER_WORK_BYTES,
        },
    );
    let mut reader = decoder
        .read_info()
        .with_context(|| format!("failed to read PNG frame header {file_name}"))?;
    let info = reader.info();
    anyhow::ensure!(
        info.animation_control.is_none() && info.frame_control.is_none(),
        "PNG frame {file_name} must not be animated"
    );
    anyhow::ensure!(
        info.color_type == png::ColorType::Rgb && info.bit_depth == png::BitDepth::Eight,
        "PNG frame {file_name} must use RGB8 encoding"
    );
    let width = info.width;
    let height = info.height;
    let decoded_bytes = validate_dimensions(width, height)?;
    let decoded_bytes = usize::try_from(decoded_bytes).context("PNG decoded size exceeds usize")?;
    anyhow::ensure!(
        reader.output_buffer_size() == Some(decoded_bytes),
        "PNG frame {file_name} decoded size does not match its dimensions"
    );
    let mut pixels = vec![0u8; decoded_bytes];
    let output = reader
        .next_frame(&mut pixels)
        .with_context(|| format!("failed to decode PNG frame {file_name}"))?;
    anyhow::ensure!(
        output.width == width
            && output.height == height
            && output.color_type == png::ColorType::Rgb
            && output.bit_depth == png::BitDepth::Eight
            && output.buffer_size() == decoded_bytes,
        "PNG frame {file_name} decoded output does not match RGB8 dimensions"
    );
    reader
        .finish()
        .with_context(|| format!("failed to finish decoding PNG frame {file_name}"))?;
    anyhow::ensure!(
        reader.info().animation_control.is_none() && reader.info().frame_control.is_none(),
        "PNG frame {file_name} must not be animated"
    );
    Ok((width, height))
}
