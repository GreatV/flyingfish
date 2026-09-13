//! Shared spatial tile layout for the visual VAE encoder and decoder.
//!
//! Both directions cut a canvas into fixed-size tiles that overlap by at least
//! `min_overlap` pixels and together cover the canvas exactly. The encoder and
//! decoder keep their own stitching order, which follows two different pinned
//! upstream implementations, but the cut itself is one contract and lives here.

use anyhow::Result;

/// Cut `length` into `tile`-sized spans overlapping by at least `min_overlap`.
///
/// Returns the tile starts, their lengths and the `count - 1` overlaps between
/// neighbours. The spare extent left after the minimum overlaps is distributed
/// round-robin in whole `ratio` steps, so every boundary stays on the VAE's
/// spatial compression grid.
pub(crate) fn split_tiles(
    length: usize,
    tile: usize,
    min_overlap: usize,
    ratio: usize,
) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    anyhow::ensure!(
        tile > 0 && ratio > 0,
        "tile size and spatial ratio must be non-zero"
    );
    anyhow::ensure!(
        min_overlap < tile,
        "tile overlap must be smaller than tile size"
    );
    if tile >= length {
        return Ok((vec![0], vec![length], vec![]));
    }
    let mut count = length.div_ceil(tile);
    while tile * count < min_overlap * (count - 1) + length {
        count += 1;
    }
    let mut overlaps = vec![min_overlap; count - 1];
    let spare = tile * count - min_overlap * (count - 1) - length;
    anyhow::ensure!(
        spare.is_multiple_of(ratio),
        "tile layout for length {length} with tile {tile}, overlap {min_overlap} and ratio {ratio} leaves {spare} unaligned spare pixels"
    );
    for index in 0..spare / ratio {
        let slot = index % overlaps.len();
        overlaps[slot] += ratio;
    }
    let mut starts = Vec::with_capacity(count);
    let mut start = 0usize;
    starts.push(start);
    for overlap in &overlaps {
        start += tile - overlap;
        starts.push(start);
    }
    Ok((starts, vec![tile; count], overlaps))
}

#[cfg(test)]
mod tests {
    use super::split_tiles;

    #[test]
    fn released_tile_layout_exactly_covers_a_real_canvas() {
        for length in [1344, 768, 896, 512] {
            let (starts, lengths, overlaps) = split_tiles(length, 256, 64, 16).unwrap();
            assert_eq!(
                starts.last().unwrap() + lengths.last().unwrap(),
                length,
                "canvas {length} is not exactly covered"
            );
            assert_eq!(starts.len(), lengths.len());
            assert_eq!(overlaps.len(), lengths.len() - 1);
            assert!(
                overlaps
                    .iter()
                    .all(|overlap| *overlap >= 64 && overlap.is_multiple_of(16))
            );
        }
    }

    #[test]
    fn a_single_tile_covers_a_canvas_no_larger_than_the_tile() {
        let (starts, lengths, overlaps) = split_tiles(200, 256, 64, 16).unwrap();
        assert_eq!(starts, vec![0]);
        assert_eq!(lengths, vec![200]);
        assert!(overlaps.is_empty());
    }

    #[test]
    fn unaligned_spare_extent_is_rejected_instead_of_overrunning_the_canvas() {
        let error = split_tiles(1350, 256, 64, 16).unwrap_err().to_string();
        assert!(error.contains("unaligned spare pixels"), "{error}");
    }

    #[test]
    fn degenerate_tile_geometry_is_rejected() {
        assert!(split_tiles(1344, 0, 64, 16).is_err());
        assert!(split_tiles(1344, 256, 64, 0).is_err());
        assert!(split_tiles(1344, 64, 64, 16).is_err());
    }
}
