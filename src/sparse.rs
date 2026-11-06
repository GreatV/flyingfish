//! Sparse lattice indexing; batch is the leading coordinate.
use anyhow::{Context, Result, ensure};

pub struct SparseMesh {
    /// Row-major [vertices,3] world coordinates.
    pub vertices: Vec<f32>,
    /// Row-major [triangles,3] indices; winding and multiplicity are retained.
    pub faces: Vec<i32>,
}

fn cube_edge_corners(edge: usize) -> [usize; 2] {
    let axis = edge / 4;
    let compressed = edge % 4;
    let lower = (compressed & ((1 << axis) - 1)) | ((compressed >> axis) << (axis + 1));
    [lower, lower | (1 << axis)]
}

fn cube_face_edges(axis: usize, side: usize) -> [usize; 4] {
    let others: Vec<_> = (0..3).filter(|&a| a != axis).collect();
    let base = side << axis;
    let corners = [
        base,
        base | (1 << others[0]),
        base | (1 << others[0]) | (1 << others[1]),
        base | (1 << others[1]),
    ];
    std::array::from_fn(|i| {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        let direction = (a ^ b).trailing_zeros() as usize;
        let lower = a.min(b);
        4 * direction + (lower & ((1 << direction) - 1)) + ((lower >> (direction + 1)) << direction)
    })
}

fn cube_components(mask: u8) -> Result<[i8; 12]> {
    let mut adjacency = [0u16; 12];
    let mut crossing = 0u16;
    for edge in 0..12 {
        let [a, b] = cube_edge_corners(edge);
        if ((mask >> a) ^ (mask >> b)) & 1 != 0 {
            crossing |= 1 << edge;
        }
    }
    for axis in 0..3 {
        for side in 0..2 {
            let rim = cube_face_edges(axis, side);
            let selected: Vec<_> = rim
                .into_iter()
                .filter(|&e| crossing & (1 << e) != 0)
                .collect();
            let mut join = |a, b| {
                adjacency[a] |= 1u16 << b;
                adjacency[b] |= 1u16 << a;
            };
            if selected.len() == 2 {
                join(selected[0], selected[1]);
            } else if selected.len() == 4 {
                for i in 0..4 {
                    let previous = cube_edge_corners(rim[(i + 3) % 4]);
                    let current = cube_edge_corners(rim[i]);
                    let corner = *current
                        .iter()
                        .find(|c| previous.contains(c))
                        .context("cube face edges do not meet")?;
                    if mask & (1 << corner) != 0 {
                        join(rim[(i + 3) % 4], rim[i]);
                    }
                }
            } else {
                ensure!(selected.is_empty(), "invalid cube face crossing parity");
            }
        }
    }
    for (edge, neighbours) in adjacency.iter().enumerate() {
        ensure!(
            neighbours.count_ones() == if crossing & (1 << edge) != 0 { 2 } else { 0 },
            "cube face graph is not a disjoint union of cycles"
        );
    }
    let mut order: Vec<_> = (0..12).collect();
    order.sort_unstable_by_key(|&edge| cube_edge_corners(edge));
    let mut labels = [-1i8; 12];
    let mut component = 0;
    for edge in order {
        if crossing & (1 << edge) == 0 || labels[edge] >= 0 {
            continue;
        }
        let mut pending = 1u16 << edge;
        while pending != 0 {
            let next = pending.trailing_zeros() as usize;
            pending &= !(1 << next);
            if labels[next] < 0 {
                labels[next] = component;
                pending |= adjacency[next];
            }
        }
        component += 1;
    }
    ensure!(component <= 4, "cube dual vertex count exceeds four");
    Ok(labels)
}

/// Rows contain 12 edge-component IDs, component count, and six alternating-face split bits.
/// Corners use x-bit0; edge = 4*axis + compressed lower corner; face = 2*axis + side.
pub fn cube_topology() -> Result<[[i8; 14]; 256]> {
    let mut table = [[-1i8; 14]; 256];
    for (mask, row) in table.iter_mut().enumerate() {
        let labels = cube_components(mask as u8)?;
        row[..12].copy_from_slice(&labels);
        row[12] = labels.into_iter().max().context("empty cube edge labels")? + 1;
        row[13] = 0;
    }
    for mask in 0..256 {
        for axis in 0..3 {
            for side in 0..2 {
                let rim = cube_face_edges(axis, side);
                if rim.iter().all(|&edge| table[mask][edge] >= 0) {
                    let own = table[mask][rim[0]];
                    let opposite = table[mask ^ 255][rim[0]];
                    if rim.iter().all(|&edge| table[mask][edge] == own)
                        && rim.iter().any(|&edge| table[mask ^ 255][edge] != opposite)
                    {
                        table[mask][13] |= 1 << (2 * axis + side);
                    }
                }
            }
        }
    }
    Ok(table)
}

pub struct SparseSites {
    coordinates: Vec<[i32; 4]>,
    sorted: Vec<([i32; 4], i32)>,
}

pub struct SparseCorners {
    pub sites: SparseSites,
    /// Row-major [cubes,8], with x-bit0 corners.
    pub cube_to_corner: Vec<u32>,
    pub contributor_offsets: Vec<u64>,
    /// Flattened cube/corner IDs, grouped by unique corner.
    pub contributors: Vec<u32>,
}

pub struct SerializedWindows {
    pub(crate) keys: [Vec<u64>; 2],
    pub(crate) order: [Vec<u32>; 2],
    pub(crate) inverse: [Vec<u32>; 2],
    pub(crate) pad: Vec<u32>,
    pub(crate) unpad: Vec<u32>,
    pub(crate) offsets: Vec<u32>,
}
impl SparseSites {
    pub fn cube_corners(&self) -> Result<SparseCorners> {
        let count = self
            .coordinates
            .len()
            .checked_mul(8)
            .filter(|&v| v <= i32::MAX as usize)
            .context("expanded cube corner count exceeds indexing capacity")?;
        ensure!(
            self.coordinates
                .iter()
                .all(|p| p[1..].iter().all(|&v| v < i32::MAX)),
            "cube corner coordinate overflow"
        );
        let mut expanded = Vec::with_capacity(count);
        for p in &self.coordinates {
            for corner in 0..8 {
                expanded.push([
                    p[0],
                    p[1] + (corner & 1),
                    p[2] + ((corner >> 1) & 1),
                    p[3] + ((corner >> 2) & 1),
                ]);
            }
        }
        let (sites, cube_to_corner) = Self::deduplicate(expanded)?;
        let unique = sites.coordinates.len();
        let mut offsets = vec![0u64; unique + 1];
        for &corner in &cube_to_corner {
            offsets[corner as usize + 1] += 1;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let mut cursor = offsets[..unique].to_vec();
        let mut contributors = vec![0u32; count];
        for (source, &corner) in cube_to_corner.iter().enumerate() {
            let at = &mut cursor[corner as usize];
            contributors[*at as usize] = source as u32;
            *at += 1;
        }
        Ok(SparseCorners {
            sites,
            cube_to_corner,
            contributor_offsets: offsets,
            contributors,
        })
    }

    /// Lexicographic unique coordinates and input-row to unique-row indices.
    pub fn deduplicate(coordinates: Vec<[i32; 4]>) -> Result<(Self, Vec<u32>)> {
        ensure!(
            !coordinates.is_empty() && coordinates.len() <= i32::MAX as usize,
            "invalid sparse corner count"
        );
        let mut order: Vec<_> = coordinates.into_iter().enumerate().collect();
        order.sort_unstable_by_key(|&(_, p)| p);
        let mut inverse = vec![0; order.len()];
        let mut unique = Vec::new();
        for (row, point) in order {
            if unique.last() != Some(&point) {
                unique.push(point);
            }
            inverse[row] = (unique.len() - 1) as u32;
        }
        Ok((Self::new(unique)?, inverse))
    }
    pub fn new(coordinates: Vec<[i32; 4]>) -> Result<Self> {
        ensure!(
            !coordinates.is_empty() && coordinates.len() <= i32::MAX as usize,
            "invalid sparse site count"
        );
        ensure!(
            coordinates.iter().flatten().all(|&x| x >= 0),
            "negative sparse coordinate"
        );
        ensure!(
            coordinates.windows(2).all(|p| p[0][0] <= p[1][0]),
            "sparse sample rows must be contiguous and batch ordered"
        );
        let mut sorted: Vec<_> = coordinates
            .iter()
            .copied()
            .enumerate()
            .map(|(i, p)| (p, i as i32))
            .collect();
        sorted.sort_unstable_by_key(|&(p, _)| p);
        ensure!(
            sorted.windows(2).all(|p| p[0].0 != p[1].0),
            "duplicate sparse coordinate"
        );
        Ok(Self {
            coordinates,
            sorted,
        })
    }
    pub fn coordinates(&self) -> &[[i32; 4]] {
        &self.coordinates
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn sorted_rows(&self) -> &[([i32; 4], i32)] {
        &self.sorted
    }
    pub fn row(&self, point: [i32; 4]) -> Option<i32> {
        self.sorted
            .binary_search_by_key(&point, |&(p, _)| p)
            .ok()
            .map(|i| self.sorted[i].1)
    }
    pub fn serialized_windows(&self, window: usize) -> Result<SerializedWindows> {
        ensure!(
            window > 0 && window <= i32::MAX as usize,
            "invalid serialized window size"
        );
        let maximum = self
            .coordinates
            .iter()
            .flat_map(|p| &p[1..])
            .copied()
            .max()
            .context("empty sparse coordinates")? as u32;
        let depth = u32::BITS - maximum.leading_zeros();
        let batches = self
            .coordinates
            .last()
            .context("empty sparse coordinates")?[0] as u64
            + 1;
        ensure!(
            depth <= 16 && 3 * depth + u64::BITS - batches.leading_zeros() <= 63,
            "serialized coordinate or batch exceeds Morton capacity"
        );
        let rows = self.coordinates.len();
        let mut result = SerializedWindows {
            keys: std::array::from_fn(|_| Vec::with_capacity(rows)),
            order: std::array::from_fn(|_| Vec::with_capacity(rows)),
            inverse: std::array::from_fn(|_| vec![0; rows]),
            pad: Vec::new(),
            unpad: Vec::with_capacity(rows),
            offsets: vec![0],
        };
        for order in 0..2 {
            for p in &self.coordinates {
                let axes = if order == 0 {
                    [p[1], p[2], p[3]]
                } else {
                    [p[2], p[1], p[3]]
                };
                let mut key = (p[0] as u64) << (3 * depth);
                for bit in 0..depth {
                    for (axis, value) in axes.into_iter().enumerate() {
                        key |= (((value as u64) >> bit) & 1) << (3 * bit + 2 - axis as u32);
                    }
                }
                result.keys[order].push(key);
            }
            result.order[order].extend(0..rows as u32);
            result.order[order].sort_unstable_by_key(|&row| result.keys[order][row as usize]);
            for (rank, &row) in result.order[order].iter().enumerate() {
                result.inverse[order][row as usize] = rank as u32;
            }
        }
        let mut first = 0;
        while first < rows {
            let count =
                self.coordinates[first..].partition_point(|p| p[0] == self.coordinates[first][0]);
            let padded = if count > window {
                count
                    .div_ceil(window)
                    .checked_mul(window)
                    .context("serialized padded count overflow")?
            } else {
                count
            };
            let base = result.pad.len();
            let end = base
                .checked_add(padded)
                .context("serialized padded count overflow")?;
            ensure!(
                end <= i32::MAX as usize,
                "serialized padded rows exceed indexing capacity"
            );
            result
                .unpad
                .extend((base..base + count).map(|row| row as u32));
            result
                .pad
                .extend((first..first + count).map(|row| row as u32));
            if padded > count {
                let tail = count % window;
                result
                    .pad
                    .extend((first + count - window..first + count - tail).map(|row| row as u32));
            }
            for start in (base..end).step_by(window) {
                result
                    .offsets
                    .push((start + window.min(end - start)) as u32);
            }
            first += count;
        }
        Ok(result)
    }
    pub fn downsample(&self, factor: i32) -> Result<(Self, Vec<u32>)> {
        ensure!(factor > 0, "invalid sparse downsample factor");
        let parents: Vec<_> = self
            .coordinates
            .iter()
            .map(|p| [p[0], p[1] / factor, p[2] / factor, p[3] / factor])
            .collect();
        let mut coordinates = parents.clone();
        coordinates.sort_unstable();
        coordinates.dedup();
        let inverse = parents
            .iter()
            .map(|p| {
                coordinates
                    .binary_search(p)
                    .ok()
                    .map(|i| i as u32)
                    .context("sparse parent not found")
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((Self::new(coordinates)?, inverse))
    }
    /// Published [kx,ky,kz] order; the missing-neighbour sentinel is -1.
    pub fn neighbours_3x3(&self) -> Result<Vec<[i32; 27]>> {
        self.coordinates
            .len()
            .checked_mul(27 * size_of::<i32>())
            .context("sparse neighbour allocation overflow")?;
        self.coordinates
            .iter()
            .map(|&p| {
                let mut map = [-1; 27];
                for (tap, slot) in map.iter_mut().enumerate() {
                    let delta = [
                        (tap / 9) as i32 - 1,
                        ((tap / 3) % 3) as i32 - 1,
                        (tap % 3) as i32 - 1,
                    ];
                    let mut neighbour = p;
                    for axis in 0..3 {
                        neighbour[axis + 1] = p[axis + 1]
                            .checked_add(delta[axis])
                            .context("sparse neighbour coordinate overflow")?;
                    }
                    *slot = self.row(neighbour).unwrap_or(-1);
                }
                Ok(map)
            })
            .collect()
    }
    /// Children retain parent order; child bit zero is X, then Y, then Z.
    pub fn subdivide(
        &self,
        active: &[[bool; 8]],
        capacity: usize,
    ) -> Result<(Self, Vec<[u32; 2]>)> {
        self.subdivide_with_offsets(
            active,
            capacity,
            &std::array::from_fn(|child| std::array::from_fn(|axis| ((child >> axis) & 1) as i32)),
        )
    }
    pub fn subdivide_with_offsets(
        &self,
        active: &[[bool; 8]],
        capacity: usize,
        offsets: &[[i32; 3]; 8],
    ) -> Result<(Self, Vec<[u32; 2]>)> {
        ensure!(
            offsets.iter().flatten().all(|&x| matches!(x, 0 | 1))
                && (0..8).all(|child| !offsets[..child].contains(&offsets[child])),
            "invalid subdivision child offsets"
        );
        ensure!(
            active.len() == self.coordinates.len(),
            "subdivision mask/site count mismatch"
        );
        let count = active.iter().flatten().filter(|&&x| x).count();
        ensure!(
            count > 0 && count <= capacity && count <= i32::MAX as usize,
            "subdivision exceeds capacity or has no active child"
        );
        let mut coordinates = Vec::with_capacity(count);
        let mut source = Vec::with_capacity(count);
        for (row, (&parent, mask)) in self.coordinates.iter().zip(active).enumerate() {
            for (child, &enabled) in mask.iter().enumerate() {
                if !enabled {
                    continue;
                }
                let mut point = parent;
                for axis in 0..3 {
                    point[axis + 1] = parent[axis + 1]
                        .checked_mul(2)
                        .and_then(|x| x.checked_add(offsets[child][axis]))
                        .context("subdivision coordinate overflow")?;
                }
                coordinates.push(point);
                source.push([row as u32, child as u32]);
            }
        }
        Ok((Self::new(coordinates)?, source))
    }
    pub fn scaled_coordinates(&self, source_grid: u32, target_grid: u32) -> Result<Vec<[f32; 3]>> {
        ensure!(
            source_grid > 0 && target_grid > 1 && target_grid <= i32::MAX as u32,
            "invalid cascade grids"
        );
        let mut coordinates = Vec::with_capacity(self.coordinates.len());
        for &p in &self.coordinates {
            let mut result = [0.0; 3];
            for axis in 1..4 {
                ensure!(
                    (p[axis] as u32) < source_grid,
                    "cascade coordinate outside source grid"
                );
                let centered = p[axis] as f32 + 0.5;
                let scaled = (centered / source_grid as f32) * (target_grid - 1) as f32;
                ensure!(
                    scaled.is_finite() && scaled.round_ties_even() < i32::MAX as f32,
                    "quantized coordinate overflow"
                );
                result[axis - 1] = scaled;
            }
            coordinates.push(result);
        }
        Ok(coordinates)
    }
    pub fn round_unique(&self, source_grid: u32, target_grid: u32) -> Result<Self> {
        let scaled = self.scaled_coordinates(source_grid, target_grid)?;
        let coordinates: Vec<_> = self
            .coordinates
            .iter()
            .zip(scaled)
            .map(|(p, v)| {
                [
                    p[0],
                    v[0].round_ties_even() as i32,
                    v[1].round_ties_even() as i32,
                    v[2].round_ties_even() as i32,
                ]
            })
            .collect();
        Ok(Self::deduplicate(coordinates)?.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn component_sets(row: &[i8; 14]) -> Vec<Vec<[usize; 2]>> {
        let mut groups = vec![Vec::new(); row[12] as usize];
        for (edge, &group) in row[..12].iter().enumerate() {
            if group >= 0 {
                groups[group as usize].push(cube_edge_corners(edge));
            }
        }
        for group in &mut groups {
            group.sort_unstable();
        }
        groups.sort_unstable();
        groups
    }

    fn oriented<const N: usize>(face: [usize; N]) -> [usize; N] {
        let first = face.iter().enumerate().min_by_key(|&(_, v)| v).unwrap().0;
        std::array::from_fn(|i| face[(first + i) % N])
    }

    fn closed_case(
        table: &[[i8; 14]; 256],
        mask: usize,
        flip: bool,
    ) -> Result<(Vec<[i8; 12]>, Vec<[usize; 4]>, Vec<[usize; 3]>)> {
        let coordinates: Vec<_> = (0..4)
            .flat_map(|x| (0..4).flat_map(move |y| (0..4).map(move |z| [0, x, y, z])))
            .collect();
        let sites = SparseSites::new(coordinates)?;
        let negative = |p: [i32; 4]| {
            let inside = p[1..].iter().all(|&v| (1..=2).contains(&v));
            let sign = inside && mask & (1 << ((p[1] - 1) + 2 * (p[2] - 1) + 4 * (p[3] - 1))) != 0;
            sign ^ flip
        };
        let masks: Vec<usize> = sites
            .coordinates()
            .iter()
            .map(|p| {
                (0..8).fold(0, |mask, i| {
                    let corner = [
                        0,
                        p[1] + (i & 1),
                        p[2] + ((i >> 1) & 1),
                        p[3] + ((i >> 2) & 1),
                    ];
                    mask | (usize::from(negative(corner)) << i)
                })
            })
            .collect();
        let mut resolved = Vec::new();
        for (row, p) in sites.coordinates().iter().enumerate() {
            let mut split = false;
            for axis in 0..3 {
                for side in 0..2 {
                    let mut q = *p;
                    q[axis + 1] += if side == 0 { -1 } else { 1 };
                    if let Some(other) = sites.row(q) {
                        split |= table[masks[row]][13] & (1 << (2 * axis + side)) != 0
                            && table[masks[other as usize]][13] & (1 << (2 * axis + 1 - side)) != 0;
                    }
                }
            }
            resolved.push(if split { masks[row] ^ 255 } else { masks[row] });
        }
        let mut owners: BTreeMap<([i32; 4], usize), Vec<(usize, usize)>> = BTreeMap::new();
        let mut labels = Vec::new();
        for (row, p) in sites.coordinates().iter().enumerate() {
            let components = table[resolved[row]];
            labels.push(std::array::from_fn(|e| components[e]));
            for (edge, &component) in components[..12].iter().enumerate() {
                if component < 0 {
                    continue;
                }
                let lower = cube_edge_corners(edge)[0];
                let point = [
                    0,
                    p[1] + (lower & 1) as i32,
                    p[2] + ((lower >> 1) & 1) as i32,
                    p[3] + ((lower >> 2) & 1) as i32,
                ];
                owners
                    .entry((point, edge / 4))
                    .or_default()
                    .push((row, row * 4 + component as usize));
            }
        }
        let mut quads = Vec::new();
        let mut faces = Vec::new();
        for ((point, axis), rows) in owners {
            assert_eq!(rows.len(), 4);
            let b = (axis + 1) % 3 + 1;
            let c = (axis + 2) % 3 + 1;
            let mut quad = [0; 4];
            for (i, (db, dc)) in [(-1, -1), (0, -1), (0, 0), (-1, 0)].into_iter().enumerate() {
                quad[i] = rows
                    .iter()
                    .find(|&&(row, _)| {
                        let p = sites.coordinates()[row];
                        p[b] - point[b] == db && p[c] - point[c] == dc
                    })
                    .unwrap()
                    .1;
            }
            if !negative(point) {
                quad.reverse();
            }
            let [a, b, c, d] = quad;
            quads.push(quad);
            faces.extend(if axis == 1 {
                [[a, b, d], [b, c, d]]
            } else {
                [[a, b, c], [a, c, d]]
            });
        }
        Ok((labels, quads, faces))
    }

    fn closed_winding(faces: &[[usize; 3]]) -> bool {
        let mut incidence: BTreeMap<[usize; 2], (usize, i32)> = BTreeMap::new();
        for face in faces {
            for i in 0..3 {
                let a = face[i];
                let b = face[(i + 1) % 3];
                let value = incidence.entry([a.min(b), a.max(b)]).or_default();
                value.0 += 1;
                value.1 += if a < b { 1 } else { -1 };
            }
        }
        incidence
            .values()
            .all(|&(count, direction)| count == 2 && direction == 0)
    }

    #[test]
    fn cube_face_graph_rotations_complements_and_closed_winding() -> Result<()> {
        let table = cube_topology()?;
        assert_eq!(table[6][12], 2);
        assert_eq!(table[249][12], 1);
        assert_eq!(table.iter().map(|row| row[12]).max(), Some(4));
        assert_eq!(table.iter().filter(|row| row[12] == 0).count(), 2);
        assert_eq!(
            (0..256)
                .filter(|&m| component_sets(&table[m]) != component_sets(&table[m ^ 255]))
                .count(),
            120
        );
        let mut rotations = 0;
        for axes in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let parity = (usize::from(axes[0] > axes[1])
                + usize::from(axes[0] > axes[2])
                + usize::from(axes[1] > axes[2]))
                % 2;
            for reflection in 0u8..8 {
                if reflection.count_ones() as usize % 2 != parity {
                    continue;
                }
                rotations += 1;
                let map = |corner: usize| {
                    axes.iter().enumerate().fold(0, |result, (a, &b)| {
                        result | ((((corner >> b) & 1) ^ usize::from((reflection >> a) & 1)) << a)
                    })
                };
                for mask in 0..256 {
                    let rotated =
                        (0..8).fold(0, |result, c| result | (((mask >> c) & 1) << map(c)));
                    let mut expected = component_sets(&table[mask]);
                    for group in &mut expected {
                        for edge in group.iter_mut() {
                            *edge = edge.map(map);
                            edge.sort_unstable();
                        }
                        group.sort_unstable();
                    }
                    expected.sort_unstable();
                    assert_eq!(component_sets(&table[rotated]), expected);
                }
            }
        }
        assert_eq!(rotations, 24);
        let mut agree = 0;
        for mask in 0..256 {
            let (a, qa, fa) = closed_case(&table, mask, false)?;
            let (b, qb, fb) = closed_case(&table, mask, true)?;
            assert!(closed_winding(&fa) && closed_winding(&fb));
            if a == b {
                agree += 1;
                let mut forward: Vec<_> = qa.into_iter().map(oriented).collect();
                let mut reverse: Vec<_> = qb
                    .into_iter()
                    .map(|mut q| {
                        q.reverse();
                        oriented(q)
                    })
                    .collect();
                forward.sort_unstable();
                reverse.sort_unstable();
                assert_eq!(forward, reverse);
            }
            if mask == 1 {
                let mut control = fa;
                control[0].reverse();
                assert!(!closed_winding(&control));
            }
        }
        assert_eq!(agree, 172);
        Ok(())
    }

    #[test]
    fn sparse_maps_preserve_batches_axes_children_and_half_even() -> Result<()> {
        let sites = SparseSites::new(vec![[0, 2, 3, 5], [0, 3, 3, 5], [0, 2, 3, 6], [1, 2, 3, 5]])?;
        let maps = sites.neighbours_3x3()?;
        assert_eq!(maps[0][13], 0);
        assert_eq!(maps[0][22], 1);
        assert_eq!(maps[0][14], 2);
        assert_eq!(maps[3][13], 3);
        assert_eq!(maps[3].iter().filter(|&&r| r >= 0).count(), 1);
        assert_ne!(maps[0][22], maps[0][14]);
        let (parents, inverse) = sites.downsample(2)?;
        assert_eq!(
            parents.coordinates(),
            &[[0, 1, 1, 2], [0, 1, 1, 3], [1, 1, 1, 2]]
        );
        assert_eq!(inverse, [0, 0, 1, 2]);
        assert!(sites.downsample(0).is_err());
        let (corners, inverse) =
            SparseSites::deduplicate(vec![[1, 0, 2, 3], [0, 4, 5, 6], [0, 4, 5, 6], [0, 1, 2, 3]])?;
        assert_eq!(
            corners.coordinates(),
            &[[0, 1, 2, 3], [0, 4, 5, 6], [1, 0, 2, 3]]
        );
        assert_eq!(inverse, [2, 1, 1, 0]);
        let coordinates: Vec<_> = (0..3)
            .flat_map(|x| (0..2).flat_map(move |y| (0..2).map(move |z| [0, x, y, z])))
            .collect();
        let cubes = SparseSites::new((0..12).map(|i| coordinates[i * 7 % 12]).collect())?;
        let corners = cubes.cube_corners()?;
        assert_eq!(corners.sites.coordinates().len(), 36);
        assert_eq!(corners.cube_to_corner.len(), 96);
        assert_eq!(corners.contributors.len(), 96);
        assert_eq!(corners.contributor_offsets.last(), Some(&96));
        let mut degrees = std::collections::BTreeSet::new();
        let mut control_mismatches = 0;
        for (unique, pair) in corners.contributor_offsets.windows(2).enumerate() {
            degrees.insert(pair[1] - pair[0]);
            for &source in &corners.contributors[pair[0] as usize..pair[1] as usize] {
                assert_eq!(corners.cube_to_corner[source as usize] as usize, unique);
                let p = cubes.coordinates()[source as usize / 8];
                let c = source % 8;
                assert_eq!(
                    corners.sites.coordinates()[unique],
                    [
                        p[0],
                        p[1] + (c & 1) as i32,
                        p[2] + ((c >> 1) & 1) as i32,
                        p[3] + ((c >> 2) & 1) as i32
                    ]
                );
                let swapped = ((c & 1) << 2) | (c & 2) | ((c & 4) >> 2);
                control_mismatches += usize::from(
                    corners.cube_to_corner[source as usize]
                        != corners.cube_to_corner[source as usize / 8 * 8 + swapped as usize],
                );
            }
        }
        assert_eq!(degrees, [1, 2, 4, 8].into_iter().collect());
        assert_eq!(control_mismatches, 48);
        assert!(
            SparseSites::new(vec![[0, i32::MAX, 0, 0]])?
                .cube_corners()
                .is_err()
        );
        let masks = [
            [false, true, false, false, false, false, false, false],
            [false; 8],
            [false; 8],
            [false, false, false, false, true, false, false, false],
        ];
        let (children, source) = sites.subdivide(&masks, 2)?;
        assert_eq!(children.coordinates(), &[[0, 5, 6, 10], [1, 4, 6, 11]]);
        assert_eq!(source, [[0, 1], [3, 4]]);
        assert!(sites.subdivide(&masks, 1).is_err());
        let z_first = std::array::from_fn(|child| {
            std::array::from_fn(|axis| ((child >> (2 - axis)) & 1) as i32)
        });
        let (children, source) = sites.subdivide_with_offsets(&masks, 2, &z_first)?;
        assert_eq!(children.coordinates(), &[[0, 4, 6, 11], [1, 5, 6, 10]]);
        assert_eq!(source, [[0, 1], [3, 4]]);
        assert!(
            sites
                .subdivide_with_offsets(&masks, 2, &[[0; 3]; 8])
                .is_err()
        );
        assert!(sites.serialized_windows(0).is_err());
        assert!(
            SparseSites::new(vec![[0, 65536, 0, 0]])?
                .serialized_windows(48)
                .is_err()
        );
        let ties = SparseSites::new(vec![[0, 0, 1, 2], [0, 1, 1, 2], [1, 0, 1, 2]])?
            .round_unique(512, 513)?;
        assert_eq!(
            ties.coordinates(),
            &[[0, 0, 2, 2], [0, 2, 2, 2], [1, 0, 2, 2]]
        );
        assert!(SparseSites::new(vec![[0, 1, 2, 3], [0, 1, 2, 3]]).is_err());
        assert!(SparseSites::new(vec![[0, 0, -1, 0]]).is_err());
        assert!(SparseSites::new(vec![[1, 0, 0, 0], [0, 0, 0, 0]]).is_err());
        assert!(
            SparseSites::new(vec![[0, i32::MAX, 0, 0]])?
                .neighbours_3x3()
                .is_err()
        );
        assert!(
            SparseSites::new(vec![[0, i32::MAX, 0, 0]])?
                .subdivide(&[[true; 8]], 8)
                .is_err()
        );
        Ok(())
    }
}
