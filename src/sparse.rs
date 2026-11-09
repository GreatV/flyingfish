//! Sparse lattice indexing; batch is the leading coordinate.
use anyhow::{Context, Result, ensure};

pub struct SparseMesh {
    /// Row-major [vertices,3] world coordinates.
    pub vertices: Vec<f32>,
    /// Row-major [triangles,3] indices; winding and multiplicity are retained.
    pub faces: Vec<i32>,
}

impl SparseMesh {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.vertices.is_empty()
                && !self.faces.is_empty()
                && self.vertices.len().is_multiple_of(3)
                && self.faces.len().is_multiple_of(3)
                && self.vertices.iter().all(|v| v.is_finite())
                && self
                    .faces
                    .iter()
                    .all(|&v| v >= 0 && (v as usize) < self.vertices.len() / 3),
            "invalid mesh buffers"
        );
        Ok(())
    }

    /// Exact-two-incidence edge components; returned maps index the original faces and vertices.
    pub fn filter_components(
        &mut self,
        minimum_face_ratio: f64,
    ) -> Result<(Vec<usize>, Vec<usize>)> {
        self.validate()?;
        ensure!(
            self.vertices
                .as_chunks::<3>()
                .0
                .iter()
                .any(|p| p != &self.vertices[..3]),
            "mesh must have non-zero extent"
        );
        ensure!(
            (0.0..=1.0).contains(&minimum_face_ratio),
            "invalid component face ratio"
        );
        let count = self.faces.len() / 3;
        let mut edges = Vec::with_capacity(self.faces.len());
        for (face, triangle) in self.faces.as_chunks::<3>().0.iter().enumerate() {
            for side in 0..3 {
                let a = triangle[side] as u32;
                let b = triangle[(side + 1) % 3] as u32;
                edges.push(((u64::from(a.min(b)) << 32) | u64::from(a.max(b)), face));
            }
        }
        edges.sort_unstable();
        let mut parent: Vec<_> = (0..count).collect();
        let mut sizes = vec![1usize; count];
        let mut order = vec![usize::MAX; count];
        let mut first = 0;
        while first < edges.len() {
            let mut end = first + 1;
            while end < edges.len() && edges[end].0 == edges[first].0 {
                end += 1;
            }
            if end - first == 2 && edges[first].1 != edges[first + 1].1 {
                let a = edges[first].1;
                let b = edges[first + 1].1;
                order[a] = order[a].min(first);
                order[b] = order[b].min(first);
                let mut a = component_root(&mut parent, a);
                let mut b = component_root(&mut parent, b);
                if a != b {
                    if sizes[a] < sizes[b] {
                        std::mem::swap(&mut a, &mut b);
                    }
                    parent[b] = a;
                    sizes[a] += sizes[b];
                }
            }
            first = end;
        }
        let isolated_start = edges.len();
        drop(edges);
        let mut first_seen = vec![usize::MAX; count];
        for (face, &seen) in order.iter().enumerate() {
            let root = component_root(&mut parent, face);
            let seen = if seen == usize::MAX {
                isolated_start
                    .checked_add(face)
                    .context("component traversal order overflow")?
            } else {
                seen
            };
            first_seen[root] = first_seen[root].min(seen);
        }
        let largest = (0..count)
            .filter(|&face| parent[face] == face)
            .max_by(|&a, &b| {
                sizes[a]
                    .cmp(&sizes[b])
                    .then(first_seen[b].cmp(&first_seen[a]))
            })
            .context("mesh has no connected face component")?;
        let selected = (0..count)
            .filter(|&face| {
                let root = component_root(&mut parent, face);
                root == largest || sizes[root] as f64 / count as f64 >= minimum_face_ratio
            })
            .collect::<Vec<_>>();
        drop(parent);
        drop(sizes);
        drop(order);
        drop(first_seen);
        let mut mapping = vec![-1i32; self.vertices.len() / 3];
        for &face in &selected {
            for &vertex in &self.faces[3 * face..3 * face + 3] {
                mapping[vertex as usize] = 0;
            }
        }
        let mut used = Vec::new();
        let mut vertices = Vec::new();
        for (vertex, destination) in mapping.iter_mut().enumerate() {
            if *destination == 0 {
                *destination =
                    i32::try_from(used.len()).context("compact vertex index overflow")?;
                used.push(vertex);
                vertices.extend_from_slice(&self.vertices[3 * vertex..3 * vertex + 3]);
            }
        }
        let mut faces = Vec::with_capacity(3 * selected.len());
        for &face in &selected {
            faces.extend(
                self.faces[3 * face..3 * face + 3]
                    .iter()
                    .map(|&v| mapping[v as usize]),
            );
        }
        self.vertices = vertices;
        self.faces = faces;
        Ok((selected, used))
    }

    /// F64 arithmetic vertex mean and maximum radial norm; storage returns at the F32 boundary.
    pub fn normalize(&mut self) -> Result<()> {
        self.validate()?;
        let mut vertices = self
            .vertices
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>();
        normalize_vertices(&mut vertices)?;
        for (destination, value) in self.vertices.iter_mut().zip(vertices) {
            *destination = value as f32;
        }
        Ok(())
    }

    pub fn fill_holes(&mut self, capacity: usize) -> Result<usize> {
        self.validate()?;
        let vertices = self.vertices.len() / 3;
        let faces = self.faces.len() / 3;
        let edges = faces
            .checked_mul(3)
            .context("boundary edge extent overflow")?;
        ensure!(
            faces <= capacity && vertices <= u32::MAX as usize,
            "boundary mesh capacity exceeded"
        );
        let edge = |ordinal: usize| {
            let face = ordinal % faces;
            let side = ordinal / faces;
            [
                self.faces[3 * face + side] as u32,
                self.faces[3 * face + (side + 1) % 3] as u32,
            ]
        };
        let mut incidence = (0..edges)
            .map(|ordinal| {
                let [a, b] = edge(ordinal);
                ((u64::from(a.min(b)) << 32) | u64::from(a.max(b)), ordinal)
            })
            .collect::<Vec<_>>();
        incidence.sort_unstable();
        let mut boundary = Vec::new();
        let mut first = 0;
        while first < incidence.len() {
            let mut end = first + 1;
            while end < incidence.len() && incidence[end].0 == incidence[first].0 {
                end += 1;
            }
            if end == first + 1 {
                boundary.push(incidence[first].1);
            }
            first = end;
        }
        drop(incidence);
        boundary.sort_unstable();
        let boundary = boundary.into_iter().map(edge).collect::<Vec<_>>();
        let mut offsets = vec![0usize; vertices + 1];
        for &[a, _] in &boundary {
            offsets[a as usize + 1] += 1;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let mut slots = offsets[..vertices].to_vec();
        let mut outgoing = vec![0; boundary.len()];
        for (index, &[a, _]) in boundary.iter().enumerate() {
            outgoing[slots[a as usize]] = index;
            slots[a as usize] += 1;
        }
        drop(slots);
        let mut used = vec![false; boundary.len()];
        let mut added = Vec::new();
        for start in 0..boundary.len() {
            if used[start] {
                continue;
            }
            let origin = boundary[start][0];
            let mut index = start;
            let mut polygon = Vec::new();
            loop {
                used[index] = true;
                let [a, b] = boundary[index];
                polygon.push(a);
                if b == origin {
                    if (3..=10_000).contains(&polygon.len()) {
                        triangulate_boundary(&self.vertices, &mut polygon, &mut added)?;
                    }
                    break;
                }
                let span = offsets[b as usize]..offsets[b as usize + 1];
                let next = outgoing[span].iter().copied().find(|&i| !used[i]);
                if let Some(next) = next {
                    index = next;
                } else {
                    break;
                }
            }
        }
        let required = faces
            .checked_add(added.len() / 3)
            .context("filled face count overflow")?;
        ensure!(
            required <= capacity,
            "filled mesh needs {required} faces, capacity {capacity}"
        );
        let count = added.len() / 3;
        self.faces.extend(added);
        Ok(count)
    }
}

fn component_root(parent: &mut [usize], mut vertex: usize) -> usize {
    while parent[vertex] != vertex {
        parent[vertex] = parent[parent[vertex]];
        vertex = parent[vertex];
    }
    vertex
}

fn normalize_vertices(vertices: &mut [f64]) -> Result<()> {
    ensure!(
        !vertices.is_empty()
            && vertices.len().is_multiple_of(3)
            && vertices.iter().all(|v| v.is_finite()),
        "invalid normalization vertices"
    );
    let mut mean = [0.0; 3];
    for point in vertices.as_chunks::<3>().0 {
        for axis in 0..3 {
            mean[axis] += point[axis];
        }
    }
    for value in &mut mean {
        *value /= (vertices.len() / 3) as f64;
    }
    let mut radius = 0.0f64;
    for point in vertices.as_chunks::<3>().0 {
        let delta: [f64; 3] = std::array::from_fn(|axis| point[axis] - mean[axis]);
        let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        radius = radius.max(length);
    }
    ensure!(
        radius.is_finite() && radius > 0.0,
        "mesh cannot be normalized to a positive finite radius"
    );
    for point in vertices.as_chunks_mut::<3>().0 {
        for axis in 0..3 {
            point[axis] = (point[axis] - mean[axis]) / radius;
        }
    }
    Ok(())
}

fn triangulate_boundary(
    vertices: &[f32],
    polygon: &mut Vec<u32>,
    output: &mut Vec<i32>,
) -> Result<()> {
    let point =
        |v: u32| <[f32; 3]>::try_from(&vertices[3 * v as usize..3 * v as usize + 3]).unwrap();
    let sub = |a: [f32; 3], b: [f32; 3]| std::array::from_fn::<_, 3, _>(|i| a[i] - b[i]);
    let dot = |a: [f32; 3], b: [f32; 3]| (0.0 + a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    while polygon.len() > 3 {
        let n = polygon.len();
        let mut center = [0f32; 3];
        for &index in polygon.iter() {
            for (sum, value) in center.iter_mut().zip(point(index)) {
                *sum += value;
            }
        }
        for value in &mut center {
            *value /= n as f32;
        }
        let tangents = (0..n)
            .map(|i| {
                let t = sub(point(polygon[(i + 1) % n]), point(polygon[i]));
                let length = dot(t, t).sqrt() + 1e-6;
                t.map(|v| v / length)
            })
            .collect::<Vec<_>>();
        let mut ear = 0;
        let mut best = f32::NEG_INFINITY;
        for i in 0..n {
            let previous = tangents[(i + n - 1) % n];
            let normal = cross(previous.map(|v| -v), tangents[i]);
            let radial = cross(
                sub(point(polygon[(i + 1) % n]), center),
                sub(point(polygon[i]), center),
            );
            let side = dot(radial, normal);
            let sign = if side > 0.0 {
                1.0
            } else if side < 0.0 {
                -1.0
            } else {
                0.0
            };
            let angle = (dot(normal, normal).sqrt() * sign).atan2(-dot(previous, tangents[i]));
            ensure!(angle.is_finite(), "nonfinite boundary angle");
            let score = if angle == 0.0 {
                f32::INFINITY
            } else {
                angle.recip()
            };
            if score > best {
                best = score;
                ear = i;
            }
        }
        output.extend([
            polygon[(ear + 1) % n] as i32,
            polygon[ear] as i32,
            polygon[(ear + n - 1) % n] as i32,
        ]);
        polygon.remove(ear);
    }
    output.extend(polygon.iter().rev().map(|&v| v as i32));
    Ok(())
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
    pub fn unit_ball_frame(&self) -> Result<[[f32; 4]; 4]> {
        let batch = self.coordinates[0][0];
        ensure!(
            self.coordinates.iter().all(|p| p[0] == batch),
            "unit-ball frame requires one sample"
        );
        let mut sum = [0u64; 3];
        for p in &self.coordinates {
            for a in 0..3 {
                sum[a] = sum[a]
                    .checked_add(p[a + 1] as u64)
                    .context("coordinate sum overflow")?;
            }
        }
        let inverse = 1.0f32 / self.coordinates.len() as f32;
        let center = sum.map(|v| v as f32 * inverse);
        let radius = self
            .coordinates
            .iter()
            .map(|p| {
                let d: [f32; 3] = std::array::from_fn(|a| p[a + 1] as f32 - center[a]);
                ((d[0] * d[0] + d[2] * d[2]) + d[1] * d[1]).sqrt()
            })
            .fold(0f32, f32::max)
            + 1e-8;
        ensure!(
            radius.is_finite() && radius > 0.0,
            "invalid unit-ball radius"
        );
        let mut frame = [[0.0; 4]; 4];
        for a in 0..3 {
            frame[a][a] = 1.0 / radius;
            frame[a][3] = -center[a] / radius;
        }
        frame[3][3] = 1.0;
        Ok(frame)
    }

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
    use crate::checkpoint::MappedCheckpoint;
    use std::collections::BTreeMap;
    use std::path::Path;

    fn integers(bank: &MappedCheckpoint, key: &str) -> Result<Vec<i32>> {
        let t = bank.tensor(key)?;
        match t.dtype() {
            safetensors::Dtype::I32 => Ok(t
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b))
                .collect()),
            safetensors::Dtype::I64 => t
                .data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i32::try_from(i64::from_le_bytes(*b)).map_err(Into::into))
                .collect(),
            _ => anyhow::bail!("mesh indices require signed integer storage"),
        }
    }

    fn canonical(faces: &[i32]) -> Vec<[i32; 3]> {
        let mut rows = faces
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| {
                let [a, b, c] = *p;
                [[a, b, c], [b, c, a], [c, a, b]].into_iter().min().unwrap()
            })
            .collect::<Vec<_>>();
        rows.sort_unstable();
        rows
    }

    fn fixture(path: &Path) -> Result<MappedCheckpoint> {
        MappedCheckpoint::open_component(
            path.parent().context("mesh fixture parent missing")?,
            path.file_stem()
                .and_then(|s| s.to_str())
                .context("mesh fixture stem missing")?,
        )
    }

    fn mesh_values(bank: &MappedCheckpoint, key: &str) -> Result<Vec<f64>> {
        let t = bank.tensor(key)?;
        match t.dtype() {
            safetensors::Dtype::F32 => Ok(t
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f64::from(f32::from_le_bytes(*b)))
                .collect()),
            safetensors::Dtype::F64 => Ok(t
                .data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b))
                .collect()),
            _ => anyhow::bail!("mesh vertices require floating point storage"),
        }
    }

    fn compare_normalization(
        root: &Path,
        arm: &str,
        input: (&str, &str),
        expected_file: &str,
        expected_key: &str,
        bounds: &serde_json::Value,
    ) -> Result<bool> {
        let source = fixture(&root.join(arm).join(input.0))?;
        let mut vertices = mesh_values(&source, input.1)?;
        normalize_vertices(&mut vertices)?;
        let reference = fixture(&root.join(arm).join(expected_file))?;
        let expected = mesh_values(&reference, expected_key)?;
        ensure!(
            vertices.len() == expected.len(),
            "normalization extent differs"
        );
        let max_bound = bounds["max_abs"].as_f64().context("max bound missing")?;
        let rms_bound = bounds["RMS"].as_f64().context("RMS bound missing")?;
        let measure = |actual: &[f64]| {
            let mut max = 0.0f64;
            let mut squared = 0.0;
            for (&a, &b) in actual.iter().zip(&expected) {
                let delta = a - b;
                max = max.max(delta.abs());
                squared += delta * delta;
            }
            (max, (squared / actual.len() as f64).sqrt())
        };
        ensure!(
            vertices.iter().all(|v| v.is_finite()),
            "nonfinite normalized mesh"
        );
        let (max, rms) = measure(&vertices);
        let boundary = vertices
            .iter()
            .map(|&v| f64::from(v as f32))
            .collect::<Vec<_>>();
        let (boundary_max, boundary_rms) = measure(&boundary);
        let control = mesh_values(&reference, "aabb_control")?;
        ensure!(
            control.len() == expected.len(),
            "normalization control extent differs"
        );
        let (control_max, control_rms) = measure(&control);
        ensure!(
            control_max > 10.0 * max_bound && control_rms > 10.0 * rms_bound,
            "normalization control does not discriminate"
        );
        let pass = max <= max_bound
            && rms <= rms_bound
            && boundary_max <= max_bound
            && boundary_rms <= rms_bound;
        println!(
            "{} N5_M6_NORMALIZE arm={arm} vertices={} max_abs={max:e} RMS={rms:e} boundary_max_abs={boundary_max:e} boundary_RMS={boundary_rms:e} tau_max={max_bound:e} tau_RMS={rms_bound:e} AABB_control_max={control_max:e} AABB_control_RMS={control_rms:e}",
            if pass { "PASS" } else { "DEFECT" },
            vertices.len() / 3
        );
        Ok(pass)
    }

    fn compare_component_fixture(root: &Path, file: &str) -> Result<bool> {
        let bank = fixture(&root.join(file))?;
        let source_key = if file.ends_with("components.safetensors") && !file.contains("corners") {
            "vertices"
        } else {
            "input_vertices"
        };
        let face_key = if source_key == "vertices" {
            "faces"
        } else {
            "input_faces"
        };
        let mut mesh = SparseMesh {
            vertices: mesh_values(&bank, source_key)?
                .iter()
                .map(|&v| v as f32)
                .collect(),
            faces: integers(&bank, face_key)?,
        };
        let (selected, _) = mesh.filter_components(0.005)?;
        let expected_vertices = mesh_values(&bank, "filtered_vertices")?
            .iter()
            .map(|&v| v as f32)
            .collect::<Vec<_>>();
        let mut pass =
            mesh.faces == integers(&bank, "filtered_faces")? && mesh.vertices == expected_vertices;
        if source_key == "vertices" {
            pass &= selected
                == integers(&bank, "selected_face_ids")?
                    .iter()
                    .map(|&v| v as usize)
                    .collect::<Vec<_>>();
            let control = integers(&bank, "vertex_control_face_ids")?;
            ensure!(
                control.len() == selected.len() + 4,
                "shared-vertex control does not discriminate"
            );
        }
        println!(
            "{} N5_M6_COMPONENTS fixture={file} retained_faces={} selected_first={:?} shared_vertex_control_extra_faces={}",
            if pass { "PASS" } else { "DEFECT" },
            mesh.faces.len() / 3,
            selected.first(),
            if source_key == "vertices" { 4 } else { 0 }
        );
        Ok(pass)
    }

    #[test]
    #[ignore = "requires frozen final-fill, component and normalization fixtures; CPU only"]
    fn sparse_mesh_postprocess_definition_cpu() -> Result<()> {
        let root = std::env::var_os("FF_HI3D_POSTPROCESS_REFERENCE")
            .context("postprocess fixtures required")?;
        let root = Path::new(&root);
        let frozen: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("FROZEN.json"))?)?;
        let mut defects = 0;
        let mut records = 0;
        for case in ["cottage512", "robot512", "cottage1024", "robot1536"] {
            let arm = format!("local-{case}-r0");
            let path = frozen["production"][case]["local"]["input"]
                .as_str()
                .context("raw mesh path missing")?;
            let raw = fixture(Path::new(path))?;
            let raw_vertices = mesh_values(&raw, "vertices")?
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<_>>();
            let original = integers(&raw, "faces")?;
            let fill = fixture(&root.join(&arm).join("fill-added.safetensors"))?;
            let expected_fill = integers(&fill, "added_faces")?;
            let mut mesh = SparseMesh {
                vertices: raw_vertices,
                faces: original.clone(),
            };
            let added = mesh.fill_holes((original.len() + expected_fill.len()) / 3)?;
            let exact = canonical(&mesh.faces[original.len()..]) == canonical(&expected_fill);
            ensure!(
                mesh.faces[..original.len()] == original,
                "final fill changed original faces"
            );
            defects += usize::from(!exact);
            records += 1;
            println!(
                "{} N5_M6_FINAL_FILL case={case} added_faces={added} expected_faces={} oriented_multiset_exact={exact}",
                if exact { "PASS" } else { "DEFECT" },
                expected_fill.len() / 3
            );
            mesh.faces = original;
            mesh.faces.extend(expected_fill);
            let (selected, used) = mesh.filter_components(0.005)?;
            let expected = fixture(&root.join(&arm).join("filter-maps.safetensors"))?;
            let exact = selected
                == integers(&expected, "selected_filled_face_ids")?
                    .iter()
                    .map(|&v| v as usize)
                    .collect::<Vec<_>>()
                && used
                    == integers(&expected, "used_original_vertex_ids")?
                        .iter()
                        .map(|&v| v as usize)
                        .collect::<Vec<_>>();
            defects += usize::from(!exact);
            records += 1;
            println!(
                "{} N5_M6_FINAL_COMPONENTS case={case} vertices={} faces={} exact_maps={exact} common_filled_input=true",
                if exact { "PASS" } else { "DEFECT" },
                mesh.vertices.len() / 3,
                mesh.faces.len() / 3
            );
            drop(mesh);
            let bounds = &frozen["float_cells"][format!("{case}.normalize")]["tau"];
            defects += usize::from(!compare_normalization(
                root,
                &arm,
                ("simplified-common.safetensors", "vertices"),
                "normalization-common.safetensors",
                "vertices_f64",
                bounds,
            )?);
            records += 1;
        }
        for file in [
            "local-bounded-r1/components.safetensors",
            "local-corners-r1/201_equal_components.safetensors",
            "local-corners-r1/201_equal_connected_components_reverse_vertices.safetensors",
            "local-corners-r1/nonmanifold_valence3.safetensors",
            "local-corners-r1/duplicate_reversed_degenerate.safetensors",
        ] {
            defects += usize::from(!compare_component_fixture(root, file)?);
            records += 1;
        }
        for case in [
            "triangle_boundary",
            "pyramid_boundary5",
            "loop10001_skipped",
        ] {
            let source = fixture(
                &root
                    .join("local-corners-r1")
                    .join(format!("{case}.safetensors")),
            )?;
            let mut mesh = SparseMesh {
                vertices: mesh_values(&source, "input_vertices")?
                    .iter()
                    .map(|&v| v as f32)
                    .collect(),
                faces: integers(&source, "input_faces")?,
            };
            let original = mesh.faces.clone();
            let expected = integers(&source, "added_faces")?;
            mesh.fill_holes((original.len() + expected.len()) / 3)?;
            let exact = canonical(&mesh.faces[original.len()..]) == canonical(&expected);
            let reversed = canonical(&integers(&source, "reversed_ear_control")?);
            ensure!(
                expected.is_empty() || canonical(&expected) != reversed,
                "reversed-ear control does not discriminate"
            );
            defects += usize::from(!exact);
            records += 1;
            println!(
                "{} N5_M6_FINAL_FILL_CORNER case={case} added_faces={} reversed_ear_discriminates={}",
                if exact { "PASS" } else { "DEFECT" },
                expected.len() / 3,
                !expected.is_empty()
            );
        }
        defects += usize::from(!compare_normalization(
            root,
            "local-bounded-r1",
            ("normalization.safetensors", "input_vertices"),
            "normalization.safetensors",
            "normalized_f64",
            &frozen["float_cells"]["bounded.normalize"]["tau"],
        )?);
        records += 1;
        println!(
            "N5_M6_POSTPROCESS_GATE records={records} defects={defects} CPU_only=true decimator=false timing=false"
        );
        ensure!(defects == 0, "final mesh operator defects");
        Ok(())
    }

    #[test]
    fn mesh_normalization_preserves_faces_and_rejects_invalid_buffers() -> Result<()> {
        let mut mesh = SparseMesh {
            vertices: vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            faces: vec![0, 1, 2],
        };
        let original = mesh.faces.clone();
        mesh.normalize()?;
        assert_eq!(mesh.faces, original);
        for axis in 0..3 {
            assert!(
                mesh.vertices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|p| f64::from(p[axis]))
                    .sum::<f64>()
                    .abs()
                    < 1e-6
            );
        }
        let mut degenerate = SparseMesh {
            vertices: vec![1.0; 9],
            faces: original.clone(),
        };
        assert!(degenerate.normalize().is_err());
        assert_eq!(degenerate.vertices, vec![1.0; 9]);
        let mut invalid = SparseMesh {
            vertices: vec![0.0; 9],
            faces: vec![-1, 1, 2],
        };
        assert!(invalid.filter_components(0.005).is_err());
        assert!(invalid.normalize().is_err());
        assert!(mesh.filter_components(f64::NAN).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires frozen raw meshes and oriented hole-fill faces; CPU only"]
    fn sparse_mesh_fill_definition_cpu() -> Result<()> {
        let root =
            std::env::var_os("FF_HI3D_CASCADE_REFERENCE").context("cascade fixtures required")?;
        let prefix = std::env::var_os("FF_HI3D_EXTRACTION_REFERENCE")
            .context("raw mesh fixtures required")?;
        let root = Path::new(&root);
        let prefix = Path::new(&prefix);
        let mut records = 0;
        let mut defects = 0;
        for high in [false, true] {
            let anchor_dir = root.join(if high {
                "transition1024-sm86-released-r1"
            } else {
                "transition-sm86-full-r1"
            });
            for case in ["open_loop", "cottage", "robot_crab"] {
                let anchor =
                    MappedCheckpoint::open_component(&anchor_dir, &format!("{case}-fill"))?;
                let source = if case == "open_loop" {
                    anchor_dir.clone()
                } else if high {
                    root.join(format!("continued1024-{}-sm86-r1", case.replace('_', "-")))
                } else {
                    prefix.join(format!("linked-sm89-{case}-r1"))
                };
                let mapped = MappedCheckpoint::open_component(
                    &source,
                    if case == "open_loop" {
                        "open_loop-fill"
                    } else {
                        "mesh"
                    },
                )?;
                let t = mapped.tensor(if case == "open_loop" {
                    "input_vertices"
                } else {
                    "vertices"
                })?;
                ensure!(
                    t.dtype() == safetensors::Dtype::F32
                        && t.shape().len() == 2
                        && t.shape()[1] == 3,
                    "mesh vertex layout differs"
                );
                let vertices = t
                    .data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect::<Vec<_>>();
                let original = if case == "open_loop" {
                    (0..7).flat_map(|i| [7, (i + 1) % 7, i]).collect::<Vec<_>>()
                } else {
                    integers(&mapped, "faces")?
                };
                let expected = integers(&anchor, "added_faces")?;
                let capacity = original.len() / 3 + expected.len() / 3;
                let mut mesh = SparseMesh {
                    vertices,
                    faces: original.clone(),
                };
                let count = mesh.fill_holes(capacity)?;
                let actual_faces = canonical(&mesh.faces[original.len()..]);
                let expected_faces = canonical(&expected);
                let exact = actual_faces == expected_faces;
                if !exact {
                    let actual = actual_faces
                        .iter()
                        .copied()
                        .collect::<std::collections::BTreeSet<_>>();
                    let expected = expected_faces
                        .iter()
                        .copied()
                        .collect::<std::collections::BTreeSet<_>>();
                    println!(
                        "N5_M4_FILL_DIAGNOSTIC case={case} high={high} missing={} extra={} first_missing={:?} first_extra={:?}",
                        expected.difference(&actual).count(),
                        actual.difference(&expected).count(),
                        expected.difference(&actual).take(8).collect::<Vec<_>>(),
                        actual.difference(&expected).take(8).collect::<Vec<_>>()
                    );
                }
                defects += usize::from(!exact);
                ensure!(
                    mesh.faces[..original.len()] == original,
                    "fill changed original faces"
                );
                println!(
                    "{} N5_M4_FILL case={case} high={high} original_faces={} added_faces={count} expected_faces={} oriented_exact={exact} skip_fill_control={}",
                    if exact { "PASS" } else { "DEFECT" },
                    original.len() / 3,
                    expected.len() / 3,
                    !expected.is_empty()
                );
                if case == "open_loop" {
                    ensure!(
                        count == 5 && capacity > 0,
                        "skip-fill control failed to discriminate"
                    );
                    mesh.faces.truncate(original.len());
                    ensure!(
                        mesh.fill_holes(capacity - 1).is_err() && mesh.faces == original,
                        "short fill capacity wrote output"
                    );
                }
                records += 1;
            }
        }
        println!(
            "N5_M4_FILL_GATE records={records} defects={defects} short_capacity=rejected timing=false"
        );
        ensure!(defects == 0, "oriented boundary fill defects");
        Ok(())
    }

    struct ClosedCubeMesh {
        components: Vec<[i8; 12]>,
        quads: Vec<[usize; 4]>,
        faces: Vec<[usize; 3]>,
    }

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

    fn closed_case(table: &[[i8; 14]; 256], mask: usize, flip: bool) -> Result<ClosedCubeMesh> {
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
        let mut owners: BTreeMap<_, Vec<_>> = BTreeMap::new();
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
        Ok(ClosedCubeMesh {
            components: labels,
            quads,
            faces,
        })
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
            let first = closed_case(&table, mask, false)?;
            let second = closed_case(&table, mask, true)?;
            assert!(closed_winding(&first.faces) && closed_winding(&second.faces));
            if first.components == second.components {
                agree += 1;
                let mut forward: Vec<_> = first.quads.into_iter().map(oriented).collect();
                let mut reverse: Vec<_> = second
                    .quads
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
                let mut control = first.faces;
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
