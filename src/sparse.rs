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

    /// Inherited plane quadrics and manifold links; returned maps index original faces and vertices.
    pub fn simplify(&mut self, target_faces: usize) -> Result<(Vec<usize>, Vec<usize>)> {
        self.validate()?;
        ensure!(target_faces > 0, "simplification target must be positive");
        if self.faces.len() / 3 <= target_faces {
            return Ok((
                (0..self.faces.len() / 3).collect(),
                (0..self.vertices.len() / 3).collect(),
            ));
        }
        let mut state = MeshDecimator::new(self)?;
        state.run(target_faces, |_, _, _, _, _| Ok(()))?;
        let (mesh, ancestry, vertices) = state.finish()?;
        *self = mesh;
        Ok((ancestry, vertices))
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

#[derive(Clone, Copy, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    fn plane(p: [f64; 4]) -> Self {
        Self([
            p[0] * p[0],
            p[0] * p[1],
            p[0] * p[2],
            p[0] * p[3],
            p[1] * p[1],
            p[1] * p[2],
            p[1] * p[3],
            p[2] * p[2],
            p[2] * p[3],
            p[3] * p[3],
        ])
    }

    fn add(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] + other.0[i]))
    }

    fn energy(self, [x, y, z]: [f64; 3]) -> f64 {
        let q = self.0;
        q[0] * x * x
            + 2.0 * q[1] * x * y
            + 2.0 * q[2] * x * z
            + 2.0 * q[3] * x
            + q[4] * y * y
            + 2.0 * q[5] * y * z
            + 2.0 * q[6] * y
            + q[7] * z * z
            + 2.0 * q[8] * z
            + q[9]
    }

    fn matrix(self, v: [f64; 3]) -> [f64; 3] {
        let q = self.0;
        [
            q[0] * v[0] + q[1] * v[1] + q[2] * v[2],
            q[1] * v[0] + q[4] * v[1] + q[5] * v[2],
            q[2] * v[0] + q[5] * v[1] + q[7] * v[2],
        ]
    }

    fn stationary(self) -> Option<[f64; 3]> {
        let q = self.0;
        let mut a = [
            [q[0], q[1], q[2], -q[3]],
            [q[1], q[4], q[5], -q[6]],
            [q[2], q[5], q[7], -q[8]],
        ];
        let scale = [q[0].abs(), q[4].abs(), q[7].abs()]
            .into_iter()
            .fold(0.0, f64::max);
        for column in 0..3 {
            let pivot =
                (column..3).max_by(|&i, &j| a[i][column].abs().total_cmp(&a[j][column].abs()))?;
            if a[pivot][column].abs() <= 1e-12 * scale {
                return None;
            }
            a.swap(column, pivot);
            let divisor = a[column][column];
            for value in &mut a[column][column..] {
                *value /= divisor;
            }
            let pivot_row = a[column];
            for (row, values) in a.iter_mut().enumerate() {
                if row != column {
                    let factor = values[column];
                    for (value, &pivot) in values[column..].iter_mut().zip(&pivot_row[column..]) {
                        *value -= factor * pivot;
                    }
                }
            }
        }
        let point = [a[0][3], a[1][3], a[2][3]];
        point.iter().all(|v| v.is_finite()).then_some(point)
    }

    fn segment(self, a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        let d = vector_sub(b, a);
        let qa = self.matrix(a);
        let numerator = vector_dot(d, [qa[0] + self.0[3], qa[1] + self.0[6], qa[2] + self.0[8]]);
        let denominator = vector_dot(d, self.matrix(d));
        let t = if denominator > 0.0 {
            (-numerator / denominator).clamp(0.0, 1.0)
        } else {
            0.0
        };
        std::array::from_fn(|i| a[i] + t * d[i])
    }

    fn candidates(self, a: [f64; 3], b: [f64; 3]) -> [Option<[f64; 3]>; 5] {
        [
            self.stationary(),
            Some(self.segment(a, b)),
            Some(a),
            Some(b),
            Some(std::array::from_fn(|i| (a[i] + b[i]) * 0.5)),
        ]
    }

    fn lower_bound(self, a: [f64; 3], b: [f64; 3]) -> f64 {
        self.candidates(a, b)
            .into_iter()
            .flatten()
            .map(|p| self.energy(p).max(0.0))
            .fold(f64::INFINITY, f64::min)
    }
}

fn vector_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}

fn vector_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vector_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn corner_next(c: u32) -> u32 {
    c / 3 * 3 + (c % 3 + 1) % 3
}

struct EdgeQueue {
    heap: Vec<u32>,
    positions: Vec<u32>,
    costs: Vec<f64>,
    peak: usize,
}

impl EdgeQueue {
    fn new(corners: usize) -> Self {
        Self {
            heap: Vec::new(),
            positions: vec![u32::MAX; corners],
            costs: vec![0.0; corners],
            peak: 0,
        }
    }

    fn less(&self, a: u32, b: u32) -> bool {
        self.costs[a as usize]
            .total_cmp(&self.costs[b as usize])
            .then(a.cmp(&b))
            .is_lt()
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.heap.swap(a, b);
        self.positions[self.heap[a] as usize] = a as u32;
        self.positions[self.heap[b] as usize] = b as u32;
    }

    fn up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if !self.less(self.heap[i], self.heap[parent]) {
                break;
            }
            self.swap(i, parent);
            i = parent;
        }
    }

    fn down(&mut self, mut i: usize) {
        loop {
            let left = i * 2 + 1;
            if left >= self.heap.len() {
                break;
            }
            let right = left + 1;
            let child = if right < self.heap.len() && self.less(self.heap[right], self.heap[left]) {
                right
            } else {
                left
            };
            if !self.less(self.heap[child], self.heap[i]) {
                break;
            }
            self.swap(i, child);
            i = child;
        }
    }

    fn set(&mut self, corner: u32, cost: f64) {
        let old = self.positions[corner as usize];
        self.costs[corner as usize] = cost;
        let i = if old == u32::MAX {
            let i = self.heap.len();
            self.positions[corner as usize] = i as u32;
            self.heap.push(corner);
            self.peak = self.peak.max(self.heap.len());
            i
        } else {
            old as usize
        };
        if i > 0 && self.less(self.heap[i], self.heap[(i - 1) / 2]) {
            self.up(i);
        } else {
            self.down(i);
        }
    }

    fn remove(&mut self, corner: u32) {
        let index = self.positions[corner as usize];
        if index == u32::MAX {
            return;
        }
        let i = index as usize;
        self.heap.swap_remove(i);
        self.positions[corner as usize] = u32::MAX;
        if i < self.heap.len() {
            self.positions[self.heap[i] as usize] = i as u32;
            if i > 0 && self.less(self.heap[i], self.heap[(i - 1) / 2]) {
                self.up(i);
            } else {
                self.down(i);
            }
        }
    }

    fn pop(&mut self) -> Option<(u32, f64)> {
        let corner = *self.heap.first()?;
        let cost = self.costs[corner as usize];
        self.remove(corner);
        Some((corner, cost))
    }
}

struct MeshDecimator {
    points: Vec<[f64; 3]>,
    faces: Vec<[u32; 3]>,
    normals: Vec<[f64; 3]>,
    quadrics: Vec<Quadric>,
    heads: Vec<u32>,
    next: Vec<u32>,
    previous: Vec<u32>,
    protected: Vec<bool>,
    queue: EdgeQueue,
    live_faces: usize,
    contractions: usize,
}

impl MeshDecimator {
    fn new(mesh: &SparseMesh) -> Result<Self> {
        ensure!(
            mesh.faces.len() < u32::MAX as usize,
            "mesh corner capacity exceeds u32"
        );
        let points: Vec<_> = mesh
            .vertices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| p.map(f64::from))
            .collect();
        let faces: Vec<_> = mesh
            .faces
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| p.map(|v| v as u32))
            .collect();
        let mut state = Self {
            normals: vec![[0.0; 3]; faces.len()],
            quadrics: vec![Quadric::default(); points.len()],
            heads: vec![u32::MAX; points.len()],
            next: vec![u32::MAX; mesh.faces.len()],
            previous: vec![u32::MAX; mesh.faces.len()],
            protected: vec![false; points.len()],
            queue: EdgeQueue::new(mesh.faces.len()),
            live_faces: faces.len(),
            contractions: 0,
            points,
            faces,
        };
        let edges = state.initialize()?;
        let mut first = 0;
        while first < edges.len() {
            let mut end = first + 1;
            while end < edges.len() && edges[end].0 == edges[first].0 {
                end += 1;
            }
            if end - first == 2 {
                let corner = edges[first..end].iter().map(|e| e.1).min().unwrap();
                state.enqueue(corner);
            }
            first = end;
        }
        ensure!(
            state.queue.heap.len() <= mesh.faces.len(),
            "live edge queue capacity exceeded"
        );
        Ok(state)
    }

    fn initialize(&mut self) -> Result<Vec<(u64, u32)>> {
        let mut edges = Vec::with_capacity(self.next.len());
        for face in 0..self.faces.len() {
            let triangle = self.faces[face];
            let [a, b, c] = triangle.map(|v| self.points[v as usize]);
            let normal = vector_cross(vector_sub(b, a), vector_sub(c, a));
            let length = vector_dot(normal, normal).sqrt();
            ensure!(length.is_finite(), "nonfinite initial face plane");
            if length > 0.0 {
                let n = normal.map(|v| v / length);
                self.normals[face] = n;
                let q = Quadric::plane([n[0], n[1], n[2], -vector_dot(n, a)]);
                for &v in &triangle {
                    self.quadrics[v as usize] = self.quadrics[v as usize].add(q);
                }
            } else {
                for &v in &triangle {
                    self.protected[v as usize] = true;
                }
            }
            for side in 0..3 {
                let corner = (3 * face + side) as u32;
                self.link(corner);
                let a = triangle[side];
                let b = triangle[(side + 1) % 3];
                edges.push(((u64::from(a.min(b)) << 32) | u64::from(a.max(b)), corner));
            }
        }
        edges.sort_unstable();
        let mut opposite = vec![u32::MAX; self.next.len()];
        let mut first = 0;
        while first < edges.len() {
            let mut end = first + 1;
            while end < edges.len() && edges[end].0 == edges[first].0 {
                end += 1;
            }
            let c = edges[first].1;
            if end - first == 2 && self.vertex(c) != self.vertex(edges[first + 1].1) {
                opposite[c as usize] = edges[first + 1].1;
                opposite[edges[first + 1].1 as usize] = c;
            } else {
                let a = self.vertex(c) as usize;
                let b = self.vertex(corner_next(c)) as usize;
                self.protected[a] = true;
                self.protected[b] = true;
            }
            first = end;
        }
        for vertex in 0..self.points.len() {
            if self.protected[vertex] || self.heads[vertex] == u32::MAX {
                continue;
            }
            let start = self.heads[vertex];
            let degree = self.incident(vertex as u32).len();
            let mut corner = start;
            let mut count = 0;
            loop {
                let next = opposite[corner as usize];
                if next == u32::MAX {
                    break;
                }
                corner = corner_next(next);
                count += 1;
                if corner == start || count >= degree {
                    break;
                }
            }
            if corner != start || count != degree {
                self.protected[vertex] = true;
            }
        }
        Ok(edges)
    }

    fn vertex(&self, corner: u32) -> u32 {
        self.faces[corner as usize / 3][corner as usize % 3]
    }

    fn link(&mut self, corner: u32) {
        let vertex = self.vertex(corner) as usize;
        let old = self.heads[vertex];
        self.next[corner as usize] = old;
        self.previous[corner as usize] = u32::MAX;
        if old != u32::MAX {
            self.previous[old as usize] = corner;
        }
        self.heads[vertex] = corner;
    }

    fn unlink(&mut self, corner: u32) {
        let vertex = self.vertex(corner) as usize;
        let previous = self.previous[corner as usize];
        let next = self.next[corner as usize];
        if previous == u32::MAX {
            self.heads[vertex] = next;
        } else {
            self.next[previous as usize] = next;
        }
        if next != u32::MAX {
            self.previous[next as usize] = previous;
        }
    }

    fn incident(&self, vertex: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let mut corner = self.heads[vertex as usize];
        while corner != u32::MAX {
            out.push(corner);
            corner = self.next[corner as usize];
        }
        out
    }

    fn edge_slot(&self, a: u32, b: u32) -> Option<u32> {
        let mut best = None;
        let mut corner = self.heads[a as usize];
        while corner != u32::MAX {
            let next = corner_next(corner);
            let previous = corner_next(next);
            let edge = if self.vertex(next) == b {
                Some(corner)
            } else if self.vertex(previous) == b {
                Some(previous)
            } else {
                None
            };
            if let Some(edge) = edge {
                best = Some(best.map_or(edge, |old: u32| old.min(edge)));
            }
            corner = self.next[corner as usize];
        }
        best
    }

    fn enqueue(&mut self, corner: u32) {
        let a = self.vertex(corner);
        let b = self.vertex(corner_next(corner));
        if self.protected[a as usize] || self.protected[b as usize] {
            return;
        }
        let q = self.quadrics[a as usize].add(self.quadrics[b as usize]);
        let cost = q.lower_bound(self.points[a as usize], self.points[b as usize]);
        self.queue.set(corner, cost);
    }

    fn star(&self, a: u32, b: u32) -> Vec<u32> {
        let mut faces = self.incident(a);
        faces.extend(self.incident(b));
        for c in &mut faces {
            *c /= 3;
        }
        faces.sort_unstable();
        faces.dedup();
        faces
    }

    fn link_valid(&self, a: u32, b: u32, star: &[u32]) -> bool {
        let mut neighbors = [Vec::new(), Vec::new()];
        let mut opposite = Vec::new();
        let mut remaining = Vec::new();
        for &face in star {
            let triangle = self.faces[face as usize];
            for (i, v) in [a, b].into_iter().enumerate() {
                if triangle.contains(&v) {
                    neighbors[i].extend(triangle.into_iter().filter(|&n| n != v));
                }
            }
            if triangle.contains(&a) && triangle.contains(&b) {
                opposite.extend(triangle.into_iter().filter(|&v| v != a && v != b));
            } else {
                let mut mapped = triangle.map(|v| if v == b { a } else { v });
                mapped.sort_unstable();
                remaining.push(mapped);
            }
        }
        if opposite.len() != 2 || opposite[0] == opposite[1] {
            return false;
        }
        for row in &mut neighbors {
            row.sort_unstable();
            row.dedup();
        }
        let common: Vec<_> = neighbors[0]
            .iter()
            .copied()
            .filter(|v| neighbors[1].binary_search(v).is_ok())
            .collect();
        opposite.sort_unstable();
        remaining.sort_unstable();
        common == opposite && !remaining.windows(2).any(|p| p[0] == p[1])
    }

    fn position_valid(&self, a: u32, b: u32, point: [f64; 3], star: &[u32]) -> bool {
        if !point.iter().all(|v| v.is_finite()) {
            return false;
        }
        for &face in star {
            let triangle = self.faces[face as usize];
            if triangle.contains(&a) && triangle.contains(&b) {
                continue;
            }
            let old = triangle.map(|v| self.points[v as usize]);
            let new = triangle.map(|v| {
                if v == a || v == b {
                    point
                } else {
                    self.points[v as usize]
                }
            });
            let normal = vector_cross(vector_sub(new[1], new[0]), vector_sub(new[2], new[0]));
            let length = vector_dot(normal, normal).sqrt();
            let previous = vector_cross(vector_sub(old[1], old[0]), vector_sub(old[2], old[0]));
            if !length.is_finite()
                || length <= 0.0
                || vector_dot(normal, self.normals[face as usize]) / length < 0.5
                || vector_dot(normal, previous) <= 0.0
            {
                return false;
            }
        }
        true
    }

    fn collapse(&mut self, a: u32, b: u32, point: [f64; 3], star: &[u32]) {
        let mut vertices = Vec::new();
        for &f in star {
            vertices.extend(self.faces[f as usize]);
        }
        vertices.sort_unstable();
        vertices.dedup();
        let mut neighborhood = Vec::new();
        for &v in &vertices {
            neighborhood.extend(self.incident(v).into_iter().map(|c| c / 3));
        }
        neighborhood.sort_unstable();
        neighborhood.dedup();
        for &f in &neighborhood {
            for side in 0..3 {
                self.queue.remove(3 * f + side);
            }
        }
        for &f in star {
            for side in 0..3 {
                self.unlink(3 * f + side);
            }
        }
        self.points[a as usize] = point;
        self.quadrics[a as usize] = self.quadrics[a as usize].add(self.quadrics[b as usize]);
        for &f in star {
            let triangle = self.faces[f as usize];
            if triangle.contains(&a) && triangle.contains(&b) {
                self.faces[f as usize] = [u32::MAX; 3];
                self.live_faces -= 1;
            } else {
                self.faces[f as usize] = triangle.map(|v| if v == b { a } else { v });
                for side in 0..3 {
                    self.link(3 * f + side);
                }
            }
        }
        let mut slots = Vec::new();
        for &f in &neighborhood {
            if self.faces[f as usize][0] == u32::MAX {
                continue;
            }
            for side in 0..3 {
                let c = 3 * f + side;
                let a = self.vertex(c);
                let b = self.vertex(corner_next(c));
                if let Some(slot) = self.edge_slot(a, b) {
                    slots.push(slot);
                }
            }
        }
        slots.sort_unstable();
        slots.dedup();
        for slot in slots {
            self.enqueue(slot);
        }
        self.contractions += 1;
    }

    fn run(
        &mut self,
        target: usize,
        mut audit: impl FnMut(&Self, u32, u32, [f64; 3], &[u32]) -> Result<()>,
    ) -> Result<()> {
        while self.live_faces > target {
            let (corner, lower) = self.queue.pop().with_context(|| {
                format!(
                    "simplification legal queue exhausted at {} faces, target {target}",
                    self.live_faces
                )
            })?;
            let x = self.vertex(corner);
            let y = self.vertex(corner_next(corner));
            let (a, b) = (x.min(y), x.max(y));
            let star = self.star(a, b);
            if !self.link_valid(a, b, &star) {
                continue;
            }
            let quadric = self.quadrics[a as usize].add(self.quadrics[b as usize]);
            let point = quadric
                .candidates(self.points[a as usize], self.points[b as usize])
                .into_iter()
                .flatten()
                .map(|p| p.map(|x| f64::from(x as f32)))
                .filter(|&p| self.position_valid(a, b, p, &star))
                .min_by(|&p, &q| quadric.energy(p).total_cmp(&quadric.energy(q)));
            let Some(point) = point else {
                continue;
            };
            let cost = quadric.energy(point).max(0.0);
            ensure!(cost.is_finite(), "nonfinite contraction cost");
            if cost > lower
                && self.queue.heap.first().is_some_and(|&next| {
                    self.queue.costs[next as usize]
                        .total_cmp(&cost)
                        .then(next.cmp(&corner))
                        .is_lt()
                })
            {
                self.queue.set(corner, cost);
                continue;
            }
            audit(self, a, b, point, &star)?;
            self.collapse(a, b, point, &star);
            if self.contractions.is_multiple_of(250_000) {
                println!(
                    "MESH_QEM_PROGRESS contractions={} faces={} queue={} queue_peak={} timing=false",
                    self.contractions,
                    self.live_faces,
                    self.queue.heap.len(),
                    self.queue.peak
                );
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<(SparseMesh, Vec<usize>, Vec<usize>)> {
        ensure!(
            self.queue.peak <= self.next.len(),
            "live queue exceeded original corner capacity"
        );
        let mut used = vec![false; self.points.len()];
        for triangle in &self.faces {
            if triangle[0] != u32::MAX {
                for &v in triangle {
                    used[v as usize] = true;
                }
            }
        }
        let mut mapping = vec![-1i32; self.points.len()];
        let mut vertices = Vec::new();
        let mut original_vertices = Vec::new();
        for (v, &live) in used.iter().enumerate() {
            if live {
                mapping[v] = i32::try_from(vertices.len() / 3)
                    .context("simplified vertex capacity overflow")?;
                vertices.extend(self.points[v].map(|x| x as f32));
                original_vertices.push(v);
            }
        }
        let mut faces = Vec::with_capacity(3 * self.live_faces);
        let mut ancestry = Vec::with_capacity(self.live_faces);
        for (f, triangle) in self.faces.iter().enumerate() {
            if triangle[0] != u32::MAX {
                faces.extend(triangle.map(|v| mapping[v as usize]));
                ancestry.push(f);
            }
        }
        let mesh = SparseMesh { vertices, faces };
        mesh.validate()?;
        Ok((mesh, ancestry, original_vertices))
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

    fn full_quadric(q: Quadric) -> [f64; 16] {
        let q = q.0;
        [
            q[0], q[1], q[2], q[3], q[1], q[4], q[5], q[6], q[2], q[5], q[7], q[8], q[3], q[6],
            q[8], q[9],
        ]
    }

    fn qem_initial(points: &[[f64; 3]], faces: &[[i32; 3]]) -> Vec<Quadric> {
        let mut q = vec![Quadric::default(); points.len()];
        for triangle in faces {
            let [a, b, c] = triangle.map(|v| points[v as usize]);
            let n = vector_cross(vector_sub(b, a), vector_sub(c, a));
            let length = vector_dot(n, n).sqrt();
            let n = n.map(|v| v / length);
            let plane = Quadric::plane([n[0], n[1], n[2], -vector_dot(n, a)]);
            for &v in triangle {
                q[v as usize] = q[v as usize].add(plane);
            }
        }
        q
    }

    fn qem_wrong(mut q: Quadric) -> Quadric {
        for i in [1, 2, 3, 5, 6, 8] {
            q.0[i] *= 0.5;
        }
        q
    }

    fn qem_float(
        name: &str,
        actual: &[f64],
        expected: &[f64],
        control: &[f64],
        field: &serde_json::Value,
    ) -> Result<usize> {
        ensure!(
            actual.len() == expected.len() && actual.len() == control.len() && !actual.is_empty(),
            "quadric field extent mismatch"
        );
        ensure!(
            actual
                .iter()
                .chain(expected)
                .chain(control)
                .all(|v| v.is_finite()),
            "nonfinite quadric field"
        );
        let stats = |values: &[f64]| {
            let delta: Vec<_> = values
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .collect();
            [
                delta.iter().copied().fold(0.0, f64::max),
                (delta.iter().map(|v| v * v).sum::<f64>() / delta.len() as f64).sqrt(),
            ]
        };
        let error = stats(actual);
        let bad = stats(control);
        let tau = [
            field["tau"]["max_abs"]
                .as_f64()
                .context("QEM max floor missing")?,
            field["tau"]["RMS"]
                .as_f64()
                .context("QEM RMS floor missing")?,
        ];
        ensure!(
            (0..2).all(|i| bad[i] > 10.0 * tau[i]),
            "QEM control does not discriminate {name}"
        );
        let defects = usize::from((0..2).any(|i| error[i] > tau[i]));
        println!(
            "N5_M6_QEM_FLOAT cell={name} max_abs={} RMS={} control_max_multiple={} control_RMS_multiple={} defects={defects}",
            error[0],
            error[1],
            bad[0] / tau[0],
            bad[1] / tau[1]
        );
        Ok(defects)
    }

    fn qem_reference() -> Result<(std::path::PathBuf, serde_json::Value)> {
        let root = std::path::PathBuf::from(
            std::env::var("FF_HI3D_QEM_REFERENCE").context("FF_HI3D_QEM_REFERENCE required")?,
        );
        let frozen = serde_json::from_slice(&std::fs::read(root.join("FROZEN.json"))?)?;
        Ok((root, frozen))
    }

    fn qem_resources() -> Result<()> {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        ensure!(
            unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0,
            "mesh gate getrusage failed"
        );
        let usage = unsafe { usage.assume_init() };
        println!(
            "N5_M6_QEM_HOST ru_maxrss_KiB={} GPU_runs=0 timing=false",
            usage.ru_maxrss
        );
        Ok(())
    }

    #[test]
    fn mesh_qem_queue_and_link_indices() -> Result<()> {
        let mut queue = EdgeQueue::new(97);
        let mut expected = BTreeMap::new();
        for i in 0..97u32 {
            let cost = ((i * 67) % 97) as f64;
            queue.set(i, cost);
            expected.insert(i, cost);
        }
        for i in 0..388u32 {
            let slot = (i * 31) % 97;
            if i % 3 == 0 {
                queue.remove(slot);
                expected.remove(&slot);
            } else {
                let cost = ((i * 43) % 103) as f64;
                queue.set(slot, cost);
                expected.insert(slot, cost);
            }
            assert!(queue.heap.len() <= 97 && queue.peak <= 97);
        }
        let mut order = expected.into_iter().collect::<Vec<_>>();
        order.sort_by(|(a, x), (b, y)| x.total_cmp(y).then(a.cmp(b)));
        for (slot, cost) in order {
            assert_eq!(queue.pop(), Some((slot, cost)));
        }
        assert!(queue.pop().is_none());
        let mut mesh = SparseMesh {
            vertices: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            faces: vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
        };
        let state = MeshDecimator::new(&mesh)?;
        assert!(!state.protected.iter().any(|&v| v));
        let star = state.star(0, 1);
        assert!(!state.link_valid(0, 1, &star));
        let original = mesh.faces.clone();
        assert!(mesh.simplify(2).is_err());
        assert_eq!(mesh.faces, original);
        assert_eq!(mesh.simplify(4)?.0, vec![0, 1, 2, 3]);
        let source = SurfaceMesh {
            points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            faces: vec![[0, 1, 2]],
        };
        let mut candidate = SurfaceMesh {
            points: source.points[..3].to_vec(),
            faces: vec![[0, 1, 2]],
        };
        assert_eq!(
            qem_inherited_faces(&source, &candidate, &[0, 1, 2])?,
            vec![true]
        );
        assert_eq!(
            qem_inherited_faces(&source, &candidate, &[0, 1, 3])?,
            vec![false]
        );
        candidate.points[2][1] = f64::from_bits(1.0f64.to_bits() + 1);
        assert_eq!(
            qem_inherited_faces(&source, &candidate, &[0, 1, 2])?,
            vec![false]
        );
        candidate.points = source.points[..3]
            .iter()
            .map(|p| [p[0] + 32.0, p[1], p[2]])
            .collect();
        assert_eq!(
            qem_inherited_faces(&source, &candidate, &[0, 1, 2])?,
            vec![false]
        );
        Ok(())
    }

    #[test]
    #[ignore]
    fn sparse_mesh_qem_definition_cpu() -> Result<()> {
        let (root, frozen) = qem_reference()?;
        let analytic = &frozen["analytic"];
        let fields = &analytic["given_edge_math_fields"];
        let old_file = Path::new(
            analytic["existing_given_edge_math_floors"]
                .as_str()
                .context("operator freeze missing")?,
        );
        let old = old_file
            .parent()
            .context("operator fixture parent missing")?;
        let mut cells = 0;
        let mut defects = 0;
        for label in ["asymmetric_tetrahedron", "asymmetric_octahedron"] {
            let name = format!("{label}.initial_quadric");
            let arm = Path::new(
                fields[&name]["provenance"][0]
                    .as_str()
                    .context("quadric provenance missing")?,
            )
            .parent()
            .context("quadric arm missing")?;
            let bank = fixture(&old.join(arm).join(format!("{label}-input.safetensors")))?;
            let p = mesh_values(&bank, "vertices")?;
            let f = integers(&bank, "faces")?;
            let mesh = SparseMesh {
                vertices: p.iter().map(|&x| x as f32).collect(),
                faces: f,
            };
            let state = MeshDecimator::new(&mesh)?;
            let expected = fixture(
                &old.join(arm)
                    .join(format!("{label}-definition-quadrics.safetensors")),
            )?;
            let expected = mesh_values(&expected, "quadric_f64")?;
            let actual: Vec<_> = state
                .quadrics
                .iter()
                .copied()
                .flat_map(full_quadric)
                .collect();
            let control: Vec<_> = state
                .quadrics
                .iter()
                .copied()
                .map(qem_wrong)
                .flat_map(full_quadric)
                .collect();
            defects += qem_float(&name, &actual, &expected, &control, &fields[&name])?;
            cells += 1;
            let records: serde_json::Value = serde_json::from_slice(&std::fs::read(
                old.join(arm).join("FRESH_COLLAPSE_ASSOCIATION.json"),
            )?)?;
            let name = format!("{label}.fresh_position_given_edge");
            let mut actual = Vec::new();
            let mut expected = Vec::new();
            let mut control = Vec::new();
            for row in records.as_array().context("fresh edge records missing")? {
                if row["case"].as_str() != Some(label) {
                    continue;
                }
                let ids = row["inferred_lost_vertices"]
                    .as_array()
                    .context("source edge missing")?;
                let q = state.quadrics[ids[0].as_u64().context("source vertex missing")? as usize]
                    .add(
                        state.quadrics[ids[1].as_u64().context("source vertex missing")? as usize],
                    );
                actual.extend(q.stationary().context("fresh edge matrix singular")?);
                control.extend(
                    qem_wrong(q)
                        .stationary()
                        .context("control matrix singular")?,
                );
                expected.extend(
                    row["publisher_position"]
                        .as_array()
                        .context("publisher position missing")?
                        .iter()
                        .map(|v| v.as_f64().unwrap()),
                );
            }
            defects += qem_float(&name, &actual, &expected, &control, &fields[&name])?;
            cells += 1;
        }
        let prefix = Path::new(
            fields["prefix1.position_given_edge"]["provenance"][0]
                .as_str()
                .context("prefix provenance missing")?,
        )
        .parent()
        .context("prefix arm missing")?;
        let first = fixture(&old.join(prefix).join("step-001.safetensors"))?;
        let points = mesh_values(&first, "vertices")?;
        let faces = integers(&first, "faces")?;
        let mut q = qem_initial(points.as_chunks::<3>().0, faces.as_chunks::<3>().0);
        for step in 1..=40 {
            let bank = fixture(&old.join(prefix).join(format!("step-{step:03}.safetensors")))?;
            let name = format!("prefix{step}.quadric_hypothesis");
            let actual: Vec<_> = q.iter().copied().flat_map(full_quadric).collect();
            let control: Vec<_> = q
                .iter()
                .copied()
                .map(qem_wrong)
                .flat_map(full_quadric)
                .collect();
            defects += qem_float(
                &name,
                &actual,
                &mesh_values(&bank, "accumulated_quadric_hypothesis")?,
                &control,
                &fields[&name],
            )?;
            let ids = integers(&bank, "source_edge")?;
            ensure!(ids.len() == 2, "prescribed edge extent differs");
            let merged = q[ids[0] as usize].add(q[ids[1] as usize]);
            let name = format!("prefix{step}.position_given_edge");
            defects += qem_float(
                &name,
                &merged.stationary().context("prefix matrix singular")?,
                &mesh_values(&bank, "publisher_position")?,
                &qem_wrong(merged)
                    .stationary()
                    .context("prefix control matrix singular")?,
                &fields[&name],
            )?;
            let points = mesh_values(&bank, "vertices")?;
            let points = points.as_chunks::<3>().0;
            let next = mesh_values(&bank, "next_vertices")?;
            q = next
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| {
                    points
                        .iter()
                        .position(|old| old == p)
                        .map_or(merged, |i| q[i])
                })
                .collect();
            cells += 2;
        }
        let exact = fixture(
            &root.join(
                analytic["exact_new"]
                    .as_str()
                    .context("exact analytic fixture missing")?,
            ),
        )?;
        let planes = mesh_values(&exact, "planes")?;
        let q = planes
            .as_chunks::<4>()
            .0
            .iter()
            .fold(Quadric::default(), |q, &p| q.add(Quadric::plane(p)));
        ensure!(
            full_quadric(q).as_slice() == mesh_values(&exact, "summed_quadric")?,
            "axis-plane quadric differs"
        );
        let point = q.stationary().context("axis matrix singular")?;
        ensure!(
            point.as_slice() == mesh_values(&exact, "expected_point")? && q.energy(point) == 0.0,
            "axis minimizer differs"
        );
        let wrong = qem_wrong(q).stationary().context("axis control singular")?;
        ensure!(q.energy(wrong) == 3.5, "cross-term control not detected");
        let segment = mesh_values(&exact, "segment")?;
        let singular = Quadric::plane([0.0, 0.0, 1.0, -2.0]);
        ensure!(
            singular.stationary().is_none()
                && singular
                    .segment(segment[..3].try_into()?, segment[3..].try_into()?)
                    .as_slice()
                    == mesh_values(&exact, "segment_minimizer")?,
            "singular segment minimizer differs"
        );
        let flat = mesh_values(&exact, "flat_segment")?;
        for t in [0.0, 0.5, 1.0] {
            ensure!(
                singular.energy(std::array::from_fn(
                    |i| flat[i] + t * (flat[i + 3] - flat[i])
                )) == 0.0,
                "flat minimizer not stationary"
            );
        }
        ensure!(cells == 84, "analytic cell coverage differs");
        println!(
            "{} N5_M6_QEM_MATH cells={cells} defects={defects} prescribed_edges=40 control=half_cross_terms timing=false",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        ensure!(defects == 0, "quadric arithmetic defects");
        Ok(())
    }

    #[test]
    #[ignore]
    fn sparse_mesh_qem_production_cpu() -> Result<()> {
        let (root, frozen) = qem_reference()?;
        let case = std::env::var("FF_HI3D_QEM_CASE").context("FF_HI3D_QEM_CASE required")?;
        let cell = &frozen["cases"][&case];
        let source = fixture(&root.join(cell["D"].as_str().context("QEM source missing")?))?;
        let mut mesh = SparseMesh {
            vertices: mesh_values(&source, "vertices")?
                .into_iter()
                .map(|v| v as f32)
                .collect(),
            faces: integers(&source, "faces")?,
        };
        let raw = fixture(Path::new(
            cell["raw_input"].as_str().context("raw input missing")?,
        ))?;
        let mut raw_mesh = SparseMesh {
            vertices: mesh_values(&raw, "vertices")?
                .into_iter()
                .map(|v| v as f32)
                .collect(),
            faces: integers(&raw, "faces")?,
        };
        raw_mesh.fill_holes(
            raw_mesh
                .faces
                .len()
                .checked_mul(4)
                .context("fill capacity overflow")?
                / 3,
        )?;
        raw_mesh.filter_components(0.005)?;
        ensure!(
            raw_mesh.vertices == mesh.vertices && raw_mesh.faces == mesh.faces,
            "full native fill/filter handoff differs from frozen D"
        );
        mesh = raw_mesh;
        drop(raw);
        drop(source);
        let target = cell["target_faces"]
            .as_u64()
            .context("target face count missing")? as usize;
        let mut state = MeshDecimator::new(&mesh)?;
        let census = fixture(
            &root.join(
                cell["census_fixture"]
                    .as_str()
                    .context("census fixture missing")?,
            ),
        )?;
        let expected_protected = integers(&census, "protected_vertices")?;
        let actual_protected = state
            .protected
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| p.then_some(i as i32))
            .collect::<Vec<_>>();
        let mut audit_defects = usize::from(expected_protected != actual_protected);
        let normals = mesh_values(&census, "original_normals")?;
        let mut inherited = state.quadrics.clone();
        let mut adjacency = vec![Vec::<u32>::new(); state.points.len()];
        for (f, t) in mesh.faces.as_chunks::<3>().0.iter().enumerate() {
            for &v in t {
                adjacency[v as usize].push(f as u32);
            }
        }
        let mut observed = 0;
        let mut minimum_original_cosine = 1.0f64;
        state.run(target, |s, a, b, point, star| {
            audit_defects += usize::from(
                s.quadrics[a as usize].0 != inherited[a as usize].0
                    || s.quadrics[b as usize].0 != inherited[b as usize].0,
            );
            inherited[a as usize] = inherited[a as usize].add(inherited[b as usize]);
            let mut expected = adjacency[a as usize].clone();
            expected.extend(&adjacency[b as usize]);
            expected.sort_unstable();
            expected.dedup();
            audit_defects += usize::from(expected != star);
            for &f in &expected {
                let old: [i32; 3] = mesh.faces[3 * f as usize..3 * f as usize + 3].try_into()?;
                let removed = old.contains(&(a as i32)) && old.contains(&(b as i32));
                if !removed {
                    let previous = old.map(|v| {
                        std::array::from_fn(|i| f64::from(mesh.vertices[3 * v as usize + i]))
                    });
                    let next = old.map(|v| {
                        if v as u32 == a || v as u32 == b {
                            point
                        } else {
                            std::array::from_fn(|i| f64::from(mesh.vertices[3 * v as usize + i]))
                        }
                    });
                    for (&v, p) in old.iter().zip(previous) {
                        audit_defects += usize::from(s.points[v as usize] != p);
                    }
                    let n =
                        vector_cross(vector_sub(next[1], next[0]), vector_sub(next[2], next[0]));
                    let length = vector_dot(n, n).sqrt();
                    let original: [f64; 3] =
                        normals[3 * f as usize..3 * f as usize + 3].try_into()?;
                    let cosine = vector_dot(n, original) / length;
                    if cosine.is_finite() {
                        minimum_original_cosine = minimum_original_cosine.min(cosine);
                    }
                    let before = vector_cross(
                        vector_sub(previous[1], previous[0]),
                        vector_sub(previous[2], previous[0]),
                    );
                    audit_defects += usize::from(
                        !cosine.is_finite()
                            || cosine < 0.5
                            || length <= 0.0
                            || vector_dot(n, before) <= 0.0,
                    );
                }
                for &v in &old {
                    if let Some(i) = adjacency[v as usize].iter().position(|&x| x == f) {
                        adjacency[v as usize].swap_remove(i);
                    } else {
                        audit_defects += 1;
                    }
                }
                if removed {
                    mesh.faces[3 * f as usize..3 * f as usize + 3].fill(-1);
                } else {
                    for v in &mut mesh.faces[3 * f as usize..3 * f as usize + 3] {
                        if *v as u32 == b {
                            *v = a as i32;
                        }
                        adjacency[*v as usize].push(f);
                    }
                }
            }
            mesh.vertices[3 * a as usize..3 * a as usize + 3]
                .copy_from_slice(&point.map(|x| x as f32));
            observed += 1;
            Ok(())
        })?;
        for (v, &head) in state.heads.iter().enumerate() {
            if head != u32::MAX {
                audit_defects += usize::from(state.quadrics[v].0 != inherited[v].0);
            }
        }
        let peak = state.queue.peak;
        let (mut mesh, ancestry, original_vertices) = state.finish()?;
        let output = std::path::PathBuf::from(
            std::env::var("FF_HI3D_QEM_OUTPUT").context("FF_HI3D_QEM_OUTPUT required")?,
        );
        std::fs::create_dir(&output)?;
        let mut arrays = BTreeMap::from([
            (
                "vertices",
                (
                    safetensors::Dtype::F32,
                    vec![mesh.vertices.len() / 3, 3],
                    mesh.vertices
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                "faces",
                (
                    safetensors::Dtype::I32,
                    vec![mesh.faces.len() / 3, 3],
                    mesh.faces
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                "original_faces",
                (
                    safetensors::Dtype::I64,
                    vec![ancestry.len()],
                    ancestry
                        .iter()
                        .flat_map(|&v| (v as i64).to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                "original_vertices",
                (
                    safetensors::Dtype::I64,
                    vec![original_vertices.len()],
                    original_vertices
                        .iter()
                        .flat_map(|&v| (v as i64).to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            ),
        ]);
        mesh.normalize()?;
        arrays.insert(
            "normalized_vertices",
            (
                safetensors::Dtype::F32,
                vec![mesh.vertices.len() / 3, 3],
                mesh.vertices
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        );
        let views = arrays
            .iter()
            .map(|(&name, (dtype, shape, data))| {
                Ok((
                    name,
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), data)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        safetensors::tensor::serialize_to_file(views, None, &output.join("native.safetensors"))?;
        println!(
            "{} N5_M6_QEM_CAPTURE case={case} vertices={} faces={} target={target} ancestry={} contractions={observed} queue_peak={peak} minimum_original_normal_cosine={minimum_original_cosine} audit_defects={audit_defects} timing=false",
            if audit_defects == 0 { "PASS" } else { "DEFECT" },
            mesh.vertices.len() / 3,
            mesh.faces.len() / 3,
            ancestry.len()
        );
        ensure!(audit_defects == 0, "streamed contraction audit defects");
        qem_resources()?;
        Ok(())
    }

    struct SurfaceMesh {
        points: Vec<[f64; 3]>,
        faces: Vec<[u32; 3]>,
    }

    impl SurfaceMesh {
        fn read(path: &Path) -> Result<Self> {
            let bank = fixture(path)?;
            let points = mesh_values(&bank, "vertices")?.as_chunks::<3>().0.to_vec();
            let faces = integers(&bank, "faces")?
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| p.map(|v| v as u32))
                .collect::<Vec<_>>();
            ensure!(
                !points.is_empty()
                    && !faces.is_empty()
                    && points.iter().flatten().all(|v| v.is_finite())
                    && faces.iter().flatten().all(|&v| (v as usize) < points.len()),
                "surface fixture layout or values invalid"
            );
            Ok(Self { points, faces })
        }

        fn triangle(&self, face: u32) -> [[f64; 3]; 3] {
            self.faces[face as usize].map(|v| self.points[v as usize])
        }

        fn frame(&mut self, frame: &serde_json::Value) -> Result<()> {
            let center = frame["center"].as_array().context("frame center missing")?;
            let radius = frame["radius"].as_f64().context("frame radius missing")?;
            ensure!(
                center.len() == 3 && radius.is_finite() && radius > 0.0,
                "invalid common frame"
            );
            for p in &mut self.points {
                for axis in 0..3 {
                    p[axis] = (p[axis]
                        - center[axis].as_f64().context("frame coordinate missing")?)
                        / radius;
                }
            }
            Ok(())
        }

        fn queries(&self, mut support: impl FnMut(u32)) -> Result<Vec<[f64; 3]>> {
            let mut total = 0.0;
            let cdf = (0..self.faces.len())
                .map(|f| {
                    let [a, b, c] = self.triangle(f as u32);
                    let n = vector_cross(vector_sub(b, a), vector_sub(c, a));
                    total += vector_dot(n, n).sqrt() * 0.5;
                    total
                })
                .collect::<Vec<_>>();
            ensure!(
                total.is_finite() && total > 0.0,
                "surface query area invalid"
            );
            let radical = |mut i: usize, base: usize| {
                let mut value = 0.0;
                let mut fraction = 1.0 / base as f64;
                while i != 0 {
                    value += (i % base) as f64 * fraction;
                    i /= base;
                    fraction /= base as f64;
                }
                value
            };
            let mut face = 0;
            let mut queries = Vec::with_capacity(32768);
            for i in 0..32768 {
                let quantile = (i as f64 + 0.5) * (total / 32768.0);
                while cdf[face] <= quantile {
                    face += 1;
                }
                let r = radical(i + 1, 2).sqrt();
                let v = radical(i + 1, 3);
                let weights = [1.0 - r, r * (1.0 - v), r * v];
                let t = self.triangle(face as u32);
                support(face as u32);
                queries.push(std::array::from_fn(|axis| {
                    t[0][axis] * weights[0] + t[1][axis] * weights[1] + t[2][axis] * weights[2]
                }));
            }
            Ok(queries)
        }
    }

    fn qem_inherited_faces(
        source: &SurfaceMesh,
        candidate: &SurfaceMesh,
        original_vertices: &[i32],
    ) -> Result<Vec<bool>> {
        ensure!(
            original_vertices.len() == candidate.points.len()
                && original_vertices
                    .iter()
                    .all(|&v| v >= 0 && (v as usize) < source.points.len()),
            "inherited triangle vertex mapping invalid"
        );
        let canonical =
            |[a, b, c]: [u32; 3]| [[a, b, c], [b, c, a], [c, a, b]].into_iter().min().unwrap();
        let mut keys = source
            .faces
            .iter()
            .copied()
            .map(canonical)
            .collect::<Vec<_>>();
        keys.sort_unstable();
        let unchanged = candidate
            .points
            .iter()
            .zip(original_vertices)
            .map(|(point, &v)| {
                point
                    .iter()
                    .zip(source.points[v as usize])
                    .all(|(&a, b)| a.to_bits() == b.to_bits())
            })
            .collect::<Vec<_>>();
        Ok(candidate
            .faces
            .iter()
            .map(|triangle| {
                triangle.iter().all(|&v| unchanged[v as usize])
                    && keys
                        .binary_search(&canonical(
                            triangle.map(|v| original_vertices[v as usize] as u32),
                        ))
                        .is_ok()
            })
            .collect())
    }

    struct SurfaceNode {
        lo: [f64; 3],
        hi: [f64; 3],
        start: usize,
        len: usize,
        children: [usize; 2],
    }

    struct SurfaceTree<'a> {
        mesh: &'a SurfaceMesh,
        ids: Vec<u32>,
        nodes: Vec<SurfaceNode>,
    }

    fn segment_distance(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
        let d = vector_sub(b, a);
        let length = vector_dot(d, d);
        let t = if length > 0.0 {
            (vector_dot(vector_sub(p, a), d) / length).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let point = std::array::from_fn(|i| (1.0 - t) * a[i] + t * b[i]);
        let delta = vector_sub(p, point);
        vector_dot(delta, delta)
    }

    fn triangle_distance(p: [f64; 3], [a, b, c]: [[f64; 3]; 3]) -> f64 {
        let mut best = segment_distance(p, a, b)
            .min(segment_distance(p, b, c))
            .min(segment_distance(p, c, a));
        let n = vector_cross(vector_sub(b, a), vector_sub(c, a));
        let length = vector_dot(n, n);
        if length > 0.0 {
            let w = [
                vector_dot(vector_cross(vector_sub(b, p), vector_sub(c, p)), n) / length,
                vector_dot(vector_cross(vector_sub(c, p), vector_sub(a, p)), n) / length,
                vector_dot(vector_cross(vector_sub(a, p), vector_sub(b, p)), n) / length,
            ];
            if w.iter().all(|v| v.is_finite() && *v >= 0.0) {
                let sum = w[0] + w[1] + w[2];
                if sum > 0.0 {
                    let u = if w[0] + w[1] > 0.0 {
                        (w[1] / (w[0] + w[1])).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let v = (w[2] / sum).clamp(0.0, 1.0);
                    let point = std::array::from_fn(|i| {
                        (1.0 - v) * ((1.0 - u) * a[i] + u * b[i]) + v * c[i]
                    });
                    let delta = vector_sub(p, point);
                    best = best.min(vector_dot(delta, delta));
                }
            }
        }
        best
    }

    impl<'a> SurfaceTree<'a> {
        fn new(mesh: &'a SurfaceMesh) -> Self {
            let mut tree = Self {
                mesh,
                ids: (0..mesh.faces.len() as u32).collect(),
                nodes: Vec::new(),
            };
            tree.partition(0, mesh.faces.len());
            tree
        }

        fn partition(&mut self, start: usize, len: usize) -> usize {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for &f in &self.ids[start..start + len] {
                for p in self.mesh.triangle(f) {
                    for axis in 0..3 {
                        lo[axis] = lo[axis].min(p[axis]);
                        hi[axis] = hi[axis].max(p[axis]);
                    }
                }
            }
            let index = self.nodes.len();
            self.nodes.push(SurfaceNode {
                lo,
                hi,
                start,
                len,
                children: [usize::MAX; 2],
            });
            if len > 8 {
                let axis = (0..3)
                    .max_by(|&i, &j| (hi[i] - lo[i]).total_cmp(&(hi[j] - lo[j])))
                    .unwrap();
                let mesh = self.mesh;
                self.ids[start..start + len].select_nth_unstable_by(len / 2, |&a, &b| {
                    let center = |f| mesh.triangle(f).into_iter().map(|p| p[axis]).sum::<f64>();
                    center(a).total_cmp(&center(b)).then(a.cmp(&b))
                });
                self.nodes[index].children = [
                    self.partition(start, len / 2),
                    self.partition(start + len / 2, len - len / 2),
                ];
            }
            index
        }

        fn box_distance(&self, p: [f64; 3], index: usize) -> f64 {
            let node = &self.nodes[index];
            (0..3)
                .map(|i| (node.lo[i] - p[i]).max(p[i] - node.hi[i]).max(0.0).powi(2))
                .sum()
        }

        fn distance(&self, p: [f64; 3]) -> f64 {
            let mut stack = [0usize; 64];
            let mut count = 1;
            let mut best = f64::INFINITY;
            while count != 0 {
                count -= 1;
                let index = stack[count];
                if self.box_distance(p, index) > best {
                    continue;
                }
                let node = &self.nodes[index];
                if node.children[0] == usize::MAX {
                    for &f in &self.ids[node.start..node.start + node.len] {
                        best = best.min(triangle_distance(p, self.mesh.triangle(f)));
                    }
                } else {
                    let [mut a, mut b] = node.children;
                    if self.box_distance(p, a) < self.box_distance(p, b) {
                        std::mem::swap(&mut a, &mut b);
                    }
                    assert!(count + 2 <= stack.len());
                    stack[count] = a;
                    stack[count + 1] = b;
                    count += 2;
                }
            }
            best.sqrt()
        }

        fn measure(
            &self,
            points: &[[f64; 3]],
            queries: &[[f64; 3]],
            mut visit: impl FnMut(usize, f64),
        ) -> Result<[f64; 2]> {
            ensure!(queries.len() == 32768, "surface query count differs");
            let mut maximum = 0.0f64;
            let mut squares = 0.0;
            for (i, &p) in points.iter().chain(queries).enumerate() {
                let d = self.distance(p);
                ensure!(d.is_finite() && d >= 0.0, "nonfinite surface distance");
                visit(i, d);
                maximum = maximum.max(d);
                if i >= points.len() {
                    squares += d * d;
                }
            }
            Ok([maximum, (squares / queries.len() as f64).sqrt()])
        }

        fn metrics(&self, points: &[[f64; 3]], queries: &[[f64; 3]]) -> Result<[f64; 2]> {
            self.measure(points, queries, |_, _| {})
        }
    }

    fn qem_proximity_validity(root: &Path, frozen: &serde_json::Value) -> Result<()> {
        let mut total = 0;
        let mut kinds = 0;
        for (path, layout) in frozen["layouts"]
            .as_object()
            .context("fixture layouts missing")?
        {
            if layout.get("queries").is_none() || layout.get("vertices").is_none() {
                continue;
            }
            let key = if layout.get("expected_distance").is_some() {
                "expected_distance"
            } else {
                "expected"
            };
            let bank = fixture(&root.join(path))?;
            let mesh = SurfaceMesh::read(&root.join(path))?;
            let queries = mesh_values(&bank, "queries")?;
            let expected = mesh_values(&bank, key)?;
            let tree = SurfaceTree::new(&mesh);
            for (&p, &d) in queries.as_chunks::<3>().0.iter().zip(&expected) {
                ensure!(
                    (tree.distance(p) - d).abs() <= 1e-12,
                    "native triangle evaluator exact fixture differs"
                );
            }
            let [a, b, c] = mesh.triangle(0);
            kinds |= if mesh.points.len() > 3 {
                8
            } else if a == b && a == c {
                4
            } else if vector_dot(
                vector_cross(vector_sub(b, a), vector_sub(c, a)),
                vector_cross(vector_sub(b, a), vector_sub(c, a)),
            ) == 0.0
            {
                2
            } else {
                1
            };
            total += expected.len();
        }
        ensure!(kinds == 15, "triangle evaluator validity coverage differs");
        ensure!(
            frozen["native_decimator_candidates"].as_u64() == Some(0),
            "reference freeze is not pre-candidate"
        );
        println!(
            "PASS N5_M6_QEM_EVALUATOR exact_queries={total} unused_vertices_excluded_from_target=true timing=false"
        );
        Ok(())
    }

    struct MeshCensus {
        labels: Vec<usize>,
        stats: Vec<[i64; 5]>,
        boundary: Vec<[u32; 2]>,
        singular: Vec<(u64, Vec<(usize, bool)>)>,
        fans: Vec<usize>,
        zero: Vec<usize>,
        duplicates: usize,
    }

    fn census_union(parent: &mut [usize], a: usize, b: usize) {
        let a = component_root(parent, a);
        let b = component_root(parent, b);
        if a != b {
            parent[a.max(b)] = a.min(b);
        }
    }

    fn mesh_census(mesh: &SurfaceMesh) -> MeshCensus {
        let mut edges = Vec::with_capacity(mesh.faces.len() * 3);
        let mut zero = Vec::new();
        for (f, triangle) in mesh.faces.iter().enumerate() {
            let [a, b, c] = mesh.triangle(f as u32);
            let n = vector_cross(vector_sub(b, a), vector_sub(c, a));
            if vector_dot(n, n) == 0.0 {
                zero.push(f);
            }
            for side in 0..3 {
                let a = triangle[side];
                let b = triangle[(side + 1) % 3];
                edges.push((
                    (u64::from(a.min(b)) << 32) | u64::from(a.max(b)),
                    3 * f + side,
                ));
            }
        }
        edges.sort_unstable();
        let vertex = |c: usize| mesh.faces[c / 3][c % 3];
        let next = |c: usize| c / 3 * 3 + (c % 3 + 1) % 3;
        let mut parent = (0..mesh.faces.len()).collect::<Vec<_>>();
        let mut corners = (0..mesh.faces.len() * 3).collect::<Vec<_>>();
        let mut first = 0;
        while first < edges.len() {
            let mut end = first + 1;
            while end < edges.len() && edges[end].0 == edges[first].0 {
                end += 1;
            }
            if end - first == 2 {
                let (a, b) = (edges[first].1, edges[first + 1].1);
                if a / 3 != b / 3 {
                    census_union(&mut parent, a / 3, b / 3);
                    if vertex(a) == vertex(b) {
                        census_union(&mut corners, a, b);
                        census_union(&mut corners, next(a), next(b));
                    } else {
                        census_union(&mut corners, a, next(b));
                        census_union(&mut corners, next(a), b);
                    }
                }
            }
            first = end;
        }
        let mut roots = vec![usize::MAX; parent.len()];
        let mut stats = Vec::<[i64; 5]>::new();
        let mut labels = Vec::with_capacity(parent.len());
        for f in 0..parent.len() {
            let root = component_root(&mut parent, f);
            if roots[root] == usize::MAX {
                roots[root] = stats.len();
                stats.push([f as i64, 0, 0, 0, 0]);
            }
            labels.push(roots[root]);
            stats[roots[root]][3] += 1;
        }
        drop(parent);
        drop(roots);
        let mut pairs = Vec::with_capacity(mesh.faces.len() * 3);
        for c in 0..corners.len() {
            let root = component_root(&mut corners, c);
            pairs.push((u64::from(vertex(c)) << 32) | root as u64);
        }
        drop(corners);
        pairs.sort_unstable();
        pairs.dedup();
        let mut fans = vec![0; mesh.points.len()];
        for &p in &pairs {
            fans[(p >> 32) as usize] += 1;
        }
        pairs.clear();
        for (f, triangle) in mesh.faces.iter().enumerate() {
            for &v in triangle {
                pairs.push(((labels[f] as u64) << 32) | u64::from(v));
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        for p in pairs {
            stats[(p >> 32) as usize][1] += 1;
        }
        let mut boundary = Vec::new();
        let mut singular = Vec::new();
        first = 0;
        while first < edges.len() {
            let mut end = first + 1;
            while end < edges.len() && edges[end].0 == edges[first].0 {
                end += 1;
            }
            let mut components = edges[first..end]
                .iter()
                .map(|e| labels[e.1 / 3])
                .collect::<Vec<_>>();
            components.sort_unstable();
            components.dedup();
            for c in components {
                stats[c][2] += 1;
            }
            if end - first == 1 {
                boundary.push([vertex(edges[first].1), vertex(next(edges[first].1))]);
            }
            if end - first > 2 {
                singular.push((
                    edges[first].0,
                    edges[first..end]
                        .iter()
                        .map(|e| (e.1 / 3, vertex(e.1) == (e.0 >> 32) as u32))
                        .collect(),
                ));
            }
            first = end;
        }
        drop(edges);
        for row in &mut stats {
            row[4] = row[1] - row[2] + row[3];
        }
        let mut triples = mesh
            .faces
            .iter()
            .map(|t| {
                let mut t = *t;
                t.sort_unstable();
                t
            })
            .collect::<Vec<_>>();
        triples.sort_unstable();
        let duplicates = (0..triples.len())
            .filter(|&i| {
                (i > 0 && triples[i] == triples[i - 1])
                    || (i + 1 < triples.len() && triples[i] == triples[i + 1])
            })
            .count();
        MeshCensus {
            labels,
            stats,
            boundary,
            singular,
            fans,
            zero,
            duplicates,
        }
    }

    fn qem_structural(
        root: &Path,
        cell: &serde_json::Value,
        source: &SurfaceMesh,
        candidate: &SurfaceMesh,
        bank: &MappedCheckpoint,
    ) -> Result<usize> {
        let reference = fixture(
            &root.join(
                cell["census_fixture"]
                    .as_str()
                    .context("census fixture missing")?,
            ),
        )?;
        let old = mesh_census(source);
        let expected_labels = integers(&reference, "face_component")?;
        let expected_stats = integers(&reference, "component_stats")?;
        ensure!(
            old.labels
                .iter()
                .copied()
                .eq(expected_labels.iter().map(|&x| x as usize))
                && old
                    .stats
                    .iter()
                    .flatten()
                    .copied()
                    .eq(expected_stats.iter().map(|&x| i64::from(x))),
            "native source component census differs from frozen oracle"
        );
        let mut old_boundary = old.boundary.clone();
        old_boundary.sort_unstable();
        let mut expected_boundary = integers(&reference, "directed_boundary")?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| p.map(|v| v as u32))
            .collect::<Vec<_>>();
        expected_boundary.sort_unstable();
        ensure!(
            old_boundary == expected_boundary
                && old.singular.len()
                    == cell["census"]["nonmanifold_edges"]
                        .as_u64()
                        .context("singular edge count missing")? as usize
                && old.fans.iter().filter(|&&n| n > 1).count()
                    == cell["census"]["multiple_fan_vertices"]
                        .as_u64()
                        .context("fan count missing")? as usize,
            "native source edge/fan census differs"
        );
        let current = mesh_census(candidate);
        let ancestry = integers(bank, "original_faces")?;
        let original_vertices = integers(bank, "original_vertices")?;
        ensure!(
            ancestry.len() == candidate.faces.len()
                && original_vertices.len() == candidate.points.len()
                && ancestry.windows(2).all(|p| p[0] < p[1])
                && original_vertices.windows(2).all(|p| p[0] < p[1]),
            "candidate ancestry layout invalid"
        );
        ensure!(
            ancestry
                .iter()
                .all(|&f| f >= 0 && (f as usize) < source.faces.len())
                && original_vertices
                    .iter()
                    .all(|&v| v >= 0 && (v as usize) < source.points.len()),
            "candidate ancestry index invalid"
        );
        let mut defects = 0;
        let mut boundary = current
            .boundary
            .iter()
            .map(|p| p.map(|v| original_vertices[v as usize] as u32))
            .collect::<Vec<_>>();
        boundary.sort_unstable();
        defects += usize::from(boundary != old_boundary);
        for v in integers(&reference, "protected_vertices")? {
            defects += usize::from(
                original_vertices
                    .binary_search(&v)
                    .map_or(true, |i| candidate.points[i] != source.points[v as usize]),
            );
        }
        for (i, &n) in current.fans.iter().enumerate() {
            defects += usize::from(n != old.fans[original_vertices[i] as usize]);
        }
        let mut singular = current
            .singular
            .iter()
            .map(|&(key, ref faces)| {
                let a = original_vertices[(key >> 32) as usize] as u32;
                let b = original_vertices[(key & 0xffffffff) as usize] as u32;
                let mut faces = faces
                    .iter()
                    .map(|&(f, direction)| (ancestry[f] as usize, direction))
                    .collect::<Vec<_>>();
                faces.sort_unstable();
                (((u64::from(a.min(b))) << 32) | u64::from(a.max(b)), faces)
            })
            .collect::<Vec<_>>();
        singular.sort_unstable();
        defects += usize::from(singular != old.singular);
        defects += usize::from(current.duplicates != old.duplicates);
        let zeros = current
            .zero
            .iter()
            .map(|&f| ancestry[f] as usize)
            .collect::<Vec<_>>();
        defects += usize::from(zeros != old.zero);
        let mut mapped = vec![usize::MAX; current.stats.len()];
        for (f, &label) in current.labels.iter().enumerate() {
            let expected = old.labels[ancestry[f] as usize];
            if mapped[label] == usize::MAX {
                mapped[label] = expected;
            }
            defects += usize::from(mapped[label] != expected);
        }
        let mut unique = mapped.clone();
        unique.sort_unstable();
        unique.dedup();
        defects += usize::from(unique.len() != old.stats.len() || mapped.len() != old.stats.len());
        for (c, &original) in mapped.iter().enumerate() {
            defects += usize::from(current.stats[c][4] != old.stats[original][4]);
        }
        let normals = mesh_values(&reference, "original_normals")?;
        let mut minimum_cosine = 1.0f64;
        for (f, &original) in ancestry.iter().enumerate() {
            let n0: [f64; 3] =
                normals[3 * original as usize..3 * original as usize + 3].try_into()?;
            if n0 == [0.0; 3] {
                continue;
            }
            let [a, b, c] = candidate.triangle(f as u32);
            let n = vector_cross(vector_sub(b, a), vector_sub(c, a));
            let length = vector_dot(n, n).sqrt();
            let cosine = vector_dot(n, n0) / length;
            if cosine.is_finite() {
                minimum_cosine = minimum_cosine.min(cosine);
            }
            defects += usize::from(!cosine.is_finite() || length <= 0.0 || cosine < 0.5);
        }
        let minimum = cell["minimum_output_faces"]
            .as_u64()
            .context("minimum face count missing")? as usize;
        let maximum = cell["target_faces"]
            .as_u64()
            .context("target face count missing")? as usize;
        defects += usize::from(!(minimum..=maximum).contains(&candidate.faces.len()));
        println!(
            "N5_M6_QEM_STRUCTURE faces={} components={} boundary={} singular={} minimum_original_normal_cosine={minimum_cosine} defects={defects}",
            candidate.faces.len(),
            current.stats.len(),
            current.boundary.len(),
            current.singular.len()
        );
        Ok(defects)
    }

    fn qem_queries(common: &Path, kind: &str) -> Result<Vec<[f64; 3]>> {
        let bank = fixture(&common.join(format!("{kind}-queries.safetensors")))?;
        let points = mesh_values(&bank, "points")?;
        ensure!(
            points.len() == 3 * 32768,
            "canonical point bank extent differs"
        );
        Ok(points.as_chunks::<3>().0.to_vec())
    }

    fn translation_certificate(input: &SurfaceMesh, output: &SurfaceMesh) -> f64 {
        let bounds = |mesh: &SurfaceMesh, shift: f64| {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for triangle in &mesh.faces {
                for &v in triangle {
                    for axis in 0..3 {
                        let p = mesh.points[v as usize][axis] + if axis == 0 { shift } else { 0.0 };
                        lo[axis] = lo[axis].min(p);
                        hi[axis] = hi[axis].max(p);
                    }
                }
            }
            (lo, hi)
        };
        let (a, b) = bounds(input, 0.0);
        let (c, d) = bounds(output, 32.0);
        (0..3)
            .map(|i| (c[i] - b[i]).max(a[i] - d[i]).max(0.0).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn qem_final_normalization(
        frozen: &serde_json::Value,
        case: &str,
        mesh: &SurfaceMesh,
        bank: &MappedCheckpoint,
    ) -> Result<usize> {
        let old: serde_json::Value = serde_json::from_slice(&std::fs::read(
            frozen["previous_operator_freeze"]
                .as_str()
                .context("normalization freeze missing")?,
        )?)?;
        let bounds = &old["float_cells"][format!("{case}.normalize")]["tau"];
        let normalized = mesh_values(bank, "normalized_vertices")?;
        ensure!(
            normalized.len() == mesh.points.len() * 3 && normalized.iter().all(|v| v.is_finite()),
            "normalized result layout invalid"
        );
        let mean: [f64; 3] = std::array::from_fn(|i| {
            mesh.points.iter().map(|p| p[i]).sum::<f64>() / mesh.points.len() as f64
        });
        let radius = mesh
            .points
            .iter()
            .map(|&p| {
                let d = vector_sub(p, mean);
                vector_dot(d, d).sqrt()
            })
            .fold(0.0, f64::max);
        ensure!(
            radius > 0.0 && radius.is_finite(),
            "normalization definition radius invalid"
        );
        let mut maximum = 0.0f64;
        let mut sum = 0.0;
        for (&point, row) in mesh.points.iter().zip(normalized.as_chunks::<3>().0) {
            for axis in 0..3 {
                let error = ((point[axis] - mean[axis]) / radius - row[axis]).abs();
                maximum = maximum.max(error);
                sum += error * error;
            }
        }
        let rms = (sum / normalized.len() as f64).sqrt();
        let defects = usize::from(
            maximum
                > bounds["max_abs"]
                    .as_f64()
                    .context("normalization max floor missing")?
                || rms
                    > bounds["RMS"]
                        .as_f64()
                        .context("normalization RMS floor missing")?,
        );
        println!(
            "N5_M6_PROCESSED_NORMALIZATION case={case} max_abs={maximum} RMS={rms} defects={defects}"
        );
        Ok(defects)
    }

    #[test]
    #[ignore]
    fn sparse_mesh_qem_reference_cpu() -> Result<()> {
        let (root, frozen) = qem_reference()?;
        qem_proximity_validity(&root, &frozen)?;
        let case = std::env::var("FF_HI3D_QEM_CASE").context("FF_HI3D_QEM_CASE required")?;
        let cell = &frozen["cases"][&case];
        let d = root.join(cell["D"].as_str().context("D fixture missing")?);
        let common = d.parent().context("common reference parent missing")?;
        let output = std::path::PathBuf::from(
            std::env::var("FF_HI3D_QEM_OUTPUT").context("FF_HI3D_QEM_OUTPUT required")?,
        )
        .join("native");
        std::fs::create_dir_all(&output)?;
        ensure!(
            !output.join("metrics.json").exists(),
            "native reference already captured"
        );
        let mut meshes = vec![
            (
                "R",
                SurfaceMesh::read(Path::new(
                    cell["raw_input"].as_str().context("RAW fixture missing")?,
                ))?,
                qem_queries(common, "R")?,
            ),
            ("D", SurfaceMesh::read(&d)?, qem_queries(common, "D")?),
            (
                "P",
                SurfaceMesh::read(Path::new(
                    cell["publisher_P"]
                        .as_str()
                        .context("publisher fixture missing")?,
                ))?,
                qem_queries(common, "P")?,
            ),
            (
                "LOW",
                SurfaceMesh::read(
                    &root.join(
                        cell["control_LOW"]
                            .as_str()
                            .context("control fixture missing")?,
                    ),
                )?,
                qem_queries(common, "LOW")?,
            ),
        ];
        for (_, mesh, _) in &mut meshes {
            mesh.frame(&cell["frame"])?;
        }
        let mut records = BTreeMap::new();
        for target in 0..meshes.len() {
            let tree = SurfaceTree::new(&meshes[target].1);
            for source in 0..meshes.len() {
                if target == source {
                    continue;
                }
                if (meshes[source].0 == "P" && meshes[target].0 == "LOW")
                    || (meshes[source].0 == "LOW" && meshes[target].0 == "P")
                {
                    continue;
                }
                let direction = format!("{}->{}", meshes[source].0, meshes[target].0);
                let file = output.join(format!(
                    "{}-to-{}.safetensors",
                    meshes[source].0, meshes[target].0
                ));
                ensure!(
                    !file.exists(),
                    "native reference direction already captured"
                );
                let control = meshes[source].0 == "LOW" || meshes[target].0 == "LOW";
                let mut distances = Vec::new();
                let values =
                    tree.measure(&meshes[source].1.points, &meshes[source].2, |_, value| {
                        if !control {
                            distances.push(value);
                        }
                    })?;
                if !control {
                    let bytes = distances
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>();
                    let tensor = safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F64,
                        vec![distances.len()],
                        &bytes,
                    )?;
                    safetensors::tensor::serialize_to_file(
                        BTreeMap::from([("distance", tensor)]),
                        None,
                        &file,
                    )?;
                }
                let row = serde_json::json!({"max":values[0],"RMS":values[1],"vertex_queries":meshes[source].1.points.len(),"surface_queries":meshes[source].2.len(),"distance_file":if control {None} else {file.file_name().and_then(|s|s.to_str())},"control":control});
                println!("N5_M6_NATIVE_REFERENCE case={case} direction={direction} {row}");
                records.insert(direction, row);
            }
        }
        use std::io::Write;
        let result = serde_json::json!({"case":case,"metrics":records,"evaluator":"explicit_convex_surface_points","candidate_data_read":false,"timing":false});
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join("metrics.json"))?;
        writeln!(file, "{}", serde_json::to_string_pretty(&result)?)?;
        println!("PASS N5_M6_NATIVE_REFERENCE {result}");
        qem_resources()?;
        Ok(())
    }

    #[test]
    #[ignore]
    fn sparse_mesh_qem_geometry_cpu() -> Result<()> {
        let (root, frozen) = qem_reference()?;
        let evaluator: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("FROZEN-evaluator.json"))
                .context("FROZEN-evaluator.json required before re-scoring")?,
        )?;
        let inherited_freeze: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("FROZEN-evaluator-2.json"))
                .context("FROZEN-evaluator-2.json required before re-scoring")?,
        )?;
        ensure!(
            inherited_freeze["parent"].as_str().map(Path::new)
                == Some(root.join("FROZEN-evaluator.json").as_path()),
            "inherited maximum addendum parent differs"
        );
        qem_proximity_validity(&root, &frozen)?;
        let case = std::env::var("FF_HI3D_QEM_CASE").context("FF_HI3D_QEM_CASE required")?;
        let cell = &frozen["cases"][&case];
        let output = std::path::PathBuf::from(
            std::env::var("FF_HI3D_QEM_OUTPUT").context("FF_HI3D_QEM_OUTPUT required")?,
        );
        ensure!(
            !output.join("SCORES.json").exists(),
            "geometry score already recorded"
        );
        let source_path = root.join(cell["D"].as_str().context("D fixture missing")?);
        let source = SurfaceMesh::read(&source_path)?;
        let candidate = SurfaceMesh::read(&output.join("native.safetensors"))?;
        let bank = fixture(&output.join("native.safetensors"))?;
        let mut defects = qem_structural(&root, cell, &source, &candidate, &bank)?;
        defects += qem_final_normalization(&frozen, &case, &candidate, &bank)?;
        let inherited =
            qem_inherited_faces(&source, &candidate, &integers(&bank, "original_vertices")?)?;
        let mut inherited_vertices = vec![false; candidate.points.len()];
        for (triangle, &same) in candidate.faces.iter().zip(&inherited) {
            if same {
                for &vertex in triangle {
                    inherited_vertices[vertex as usize] = true;
                }
            }
        }
        drop(bank);
        let frame = &cell["frame"];
        let common = source_path
            .parent()
            .context("common query parent missing")?;
        let mut inputs = vec![
            (
                "R",
                SurfaceMesh::read(Path::new(
                    cell["raw_input"].as_str().context("RAW fixture missing")?,
                ))?,
                qem_queries(common, "R")?,
            ),
            ("D", source, qem_queries(common, "D")?),
        ];
        let mut outputs = vec![
            (
                "P",
                SurfaceMesh::read(Path::new(
                    cell["publisher_P"]
                        .as_str()
                        .context("publisher fixture missing")?,
                ))?,
                qem_queries(common, "P")?,
            ),
            ("C", candidate, Vec::new()),
            (
                "LOW",
                SurfaceMesh::read(
                    &root.join(
                        cell["control_LOW"]
                            .as_str()
                            .context("control fixture missing")?,
                    ),
                )?,
                qem_queries(common, "LOW")?,
            ),
        ];
        for (_, mesh, _) in &mut inputs {
            mesh.frame(frame)?;
        }
        let mut query_faces = Vec::new();
        for (name, mesh, queries) in &mut outputs {
            mesh.frame(frame)?;
            if *name == "C" {
                *queries = mesh.queries(|face| query_faces.push(face))?;
            }
            if *name == "P" {
                ensure!(
                    *queries == mesh.queries(|_| {})?,
                    "native area-query construction differs from canonical reference"
                );
            }
        }
        let raw_max_field = &evaluator["cases"][&case]["fields"]["P->R.max"];
        let raw_max_bound = raw_max_field["quality_bound"]
            .as_f64()
            .context("raw maximum bound missing")?;
        let raw_max_tau = raw_max_field["tau"]
            .as_f64()
            .context("raw maximum floor missing")?;
        let mut lifted_queries = 0usize;
        let mut largest_lift = 0.0f64;
        let mut maximum_unlifted = 0.0f64;
        let mut inherited_violations = 0usize;
        let mut changed_violations = 0usize;
        let mut metrics = BTreeMap::new();
        for (name, mesh, _) in &outputs {
            let tree = SurfaceTree::new(mesh);
            for (input, source, queries) in &inputs {
                let error = tree.metrics(&source.points, queries)?;
                metrics.insert((name.to_string(), format!("{input}->P")), error);
                println!(
                    "N5_M6_QEM_DISTANCE case={case} arm={name} direction={input}->{name} max={} RMS={}",
                    error[0], error[1]
                );
            }
        }
        let mut translation = BTreeMap::new();
        for (input, mesh, _) in &inputs {
            translation.insert(
                input.to_string(),
                translation_certificate(mesh, &outputs[0].1),
            );
            let tree = SurfaceTree::new(mesh);
            for (name, source, queries) in &outputs {
                let error = tree.measure(&source.points, queries, |index, distance| {
                    if *input != "R" || *name != "C" {
                        return;
                    }
                    let same = if index < source.points.len() {
                        inherited_vertices[index]
                    } else {
                        inherited[query_faces[index - source.points.len()] as usize]
                    };
                    if distance > raw_max_bound && same {
                        let point = if index < source.points.len() {
                            source.points[index]
                        } else {
                            queries[index - source.points.len()]
                        };
                        let input_distance = tree.distance(point);
                        assert_eq!(input_distance.to_bits(), distance.to_bits());
                        let permitted = raw_max_bound.max(input_distance + raw_max_tau);
                        largest_lift = largest_lift.max(permitted - raw_max_bound);
                        lifted_queries += 1;
                        inherited_violations += usize::from(distance > permitted);
                    } else {
                        maximum_unlifted = maximum_unlifted.max(distance);
                        changed_violations += usize::from(distance > raw_max_bound);
                    }
                })?;
                metrics.insert((name.to_string(), format!("P->{input}")), error);
                println!(
                    "N5_M6_QEM_DISTANCE case={case} arm={name} direction={name}->{input} max={} RMS={}",
                    error[0], error[1]
                );
            }
        }
        let mut records = BTreeMap::new();
        let mut best_control = 0.0f64;
        let mut minimum_control = f64::INFINITY;
        let mut minimum_translation = f64::INFINITY;
        for (key, field) in evaluator["cases"][&case]["fields"]
            .as_object()
            .context("quality fields missing")?
        {
            let (direction, metric) = key.rsplit_once('.').context("metric key invalid")?;
            let i = usize::from(metric == "RMS");
            let reference = metrics[&("P".to_string(), direction.to_string())][i];
            let actual = metrics[&("C".to_string(), direction.to_string())][i];
            let control = metrics[&("LOW".to_string(), direction.to_string())][i];
            let expected = field["native_E"]
                .as_f64()
                .context("publisher metric missing")?;
            let tau = field["tau"].as_f64().context("metric floor missing")?;
            let bound = field["quality_bound"]
                .as_f64()
                .context("quality bound missing")?;
            let mut formula = (1.25 * expected).max(tau);
            if direction.contains('R') {
                formula = formula.max(
                    field["input_floor"]
                        .as_f64()
                        .context("R-direction input floor missing")?
                        + tau,
                );
            }
            ensure!(bound == formula, "evaluator addendum bound formula differs");
            ensure!(
                (reference - expected).abs() <= tau,
                "native evaluator disagrees with frozen publisher metric {key}"
            );
            let input = if direction.contains('R') { "R" } else { "D" };
            let certified = translation[input] / bound;
            ensure!(
                certified > 10.0,
                "translation validity control failed {key}"
            );
            minimum_translation = minimum_translation.min(certified);
            if input == "D" {
                best_control = best_control.max(control / bound);
                minimum_control = minimum_control.min(control / bound);
            }
            let defect = if key == "P->R.max" {
                maximum_unlifted > bound || inherited_violations != 0
            } else {
                actual > bound
            };
            defects += usize::from(defect);
            println!(
                "N5_M6_QEM_SCORE case={case} field={key} value={actual} bound={bound} publisher_native={reference} publisher_frozen={expected} publisher_blackbox={} control_multiple={} defect={defect}",
                field["blackbox_E"],
                control / bound
            );
            records.insert(key.clone(), serde_json::json!({"value":actual,"quality_bound":bound,"reference_native":reference,"reference_frozen":expected,"reference_blackbox":field["blackbox_E"],"input_floor":field["input_floor"],"reference_delta":(reference-expected).abs(),"control_value":control,"control_multiple":control/bound,"translation_certificate_multiple":certified,"defect":defect}));
        }
        ensure!(
            best_control > 2.0,
            "realistic degradation control failed frozen detector"
        );
        let inheritance = serde_json::json!({"inherited_faces":inherited.iter().filter(|&&same|same).count(),"inherited_vertices":inherited_vertices.iter().filter(|&&same|same).count(),"inherited_surface_queries":query_faces.iter().filter(|&&face|inherited[face as usize]).count(),"max_queries_lifted":lifted_queries,"largest_lift":largest_lift,"maximum_unlifted_distance":maximum_unlifted,"changed_query_violations":changed_violations,"inherited_query_violations":inherited_violations,"R_to_C_queries_lifted":0,"RMS_lifted":false});
        println!("N5_M6_QEM_INHERITED_MAX case={case} {inheritance}");
        let result = serde_json::json!({"case":case,"metric_cells":records.len(),"fields":records,"defects":defects,"minimum_realistic_control_multiple":minimum_control,"best_realistic_control_multiple":best_control,"minimum_translation_certificate_multiple":minimum_translation,"inherited_max":inheritance,"timing":false});
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join("SCORES.json"))?;
        writeln!(file, "{}", serde_json::to_string_pretty(&result)?)?;
        println!(
            "{} N5_M6_QEM_GEOMETRY {result}",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        qem_resources()?;
        ensure!(defects == 0, "mesh simplification defects");
        Ok(())
    }

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
            safetensors::Dtype::I64 => Ok(t
                .data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b) as f64)
                .collect()),
            _ => anyhow::bail!("mesh numeric storage requires F32, F64 or I64"),
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
            let expected_selected = integers(&bank, "selected_face_ids")?;
            pass &= selected
                == expected_selected
                    .iter()
                    .map(|&v| v as usize)
                    .collect::<Vec<_>>();
            let control = integers(&bank, "vertex_control_face_ids")?;
            ensure!(
                control.len() == expected_selected.len() + 4,
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
            let filled_faces = mesh.faces.clone();
            let (selected, used) = mesh.filter_components(0.005)?;
            let expected = fixture(&root.join(&arm).join("filter-maps.safetensors"))?;
            let expected_selected = integers(&expected, "selected_filled_face_ids")?;
            let expected_used = integers(&expected, "used_original_vertex_ids")?;
            let raw_vertices = raw.tensor("vertices")?;
            let vertices_exact = mesh.vertices.len() == 3 * expected_used.len()
                && mesh
                    .vertices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .zip(&expected_used)
                    .all(|(point, &original)| {
                        (0..3).all(|axis| {
                            let offset = 4 * (3 * original as usize + axis);
                            point[axis]
                                == f32::from_le_bytes(
                                    raw_vertices.data()[offset..offset + 4].try_into().unwrap(),
                                )
                        })
                    });
            let faces_exact = mesh.faces.len() == 3 * expected_selected.len()
                && mesh
                    .faces
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .zip(&expected_selected)
                    .all(|(triangle, &original)| {
                        (0..3).all(|axis| {
                            triangle[axis] >= 0
                                && expected_used.get(triangle[axis] as usize)
                                    == Some(&filled_faces[3 * original as usize + axis])
                        })
                    });
            let exact = selected
                == expected_selected
                    .iter()
                    .map(|&v| v as usize)
                    .collect::<Vec<_>>()
                && used
                    == expected_used
                        .iter()
                        .map(|&v| v as usize)
                        .collect::<Vec<_>>()
                && vertices_exact
                && faces_exact;
            defects += usize::from(!exact);
            records += 1;
            println!(
                "{} N5_M6_FINAL_COMPONENTS case={case} vertices={} faces={} exact_maps={exact} complete_vertices={vertices_exact} complete_faces={faces_exact} common_filled_input=true",
                if exact { "PASS" } else { "DEFECT" },
                mesh.vertices.len() / 3,
                mesh.faces.len() / 3
            );
            drop(filled_faces);
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
