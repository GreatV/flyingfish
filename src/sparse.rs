//! Sparse lattice indexing; batch is the leading coordinate.
use anyhow::{Context, Result, ensure};

pub struct SparseSites {
    coordinates: Vec<[i32; 4]>,
    sorted: Vec<([i32; 4], i32)>,
}
impl SparseSites {
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
    pub fn row(&self, point: [i32; 4]) -> Option<i32> {
        self.sorted
            .binary_search_by_key(&point, |&(p, _)| p)
            .ok()
            .map(|i| self.sorted[i].1)
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
                        .and_then(|x| x.checked_add(((child >> axis) & 1) as i32))
                        .context("subdivision coordinate overflow")?;
                }
                coordinates.push(point);
                source.push([row as u32, child as u32]);
            }
        }
        Ok((Self::new(coordinates)?, source))
    }
    #[cfg(test)]
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
    #[cfg(test)]
    pub fn round_unique(&self, source_grid: u32, target_grid: u32) -> Result<Self> {
        let scaled = self.scaled_coordinates(source_grid, target_grid)?;
        let mut coordinates: Vec<_> = self
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
        coordinates.sort_unstable();
        coordinates.dedup();
        Self::new(coordinates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
