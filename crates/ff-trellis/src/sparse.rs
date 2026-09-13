//! Coordinate-preserving sparse geometry for TRELLIS-1. Features stay in Candle;
//! CPU maps only describe connectivity, pooling and attention partitions.
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug)]
pub struct Grid {
    pub coords: Vec<[u32; 4]>,
    neighbors: Vec<u32>,
}

pub struct PoolMap {
    pub coarse: Grid,
    pub inverse: Vec<u32>,
    divisors: Vec<f32>,
}

impl Grid {
    pub fn new(coords: Vec<[u32; 4]>) -> Result<Self> {
        anyhow::ensure!(!coords.is_empty(), "sparse grid is empty");
        let sentinel = u32::try_from(coords.len()).context("too many sparse coordinates")?;
        let lookup = coords
            .iter()
            .enumerate()
            .map(|(i, &coord)| (coord, i as u32))
            .collect::<HashMap<_, _>>();
        anyhow::ensure!(
            lookup.len() == coords.len(),
            "sparse coordinates must be unique"
        );
        let mut neighbors = Vec::with_capacity(
            coords
                .len()
                .checked_mul(27)
                .context("neighbor map overflow")?,
        );
        for &[batch, x, y, z] in &coords {
            for dx in -1i64..=1 {
                for dy in -1i64..=1 {
                    for dz in -1i64..=1 {
                        let xyz = [i64::from(x) + dx, i64::from(y) + dy, i64::from(z) + dz];
                        let index = if xyz.iter().all(|&v| (0..=i64::from(u32::MAX)).contains(&v)) {
                            lookup
                                .get(&[batch, xyz[0] as u32, xyz[1] as u32, xyz[2] as u32])
                                .copied()
                                .unwrap_or(sentinel)
                        } else {
                            sentinel
                        };
                        neighbors.push(index);
                    }
                }
            }
        }
        Ok(Self { coords, neighbors })
    }

    pub fn downsample(&self) -> Result<PoolMap> {
        let mut counts = BTreeMap::<[u32; 4], usize>::new();
        let parent = |c: &[u32; 4]| [c[0], c[1] / 2, c[2] / 2, c[3] / 2];
        for coord in &self.coords {
            *counts.entry(parent(coord)).or_default() += 1;
        }
        let coarse = Grid::new(counts.keys().copied().collect())?;
        let indices = coarse
            .coords
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect::<HashMap<_, _>>();
        let inverse = self.coords.iter().map(|c| indices[&parent(c)]).collect();
        let divisors = counts.values().map(|&count| (count + 1) as f32).collect();
        Ok(PoolMap {
            coarse,
            inverse,
            divisors,
        })
    }

    pub(crate) fn neighbor_tensor(&self, device: &Device) -> Result<Tensor> {
        Ok(Tensor::from_vec(
            self.neighbors.clone(),
            self.neighbors.len(),
            device,
        )?)
    }

    pub(crate) fn partitions(&self, window: Option<(usize, usize)>) -> Result<Vec<Vec<u32>>> {
        if let Some((width, _)) = window {
            anyhow::ensure!(width > 0, "attention window must be positive");
        }
        let mut groups = BTreeMap::<[u64; 4], Vec<u32>>::new();
        for (row, coord) in self.coords.iter().enumerate() {
            let mut key = [u64::from(coord[0]), 0, 0, 0];
            if let Some((width, shift)) = window {
                for axis in 1..4 {
                    key[axis] = (u64::from(coord[axis]) + shift as u64) / width as u64;
                }
            }
            groups.entry(key).or_default().push(row as u32);
        }
        Ok(groups.into_values().collect())
    }
}

impl PoolMap {
    pub fn pool(&self, features: &Tensor) -> Result<Tensor> {
        let (rows, width) = features.dims2()?;
        anyhow::ensure!(
            rows == self.inverse.len(),
            "pooling feature/coordinate mismatch"
        );
        let indices = Tensor::from_vec(self.inverse.clone(), rows, features.device())?;
        let zero = Tensor::zeros(
            (self.coarse.coords.len(), width),
            DType::F32,
            features.device(),
        )?;
        let sum = zero.index_add(&indices, &features.to_dtype(DType::F32)?, 0)?;
        let divisor = Tensor::from_vec(
            self.divisors.clone(),
            (self.divisors.len(), 1),
            features.device(),
        )?;
        Ok(sum.broadcast_div(&divisor)?.to_dtype(features.dtype())?)
    }
    pub fn unpool(&self, features: &Tensor) -> Result<Tensor> {
        anyhow::ensure!(
            features.dim(0)? == self.coarse.coords.len(),
            "unpooling feature/coordinate mismatch"
        );
        Ok(features.index_select(
            &Tensor::from_vec(self.inverse.clone(), self.inverse.len(), features.device())?,
            0,
        )?)
    }
}

/// Submanifold cross-correlation with spconv's `[out,kx,ky,kz,in]` weights.
/// One bounded im2col tile forms a single GEMM, so half-precision convolution
/// accumulates inside the matmul rather than summing 27 rounded half results.
pub(crate) fn conv3d(
    features: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    neighbors: &Tensor,
    chunk: usize,
) -> Result<Tensor> {
    let (rows, input) = features.dims2()?;
    let (output, kx, ky, kz, wi) = weight.dims5()?;
    anyhow::ensure!(
        chunk > 0
            && [kx, ky, kz] == [3, 3, 3]
            && wi == input
            && neighbors.elem_count() == rows * 27,
        "invalid submanifold convolution geometry"
    );
    let padded = Tensor::cat(
        &[
            features,
            &Tensor::zeros((1, input), features.dtype(), features.device())?,
        ],
        0,
    )?;
    let matrix = weight.reshape((output, 27 * input))?.t()?;
    let mut tiles = Vec::new();
    for start in (0..rows).step_by(chunk) {
        let count = chunk.min(rows - start);
        let patch = padded
            .index_select(&neighbors.narrow(0, start * 27, count * 27)?, 0)?
            .reshape((count, 27 * input))?;
        tiles.push(patch.matmul(&matrix)?.broadcast_add(bias)?);
    }
    Ok(Tensor::cat(&tiles, 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pooling_includes_the_initial_zero_and_unpool_restores_input_order() {
        let grid = Grid::new(vec![[0, 1, 1, 1], [0, 0, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0]]).unwrap();
        let map = grid.downsample().unwrap();
        let x = Tensor::new(&[[3f32], [6.], [8.], [10.]], &Device::Cpu).unwrap();
        let coarse = map.pool(&x).unwrap();
        assert_eq!(
            coarse.to_vec2::<f32>().unwrap(),
            vec![vec![3.], vec![4.], vec![5.]]
        );
        assert_eq!(
            map.unpool(&coarse).unwrap().to_vec2::<f32>().unwrap(),
            vec![vec![3.], vec![3.], vec![4.], vec![5.]]
        );
        assert!(Grid::new(vec![[0, 0, 0, 0]; 2]).is_err());
    }
    #[test]
    fn convolution_preserves_axis_order_and_never_connects_different_samples() {
        let grid = Grid::new(vec![
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ])
        .unwrap();
        let x = Tensor::new(&[[1f32], [2.], [3.], [4.], [99.]], &Device::Cpu).unwrap();
        let weights = Tensor::from_vec(
            (1..=27).map(|x| x as f32).collect::<Vec<_>>(),
            (1, 3, 3, 3, 1),
            &Device::Cpu,
        )
        .unwrap();
        let out = conv3d(
            &x,
            &weights,
            &Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap(),
            &grid.neighbor_tensor(&Device::Cpu).unwrap(),
            2,
        )
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
        assert_eq!(out[0][0], 14. + 23. * 2. + 17. * 3. + 15. * 4.);
        assert_eq!(out[4][0], 14. * 99.);
        assert_ne!(
            grid.partitions(Some((2, 0))).unwrap(),
            grid.partitions(Some((2, 1))).unwrap()
        );
    }
}
