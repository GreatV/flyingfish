//! Flexible dual-grid extraction and a portable colored triangle-mesh PLY.
use crate::sparse::Grid;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::{collections::HashMap, io::Write};

pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
    pub attributes: Vec<[f32; 6]>,
}
impl Mesh {
    pub fn from_dual_grid(
        grid: &Grid,
        decoded: &Tensor,
        texture: &Tensor,
        resolution: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            resolution > 0
                && decoded.dims() == [grid.coords.len(), 7]
                && texture.dims() == [grid.coords.len(), 6],
            "dual-grid geometry mismatch"
        );
        let features = decoded
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let colors = texture
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let mut vertices = Vec::with_capacity(features.len());
        let mut attributes = Vec::with_capacity(features.len());
        let mut split = Vec::with_capacity(features.len());
        for (coord, (row, color)) in grid.coords.iter().zip(features.iter().zip(colors)) {
            anyhow::ensure!(
                row.iter().chain(&color).all(|v| v.is_finite()),
                "dual grid has non-finite values"
            );
            let mut vertex = [0.; 3];
            for axis in 0..3 {
                vertex[axis] = (coord[axis + 1] as f32 + 2. / (1. + (-row[axis]).exp()) - 0.5)
                    / resolution as f32
                    - 0.5;
            }
            vertices.push(vertex);
            attributes.push(
                color
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("invalid material channels"))?,
            );
            split.push(row[6].max(0.) + (-row[6].abs()).exp().ln_1p());
        }
        let lookup = grid
            .coords
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect::<HashMap<_, _>>();
        let offsets = [
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        ];
        let mut triangles = Vec::new();
        for (coord, row) in grid.coords.iter().zip(features) {
            for axis in 0..3 {
                if row[3 + axis] > 0. {
                    let mut quad = [0; 4];
                    let mut valid = true;
                    for corner in 0..4 {
                        let mut key = *coord;
                        for dimension in 0..3 {
                            key[dimension + 1] = key[dimension + 1]
                                .checked_add(offsets[axis][corner][dimension])
                                .context("dual-grid coordinate overflow")?;
                        }
                        if let Some(&index) = lookup.get(&key) {
                            quad[corner] = index;
                        } else {
                            valid = false;
                            break;
                        }
                    }
                    if valid {
                        let pattern = if split[quad[0] as usize] * split[quad[2] as usize]
                            > split[quad[1] as usize] * split[quad[3] as usize]
                        {
                            [0, 1, 2, 0, 2, 3]
                        } else {
                            [0, 1, 3, 3, 1, 2]
                        };
                        triangles.extend([
                            [quad[pattern[0]], quad[pattern[1]], quad[pattern[2]]],
                            [quad[pattern[3]], quad[pattern[4]], quad[pattern[5]]],
                        ]);
                    }
                }
            }
        }
        anyhow::ensure!(
            !triangles.is_empty(),
            "dual-grid decoder produced no triangles"
        );
        Ok(Self {
            vertices,
            triangles,
            attributes,
        })
    }
    /// Vertex material values are the decoder's voxel attributes. This export
    /// does not perform UV unwrapping, texture baking or hole filling.
    pub fn write_ply(&self, mut writer: impl Write) -> Result<()> {
        anyhow::ensure!(
            self.vertices.len() == self.attributes.len(),
            "mesh attribute count mismatch"
        );
        writeln!(
            writer,
            "ply\nformat binary_little_endian 1.0\ncomment TRELLIS.2 raw mesh; source axes\nelement vertex {}",
            self.vertices.len()
        )?;
        for axis in ["x", "y", "z"] {
            writeln!(writer, "property float {axis}")?;
        }
        for channel in ["red", "green", "blue", "alpha"] {
            writeln!(writer, "property uchar {channel}")?;
        }
        writeln!(
            writer,
            "property float metallic\nproperty float roughness\nelement face {}\nproperty list uchar uint vertex_indices\nend_header",
            self.triangles.len()
        )?;
        for (vertex, attribute) in self.vertices.iter().zip(&self.attributes) {
            for &value in vertex {
                writer.write_all(&value.to_le_bytes())?;
            }
            for i in [0, 1, 2, 5] {
                writer.write_all(&[(attribute[i].clamp(0., 1.) * 255.).round() as u8])?;
            }
            for i in [3, 4] {
                writer.write_all(&attribute[i].clamp(0., 1.).to_le_bytes())?;
            }
        }
        for triangle in &self.triangles {
            writer.write_all(&[3])?;
            for index in triangle {
                writer.write_all(&index.to_le_bytes())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dual_grid_extracts_the_expected_quad_and_material_fields() {
        let grid = Grid::new(vec![[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0]]).unwrap();
        let mut raw = vec![0f32; 28];
        raw[5] = 1.;
        let raw = Tensor::from_vec(raw, (4, 7), &Device::Cpu).unwrap();
        let material = Tensor::from_vec(vec![0.5f32; 24], (4, 6), &Device::Cpu).unwrap();
        let mesh = Mesh::from_dual_grid(&grid, &raw, &material, 2).unwrap();
        assert_eq!(mesh.vertices[0], [-0.25, -0.25, -0.25]);
        assert_eq!(mesh.triangles, [[0, 1, 3], [3, 1, 2]]);
        let mut bytes = Vec::new();
        mesh.write_ply(&mut bytes).unwrap();
        let end = bytes
            .windows(11)
            .position(|v| v == b"end_header\n")
            .unwrap()
            + 11;
        assert_eq!(bytes.len() - end, 4 * 24 + 2 * 13);
    }
}
