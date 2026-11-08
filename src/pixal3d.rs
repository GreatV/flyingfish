//! Published Pixal3D flow geometry and checkpoint contracts.
use crate::checkpoint::{MappedCheckpoint, PublishedTensor};
use anyhow::{Context, Result, ensure};
use safetensors::Dtype;
use serde::Deserialize;
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug, Deserialize)]
pub struct FlowSpec {
    pub resolution: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub model_channels: usize,
    pub cond_channels: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    pub pe_mode: String,
    pub share_mod: bool,
    pub qk_rms_norm: bool,
    pub qk_rms_norm_cross: bool,
    pub image_attn_mode: String,
    pub dtype: String,
    pub proj_in_channels: Option<usize>,
}

impl FlowSpec {
    pub fn sparse(&self) -> bool {
        self.in_channels != 8
    }
    pub fn texture(&self) -> bool {
        self.in_channels == 64
    }
    pub fn projection_width(&self) -> usize {
        self.proj_in_channels.unwrap_or(self.cond_channels)
    }
    pub fn feed_forward_width(&self) -> usize {
        (self.model_channels as f64 * self.mlp_ratio) as usize
    }

    pub fn read(dir: &Path, stem: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct Definition {
            name: String,
            args: FlowSpec,
        }
        let d: Definition =
            serde_json::from_slice(&std::fs::read(dir.join(format!("{stem}.json")))?)?;
        let s = d.args;
        ensure!(
            (d.name == "SparseStructureFlowModel" && s.in_channels == 8 && s.resolution == 16)
                || (d.name == "ElasticSLatFlowModel"
                    && matches!((s.in_channels, s.out_channels), (32, 32) | (64, 32))
                    && matches!(s.resolution, 32 | 64)),
            "unsupported Pixal3D flow definition"
        );
        ensure!(
            s.out_channels == if s.sparse() { 32 } else { 8 }
                && (!s.texture() || s.resolution == 64)
                && s.model_channels == 1536
                && s.cond_channels == 1024
                && s.num_blocks == 30
                && s.num_heads == 12
                && s.feed_forward_width() == 8192
                && s.pe_mode == "rope"
                && s.share_mod
                && s.qk_rms_norm
                && s.qk_rms_norm_cross
                && s.image_attn_mode == "proj"
                && s.dtype == "bfloat16"
                && s.projection_width() == if s.sparse() { 2048 } else { 1024 },
            "unsupported Pixal3D flow geometry/arithmetic"
        );
        Ok(s)
    }

    pub(crate) fn inventory(&self) -> BTreeMap<String, (Vec<usize>, bool)> {
        let mut out = BTreeMap::new();
        let h = self.model_channels;
        let mut linear = |name: String, output: usize, input: usize, bf16: bool| {
            out.insert(format!("{name}.weight"), (vec![output, input], bf16));
            out.insert(format!("{name}.bias"), (vec![output], bf16));
        };
        for (name, output, input) in [
            ("input_layer", h, self.in_channels),
            ("out_layer", self.out_channels, h),
            ("t_embedder.mlp.0", h, 256),
            ("t_embedder.mlp.2", h, h),
            ("adaLN_modulation.1", 6 * h, h),
        ] {
            linear(name.into(), output, input, false);
        }
        for i in 0..self.num_blocks {
            for (name, output, input) in [
                ("self_attn.to_qkv", 3 * h, h),
                ("self_attn.to_out", h, h),
                ("cross_attn.cross_attn_block.to_q", h, h),
                (
                    "cross_attn.cross_attn_block.to_kv",
                    2 * h,
                    self.cond_channels,
                ),
                ("cross_attn.cross_attn_block.to_out", h, h),
                ("cross_attn.proj_linear", h, self.projection_width()),
                ("mlp.mlp.0", self.feed_forward_width(), h),
                ("mlp.mlp.2", h, self.feed_forward_width()),
            ] {
                linear(format!("blocks.{i}.{name}"), output, input, true);
            }
        }
        for i in 0..self.num_blocks {
            out.insert(format!("blocks.{i}.modulation"), (vec![6 * h], false));
            for name in ["norm2.weight", "norm2.bias"] {
                out.insert(format!("blocks.{i}.{name}"), (vec![h], false));
            }
            for prefix in ["self_attn", "cross_attn.cross_attn_block"] {
                for qk in ["q", "k"] {
                    out.insert(
                        format!("blocks.{i}.{prefix}.{qk}_rms_norm.gamma"),
                        (vec![self.num_heads, h / self.num_heads], false),
                    );
                }
            }
        }
        if !self.sparse() {
            out.insert(
                "rope_phases".into(),
                (vec![self.resolution.pow(3), h / self.num_heads / 2], false),
            );
        }
        out
    }

    pub(crate) fn validate_tensors(
        &self,
        tensors: &BTreeMap<String, PublishedTensor<'_>>,
    ) -> Result<usize> {
        let inventory = self.inventory();
        ensure!(
            tensors.keys().eq(inventory.keys()),
            "Pixal3D checkpoint missing/unexpected keys"
        );
        let mut bytes = 0usize;
        for (name, (shape, bf16)) in inventory {
            let tensor = &tensors[&name];
            let dtype = if name == "rope_phases" { "C64" } else { "F32" };
            ensure!(
                tensor.shape == shape && tensor.dtype == dtype,
                "{name}: stored {} {:?}, expected {dtype} {shape:?}",
                tensor.dtype,
                tensor.shape
            );
            bytes = bytes
                .checked_add(if bf16 {
                    tensor.data.len() / 2
                } else {
                    tensor.data.len()
                })
                .context("Pixal3D runtime weight bytes overflow")?;
        }
        Ok(bytes)
    }
}

pub struct FlowCheckpoint {
    pub spec: FlowSpec,
    pub mapped: MappedCheckpoint,
    pub runtime_bytes: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hi3DFlowSpec {
    pub resolution: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub model_channels: usize,
    pub cond_channels: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    pub patch_size: usize,
    pub num_io_res_blocks: usize,
    pub io_block_channels: Vec<usize>,
    pub use_fp16: bool,
    pub latent_mean: Vec<f32>,
    pub latent_std: Vec<f32>,
}

impl Hi3DFlowSpec {
    pub fn validate_grid(&self, coordinates: &[[i32; 3]], resolution: usize) -> Result<()> {
        ensure!(
            matches!((self.resolution, resolution), (64, 64) | (128, 128 | 192)),
            "unsupported flow training/execution grids {}/{}",
            self.resolution,
            resolution
        );
        ensure!(
            !coordinates.is_empty()
                && coordinates
                    .iter()
                    .flatten()
                    .all(|&v| v >= 0 && (v as usize) < resolution),
            "flow coordinates outside execution grid {resolution}"
        );
        Ok(())
    }

    pub fn read(root: &Path, component: &str) -> Result<Self> {
        let resolution = match component {
            "transformer_512" => 64,
            "transformer_1024" => 128,
            _ => anyhow::bail!("unsupported Hi3D flow component {component}"),
        };
        let spec: Self =
            serde_json::from_slice(&std::fs::read(root.join(component).join("config.json"))?)?;
        ensure!(
            spec.resolution == resolution
                && spec.in_channels == 32
                && spec.out_channels == 32
                && spec.model_channels == 2112
                && spec.cond_channels == 2048
                && spec.num_blocks == 24
                && spec.num_heads == 22
                && spec.mlp_ratio == 4.0
                && spec.patch_size == 2
                && spec.num_io_res_blocks == 2
                && spec.io_block_channels == [256]
                && spec.use_fp16,
            "unsupported Hi3D flow geometry/arithmetic"
        );
        ensure!(
            spec.latent_mean.len() == spec.in_channels
                && spec.latent_std.len() == spec.in_channels
                && spec.latent_mean.iter().all(|x| x.is_finite())
                && spec.latent_std.iter().all(|x| x.is_finite() && *x > 0.0),
            "invalid Hi3D latent normalization"
        );
        Ok(spec)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn flow_spec(&self) -> FlowSpec {
        FlowSpec {
            resolution: self.resolution,
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            model_channels: self.model_channels,
            cond_channels: self.cond_channels,
            num_blocks: self.num_blocks,
            num_heads: self.num_heads,
            mlp_ratio: self.mlp_ratio,
            pe_mode: "rope".into(),
            share_mod: true,
            qk_rms_norm: true,
            qk_rms_norm_cross: false,
            image_attn_mode: "full".into(),
            dtype: "float16".into(),
            proj_in_channels: None,
        }
    }

    pub(crate) fn inventory(&self) -> BTreeMap<String, (Vec<usize>, &'static str)> {
        let mut out = BTreeMap::new();
        let h = self.model_channels;
        let mut linear = |name: String, output: usize, input: usize, dtype| {
            out.insert(format!("{name}.weight"), (vec![output, input], dtype));
            out.insert(format!("{name}.bias"), (vec![output], dtype));
        };
        for (name, output, input) in [
            ("input_layer", 256, self.in_channels),
            ("out_layer", self.out_channels, 256),
            ("t_embedder.mlp.0", h, 256),
            ("t_embedder.mlp.2", h, h),
            ("adaLN_modulation.1", 6 * h, h),
        ] {
            linear(name.into(), output, input, "F32");
        }
        for block in 0..self.num_blocks {
            for (name, output, input) in [
                ("self_attn.to_qkv", 3 * h, h),
                ("self_attn.to_out", h, h),
                ("cross_attn.to_q", h, h),
                ("cross_attn.to_kv", 2 * h, self.cond_channels),
                ("cross_attn.to_out", h, h),
                ("mlp.mlp.0", 4 * h, h),
                ("mlp.mlp.2", h, 4 * h),
            ] {
                linear(format!("blocks.{block}.{name}"), output, input, "F16");
            }
        }
        for (name, input, output) in [
            ("input_blocks.0", 256, 256),
            ("input_blocks.1", 256, h),
            ("out_blocks.0", 2 * h, 256),
            ("out_blocks.1", 512, 256),
        ] {
            let hidden = output / 2;
            linear(format!("{name}.emb_layers.1"), 2 * hidden, h, "F16");
            linear(format!("{name}.linear"), output, hidden, "F16");
            if input != output {
                linear(format!("{name}.skip_connection"), output, input, "F16");
            }
        }
        for block in 0..self.num_blocks {
            let prefix = format!("blocks.{block}");
            out.insert(format!("{prefix}.modulation"), (vec![1, 6 * h], "F32"));
            for name in ["norm2.weight", "norm2.bias"] {
                out.insert(format!("{prefix}.{name}"), (vec![h], "F32"));
            }
            for role in ["q", "k"] {
                out.insert(
                    format!("{prefix}.self_attn.{role}_rms_norm.gamma"),
                    (vec![self.num_heads, h / self.num_heads], "F32"),
                );
            }
        }
        for (name, input, output) in [
            ("input_blocks.0", 256, 256),
            ("input_blocks.1", 256, h),
            ("out_blocks.0", 2 * h, 256),
            ("out_blocks.1", 512, 256),
        ] {
            let hidden = output / 2;
            for (conv, channels) in [("conv1", input), ("conv2", hidden)] {
                out.insert(
                    format!("{name}.{conv}.conv.weight"),
                    (vec![hidden, 3, 3, 3, channels], "F16"),
                );
                out.insert(format!("{name}.{conv}.conv.bias"), (vec![hidden], "F16"));
            }
            for parameter in ["weight", "bias"] {
                out.insert(format!("{name}.norm1.{parameter}"), (vec![input], "F32"));
            }
        }
        out
    }

    pub(crate) fn validate_tensors(
        &self,
        tensors: &BTreeMap<String, PublishedTensor<'_>>,
    ) -> Result<usize> {
        let inventory = self.inventory();
        ensure!(
            tensors.keys().eq(inventory.keys()),
            "Hi3D checkpoint missing/unexpected keys"
        );
        let mut bytes = 0usize;
        for (name, (shape, dtype)) in inventory {
            let tensor = &tensors[&name];
            ensure!(
                tensor.shape == shape && tensor.dtype == dtype,
                "{name}: stored {} {:?}, expected {dtype} {shape:?}",
                tensor.dtype,
                tensor.shape
            );
            bytes = bytes
                .checked_add(tensor.data.len())
                .context("flow weight bytes overflow")?;
        }
        Ok(bytes)
    }
}

pub struct Hi3DFlowCheckpoint {
    pub spec: Hi3DFlowSpec,
    pub mapped: MappedCheckpoint,
    pub runtime_bytes: usize,
}
impl Hi3DFlowCheckpoint {
    pub fn open(root: &Path, component: &str) -> Result<Self> {
        let spec = Hi3DFlowSpec::read(root, component)?;
        let mapped = MappedCheckpoint::open_component(&root.join(component), "model")?;
        let runtime_bytes = spec.validate_tensors(&mapped.component_tensors()?)?;
        Ok(Self {
            spec,
            mapped,
            runtime_bytes,
        })
    }
}
impl FlowCheckpoint {
    pub fn open(dir: &Path, stem: &str) -> Result<Self> {
        let mapped = MappedCheckpoint::open_component(dir, stem)?;
        let spec = FlowSpec::read(dir, stem)?;
        let runtime_bytes = spec.validate_tensors(&mapped.component_tensors()?)?;
        Ok(Self {
            spec,
            mapped,
            runtime_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DecoderKind {
    Structure,
    Shape,
    Texture,
    PointTransformer,
}
pub struct DecoderCheckpoint {
    pub kind: DecoderKind,
    pub mapped: MappedCheckpoint,
    pub runtime_bytes: usize,
}
impl DecoderKind {
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn validate_point_rows(rows: usize) -> Result<()> {
        ensure!(
            (1..=12_000_000).contains(&rows),
            "point residual stage requires 1..=12000000 sites"
        );
        Ok(())
    }
    pub fn stem(self) -> &'static str {
        match self {
            Self::Structure => "ss_dec_conv3d_16l8_fp16",
            Self::Shape => "shape_dec_next_dc_f16c32_fp16",
            Self::Texture => "tex_dec_next_dc_f16c32_fp16",
            Self::PointTransformer => "model",
        }
    }
    pub(crate) fn inventory(self) -> BTreeMap<String, (Vec<usize>, Dtype, bool)> {
        let mut tensors = BTreeMap::new();
        let mut pair =
            |name: String, shape: Vec<usize>, width: usize, dtype: Dtype, f32_runtime: bool| {
                tensors.insert(format!("{name}.weight"), (shape, dtype, f32_runtime));
                tensors.insert(format!("{name}.bias"), (vec![width], dtype, f32_runtime));
            };
        match self {
            Self::PointTransformer => return Self::point_transformer_inventory(),
            Self::Structure => {
                pair(
                    "input_layer".into(),
                    vec![512, 8, 3, 3, 3],
                    512,
                    Dtype::F32,
                    true,
                );
                pair(
                    "out_layer.2".into(),
                    vec![1, 32, 3, 3, 3],
                    1,
                    Dtype::F32,
                    true,
                );
                pair("out_layer.0".into(), vec![32], 32, Dtype::F32, true);
                for (name, c) in [
                    ("middle_block.0", 512),
                    ("middle_block.1", 512),
                    ("blocks.0", 512),
                    ("blocks.1", 512),
                    ("blocks.3", 128),
                    ("blocks.4", 128),
                    ("blocks.6", 32),
                    ("blocks.7", 32),
                ] {
                    for j in 1..=2 {
                        pair(format!("{name}.norm{j}"), vec![c], c, Dtype::F32, true);
                        pair(
                            format!("{name}.conv{j}"),
                            vec![c, c, 3, 3, 3],
                            c,
                            Dtype::F16,
                            false,
                        );
                    }
                }
                for (block, ci, co) in [(2, 512, 1024), (5, 128, 256)] {
                    pair(
                        format!("blocks.{block}.conv"),
                        vec![co, ci, 3, 3, 3],
                        co,
                        Dtype::F16,
                        false,
                    );
                }
            }
            Self::Shape | Self::Texture => {
                pair("from_latent".into(), vec![1024, 32], 1024, Dtype::F16, true);
                let output = if matches!(self, Self::Texture) { 6 } else { 7 };
                pair(
                    "output_layer".into(),
                    vec![output, 64],
                    output,
                    Dtype::F16,
                    true,
                );
                for (stage, (c, blocks, co)) in
                    [(1024, 4, 512), (512, 16, 256), (256, 8, 128), (128, 4, 64)]
                        .into_iter()
                        .enumerate()
                {
                    for block in 0..blocks {
                        let name = format!("blocks.{stage}.{block}");
                        pair(
                            format!("{name}.conv"),
                            vec![c, 3, 3, 3, c],
                            c,
                            Dtype::F16,
                            false,
                        );
                        pair(format!("{name}.norm"), vec![c], c, Dtype::F16, true);
                        pair(
                            format!("{name}.mlp.0"),
                            vec![4 * c, c],
                            4 * c,
                            Dtype::F16,
                            false,
                        );
                        pair(
                            format!("{name}.mlp.2"),
                            vec![c, 4 * c],
                            c,
                            Dtype::F16,
                            false,
                        );
                    }
                    let name = format!("blocks.{stage}.{blocks}");
                    pair(format!("{name}.norm1"), vec![c], c, Dtype::F16, true);
                    pair(
                        format!("{name}.conv1"),
                        vec![8 * co, 3, 3, 3, c],
                        8 * co,
                        Dtype::F16,
                        false,
                    );
                    pair(
                        format!("{name}.conv2"),
                        vec![co, 3, 3, 3, co],
                        co,
                        Dtype::F16,
                        false,
                    );
                    if matches!(self, Self::Shape) {
                        pair(
                            format!("{name}.to_subdiv"),
                            vec![8, c],
                            8,
                            Dtype::F16,
                            false,
                        );
                    }
                }
            }
        }
        tensors
    }
    fn point_transformer_inventory() -> BTreeMap<String, (Vec<usize>, Dtype, bool)> {
        let mut tensors = BTreeMap::new();
        let mut norms = vec![("decoder.conv_out.conv.0".into(), 128)];
        let mut convolutions = vec![
            ("decoder.z_to_f.conv.0".into(), 32, 512),
            ("decoder.conv_out.conv.2".into(), 128, 40),
        ];
        let mut linears = Vec::new();
        let mut residuals = vec![
            ("decoder.mid.block_in".into(), 512, 512),
            ("decoder.mid.block_out".into(), 512, 512),
        ];
        for (stage, input, output) in [(3, 512, 512), (2, 512, 256), (1, 256, 128), (0, 128, 128)] {
            for (block, ci) in [(0, input), (1, output)] {
                residuals.push((format!("decoder.dec.dec{stage}.block{block}"), ci, output));
            }
            if stage > 0 {
                let name = format!("decoder.dec.dec{stage}.up.upsample");
                norms.push((format!("{name}.cls_head.fc.0"), output));
                convolutions.push((format!("{name}.cls_head.fc.2"), output, 8));
                linears.push((format!("{name}.proj"), output, 8 * output, false));
                tensors.insert(
                    format!("{name}.inv_grid_coord"),
                    (vec![8, 3], Dtype::I64, false),
                );
            }
        }
        for (name, input, output) in residuals {
            for (index, channels) in [(0, input), (3, output)] {
                norms.push((format!("{name}.block.{index}"), channels));
            }
            for (index, channels) in [(2, input), (5, output)] {
                convolutions.push((format!("{name}.block.{index}"), channels, output));
            }
            if input != output {
                linears.push((format!("{name}.res_conv"), input, output, true));
            }
        }
        for block in 0..2 {
            let name = format!("decoder.mid.block{block}");
            convolutions.push((format!("{name}.cpe.0"), 512, 512));
            for (suffix, input, output) in [
                ("cpe.1", 512, 512),
                ("attn.qkv", 512, 1536),
                ("attn.proj", 512, 512),
                ("mlp.0.fc1", 512, 2048),
                ("mlp.0.fc2", 2048, 512),
            ] {
                linears.push((format!("{name}.{suffix}"), input, output, true));
            }
            for suffix in ["cpe.2", "norm1.0", "norm2.0"] {
                for parameter in ["weight", "bias"] {
                    tensors.insert(
                        format!("{name}.{suffix}.{parameter}"),
                        (vec![512], Dtype::F32, true),
                    );
                }
            }
        }
        for (name, channels) in norms {
            for parameter in ["weights", "bias"] {
                tensors.insert(
                    format!("{name}.{parameter}"),
                    (vec![1, channels], Dtype::F32, true),
                );
            }
        }
        for (name, input, output) in convolutions {
            tensors.insert(
                format!("{name}.weight"),
                (vec![output, 3, 3, 3, input], Dtype::F32, true),
            );
            tensors.insert(format!("{name}.bias"), (vec![output], Dtype::F32, true));
        }
        for (name, input, output, bias) in linears {
            tensors.insert(
                format!("{name}.weight"),
                (vec![output, input], Dtype::F32, false),
            );
            if bias {
                tensors.insert(format!("{name}.bias"), (vec![output], Dtype::F32, false));
            }
        }
        tensors
    }
    fn validate(self, config: &Value) -> Result<()> {
        if matches!(self, Self::PointTransformer) {
            ensure!(
                config["model"]["type"] == "SparconvVAE"
                    && config["model"]["decoder_cfg"]["type"] == "PTv3MeshDecoder"
                    && config["model"]["decoder_cfg"]["z_channels"] == 32,
                "unsupported point transformer decoder definition"
            );
            return Ok(());
        }
        let args = &config["args"];
        ensure!(
            args["use_fp16"] == true,
            "decoder precision differs from published contract"
        );
        match self {
            Self::Structure => ensure!(
                config["name"] == "SparseStructureDecoder"
                    && args["out_channels"] == 1
                    && args["latent_channels"] == 8
                    && args["num_res_blocks"] == 2
                    && args["num_res_blocks_middle"] == 2
                    && args["channels"] == serde_json::json!([512, 128, 32])
                    && args.get("norm_type").is_none_or(|v| v == "layer"),
                "unsupported dense structure decoder"
            ),
            Self::Shape | Self::Texture => ensure!(
                (if matches!(self, Self::Shape) {
                    config["name"] == "FlexiDualGridVaeDecoder" && args["resolution"] == 256
                } else {
                    config["name"] == "SparseUnetVaeDecoder"
                        && args["out_channels"] == 6
                        && args["pred_subdiv"] == false
                }) && args["latent_channels"] == 32
                    && args["model_channels"] == serde_json::json!([1024, 512, 256, 128, 64])
                    && args["num_blocks"] == serde_json::json!([4, 16, 8, 4, 0])
                    && args["block_type"] == serde_json::json!(vec!["SparseConvNeXtBlock3d"; 5])
                    && args["up_block_type"] == serde_json::json!(vec!["SparseResBlockC2S3d"; 4])
                    && args["block_args"] == serde_json::json!([{}, {}, {}, {}, {}]),
                "unsupported sparse shape decoder"
            ),
            Self::PointTransformer => unreachable!(),
        }
        Ok(())
    }
    pub(crate) fn validate_tensors(
        self,
        tensors: &BTreeMap<String, PublishedTensor<'_>>,
    ) -> Result<()> {
        let inventory = self.inventory();
        ensure!(
            tensors.keys().eq(inventory.keys()),
            "decoder checkpoint missing/unexpected keys"
        );
        for (name, (shape, dtype, _)) in inventory {
            let tensor = &tensors[&name];
            ensure!(
                tensor.shape == shape && tensor.dtype == format!("{dtype:?}"),
                "decoder tensor {name} has unexpected stored dtype/layout"
            );
            if dtype == Dtype::I64 {
                for (index, element) in tensor.data.as_chunks::<8>().0.iter().enumerate() {
                    let child = index / 3;
                    let axis = index % 3;
                    let expected = ((child >> (2 - axis)) & 1) as i64;
                    ensure!(
                        *element == expected.to_le_bytes(),
                        "decoder child offset {name} differs at {index}"
                    );
                }
            }
        }
        Ok(())
    }
}
impl DecoderCheckpoint {
    pub fn open(dir: &Path, kind: DecoderKind) -> Result<Self> {
        let config_name = if matches!(kind, DecoderKind::PointTransformer) {
            "config"
        } else {
            kind.stem()
        };
        let config: Value =
            serde_json::from_slice(&std::fs::read(dir.join(format!("{config_name}.json")))?)?;
        kind.validate(&config)?;
        let mapped = MappedCheckpoint::open_component(dir, kind.stem())?;
        kind.validate_tensors(&mapped.component_tensors()?)?;
        let mut runtime_bytes = 0usize;
        for (name, (_, dtype, float_runtime)) in kind.inventory() {
            let tensor = mapped.tensor(&name)?;
            let count = tensor.shape().iter().try_fold(1usize, |n, &d| {
                n.checked_mul(d).context("decoder shape overflow")
            })?;
            let width = if dtype == Dtype::I64 {
                8
            } else if float_runtime {
                4
            } else {
                2
            };
            runtime_bytes = runtime_bytes
                .checked_add(count.checked_mul(width).context("decoder bytes overflow")?)
                .context("decoder bytes overflow")?;
        }
        Ok(Self {
            kind,
            mapped,
            runtime_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires FF_HI3D_MODEL; CPU only"]
    fn hi3d_decoder_checkpoint_cpu() -> Result<()> {
        DecoderKind::validate_point_rows(12_000_000)?;
        ensure!(
            DecoderKind::validate_point_rows(12_000_001).is_err(),
            "unsupported residual branch accepted"
        );
        let root = std::env::var_os("FF_HI3D_MODEL").context("FF_HI3D_MODEL required")?;
        let checkpoint =
            DecoderCheckpoint::open(&Path::new(&root).join("vae"), DecoderKind::PointTransformer)?;
        ensure!(
            checkpoint.runtime_bytes == 357_128_768,
            "decoder runtime byte count"
        );
        let mut tensors = checkpoint.mapped.component_tensors()?;
        ensure!(tensors.len() == 144, "published decoder tensor count");
        ensure!(
            tensors.values().map(|t| t.data.len()).sum::<usize>() == 376_614_208,
            "published decoder byte count"
        );
        let name = "decoder.z_to_f.conv.0.weight";
        let original = tensors[name].shape.clone();
        tensors.get_mut(name).context("convolution missing")?.shape = vec![512, 1, 9, 3, 32];
        ensure!(
            checkpoint.kind.validate_tensors(&tensors).is_err(),
            "same-numel convolution extent control accepted"
        );
        tensors.get_mut(name).context("convolution missing")?.shape = original;
        let name = "decoder.dec.dec1.up.upsample.inv_grid_coord";
        let mut wrong = tensors[name].data.to_vec();
        wrong[3 * 8] = 1;
        wrong[5 * 8] = 0;
        tensors.get_mut(name).context("child table missing")?.data = &wrong;
        ensure!(
            checkpoint.kind.validate_tensors(&tensors).is_err(),
            "X/Z child offset control accepted"
        );
        let reference = std::env::var_os("FF_HI3D_DECODER_REFERENCE")
            .context("FF_HI3D_DECODER_REFERENCE required")?;
        let reference = Path::new(&reference);
        let freeze: Value = serde_json::from_slice(&std::fs::read(reference.join("FROZEN.json"))?)?;
        ensure!(
            freeze["M0_complete"] == true && freeze["reference_protocol"] == "PASS",
            "decoder reference freeze is incomplete"
        );
        let maps = std::fs::read(reference.join("CPU-r3/maps.safetensors"))?;
        let maps = safetensors::SafeTensors::deserialize(&maps)?;
        let read = |name: &str| -> Result<Vec<i64>> {
            let tensor = maps.tensor(name)?;
            match tensor.dtype() {
                Dtype::I64 => Ok(tensor
                    .data()
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|v| i64::from_le_bytes(*v))
                    .collect()),
                Dtype::I32 => Ok(tensor
                    .data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| i32::from_le_bytes(*v) as i64)
                    .collect()),
                _ => anyhow::bail!("invalid sparse map dtype"),
            }
        };
        let cases = freeze["exact_serialization_maps"]
            .as_object()
            .context("serialization cells missing")?;
        ensure!(cases.len() == 9, "serialization case count");
        for (name, _) in cases {
            let name = name
                .strip_prefix("serialization.")
                .context("serialization prefix missing")?;
            let grid = read(&format!("{name}.grid"))?;
            let batch = read(&format!("{name}.batch"))?;
            let coordinates = grid
                .as_chunks::<3>()
                .0
                .iter()
                .zip(batch)
                .map(|(p, b)| {
                    Ok([
                        i32::try_from(b)?,
                        i32::try_from(p[0])?,
                        i32::try_from(p[1])?,
                        i32::try_from(p[2])?,
                    ])
                })
                .collect::<Result<Vec<_>>>()?;
            let sites = crate::sparse::SparseSites::new(coordinates)?;
            let actual = sites.serialized_windows(48)?;
            for (suffix, value) in [
                (
                    "keys",
                    actual
                        .keys
                        .iter()
                        .flatten()
                        .map(|&x| x as i64)
                        .collect::<Vec<_>>(),
                ),
                (
                    "order",
                    actual.order.iter().flatten().map(|&x| x as i64).collect(),
                ),
                (
                    "inverse",
                    actual.inverse.iter().flatten().map(|&x| x as i64).collect(),
                ),
                ("pad", actual.pad.iter().map(|&x| x as i64).collect()),
                ("unpad", actual.unpad.iter().map(|&x| x as i64).collect()),
                (
                    "cu_seqlens",
                    actual.offsets.iter().map(|&x| x as i64).collect(),
                ),
            ] {
                ensure!(
                    value == read(&format!("{name}.{suffix}"))?,
                    "serialized map {name}.{suffix} differs"
                );
            }
            println!(
                "PASS HI3D_DECODER_MAP case={name} rows={} padded_rows={} windows={} exact=true",
                sites.coordinates().len(),
                actual.pad.len(),
                actual.offsets.len() - 1
            );
        }
        println!(
            "PASS HI3D_DECODER_LOAD tensors=144 published_bytes=376614208 runtime_bytes=357128768 CPU_child_bytes=576 convolutions=27 child_tables=3 missing=0 unexpected=0 initialized=0 physical_extent_control=rejected child_axis_control=rejected"
        );
        Ok(())
    }
    #[test]
    #[ignore = "requires FF_HI3D_MODEL; CPU only"]
    fn hi3d_flow_checkpoint_cpu() -> Result<()> {
        let root = std::env::var_os("FF_HI3D_MODEL").context("FF_HI3D_MODEL required")?;
        for component in ["transformer_512", "transformer_1024"] {
            let checkpoint = Hi3DFlowCheckpoint::open(Path::new(&root), component)?;
            let resolution = checkpoint.spec.resolution;
            checkpoint.spec.validate_grid(&[[0, 1, 2]], resolution)?;
            ensure!(
                checkpoint
                    .spec
                    .validate_grid(&[[0, 1, resolution as i32]], resolution)
                    .is_err(),
                "out-of-grid coordinate control accepted"
            );
            if resolution == 128 {
                checkpoint.spec.validate_grid(&[[128, 190, 191]], 192)?;
                ensure!(
                    checkpoint
                        .spec
                        .validate_grid(&[[128, 190, 191]], 128)
                        .is_err()
                        && checkpoint.spec.validate_grid(&[[0, 1, 2]], 256).is_err(),
                    "training/execution grid control accepted"
                );
            } else {
                ensure!(
                    checkpoint.spec.validate_grid(&[[0, 1, 2]], 192).is_err(),
                    "low-grid weights accepted for high-grid execution"
                );
            }
            ensure!(
                checkpoint.runtime_bytes == 3_675_626_624,
                "published flow byte count"
            );
            let mut tensors = checkpoint.mapped.component_tensors()?;
            ensure!(tensors.len() == 512, "published flow tensor count");
            let key = "blocks.0.self_attn.q_rms_norm.gamma";
            tensors
                .get_mut(key)
                .context("gain missing")?
                .shape
                .swap(0, 1);
            ensure!(
                checkpoint.spec.validate_tensors(&tensors).is_err(),
                "same-element-count gain axis control accepted"
            );
            tensors
                .get_mut(key)
                .context("gain missing")?
                .shape
                .swap(0, 1);
            tensors
                .get_mut("blocks.0.self_attn.q_rms_norm.gamma")
                .context("gain missing")?
                .dtype = "F16".into();
            ensure!(
                checkpoint.spec.validate_tensors(&tensors).is_err(),
                "cast gain control accepted"
            );
            println!(
                "HI3D_FLOW_LOAD PASS component={component} tensors=512 published_bytes=3675626624 missing=0 unexpected=0 initialized=0 gain_dtype_control=rejected"
            );
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires the published Pixal3D checkpoint; CPU only"]
    fn pixal3d_flow_checkpoint_cpu() -> Result<()> {
        let root = std::env::var_os("FF_PIXAL_MODEL").context("FF_PIXAL_MODEL required")?;
        let root = Path::new(&root).join("ckpts");
        for stem in [
            "ss_flow_img_dit_1_3B_64_bf16",
            "ss_flow_img_dit_1_3B_64_bf16_mv",
            "slat_flow_img2shape_dit_1_3B_512_bf16",
            "slat_flow_img2shape_dit_1_3B_1024_bf16",
            "slat_flow_img2shape_dit_1_3B_512_bf16_mv",
            "slat_flow_img2shape_dit_1_3B_1024_bf16_mv",
            "slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
            "slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv",
        ] {
            let checkpoint = FlowCheckpoint::open(&root, stem)?;
            let expected = if checkpoint.spec.texture() {
                2_808_686_720
            } else if checkpoint.spec.sparse() {
                2_808_490_112
            } else {
                2_715_920_416
            };
            ensure!(
                checkpoint.runtime_bytes == expected,
                "runtime cast inventory differs from definition"
            );
            let mut tensors = checkpoint.mapped.component_tensors()?;
            tensors.remove("blocks.0.self_attn.q_rms_norm.gamma");
            ensure!(
                checkpoint.spec.validate_tensors(&tensors).is_err(),
                "missing Q norm control accepted"
            );
            println!(
                "PASS PIXAL_FLOW_LOAD component={stem} runtime_bytes={expected} missing=0 unexpected=0 new=0 missing_norm_control=rejected"
            );
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires published Pixal3D checkpoints; CPU only"]
    fn pixal3d_decoder_checkpoint_cpu() -> Result<()> {
        let root = std::env::var_os("FF_PIXAL_MODEL").context("FF_PIXAL_MODEL required")?;
        for (kind, keys, expected) in [
            (DecoderKind::Structure, 74, 147_585_412),
            (DecoderKind::Shape, 292, 948_598_492),
            (DecoderKind::Texture, 284, 948_567_448),
        ] {
            let checkpoint = DecoderCheckpoint::open(&Path::new(&root).join("ckpts"), kind)?;
            ensure!(
                checkpoint.mapped.index.weight_map.len() == keys,
                "decoder key count differs"
            );
            ensure!(
                checkpoint.runtime_bytes == expected,
                "decoder runtime dtype inventory differs"
            );
            println!(
                "PASS PIXAL_DECODER_LOAD kind={kind:?} keys={keys} runtime_bytes={} missing=0 unexpected=0",
                checkpoint.runtime_bytes
            );
        }
        Ok(())
    }
}
