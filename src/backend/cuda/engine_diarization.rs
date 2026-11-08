//! Resident F32 bidirectional encoder and subpixel frame-classification head.
use super::{
    Device, blas,
    contracts::{Tensor, TensorMut, view},
    cubin,
    flash::{F32Attention, F32Mask},
    ops,
};
use crate::{
    audio::{LogMel, LogMelBatch},
    backend::gpu::{CudaFunction, CudaSlice, CudaStream, PushKernelArg},
    checkpoint::MappedCheckpoint,
    diarization::DiarizationSpec,
};
use anyhow::{Context, Result, ensure};
use cudarc::cublas::{CudaBlas, sys as bs};
use std::{borrow::Cow, path::Path, sync::Arc};

struct Linear {
    weight: CudaSlice<f32>,
    bias: Option<CudaSlice<f32>>,
    shape: [u32; 2],
}

struct Norm {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
}

struct Layer {
    norm1: Norm,
    norm2: Norm,
    qkv: [Linear; 3],
    output: Linear,
    up: Linear,
    down: Linear,
}

struct Kernels {
    norm: CudaFunction,
    gelu: CudaFunction,
    relu: CudaFunction,
    bias: CudaFunction,
    residual: CudaFunction,
    rotary: CudaFunction,
    pad: CudaFunction,
    im2col: CudaFunction,
}

struct Workspace {
    features: CudaSlice<f32>,
    embeds: CudaSlice<f32>,
    hidden: CudaSlice<f32>,
    normalized: CudaSlice<f32>,
    qkv: [CudaSlice<f32>; 3],
    attention: CudaSlice<f32>,
    delta: CudaSlice<f32>,
    up: CudaSlice<f32>,
    projection: CudaSlice<f32>,
    padded: CudaSlice<f32>,
    columns: CudaSlice<f32>,
    subpixel: CudaSlice<f32>,
    classifier: CudaSlice<f32>,
    logits: CudaSlice<f32>,
}

pub struct Diarization {
    spec: DiarizationSpec,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    kernels: Kernels,
    embedding: Linear,
    input_norm: Norm,
    layers: Vec<Layer>,
    final_norm: Norm,
    projection: Linear,
    convolution: Linear,
    dense: Linear,
    output: Linear,
    cos: CudaSlice<f32>,
    sin: CudaSlice<f32>,
    max_rows: u32,
    attention: F32Attention,
    workspace: Workspace,
    frontend: LogMel,
}

fn upload(
    cp: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    name: &str,
    shape: &[usize],
) -> Result<CudaSlice<f32>> {
    let values = cp.f32_tensor(name, shape)?;
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "non-finite published tensor {name}"
    );
    Ok(stream.clone_htod(&values)?)
}

fn linear(
    cp: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    prefix: &str,
    shape: [u32; 2],
    biased: bool,
) -> Result<Linear> {
    Ok(Linear {
        weight: upload(
            cp,
            stream,
            &format!("{prefix}.weight"),
            &[shape[0] as usize, shape[1] as usize],
        )?,
        bias: biased
            .then(|| upload(cp, stream, &format!("{prefix}.bias"), &[shape[0] as usize]))
            .transpose()?,
        shape,
    })
}

fn norm(cp: &MappedCheckpoint, stream: &Arc<CudaStream>, prefix: &str, width: u32) -> Result<Norm> {
    Ok(Norm {
        weight: upload(cp, stream, &format!("{prefix}.weight"), &[width as usize])?,
        bias: upload(cp, stream, &format!("{prefix}.bias"), &[width as usize])?,
    })
}

fn matrix(
    x: &CudaSlice<f32>,
    stream: &Arc<CudaStream>,
    rows: u32,
    width: u32,
) -> Result<Tensor<f32, 2>> {
    view(x, stream, 0, [rows, width], [u64::from(width), 1])
}

impl Diarization {
    pub fn load(directory: &Path, device: &Device) -> Result<Self> {
        let spec = DiarizationSpec::read(directory)?;
        let frontend = LogMel::new(spec.mel)?;
        let cp = MappedCheckpoint::open(directory)?;
        let (initial_free, _) = device.ctx.mem_get_info()?;
        let stream = device.stream.clone();
        let blas = CudaBlas::new(stream.clone())?;
        unsafe {
            bs::cublasSetMathMode(
                *blas.handle(),
                bs::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
            )
            .result()?;
        }
        let h = spec.hidden;
        let i = spec.intermediate;
        let c = spec.head_hidden;
        let factor = spec.factor;
        let max_rows = spec.max_encoder_rows()?;
        let common = cubin::module(&device.ctx, "q38_common")?;
        let vision = cubin::module(&device.ctx, "vision_ops")?;
        let kernels = Kernels {
            norm: common.load_function("q38_layer_norm_f32")?,
            gelu: common.load_function("q38_gelu_f32")?,
            relu: vision.load_function("activation_f32")?,
            bias: vision.load_function("vision_bias_f32")?,
            residual: cubin::module(&device.ctx, "speech_ops")?
                .load_function("speech_residual_f32")?,
            rotary: cubin::module(&device.ctx, "rope")?.load_function("partial_rotary_f32")?,
            pad: common.load_function("pad_rows_n_f32")?,
            im2col: common.load_function("conv_im2col_f32")?,
        };
        let embedding = linear(
            &cp,
            &stream,
            "model.audio_tower.embedder.projection",
            [h, factor * spec.mel.mel_bins],
            false,
        )?;
        let input_norm = norm(&cp, &stream, "model.audio_tower.input_layer_norm", h)?;
        let mut layers = Vec::with_capacity(spec.layers as usize);
        for index in 0..spec.layers {
            let prefix = format!("model.audio_tower.layers.{index}");
            layers.push(Layer {
                norm1: norm(&cp, &stream, &format!("{prefix}.layer_norm1"), h)?,
                norm2: norm(&cp, &stream, &format!("{prefix}.layer_norm2"), h)?,
                qkv: [
                    linear(
                        &cp,
                        &stream,
                        &format!("{prefix}.self_attn.q_proj"),
                        [h, h],
                        false,
                    )?,
                    linear(
                        &cp,
                        &stream,
                        &format!("{prefix}.self_attn.k_proj"),
                        [h, h],
                        false,
                    )?,
                    linear(
                        &cp,
                        &stream,
                        &format!("{prefix}.self_attn.v_proj"),
                        [h, h],
                        false,
                    )?,
                ],
                output: linear(
                    &cp,
                    &stream,
                    &format!("{prefix}.self_attn.o_proj"),
                    [h, h],
                    true,
                )?,
                up: linear(&cp, &stream, &format!("{prefix}.mlp.fc1"), [i, h], true)?,
                down: linear(&cp, &stream, &format!("{prefix}.mlp.fc2"), [h, i], true)?,
            });
        }
        let final_norm = norm(&cp, &stream, "model.audio_tower.layer_norm", h)?;
        let projection = linear(&cp, &stream, "model.proj", [c, h], true)?;
        let convolution = Linear {
            weight: upload(
                &cp,
                &stream,
                "model.upsampler.conv.weight",
                &[(factor * c) as usize, c as usize, 3],
            )?,
            bias: Some(upload(
                &cp,
                &stream,
                "model.upsampler.conv.bias",
                &[(factor * c) as usize],
            )?),
            shape: [factor * c, 3 * c],
        };
        let dense = linear(&cp, &stream, "classifier.dense", [c, c], true)?;
        let output = linear(
            &cp,
            &stream,
            "classifier.out_proj",
            [spec.speakers, c],
            true,
        )?;
        let dim = h / spec.heads;
        let mut cos = Vec::with_capacity((max_rows * dim) as usize);
        let mut sin = Vec::with_capacity((max_rows * dim) as usize);
        for position in 0..max_rows {
            for d in 0..dim {
                let inverse = 1.0f32
                    / spec
                        .rope_theta
                        .powf((2 * (d % (dim / 2))) as f32 / dim as f32);
                let angle = position as f32 * inverse;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let cos = stream.clone_htod(&cos)?;
        let sin = stream.clone_htod(&sin)?;
        let alloc = |width: u32| stream.alloc_zeros::<f32>((max_rows as usize) * (width as usize));
        let workspace = Workspace {
            features: alloc(factor * spec.mel.mel_bins)?,
            embeds: alloc(h)?,
            hidden: alloc(h)?,
            normalized: alloc(h)?,
            qkv: [alloc(h)?, alloc(h)?, alloc(h)?],
            attention: alloc(h)?,
            delta: alloc(h)?,
            up: alloc(i)?,
            projection: alloc(c)?,
            padded: stream.alloc_zeros(((max_rows + 2) * c) as usize)?,
            columns: alloc(3 * c)?,
            subpixel: alloc(factor * c)?,
            classifier: alloc(factor * c)?,
            logits: alloc(factor * spec.speakers)?,
        };
        let geometry = [spec.heads, max_rows, max_rows, dim, dim];
        let available = device.workspace_budget(initial_free, initial_free, 0)?;
        let bytes = F32Attention::required_bytes(geometry)?.min(available);
        let attention = F32Attention::new(stream.clone(), geometry, bytes)?;
        Ok(Self {
            spec,
            stream,
            blas,
            kernels,
            embedding,
            input_norm,
            layers,
            final_norm,
            projection,
            convolution,
            dense,
            output,
            cos,
            sin,
            max_rows,
            attention,
            workspace,
            frontend,
        })
    }

    fn apply_linear(
        &self,
        input: Tensor<f32, 2>,
        weight: &Linear,
        destination: Tensor<f32, 2>,
    ) -> Result<()> {
        let w = matrix(
            &weight.weight,
            &self.stream,
            weight.shape[0],
            weight.shape[1],
        )?;
        blas::linear_f32_on(&self.blas, w, input, &mut TensorMut(destination))?;
        if let Some(bias) = &weight.bias {
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.bias)
                    .arg(&destination.ptr)
                    .arg(bias)
                    .arg(&destination.ptr)
                    .arg(&destination.shape[0])
                    .arg(&destination.shape[1])
                    .launch(ops::flat(
                        (destination.shape[0] as usize) * (destination.shape[1] as usize),
                    ))?;
            }
        }
        Ok(())
    }

    fn normalize(
        &self,
        input: Tensor<f32, 2>,
        weight: &Norm,
        output: Tensor<f32, 2>,
    ) -> Result<()> {
        ensure!(
            input.shape == output.shape,
            "diarization normalization extent"
        );
        let [rows, width] = input.shape;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.norm)
                .arg(&input.ptr)
                .arg(&weight.weight)
                .arg(&weight.bias)
                .arg(&output.ptr)
                .arg(&rows)
                .arg(&1u32)
                .arg(&width)
                .arg(&u64::from(width))
                .arg(&u64::from(width))
                .arg(&1u64)
                .arg(&u64::from(width))
                .arg(&u64::from(width))
                .arg(&1u64)
                .arg(&1e-5f32)
                .launch(ops::grid(rows as usize, 256))?;
        }
        Ok(())
    }

    fn activation(&self, values: Tensor<f32, 2>, gelu: bool) -> Result<()> {
        let [rows, width] = values.shape;
        let count = (rows as usize) * (width as usize);
        unsafe {
            if gelu {
                self.stream
                    .launch_builder(&self.kernels.gelu)
                    .arg(&values.ptr)
                    .arg(&values.ptr)
                    .arg(&rows)
                    .arg(&1u32)
                    .arg(&width)
                    .arg(&u64::from(width))
                    .arg(&u64::from(width))
                    .arg(&1u64)
                    .arg(&u64::from(width))
                    .arg(&u64::from(width))
                    .arg(&1u64)
                    .arg(&0i32)
                    .launch(ops::flat(count))?;
            } else {
                self.stream
                    .launch_builder(&self.kernels.relu)
                    .arg(&values.ptr)
                    .arg(&values.ptr)
                    .arg(&rows)
                    .arg(&width)
                    .arg(&0i32)
                    .launch(ops::flat(count))?;
            }
        }
        Ok(())
    }

    fn residual(&self, hidden: Tensor<f32, 2>, delta: Tensor<f32, 2>) -> Result<()> {
        ensure!(hidden.shape == delta.shape, "diarization residual extent");
        let count = u64::from(hidden.shape[0]) * u64::from(hidden.shape[1]);
        unsafe {
            self.stream
                .launch_builder(&self.kernels.residual)
                .arg(&hidden.ptr)
                .arg(&delta.ptr)
                .arg(&hidden.ptr)
                .arg(&count)
                .arg(&1f32)
                .launch(ops::flat(count as usize))?;
        }
        Ok(())
    }

    fn rotary(&self, values: Tensor<f32, 2>) -> Result<()> {
        let rows = values.shape[0];
        let dim = self.spec.hidden / self.spec.heads;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rotary)
                .arg(&values.ptr)
                .arg(&self.cos)
                .arg(&self.sin)
                .arg(&(rows as i32))
                .arg(&(self.spec.heads as i32))
                .arg(&(dim as i32))
                .arg(&(dim as i32))
                .launch(ops::flat((rows as usize) * (self.spec.hidden as usize) / 2))?;
        }
        Ok(())
    }

    fn encode(&self, rows: u32, valid: u32) -> Result<()> {
        let h = self.spec.hidden;
        let hidden = matrix(&self.workspace.hidden, &self.stream, rows, h)?;
        let normalized = matrix(&self.workspace.normalized, &self.stream, rows, h)?;
        self.normalize(
            matrix(&self.workspace.embeds, &self.stream, rows, h)?,
            &self.input_norm,
            hidden,
        )?;
        for layer in &self.layers {
            self.normalize(hidden, &layer.norm1, normalized)?;
            for (weight, values) in layer.qkv.iter().zip(&self.workspace.qkv) {
                self.apply_linear(normalized, weight, matrix(values, &self.stream, rows, h)?)?;
            }
            self.rotary(matrix(&self.workspace.qkv[0], &self.stream, rows, h)?)?;
            self.rotary(matrix(&self.workspace.qkv[1], &self.stream, rows, h)?)?;
            let heads = self.spec.heads;
            let dim = h / heads;
            let head_view = |buffer| {
                view(
                    buffer,
                    &self.stream,
                    0,
                    [heads, rows, dim],
                    [u64::from(dim), u64::from(h), 1],
                )
            };
            self.attention.run_cublas(
                &self.blas,
                head_view(&self.workspace.qkv[0])?,
                head_view(&self.workspace.qkv[1])?,
                head_view(&self.workspace.qkv[2])?,
                &mut TensorMut(head_view(&self.workspace.attention)?),
                F32Mask {
                    causal: false,
                    valid_keys: valid,
                    query_base: 0,
                },
            )?;
            let delta = matrix(&self.workspace.delta, &self.stream, rows, h)?;
            self.apply_linear(
                matrix(&self.workspace.attention, &self.stream, rows, h)?,
                &layer.output,
                delta,
            )?;
            self.residual(hidden, delta)?;
            self.normalize(hidden, &layer.norm2, normalized)?;
            let up = matrix(
                &self.workspace.up,
                &self.stream,
                rows,
                self.spec.intermediate,
            )?;
            self.apply_linear(normalized, &layer.up, up)?;
            self.activation(up, true)?;
            self.apply_linear(up, &layer.down, delta)?;
            self.residual(hidden, delta)?;
        }
        self.normalize(hidden, &self.final_norm, normalized)
    }

    fn classify(&self, rows: u32) -> Result<()> {
        let c = self.spec.head_hidden;
        let projected = matrix(&self.workspace.projection, &self.stream, rows, c)?;
        self.apply_linear(
            matrix(
                &self.workspace.normalized,
                &self.stream,
                rows,
                self.spec.hidden,
            )?,
            &self.projection,
            projected,
        )?;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.pad)
                .arg(&projected.ptr)
                .arg(&self.workspace.padded)
                .arg(&i64::from(rows))
                .arg(&1i64)
                .arg(&i64::from(c))
                .launch(ops::flat(((rows + 2) * c) as usize))?;
            self.stream
                .launch_builder(&self.kernels.im2col)
                .arg(&self.workspace.padded)
                .arg(&self.workspace.columns)
                .arg(&i64::from(rows))
                .arg(&(c as i32))
                .arg(&3i32)
                .arg(&1i32)
                .launch(ops::flat((rows * c * 3) as usize))?;
        }
        self.apply_linear(
            matrix(&self.workspace.columns, &self.stream, rows, c * 3)?,
            &self.convolution,
            matrix(
                &self.workspace.subpixel,
                &self.stream,
                rows,
                c * self.spec.factor,
            )?,
        )?;
        let output_rows = rows * self.spec.factor;
        let subpixel = matrix(&self.workspace.subpixel, &self.stream, output_rows, c)?;
        self.activation(subpixel, false)?;
        let classified = matrix(&self.workspace.classifier, &self.stream, output_rows, c)?;
        self.apply_linear(subpixel, &self.dense, classified)?;
        self.activation(classified, false)?;
        self.apply_linear(
            classified,
            &self.output,
            matrix(
                &self.workspace.logits,
                &self.stream,
                output_rows,
                self.spec.speakers,
            )?,
        )
    }

    /// Includes the zero-padded final stack's subpixel rows.
    pub fn frame_logits(&mut self, features: &[f32], valid_frames: usize) -> Result<Vec<f32>> {
        let mel = self.spec.mel.mel_bins as usize;
        ensure!(
            !features.is_empty()
                && features.len().is_multiple_of(mel)
                && features.iter().all(|x| x.is_finite()),
            "invalid diarization mel features"
        );
        let frames = features.len() / mel;
        ensure!(
            valid_frames > 0 && valid_frames <= frames,
            "invalid diarization valid frames"
        );
        let rows = u32::try_from(frames.div_ceil(self.spec.factor as usize))
            .context("diarization rows overflow")?;
        ensure!(
            rows <= self.max_rows,
            "diarization window {rows} exceeds {}",
            self.max_rows
        );
        let length = (rows as usize) * (self.spec.factor as usize) * mel;
        let padded = if length == features.len() {
            Cow::Borrowed(features)
        } else {
            let mut values = features.to_vec();
            values.resize(length, 0.0);
            Cow::Owned(values)
        };
        {
            let mut destination = self.workspace.features.slice_mut(..padded.len());
            self.stream.memcpy_htod(padded.as_ref(), &mut destination)?;
        }
        self.apply_linear(
            matrix(
                &self.workspace.features,
                &self.stream,
                rows,
                self.spec.factor * self.spec.mel.mel_bins,
            )?,
            &self.embedding,
            matrix(&self.workspace.embeds, &self.stream, rows, self.spec.hidden)?,
        )?;
        let valid = u32::try_from(valid_frames.div_ceil(self.spec.factor as usize))?;
        self.encode(rows, valid)?;
        self.classify(rows)?;
        Ok(self.stream.clone_dtoh(
            &self
                .workspace
                .logits
                .slice(..(rows * self.spec.factor * self.spec.speakers) as usize),
        )?)
    }

    pub fn attention_workspace_bytes(&self) -> usize {
        self.attention.workspace_bytes()
    }

    pub fn audio_features(
        &self,
        waveform: &[f32],
        centered: bool,
        last: bool,
    ) -> Result<LogMelBatch> {
        self.frontend.extract(&[waveform], false, centered, last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{Dtype, SafeTensors};
    use serde_json::Value;
    use std::path::PathBuf;

    #[test]
    #[ignore = "requires GPU lease and external fixtures"]
    fn diarization_encoder_and_head_match_frozen_reference() -> Result<()> {
        let model = PathBuf::from(
            std::env::var_os("FF_DIARIZATION_MODEL").context("diarization model missing")?,
        );
        let fixtures = PathBuf::from(
            std::env::var_os("FF_DIARIZATION_FIXTURES").context("diarization fixtures missing")?,
        );
        let floors = PathBuf::from(
            std::env::var_os("FF_DIARIZATION_FLOORS").context("diarization floors missing")?,
        );
        let floors: Value = serde_json::from_slice(&std::fs::read(floors)?)?;
        ensure!(
            floors["status"] == "FROZEN",
            "diarization references are not frozen"
        );
        let device = Device::new(0)?;
        let mut engine = Diarization::load(&model, &device)?;
        let difference = |left: &[f32], right: &[f32]| -> Result<(f64, f64)> {
            ensure!(
                left.len() == right.len() && !left.is_empty(),
                "frame-logit extent mismatch"
            );
            ensure!(
                left.iter().chain(right).all(|x| x.is_finite()),
                "non-finite frame logits"
            );
            let mut maximum = 0f64;
            let mut square = 0f64;
            for (&a, &b) in left.iter().zip(right) {
                let d = f64::from(a) - f64::from(b);
                maximum = maximum.max(d.abs());
                square += d * d;
            }
            Ok((maximum, (square / left.len() as f64).sqrt()))
        };
        let mut defects = 0;
        let mut first = None;
        for (case, valid) in [
            ("tail7", 7),
            ("production684", 5471),
            ("tail8", 8),
            ("production541", 4327),
            ("tail9", 9),
            ("tail65", 63),
        ] {
            let bytes = std::fs::read(fixtures.join(format!("components/{case}-b1.safetensors")))?;
            let fixture = SafeTensors::deserialize(&bytes)?;
            let read = |key: &str| -> Result<Vec<f32>> {
                let tensor = fixture.tensor(key)?;
                ensure!(tensor.dtype() == Dtype::F32, "frame fixture must be F32");
                Ok(tensor
                    .data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
                    .collect())
            };
            let features = read("features")?;
            let expected = read("logits")?;
            let actual = engine.frame_logits(&features, valid)?;
            let limit = &floors["metrics"][format!("components/{case}.safetensors/logits")]["tau"];
            let max_limit = limit["max_abs"]
                .as_f64()
                .context("missing frame max_abs floor")?;
            let rms_limit = limit["rmse"].as_f64().context("missing frame RMSE floor")?;
            let (maximum, rmse) = difference(&expected, &actual)?;
            let pass = maximum <= max_limit && rmse <= rms_limit;
            defects += usize::from(!pass);
            println!(
                "DIARIZATION_FRAME case={case} max_abs={maximum:e} tau={max_limit:e} rmse={rmse:e} tau_rmse={rms_limit:e} {}",
                if pass { "PASS" } else { "DEFECT" }
            );
            let mut mutant = actual.clone();
            let speakers = engine.spec.speakers as usize;
            for row in 0..actual.len() / speakers {
                let source = (row / 8 * 8) + ((row % 8) ^ 1);
                mutant[row * speakers..(row + 1) * speakers]
                    .copy_from_slice(&actual[source * speakers..(source + 1) * speakers]);
            }
            let (control_abs, control_rmse) = difference(&expected, &mutant)?;
            let detected = control_abs > 10. * max_limit && control_rmse > 10. * rms_limit;
            defects += usize::from(!detected);
            println!(
                "DIARIZATION_PHASE_CONTROL case={case} max_abs={control_abs:e} rmse={control_rmse:e} detected={detected}"
            );
            if case == "tail7" {
                first = Some((features, actual));
            }
        }
        let (features, expected) = first.context("missing repeat case")?;
        let repeat = engine.frame_logits(&features, 7)?;
        let reset = expected
            .iter()
            .zip(&repeat)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        defects += usize::from(!reset);
        let oversized = vec![
            0f32;
            ((engine.max_rows + 1) * engine.spec.factor * engine.spec.mel.mel_bins)
                as usize
        ];
        ensure!(
            engine
                .frame_logits(&oversized, oversized.len() / 128)
                .is_err(),
            "capacity bound accepted oversized window"
        );
        let verdict = if defects == 0 { "PASS" } else { "DEFECT" };
        println!(
            "DIARIZATION_FRAME_GATE {verdict} cells=6 layers=31 phases=8 eager=true scratch_replay_exact={reset} attention_workspace_bytes={} defects={defects}",
            engine.attention_workspace_bytes()
        );
        ensure!(
            defects == 0,
            "diarization frame classification has {defects} DEFECTs"
        );
        Ok(())
    }
}
