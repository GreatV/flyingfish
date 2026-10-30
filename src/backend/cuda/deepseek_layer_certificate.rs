use super::*;
use crate::backend::cuda::cubin;
use crate::backend::cuda::latent_window_attention::certificate;
use crate::backend::gpu::{LaunchConfig, PushKernelArg};

pub(super) struct Certificate {
    values: [Vec<bf16>; 2],
    valid: bool,
}
impl Certificate {
    pub(super) fn covers(&self, index: usize, values: [f32; 2]) -> bool {
        self.valid && (0..2).all(|side| self.values[side][index].to_f32() == values[side])
    }
}

struct Record<'a>(&'a Path);
impl Record<'_> {
    fn bytes(&self, name: &str, count: usize) -> Result<Vec<u8>> {
        let bytes = std::fs::read(self.0.join(format!("{name}.bin")))?;
        ensure!(
            bytes.len() == count,
            "certificate {name} extent {} != {count}",
            bytes.len()
        );
        Ok(bytes)
    }
    fn bf(&self, name: &str, count: usize) -> Result<Vec<bf16>> {
        Ok(self
            .bytes(name, count * 2)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]])))
            .collect())
    }
    fn f32(&self, name: &str, count: usize) -> Result<Vec<f32>> {
        Ok(self
            .bytes(name, count * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect())
    }
}

fn gamma(n: usize) -> f64 {
    crate::backend::cuda::gamma(n, 2.0f64.powi(-24))
}
fn value(v: bf16) -> f64 {
    f64::from(v.to_f32())
}
fn rounded(v: f64) -> f64 {
    value(bf16::from_f32(v as f32))
}
fn scale(v: u8) -> f64 {
    2.0f64.powi(i32::from(v) - 127)
}

struct Proof {
    layer: usize,
    position: usize,
    defects: usize,
}
impl Proof {
    fn store(
        &mut self,
        stage: &str,
        index: usize,
        values: [bf16; 2],
        reductions: [[f64; 2]; 2],
        n: usize,
    ) {
        let [a, b] = reductions;
        let low = rounded((a[0] - gamma(n) * a[1]).min(b[0] - gamma(n) * b[1]));
        let high = rounded((a[0] + gamma(n) * a[1]).max(b[0] + gamma(n) * b[1]));
        self.interval(stage, index, values, [low, high], reductions, n);
    }
    fn interval(
        &mut self,
        stage: &str,
        index: usize,
        values: [bf16; 2],
        bounds: [f64; 2],
        reductions: [[f64; 2]; 2],
        n: usize,
    ) {
        let valid = values
            .iter()
            .all(|v| value(*v).is_finite() && value(*v) >= bounds[0] && value(*v) <= bounds[1]);
        self.defects += usize::from(!valid);
        if !valid || values[0].to_bits() != values[1].to_bits() {
            println!(
                "LAYER_CHAIN_STORE {}",
                serde_json::json!({"layer":self.layer,"position":self.position,"stage":stage,"index":index,"n":n,"gamma":gamma(n),"candidate":value(values[0]),"publisher":value(values[1]),"lower":bounds[0],"upper":bounds[1],"reductions":reductions,"both_inside":valid})
            );
        }
    }
    fn quant(
        &mut self,
        stage: &str,
        inputs: [&[bf16]; 2],
        codes: [&[u8]; 2],
        scales: [&[u8]; 2],
    ) -> Result<[Vec<bf16>; 2]> {
        let n = inputs[0].len();
        ensure!(
            n.is_multiple_of(32)
                && inputs[1].len() == n
                && codes.iter().all(|x| x.len() == n)
                && scales.iter().all(|x| x.len() == n / 32),
            "quant certificate geometry"
        );
        let mut out = [Vec::with_capacity(n), Vec::with_capacity(n)];
        for block in 0..n / 32 {
            let mut expected = [0.0; 2];
            for side in 0..2 {
                let maximum = inputs[side][block * 32..block * 32 + 32]
                    .iter()
                    .map(|v| v.to_f32().abs())
                    .fold(1e-4f32, f32::max);
                expected[side] = 2.0f32.powf((maximum * (1.0f32 / 448.0)).log2().ceil());
                self.defects +=
                    usize::from(f64::from(expected[side]) != scale(scales[side][block]));
            }
            self.defects += usize::from(scales[0][block] != scales[1][block]);
            for j in 0..32 {
                let i = block * 32 + j;
                for side in 0..2 {
                    let code = crate::fp8::encode_e4m3(
                        (inputs[side][i].to_f32() / expected[side]).clamp(-448.0, 448.0),
                    );
                    self.defects += usize::from(code != codes[side][i]);
                    out[side].push(bf16::from_f32(
                        crate::fp8::decode_e4m3(codes[side][i]) * scale(scales[side][block]) as f32,
                    ));
                }
                let ends = [
                    inputs[0][i].to_f32().min(inputs[1][i].to_f32()),
                    inputs[0][i].to_f32().max(inputs[1][i].to_f32()),
                ];
                let bounds = ends.map(|v| {
                    f64::from(
                        crate::fp8::decode_e4m3(crate::fp8::encode_e4m3(
                            (v / expected[0]).clamp(-448.0, 448.0),
                        )) * expected[0],
                    )
                });
                self.interval(stage, i, [out[0][i], out[1][i]], bounds, [[0.0; 2]; 2], 0);
            }
        }
        Ok(out)
    }
    fn projection(
        &mut self,
        stage: &str,
        input: [&[bf16]; 2],
        output: [&[bf16]; 2],
        weights: (&[u8], &[u8]),
        groups: usize,
    ) -> Result<()> {
        let n = output[0].len();
        let k = input[0].len() / groups;
        ensure!(
            input[0].len() == input[1].len()
                && n == output[1].len()
                && n.is_multiple_of(groups)
                && weights.0.len() == n * k
                && weights.1.len() == n.div_ceil(32) * k.div_ceil(32),
            "projection certificate geometry {stage}"
        );
        for (i, (&a, &b)) in output[0].iter().zip(output[1]).enumerate() {
            let offset = (i / (n / groups)) * k;
            let mut reductions = [[0.0; 2]; 2];
            for j in 0..k {
                let w = f64::from(crate::fp8::decode_e4m3(weights.0[i * k + j]))
                    * scale(weights.1[(i / 32) * k.div_ceil(32) + j / 32]);
                for side in 0..2 {
                    let product = value(input[side][offset + j]) * w;
                    reductions[side][0] += product;
                    reductions[side][1] += product.abs();
                }
            }
            self.store(stage, i, [a, b], reductions, k);
        }
        Ok(())
    }
    fn rotary(
        &mut self,
        stage: &str,
        inputs: [&[bf16]; 2],
        outputs: [&[bf16]; 2],
        coefficients: [&[f32]; 2],
        inverse: bool,
    ) -> Result<()> {
        ensure!(
            inputs
                .iter()
                .chain(outputs.iter())
                .all(|v| v.len() == 32768)
                && coefficients.iter().all(|v| v.len() == 64),
            "rotary certificate geometry"
        );
        for i in 0..32768 {
            let col = i % 512;
            if col < 448 {
                self.interval(
                    stage,
                    i,
                    [outputs[0][i], outputs[1][i]],
                    [
                        value(inputs[0][i]).min(value(inputs[1][i])),
                        value(inputs[0][i]).max(value(inputs[1][i])),
                    ],
                    [[0.0; 2]; 2],
                    0,
                );
                continue;
            }
            let pair = (col - 448) / 2;
            let first = i - col + 448 + pair * 2;
            if stage.starts_with("query") && outputs[0][i] != outputs[1][i] {
                self.defects += usize::from((0..2).any(|j| {
                    coefficients[0][pair * 2 + j].to_bits()
                        != coefficients[1][pair * 2 + j].to_bits()
                }));
            }
            let mut sums = [[0.0; 2]; 2];
            for side in 0..2 {
                let c = f64::from(coefficients[1][pair * 2]);
                let sn =
                    f64::from(coefficients[1][pair * 2 + 1]) * if inverse { -1.0 } else { 1.0 };
                let x = value(inputs[side][first]);
                let y = value(inputs[side][first + 1]);
                let products = if col.is_multiple_of(2) {
                    [x * c, -y * sn]
                } else {
                    [x * sn, y * c]
                };
                sums[side] = [
                    products.iter().sum(),
                    products.iter().map(|v| v.abs()).sum(),
                ];
            }
            self.store(stage, i, [outputs[0][i], outputs[1][i]], sums, 2);
        }
        Ok(())
    }
}

fn trace(engine: &Engine, layer: usize, position: usize) -> Result<(Vec<f32>, Vec<bf16>)> {
    let s = &engine.device.stream;
    let mixer = engine.deepseek_mixer.as_ref().context("mixer missing")?;
    let weights = mixer.weights[layer]
        .as_ref()
        .context("mixer weights missing")?;
    let ring = engine.state.latent_ring(layer)?;
    let mut capacity = 128.max(position + 1);
    let dummy = s.alloc_zeros::<u8>(256 * capacity)?;
    let selections = s.clone_htod(&vec![-1i32; 512])?;
    let count = s.clone_htod(&[position as u32 + 1])?;
    let mut packed = dummy.device_ptr(s).0;
    let mut scales = packed;
    let mut selected = selections.device_ptr(s).0;
    let mut committed = count.device_ptr(s).0;
    let ratio = usize::from(mixer.compress_ratios[layer]).max(1);
    if mixer.compress_ratios[layer] != 0 {
        let state = engine
            .state
            .compressed
            .as_ref()
            .context("compressed state missing")?;
        let owners = state.owners[layer].context("compressed owners missing")?;
        let kv = state
            .kv
            .iter()
            .find(|v| v.owner == owners.kv)
            .context("KV owner missing")?;
        let index = state
            .index
            .iter()
            .find(|v| v.owner == owners.index)
            .context("index owner missing")?;
        packed = kv.codes.device_ptr(s).0;
        scales = kv.scales.device_ptr(s).0;
        selected = index.positions.device_ptr(s).0;
        committed = kv.count.device_ptr(s).0;
        capacity = kv.codes.len() / 256;
    }
    let sequence = s.clone_htod(&[
        position as i32,
        1,
        position as i32 + 1,
        (capacity * ratio).max(position + 1) as i32,
        1,
        0,
        0,
        0,
    ])?;
    let out = s.alloc_zeros::<bf16>(32768)?;
    let stores = s.clone_htod(&vec![f32::NAN; certificate::TRACE_LEN])?;
    let kernel = cubin::module(&engine.device.ctx, "latent_window_attention")?
        .load_function("latent_indexed_attention_trace")?;
    let dims = [64i32, 512, 128, capacity as i32, ratio as i32, 0, 64];
    let scaling = 512.0f32.sqrt().recip();
    unsafe {
        s.launch_builder(&kernel)
            .arg(&engine.work.qkv.device_ptr(s).0)
            .arg(&ring.codes.device_ptr(s).0)
            .arg(&ring.scales.device_ptr(s).0)
            .arg(&packed)
            .arg(&scales)
            .arg(&selected)
            .arg(&committed)
            .arg(&weights.attn_sink.device_ptr(s).0)
            .arg(&mixer.rotary_inv.device_ptr(s).0)
            .arg(&sequence.device_ptr(s).0)
            .arg(&out.device_ptr(s).0)
            .arg(&dims[0])
            .arg(&dims[1])
            .arg(&dims[2])
            .arg(&dims[3])
            .arg(&dims[4])
            .arg(&dims[5])
            .arg(&dims[6])
            .arg(&scaling)
            .arg(&stores.device_ptr(s).0)
            .launch(LaunchConfig {
                grid_dim: (64, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 6656,
            })?;
    }
    Ok((s.clone_dtoh(&stores)?, s.clone_dtoh(&out)?))
}

fn query(
    proof: &mut Proof,
    engine: &Engine,
    input: &[u16],
    record: &Record<'_>,
    actual: &[bf16],
    expected: &[bf16],
) -> Result<()> {
    let s = &engine.device.stream;
    let mixer = engine.deepseek_mixer.as_ref().context("mixer missing")?;
    let weights = mixer.weights[proof.layer]
        .as_ref()
        .context("mixer weights missing")?;
    let stores = mixer.query_stores.borrow();
    let qa = stores.get("q_a").context("Q source wq_a capture missing")?;
    let qn = stores
        .get("q_norm")
        .context("Q source norm capture missing")?;
    let qb = stores.get("q_b").context("Q source wq_b capture missing")?;
    let ra = record.bf("q_a", 1280)?;
    let rn = record.bf("q_norm", 1280)?;
    let rb = record.bf("q_b", 32768)?;
    let source: Vec<_> = input.iter().map(|v| bf16::from_bits(*v)).collect();
    let sc = s.clone_dtoh(&mixer.act_hidden_codes.slice(..5120))?;
    let ss = s.clone_dtoh(&mixer.act_hidden_scales.slice(..160))?;
    let rc = record.bytes("source_codes", 5120)?;
    let rs = record.bytes("source_scales", 160)?;
    let x = proof.quant("source_quant", [&source, &source], [&sc, &rc], [&ss, &rs])?;
    proof.projection(
        "wq_a",
        [&x[0], &x[1]],
        [qa, &ra],
        (
            &s.clone_dtoh(&weights.wq_a.codes)?,
            &s.clone_dtoh(&weights.wq_a.scale)?,
        ),
        1,
    )?;
    let norm_bytes = s.clone_dtoh(&weights.q_norm)?;
    let norm: Vec<_> = norm_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]])))
        .collect();
    let u = 2.0f64.powi(-24);
    let mut bounds = vec![[f64::INFINITY, f64::NEG_INFINITY]; 1280];
    for operand in [qa, &ra] {
        let exact: f64 = operand.iter().map(|v| value(*v).powi(2)).sum();
        let error = gamma(1280) * exact;
        let lo = (exact - error) / 1280.0 * (1.0 - u);
        let hi = (exact + error) / 1280.0 * (1.0 + u);
        let il = ((hi + f64::from(mixer.eps)) * (1.0 + u)).sqrt().recip() * (1.0 - u);
        let ih = ((lo + f64::from(mixer.eps)) * (1.0 - u)).sqrt().recip() * (1.0 + u);
        for i in 0..1280 {
            let a = value(operand[i]) * il;
            let b = value(operand[i]) * ih;
            let l = a.min(b);
            let h = a.max(b);
            let a = (l - u * l.abs()) * value(norm[i]);
            let b = (h + u * h.abs()) * value(norm[i]);
            let l = a.min(b);
            let h = a.max(b);
            bounds[i][0] = bounds[i][0].min(rounded(l - u * l.abs()));
            bounds[i][1] = bounds[i][1].max(rounded(h + u * h.abs()));
        }
    }
    for i in 0..1280 {
        proof.interval("q_norm", i, [qn[i], rn[i]], bounds[i], [[0.0; 2]; 2], 1280);
    }
    let qc = s.clone_dtoh(&mixer.act_lora_codes.slice(..1280))?;
    let qs = s.clone_dtoh(&mixer.act_lora_scales.slice(..40))?;
    let rc = record.bytes("query_codes", 1280)?;
    let rs = record.bytes("query_scales", 40)?;
    let x = proof.quant("query_quant", [qn, &rn], [&qc, &rc], [&qs, &rs])?;
    proof.projection(
        "wq_b",
        [&x[0], &x[1]],
        [qb, &rb],
        (
            &s.clone_dtoh(&weights.wq_b.codes)?,
            &s.clone_dtoh(&weights.wq_b.scale)?,
        ),
        1,
    )?;
    let coefficients = s.clone_dtoh(&mixer.rotary_fwd.slice(..64))?;
    let reference = record.f32("rotary", 64)?;
    proof.rotary(
        "query_rotary",
        [qb, &rb],
        [actual, expected],
        [&coefficients, &reference],
        false,
    )?;
    let before = proof.defects;
    let mut mutant = actual.to_vec();
    mutant[19931] = bf16::from_f32(mutant[19931].to_f32() + 1.0);
    proof.rotary(
        "query_control",
        [qb, &rb],
        [&mutant, expected],
        [&coefficients, &reference],
        false,
    )?;
    ensure!(proof.defects > before, "Q-store control escaped");
    proof.defects = before;
    println!(
        "LAYER_QUERY_CERTIFICATE layer={} position={} defects={} control=true",
        proof.layer, proof.position, proof.defects
    );
    Ok(())
}

pub(super) fn prove(
    engine: &Engine,
    layer: usize,
    position: usize,
    source: &[u16],
    root: &Path,
) -> Result<Certificate> {
    let record = Record(root);
    let s = &engine.device.stream;
    let mixer = engine.deepseek_mixer.as_ref().context("mixer missing")?;
    let weights = mixer.weights[layer]
        .as_ref()
        .context("mixer weights missing")?;
    let mut proof = Proof {
        layer,
        position,
        defects: 0,
    };
    let q = s.clone_dtoh(&engine.work.qkv.slice(..32768))?;
    let rq = record.bf("query", 32768)?;
    if q != rq {
        query(&mut proof, engine, source, &record, &q, &rq)?;
    }
    let (mut stores, traced) = trace(engine, layer, position)?;
    let (post, groups) = mixer.attn_output_host(s, 1)?;
    ensure!(
        traced == post,
        "certificate trace changes production attention stores"
    );
    let publisher = record.f32("trace", certificate::TRACE_LEN)?;
    let raw: Vec<_> = stores[853952..886720]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect();
    let expected_raw = record.bf("raw", 32768)?;
    let sink_bytes = s.clone_dtoh(&weights.attn_sink)?;
    let sink: Vec<_> = sink_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    let qbits: Vec<_> = q.iter().map(|v| v.to_bits()).collect();
    let rqbits: Vec<_> = rq.iter().map(|v| v.to_bits()).collect();
    let rawbits: Vec<_> = expected_raw.iter().map(|v| v.to_bits()).collect();
    let sparse = |candidate: &[f32]| {
        certificate::prove(certificate::Inputs {
            layer,
            position,
            query: [&qbits, &rqbits],
            sink: &sink,
            candidate,
            publisher: &publisher,
            output: &raw,
            expected: &rawbits,
        })
    };
    let result = sparse(&stores)?;
    proof.defects += result.store_defects + result.operand_defects;
    if position == 127 || (layer == 4 && position == 128) {
        stores[0] += 1.0;
        let control = sparse(&stores)?;
        ensure!(control.store_defects > 0, "QK+1 control escaped");
        println!("LAYER_ATTENTION_CONTROL layer={layer} position={position} detected=true");
    }
    let rpost = record.bf("post_rotary", 32768)?;
    let rotary = record.f32("rotary", 64)?;
    let actual_rotary = s.clone_dtoh(&mixer.rotary_fwd.slice(..64))?;
    proof.rotary(
        "inverse_rotary",
        [&raw, &expected_raw],
        [&post, &rpost],
        [&actual_rotary, &rotary],
        true,
    )?;
    let codes = s.clone_dtoh(&mixer.o_codes.slice(..32768))?;
    let scales = s.clone_dtoh(&mixer.o_scales.slice(..1024))?;
    let rc = record.bytes("attention_codes", 32768)?;
    let rs = record.bytes("attention_scales", 1024)?;
    let x = proof.quant(
        "attention_quant",
        [&post, &rpost],
        [&codes, &rc],
        [&scales, &rs],
    )?;
    let rgroups = record.bf("wo_b_input", 8192)?;
    proof.projection(
        "wo_a",
        [&x[0], &x[1]],
        [&groups, &rgroups],
        (
            &s.clone_dtoh(&weights.wo_a)?,
            &s.clone_dtoh(&weights.wo_a_scale)?,
        ),
        8,
    )?;
    let codes = s.clone_dtoh(&mixer.act_o_codes.slice(..8192))?;
    let scales = s.clone_dtoh(&mixer.act_o_scales.slice(..256))?;
    let rc = record.bytes("wo_b_codes", 8192)?;
    let rs = record.bytes("wo_b_scales", 256)?;
    let x = proof.quant(
        "wo_b_quant",
        [&groups, &rgroups],
        [&codes, &rc],
        [&scales, &rs],
    )?;
    let actual = s.clone_dtoh(&engine.work.out.slice(..5120))?;
    let expected = record.bf("output", 5120)?;
    proof.projection(
        "wo_b",
        [&x[0], &x[1]],
        [&actual, &expected],
        (
            &s.clone_dtoh(&weights.wo_b.codes)?,
            &s.clone_dtoh(&weights.wo_b.scale)?,
        ),
        1,
    )?;
    println!(
        "LAYER_CHAIN_CERTIFICATE layer={layer} position={position} defects={} computed=true",
        proof.defects
    );
    Ok(Certificate {
        values: [actual, expected],
        valid: proof.defects == 0,
    })
}
