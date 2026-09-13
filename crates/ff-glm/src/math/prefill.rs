//! The pinned Transformers KDA prefill formula, with its fixed 64-token chunks.
//! This is not a token-recurrent approximation or a selectable chunk policy.

use super::l2_normalize_f32;
use anyhow::{Context, Result, ensure};
use candle_core::{D, DType, Tensor};

const CHUNK: usize = 64;

fn sum_key_axis(input: &Tensor) -> Result<Tensor> {
    if input.device().is_cpu() {
        return Ok(input.sum(D::Minus1)?);
    }
    ensure!(
        input.device().is_cuda() && input.dtype() == DType::F32 && input.dim(D::Minus1)? == 128,
        "KDA CUDA key reduction requires F32 width 128"
    );
    let rows = input.elem_count() / 128;
    let lanes = input.contiguous()?.reshape((rows, 32, 4))?;
    let mut values = Tensor::zeros((rows, 32, 1), DType::F32, input.device())?;
    for slot in 0..4 {
        values = values.add(&lanes.narrow(2, slot, 1)?)?;
    }
    for offset in [16, 8, 4, 2, 1] {
        values = values
            .narrow(1, 0, offset)?
            .add(&values.narrow(1, offset, offset)?)?;
    }
    Ok(values.reshape(&input.dims()[..input.rank() - 1])?)
}

fn normalize_key(input: &Tensor) -> Result<Tensor> {
    if input.device().is_cpu() {
        return l2_normalize_f32(input, 1e-6);
    }
    let denominator = (sum_key_axis(&input.sqr()?)? + 1e-6)?
        .sqrt()?
        .unsqueeze(input.rank() - 1)?;
    Ok(input.broadcast_div(&denominator)?)
}

pub(super) fn sum_short_strided_rows(input: &Tensor) -> Result<Tensor> {
    if input.device().is_cpu() {
        return Ok(input.sum(2)?);
    }
    let (heads, chunks, rows, columns) = input.dims4()?;
    ensure!(
        input.device().is_cuda() && input.dtype() == DType::F32 && rows > 0 && rows < CHUNK,
        "KDA history reduction requires F32 CUDA rows in 1..64"
    );
    let zero = Tensor::zeros((heads, chunks, 1, columns), DType::F32, input.device())?;
    let mut accumulators = [zero.clone(), zero.clone(), zero.clone(), zero];
    for row in 0..rows {
        let slot = row % 4;
        accumulators[slot] = accumulators[slot].add(&input.narrow(2, row, 1)?)?;
    }
    Ok(accumulators[0]
        .add(&accumulators[1])?
        .add(&accumulators[2])?
        .add(&accumulators[3])?
        .squeeze(2)?)
}

/// Prefill `[tokens, heads, width]` Q/K/V, F32 per-key log decay and
/// `[tokens, heads]` beta. Returns attended values in Q's dtype and the final
/// F32 `[heads, key_width, value_width]` recurrent state, ready for decode.
/// Padding is exactly the native operator's zero padding to 64-token chunks.
pub fn kda_prefill(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    initial_state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (tokens, heads, key_width) = query
        .dims3()
        .context("KDA prefill Q must be [tokens, heads, width]")?;
    let (value_tokens, value_heads, value_width) = value
        .dims3()
        .context("KDA prefill V must be [tokens, heads, width]")?;
    ensure!(
        tokens > 0 && tokens <= 2048,
        "KDA prefill requires 1..=2048 tokens"
    );
    ensure!(
        heads > 0 && key_width > 0 && value_width > 0,
        "KDA prefill dimensions must be positive"
    );
    ensure!(key.dims() == query.dims(), "KDA prefill Q/K shapes differ");
    ensure!(
        value_tokens == tokens && value_heads == heads,
        "KDA prefill Q/V token or head counts differ"
    );
    ensure!(
        log_decay.dims() == query.dims(),
        "KDA prefill log decay must match Q"
    );
    ensure!(
        beta.dims() == [tokens, heads],
        "KDA prefill beta must be [tokens, heads]"
    );
    ensure!(
        initial_state.dims() == [heads, key_width, value_width],
        "KDA prefill recurrent state has the wrong shape"
    );
    ensure!(
        query.dtype() == key.dtype() && query.dtype() == value.dtype(),
        "KDA prefill Q/K/V dtypes differ"
    );
    ensure!(
        query.device().is_cpu() || (query.device().is_cuda() && key_width == 128),
        "KDA prefill CUDA profile requires key width 128"
    );
    let dtype = query.dtype();
    let device = query.device();
    let chunks = tokens.div_ceil(CHUNK);
    let padding = chunks * CHUNK - tokens;
    let head_first = |tensor: &Tensor| -> Result<Tensor> {
        Ok(tensor.transpose(0, 1)?.contiguous()?.to_dtype(DType::F32)?)
    };
    let query = normalize_key(&head_first(query)?)?
        .pad_with_zeros(1, 0, padding)?
        .affine(1.0 / (key_width as f64).sqrt(), 0.0)?;
    let key = normalize_key(&head_first(key)?)?.pad_with_zeros(1, 0, padding)?;
    let value = head_first(value)?.pad_with_zeros(1, 0, padding)?;
    let beta = head_first(beta)?
        .pad_with_zeros(1, 0, padding)?
        .unsqueeze(2)?;
    let v_beta = value
        .broadcast_mul(&beta)?
        .reshape((heads, chunks, CHUNK, value_width))?;
    let k_beta = key
        .broadcast_mul(&beta)?
        .reshape((heads, chunks, CHUNK, key_width))?;
    let query = query.reshape((heads, chunks, CHUNK, key_width))?;
    let key = key.reshape((heads, chunks, CHUNK, key_width))?;
    let g = head_first(log_decay)?
        .pad_with_zeros(1, 0, padding)?
        .reshape((heads, chunks, CHUNK, key_width))?;

    let mut cumulative = Vec::with_capacity(CHUNK);
    let mut running = Tensor::zeros((heads, chunks, 1, key_width), DType::F32, device)?;
    for token in 0..CHUNK {
        running = running.add(&g.narrow(2, token, 1)?)?;
        cumulative.push(running.clone());
    }
    let g = Tensor::cat(&cumulative, 2)?;
    let decay = g.unsqueeze(3)?.broadcast_sub(&g.unsqueeze(2)?)?.exp()?;
    let mask = |strict: bool| -> Result<Tensor> {
        let values = (0..CHUNK)
            .flat_map(|row| {
                (0..CHUNK).map(move |col| u8::from(if strict { col < row } else { col <= row }))
            })
            .collect::<Vec<_>>();
        Ok(Tensor::from_vec(values, (CHUNK, CHUNK), device)?
            .broadcast_as((heads, chunks, CHUNK, CHUNK))?)
    };
    let zero = Tensor::zeros((heads, chunks, CHUNK, CHUNK), DType::F32, device)?;
    let mut attn = mask(true)?.where_cond(
        &sum_key_axis(
            &k_beta
                .unsqueeze(3)?
                .broadcast_mul(&key.unsqueeze(2)?)?
                .mul(&decay)?,
        )?
        .neg()?,
        &zero,
    )?;
    for row_index in 1..CHUNK {
        let row = attn
            .narrow(2, row_index, 1)?
            .narrow(3, 0, row_index)?
            .squeeze(2)?
            .contiguous()?;
        let sub = attn
            .narrow(2, 0, row_index)?
            .narrow(3, 0, row_index)?
            .contiguous()?;
        let updated = row
            .add(&sum_short_strided_rows(
                &row.unsqueeze(3)?.broadcast_mul(&sub)?,
            )?)?
            .unsqueeze(2)?;
        attn = attn.slice_assign(
            &[0..heads, 0..chunks, row_index..row_index + 1, 0..row_index],
            &updated,
        )?;
    }
    let attn = attn.broadcast_add(&Tensor::eye(CHUNK, DType::F32, device)?)?;
    let value = attn.matmul(&v_beta)?;
    let k_cumdecay = attn.matmul(&k_beta.mul(&g.exp()?)?)?;
    let intra = mask(false)?.where_cond(
        &sum_key_axis(
            &query
                .unsqueeze(3)?
                .broadcast_mul(&key.unsqueeze(2)?)?
                .mul(&decay)?,
        )?,
        &zero,
    )?;
    let mut state = initial_state.to_dtype(DType::F32)?.contiguous()?;
    let mut outputs = Vec::with_capacity(chunks);
    for chunk in 0..chunks {
        let take = |tensor: &Tensor| -> Result<Tensor> {
            Ok(tensor.narrow(1, chunk, 1)?.squeeze(1)?.contiguous()?)
        };
        let q_i = take(&query)?;
        let k_i = take(&key)?;
        let g_i = take(&g)?;
        let attn_inter = q_i.mul(&g_i.exp()?)?.matmul(&state)?;
        let v_new = take(&value)?.sub(&take(&k_cumdecay)?.matmul(&state)?)?;
        outputs.push(attn_inter.add(&take(&intra)?.matmul(&v_new)?)?);
        let g_last = g_i.narrow(1, CHUNK - 1, 1)?;
        state = state
            .broadcast_mul(&g_last.squeeze(1)?.exp()?.unsqueeze(2)?)?
            .add(
                &k_i.mul(&g_last.broadcast_sub(&g_i)?.exp()?)?
                    .transpose(1, 2)?
                    .contiguous()?
                    .matmul(&v_new)?,
            )?;
    }
    let output = Tensor::cat(&outputs, 1)?
        .narrow(1, 0, tokens)?
        .transpose(0, 1)?
        .contiguous()?
        .to_dtype(dtype)?;
    Ok((output, state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, IndexOp};

    #[test]
    fn chunked_prefill_agrees_with_recurrence_on_small_f32_inputs() {
        let device = Device::Cpu;
        for tokens in [1, 3, 63, 64, 65, 129] {
            let make = |width: usize, offset: f32| {
                Tensor::from_vec(
                    (0..tokens * 2 * width)
                        .map(|i| ((i % 17) as f32 - 8.0) / 17.0 + offset)
                        .collect::<Vec<_>>(),
                    (tokens, 2, width),
                    &device,
                )
                .unwrap()
            };
            let q = make(3, 0.01);
            let k = make(3, -0.02);
            let v = make(4, 0.03);
            let g = Tensor::full(-0.1f32, (tokens, 2, 3), &device).unwrap();
            let beta = Tensor::full(0.3f32, (tokens, 2), &device).unwrap();
            let initial = Tensor::full(0.01f32, (2, 3, 4), &device).unwrap();
            let (actual, final_state) = kda_prefill(&q, &k, &v, &g, &beta, &initial).unwrap();
            let mut state = initial;
            let mut expected = Vec::new();
            for token in 0..tokens {
                let (row, next) = super::super::kda_single_token(
                    &q.i(token).unwrap(),
                    &k.i(token).unwrap(),
                    &v.i(token).unwrap(),
                    &g.i(token).unwrap(),
                    &beta.i(token).unwrap(),
                    &state,
                )
                .unwrap();
                state = next;
                expected.push(row.unsqueeze(0).unwrap());
            }
            for (a, b) in [
                (actual, Tensor::cat(&expected, 0).unwrap()),
                (final_state, state),
            ] {
                for (a, b) in a
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .into_iter()
                    .zip(b.flatten_all().unwrap().to_vec1::<f32>().unwrap())
                {
                    assert!((a - b).abs() < 2e-6, "{tokens} tokens: {a} != {b}");
                }
            }
        }
    }

    #[test]
    fn prefill_rejects_invalid_geometry_before_computation() {
        let d = Device::Cpu;
        let q = Tensor::zeros((0, 2, 3), DType::F32, &d).unwrap();
        let v = Tensor::zeros((0, 2, 4), DType::F32, &d).unwrap();
        let beta = Tensor::zeros((0, 2), DType::F32, &d).unwrap();
        let state = Tensor::zeros((2, 3, 4), DType::F32, &d).unwrap();
        assert!(
            kda_prefill(&q, &q, &v, &q, &beta, &state)
                .unwrap_err()
                .to_string()
                .contains("1..=2048")
        );
    }
}
