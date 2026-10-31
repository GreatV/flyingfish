"""Gemma 4 vision operator fixtures from the transformers model definition."""
import argparse
import json
from pathlib import Path

import torch
import transformers
from safetensors.torch import save
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4VisionPatchEmbedder,
    Gemma4VisionPooler,
    Gemma4VisionRotaryEmbedding,
    Gemma4VisionAttention,
    apply_multidimensional_rope,
    eager_attention_forward,
)


def pattern(shape, period, multiplier, shift, divisor, device):
    count = 1
    for extent in shape:
        count *= extent
    values = (torch.arange(count) * multiplier % period - shift).float()
    return (values / divisor).reshape(shape).to(device)


@torch.inference_mode()
def capture(config, height, width, device):
    rows, hidden, heads, dim = height * width, config.hidden_size, config.num_attention_heads, config.head_dim
    assert (hidden, heads, dim) == (768, 12, 64)
    positions = torch.stack(torch.meshgrid(torch.arange(width, device=device),
                                         torch.arange(height, device=device), indexing="xy"), -1).reshape(1, rows, 2)
    pixels = pattern((rows, 768), 251, 37, 0, 250, device)
    scaled = (2 * (pixels - 0.5)).bfloat16()
    lower, upper = torch.tensor([-0.625], device=device).bfloat16(), torch.tensor([0.75], device=device).bfloat16()
    table = pattern((2, config.position_embedding_size, hidden), 257, 31, 128, 64, device).bfloat16()
    embedder = Gemma4VisionPatchEmbedder(config).to(device=device, dtype=torch.bfloat16)
    embedder.position_embedding_table = torch.nn.Parameter(table, requires_grad=False)
    pad = torch.zeros((1, rows), dtype=torch.bool, device=device)
    positioned = (scaled.unsqueeze(0) + embedder._position_embeddings(positions, pad))[0]
    pool_input = pattern((1, rows, hidden), 127, 17, -1, 1, device).bfloat16()
    pool = Gemma4VisionPooler(config)
    pooled, mask = pool(pool_input, positions, pad, output_length=rows // 9)
    pooled = pooled[mask]
    bias = pattern((hidden,), 263, 19, 131, 16, device).bfloat16()
    multiplier = pattern((hidden,), 269, 23, -1, 256, device).bfloat16()
    standardized = ((pooled - bias.float()) * multiplier.float()).bfloat16()
    padded = torch.full((1, 2520, hidden), 123.0, dtype=torch.bfloat16, device=device)
    padded[:, :rows] = pool_input
    padded_positions = torch.full((1, 2520, 2), -1, device=device)
    padded_positions[:, :rows] = positions
    padded_pool, padded_mask = pool(padded, padded_positions, (padded_positions == -1).all(-1), output_length=280)
    assert torch.equal(padded_pool[padded_mask].bfloat16(), pooled.bfloat16())
    q = pattern((1, rows, heads, dim), 257, 29, 128, 64, device).bfloat16()
    k = pattern(q.shape, 263, 31, 131, 64, device).bfloat16()
    v = pattern(q.shape, 269, 37, 134, 64, device).bfloat16()
    rotary = Gemma4VisionRotaryEmbedding(config).to(device)
    cosine, sine = rotary(q, positions)
    compact_cos = torch.cat((cosine[..., :dim // 4], cosine[..., dim // 2:3 * dim // 4]), -1)
    compact_sin = torch.cat((sine[..., :dim // 4], sine[..., dim // 2:3 * dim // 4]), -1)
    angles = torch.stack((compact_cos, compact_sin), -1)[0]
    rotated_q = apply_multidimensional_rope(q, cosine, sine, positions)[0]
    rotated_k = apply_multidimensional_rope(k, cosine, sine, positions)[0].transpose(0, 1).contiguous()
    wrong_cos, wrong_sin = rotary(q, positions.flip(-1))
    wrong_q = apply_multidimensional_rope(q, wrong_cos, wrong_sin, positions)[0]
    wrong_pool, wrong_mask = pool(pool_input, positions.flip(-1), pad, output_length=rows // 9)
    controls = {
        "clamp": int((scaled != scaled.clamp(lower, upper)).sum()),
        "position": int((positioned != (scaled.unsqueeze(0) + embedder._position_embeddings(positions.flip(-1), pad))[0]).sum()),
        "pool": int((pooled.bfloat16() != wrong_pool[wrong_mask].bfloat16()).sum()),
        "rope": int((rotated_q != wrong_q).sum()),
    }
    assert all(value > 0 for value in controls.values()), controls
    tensors = dict(pixels=pixels, scaled=scaled, lower=lower, upper=upper,
                   clamped=scaled.clamp(lower, upper), table=table, positioned=positioned,
                   pool_input=pool_input[0], pooled=pooled.bfloat16(), bias=bias,
                   multiplier=multiplier, standardized=standardized,
                   q=q[0], k=k[0], v=v[0], angles=angles, rotated_q=rotated_q,
                   rotated_k=rotated_k, transposed_v=v[0].transpose(0, 1))
    return {name: value.cpu().contiguous() for name, value in tensors.items()}, controls


@torch.inference_mode()
def capture_attention(config, rows, device):
    heads, dim = config.num_attention_heads, config.head_dim
    assert dim == 64 and config.num_key_value_heads == heads
    keys = rows + 17 if rows < 2304 else rows
    capacity, pitch = keys + 23, heads * dim + 64
    q = pattern((1, rows, heads, dim), 257, 29, 128, 64, device).bfloat16().transpose(1, 2)
    k = pattern((1, heads, keys, dim), 263, 31, 131, 64, device).bfloat16()
    v = pattern((1, heads, keys, dim), 269, 37, 134, 64, device).bfloat16()
    module = Gemma4VisionAttention(config, 0).eval()
    assert module.scaling == 1.0 and not module.is_causal
    def run(query, key, value, scale=1.0):
        return eager_attention_forward(module, query, key, value, None, scaling=scale)[0][0]
    expected = run(q.float(), k.float(), v.float())
    baseline = run(q, k, v)
    repeat = run(q, k, v)
    assert torch.equal(baseline, repeat)
    varied = torch.cat([run(q[:, :, start:start + 17], k, v)
                       for start in range(0, rows, 17)])
    wrong = run(q, k, v, scale=0.125)
    packed_q = torch.full((rows, pitch), 37.0, dtype=torch.bfloat16, device=device)
    packed_q[:, :heads * dim] = q[0].transpose(0, 1).reshape(rows, heads * dim)
    packed_k = torch.full((heads, capacity, dim), 17.0, dtype=torch.bfloat16, device=device)
    packed_v = torch.full_like(packed_k, 31.0)
    packed_k[:, :keys], packed_v[:, :keys] = k[0], v[0]
    tensors = dict(q=packed_q, k=packed_k, v=packed_v, oracle=expected,
                   baseline=baseline, varied=varied, wrong_scale=wrong)
    controls = {"wrong_scale": int((wrong != baseline).sum())}
    assert controls["wrong_scale"] > 0
    return {name: value.cpu().contiguous() for name, value in tensors.items()}, dict(
        rows=rows, heads=heads, dim=dim, keys=keys, capacity=capacity, pitch=pitch,
        repeat_mismatches=0, controls=controls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--stage", choices=("operators", "attention"), default="operators")
    args = parser.parse_args()
    assert transformers.__version__ == "5.21.0"
    assert not args.output.resolve().is_relative_to(args.model.resolve())
    config = transformers.AutoConfig.from_pretrained(args.model, local_files_only=True)
    assert config.architectures == ["Gemma4ForConditionalGeneration"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"definition": "transformers Gemma4 vision model definition", "weights_loaded": False,
              "transformers": transformers.__version__, "torch": torch.__version__, "device": args.device,
              "cells": {}}
    if args.stage == "attention":
        for rows in [1, 63, 64, 65, 135, 2304, 2520]:
            tensors, cell = capture_attention(config.vision_config, rows, args.device)
            name = f"vision-attention-{rows}"
            with (args.output / f"{name}.safetensors").open("xb") as output:
                output.write(save(tensors, metadata={"definition": report["definition"]}))
            report["cells"][name] = cell
            print(f"PASS {name} reference_repeat_bits=0 wrong_scale={cell['controls']['wrong_scale']}", flush=True)
        with (args.output / "REFERENCE.json").open("x") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
        return
    for height, width in [(9, 15), (48, 48), (42, 60)]:
        tensors, controls = capture(config.vision_config, height, width, args.device)
        repeat, repeat_controls = capture(config.vision_config, height, width, args.device)
        assert controls == repeat_controls
        mismatches = {name: int((value != repeat[name]).sum()) for name, value in tensors.items()}
        assert all(value == 0 for value in mismatches.values()), mismatches
        name = f"vision-ops-{height}x{width}"
        with (args.output / f"{name}.safetensors").open("xb") as output:
            output.write(save(tensors, metadata={"definition": report["definition"]}))
        report["cells"][name] = {"repeat_mismatches": mismatches, "controls": controls, "packed_padded_mismatches": 0}
        print(f"PASS {name} reference_repeat_bits=0 packed_padded_bits=0 controls={controls}", flush=True)
    with (args.output / "REFERENCE.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
