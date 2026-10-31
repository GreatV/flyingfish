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


@torch.inference_mode()
def prepare_media(args):
    from PIL import Image
    from transformers import AutoProcessor
    assert args.images and len(args.images) == 2
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    sentence = processor.tokenizer(" A photograph shows objects and buildings in the surrounding scene.",
                                   add_special_tokens=False).input_ids
    cells = [("street", 0, (768, 768), 17), ("winter", 1, (768, 768), 17),
             ("wide", 0, (960, 672), 17), ("long", 0, (768, 768), 513)]
    metadata = {}
    for name, index, size, rows in cells:
        picture = Image.open(args.images[index]).convert("RGB").resize(size, Image.Resampling.BICUBIC)
        batch = processor(text=processor.image_token + " Describe the visible scene in two sentences.",
                          images=picture, return_tensors="pt")
        prompt_length = batch.input_ids.shape[1]
        suffix = (sentence * ((rows + len(sentence) - 1) // len(sentence)))[:rows - 1]
        ids = torch.cat((batch.input_ids, torch.tensor([suffix], dtype=torch.long)), dim=1)
        modalities = torch.cat((batch.mm_token_type_ids, torch.zeros((1, len(suffix)), dtype=torch.long)), dim=1)
        valid = (batch.image_position_ids[0] >= 0).all(-1)
        positions = batch.image_position_ids[:, valid]
        packed = batch.pixel_values[:, valid]
        grid = (positions[0].amax(0) + 1).tolist()
        inputs = dict(ids=ids[0].int(), modalities=modalities[0].to(torch.uint8),
                      pixels=packed[0].float(), positions=positions[0].int(),
                      padded_pixels=batch.pixel_values[0].float(), padded_positions=batch.image_position_ids[0].int())
        (args.output / f"{name}-input.safetensors").write_bytes(save(inputs))
        metadata[name] = dict(grid_xy=grid, prompt_length=prompt_length, score_first=prompt_length - 1,
                              score_rows=rows, total_tokens=ids.shape[1])
    (args.output / "INPUT.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("PASS media_inputs cells=4 same_grid_two_media=true production_patches=2304,2520", flush=True)


@torch.inference_mode()
def capture_model(args, config):
    from safetensors.torch import load_file
    assert args.inputs
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model_class = getattr(transformers, config.architectures[0])
    model, load = model_class.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager", output_loading_info=True, device_map="cpu")
    assert all(not load[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), load
    assert not any(value.is_meta for value in model.state_dict().values())
    model.eval().requires_grad_(False)
    lm, vision = model.model.language_model, model.model.vision_tower
    assert config.text_config.use_bidirectional_attention is None
    assert model.lm_head.weight is lm.embed_tokens.weight

    def move(module):
        module.to(device=args.device, dtype=dtype)
        for child in module.modules():
            if hasattr(child, "scalar_embed_scale"):
                child.embed_scale = torch.tensor(child.scalar_embed_scale, device=args.device, dtype=dtype)

    move(lm.embed_tokens)
    model.lm_head.weight = lm.embed_tokens.weight
    lm.rotary_emb.to(args.device)
    vision.encoder.rotary_emb.to(args.device)
    hooks = []
    def stream(module):
        hooks.append(module.register_forward_pre_hook(lambda mod, inputs: move(mod)))
        def offload(mod, inputs, output):
            mod.to(device="cpu", dtype=torch.bfloat16)
        hooks.append(module.register_forward_hook(offload))
    for module in (vision.patch_embedder, *vision.encoder.layers, model.model.embed_vision,
                   lm.embed_tokens_per_layer, lm.per_layer_model_projection,
                   lm.per_layer_projection_norm, *lm.layers, lm.norm):
        stream(module)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    report = dict(definition="transformers Gemma4 model definition", transformers=transformers.__version__,
                  torch=torch.__version__, dtype=args.dtype, device=args.device,
                  clean_load={key: list(load[key]) for key in
                              ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")},
                  cpu_resident_weights="published BF16", cells={})
    (args.output / "LOAD.json").write_text(json.dumps(report, indent=2) + "\n")
    counted = [0]
    features = []
    ple = []
    def encoder_called(module, inputs, output):
        counted[0] += 1
    hooks.append(vision.register_forward_hook(encoder_called))
    hooks.append(model.model.embed_vision.register_forward_hook(
        lambda module, inputs, output: features.append(output.detach().cpu().contiguous())))
    def capture_ple(module, inputs):
        ple.append(inputs[1].detach().cpu().contiguous())
    hooks.append(lm.layers[0].register_forward_pre_hook(capture_ple))
    media_logits = []
    for name, cell in json.loads((args.inputs / "INPUT.json").read_text()).items():
        if args.stage == "media" and name not in ("street", "winter"):
            continue
        data = load_file(args.inputs / f"{name}-input.safetensors")
        ids, modalities = data["ids"][None].long(), data["modalities"][None].long()
        rows, prompt_length = cell["score_rows"], cell["prompt_length"]
        cell.update(encoder_calls={}, decoded=None)
        for variant in (() if args.stage == "media" else ("packed", "padded")):
            before = counted[0]
            features.clear()
            ple.clear()
            pixels = data["pixels" if variant == "packed" else "padded_pixels"][None]
            coordinates = data["positions" if variant == "packed" else "padded_positions"][None].long()
            output = model(input_ids=ids.to(args.device), mm_token_type_ids=modalities.to(args.device),
                           pixel_values=pixels.to(args.device), image_position_ids=coordinates.to(args.device),
                           use_cache=False, logits_to_keep=rows)
            logits = output.logits[0].float().cpu().contiguous()
            assert torch.isfinite(logits).all() and len(features) == 1
            cell["encoder_calls"][variant] = counted[0] - before
            assert cell["encoder_calls"][variant] == 1
            (args.output / f"{name}-{variant}.safetensors").write_bytes(
                save(dict(logits=logits, features=features[0].float(), ple=ple[0][0].float())))
            if name in ("street", "winter") and variant == "packed":
                media_logits.append(logits[0].clone())
            del output, logits
        if name in ("street", "winter") and args.dtype == "bf16":
            before = counted[0]
            image_end = (ids[0] == config.eoi_token_id).nonzero().item() + 1
            caption = tokenizer(" Caption: A photograph of", add_special_tokens=False).input_ids
            review_ids = torch.cat((ids[:, :image_end], torch.tensor([caption])), dim=1)
            review_modalities = torch.cat((modalities[:, :image_end], torch.zeros((1, len(caption)), dtype=torch.long)), dim=1)
            output = model(input_ids=review_ids.to(args.device),
                           mm_token_type_ids=review_modalities.to(args.device),
                           pixel_values=data["padded_pixels"][None].to(args.device),
                           image_position_ids=data["padded_positions"][None].long().to(args.device),
                           use_cache=True, logits_to_keep=1)
            generated = []
            media_logits.append(output.logits[0, -1].float().cpu())
            for step in range(32):
                token = output.logits[:, -1].argmax(-1, keepdim=True)
                generated.append(token.item())
                if step != 31:
                    output = model(input_ids=token, past_key_values=output.past_key_values,
                                   use_cache=True, logits_to_keep=1)
            cell["decoded"] = tokenizer.decode(generated, skip_special_tokens=False)
            cell["generated_ids"] = generated
            cell["review_prompt_suffix"] = " Caption: A photograph of"
            del output
            cell["generation_encoder_calls"] = counted[0] - before
            assert cell["generation_encoder_calls"] == 1
            print(f"MEDIA {name} decoded={cell['decoded']!r}", flush=True)
        report["cells"][name] = cell
        (args.output / "REFERENCE.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASS {name} model_reference rows={rows} encoder_calls={cell['encoder_calls']}", flush=True)
    assert len(media_logits) in (2, 4)
    assert not torch.equal(media_logits[0], media_logits[len(media_logits) // 2]), "two-media control produced identical logits"
    report["two_media_logit_mismatches"] = int((media_logits[0] != media_logits[len(media_logits) // 2]).sum())
    (args.output / "REFERENCE.json").write_text(json.dumps(report, indent=2) + "\n")
    for hook in hooks:
        hook.remove()


def freeze_model(args):
    from safetensors.torch import load_file
    import shutil
    assert args.oracle and args.arms and args.inputs
    metadata = json.loads((args.inputs / "INPUT.json").read_text())
    frozen = dict(status="FROZEN", rule="2 * maximum reference-only distance to FP32 oracle", cells={})
    for name, cell in metadata.items():
        expected = load_file(args.oracle / f"{name}-packed.safetensors")
        ids = load_file(args.inputs / f"{name}-input.safetensors")["ids"].long()[cell["score_first"]:]
        bounds = {key: dict(output_max=0.0, output_rmse=0.0) for key in ("logits", "features", "ple")}
        bounds["logits"]["nll_max"] = 0.0
        for arm in args.arms:
            for variant in ("packed", "padded"):
                actual = load_file(arm / f"{name}-{variant}.safetensors")
                for key in bounds:
                    assert actual[key].shape == expected[key].shape
                    assert torch.isfinite(actual[key]).all() and torch.isfinite(expected[key]).all()
                    delta = actual[key].double() - expected[key].double()
                    bounds[key]["output_max"] = max(bounds[key]["output_max"], 2 * delta.abs().max().item())
                    bounds[key]["output_rmse"] = max(bounds[key]["output_rmse"], 2 * delta.square().mean().sqrt().item())
                    del delta
                def loss(logits):
                    total = 0.0
                    for row, target in zip(logits[:-1], ids[1:]):
                        values = row.double()
                        total += (values.logsumexp(-1) - values[target]).item()
                    return total / (len(ids) - 1)
                bounds["logits"]["nll_max"] = max(bounds["logits"]["nll_max"],
                    2 * abs(loss(actual["logits"]) - loss(expected["logits"])))
                del actual
        bounds["logits"]["margin_cut"] = 2 * bounds["logits"]["output_max"]
        frozen["cells"][name] = bounds
        print(f"FREEZE {name} {json.dumps(bounds, sort_keys=True)}", flush=True)
    shutil.copytree(args.inputs, args.output / "inputs")
    (args.output / "oracle").symlink_to(args.oracle.resolve(), target_is_directory=True)
    (args.output / "FROZEN.json").write_text(json.dumps(frozen, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--stage", choices=("operators", "attention", "inputs", "model", "media", "freeze"), default="operators")
    parser.add_argument("--images", type=Path, nargs=2)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--arms", type=Path, nargs="+")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
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
    if args.stage == "freeze":
        freeze_model(args)
        return
    if args.stage == "inputs":
        prepare_media(args)
        return
    if args.stage in ("model", "media"):
        capture_model(args, config)
        return
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
