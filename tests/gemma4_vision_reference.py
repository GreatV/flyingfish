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
    if args.audio:
        prepare_audio(args)
        return
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


def prepare_audio(args):
    import subprocess
    import numpy as np
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    waves = []
    for path in args.audio:
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels",
                                "-of", "json", str(path)], check=True, capture_output=True)
        streams = json.loads(probe.stdout)["streams"]
        assert len(streams) == 1 and streams[0] == dict(sample_rate="16000", channels=1)
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
                                  "-acodec", "pcm_f32le", "-"], check=True, capture_output=True)
        waves.append(np.frombuffer(decoded.stdout, dtype="<f4").copy())
    long = np.tile(waves[0], (480000 + len(waves[0]) - 1) // len(waves[0]))[:480000]
    suffix = processor.tokenizer(" Transcription: The speaker describes a scene in a story.",
                                 add_special_tokens=False).input_ids
    metadata = {}
    for name, clips, rows in [("stew", waves[:1], 17), ("wizard", waves[1:], 17),
                              ("pair", waves, 17), ("long", [long], 513)]:
        processed = [processor(text=processor.audio_token, audio=wave, sampling_rate=16000,
                               return_tensors="pt") for wave in clips]
        frames = max(batch.input_features.shape[1] for batch in processed)
        padded_frames = min(3000, ((frames + 47) // 48) * 48)
        def pack(extent):
            features = torch.zeros(len(clips), extent, 128)
            mask = torch.zeros(len(clips), extent, dtype=torch.bool)
            for i, batch in enumerate(processed):
                n = batch.input_features.shape[1]
                features[i, :n] = batch.input_features[0]
                mask[i, :n] = batch.input_features_mask[0]
            return features, mask
        features, mask = pack(frames)
        padded, padded_mask = pack(padded_frames)
        base = processed[0].input_ids[0].tolist()
        for batch in processed[1:]:
            assert batch.input_ids[0, 0] == base[0]
            base.extend(batch.input_ids[0, 1:].tolist())
        prompt = base + processor.tokenizer(" Transcription:", add_special_tokens=False).input_ids
        continuation = (suffix * ((rows + len(suffix) - 1) // len(suffix)))[:rows - 1]
        ids = torch.tensor(prompt + continuation, dtype=torch.int32)
        modalities = (ids == processor.audio_token_id).to(torch.uint8) * 3
        assert int(mask[:, ::4].sum()) == int((modalities == 3).sum())
        assert torch.isfinite(features).all() and frames <= 3000
        (args.output / f"{name}-input.safetensors").write_bytes(save(dict(
            ids=ids, modalities=modalities, audio_features=features, audio_mask=mask,
            padded_audio_features=padded, padded_audio_mask=padded_mask)))
        metadata[name] = dict(kind="audio", prompt_length=len(prompt), score_first=len(prompt)-1,
                              score_rows=rows, total_tokens=len(ids), clips=len(clips), frames=frames,
                              valid_frames=mask.sum(1).tolist(), soft_tokens=mask[:, ::4].sum(1).tolist())
    (args.output / "INPUT.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("PASS audio_inputs cells=4 two_clips=true production_soft_tokens=750", flush=True)


@torch.inference_mode()
def capture_model(args, config):
    from safetensors.torch import load_file
    assert args.inputs
    assert args.stage != "media" or args.dtype == "bf16"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    metadata = json.loads((args.inputs / "INPUT.json").read_text())
    audio = all(cell.get("kind") == "audio" for cell in metadata.values())
    model_class = getattr(transformers, config.architectures[0])
    model, load = model_class.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager", output_loading_info=True)
    assert all(not load[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), load
    assert not any(value.is_meta for value in model.state_dict().values())
    model.eval().requires_grad_(False)
    lm = model.model.language_model
    tower = model.model.audio_tower if audio else model.model.vision_tower
    embedder = model.model.embed_audio if audio else model.model.embed_vision
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
    if audio:
        relative_timescales_dtype = str(tower.rel_pos_enc.inv_timescales.dtype)
        move(tower)
        move(embedder)
    else:
        tower.encoder.rotary_emb.to(args.device)
    hooks = []
    def stream(module):
        hooks.append(module.register_forward_pre_hook(lambda mod, inputs: move(mod)))
        def offload(mod, inputs, output):
            mod.to(device="cpu", dtype=torch.bfloat16)
        hooks.append(module.register_forward_hook(offload))
    if not audio:
        for module in (tower.patch_embedder, *tower.encoder.layers, embedder):
            stream(module)
    text_modules = (lm.embed_tokens_per_layer, lm.per_layer_model_projection,
                    lm.per_layer_projection_norm, *lm.layers, lm.norm)
    text_bytes = sum(p.numel() * torch.empty((), dtype=dtype).element_size()
                     for module in text_modules for p in module.parameters())
    rows = max(cell["total_tokens"] for cell in metadata.values())
    score_rows = max(cell["score_rows"] for cell in metadata.values())
    text_config = config.text_config
    workspace_bytes = (2 * 1024**3 + 12 * rows * rows * text_config.num_attention_heads
                       + 12 * rows * text_config.intermediate_size
                       + 12 * score_rows * text_config.vocab_size)
    free_bytes, _ = torch.cuda.mem_get_info(args.device)
    resident = text_bytes + workspace_bytes <= free_bytes
    for module in text_modules:
        move(module) if resident else stream(module)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    report = dict(definition="transformers Gemma4 model definition", transformers=transformers.__version__,
                  torch=torch.__version__, dtype=args.dtype, device=args.device,
                  clean_load={key: list(load[key]) for key in
                              ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")},
                  cpu_resident_weights="published BF16", cells={})
    report["text_placement"] = dict(resident=resident, additional_weight_bytes=text_bytes,
                                  workspace_bytes=workspace_bytes, free_bytes=free_bytes)
    if audio:
        report["relative_timescales_dtype"] = relative_timescales_dtype
        report["bf16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    (args.output / "LOAD.json").write_text(json.dumps(report, indent=2) + "\n")
    counted = [0]
    features = []
    ple = []
    stages = {}
    text_stages = {}
    if audio:
        for index in (0, len(lm.layers) // 2, len(lm.layers) - 1):
            hooks.append(lm.layers[index].register_forward_hook(
                lambda module, inputs, output, index=index:
                    text_stages.update({f"text.layer.{index}": output[0].detach().float().cpu().contiguous()})))
        hooks.append(tower.subsample_conv_projection.register_forward_hook(
            lambda module, inputs, output: stages.update(input=output[0].detach().cpu())))
        ff = tower.layers[0].feed_forward1
        for key, module in (("pre", ff.pre_layer_norm), ("up", ff.ffw_layer_1),
                            ("act", ff.act_fn), ("down", ff.ffw_layer_2)):
            hooks.append(module.register_forward_hook(
                lambda module, inputs, output, key=key: stages.update({f"ff1.{key}": output.detach().cpu()})))
        for index, module in enumerate((tower.subsample_conv_projection.layer0,
                                        tower.subsample_conv_projection.layer1)):
            def subsample_hook(module, inputs, output, index=index):
                hidden, mask = output
                stages[f"subsample{index}"] = hidden.permute(0, 2, 3, 1)[mask].flatten(0, 1).float().cpu().contiguous()
            hooks.append(module.register_forward_hook(subsample_hook))
        for index, module in enumerate(tower.layers):
            hooks.append(module.register_forward_hook(
                lambda module, inputs, output, index=index:
                    stages.update({f"layer{index}": output.detach().cpu()})))
        for key, module in (("ff1", tower.layers[0].feed_forward1),
                            ("ff2", tower.layers[0].feed_forward2)):
            hooks.append(module.register_forward_hook(
                lambda module, inputs, output, key=key: stages.update({key: output.detach().cpu()})))
        hooks.append(tower.layers[0].lconv1d.register_forward_pre_hook(
            lambda module, inputs: stages.update(attention=inputs[0].detach().cpu())))
        hooks.append(tower.register_forward_hook(
            lambda module, inputs, output: stages.update(tower=output.last_hidden_state.detach().cpu())))
    def encoder_called(module, inputs, output):
        counted[0] += 1
    hooks.append(tower.register_forward_hook(encoder_called))
    hooks.append(embedder.register_forward_hook(
        lambda module, inputs, output: features.append(output.detach().cpu().contiguous())))
    def capture_ple(module, inputs):
        ple.append(inputs[1].detach().cpu().contiguous())
    hooks.append(lm.layers[0].register_forward_pre_hook(capture_ple))
    media_logits = []
    semantic_cells = ("stew", "wizard") if audio else ("street", "winter")
    for name, cell in metadata.items():
        if args.stage == "media" and name not in semantic_cells:
            continue
        data = load_file(args.inputs / f"{name}-input.safetensors")
        ids, modalities = data["ids"][None].long(), data["modalities"][None].long()
        rows, prompt_length = cell["score_rows"], cell["prompt_length"]
        cell.update(encoder_calls={}, decoded=None)
        for variant in (() if args.stage == "media" else ("packed", "padded")):
            before = counted[0]
            features.clear()
            ple.clear()
            media = reference_media(data, variant, args.device, audio)
            arguments = dict(input_ids=ids.to(args.device), mm_token_type_ids=modalities.to(args.device),
                             **media, use_cache=False, logits_to_keep=rows)
            output = model(**arguments)
            logits = output.logits[0].float().cpu().contiguous()
            assert torch.isfinite(logits).all() and len(features) == 1
            cell["encoder_calls"][variant] = counted[0] - before
            assert cell["encoder_calls"][variant] == 1
            def valid_features():
                return (features[0][media["input_features_mask"].cpu()[:, ::4]] if audio else features[0]).float()
            captured = dict(logits=logits, features=valid_features(), ple=ple[0][0].float())
            captured.update(text_stages)
            if audio:
                for key, value in stages.items():
                    captured[key] = (value[media["input_features_mask"].cpu()[:, ::4]] if key not in ("subsample0", "subsample1") else value).float().contiguous()
            del output
            if variant == "packed":
                features.clear()
                ple.clear()
                repeated = model(**arguments)
                repeat = dict(logits=repeated.logits[0].float().cpu(), features=valid_features(), ple=ple[0][0].float())
                repeat.update(text_stages)
                if audio:
                    for key, value in stages.items():
                        repeat[key] = (value[media["input_features_mask"].cpu()[:, ::4]] if key not in ("subsample0", "subsample1") else value).float().contiguous()
                cell["repeat_mismatches"] = {key: int((captured[key] != repeat[key]).sum()) for key in captured}
                assert not any(cell["repeat_mismatches"].values()), cell["repeat_mismatches"]
                assert counted[0] - before == 2
                cell["repeat_encoder_calls"] = 1
                del repeated, repeat
            (args.output / f"{name}-{variant}.safetensors").write_bytes(save(captured))
            if name in semantic_cells and variant == "packed":
                media_logits.append(logits[0].clone())
            del logits, captured, arguments
        if audio and args.stage == "model" and name != "long":
            before = counted[0]
            media = reference_media(data, "packed", args.device, audio)
            output = model(input_ids=ids[:, :prompt_length].to(args.device),
                           mm_token_type_ids=modalities[:, :prompt_length].to(args.device),
                           **media, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            decoded_logits = [output.logits[0].float().cpu()]
            del output
            for position in range(prompt_length, ids.shape[1]):
                output = model(input_ids=ids[:, position:position + 1].to(args.device),
                               past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                decoded_logits.append(output.logits[0].float().cpu())
                del output
            logits = torch.cat(decoded_logits)
            assert logits.shape[0] == rows and torch.isfinite(logits).all()
            assert counted[0] - before == 1
            (args.output / f"{name}-decode.safetensors").write_bytes(save(dict(logits=logits.contiguous())))
            cell["encoder_calls"]["decode"] = 1
            del logits, decoded_logits, cache, media
        if args.stage == "media":
            before = counted[0]
            image_end = (ids[0] == (config.eoa_token_id if audio else config.eoi_token_id)).nonzero().item() + 1
            review_suffix = ' Audio transcript: "' if audio else " Caption: A photograph of"
            caption = tokenizer(review_suffix, add_special_tokens=False).input_ids
            review_ids = torch.cat((ids[:, :image_end], torch.tensor([caption])), dim=1)
            review_modalities = torch.cat((modalities[:, :image_end], torch.zeros((1, len(caption)), dtype=torch.long)), dim=1)
            output = model(input_ids=review_ids.to(args.device),
                           mm_token_type_ids=review_modalities.to(args.device),
                           **reference_media(data, "padded", args.device, audio),
                           use_cache=True, logits_to_keep=1)
            generated = []
            media_logits.append(output.logits[0, -1].float().cpu())
            review_tokens = 64 if audio else 32
            for step in range(review_tokens):
                token = output.logits[:, -1].argmax(-1, keepdim=True)
                generated.append(token.item())
                if step != review_tokens - 1:
                    output = model(input_ids=token, past_key_values=output.past_key_values,
                                   use_cache=True, logits_to_keep=1)
            cell["decoded"] = tokenizer.decode(generated, skip_special_tokens=False)
            cell["generated_ids"] = generated
            cell["review_prompt_suffix"] = review_suffix
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


def reference_media(data, variant, device, audio):
    prefix = "" if variant == "packed" else "padded_"
    if audio:
        return dict(input_features=data[prefix + "audio_features"].to(device),
                    input_features_mask=data[prefix + "audio_mask"].to(device))
    return dict(pixel_values=data[prefix + "pixels"][None].to(device),
                image_position_ids=data[prefix + "positions"][None].long().to(device))


@torch.inference_mode()
def capture_audio_operators(args, config):
    from types import SimpleNamespace
    from transformers.models.gemma4 import modeling_gemma4 as definition
    import torch.nn.functional as F
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    config.audio_config._attn_implementation = "eager"
    report = dict(torch=torch.__version__, transformers=transformers.__version__, device=args.device,
                  capability=list(torch.cuda.get_device_capability(args.device)) if str(args.device).startswith("cuda") else None,
                  dtype=args.dtype, definition="transformers Gemma4 audio model definition", cells={})
    initial_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        position = definition.Gemma4AudioRelPositionalEncoding(config.audio_config)
    finally:
        torch.set_default_dtype(initial_dtype)
    position.to(device=args.device, dtype=dtype)
    positions = position(torch.zeros(1, 1, 1024, device=args.device, dtype=dtype))[0]
    old_position = definition.Gemma4AudioRelPositionalEncoding(config.audio_config).to(args.device)
    wrong = old_position(torch.zeros(1, 1, 1024, device=args.device)).bfloat16()[0]
    (args.output / "audio-relative-table.safetensors").write_bytes(save(dict(
        positions=positions.float().cpu().contiguous(), wrong=wrong.float().cpu().contiguous())))
    report["cells"]["audio-relative-table"] = dict(shape=[13, 1024], outputs=["positions"],
        controls={"positions": "wrong"}, exact_bf16=True)
    print("PASS audio_relative_table elements=13312 staged_bf16_initialization=true", flush=True)
    class Projection(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value
        def forward(self, unused):
            return self.value
    for rows in [1, 11, 12, 13, 25, 750]:
        def values(shape, period=1009, multiplier=37, shift=504, divisor=128):
            return pattern(shape, period, multiplier, shift, divisor, args.device).bfloat16()
        tensors = dict(x=values((rows, 1024)), weight=values((1024,), 1013, 31, -1000, 2048),
                       branch=values((rows, 1024), 1019, 41, 509, 256),
                       gate=values((rows, 2048), 1021, 43, 510, 128),
                       conv_weight=values((1024, 1, 5), 1031, 47, 515, 1024),
                       q=values((1, rows, 1024), 1033, 53, 516, 512),
                       k=values((1, rows, 1024), 1039, 59, 519, 512),
                       v=values((1, rows, 1024), 1049, 61, 524, 256),
                       relative=values((1, 13, 1024), 1051, 67, 525, 512),
                       scale=values((128,), 1061, 71, 530, 512))
        mask = torch.ones((1, rows), dtype=torch.bool, device=args.device)
        if rows > 12:
            mask[:, -2:] = False
            mask[:, 3] = False
        if rows >= 25:
            mask[:, :12] = False
        tensors["mask"] = mask.bfloat16()
        tensors["branch"][rows // 2, 3] = 2e10
        x, branch, weight = [tensors[k].to(dtype) for k in ("x", "branch", "weight")]
        norm = definition.Gemma4RMSNorm(1024).to(device=args.device, dtype=dtype)
        norm.weight.copy_(weight)
        expected = dict(relu=F.relu(x), silu=F.silu(x), glu=F.glu(tensors["gate"].to(dtype)),
                        softplus=F.softplus(tensors["scale"].to(dtype)),
                        branch_out=(norm(branch.clamp(-1e10, 1e10)) * 0.5 + x))
        expected["branch_wrong"] = (norm(branch.clamp(-1e10, 1e10)) + x) * 0.5
        halves = tensors["gate"].to(dtype).chunk(2, -1)
        expected["glu_wrong"] = F.glu(torch.cat(halves[::-1], -1))
        conv = definition.Gemma4AudioCausalConv1d(1024, 1024, 5, groups=1024, bias=False).to(
            device=args.device, dtype=dtype)
        conv.weight.copy_(tensors["conv_weight"].to(dtype))
        expected["conv"] = conv(x.T[None])[0].T.contiguous()
        batched = F.pad(x.T, (0, 13))[None].repeat(2, 1, 1)
        expected["conv_shape"] = conv(batched)[0, :, :rows].T.contiguous()
        expected["conv_wrong"] = conv(x.roll(1, 0).T[None])[0].T.contiguous()
        attention = definition.Gemma4AudioAttention(config.audio_config, 0).to(device=args.device, dtype=dtype)
        attention.q_proj = Projection(tensors["q"].to(dtype))
        attention.k_proj = Projection(tensors["k"].to(dtype))
        attention.v_proj = Projection(tensors["v"].to(dtype))
        attention.post = torch.nn.Identity()
        attention.relative_k_proj = torch.nn.Identity()
        attention.per_dim_scale.copy_(tensors["scale"].to(dtype))
        hidden = x[None]
        def run_attention(left, padded=False):
            inputs = F.pad(hidden, (0, 0, 0, 12)) if padded else hidden
            valid = F.pad(mask, (0, 12), value=False) if padded else mask
            for key in ("q", "k", "v"):
                value = tensors[key].to(dtype)
                getattr(attention, key + "_proj").value = F.pad(value, (0, 0, 0, 12)) if padded else value
            dense_mask = definition.create_bidirectional_mask(config=config.audio_config, inputs_embeds=inputs,
                attention_mask=valid, and_mask_function=definition.sliding_window_mask_function((left, 0)))
            blocked = definition.Gemma4AudioModel._convert_4d_mask_to_blocked_5d(
                SimpleNamespace(config=config.audio_config), dense_mask)
            return attention(inputs, tensors["relative"].to(dtype), blocked)[0][0, :rows]
        expected["attention"] = run_attention(12)
        expected["attention_wrong"] = run_attention(13)
        expected["attention_repeat"] = run_attention(12)
        assert torch.equal(expected["attention"], expected["attention_repeat"])
        expected["attention_shape"] = run_attention(12, padded=True)
        expected["branch_out_shape"] = (norm(F.pad(branch, (0, 0, 0, 13)).clamp(-1e10, 1e10))[:rows] * 0.5 + x)
        outputs = ["relu", "silu", "glu", "softplus", "conv", "branch_out", "attention"]
        for key, value in expected.items():
            assert torch.isfinite(value).all(), key
            tensors[key] = value.float()
        name = f"audio-ops-{rows}"
        (args.output / f"{name}.safetensors").write_bytes(save({k: v.cpu().contiguous() for k,v in tensors.items()}))
        report["cells"][name] = dict(rows=rows, outputs=outputs,
            variations={"conv": "conv_shape", "attention": "attention_shape", "branch_out": "branch_out_shape"},
            controls={"branch_out": "branch_wrong", "glu": "glu_wrong", **({"conv": "conv_wrong"} if rows > 1 else {}),
                      **({"attention": "attention_wrong"} if rows >= 13 else {})})
        print(f"PASS {name} reference_repeat_bits=0", flush=True)
    for time, frequency, channels in [(7, 5, 3), (2999, 128, 1), (1500, 64, 128)]:
        x = pattern((time, frequency, channels), 1063, 73, 531, 256, args.device).bfloat16()
        mask = torch.ones(time, device=args.device, dtype=torch.bool)
        mask[-1] = False
        masked = x * mask[:, None, None]
        patches = F.unfold(masked.permute(2, 0, 1)[None].to(dtype), (3, 3), padding=1, stride=2)[0].T
        wrong = F.unfold(x.permute(2, 0, 1)[None].to(dtype), (3, 3), padding=1, stride=2)[0].T
        name = f"audio-patch-{time}-{channels}"
        (args.output / f"{name}.safetensors").write_bytes(save(dict(x=x.cpu(), mask=mask.bfloat16().cpu(),
            patches=patches.float().cpu().contiguous(), wrong=wrong.float().cpu().contiguous())))
        report["cells"][name] = dict(shape=[time, frequency, channels], outputs=["patches"], controls={"patches":"wrong"})
        print(f"PASS {name} masked_rows=true", flush=True)
    (args.output / "AUDIO_OPS.json").write_text(json.dumps(report, indent=2) + "\n")


def freeze_audio_operators(args):
    from safetensors.torch import load_file
    import shutil
    reports = [json.loads((p / "AUDIO_OPS.json").read_text()) for p in [args.oracle, *args.arms]]
    assert len({(r["torch"], r["transformers"]) for r in reports}) == 1
    assert {tuple(r["capability"]) for r in reports} == {(8, 9), (8, 6)}
    assert {r["dtype"] for r in reports} == {"bf16", "fp32"}
    frozen = dict(status="FROZEN", candidate_observations=0,
                  rule="2 * maximum reference-only distance to FP32 oracle", reference_reports=reports, cells={})
    for name, cell in reports[0]["cells"].items():
        paths = [args.oracle, *args.arms]
        selected = [p for p, r in zip(paths, reports) if not cell.get("exact_bf16") or r["dtype"] == "bf16"]
        oracle_path = selected[0] if cell.get("exact_bf16") else args.oracle
        oracle = load_file(oracle_path / f"{name}.safetensors")
        bounds = {key: dict(output_max=0.0, output_rmse=0.0) for key in cell["outputs"]}
        for path in selected:
            arm = load_file(path / f"{name}.safetensors")
            for key, bound in bounds.items():
                for variant in (key, cell.get("variations", {}).get(key, key)):
                    delta = arm[variant].double() - oracle[key].double()
                    if cell.get("exact_bf16"):
                        assert torch.equal(arm[variant], oracle[key]), "BF16 relative table varies across GPUs"
                    bound["output_max"] = max(bound["output_max"], 2*delta.abs().max().item())
                    bound["output_rmse"] = max(bound["output_rmse"], 2*delta.square().mean().sqrt().item())
        for key, control in cell["controls"].items():
            delta = oracle[control].double() - oracle[key].double()
            assert delta.abs().max().item() > bounds[key]["output_max"] or delta.square().mean().sqrt().item() > bounds[key]["output_rmse"], (name,key)
        frozen["cells"][name] = dict(**cell, bounds=bounds)
        shutil.copyfile(oracle_path / f"{name}.safetensors", args.output / f"{name}.safetensors")
    (args.output / "AUDIO_FROZEN.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"PASS audio_operator_freeze cells={len(frozen['cells'])} candidate_observations=0", flush=True)


def freeze_model(args):
    from safetensors.torch import load_file
    import shutil
    assert args.oracle and args.arms and args.inputs
    versions = [json.loads((arm / "LOAD.json").read_text()) for arm in [args.oracle, *args.arms]]
    assert len({(v["torch"], v["transformers"]) for v in versions}) == 1, "mixed reference software versions"
    metadata = json.loads((args.inputs / "INPUT.json").read_text())
    frozen = dict(status="FROZEN", candidate_observations=0, rule="2 * maximum reference-only distance to FP32 oracle", cells={})
    for name, cell in metadata.items():
        expected = load_file(args.oracle / f"{name}-packed.safetensors")
        ids = load_file(args.inputs / f"{name}-input.safetensors")["ids"].long()[cell["score_first"]:]
        bounds = {key: dict(output_max=0.0, output_rmse=0.0) for key in expected}
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
    parser.add_argument("--stage", choices=("operators", "attention", "inputs", "model", "media", "freeze", "audio-operators", "audio-freeze"), default="operators")
    parser.add_argument("--images", type=Path, nargs=2)
    parser.add_argument("--audio", type=Path, nargs=2)
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
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.stage == "audio-operators":
        capture_audio_operators(args, config)
        return
    if args.stage == "audio-freeze":
        freeze_audio_operators(args)
        return
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
