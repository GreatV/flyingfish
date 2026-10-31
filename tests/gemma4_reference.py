"""Gemma text model-definition oracle over published BF16 tensor ranges."""
import argparse
import json
from pathlib import Path
import struct

import torch
import transformers
from safetensors.torch import load_file, save_file
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import apply_renamings


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def host_cache_budget():
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    group = next(line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    mount = Path("/sys/fs/cgroup")
    directory = mount / group.lstrip("/")
    while directory != mount:
        maximum = (directory / "memory.max").read_text().strip()
        if maximum != "max":
            current = int((directory / "memory.current").read_text())
            stats = dict(line.split() for line in (directory / "memory.stat").read_text().splitlines())
            non_file = current - int(stats["file"]) + int(stats["shmem"])
            available = min(available, max(0, int(maximum) - non_file))
        directory = directory.parent
    return max(0, available - 8 * 1024 ** 3)


class PublishedWeights:
    def __init__(self, directory, model_type, cache_budget=0):
        self.entries = {}
        self.consumed = set()
        self.cache = {}
        self.cache_budget = cache_budget
        self.cached_bytes = 0
        renamings = get_checkpoint_conversion_mapping(model_type) or []
        for path in sorted(directory.glob("*.safetensors")):
            with path.open("rb") as source:
                header_size = struct.unpack("<Q", source.read(8))[0]
                require(header_size <= 16 * 1024 * 1024, "oversized tensor header")
                header = json.loads(source.read(header_size))
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                target = apply_renamings(name, renamings)
                require(target not in self.entries, f"duplicate tensor {target}")
                require(info["dtype"] == "BF16", f"non-BF16 published tensor {name}")
                start, end = info["data_offsets"]
                require(0 <= start < end <= path.stat().st_size - 8 - header_size,
                        f"invalid tensor extent {name}")
                self.entries[target] = (path, 8 + header_size + start, end - start, info["shape"])

    def inventory(self, model):
        shapes = model.state_dict()
        require(set(shapes) - set(self.entries) == {"lm_head.weight"},
                "definition has unbound parameters other than the tied head")
        for name, (_, _, _, shape) in self.entries.items():
            require(name in shapes and list(shapes[name].shape) == shape,
                    f"published tensor/definition shape mismatch {name}")
        require(model._tied_weights_keys == {"lm_head.weight": "model.language_model.embed_tokens.weight"},
                "unexpected tied-weight declaration")

    def read(self, name, dtype, device):
        path, offset, length, shape = self.entries[name]
        data = self.cache.get(name)
        if data is None:
            with path.open("rb", buffering=0) as source:
                source.seek(offset)
                data = bytearray(length)
                view = memoryview(data)
                read = 0
                while read < length:
                    count = source.readinto(view[read:])
                    require(count, f"short published tensor read {name}")
                    read += count
            if ".layers." in name and self.cached_bytes + length <= self.cache_budget:
                self.cache[name] = data
                self.cached_bytes += length
        value = torch.frombuffer(data, dtype=torch.bfloat16).reshape(shape).to(device=device, dtype=dtype)
        require(bool(torch.isfinite(value).all()), f"nonfinite published tensor {name}")
        self.consumed.add(name)
        return value

    def bind(self, module, prefix, dtype, device):
        names = list(module.state_dict())
        for name in names:
            value = self.read(prefix + name, dtype, device)
            parent, _, leaf = name.rpartition(".")
            owner = module.get_submodule(parent) if parent else module
            if leaf in owner._parameters:
                setattr(owner, leaf, torch.nn.Parameter(value, requires_grad=False))
            else:
                setattr(owner, leaf, value)
        require(all(not value.is_meta for value in module.state_dict().values()),
                f"unbound executed module {prefix}")


def stream_model(model_class, config, weights, dtype, device):
    with torch.device("meta"):
        model = model_class(config)
    weights.inventory(model)
    model.eval().requires_grad_(False)
    text = model.model.language_model
    weights.bind(text.embed_tokens, "model.language_model.embed_tokens.", dtype, device)
    text.embed_tokens.embed_scale = torch.tensor(config.text_config.hidden_size ** 0.5, device=device)
    weights.bind(text.norm, "model.language_model.norm.", dtype, device)
    model.lm_head.weight = text.embed_tokens.weight
    text.rotary_emb = type(text.rotary_emb)(config.text_config).to(device)
    hooks = []
    for index, layer in enumerate(text.layers):
        prefix = f"model.language_model.layers.{index}."

        def enter(module, inputs, prefix=prefix):
            weights.bind(module, prefix, dtype, device)

        def leave(module, inputs, output):
            module.to_empty(device="meta")

        hooks.append(layer.register_forward_pre_hook(enter))
        hooks.append(layer.register_forward_hook(leave))
    return model, hooks


def bind_media(model, weights, dtype, device):
    core = model.model
    modules = ["embed_vision"]
    if hasattr(core, "vision_tower"):
        modules.append("vision_tower")
    if getattr(core, "embed_audio", None) is not None:
        modules.append("embed_audio")
    for name in modules:
        weights.bind(getattr(core, name), f"model.{name}.", dtype, device)
    if hasattr(core, "vision_tower"):
        tower = core.vision_tower
        tower.encoder.rotary_emb = type(tower.encoder.rotary_emb)(model.config.vision_config).to(device)
    return {name for name in weights.entries if any(name.startswith(f"model.{part}.") for part in modules)}


@torch.inference_mode()
def prepare_media(args):
    from PIL import Image
    processor = transformers.AutoProcessor.from_pretrained(args.model, local_files_only=True)
    require(args.images and len(args.images) == 2, "two media-control images required")
    images = [Image.open(path).convert("RGB") for path in args.images]
    images = [picture.resize((768, 768), Image.Resampling.BICUBIC) for picture in images]
    sentence = processor.tokenizer(" Objects and buildings appear in the surrounding scene.",
                                   add_special_tokens=False).input_ids
    cases = [("street", [images[0]], [], processor.image_token + " Caption: A photograph of", 16),
             ("winter", [images[1]], [], processor.image_token + " Caption: A photograph of", 16),
             ("wide", [images[0].resize((960, 672), Image.Resampling.BICUBIC)], [],
              processor.image_token + " Describe the image.", 16),
             ("pair", images, [], processor.image_token + " First picture. " + processor.image_token + " Compare.", 16),
             ("long", [images[0]], [], processor.image_token + " Describe the image.", 1024)]
    raw = json.loads((args.model / "config.json").read_text())
    if raw.get("audio_config") is not None:
        require(args.audio and len(args.audio) == 2, "two raw-audio controls required")
        waves = []
        for path in args.audio:
            pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
                                  "-acodec", "pcm_f32le", "-ac", "1", "-ar", "16000", "-"],
                                 check=True, capture_output=True).stdout
            waves.append(torch.frombuffer(bytearray(pcm), dtype=torch.float32).numpy().copy())
        token = processor.audio_token
        cases += [("stew", [], [waves[0]], token + " Transcription:", 16),
                  ("wizard", [], [waves[1]], token + " Transcription:", 16),
                  ("audio_pair", [], waves, token + " First clip. " + token + " Transcribe.", 16),
                  ("mixed", [images[0]], [waves[1]], token + " Sound. " + processor.image_token + " Describe both.", 16)]
    metadata = {}
    for name, pictures, waves, text, extra in cases:
        arguments = dict(text=text, return_tensors="pt")
        if pictures:
            arguments["images"] = pictures
        if waves:
            arguments.update(audio=waves, sampling_rate=16000)
        batch = processor(**arguments)
        prompt = batch.input_ids.shape[1]
        suffix = (sentence * (extra // len(sentence) + 1))[:extra]
        ids = torch.cat((batch.input_ids[0], torch.tensor(suffix)))
        modalities = torch.cat((batch.mm_token_type_ids[0], torch.zeros(len(suffix), dtype=torch.long)))
        values = dict(ids=ids.int(), modalities=modalities.to(torch.uint8))
        grids = []
        if pictures:
            masks = (batch.image_position_ids >= 0).all(-1)
            values.update(padded_pixels=batch.pixel_values.float(), padded_positions=batch.image_position_ids.int())
            values["pixels"] = batch.pixel_values[masks].float()
            values["positions"] = batch.image_position_ids[masks].int()
            lengths = masks.sum(-1).tolist()
            require(len(set(lengths)) == 1, "packed reference images must have matching patch counts")
            for coordinates, mask in zip(batch.image_position_ids, masks):
                x, y = (coordinates[mask].amax(0) + 1).tolist()
                grids.append([1, y, x])
        if waves:
            values.update(audio=batch.input_features.float(), audio_mask=batch.input_features_mask.bool())
        save_file(values, str(args.out / f"{name}-input.safetensors"))
        rows = 513 if name == "long" else 17
        metadata[name] = dict(prompt_length=prompt, score_first=len(ids) - rows,
                              score_rows=rows, total_tokens=len(ids), grids=grids,
                              audio_clips=len(waves), image_count=len(pictures))
    require(metadata["long"]["total_tokens"] > raw["text_config"]["sliding_window"], "long does not wrap KV")
    (args.out / "INPUT.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"PASS large_media_inputs cells={len(cases)} GPU_calls=0", flush=True)


def media_arguments(data, cell, variant, device):
    arguments = dict(input_ids=data["ids"][None].long().to(device),
                     mm_token_type_ids=data["modalities"][None].long().to(device))
    if cell["image_count"]:
        if variant == "packed":
            pixels = data["pixels"].reshape(cell["image_count"], -1, data["pixels"].shape[-1])
            positions = data["positions"].reshape(cell["image_count"], -1, 2)
        else:
            pixels, positions = data["padded_pixels"], data["padded_positions"]
        arguments.update(pixel_values=pixels.to(device), image_position_ids=positions.long().to(device))
    if cell["audio_clips"]:
        samples, mask = data["audio"], data["audio_mask"]
        if variant == "packed":
            samples = samples[mask][None]
            mask = torch.ones(samples.shape[:2], dtype=torch.bool)
        else:
            samples = torch.nn.functional.pad(samples, (0, 0, 0, 7))
            mask = torch.nn.functional.pad(mask, (0, 7))
        arguments.update(input_features=samples.to(device), input_features_mask=mask.to(device))
    return arguments


@torch.inference_mode()
def capture_media(args, model, weights, hooks, expected, device):
    metadata = json.loads((args.media_inputs / "INPUT.json").read_text())
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    core = model.model
    calls = dict(image=0, audio=0)
    recorded = {}
    active = {}

    def image_output(module, inputs, output):
        calls["image"] += 1
        values = output.pooler_output if hasattr(output, "pooler_output") else output
        values = values.reshape(-1, values.shape[-1])
        if hasattr(output, "pooler_output"):
            valid = (active["image_position_ids"] >= 0).all(-1).reshape(-1)
            values = values[valid]
        recorded["image"] = values.float().cpu().contiguous()

    def audio_output(module, inputs, output):
        calls["audio"] += 1
        recorded["audio"] = output[active["input_features_mask"]].float().cpu().contiguous()

    hooks.append(core.embed_vision.register_forward_hook(image_output))
    if hasattr(core.embed_vision, "patch_ln1"):
        stages = [("patch_ln1", "vision.patch_ln1"), ("patch_dense", "vision.patch_dense"),
                  ("patch_ln2", "vision.patch_ln2"), ("pos_norm", "vision.pos_norm"),
                  ("multimodal_embedder.embedding_pre_projection_norm", "vision.rms")]
        for path, label in stages:
            def record_image(module, inputs, output, label=label):
                valid = (active["image_position_ids"] >= 0).all(-1)
                recorded[label] = output[valid].float().cpu().contiguous()
            hooks.append(core.embed_vision.get_submodule(path).register_forward_hook(record_image))
        def record_position(module, inputs):
            valid = (active["image_position_ids"] >= 0).all(-1)
            recorded["vision.position"] = inputs[0][valid].float().cpu().contiguous()
        hooks.append(core.embed_vision.pos_norm.register_forward_pre_hook(record_position))
    if getattr(core, "embed_audio", None) is not None:
        hooks.append(core.embed_audio.register_forward_hook(audio_output))
        def record_audio(module, inputs, output):
            recorded["audio.raw_norm"] = output[active["input_features_mask"]].float().cpu().contiguous()
        hooks.append(core.embed_audio.embedding_pre_projection_norm.register_forward_hook(record_audio))
    report = dict(class_name=type(model).__name__, torch=torch.__version__, transformers=transformers.__version__,
                  device=torch.cuda.get_device_name(0), dtype=args.dtype, strict=args.strict_reduction,
                  unbound_executed=0, unexpected=0, newly_initialized_executed=0, full_class_load=False, cells={})
    controls = {}
    for name, cell in metadata.items():
        if args.media_review and name not in ("street", "winter", "stew", "wizard"):
            continue
        data = load_file(str(args.media_inputs / f"{name}-input.safetensors"))
        cell = dict(cell)
        variants = ("padded",) if args.media_review else ("packed", "padded", "chunk") if name == "long" else ("packed", "padded")
        for variant in variants:
            arguments = media_arguments(data, cell, variant, device)
            active = arguments
            before = dict(calls)
            recorded.clear()
            if args.media_review:
                length = cell["prompt_length"]
                for key in ("input_ids", "mm_token_type_ids"):
                    arguments[key] = arguments[key][:, :length]
                output = model(**arguments, use_cache=True, logits_to_keep=1)
                logits = output.logits[0, -1].float().cpu()
                generated = []
                for step in range(32):
                    token = output.logits[:, -1].argmax(-1, keepdim=True)
                    generated.append(token.item())
                    if step != 31:
                        output = model(input_ids=token, past_key_values=output.past_key_values,
                                       use_cache=True, logits_to_keep=1)
                cell.update(decoded=tokenizer.decode(generated, skip_special_tokens=False), generated_ids=generated)
                controls[name] = logits
                print(f"MEDIA {name} decoded={cell['decoded']!r}", flush=True)
            else:
                rows = cell["score_rows"]
                if variant == "chunk":
                    prompt = cell["prompt_length"]
                    prefix = {key: value for key, value in arguments.items()}
                    prefix["input_ids"] = arguments["input_ids"][:, :prompt]
                    prefix["mm_token_type_ids"] = arguments["mm_token_type_ids"][:, :prompt]
                    output = model(**prefix, use_cache=True, logits_to_keep=1)
                    cache, outputs = output.past_key_values, []
                    for first in range(prompt, data["ids"].numel(), 512):
                        end = min(first + 512, data["ids"].numel())
                        output = model(input_ids=arguments["input_ids"][:, first:end],
                                       mm_token_type_ids=arguments["mm_token_type_ids"][:, first:end],
                                       position_ids=torch.arange(first, end, device=device)[None],
                                       past_key_values=cache, use_cache=True)
                        cache = output.past_key_values
                        selected = max(0, cell["score_first"] - first)
                        if selected < end - first:
                            outputs.append(output.logits[0, selected:].float().cpu())
                    logits = torch.cat(outputs)
                else:
                    output = model(**arguments, use_cache=False, logits_to_keep=rows)
                    logits = output.logits[0].float().cpu().contiguous()
                captured = dict(logits=logits)
                chunks = []
                image_offset = audio_offset = 0
                kinds, counts = torch.unique_consecutive(data["modalities"], return_counts=True)
                for kind, count in zip(kinds.tolist(), counts.tolist()):
                    if kind == 1:
                        chunks.append(recorded["image"][image_offset:image_offset + count])
                        image_offset += count
                    elif kind == 3:
                        chunks.append(recorded["audio"][audio_offset:audio_offset + count])
                        audio_offset += count
                require(chunks, "reference did not consume media")
                captured["features"] = torch.cat(chunks)
                captured.update({key: value for key, value in recorded.items() if "." in key})
                require(all(bool(torch.isfinite(value).all()) for value in captured.values()), "nonfinite media reference")
                save_file(captured, str(args.out / f"{name}-{variant}.safetensors"))
                if variant == "packed":
                    recorded.clear()
                    repeated = model(**arguments, use_cache=False, logits_to_keep=rows)
                    require(torch.equal(repeated.logits[0].float().cpu(), logits), "media explicit repeat differs")
                    del repeated
                    cell["repeat_bits"] = 0
            multiplier = 2 if not args.media_review and variant == "packed" else 1
            observed = {key: calls[key] - before[key] for key in calls}
            require(observed == dict(image=multiplier * bool(cell["image_count"]),
                                     audio=multiplier * bool(cell["audio_clips"])), f"encoder calls {observed}")
            cell[variant + "_calls"] = observed
            del output, arguments, logits
        report["cells"][name] = cell
        (args.out / "REFERENCE.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASS media_reference {name}", flush=True)
    require(weights.consumed == expected, "streamed media executed-key mismatch")
    report["consumed_keys"] = len(weights.consumed)
    if args.media_review:
        for a, b in (("street", "winter"), ("stew", "wizard")):
            if a in controls:
                require(not torch.equal(controls[a], controls[b]), "two-media logits are identical")
                require(report["cells"][a]["generated_ids"] != report["cells"][b]["generated_ids"], "two-media decoded output identical")
    (args.out / "REFERENCE.json").write_text(json.dumps(report, indent=2) + "\n")
    for hook in hooks:
        hook.remove()
    print(f"PASS large_media_reference cells={len(report['cells'])} executed_keys={len(expected)}", flush=True)


def freeze_media(args):
    import shutil
    arms = args.freeze_media_from
    metadata = json.loads((args.media_inputs / "INPUT.json").read_text())
    reports = [json.loads((arm / "REFERENCE.json").read_text()) for arm in arms]
    require(reports[0]["dtype"] == "fp32", "media oracle must be FP32")
    devices = {report["device"] for report in reports}
    coverage = {(report["device"], report["dtype"], report["strict"]) for report in reports}
    require(len(devices) == 2 and coverage == {(device, dtype, strict) for device in devices
            for dtype, strict in (("fp32", False), ("bf16", False), ("bf16", True))}, "incomplete media reference coverage")
    for report in reports:
        require(report["torch"] == torch.__version__ and report["transformers"] == transformers.__version__,
                "media reference versions differ")
        require(report["class_name"] == reports[0]["class_name"] and report["unbound_executed"] ==
                report["unexpected"] == report["newly_initialized_executed"] == 0, "unclean media reference")
    frozen = dict(status="FROZEN", policy="2*maximum reference-only distance to FP32 oracle", cells={}, arms=reports)
    for name, cell in metadata.items():
        expected = load_file(str(arms[0] / f"{name}-packed.safetensors"))
        ids = load_file(str(args.media_inputs / f"{name}-input.safetensors"))["ids"].long()[cell["score_first"]:]
        bounds = {key: dict(output_max=0.0, output_rmse=0.0) for key in expected}
        bounds["logits"]["nll_max"] = 0.0
        for arm, report in zip(arms, reports):
            require(report["cells"][name]["repeat_bits"] == 0, "media repeat mismatch")
            for variant in (("packed", "padded", "chunk") if name == "long" else ("packed", "padded")):
                actual = load_file(str(arm / f"{name}-{variant}.safetensors"))
                for key in expected:
                    require(actual[key].shape == expected[key].shape and bool(torch.isfinite(actual[key]).all()),
                            "media freeze shape or finite check")
                    delta = actual[key].double() - expected[key].double()
                    bounds[key]["output_max"] = max(bounds[key]["output_max"], 2 * float(delta.abs().max()))
                    bounds[key]["output_rmse"] = max(bounds[key]["output_rmse"], 2 * float(delta.square().mean().sqrt()))
                metric = distances(actual["logits"], expected["logits"], ids[1:].tolist())
                bounds["logits"]["nll_max"] = max(bounds["logits"]["nll_max"], 2 * metric["nll_max"])
        bounds["logits"]["margin_cut"] = 2 * bounds["logits"]["output_max"]
        frozen["cells"][name] = bounds
        print(f"FREEZE {name} {json.dumps(bounds, sort_keys=True)}", flush=True)
    shutil.copytree(args.media_inputs, args.out / "inputs")
    (args.out / "oracle").symlink_to(arms[0].resolve(), target_is_directory=True)
    with (args.out / "FROZEN.json").open("x") as destination:
        destination.write(json.dumps(frozen, indent=2) + "\n")
    print(f"PASS frozen_media_limits cells={len(metadata)} arms=6 candidate_observations=0", flush=True)


@torch.inference_mode()
def capture(model, ids, chunk, keep, device, prefix=0):
    cache = None
    values = []
    starts = ([0] if prefix else []) + list(range(prefix, len(ids), chunk))
    for start in starts:
        piece = ids[start:prefix if prefix and start == 0 else start + chunk]
        end = start + len(piece)
        selected = max(start, len(ids) - keep)
        retained = end - selected
        output = model(
            input_ids=torch.tensor([piece], device=device),
            position_ids=torch.arange(start, end, device=device).unsqueeze(0),
            past_key_values=cache, use_cache=True,
            logits_to_keep=max(1, retained),
        )
        cache = output.past_key_values
        if retained > 0:
            values.append(output.logits[0].float().cpu().contiguous())
        del output
    result = torch.cat(values)
    require(result.shape == (keep, model.config.text_config.vocab_size), "reference logits extent")
    require(bool(torch.isfinite(result).all()), "nonfinite reference logits")
    return result


def distances(actual, expected, targets):
    require(actual.shape == expected.shape, "metric shape mismatch")
    delta = actual.double() - expected.double()
    losses = []
    for values in (actual, expected):
        rows = values[:len(targets)].double()
        loss = rows.logsumexp(-1) - rows.gather(1, torch.tensor(targets).unsqueeze(1)).squeeze(1)
        losses.append(float(loss.mean()))
    return {"output_max": float(delta.abs().max()),
            "output_rmse": float(delta.square().mean().sqrt()),
            "nll_max": abs(losses[0] - losses[1])}


def freeze(arms, output):
    inputs = json.loads((arms[0] / "ids.json").read_text())
    oracle_meta = json.loads((arms[0] / "LOAD.json").read_text())
    require(oracle_meta["dtype"] == "fp32", "first freeze arm must be the FP32 oracle")
    oracle = load_file(str(arms[0] / "reference.safetensors"))
    families = {tag: {key: 0.0 for key in ("output_max", "output_rmse", "nll_max")}
                for tag in ("a", "b", "wrap", "long")}
    references, coverage = [], set()
    for arm in arms:
        require(json.loads((arm / "ids.json").read_text()) == inputs, "reference token IDs differ")
        meta = json.loads((arm / "LOAD.json").read_text())
        require(meta["unbound_executed"] == meta["unexpected"] == meta["newly_initialized_executed"] == 0,
                "unclean streamed reference arm")
        require(meta["class"] == oracle_meta["class"], "reference model classes differ")
        require(meta["torch"] == oracle_meta["torch"] == torch.__version__
                and meta["transformers"] == oracle_meta["transformers"] == transformers.__version__,
                "reference software versions differ")
        coverage.add((meta["device"], meta["dtype"]))
        noise = json.loads((arm / "NOISE.json").read_text())
        values = load_file(str(arm / "reference.safetensors"))
        measured = {}
        for tag, limits in families.items():
            require(all(value == 0 for value in noise[tag]["repeat"].values()), "reference repeat is not exact")
            ids = inputs["short"][0 if tag == "a" else 1] if tag in ("a", "b") else inputs[tag]
            for mode in ("prefill", "decode"):
                name = f"{tag}.{mode}"
                target = oracle[f"{tag}.prefill"]
                delta = distances(values[name], target, ids[len(ids) - len(target) + 1:])
                measured[name] = delta
                for key in limits:
                    limits[key] = max(limits[key], 2 * delta[key])
        references.append({"device": meta["device"], "dtype": meta["dtype"], "distances": measured})
        del values
    devices = {device for device, _ in coverage}
    require(len(devices) == 2 and coverage == {(device, dtype) for device in devices for dtype in ("bf16", "fp32")},
            "freeze requires BF16 and FP32 on both GPUs")
    for limits in families.values():
        limits["margin_cut"] = 2 * limits["output_max"]
    for tag in families:
        oracle[f"{tag}.decode"] = oracle[f"{tag}.prefill"].clone()
    save_file(oracle, str(output / "reference.safetensors"))
    (output / "ids.json").write_text(json.dumps(inputs, indent=2) + "\n")
    frozen = {"status": "FROZEN", "policy": "2*max reference-only distance to FP32 prefill oracle",
              "families": families, "arms": references}
    with (output / "FROZEN.json").open("x") as destination:
        destination.write(json.dumps(frozen, indent=2) + "\n")
    print("PASS frozen_reference_limits arms=4 candidate_observations=0", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--full-load-only", action="store_true")
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--validate-full", type=Path)
    parser.add_argument("--freeze-from", type=Path, nargs=4)
    args = parser.parse_args()
    require(not args.out.resolve().is_relative_to(args.model.resolve()), "output inside read-only weights")
    require(transformers.__version__ == "5.21.0", "reference definition version mismatch")
    require(torch.__version__ == "2.14.1+cu130", "reference torch version mismatch")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.freeze_from:
        freeze(args.freeze_from, args.out)
        return
    raw = json.loads((args.model / "config.json").read_text())
    require(raw["architectures"] in (["Gemma4ForConditionalGeneration"],
                                    ["Gemma4UnifiedForConditionalGeneration"]), "reference class mismatch")
    model_class = getattr(transformers, raw["architectures"][0])
    if args.validate_full:
        require(args.dtype == "bf16" and args.ids is not None, "full validation requires BF16 and shared IDs")
        model, loading = model_class.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="eager", output_loading_info=True)
        require(all(not loading[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
                "unclean full GPU reference load")
        require(all(not value.is_meta for value in model.state_dict().values()), "unbound full GPU reference")
        model.to(torch.device("cuda", 0)).eval()
        inputs = json.loads(args.ids.read_text())
        expected = load_file(str(args.validate_full / "reference.safetensors"))
        for tag, ids in zip(("a", "b"), inputs["short"]):
            actual = capture(model, ids, len(ids), len(ids), torch.device("cuda", 0))
            require(torch.equal(actual, expected[f"{tag}.prefill"]), f"full/streamed model-definition mismatch {tag}")
        (args.out / "LOAD.json").write_text(json.dumps({"class": model_class.__name__, "full_class_load": True,
            "missing": 0, "unexpected": 0, "newly_initialized": 0, "full_streamed_bits": 0, "cells": 2}, indent=2) + "\n")
        print("PASS full_streamed_definition cells=2 bits=0 missing=0 unexpected=0 newly_initialized=0", flush=True)
        return
    if args.full_load_only:
        model, loading = model_class.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="eager", output_loading_info=True)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            require(key in loading and not loading[key], f"unclean reference load {key}: {loading.get(key)}")
        require(all(not value.is_meta for value in model.state_dict().values()), "meta parameter after full load")
        report = {"class": model_class.__name__, "loading": {key: list(value) for key, value in loading.items()},
                  "missing": 0, "unexpected": 0, "newly_initialized": 0, "full_class_load": True}
        (args.out / "LOAD.json").write_text(json.dumps(report, indent=2) + "\n")
        print("PASS full_class_load missing=0 unexpected=0 newly_initialized=0", flush=True)
        return
    config = model_class.config_class.from_pretrained(args.model, local_files_only=True)
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    device = torch.device("cuda", 0)
    weights = PublishedWeights(args.model, raw["model_type"], host_cache_budget())
    model, hooks = stream_model(model_class, config, weights, dtype, device)
    window = config.text_config.sliding_window
    if args.ids:
        inputs = json.loads(args.ids.read_text())
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        texts = ["The small brown bird rested on a smooth stone beside the river.",
                 "During the winter workshop, students designed a careful map of the old railway."]
        short = [tokenizer(text).input_ids for text in texts]
        seed = short[0] + short[1]
        wrap = (seed * ((window + 8) // len(seed) + 1))[:window + 8]
        inputs = {"texts": texts, "short": short, "wrap": wrap, "long": wrap}
    short, wrap = inputs["short"], inputs["wrap"]
    require(len(short) == 2 and all(8 <= len(ids) <= 32 for ids in short), "short reference input extent")
    require(len(wrap) == window + 8 and inputs["long"] == wrap, "window reference input extent")
    require(all(type(token) is int and 0 <= token < config.text_config.vocab_size
                for ids in short + [wrap] for token in ids), "invalid reference token ID")
    (args.out / "ids.json").write_text(json.dumps(inputs, indent=2) + "\n")
    tensors, noise = {}, {}
    cases = [("a", short[0], len(short[0]), 1, 0), ("b", short[1], len(short[1]), 1, 0),
             ("wrap", wrap, 8, 1, window), ("long", wrap, 513, window // 2, 0)]
    for tag, ids, keep, chunk, prefix in cases:
        direct = capture(model, ids, len(ids), keep, device)
        again = capture(model, ids, len(ids), keep, device)
        require(torch.equal(direct, again), f"non-identical explicit reference repeat {tag}")
        varied = capture(model, ids, chunk, keep, device, prefix)
        targets = ids[len(ids) - keep + 1:]
        tensors[f"{tag}.prefill"] = direct
        tensors[f"{tag}.decode"] = varied
        noise[tag] = {"repeat": distances(again, direct, targets),
                      "shape": distances(varied, direct, targets), "chunk": chunk, "prefix": prefix}
        save_file({f"{tag}.prefill": direct, f"{tag}.decode": varied}, str(args.out / f"{tag}.safetensors"))
        (args.out / f"{tag}.json").write_text(json.dumps(noise[tag], indent=2) + "\n")
        print(f"REFERENCE_CELL {tag} rows={keep} repeat_bits=0 chunk={chunk}", flush=True)
    expected = {name for name in weights.entries if name.startswith("model.language_model.")}
    require(weights.consumed == expected, "streamed text tensor key-consumption mismatch")
    report = {"class": model_class.__name__, "transformers": transformers.__version__,
              "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "dtype": args.dtype,
              "full_class_load": False, "checkpoint_tensors": len(weights.entries),
              "consumed_text_tensors": len(weights.consumed), "unbound_executed": 0,
              "unexpected": 0, "newly_initialized_executed": 0, "tied_head": True,
              "published_host_cache_bytes": weights.cached_bytes}
    save_file(tensors, str(args.out / "reference.safetensors"))
    (args.out / "NOISE.json").write_text(json.dumps(noise, indent=2) + "\n")
    (args.out / "LOAD.json").write_text(json.dumps(report, indent=2) + "\n")
    for tag, *_ in cases:
        (args.out / f"{tag}.safetensors").unlink()
        (args.out / f"{tag}.json").unlink()
    for hook in hooks:
        hook.remove()
    print(f"PASS streamed_reference cells={len(cases)} text_keys={len(expected)}", flush=True)


if __name__ == "__main__":
    main()
