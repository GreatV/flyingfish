"""DiffusionGemma model-definition reference over published BF16 tensor ranges."""
import argparse
import gc
import json
from pathlib import Path
import struct

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    DiffusionGemmaGenerationConfig,
    EntropyBoundSampler,
)
from transformers.models.diffusion_gemma import modeling_diffusion_gemma as definition


TAPS = (0, 5, 17, 29)
CASE_NAMES = ("encoder_p300", "encoder_p1500", "canvas_p300", "canvas_p1500", "commit256_p300")
FLOAT_METRICS = ("max_abs", "rmse", "cosine_defect")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tensor(value):
    return value.detach().cpu().contiguous()


def save_case(directory, name, tensors, meta):
    tensor_path = directory / f"{name}.safetensors"
    meta_path = directory / f"{name}.json"
    require(not tensor_path.exists() and not meta_path.exists(), f"fixture output exists {name}")
    for key, value in tensors.items():
        require(value.device.type == "cpu" and value.is_contiguous(), f"non-CPU output {name}/{key}")
        require(not value.dtype.is_floating_point or bool(torch.isfinite(value).all()),
                f"nonfinite output {name}/{key}")
    save_file(tensors, str(tensor_path))
    meta_path.write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")


def load_case(directory, name):
    return load_file(str(directory / f"{name}.safetensors")), json.loads((directory / f"{name}.json").read_text())


def optional_tensor(directory, name, key):
    if directory is None:
        return None
    with safe_open(str(directory / f"{name}.safetensors"), framework="pt", device="cpu") as source:
        return source.get_tensor(key) if key in source.keys() else None


def cache_extents(cache):
    stored, cumulative, offset = [], [], []
    for index, layer in enumerate(cache.layers):
        stored.append(layer.keys.shape[-2] if layer.is_initialized else 0)
        cumulative.append(layer.get_seq_length())
        _, start = cache.get_mask_sizes(0, index)
        offset.append(start)
    return {
        "kv_stored_len": torch.tensor(stored, dtype=torch.int64),
        "kv_cumulative_len": torch.tensor(cumulative, dtype=torch.int64),
        "kv_offset": torch.tensor(offset, dtype=torch.int64),
    }


def distance(actual, expected):
    require(actual.shape == expected.shape, "distance shape mismatch")
    if not (actual.dtype.is_floating_point and expected.dtype.is_floating_point):
        return {"mismatch": int((actual != expected).sum())}
    if torch.equal(actual, expected):
        return {key: 0.0 for key in FLOAT_METRICS}
    a = actual.reshape(-1)
    b = expected.reshape(-1)
    maximum = 0.0
    square = 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for start in range(0, a.numel(), 1 << 20):
        x = a[start:start + (1 << 20)].double()
        y = b[start:start + (1 << 20)].double()
        delta = x - y
        maximum = max(maximum, float(delta.abs().max()))
        square += float(delta.square().sum())
        dot += float((x * y).sum())
        norm_a += float(x.square().sum())
        norm_b += float(y.square().sum())
    cosine = dot / ((norm_a * norm_b) ** 0.5)
    return {"max_abs": maximum, "rmse": (square / a.numel()) ** 0.5,
            "cosine_defect": max(0.0, 1.0 - cosine)}


class PublishedWeights:
    def __init__(self, directory):
        self.entries = {}
        self.consumed = set()
        self.read_counts = {}
        for path in sorted(directory.glob("*.safetensors")):
            with path.open("rb") as source:
                header_size = struct.unpack("<Q", source.read(8))[0]
                require(header_size <= 16 * 1024 * 1024, "oversized tensor header")
                header = json.loads(source.read(header_size))
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                require(name not in self.entries, f"duplicate tensor {name}")
                require(info["dtype"] == "BF16", f"non-BF16 published tensor {name}")
                start, end = info["data_offsets"]
                require(0 <= start < end <= path.stat().st_size - 8 - header_size,
                        f"invalid tensor extent {name}")
                self.entries[name] = (path, 8 + header_size + start, end - start, info["shape"])

    @staticmethod
    def source_for(model_key):
        if model_key == "lm_head.weight":
            return "model.decoder.embed_tokens.weight"
        if model_key == "model.encoder.language_model.norm.weight":
            return "model.decoder.norm.weight"
        if model_key == "model.encoder.language_model.embed_tokens.weight":
            return "model.decoder.embed_tokens.weight"
        prefix = "model.encoder.language_model.layers."
        if model_key.startswith(prefix) and not model_key.endswith(".layer_scalar"):
            return model_key.replace(prefix, "model.decoder.layers.", 1)
        return model_key

    @staticmethod
    def is_text_key(name):
        return name.startswith("model.decoder.") or (
            name.startswith("model.encoder.language_model.layers.") and name.endswith(".layer_scalar")
        )

    def inventory(self, model):
        shapes = model.state_dict()
        for name, (_, _, _, shape) in self.entries.items():
            require(name in shapes and list(shapes[name].shape) == shape,
                    f"published tensor/definition shape mismatch {name}")
        for name, value in shapes.items():
            source = self.source_for(name)
            require(source in self.entries and list(value.shape) == self.entries[source][3],
                    f"unbound definition tensor {name}")
        text = {name for name in self.entries if self.is_text_key(name)}
        vision = set(self.entries) - text
        require(len(self.entries) == 1047 and len(text) == 691 and len(vision) == 356,
                "published tensor census mismatch")
        require(model._tied_weights_keys == {"lm_head.weight": "model.decoder.embed_tokens.weight"},
                "unexpected head tie")
        require("encoder.language_model.embed_tokens.weight" in model.model._tied_weights_keys,
                "missing encoder embedding tie")
        return text, vision

    def read(self, name, dtype, device, once):
        require(name in self.entries, f"unexpected checkpoint key {name}")
        if once:
            require(name not in self.consumed, f"checkpoint key bound twice {name}")
        path, offset, length, shape = self.entries[name]
        with path.open("rb", buffering=0) as source:
            source.seek(offset)
            data = bytearray(length)
            view = memoryview(data)
            read = 0
            while read < length:
                count = source.readinto(view[read:])
                require(count, f"short published tensor read {name}")
                read += count
        value = torch.frombuffer(data, dtype=torch.bfloat16).reshape(shape).to(device=device, dtype=dtype)
        if name not in self.consumed:
            require(bool(torch.isfinite(value).all()), f"nonfinite published tensor {name}")
        self.consumed.add(name)
        self.read_counts[name] = self.read_counts.get(name, 0) + 1
        return value

    @staticmethod
    def assign(module, name, value):
        parent, _, leaf = name.rpartition(".")
        owner = module.get_submodule(parent) if parent else module
        if leaf in owner._parameters:
            setattr(owner, leaf, torch.nn.Parameter(value, requires_grad=False))
        else:
            setattr(owner, leaf, value)

    def bind(self, module, prefix, dtype, device, once, encoder_layer=False):
        for name in list(module.state_dict()):
            source = name
            source_prefix = prefix
            if encoder_layer and not name.endswith("layer_scalar"):
                source_prefix = prefix.replace("model.encoder.language_model.layers.",
                                               "model.decoder.layers.")
            value = self.read(source_prefix + source, dtype, device, once)
            self.assign(module, name, value)
        require(all(not value.is_meta for value in module.state_dict().values()),
                f"unbound executed module {prefix}")


class StreamedDefinition:
    def __init__(self, model_dir, dtype, device, hooks):
        raw = json.loads((model_dir / "config.json").read_text())
        require(raw["architectures"] == ["DiffusionGemmaForBlockDiffusion"], "reference class mismatch")
        require(raw["canvas_length"] == 256 and raw["text_config"]["num_hidden_layers"] == 30,
                "reference geometry mismatch")
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        config._attn_implementation = "eager"
        config.text_config._attn_implementation = "eager"
        with torch.device("meta"):
            model = definition.DiffusionGemmaForBlockDiffusion(config)
        self.weights = PublishedWeights(model_dir)
        self.text_keys, self.vision_keys = self.weights.inventory(model)
        self.model = model.eval().requires_grad_(False)
        self.model.generation_config = DiffusionGemmaGenerationConfig.from_pretrained(
            model_dir, local_files_only=True
        )
        self.dtype = dtype
        self.device = device
        self.hooks = []
        if device.type == "meta":
            return
        decoder = model.model.decoder
        encoder = model.model.encoder.language_model
        self.weights.bind(decoder.embed_tokens, "model.decoder.embed_tokens.", dtype, device, hooks)
        encoder.embed_tokens.weight = decoder.embed_tokens.weight
        model.lm_head.weight = decoder.embed_tokens.weight
        for embedding in (decoder.embed_tokens, encoder.embed_tokens):
            embedding.embed_scale = torch.tensor(config.text_config.hidden_size ** 0.5, device=device)
        self.weights.bind(decoder.norm, "model.decoder.norm.", dtype, device, hooks)
        encoder.norm.weight = decoder.norm.weight
        self.weights.bind(decoder.self_conditioning, "model.decoder.self_conditioning.", dtype, device, hooks)
        decoder.rotary_emb = definition.DiffusionGemmaTextRotaryEmbedding(config.text_config).to(device)
        encoder.rotary_emb = definition.DiffusionGemmaTextRotaryEmbedding(config.text_config).to(device)
        if hooks:
            for tower, layers, prefix in (
                ("decoder", decoder.layers, "model.decoder.layers."),
                ("encoder", encoder.layers, "model.encoder.language_model.layers."),
            ):
                for index, layer in enumerate(layers):
                    layer_prefix = f"{prefix}{index}."

                    def enter(module, inputs, prefix=layer_prefix, tower=tower):
                        self.weights.bind(module, prefix, dtype, device, False, tower == "encoder")

                    def leave(module, inputs, output):
                        module.to_empty(device="meta")

                    self.hooks.append(layer.register_forward_pre_hook(enter))
                    self.hooks.append(layer.register_forward_hook(leave))

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class RecordingSampler(EntropyBoundSampler):
    def __init__(self, config, canvas_length, vocab_size, steps, recorder):
        super().__init__(config, canvas_length, vocab_size, steps)
        self.recorder = recorder

    def initialize_canvas(self, batch_size, device):
        value = super().initialize_canvas(batch_size, device)
        self.recorder.random_canvas(value)
        return value

    def accept_canvas(self, current_canvas, denoiser_canvas, logits, cur_step):
        value = super().accept_canvas(current_canvas, denoiser_canvas, logits, cur_step)
        entropy = torch.distributions.Categorical(logits=logits).entropy()
        sorted_entropy, _ = torch.sort(entropy, dim=-1)
        cumulative = torch.cumsum(sorted_entropy, dim=-1)
        self.recorder.accept(cur_step, denoiser_canvas, entropy, self.accepted_token_mask,
                             self.entropy_bound - (cumulative - sorted_entropy))
        return value


class Recorder:
    def __init__(self, generation_config):
        self.config = generation_config
        self.active = None
        self.steps = {}
        self.first_full = {}
        self.latest_full = None

    def entry(self, step):
        return self.steps.setdefault(int(step), {})

    def random_canvas(self, value):
        if self.active is None:
            self.initial_canvas = tensor(value[0])
        else:
            self.entry(self.active)["renoise"] = tensor(value[0])

    def accept(self, step, samples, entropy, mask, slack):
        self.entry(step).update({
            "multinomial": tensor(samples[0]), "entropy": tensor(entropy[0]),
            "accept": tensor(mask[0]), "slack": tensor(slack[0]),
        })

    def router(self, layer, weights, ids):
        self.entry(self.active).update({
            f"router{layer:02d}_weights": tensor(weights), f"router{layer:02d}_ids": tensor(ids),
        })

    def capture(self, kwargs, logits):
        step = self.active
        self.entry(step)["canvas"] = tensor(kwargs["decoder_input_ids"][0])
        full = {
            "logits": tensor(logits[0].float()),
            "self_conditioning": tensor(kwargs["self_conditioning_logits"][0])
            if kwargs["self_conditioning_logits"] is not None else None,
        }
        self.entry(step)["top2"] = tensor(torch.topk(logits[0].float(), 2, dim=-1).values)
        if step in (48, 47):
            self.first_full[step] = full
        self.latest_full = (step, full)

    def finish(self, step, argmax, stop):
        self.entry(step).update({"argmax": tensor(argmax[0]), "stop": tensor(stop[0:1])})

    def tensors(self):
        result = {}
        for step, values in self.steps.items():
            prefix = f"step{step}_"
            for source, suffix in (
                ("canvas", "canvas_ids"), ("next", "next_canvas_ids"),
                ("multinomial", "multinomial_ids"), ("renoise", "renoise_ids"),
                ("entropy", "entropy_bf16_f32"), ("accept", "accept_mask"),
                ("slack", "accept_slack_bf16_f32"), ("argmax", "argmax_ids"),
                ("stop", "stop"), ("top2", "top2_bf16_f32"),
            ):
                result[prefix + suffix] = values[source]
            result[prefix + "temperature_f32"] = torch.tensor([
                self.config.t_min + (self.config.t_max - self.config.t_min) * step / self.config.max_denoising_steps
            ])
            for layer in (0, 5):
                result[f"step{step}_layer{layer:02d}_router_ids"] = values[f"router{layer:02d}_ids"]
                result[f"step{step}_layer{layer:02d}_router_weights_f32"] = values[f"router{layer:02d}_weights"]
        full = dict(self.first_full)
        if self.latest_full is not None:
            full[self.latest_full[0]] = self.latest_full[1]
        for step, values in full.items():
            result[f"step{step}_logits_bf16_f32"] = values["logits"]
            if values["self_conditioning"] is not None:
                result[f"step{step}_self_conditioning_bf16"] = values["self_conditioning"]
        return result

    def selected(self):
        steps = set(self.first_full)
        if self.latest_full is not None:
            steps.add(self.latest_full[0])
        return sorted(steps, reverse=True)


def generation_setup(model):
    config, model_kwargs = model._prepare_generation_config(None)
    require(config.max_new_tokens == 256 and config.max_denoising_steps == 48,
            "generation defaults mismatch")
    require(config.t_min == 0.4 and config.t_max == 0.8 and config.stability_threshold == 1,
            "generation schedule mismatch")
    require(config.confidence_threshold == 0.005 and config.sampler_config.entropy_bound == 0.1,
            "generation sampler mismatch")
    require(config.eos_token_id == [1, 106, 50] and config.pad_token_id == 0,
            "generation token config mismatch")
    return config, model_kwargs


def record_router_hooks(model, recorder):
    handles = []
    for index in (0, 5):
        def tap(module, inputs, output, index=index):
            recorder.router(index, output[1], output[2])
        handles.append(model.model.decoder.layers[index].router.register_forward_hook(tap))
    return handles


def denoise_one(model, config, model_kwargs, sampler, stopping, processor, recorder, cache, input_ids,
                decoder_positions, mask_mapping, current, self_conditioning, argmax, finished, step):
    recorder.active = step

    def forward(**kwargs):
        output = model.forward(**kwargs)
        recorder.capture(kwargs, output.logits)
        return output

    current, argmax, self_conditioning, finished = model._denoising_step(
        decoder_forward=forward, current_canvas=current, argmax_canvas=argmax, input_ids=input_ids,
        decoder_position_ids=decoder_positions, self_conditioning_logits=self_conditioning,
        mask_mapping=mask_mapping, past_key_values=cache, finished_denoising=finished, cur_step=step,
        sampler=sampler, logits_processor=processor, diffusion_stopping_criteria=stopping, **model_kwargs,
    )
    recorder.entry(step)["next"] = tensor(current[0])
    recorder.finish(step, argmax, finished)
    return current, self_conditioning, argmax, finished


def record_initial(stream, ids, seed, maximum_steps):
    model = stream.model
    device = stream.device
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config, model_kwargs = generation_setup(model)
    ids = torch.tensor([ids], device=device)
    batch, prefix = ids.shape
    cache = model._prepare_cache_for_generation(config, batch, prefix + 256)
    encoder_positions = torch.arange(prefix, dtype=torch.int32, device=device).unsqueeze(0)
    decoder_positions = torch.arange(prefix, prefix + 256, dtype=torch.int32, device=device).unsqueeze(0)
    attention = torch.ones((batch, prefix), dtype=torch.bool, device=device)
    decoder_attention = torch.nn.functional.pad(attention, (0, 256), value=True)
    unprocessed, encoder_mask = model._prepare_encoder_inputs(
        input_ids=ids, attention_mask=attention, encoder_position_ids=encoder_positions,
        past_key_values=cache, is_prefill=True, canvas_length=256, batch_size=batch, **model_kwargs,
    )
    taps = {}
    handles = []
    for index in TAPS:
        def tap(module, inputs, output, index=index):
            taps[f"layer_{index:02d}_hidden_bf16"] = tensor(output[0])
        handles.append(model.model.encoder.language_model.layers[index].register_forward_hook(tap))
    output = model.model.encoder(input_ids=unprocessed, attention_mask=encoder_mask, past_key_values=cache,
                                 position_ids=encoder_positions, **model_kwargs)
    for handle in handles:
        handle.remove()
    cache = output.past_key_values
    recorder = Recorder(config)
    router_handles = record_router_hooks(model, recorder)
    sampler = RecordingSampler(config.sampler_config, 256, model.config.text_config.vocab_size, 48, recorder)
    processor = model._prepare_logits_processor(config, None)
    stopping = model._prepare_diffusion_stopping_criteria(config)
    ar_stopping = model._prepare_ar_stopping_criteria(config, None)
    current, conditioning, mask, finished = model._prepare_denoiser_inputs(
        decoder_attention_mask=decoder_attention, past_key_values=cache, sampler=sampler,
        diffusion_stopping_criteria=stopping, batch_size=batch, device=device, model_kwargs=model_kwargs,
    )
    argmax = current
    steps = list(reversed(range(1, 49)))[:maximum_steps]
    for step in steps:
        current, conditioning, argmax, finished = denoise_one(
            model, config, model_kwargs, sampler, stopping, processor, recorder, cache, ids,
            decoder_positions, mask, current, conditioning, argmax, finished, step,
        )
        if bool(torch.all(finished)):
            break
    for handle in router_handles:
        handle.remove()
    input_ids = torch.cat([ids, argmax], dim=-1)
    eos = torch.tensor(config.eos_token_id, device=device)
    input_ids, finished_sequences = model._finalize_canvas(
        input_ids=input_ids, finished_sequences=torch.zeros(batch, dtype=torch.bool, device=device),
        generation_config=config, stopping_criteria=ar_stopping, canvas_length=256, eos_tensor=eos,
    )
    tensors = recorder.tensors()
    tensors.update(taps)
    tensors["prompt_ids"] = tensor(ids[0])
    tensors["position_ids"] = tensor(encoder_positions[0].long())
    tensors.update(cache_extents(cache))
    tensors["commit_ids"] = tensor(input_ids[0, -256:])
    next_len, next_decoder_attention, next_attention, next_encoder_positions, _ = (
        model._prepare_kwargs_for_next_canvas(
            attention_mask=attention, decoder_attention_mask=decoder_attention,
            decoder_position_ids=decoder_positions, canvas_length=256, cur_len=prefix,
        )
    )
    continuation = {
        "cache": cache, "input_ids": input_ids, "attention": next_attention,
        "decoder_attention": next_decoder_attention, "encoder_positions": next_encoder_positions,
        "cur_len": next_len, "config": config, "model_kwargs": model_kwargs,
    }
    meta = {"seed": seed, "prefix": prefix, "executed_steps": [int(step) for step in recorder.steps],
            "selected_steps": recorder.selected(), "finished_sequence": bool(finished_sequences[0])}
    return tensors, meta, continuation


def record_commit(stream, prior, seed):
    model = stream.model
    device = stream.device
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = prior["config"]
    model_kwargs = dict(prior["model_kwargs"])
    cache = prior["cache"]
    before = cache_extents(cache)
    ids = prior["input_ids"]
    unprocessed, encoder_mask = model._prepare_encoder_inputs(
        input_ids=ids, attention_mask=prior["attention"], encoder_position_ids=prior["encoder_positions"],
        past_key_values=cache, is_prefill=False, canvas_length=256, batch_size=1, **model_kwargs,
    )
    output = model.model.encoder(input_ids=unprocessed, attention_mask=encoder_mask, past_key_values=cache,
                                 position_ids=prior["encoder_positions"], **model_kwargs)
    cache = output.past_key_values
    cur_len = prior["cur_len"]
    require(prior["attention"].shape == (1, cur_len), "commit attention extent mismatch")
    decoder_attention = prior["decoder_attention"]
    require(decoder_attention.shape == (1, cur_len + 256), "next-canvas attention extent mismatch")
    decoder_positions = torch.arange(cur_len, cur_len + 256, dtype=torch.int32, device=device).unsqueeze(0)
    recorder = Recorder(config)
    router_handles = record_router_hooks(model, recorder)
    sampler = RecordingSampler(config.sampler_config, 256, model.config.text_config.vocab_size, 48, recorder)
    processor = model._prepare_logits_processor(config, None)
    stopping = model._prepare_diffusion_stopping_criteria(config)
    current, conditioning, mask, finished = model._prepare_denoiser_inputs(
        decoder_attention_mask=decoder_attention, past_key_values=cache, sampler=sampler,
        diffusion_stopping_criteria=stopping, batch_size=1, device=device, model_kwargs=model_kwargs,
    )
    current, conditioning, argmax, finished = denoise_one(
        model, config, model_kwargs, sampler, stopping, processor, recorder, cache, ids,
        decoder_positions, mask, current, conditioning, current, finished, 48,
    )
    for handle in router_handles:
        handle.remove()
    tensors = recorder.tensors()
    tensors.update({f"before_{key}": value for key, value in before.items()})
    tensors.update(cache_extents(cache))
    tensors["prompt_ids"] = tensor(ids[0, :-256])
    tensors["commit_ids"] = tensor(ids[0, -256:])
    meta = {"seed": seed, "prefix": cur_len, "executed_steps": [48], "selected_steps": [48],
            "finished_sequence": False, "teacher_forced_continuation": prior["finished_sequence"]}
    return tensors, meta


def prompt_ids(tokenizer, length):
    paragraphs = [
        "The small brown bird rested on a smooth stone beside the river while morning light crossed the water.",
        "During the winter workshop, students designed a careful map of the old railway and compared their notes.",
        "A patient engineer measured the bridge, recorded each result, and explained the observations to the group.",
    ]
    values = tokenizer(paragraphs[0]).input_ids
    index = 1
    while len(values) < length:
        values.extend(tokenizer(paragraphs[index % len(paragraphs)], add_special_tokens=False).input_ids)
        index += 1
    return values[:length]


def common_meta(stream, role):
    dtype_name = {torch.bfloat16: "bf16", torch.float32: "fp32"}[stream.dtype]
    return {
        "role": role, "class": stream.model.__class__.__name__, "torch": torch.__version__,
        "transformers": transformers.__version__, "device": torch.cuda.get_device_name(stream.device),
        "capability": list(torch.cuda.get_device_capability(stream.device)),
        "dtype": dtype_name, "checkpoint_tensors": 1047,
        "text_tensors": 691, "vision_excluded": 356, "unexpected": 0, "newly_initialized_executed": 0,
        "unbound_executed": 0, "full_class_load": False,
        "bf16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        "max_memory_allocated": torch.cuda.max_memory_allocated(stream.device),
    }


def run_record(args):
    device = torch.device("cuda", 0)
    stream = StreamedDefinition(args.model, torch.bfloat16, device, True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids300 = prompt_ids(tokenizer, 300)
    ids1500 = prompt_ids(tokenizer, 1500)
    require(ids300 == ids1500[:300], "prompt prefix mismatch")
    if args.case in ("all", "p300"):
        p300, meta300, prior = record_initial(stream, ids300, 47, 48)
        prior["finished_sequence"] = meta300["finished_sequence"]
        commit, commit_meta = record_commit(stream, prior, 71)
        common = common_meta(stream, "record_bf16")
        for name, tensors, meta in (
            ("encoder_p300", {key: value for key, value in p300.items() if not key.startswith("step")
                              and key != "commit_ids"}, meta300),
            ("canvas_p300", {key: value for key, value in p300.items() if key.startswith("step") or key in
                             ("prompt_ids", "commit_ids")}, meta300),
            ("commit256_p300", commit, commit_meta),
        ):
            save_case(args.out, name, tensors, common | meta)
        (args.out / "RECORD-p300.json").write_text(json.dumps(common, indent=2) + "\n")
        print(f"PASS diffusion_record_case case=p300 steps={len(meta300['executed_steps'])} "
              f"fixtures=3 max_memory_allocated={common['max_memory_allocated']}", flush=True)
        del p300, commit, prior
        gc.collect()
    if args.case in ("all", "p1500"):
        p1500, meta1500, _ = record_initial(stream, ids1500, 71, 2)
        common = common_meta(stream, "record_bf16")
        for name, tensors, meta in (
            ("encoder_p1500", {key: value for key, value in p1500.items() if not key.startswith("step")
                               and key != "commit_ids"}, meta1500),
            ("canvas_p1500", {key: value for key, value in p1500.items() if key.startswith("step") or
                              key == "prompt_ids"}, meta1500),
        ):
            save_case(args.out, name, tensors, common | meta)
        (args.out / "RECORD-p1500.json").write_text(json.dumps(common, indent=2) + "\n")
        print(f"PASS diffusion_record_case case=p1500 steps={len(meta1500['executed_steps'])} "
              f"fixtures=2 max_memory_allocated={common['max_memory_allocated']}", flush=True)
    common = common_meta(stream, "record_bf16")
    require(common["max_memory_allocated"] < 8 * 1024 ** 3, "record GPU memory bound exceeded")
    require(stream.weights.consumed == stream.text_keys, "record text key-consumption mismatch")
    if all((args.out / f"{name}.safetensors").exists() for name in CASE_NAMES):
        require(not (args.out / "LOAD.json").exists(), "record LOAD output exists")
        (args.out / "LOAD.json").write_text(json.dumps(common, indent=2) + "\n")
        print(f"PASS diffusion_record cases=5 text_keys=691 unexpected=0 "
              f"max_memory_allocated={common['max_memory_allocated']}", flush=True)


class ReplaySampler(EntropyBoundSampler):
    def __init__(self, config, canvas, vocab, steps, renoise):
        super().__init__(config, canvas, vocab, steps)
        self.recorded_renoise = renoise

    def initialize_canvas(self, batch_size, device):
        return self.recorded_renoise.to(device).unsqueeze(0)


def sampler_products(model, config, raw_logits, current, recorded, step, previous_argmax):
    processor = model._prepare_logits_processor(config, None)
    processed = processor(current, raw_logits, cur_step=torch.tensor(step, device=raw_logits.device, dtype=torch.int32))
    sampler = ReplaySampler(config.sampler_config, 256, model.config.text_config.vocab_size, 48,
                            recorded[f"step{step}_renoise_ids"])
    accepted = sampler.accept_canvas(
        current, recorded[f"step{step}_multinomial_ids"].to(raw_logits.device).unsqueeze(0), processed, step
    )
    next_canvas = sampler.renoise_canvas(accepted, step)
    stopping = model._prepare_diffusion_stopping_criteria(config)
    if previous_argmax is not None:
        stopping.argmax_canvas_history = previous_argmax.to(raw_logits.device).reshape(1, 1, 256).clone()
    argmax = torch.argmax(processed, dim=-1)
    stop = stopping(argmax, processed)
    entropy = torch.distributions.Categorical(logits=processed).entropy()
    sorted_entropy, _ = torch.sort(entropy, dim=-1)
    slack = config.sampler_config.entropy_bound - (torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy)
    return {
        "argmax": tensor(argmax[0]), "stop": tensor(stop[0:1]), "next": tensor(next_canvas[0]),
        "entropy": tensor(entropy[0]), "accept": tensor(sampler.accepted_token_mask[0]),
        "slack": tensor(slack[0]), "top2": tensor(torch.topk(raw_logits[0].float(), 2, dim=-1).values),
    }


def replay_products(tensors, meta, step, products, oracle):
    result = dict(tensors)
    names = {
        "logits": f"step{step}_logits_oracle_f32" if oracle else f"step{step}_logits_bf16_f32",
        "argmax": f"step{step}_oracle_argmax_ids" if oracle else f"step{step}_argmax_ids",
        "stop": f"step{step}_oracle_stop" if oracle else f"step{step}_stop",
        "entropy": f"step{step}_entropy_oracle_f32" if oracle else f"step{step}_entropy_bf16_f32",
        "accept": f"step{step}_oracle_accept_mask" if oracle else f"step{step}_accept_mask",
        "slack": f"step{step}_accept_slack_oracle_f32" if oracle else f"step{step}_accept_slack_bf16_f32",
        "top2": f"step{step}_oracle_top2_f32" if oracle else f"step{step}_top2_bf16_f32",
    }
    result[names["logits"]] = products.pop("logits")
    for source in ("argmax", "stop", "entropy", "accept", "slack", "top2"):
        result[names[source]] = products[source]
    if not oracle:
        result[f"step{step}_next_canvas_ids"] = products["next"]
    return result


def router_products(call, oracle):
    result = {}
    for layer in (0, 5):
        weights, ids = call[f"router{layer:02d}"]
        if oracle:
            result[f"step{call['step']}_layer{layer:02d}_router_oracle_ids"] = ids
            result[f"step{call['step']}_layer{layer:02d}_router_oracle_weights_f32"] = weights
        else:
            result[f"step{call['step']}_layer{layer:02d}_router_ids"] = ids
            result[f"step{call['step']}_layer{layer:02d}_router_weights_f32"] = weights
    return result


def bf16_name(key):
    replacements = {
        "logits_oracle_f32": "logits_bf16_f32", "entropy_oracle_f32": "entropy_bf16_f32",
        "accept_slack_oracle_f32": "accept_slack_bf16_f32", "oracle_accept_mask": "accept_mask",
        "oracle_argmax_ids": "argmax_ids", "oracle_stop": "stop",
        "oracle_top2_f32": "top2_bf16_f32", "hidden_oracle_f32": "hidden_bf16",
        "router_oracle_ids": "router_ids", "router_oracle_weights_f32": "router_weights_f32",
    }
    for source, target in replacements.items():
        if source in key:
            return key.replace(source, target)
    return key


def oracle_name(key):
    replacements = {
        "logits_bf16_f32": "logits_oracle_f32", "entropy_bf16_f32": "entropy_oracle_f32",
        "accept_slack_bf16_f32": "accept_slack_oracle_f32", "accept_mask": "oracle_accept_mask",
        "argmax_ids": "oracle_argmax_ids", "stop": "oracle_stop",
        "top2_bf16_f32": "oracle_top2_f32", "hidden_bf16": "hidden_oracle_f32",
        "router_ids": "router_oracle_ids", "router_weights_f32": "router_oracle_weights_f32",
    }
    for source, target in replacements.items():
        if source in key:
            return key.replace(source, target)
    return key


def softcap(model, hidden, split):
    decoder = model.model.decoder
    if split:
        hidden = torch.cat([decoder.norm(hidden[:, :128]), decoder.norm(hidden[:, 128:])], dim=1)
        logits = torch.cat([model.lm_head(hidden[:, :128]), model.lm_head(hidden[:, 128:])], dim=1)
    else:
        hidden = decoder.norm(hidden)
        logits = model.lm_head(hidden)
    logits = logits.to(torch.float32)
    return torch.tanh(logits / model.final_logit_softcapping) * model.final_logit_softcapping


def run_replay(args):
    oracle = args.dtype == "fp32"
    dtype = torch.float32 if oracle else torch.bfloat16
    device = torch.device("cuda", 0)
    case_names = CASE_NAMES if args.case == "all" else (
        ("encoder_p300", "canvas_p300", "commit256_p300") if args.case == "p300"
        else ("encoder_p1500", "canvas_p1500")
    )
    stream = StreamedDefinition(args.model, dtype, device, False)
    model = stream.model
    weights = stream.weights
    config, _ = generation_setup(model)
    records = {name: load_case(args.record, name) for name in case_names}
    selected = {name: records[name][1]["selected_steps"] for name in
                ("canvas_p300", "canvas_p1500", "commit256_p300") if name in records}
    prompts = {}
    if "canvas_p300" in records:
        prompts[300] = records["canvas_p300"][0]["prompt_ids"]
    if "canvas_p1500" in records:
        prompts[1500] = records["canvas_p1500"][0]["prompt_ids"]
    commit_ids = records["commit256_p300"][0]["commit_ids"] if "commit256_p300" in records else None
    chains = {}
    chain_specs = []
    if "canvas_p300" in records:
        chain_specs.extend((("p300", 300, [prompts[300]]), ("commit", 300, [prompts[300], commit_ids]),
                            ("commit_split", 300, [prompts[300], commit_ids[:128], commit_ids[128:]])))
    if "canvas_p1500" in records:
        chain_specs.append(("p1500", 1500, [prompts[1500]]))
    for name, prompt_length, chunks in chain_specs:
        cache = DynamicCache(config=model.config.text_config)
        mask_cache = DynamicCache(config=model.config.text_config)
        calls = []
        start = 0
        for chunk_index, ids in enumerate(chunks):
            ids = ids.to(device)
            hidden = model.model.encoder.language_model.embed_tokens(ids.unsqueeze(0))
            positions = torch.arange(start, start + len(ids), dtype=torch.int32, device=device).unsqueeze(0)
            attention = torch.ones((1, start + len(ids)), dtype=torch.bool, device=device)
            mask = model.model.encoder.create_masks_for_generate(
                config=model.config, inputs_embeds=hidden, attention_mask=attention,
                past_key_values=mask_cache, position_ids=positions, mm_token_type_ids=None,
            )
            calls.append({"ids": ids, "hidden": hidden, "positions": positions, "start": start,
                          "mask": mask, "tap": chunk_index == 0 and name in ("p300", "p1500")})
            dummy = torch.zeros((1, 1, len(ids), 1), dtype=dtype, device=device)
            for layer_index in range(30):
                mask_cache.update(dummy, dummy, layer_index)
            start += len(ids)
        chains[name] = {"cache": cache, "calls": calls, "prefix": start}
    decoder_calls = []
    for case_name, chain_name in (("canvas_p300", "p300"), ("canvas_p1500", "p1500"),
                                  ("commit256_p300", "commit")):
        if case_name not in records:
            continue
        source = records[case_name][0]
        chain = chains[chain_name]
        for step in selected[case_name]:
            ids = source[f"step{step}_canvas_ids"].to(device).unsqueeze(0)
            conditioning_name = f"step{step}_self_conditioning_bf16"
            conditioning = source[conditioning_name].to(device).unsqueeze(0) if conditioning_name in source else None
            inputs = model.model.decoder.embed_tokens(ids)
            if conditioning is not None:
                soft = torch.matmul(conditioning.softmax(dim=-1, dtype=torch.float32).to(dtype),
                                    model.model.decoder.embed_tokens.weight)
                soft = soft * model.model.decoder.embed_tokens.embed_scale.to(inputs.dtype)
            else:
                soft = torch.zeros_like(inputs)
            hidden = model.model.decoder.self_conditioning(inputs, soft)
            positions = torch.arange(chain["prefix"], chain["prefix"] + 256, dtype=torch.int32,
                                     device=device).unsqueeze(0)
            previous = source[f"step{step + 1}_argmax_ids"] if step < 48 else None
            call = {"case": case_name, "step": step, "chain": chain_name, "ids": ids,
                    "hidden": hidden, "positions": positions, "previous": previous, "append_variant": False}
            decoder_calls.append(call)
            if case_name == "commit256_p300":
                decoder_calls.append(call | {"chain": "commit_split", "append_variant": True})
    outputs = {name: {} for name in case_names}
    noise = {name: {"direct_record": {}, "split_direct": {}, "direct_oracle": {}, "split_oracle": {},
                    "append_direct": {}, "append_oracle": {}} for name in case_names}
    active_decoder = {}
    replay_router_handles = []
    for router_index in (0, 5):
        def router_tap(module, inputs, output, index=router_index):
            active_decoder["call"][f"router{index:02d}"] = (tensor(output[1]), tensor(output[2]))
        replay_router_handles.append(
            model.model.decoder.layers[router_index].router.register_forward_hook(router_tap)
        )
    for index in range(30):
        prefix = f"model.decoder.layers.{index}."
        decoder_layer = model.model.decoder.layers[index]
        encoder_layer = model.model.encoder.language_model.layers[index]
        weights.bind(decoder_layer, prefix, dtype, device, True)
        for name in list(encoder_layer.state_dict()):
            if name == "layer_scalar":
                value = weights.read(f"model.encoder.language_model.layers.{index}.layer_scalar",
                                     dtype, device, True)
            else:
                owner_name = name
                parent, _, leaf = owner_name.rpartition(".")
                decoder_owner = decoder_layer.get_submodule(parent) if parent else decoder_layer
                value = getattr(decoder_owner, leaf)
            weights.assign(encoder_layer, name, value)
        require(all(not value.is_meta for value in encoder_layer.state_dict().values()),
                f"unbound encoder layer {index}")
        layer_type = model.config.text_config.layer_types[index]
        for chain in chains.values():
            for call in chain["calls"]:
                hidden = call["hidden"]
                positions = model.model.encoder.language_model.rotary_emb(hidden, call["positions"], layer_type)
                hidden = encoder_layer(hidden, position_embeddings=positions,
                                       attention_mask=call["mask"][layer_type], position_ids=call["positions"],
                                       past_key_values=chain["cache"])
                call["hidden"] = hidden
                if call["tap"] and index in TAPS:
                    case = "encoder_p300" if "p300" in chains and chain is chains["p300"] else "encoder_p1500"
                    key = f"layer_{index:02d}_hidden_{'oracle_f32' if oracle else 'bf16'}"
                    outputs[case][key] = tensor(hidden[0])
        for call in decoder_calls:
            chain = chains[call["chain"]]
            active_decoder["call"] = call
            positions = model.model.decoder.rotary_emb(call["hidden"], call["positions"], layer_type)
            call["hidden"] = decoder_layer(
                call["hidden"], position_embeddings=positions, attention_mask=None,
                position_ids=call["positions"], past_key_values=chain["cache"],
            )
        decoder_layer.to_empty(device="meta")
        encoder_layer.to_empty(device="meta")
        gc.collect()
    for handle in replay_router_handles:
        handle.remove()
    for chain in chains.values():
        model.model.encoder.language_model.norm(chain["calls"][-1]["hidden"])
    for call in decoder_calls:
        case = call["case"]
        step = call["step"]
        if call["append_variant"]:
            raw = softcap(model, call["hidden"], False)
            products = sampler_products(model, config, raw, call["ids"], records[case][0], step,
                                        call["previous"])
            products["logits"] = tensor(raw[0])
            values = replay_products({}, records[case][1], step, products, oracle)
            values.update(router_products(call, oracle))
            for key, value in values.items():
                canonical = bf16_name(key)
                if key in outputs[case]:
                    noise[case]["append_direct"][canonical] = distance(value, outputs[case][key])
                other = optional_tensor(args.oracle, case, oracle_name(canonical))
                if other is not None:
                    noise[case]["append_oracle"][canonical] = distance(value, other)
            del values, products, raw
            continue
        raw = softcap(model, call["hidden"], False)
        split_raw = softcap(model, call["hidden"], True)
        products = sampler_products(model, config, raw, call["ids"], records[case][0], step, call["previous"])
        split_products = sampler_products(model, config, split_raw, call["ids"], records[case][0], step,
                                          call["previous"])
        products["logits"] = tensor(raw[0])
        split_products["logits"] = tensor(split_raw[0])
        direct = replay_products(outputs[case], records[case][1], step, products, oracle)
        direct.update(router_products(call, oracle))
        outputs[case].update(direct)
        split_values = replay_products({}, records[case][1], step, split_products, oracle)
        for key, value in split_values.items():
            canonical = bf16_name(key)
            if key in direct:
                noise[case]["split_direct"][canonical] = distance(value, direct[key])
            other = optional_tensor(args.oracle, case, oracle_name(canonical))
            if other is not None:
                noise[case]["split_oracle"][canonical] = distance(value, other)
        del split_values, split_products, split_raw
    for name, chain_name in (("encoder_p300", "p300"), ("encoder_p1500", "p1500"),
                             ("commit256_p300", "commit")):
        if name not in records:
            continue
        for key, value in cache_extents(chains[chain_name]["cache"]).items():
            outputs[name][key] = value
    if "commit256_p300" in records:
        for key, value in cache_extents(chains["commit_split"]["cache"]).items():
            require(torch.equal(value, outputs["commit256_p300"][key]), f"commit shape changed {key}")
    for name in case_names:
        source = records[name][0]
        for key, value in outputs[name].items():
            canonical = bf16_name(key)
            if canonical in source:
                noise[name]["direct_record"][canonical] = distance(value, source[canonical])
            other = optional_tensor(args.oracle, name, oracle_name(canonical))
            if other is not None:
                noise[name]["direct_oracle"][canonical] = distance(value, other)
    if not oracle and args.bf16_reduction == "default" and torch.cuda.get_device_capability(device) == (8, 9):
        for case, groups in noise.items():
            for key, values in groups["direct_record"].items():
                require(all(value == 0 for value in values.values()),
                        f"BF16 record/replay mismatch {case}/{key}: {values}")
    require(weights.consumed == stream.text_keys, "replay text key-consumption mismatch")
    require(all(weights.read_counts[name] == 1 for name in stream.text_keys),
            "replay did not bind every text key exactly once")
    common = common_meta(stream, "fp32_oracle" if oracle else f"bf16_{args.bf16_reduction}_replay")
    limit = 14 * 1024 ** 3 if oracle else 8 * 1024 ** 3
    require(common["max_memory_allocated"] < limit, "streamed reference GPU memory bound exceeded")
    for name in case_names:
        tensors = outputs[name]
        for key in ("prompt_ids", "position_ids", "commit_ids"):
            if key in records[name][0] and (name.startswith("encoder") or name.startswith("commit")):
                tensors[key] = records[name][0][key]
        save_case(args.out, name, tensors, records[name][1] | common)
    noise_part = args.out / f"NOISE-{args.dtype}-{args.case}.json"
    require(not noise_part.exists(), "replay noise part exists")
    noise_part.write_text(json.dumps(noise, indent=2, allow_nan=False) + "\n")
    print(f"PASS diffusion_replay_case case={args.case} dtype={args.dtype} "
          f"selected_steps={sum(map(len, selected.values()))} text_keys=691 "
          f"max_memory_allocated={common['max_memory_allocated']}", flush=True)
    if all((args.out / f"{name}.safetensors").exists() for name in CASE_NAMES):
        combined = {}
        for path in sorted(args.out.glob(f"NOISE-{args.dtype}-*.json")):
            combined.update(json.loads(path.read_text()))
        require(set(combined) == set(CASE_NAMES), "replay noise cases incomplete")
        require(not (args.out / "NOISE.json").exists() and not (args.out / "LOAD.json").exists(),
                "replay final reports exist")
        (args.out / "NOISE.json").write_text(json.dumps(combined, indent=2, allow_nan=False) + "\n")
        (args.out / "LOAD.json").write_text(json.dumps(common, indent=2) + "\n")
        print(f"PASS diffusion_replay dtype={args.dtype} cases=5 text_keys=691 "
              f"max_memory_allocated={common['max_memory_allocated']}", flush=True)


def metric_family(key):
    for token in ("logits", "entropy", "accept_slack", "accept_mask", "argmax", "stop", "router", "hidden"):
        if token in key:
            return token
    return "other"


def run_freeze(args):
    canonical = args.canonical
    arms = [Path(value) for value in args.arms]
    require(len(arms) >= 4, "freeze requires local BF16, two local oracle observations, and sm86")
    oracle_dir = next(path for path in arms if json.loads((path / "LOAD.json").read_text())["dtype"] == "fp32")
    limits = {}
    sources = {}
    for arm in arms:
        meta = json.loads((arm / "LOAD.json").read_text())
        noise = json.loads((arm / "NOISE.json").read_text())
        for case, groups in noise.items():
            for group in ("direct_oracle", "split_oracle", "append_oracle"):
                for key, values in groups[group].items():
                    target = limits.setdefault(case, {}).setdefault(key, {metric: 0.0 for metric in values})
                    for metric, value in values.items():
                        target[metric] = max(target.get(metric, 0.0), 2 * value)
                    sources.setdefault(case, {}).setdefault(key, []).append(
                        {"arm": str(arm), "device": meta["device"], "dtype": meta["dtype"],
                         "group": group, "distance": values}
                    )
    arm_metas = [json.loads((path / "LOAD.json").read_text()) for path in arms]
    require(any(meta["capability"] == [8, 6] for meta in arm_metas), "freeze lacks sm86")
    require(any(meta["dtype"] == "bf16" and meta["bf16_reduced_precision_reduction"] for meta in arm_metas)
            and any(meta["dtype"] == "bf16" and not meta["bf16_reduced_precision_reduction"]
                    for meta in arm_metas), "freeze lacks default/strict BF16 coverage")
    frozen_limits = {}
    for case, tensors in limits.items():
        frozen_limits[case] = {}
        trajectory = {}
        for key, values in tensors.items():
            frozen_limits[case][key] = values
            family = metric_family(key)
            aggregate = trajectory.setdefault(family, {})
            for metric, value in values.items():
                aggregate[metric] = max(aggregate.get(metric, 0.0), value)
        for key in list(frozen_limits[case]):
            if "logits" in key:
                frozen_limits[case][key]["margin_cut"] = 2 * frozen_limits[case][key]["max_abs"]
        if "logits" in trajectory:
            trajectory["logits"]["margin_cut"] = 2 * trajectory["logits"]["max_abs"]
        frozen_limits[case]["trajectory"] = trajectory
    common = json.loads((canonical / "LOAD.json").read_text())
    for name in CASE_NAMES:
        tensors, meta = load_case(canonical, name)
        oracle_tensors, _ = load_case(oracle_dir, name)
        for key, value in oracle_tensors.items():
            if "oracle" in key:
                require(key not in tensors, f"duplicate canonical tensor {key}")
                tensors[key] = value
        save_case(args.out, name, tensors, common | meta)
    frozen = {"status": "FROZEN", "policy": "2*max reference-only distance to FP32 oracle",
              "candidate_observations": 0, "limits": frozen_limits, "sources": sources,
              "arms": [str(path) for path in arms]}
    with (args.out / "FROZEN.json").open("x") as destination:
        destination.write(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    print(f"PASS diffusion_freeze cases={len(CASE_NAMES)} arms={len(arms)} candidate_observations=0", flush=True)


def run_dry(args):
    require(not torch.cuda.is_initialized(), "dry run initialized CUDA")
    stream = StreamedDefinition(args.model, torch.bfloat16, torch.device("meta"), False)
    require(len(stream.text_keys) == 691 and len(stream.vision_keys) == 356, "dry census mismatch")
    config, _ = generation_setup(stream.model)
    report = {"class": stream.model.__class__.__name__, "checkpoint_tensors": 1047, "text_tensors": 691,
              "vision_excluded": 356, "unexpected": 0, "newly_initialized": 0,
              "generation_steps": config.max_denoising_steps, "torch": torch.__version__,
              "transformers": transformers.__version__}
    (args.out / "DRY.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS diffusion_reference_dry checkpoint=1047 text=691 vision_excluded=356 unexpected=0", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry", "record", "replay", "freeze"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--case", choices=("all", "p300", "p1500"), default="all")
    parser.add_argument("--bf16-reduction", choices=("default", "strict"), default="default")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--canonical", type=Path)
    parser.add_argument("--arms", type=Path, nargs="+")
    args = parser.parse_args()
    require(not args.out.resolve().is_relative_to(args.model.resolve()), "output inside read-only weights")
    require(transformers.__version__ == "5.21.0", "reference definition version mismatch")
    require(torch.__version__ == "2.14.1+cu130", "reference torch version mismatch")
    args.out.mkdir(parents=True, exist_ok=args.mode in ("record", "replay"))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        args.dtype == "bf16" and args.bf16_reduction == "default"
    )
    torch.set_float32_matmul_precision("highest")
    if args.mode == "dry":
        run_dry(args)
    elif args.mode == "record":
        run_record(args)
    elif args.mode == "replay":
        require(args.record is not None, "replay requires --record")
        run_replay(args)
    else:
        require(args.canonical is not None and args.arms, "freeze requires --canonical and --arms")
        run_freeze(args)


if __name__ == "__main__":
    main()
