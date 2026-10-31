"""Gemma text model-definition oracle over published BF16 tensor ranges."""
import argparse
import json
from pathlib import Path
import struct

import torch
import transformers
from safetensors.torch import save_file
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import apply_renamings


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class PublishedWeights:
    def __init__(self, directory, model_type):
        self.entries = {}
        self.consumed = set()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--full-load-only", action="store_true")
    args = parser.parse_args()
    require(not args.out.resolve().is_relative_to(args.model.resolve()), "output inside read-only weights")
    require(transformers.__version__ == "5.21.0", "reference definition version mismatch")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    raw = json.loads((args.model / "config.json").read_text())
    require(raw["architectures"] in (["Gemma4ForConditionalGeneration"],
                                    ["Gemma4UnifiedForConditionalGeneration"]), "reference class mismatch")
    model_class = getattr(transformers, raw["architectures"][0])
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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    config = model_class.config_class.from_pretrained(args.model, local_files_only=True)
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    device = torch.device("cuda", 0)
    weights = PublishedWeights(args.model, raw["model_type"])
    model, hooks = stream_model(model_class, config, weights, dtype, device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    texts = ["The small brown bird rested on a smooth stone beside the river.",
             "During the winter workshop, students designed a careful map of the old railway."]
    short = [tokenizer(text).input_ids for text in texts]
    require(all(8 <= len(ids) <= 32 for ids in short), "short reference input extent")
    seed = short[0] + short[1]
    window = config.text_config.sliding_window
    wrap = (seed * ((window + 8) // len(seed) + 1))[:window + 8]
    (args.out / "ids.json").write_text(json.dumps({"texts": texts, "short": short, "wrap": wrap, "long": wrap}, indent=2) + "\n")
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
        print(f"REFERENCE_CELL {tag} rows={keep} repeat_bits=0 chunk={chunk}", flush=True)
    expected = {name for name in weights.entries if name.startswith("model.language_model.")}
    require(weights.consumed == expected, "streamed text tensor key-consumption mismatch")
    report = {"class": model_class.__name__, "transformers": transformers.__version__,
              "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "dtype": args.dtype,
              "full_class_load": False, "checkpoint_tensors": len(weights.entries),
              "consumed_text_tensors": len(weights.consumed), "unbound_executed": 0,
              "unexpected": 0, "newly_initialized_executed": 0, "tied_head": True}
    save_file(tensors, str(args.out / "reference.safetensors"))
    (args.out / "NOISE.json").write_text(json.dumps(noise, indent=2) + "\n")
    (args.out / "LOAD.json").write_text(json.dumps(report, indent=2) + "\n")
    for hook in hooks:
        hook.remove()
    print(f"PASS streamed_reference cells={len(cases)} text_keys={len(expected)}", flush=True)


if __name__ == "__main__":
    main()
