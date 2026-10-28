"""MiniCPM-V-4.7 published FP8 model-definition reference, text only."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import torch
from accelerate.hooks import remove_hook_from_module
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from safetensors.torch import save_file
import transformers


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_model(directory):
    config = transformers.AutoConfig.from_pretrained(directory, local_files_only=True)
    check(config.architectures == ["MiniCPMV4_7ForConditionalGeneration"], "architecture mismatch")
    text = config.text_config
    check((text.num_hidden_layers, text.hidden_size, text.num_experts,
           text.moe_intermediate_size, text.vocab_size) == (40, 2048, 256, 512, 248144),
          "published model geometry mismatch")
    placement = {"": "cpu"}
    for layer in range(text.num_hidden_layers):
        placement[f"model.language_model.layers.{layer}.mlp.experts"] = "meta"
    placements = []

    def dispatch(model, device_map, **_kwargs):
        check(device_map == placement, "reference placement changed")
        for name, value in model.named_parameters():
            check(value.is_meta == (".mlp.experts." in name), f"unexpected placement {name}")
        model.hf_device_map = device_map
        placements.append(True)
        return model

    architecture = getattr(transformers, config.architectures[0])
    initialized = []
    initialization_calls = []
    initialize = architecture._initialize_missing_keys

    def record_initialization(model, *args, **kwargs):
        initialization_calls.append(True)
        initialized.extend(name for name, value in model.state_dict(keep_vars=True).items()
                           if not getattr(value, "_is_hf_initialized", False))
        check(not initialized, f"newly initialized model keys: {initialized}")
        return initialize(model, *args, **kwargs)

    with patch("transformers.integrations.accelerate.dispatch_model", dispatch), \
            patch.object(architecture, "_initialize_missing_keys", record_initialization):
        model, loading = architecture.from_pretrained(
            directory, dtype=torch.bfloat16, device_map=placement,
            output_loading_info=True, local_files_only=True,
        )
    check(placements == [True], "reference placement call count")
    check(initialization_calls == [True], "reference initialization call count")
    clean = {}
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        check(key in loading and not loading[key], f"unclean model load {key}: {loading.get(key)}")
        clean[key] = []
    clean["newly_initialized_keys"] = initialized
    check(not clean["newly_initialized_keys"], "newly initialized model keys")
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if not value.is_meta:
            set_module_tensor_to_device(model, name, "cuda:0")
    model.set_attn_implementation("sdpa")
    model.config.text_config._experts_implementation = "eager"
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    gate_up = torch.zeros((256, 1024, 2048), dtype=torch.bfloat16, device="cuda")
    down = torch.zeros((256, 2048, 512), dtype=torch.bfloat16, device="cuda")
    calls = [0] * text.num_hidden_layers

    def weight(name, shards):
        codes = shards[index[name + ".weight"]].get_tensor(name + ".weight")
        scale = shards[index[name + ".weight_scale"]].get_tensor(name + ".weight_scale")
        check(codes.dtype == torch.float8_e4m3fn and scale.dtype == torch.bfloat16,
              "published expert format changed")
        check(scale.shape == (codes.shape[0], 1), "published channel scale layout changed")
        return (codes.float() * scale.float()).bfloat16()

    for layer, block in enumerate(model.model.language_model.layers):
        experts = block.mlp.experts
        check(experts.gate_up_proj.is_meta and experts.down_proj.is_meta, "expert residency changed")
        remove_hook_from_module(experts)

        def onload(module, args, layer=layer):
            calls[layer] += 1
            selected = torch.unique(args[1]).cpu().tolist()
            names = {
                index[f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.{suffix}"]
                for expert in selected for projection in ("gate_proj", "up_proj", "down_proj")
                for suffix in ("weight", "weight_scale")
            }
            with ExitStack() as stack:
                shards = {name: stack.enter_context(safe_open(directory / name, framework="pt", device="cpu"))
                          for name in names}
                for expert in selected:
                    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
                    gate_up[expert, :512].copy_(weight(prefix + ".gate_proj", shards))
                    gate_up[expert, 512:].copy_(weight(prefix + ".up_proj", shards))
                    down[expert].copy_(weight(prefix + ".down_proj", shards))
            module._parameters["gate_up_proj"] = torch.nn.Parameter(gate_up, requires_grad=False)
            module._parameters["down_proj"] = torch.nn.Parameter(down, requires_grad=False)

        def release(module, _args, output, layer=layer):
            set_module_tensor_to_device(module, "gate_up_proj", "meta")
            set_module_tensor_to_device(module, "down_proj", "meta")
            if calls[layer] == 1 and layer % 10 == 9:
                print(f"MINICPM_REFERENCE first_prefill_layers={layer + 1}", flush=True)
            return output

        experts.register_forward_pre_hook(onload)
        experts.register_forward_hook(release)
    model.eval()
    print(f"MINICPM_REFERENCE_LOAD missing={len(clean['missing_keys'])} "
          f"unexpected={len(clean['unexpected_keys'])} initialized={len(clean['newly_initialized_keys'])} "
          f"errors={len(clean['error_msgs'])} PASS", flush=True)
    return model, clean, calls


@torch.inference_mode()
def sequence(model, ids, forced, chunk, steps):
    model.model.rope_deltas = None
    cache = transformers.cache_utils.DynamicCache(config=model.config.text_config)
    position = 0
    logits, tokens, margins = [], [], []

    def forward(piece):
        nonlocal cache, position
        inputs = torch.tensor([piece], dtype=torch.long, device="cuda")
        mask = torch.ones((1, position + len(piece)), dtype=torch.long, device="cuda")
        arguments = {"attention_mask": mask, "past_key_values": cache, "use_cache": True}
        positions = model._prepare_position_ids_for_generation(inputs, arguments)
        positions = positions[..., -len(piece):].contiguous()
        output = model(input_ids=inputs, position_ids=positions, logits_to_keep=1, **arguments)
        cache = output.past_key_values
        position += len(piece)
        return output.logits[0, -1].float().cpu().contiguous()

    for start in range(0, len(ids), chunk):
        row = forward(ids[start:start + chunk])
    for step in range(steps):
        if step:
            row = forward([forced[step - 1]])
        check(row.numel() == 248144 and torch.isfinite(row).all(), "invalid full-vocabulary logits")
        values = torch.topk(row, 2).values
        logits.append(row)
        tokens.append(int(row.argmax()))
        margins.append(float(values[0] - values[1]))
    return {"logits": torch.stack(logits), "tokens": torch.tensor(tokens),
            "margins": torch.tensor(margins), "positions": torch.arange(len(ids), len(ids) + steps)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    check(args.repeats > 0, "reference repeat count must be positive")
    check(not args.output.resolve().is_relative_to(args.model.resolve()), "output is inside weights")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fixtures = json.loads(args.fixtures.read_text())
    model, loading, calls = load_model(args.model)
    report = {"definition": "transformers MiniCPMV4_7ForConditionalGeneration model definition",
              "torch": torch.__version__, "transformers": transformers.__version__,
              "loading": loading, "cells": [], "saved_converted_weights": False}
    (args.output / "load.json").write_text(json.dumps(report, indent=2) + "\n")
    for case in fixtures["cases"]:
        ids = [(i * 269 + 23) % 248144 for i in range(case["rows"])]
        check(len(case["forced"]) + 1 == fixtures["steps"], "forced continuation extent mismatch")
        for chunk in case["chunks"]:
            for repeat in range(args.repeats):
                before = calls.copy()
                result = sequence(model, ids, case["forced"], chunk, fixtures["steps"])
                expected = (len(ids) + chunk - 1) // chunk + fixtures["steps"] - 1
                check(all(after - start == expected for after, start in zip(calls, before)),
                      "model-definition expert layer call count mismatch")
                name = f'{case["name"]}-chunk{chunk}-repeat{repeat}'
                save_file(result, args.output / (name + ".safetensors"))
                report["cells"].append({"name": name, "layer_calls": expected,
                                        "min_top_two_margin": float(result["margins"].min())})
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(f"MINICPM_REFERENCE cell={name} layers=40 outputs_finite=true PASS", flush=True)


if __name__ == "__main__":
    main()
