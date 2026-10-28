"""Qwen3.8 model definition with bounded, in-memory published FP8 dequantization."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
import transformers
from safetensors.torch import load_file, save_file
from accelerate.utils import set_module_tensor_to_device
from transformers.integrations.finegrained_fp8 import FP8Linear
from minicpm_reference import check, sequence


def load_model(directory):
    config = transformers.AutoConfig.from_pretrained(directory, local_files_only=True)
    check(config.architectures == ["Qwen3_5ForConditionalGeneration"], "Qwen architecture mismatch")
    text = config.text_config
    check((text.hidden_size, text.num_hidden_layers, text.num_attention_heads,
           text.num_key_value_heads, text.head_dim, text.vocab_size) ==
          (5120, 64, 24, 4, 256, 248320), "Qwen published geometry mismatch")
    architecture = getattr(transformers, config.architectures[0])
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    native_bytes = sum((directory / shard).stat().st_size for shard in set(index["weight_map"].values()))
    free_bytes, _ = torch.cuda.mem_get_info()
    resident = native_bytes + 8 * 1024**3 <= free_bytes
    initialized, initialization_calls = [], []
    initialize = architecture._initialize_missing_keys

    def record_initialization(model, *args, **kwargs):
        initialization_calls.append(True)
        initialized.extend(name for name, value in model.state_dict(keep_vars=True).items()
                           if not getattr(value, "_is_hf_initialized", False))
        check(not initialized, f"newly initialized keys: {initialized}")
        return initialize(model, *args, **kwargs)

    with patch.object(architecture, "_initialize_missing_keys", record_initialization):
        model, loading = architecture.from_pretrained(
            directory, dtype=torch.bfloat16, device_map="cuda:0" if resident else "cpu",
            attn_implementation="eager",
            output_loading_info=True, local_files_only=True,
        )
    check(initialization_calls == [True], "reference initialization call count")
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        check(key in loading and not loading[key], f"unclean reference {key}: {loading.get(key)}")
    loading = {key: list(loading[key]) for key in
               ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    loading["newly_initialized_keys"] = initialized
    projections = [module for module in model.modules() if isinstance(module, FP8Linear)]
    check(projections, "no published FP8 projections loaded")
    for module in projections:
        check(module.weight.dtype == torch.float8_e4m3fn and tuple(module.block_size) == (128, 128),
              "reference changed native FP8 storage")
    if not resident:
        native_weights = {id(module.weight) for module in projections}
        for name, value in list(model.named_parameters()) + list(model.named_buffers()):
            if id(value) not in native_weights:
                set_module_tensor_to_device(model, name, "cuda:0")

    def forward(module, inputs):
        n, k = module.weight.shape
        check(n % 128 == 0 and k % 128 == 0, "reference block tail not declared")
        weights = module.weight.to(inputs.device).float().view(n // 128, 128, k // 128, 128)
        weights = (weights * module.weight_scale_inv.float()[:, None, :, None]).bfloat16().view(n, k)
        return torch.nn.functional.linear(inputs, weights, module.bias)

    for module in projections:
        module.forward = forward.__get__(module, type(module))
    calls = [0] * text.num_hidden_layers
    for index, layer in enumerate(model.model.language_model.layers):
        def count(_module, _inputs, _output, index=index):
            calls[index] += 1
        layer.register_forward_hook(count)
    model.eval()
    print(f"Q38_REFERENCE_LOAD projections={len(projections)} missing=0 unexpected=0 "
          f"initialized=0 errors=0 native_fp8_resident={resident} "
          f"native_bytes={native_bytes} free_bytes={free_bytes} bounded_bf16_projection=true PASS", flush=True)
    return model, loading, calls


@torch.inference_mode()
def media_sequence(model, inputs, forced, steps, head_all=False):
    model.model.rope_deltas = None
    ids = inputs["input_ids"].cuda()
    cache = transformers.cache_utils.DynamicCache(config=model.config.text_config)
    arguments = {key: value.cuda() for key, value in inputs.items()
                 if key not in ("input_ids", "target_sizes")}
    arguments["pixel_values"] = arguments["pixel_values"].bfloat16()
    arguments.update(past_key_values=cache, use_cache=True)
    logits, tokens, margins = [], [], []
    for step in range(steps):
        if step:
            ids = torch.tensor([[tokens[-1] if forced is None else forced[step - 1]]], device="cuda")
            arguments = dict(attention_mask=torch.ones((1, inputs["input_ids"].numel() + step),
                                                       dtype=torch.long, device="cuda"),
                             past_key_values=cache, use_cache=True)
        positions = model._prepare_position_ids_for_generation(ids, arguments)[..., -ids.shape[1]:]
        result = model(input_ids=ids, position_ids=positions, logits_to_keep=0 if head_all and step == 0 else 1, **arguments)
        cache = result.past_key_values
        row = result.logits[0, -1].float().cpu().contiguous()
        check(row.shape == (248320,) and torch.isfinite(row).all(), "invalid Qwen media logits")
        top = row.topk(2).values
        logits.append(row)
        tokens.append(int(row.argmax()))
        margins.append(float(top[0] - top[1]))
    return dict(logits=torch.stack(logits), tokens=torch.tensor(tokens), margins=torch.tensor(margins))


def media_reference(model, args, loading, calls):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    processor = None
    if args.forced_reference is None:
        processor = transformers.AutoProcessor.from_pretrained(args.model, local_files_only=True)
    encoder = []
    handle = model.model.visual.register_forward_hook(lambda *_: encoder.append(True))
    report = dict(loading=loading, cells=[], torch=torch.__version__, transformers=transformers.__version__)
    outputs = []
    for path in args.images:
        name = "image_" + path.stem
        if args.forced_reference is None:
            from PIL import Image
            with Image.open(path) as source:
                picture = source.convert("RGB").resize((256, 256))
            messages = [{"role": "user", "content": [{"type": "image"},
                         {"type": "text", "text": "Describe this image briefly."}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            inputs = dict(processor(text=[text], images=[picture], return_tensors="pt"))
            check(inputs["image_grid_thw"].shape == (1, 3)
                  and int(inputs["image_grid_thw"][0, 0]) == 1, "image grid layout")
            inputs["target_sizes"] = inputs["image_grid_thw"][:, 1:].clone().contiguous()
            forced = None
        else:
            inputs = load_file(args.forced_reference / (name + "-input.safetensors"))
            forced = load_file(args.forced_reference / (name + ".safetensors"))["tokens"].tolist()
        save_file(inputs, args.output / (name + "-input.safetensors"))
        primary = None
        for variant in ("eager", "sdpa", "all-head"):
            model.set_attn_implementation("sdpa" if variant == "sdpa" else "eager")
            first = None
            for repeat in range(args.repeats):
                before, layers = len(encoder), calls.copy()
                result = media_sequence(model, inputs, forced, 32, variant == "all-head")
                check(len(encoder) == before + 1, "image encoder call count")
                check(all(x - y == 32 for x, y in zip(calls, layers)), "image language layer call count")
                if first is None:
                    first = result["logits"]
                    if forced is None:
                        forced = result["tokens"].tolist()
                else:
                    check(torch.equal(first.view(torch.int32), result["logits"].view(torch.int32)),
                          "image reference repeat bits differ")
                suffix = ("" if variant == "eager" else "-" + variant) + ("" if repeat == 0 else f"-repeat{repeat}")
                save_file(result, args.output / (name + suffix + ".safetensors"))
                cell = dict(name=name, variant=variant, repeat=repeat, encoder_calls=1, steps=32,
                            rows=inputs["input_ids"].numel(), min_margin=float(result["margins"].min()),
                            text=tokenizer.decode(result["tokens"], skip_special_tokens=True))
                report["cells"].append(cell)
                print("Q38_MEDIA_REFERENCE " + json.dumps(cell), flush=True)
            if variant == "eager":
                primary = first
        outputs.append(primary)
    check(len(outputs) >= 2 and all(not torch.equal(a, b) for i, a in enumerate(outputs)
                                   for b in outputs[i + 1:]), "two-image control failed")
    handle.remove()
    report.update(encoder_calls=len(encoder), distinct_media=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--images", type=Path, nargs="+")
    parser.add_argument("--forced-reference", type=Path)
    args = parser.parse_args()
    check(args.repeats > 0, "reference repeats must be positive")
    check(not args.output.resolve().is_relative_to(args.model.resolve()), "output inside weights")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fixtures = json.loads(args.fixtures.read_text())
    model, loading, calls = load_model(args.model)
    if args.images:
        media_reference(model, args, loading, calls)
        return
    report = {"definition": "transformers Qwen3_5ForConditionalGeneration model definition",
              "weight_contract": "BF16-rounded published E4M3 times block scale, BF16 activations",
              "torch": torch.__version__, "transformers": transformers.__version__,
              "loading": loading, "cells": [], "saved_converted_weights": False}
    (args.output / "load.json").write_text(json.dumps(report, indent=2) + "\n")
    for case in fixtures["cases"]:
        ids = [(i * 269 + 23) % 248320 for i in range(case["rows"])]
        check(len(case["forced"]) + 1 == fixtures["steps"], "forced continuation extent mismatch")
        for chunk in case["chunks"]:
            for repeat in range(args.repeats):
                before = calls.copy()
                result = sequence(model, ids, case["forced"], chunk, fixtures["steps"])
                expected = (len(ids) + chunk - 1) // chunk + fixtures["steps"] - 1
                check(all(after - start == expected for after, start in zip(calls, before)),
                      "model-definition layer call count mismatch")
                name = f'{case["name"]}-chunk{chunk}-repeat{repeat}'
                save_file(result, args.output / (name + ".safetensors"))
                report["cells"].append({"name": name, "layer_calls": expected,
                                        "min_top_two_margin": float(result["margins"].min())})
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(f"Q38_REFERENCE cell={name} layers=64 full_vocab=248320 finite=true PASS", flush=True)


if __name__ == "__main__":
    main()
