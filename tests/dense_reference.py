"""Dense Llama model-definition oracle with explicit positions and frozen input IDs."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import LlamaForCausalLM
import transformers


def check(value, message):
    if not value:
        raise RuntimeError(message)


@torch.inference_mode()
def sequence(model, ids, forced, chunk, steps):
    cache = None
    position = 0
    output = None
    for start in range(0, len(ids), chunk):
        piece = ids[start:start + chunk]
        x = torch.tensor([piece], dtype=torch.long, device="cuda")
        positions = torch.arange(position, position + len(piece), device="cuda").unsqueeze(0)
        output = model(input_ids=x, position_ids=positions,
                       attention_mask=torch.ones((1, position + len(piece)), dtype=torch.long, device="cuda"),
                       past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        position += len(piece)
    logits, tokens, margins = [], [], []
    for step in range(steps):
        if step:
            token = forced[step - 1] if forced is not None else tokens[-1]
            output = model(input_ids=torch.tensor([[token]], device="cuda"),
                           position_ids=torch.tensor([[position]], device="cuda"),
                           attention_mask=torch.ones((1, position + 1), dtype=torch.long, device="cuda"),
                           past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            position += 1
        row = output.logits[0, -1].float().cpu().contiguous()
        check(row.numel() == 130560 and torch.isfinite(row).all(), "reference full logits invalid")
        top = torch.topk(row, 2).values
        logits.append(row)
        tokens.append(int(row.argmax()))
        margins.append(float(top[0] - top[1]))
    return {"logits": torch.stack(logits), "tokens": torch.tensor(tokens),
            "margins": torch.tensor(margins),
            "positions": torch.arange(len(ids), len(ids) + steps)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    check(args.repeats > 0, "reference repeat count must be positive")
    check(not args.output.resolve().is_relative_to(args.model.resolve()), "output is inside readonly weights")
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = json.loads(args.fixtures.read_text())
    config = json.loads((args.model / "config.json").read_text())
    check(config["architectures"] == ["LlamaForCausalLM"], "reference architecture mismatch")
    check((config["hidden_size"], config["num_hidden_layers"], config["num_attention_heads"], config["num_key_value_heads"], config["vocab_size"]) == (2048, 42, 16, 2, 130560), "reference geometry mismatch")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, loading = LlamaForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                    attn_implementation="eager", local_files_only=True,
                                                    output_loading_info=True)
    for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]:
        check(key in loading and not loading[key], f"unclean reference load {key}: {loading.get(key)}")
    model.eval().to("cuda")
    print("DENSE_REFERENCE_LOAD missing=0 unexpected=0 mismatched=0 errors=0 PASS", flush=True)
    report = {"torch": torch.__version__, "transformers": transformers.__version__,
              "hip": torch.version.hip, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(0), "definition": "transformers LlamaForCausalLM model definition",
              "loading": {key: list(value) for key, value in loading.items()}, "cells": []}
    for fixture in fixtures["cases"]:
        rows = fixture["rows"]
        ids = [(i * 269 + 23) % 130560 for i in range(rows)]
        check(len(fixture["forced"]) == fixtures["steps"] - 1, "fixed continuation extent mismatch")
        greedy = sequence(model, ids, None, rows, fixtures["steps"])
        save_file(greedy, str(args.output / (fixture["name"] + "-greedy.safetensors")))
        for repeat in range(args.repeats):
            for chunk in [rows, *fixtures["chunks"]]:
                tensors = sequence(model, ids, fixture["forced"], chunk, fixtures["steps"])
                name = f"{fixture['name']}-chunk{chunk}-repeat{repeat}"
                save_file(tensors, str(args.output / (name + ".safetensors")))
                report["cells"].append({"name": name, "min_margin": tensors["margins"].min().item(),
                                        "positions": tensors["positions"].tolist(), "tokens": tensors["tokens"].tolist()})
                print(f"DENSE_REFERENCE cell={name} rows={rows} steps={fixtures['steps']} vocab=130560 finite=true PASS", flush=True)
    (args.output / "fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"DENSE_REFERENCE cells={len(report['cells'])} clean_load=true PASS", flush=True)


if __name__ == "__main__":
    main()
