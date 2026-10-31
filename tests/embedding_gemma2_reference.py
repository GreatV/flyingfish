"""EmbeddingGemma-2 model-definition and sentence-transformers reference."""
import argparse
import json
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file, save
from sentence_transformers.base.modules.normalize import Normalize
from sentence_transformers.sentence_transformer.modules.pooling import Pooling


CASES = ("short", "batch", "radius", "radius_batch", "semantic")
STAGES = ("embedding", "ple", "layer0", "layer5", "layer17", "layer23", "norm", "projected")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_bytes(path, data):
    with path.open("xb") as destination:
        destination.write(data)


def write_json(path, value):
    write_bytes(path, (json.dumps(value, indent=2) + "\n").encode())


def save_tensors(path, tensors):
    write_bytes(path, save({key: value.contiguous().cpu() for key, value in tensors.items()}))


def read_json(path):
    return json.loads(path.read_text())


def tensor_values(tensors, prefix=""):
    if prefix:
        return {key: value for key, value in tensors.items() if key.startswith(prefix)}
    return {
        key: value
        for key, value in tensors.items()
        if not key.startswith("repeat.") and not key.startswith("split.")
    }


def distances(actual, expected):
    a = actual.double()
    b = expected.double()
    require(a.shape == b.shape, f"metric shape mismatch {a.shape} != {b.shape}")
    delta = a - b
    cosine = 0.0
    if a.ndim == 1:
        cosine = 1.0 - float(torch.dot(a, b) / (a.norm() * b.norm()))
    return {
        "output_max": float(delta.abs().max()),
        "output_rmse": float(delta.square().mean().sqrt()),
        **({"cosine_distance": cosine} if a.ndim == 1 else {}),
    }


def tokenized(tokenizer, text, length):
    ids = tokenizer(text, add_special_tokens=True).input_ids
    require(len(ids) >= length, f"text has {len(ids)} tokens, need {length}")
    return ids[:length]


def prepare_inputs(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    require(tokenizer.padding_side == "right", "reference requires published right padding")
    street = "A city street with cars and buildings. "
    road = "A photograph of buildings beside a road. "
    kitchen = "A saucepan on a wooden kitchen table. "
    long = (street + road + kitchen) * 80
    raw = {
        "short": [tokenized(tokenizer, street * 2, 17)],
        "batch": [tokenized(tokenizer, street * 2, 17), tokenized(tokenizer, kitchen * 2, 9)],
        "radius": [tokenized(tokenizer, long, 600)],
        "radius_batch": [tokenized(tokenizer, long, 600), tokenized(tokenizer, road * 70, 512)],
        "semantic": [
            tokenized(tokenizer, street * 4, 32),
            tokenized(tokenizer, road * 5, 29),
            tokenized(tokenizer, kitchen * 5, 24),
        ],
    }
    metadata = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=False, exist_ok=False)
    for name, rows in raw.items():
        width = max(map(len, rows))
        ids = torch.zeros((len(rows), width), dtype=torch.int32)
        mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for row, values in enumerate(rows):
            ids[row, : len(values)] = torch.tensor(values, dtype=torch.int32)
            mask[row, : len(values)] = True
        save_tensors(args.out / f"{name}.safetensors", {"ids": ids, "mask": mask})
        metadata[name] = {"rows": len(rows), "tokens": width, "valid": [len(row) for row in rows]}
    write_json(args.out / "INPUT.json", metadata)
    print(f"PASS embedding_text_inputs cells={len(raw)}", flush=True)


def load_model(args, device):
    config = transformers.AutoConfig.from_pretrained(args.model, local_files_only=True)
    require(config.architectures == ["EmbeddingGemma2Model"], "declared architecture changed")
    model_class = getattr(transformers, config.architectures[0])
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    model, loading = model_class.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
        output_loading_info=True,
    )
    newly_initialized = list(loading["missing_keys"])
    report = {
        "architecture": config.architectures[0],
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "dtype": args.dtype,
        "bf16_reduction": args.bf16_reduction,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "missing_keys": sorted(loading["missing_keys"]),
        "unexpected_keys": sorted(loading["unexpected_keys"]),
        "mismatched_keys": sorted([list(item[:2]) for item in loading["mismatched_keys"]]),
        "error_msgs": list(loading["error_msgs"]),
        "newly_initialized_keys": sorted(newly_initialized),
        "published_file_bytes": (args.model / "model.safetensors").stat().st_size,
        "tensor_count": len(model.state_dict()),
    }
    require(not loading["missing_keys"], "missing checkpoint keys")
    require(not loading["unexpected_keys"], "unexpected checkpoint keys")
    require(not loading["mismatched_keys"], "mismatched checkpoint keys")
    require(not loading["error_msgs"], "checkpoint load errors")
    require(not newly_initialized, f"newly initialized model keys: {newly_initialized}")
    require(report["published_file_bytes"] == 1_488_915_288, "published file size changed")
    require(report["tensor_count"] == 1376, "model tensor count changed")
    model.eval().requires_grad_(False).to(device)
    return model, report


def capture_modules(model):
    language = model.language_model
    stages = {}

    def put(name, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        stages[name] = output.detach().float().cpu().contiguous()

    hooks = [
        language.embed_tokens.register_forward_hook(lambda module, inputs, output: put("embedding", output)),
        language.ple.register_forward_hook(lambda module, inputs, output: put("ple", output)),
        language.norm.register_forward_hook(lambda module, inputs, output: put("norm", output)),
        language.embedding_projection.register_forward_hook(lambda module, inputs, output: put("projected", output)),
    ]
    for index in (0, 5, 17, 23):
        hooks.append(language.layers[index].register_forward_hook(
            lambda module, inputs, output, index=index: put(f"layer{index}", output)
        ))
    return stages, hooks


def run_case(model, pooling, normalize, data, stages, device):
    stages.clear()
    ids = data["ids"].long().to(device)
    mask = data["mask"].to(device)
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=mask)
        projected = output.last_hidden_state
        features = pooling({"token_embeddings": projected, "attention_mask": mask})
        pooled = features["sentence_embedding"]
        vector = normalize({"sentence_embedding": pooled})["sentence_embedding"]
    result = {key: stages[key] for key in STAGES}
    result.update(pooled=pooled.float().cpu(), vector=vector.float().cpu())
    require(set(result) == set(STAGES) | {"pooled", "vector"}, "missing reference stage")
    for value in result.values():
        require(bool(torch.isfinite(value).all()), "nonfinite reference output")
    norms = vector.float().norm(dim=-1)
    require(bool((norms > 0).all()), "reference vector has zero norm")
    return result


def prefixed(tensors, prefix):
    return {prefix + key: value for key, value in tensors.items()}


def flatten_valid(result, valid):
    output = {}
    for key, value in result.items():
        if key in ("pooled", "vector"):
            output[key] = value
        elif key == "ple":
            output[key] = torch.cat([value[row, :count] for row, count in enumerate(valid)])
        else:
            output[key] = torch.cat([value[row, :count] for row, count in enumerate(valid)])
    return output


def split_rows(model, pooling, normalize, data, stages, device):
    valid = data["mask"].sum(1).tolist()
    rows = []
    for row, count in enumerate(valid):
        row_data = {
            "ids": data["ids"][row : row + 1, :count],
            "mask": data["mask"][row : row + 1, :count],
        }
        rows.append(run_case(model, pooling, normalize, row_data, stages, device))
    combined = {}
    for key in STAGES:
        combined[key] = torch.cat([row[key][0] for row in rows])
    combined["pooled"] = torch.cat([row["pooled"] for row in rows])
    combined["vector"] = torch.cat([row["vector"] for row in rows])
    return combined


def record(args):
    require(torch.cuda.is_available(), "CUDA reference device unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = args.bf16_reduction == "default"
    device = torch.device("cuda", args.ordinal)
    model, load_report = load_model(args, device)
    pooling = Pooling.load(str(args.model / "1_Pooling"))
    normalize = Normalize.load(str(args.model / "2_Normalize"))
    require(pooling.get_embedding_dimension() == 768, "pooling width changed")
    require(pooling.include_prompt, "published pooling excludes prompt tokens")
    stages, hooks = capture_modules(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=False, exist_ok=False)
    write_json(args.out / "LOAD.json", load_report)
    reports = {"cases": {}, "controls": {}}
    for name in CASES:
        data = load_file(args.inputs / f"{name}.safetensors")
        result = run_case(model, pooling, normalize, data, stages, device)
        repeat = run_case(model, pooling, normalize, data, stages, device)
        repeat_distances = {key: distances(repeat[key], result[key]) for key in result}
        saved = dict(result)
        if any(item["output_max"] != 0 for item in repeat_distances.values()):
            saved.update(prefixed(repeat, "repeat."))
        case_report = {
            "repeat": repeat_distances,
            "unit_norm_max": float((result["vector"].norm(dim=-1) - 1).abs().max()),
        }
        if data["ids"].shape[0] > 1:
            split = split_rows(model, pooling, normalize, data, stages, device)
            valid = flatten_valid(result, data["mask"].sum(1).tolist())
            saved.update(prefixed(split, "split."))
            case_report["batch_split"] = {key: distances(split[key], valid[key]) for key in result}
        if name == "radius":
            changed = dict(data)
            changed["ids"] = data["ids"].clone()
            changed["ids"][0, 512] = 0
            late = run_case(model, pooling, normalize, changed, stages, device)
            require(float((late["layer0"][0, 0] - result["layer0"][0, 0]).abs().max()) > 0,
                    "late token did not alter earlier bidirectional output")
            changed["ids"] = data["ids"].clone()
            changed["ids"][0, 0] = 0
            boundary = run_case(model, pooling, normalize, changed, stages, device)
            require(torch.equal(boundary["layer0"][0, 599], result["layer0"][0, 599]),
                    "token zero crossed the sliding radius")
            require(float((boundary["layer0"][0, 512] - result["layer0"][0, 512]).abs().max()) > 0,
                    "inclusive radius boundary did not reach token 512")
            reports["controls"]["bidirectional_radius"] = {
                "late_to_early_max": float((late["layer0"][0, 0] - result["layer0"][0, 0]).abs().max()),
                "outside_radius_max": float((boundary["layer0"][0, 599] - result["layer0"][0, 599]).abs().max()),
                "inclusive_boundary_max": float((boundary["layer0"][0, 512] - result["layer0"][0, 512]).abs().max()),
            }
        if name == "batch":
            changed = dict(data)
            changed["ids"] = data["ids"].clone()
            changed["ids"][~data["mask"]] = 1
            padded = run_case(model, pooling, normalize, changed, stages, device)
            valid = flatten_valid(result, data["mask"].sum(1).tolist())
            padded_valid = flatten_valid(padded, data["mask"].sum(1).tolist())
            require(all(torch.equal(padded_valid[key], valid[key]) for key in valid),
                    "padding mutation changed valid outputs")
            reports["controls"]["padding"] = {"changed_positions": int((~data["mask"]).sum()), "valid_max": 0.0}
        save_tensors(args.out / f"{name}.safetensors", saved)
        reports["cases"][name] = case_report
        print(f"PASS reference_case {name} rows={data['ids'].shape[0]} tokens={data['ids'].shape[1]}", flush=True)
    for hook in hooks:
        hook.remove()
    vectors = load_file(args.out / "semantic.safetensors")["vector"].double()
    similarity = torch.nn.functional.cosine_similarity(vectors[:, None], vectors[None, :], dim=-1)
    require(float(similarity[0, 1]) > float(similarity[0, 2]), "named semantic ordering reversed")
    reports["controls"]["semantic"] = {"street_road": float(similarity[0, 1]), "street_kitchen": float(similarity[0, 2])}
    write_json(args.out / "REPORT.json", reports)
    print(f"PASS embedding_reference cells={len(CASES)} dtype={args.dtype} reduction={args.bf16_reduction}", flush=True)


def freeze(args):
    oracle = args.arms[0]
    oracle_load = read_json(oracle / "LOAD.json")
    require(oracle_load["dtype"] == "fp32" and oracle_load["capability"] == [8, 9],
            "first freeze arm must be the sm89 FP32 oracle")
    loads = [read_json(arm / "LOAD.json") for arm in args.arms]
    required = {
        ((8, 9), "fp32", "strict"),
        ((8, 9), "bf16", "default"),
        ((8, 9), "bf16", "strict"),
        ((8, 6), "fp32", "strict"),
        ((8, 6), "bf16", "default"),
        ((8, 6), "bf16", "strict"),
    }
    actual = {(tuple(report["capability"]), report["dtype"], report["bf16_reduction"]) for report in loads}
    require(actual == required, f"freeze arms differ: {actual}")
    require(all(report["transformers"] == "5.21.0" and report["torch"] == loads[0]["torch"] for report in loads),
            "reference software versions differ")
    require(all(not report["missing_keys"] and not report["unexpected_keys"]
                and not report["mismatched_keys"] and not report["newly_initialized_keys"] for report in loads),
            "freeze requires clean full-model loads")
    frozen = {
        "status": "FROZEN",
        "candidate_observations": 0,
        "policy": "2 * maximum reference-only distance to the sm89 FP32 oracle",
        "unit_norm_limit": 0.0,
        "arms": loads,
        "cells": {},
    }
    for case in CASES:
        expected = load_file(oracle / f"{case}.safetensors")
        candidates = {arm: load_file(arm / f"{case}.safetensors") for arm in args.arms}
        for candidate in candidates.values():
            for vector_key, vector in candidate.items():
                if vector_key.endswith("vector"):
                    error = float((vector.double().norm(dim=-1) - 1).abs().max())
                    frozen["unit_norm_limit"] = max(frozen["unit_norm_limit"], 2 * error)
        groups = {""}
        if any(key.startswith("repeat.") for key in expected):
            groups.add("repeat.")
        if any(key.startswith("split.") for key in expected):
            groups.add("split.")
        bounds = {}
        for group in groups:
            for key in tensor_values(expected, group):
                stage = key[len(group):]
                for arm in args.arms:
                    candidate = candidates[arm]
                    require(key in candidate, f"missing {arm}/{key}")
                    require(candidate[key].shape == expected[key].shape, f"shape mismatch {arm}/{key}")
                    for metric, value in distances(candidate[key], expected[key]).items():
                        limit = 2 * value
                        bounds.setdefault(stage, {})[metric] = max(bounds.setdefault(stage, {}).get(metric, 0.0), limit)
        frozen["cells"][case] = bounds
    write_json(args.out / "FROZEN.json", frozen)
    print(f"PASS embedding_reference_freeze cells={len(CASES)} candidate_observations=0", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inputs", "record", "freeze"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--bf16-reduction", choices=("default", "strict"), default="strict")
    parser.add_argument("--ordinal", type=int, default=0)
    parser.add_argument("--arms", type=Path, nargs=6)
    args = parser.parse_args()
    if args.mode == "inputs":
        prepare_inputs(args)
    elif args.mode == "record":
        require(args.inputs is not None, "record requires shared inputs")
        record(args)
    else:
        require(args.arms is not None, "freeze requires six arms")
        freeze(args)


if __name__ == "__main__":
    main()
