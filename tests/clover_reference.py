"""Pinned Clover model-definition oracle for the GDN mixer boundaries."""
import argparse
from contextlib import ExitStack, contextmanager
import copy
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file


REVISION = "4b0402b65e523d0183a14c431955438714cb16ff"


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def distance(value, oracle):
    delta = value.double() - oracle.double()
    check(bool(torch.isfinite(delta).all()), "nonfinite reference")
    return {"max_abs": delta.abs().max().item(),
            "relative_l2": (delta.norm() / oracle.double().norm().clamp_min(1e-30)).item()}


def pattern(shape, seed, scale=1.0):
    count = 1
    for dimension in shape:
        count *= dimension
    values = torch.arange(count, device="cuda", dtype=torch.int64)
    return (((values * 73 + seed * 31) % 1009).float() / 504 - 1).reshape(shape) * scale


def declared_model(definition):
    check(definition.name == REVISION, "definition directory must name the pinned revision")
    check(transformers.__version__ == "5.19.0", "reference requires transformers 5.19.0")
    config = transformers.AutoConfig.from_pretrained(
        definition, trust_remote_code=True, local_files_only=True)
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, dtype=torch.bfloat16)
    check(type(model).__name__ == config.architectures[0] == "CloverMoeModelLM",
          "declared model class mismatch")
    return model


@contextmanager
def mapped_model(args):
    from collections import OrderedDict

    check(args.device_budget_gib is not None and args.device_budget_gib > 0,
          "mapped reference requires its declared device budget")
    model, index = schema(args.model, args.definition)
    definition = importlib.import_module(type(model).__module__)
    original_linear = torch.nn.functional.linear
    original_wide = definition.linear_fp32_output
    staged = [0, 0, 0]
    cache, capacity, occupied = OrderedDict(), [0], [0]

    def clear():
        cache.clear()
        occupied[0] = 0
        torch.cuda.empty_cache()

    def plan():
        free, _total = torch.cuda.mem_get_info()
        available = min(free, (args.device_budget_gib << 30) - torch.cuda.memory_allocated())
        capacity[0] = available - (1 << 30)
        check(capacity[0] >= 2048 * 1024 * 4, "mapped reference cannot fit its selected expert slice")
        print(f"CLOVER_MAPPED_CACHE capacity_bytes={capacity[0]} dense_bytes={torch.cuda.memory_allocated()} "
              "reserve_bytes=1073741824 arithmetic=model_definition", flush=True)

    def weight_on_input(x, weight):
        if weight.device.type == "cpu":
            check(x.is_cuda and weight.ndim == 2, "mapped expert invocation layout")
            key = (weight.data_ptr(), tuple(weight.shape), x.dtype, x.device)
            if key in cache:
                staged[2] += 1
                cache.move_to_end(key)
                return cache[key]
            size = weight.numel() * x.element_size()
            check(size <= capacity[0], "selected expert slice exceeds derived cache capacity")
            while occupied[0] + size > capacity[0]:
                _old_key, old = cache.popitem(last=False)
                occupied[0] -= old.numel() * old.element_size()
            staged[0] += 1
            staged[1] += weight.numel() * weight.element_size()
            value = weight.to(device=x.device, dtype=x.dtype)
            cache[key] = value
            occupied[0] += size
            return value
        return weight

    def linear(x, weight, bias=None):
        return original_linear(x, weight_on_input(x, weight), bias)

    def wide(x, weight):
        return original_wide(x, weight_on_input(x, weight))

    with ExitStack() as stack:
        handles = {name: stack.enter_context(safe_open(args.model / name, framework="pt", device="cpu"))
                   for name in sorted(set(index.values()))}
        weights = {}
        for name, shard in index.items():
            value = handles[shard].get_tensor(name)
            weights[name] = value if ".mlp.experts." in name else value.to("cuda")
        report = model.load_state_dict(weights, strict=True, assign=True)
        check(not report.missing_keys and not report.unexpected_keys
              and all(not p.is_meta for p in model.parameters()), "unclean mapped full-model load")
        check(sum(p.device.type == "cpu" for p in model.parameters()) == 80,
              "mapped expert parameter census")
        del weights
        model.eval()
        model.generation_config = transformers.GenerationConfig.from_pretrained(
            args.model, local_files_only=True)
        definition.validate_penalties(model.generation_config, model.config.vocab_size)
        stack.enter_context(patch.object(torch.nn.functional, "linear", linear))
        stack.enter_context(patch.object(definition, "linear_fp32_output", wide))
        plan()
        print("CLOVER_MAPPED_LOAD PASS keys=813 missing=0 unexpected=0 meta=0 cpu_expert_parameters=80", flush=True)
        yield model, definition, staged, (clear, plan)


def generation_dtype(model, dtype, placement):
    placement[0]()
    for module in model.modules():
        for name, value in module.named_parameters(recurse=False):
            if value.is_cuda:
                setattr(module, name, torch.nn.Parameter(value.to(dtype=dtype), requires_grad=False))
    del value
    torch.cuda.empty_cache()
    placement[1]()


def validate_mapped(model, args, dtype):
    check(args.validation is not None, "mapped generation requires frozen layer validation")
    table = json.loads((args.validation / "thresholds.json").read_text())
    check(table["revision"] == REVISION and len(table["layer_cases"]) == 8,
          "mapped validation fixture identity")
    arm = "bf" if dtype == torch.bfloat16 else "fp"
    caches = {}
    with safe_open(args.validation / "layers.safetensors", framework="pt", device="cpu") as source:
        for case in table["layer_cases"]:
            tag, layer = case["tag"], case["layer"]
            x = source.get_tensor(tag + ".x").to(device="cuda", dtype=dtype)
            if dtype == torch.bfloat16 or layer not in caches:
                cache = {name.rsplit(".", 1)[-1]: source.get_tensor(name).to(device="cuda")
                         for name in source.keys() if name.startswith(tag + ".cache.")}
                if dtype == torch.float32:
                    cache = {name: value.float() for name, value in cache.items()}
            else:
                cache = caches[layer]
            got, state = layer_run(model.model.layers[layer], x, cache, case["prefix"], case["preview"])
            for name, value in got.items():
                if name == "ids":
                    continue
                want = source.get_tensor(f"{tag}.{arm}.{name}").to("cuda")
                error = distance(value, want)
                check(all(error[key] <= case["bounds"][name][key] for key in error),
                      f"mapped placement validation: {tag}/{arm}/{name}: {error}")
            if not case["preview"]:
                caches[layer] = state
            print(f"CLOVER_MAPPED_VALIDATE PASS tag={tag} precision={arm}", flush=True)


def cache_tensors(cache):
    if cache is None:
        return {}
    tensors = {"attention_mask": cache.attention_mask}
    for index, layer in enumerate(cache.layers):
        tensors.update({f"layer{index}.{name}": value for name, value in layer.items()})
    return tensors


def generation_pack(penalized, raw):
    changed = (penalized.view(torch.int32) != raw.float().view(torch.int32)).flatten().nonzero().flatten()
    return {"penalty_indices": changed.int(), "penalty_values": penalized.flatten()[changed]}


def generation_unpack(values, raw):
    penalized = raw.float().clone()
    penalized.flatten()[values["penalty_indices"].long()] = values["penalty_values"]
    return {"logits": raw, "penalized": penalized,
            "picks": values["picks"], "confidence": values["confidence"]}


def matching_generation_calls(context, tokens, case, calls, canonical):
    prefix = len(context)
    prompt = case["prompt"]
    if prefix < len(prompt) or context != prompt + case["response"][:prefix - len(prompt)]:
        return []
    return [entry for entry in calls if entry["preview"] and entry["case"] == case["tag"]
            and entry["prefix"] == prefix
            and torch.equal(tokens.cpu().int(), canonical.get_tensor(entry["key"] + ".tokens"))]


def generation_pending_calls(calls, completed):
    check(completed <= {call["key"] for call in calls}, "checkpoint contains an unknown call")
    if all(call["key"] in completed for call in calls):
        return []
    return [call for call in calls if not call["preview"] or call["key"] not in completed]


@torch.inference_mode()
def generation_fixtures(args):
    import inspect
    import sys

    check(args.resume == args.output.exists(), "existing generation output requires explicit resume")
    if not args.resume:
        args.output.mkdir(parents=True)
    check(not (args.output / "generation.json").exists(), "generation reference is already complete")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = args.reduction == "original"
    identity = {"revision": REVISION, "keys": 813, "device": torch.cuda.get_device_name(),
                "torch": torch.__version__, "reduction": args.reduction,
                "candidate_observations": 0, "progress_version": 1}
    if args.resume:
        table = json.loads((args.output / "PROGRESS.json").read_text())
        check(all(table.get(key) == value for key, value in identity.items()), "generation checkpoint identity")
    else:
        table = {**identity, "cases": [], "calls": [], "replays": []}

    def save(path, tensors):
        save_file({name: value.detach().cpu().contiguous().clone() for name, value in tensors.items()}, path)

    with mapped_model(args) as (model, definition, staged, placement):
        validate_mapped(model, args, torch.bfloat16)
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
        prompts = {
            33: "Prove the Pythagorean theorem using two different geometric arguments and explain each step clearly with all geometric assumptions stated.",
            31: "Prove the Pythagorean theorem using two different geometric arguments and explain each step clearly in elementary terms.",
        }
        function = inspect.unwrap(type(model).generate)
        lines, first = inspect.getsourcelines(function)
        fill_line = first + next(i for i, line in enumerate(lines) if "if not bool(pending.any()):" in line)
        commit_line = first + next(i for i, line in enumerate(lines) if "if len(response) < maximum and not stopped:" in line)
        original_predict = definition.predict_with_confidence
        current_case = [None]
        last_call = [None]
        last_raw = [None]
        snapshots = [None]

        def model_after(_module, inputs, kwargs, output):
            if kwargs.get("preview", False):
                return
            ids = kwargs.get("input_ids", inputs[0] if inputs else None)
            cache = kwargs.get("past_key_values")
            prefix = 0 if cache is None else cache.attention_mask.shape[1]
            check(all(bool(torch.isfinite(value).all()) for value in cache_tensors(output.past_key_values).values()),
                  "nonfinite committed generation reference")
            key = f"call{len(table['calls']):04d}"
            values = {"tokens": ids.int(), **{"cache." + k: v for k, v in cache_tensors(output.past_key_values).items()}}
            save(args.output / (key + ".safetensors"), values)
            table["calls"].append({"key": key, "case": current_case[0], "preview": False,
                                   "prefix": prefix, "rows": ids.shape[1]})
            snapshots[0] = {k: v.clone() for k, v in cache_tensors(output.past_key_values).items()}

        def preview_after(_module, inputs, kwargs, output):
            check(kwargs.get("preview") is True, "generation LM call must be a preview")
            ids = kwargs.get("input_ids", inputs[0] if inputs else None)
            cache = kwargs["past_key_values"]
            check(bool(torch.isfinite(output.logits).all()), "nonfinite generation reference logits")
            check(output.past_key_values is cache, "reference preview replaced committed cache")
            check(all(torch.equal(v, cache_tensors(cache)[k]) for k, v in snapshots[0].items()),
                  "reference preview mutated committed cache")
            key = f"call{len(table['calls']):04d}"
            save(args.output / (key + ".safetensors"), {"tokens": ids.int(), "logits": output.logits[0]})
            table["calls"].append({"key": key, "case": current_case[0], "preview": True,
                                   "prefix": cache.attention_mask.shape[1], "rows": ids.shape[1]})
            last_call[0] = key
            last_raw[0] = output.logits[0]

        def predictions(logits, gen):
            picks, confidence = original_predict(logits, gen)
            save(args.output / (last_call[0] + ".policy.safetensors"),
                 {**generation_pack(logits, last_raw[0]), "picks": picks.int(), "confidence": confidence})
            return picks, confidence

        def trace(frame, event, _arg):
            if frame.f_code is function.__code__ and event == "line":
                values = frame.f_locals
                if frame.f_lineno == fill_line:
                    penalties = values["penalties"]
                    save(args.output / (last_call[0] + ".fill.safetensors"),
                         {"block": values["block"].int(), "pending": values["pending"].int(),
                          "counts": penalties.counts})
                    table["calls"][-1]["length"] = penalties.length
                if frame.f_lineno == commit_line:
                    table["cases"][-1]["commits"].append({"response": list(values["response"]),
                        "stopped": values["stopped"], "remaining": values["remaining"],
                        "length": values["penalties"].length})
            return trace

        if not args.resume:
            with ExitStack() as hooks:
                hooks.callback(model.model.register_forward_hook(model_after, with_kwargs=True).remove)
                hooks.callback(model.register_forward_hook(preview_after, with_kwargs=True).remove)
                hooks.enter_context(patch.object(definition, "predict_with_confidence", predictions))
                previous_trace = sys.gettrace()
                try:
                    sys.settrace(trace)
                    for rows, count in ((33, 33), (31, 32)):
                        tag = f"p{rows}_n{count}"
                        current_case[0] = tag
                        text = prompts[rows]
                        rendered = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                            tokenize=False, add_generation_prompt=True, enable_thinking=True)
                        prompt = tokenizer.encode(rendered, add_special_tokens=False)
                        check(len(prompt) == rows, "model-card generation prompt length differs")
                        tokens = torch.tensor([prompt], device="cuda")
                        table["cases"].append({"tag": tag, "prompt": prompt, "prompt_text": text,
                            "input": "model-card example; enable_thinking=True", "maximum": count, "commits": []})
                        output = model.generate(tokens, max_new_tokens=count)
                        result = output[0, rows:].tolist()
                        print(f"CLOVER_GENERATION_REFERENCE case={tag} returned={len(result)} ids={result} decoded={tokenizer.decode(result)!r}", flush=True)
                        check(len(result) == count, "reference case stopped before required trajectory coverage")
                        table["cases"][-1]["response"] = result
                        table["cases"][-1]["decoded"] = tokenizer.decode(result)
                finally:
                    sys.settrace(previous_trace)

            snapshots[0], last_raw[0] = None, None
            with (args.output / "CANONICAL.json").open("x") as handle:
                json.dump({"revision": REVISION, "cases": table["cases"], "calls": table["calls"]}, handle, indent=2)
                handle.write("\n")
        canonical = args.output if args.replay is None else args.replay
        canonical_table = json.loads((canonical / "CANONICAL.json").read_text())
        if args.resume:
            check(table["canonical_calls"] == canonical_table["calls"]
                  and table["canonical_cases"] == canonical_table["cases"], "checkpoint canonical input mismatch")
        else:
            table["canonical_calls"], table["canonical_cases"] = canonical_table["calls"], canonical_table["cases"]

        def checkpoint():
            table["expert_slice_copies"], table["expert_source_bytes"] = staged[:2]
            table["expert_cache_hits"], table["expert_count_scope"] = staged[2], "current_reference_lease"
            temporary = args.output / "PROGRESS.tmp"
            with temporary.open("w") as handle:
                json.dump(table, handle, indent=2)
                handle.write("\n")
            temporary.replace(args.output / "PROGRESS.json")

        pairs = [(entry["mode"], entry["key"]) for entry in table["replays"]]
        check(len(set(pairs)) == len(pairs)
              and all(mode in ("bf", "fp", "repeat", "shape", "split") for mode, _key in pairs),
              "checkpoint replay census")
        checkpoint()
        for mode in ("bf", "fp", "repeat", "shape", "split"):
            completed = {entry["key"] for entry in table["replays"] if entry["mode"] == mode}
            pending_calls = generation_pending_calls(canonical_table["calls"], completed)
            if not pending_calls:
                continue
            dtype = torch.float32 if mode == "fp" else torch.bfloat16
            generation_dtype(model, dtype, placement)
            if mode == "fp":
                validate_mapped(model, args, dtype)
            cache, case = None, None
            for call in pending_calls:
                if call["case"] != case:
                    cache, case = None, call["case"]
                with safe_open(canonical / (call["key"] + ".safetensors"), framework="pt", device="cpu") as source:
                    ids = source.get_tensor("tokens").long().to("cuda")
                if mode == "shape":
                    ids = ids.repeat(2, 1)
                if call["preview"]:
                    before = {k: v.clone() for k, v in cache_tensors(cache).items()}
                    output = model(ids, past_key_values=cache, use_cache=True, preview=True)
                    check(bool(torch.isfinite(output.logits).all()), "nonfinite replay reference logits")
                    check(all(torch.equal(v, cache_tensors(cache)[k]) for k, v in before.items()), "replay preview changed committed cache")
                    value = {"logits": output.logits[0]}
                    sampling = importlib.import_module(definition.CloverPenaltyState.__module__)
                    gen = model.generation_config
                    eos = gen.eos_token_id if isinstance(gen.eos_token_id, list) else [gen.eos_token_id]
                    penalties = sampling.CloverPenaltyState(gen, model.config.vocab_size, "cuda", eos + gen.stop_token_ids + [gen.pad_token_id])
                    with safe_open(canonical / (call["key"] + ".fill.safetensors"), framework="pt", device="cpu") as source:
                        penalties.counts = source.get_tensor("counts").to("cuda")
                    penalties.length = call["length"]
                    penalized = output.logits[0].float().clone()
                    penalized[:, model.config.mask_token_id] = -float("inf")
                    penalties.apply(penalized, ids[:1], model.config.mask_token_id)
                    picks, confidence = definition.predict_with_confidence(penalized, gen)
                    value.update({"penalized": penalized, "picks": picks.int(), "confidence": confidence})
                else:
                    chunks = ids.split(32, dim=1) if mode == "split" else (ids,)
                    for chunk in chunks:
                        cache = model.model(chunk, past_key_values=cache, use_cache=True).past_key_values
                    value = {"cache." + k: v[:1] for k, v in cache_tensors(cache).items()}
                if call["key"] in completed:
                    continue
                entry = {"key": call["key"], "case": case, "mode": mode, "preview": call["preview"]}
                if mode != "bf":
                    with safe_open(args.output / f"bf.{call['key']}.safetensors", framework="pt", device="cpu") as source:
                        base = {key: source.get_tensor(key).to("cuda") for key in source.keys()}
                    if call["preview"]:
                        base = generation_unpack(base, base["logits"])
                    noise = {}
                    for key in set(base) - {"picks"}:
                        a, b = value[key], base[key]
                        finite = torch.isfinite(a)
                        check(torch.equal(finite, torch.isfinite(b)), "reference replay finite-mask mismatch")
                        forward, backward = distance(a[finite], b[finite]), distance(b[finite], a[finite])
                        noise[key] = {metric: max(forward[metric], backward[metric]) for metric in forward}
                    entry["noise"] = noise
                if mode in ("bf", "fp"):
                    name = f"{mode}.{call['key']}.safetensors"
                    if call["preview"]:
                        value = {"logits": value["logits"], **generation_pack(value["penalized"], value["logits"]),
                                 "picks": value["picks"], "confidence": value["confidence"]}
                    save(args.output / name, value)
                    entry["file"] = name
                table["replays"].append(entry)
                checkpoint()
                print(f"CLOVER_GENERATION_REPLAY mode={mode} call={call['key']} case={case} preview={call['preview']} PASS", flush=True)
        table["canonical_cases"] = canonical_table["cases"]
        table["canonical_calls"] = canonical_table["calls"]
        table["expert_slice_copies"], table["expert_source_bytes"] = staged[:2]
        table["expert_cache_hits"] = staged[2]
    with (args.output / "generation.json").open("x") as handle:
        json.dump(table, handle, indent=2); handle.write("\n")
    print(f"CLOVER_GENERATION_REFERENCE PASS cases=2 keys=813 candidate_observations=0 replays={len(table['replays'])}", flush=True)


def generation_bounds(args):
    from itertools import combinations
    import shutil

    check(not args.output.exists(), "generation freeze output must not exist")
    directories = args.trajectory_reference
    tables = [json.loads((directory / "generation.json").read_text()) for directory in directories]
    canonical = tables[0]["canonical_calls"]
    check(all(table["revision"] == REVISION and table["keys"] == 813
              and table["candidate_observations"] == 0
              and table["canonical_calls"] == canonical for table in tables),
          "generation reference arms do not replay the same canonical calls")
    check(len({(table["device"], table["reduction"]) for table in tables}) == 4,
          "generation freeze requires original/strict on both devices")
    check(tables[0]["calls"] == canonical, "first freeze arm must own the canonical trajectory")
    check(args.policy_reference is not None, "generation freeze requires the frozen publisher policy fixtures")
    policy = json.loads((args.policy_reference / "thresholds.json").read_text())
    check(policy["revision"] == REVISION and policy["bounds"] ==
          {"logit_bits": 0, "ids": 0, "pending": 0, "counts": 0, "returned": 0},
          "generation policy fixture identity")
    frozen = {"revision": REVISION, "keys": 813, "candidate_observations": 0,
              "method": "2x maximum within-arm repeat/shape/split/BF16-FP32 and same-precision cross-arm/device distance",
              "r2_changed": False, "structural_tolerance": 0,
              "cases": tables[0]["canonical_cases"], "calls": [], "prefix_bounds": {}}
    for call in canonical:
        entries, noise_legs = [], []
        with ExitStack() as stack:
            for directory, table in zip(directories, tables):
                replays = [entry for entry in table["replays"] if entry["key"] == call["key"]]
                check({entry["mode"] for entry in replays} == {"bf", "repeat", "shape", "split", "fp"},
                      "generation replay noise coverage")
                for entry in replays:
                    if entry["mode"] != "bf":
                        noise_legs.append(entry["noise"])
                    if entry["mode"] in ("bf", "fp"):
                        handle = stack.enter_context(safe_open(directory / entry["file"], framework="pt", device="cpu"))
                        values = {name: handle.get_tensor(name) for name in handle.keys()}
                        if call["preview"]:
                            values = generation_unpack(values, values["logits"])
                        entries.append((entry["mode"], values))
            keys = set(entries[0][1]) - {"picks"}
            check(all(set(entry) - {"picks"} == keys for _mode, entry in entries), "generation replay tensor census")
            bounds = {}
            for name in sorted(keys):
                values = [(mode, entry[name].float()) for mode, entry in entries]
                finite = torch.isfinite(values[0][1])
                check(all(torch.equal(torch.isfinite(value), finite) for _mode, value in values), "generation reference finite-mask disagreement")
                check(bool(finite.all()) or (name == "penalized" and int((~finite).sum()) == 32),
                      "nonfinite generation reference")
                if not bool(finite.all()):
                    values = [(mode, value[finite]) for mode, value in values]
                maxima = {key: max(leg[name][key] for leg in noise_legs) for key in ("max_abs", "relative_l2")}
                for (left_mode, left), (right_mode, right) in combinations(values, 2):
                    if left_mode != right_mode:
                        continue
                    for a, b in ((left, right), (right, left)):
                        measured = distance(a, b)
                        maxima = {key: max(maxima[key], measured[key]) for key in maxima}
                bounds[name] = {key: 2 * value for key, value in maxima.items()}
            frozen["calls"].append({**call, "bounds": bounds})
            if call["preview"]:
                group = f"{call['case']}.prefix{call['prefix']}"
                envelope = frozen["prefix_bounds"].setdefault(group, {})
                for name, limits in bounds.items():
                    previous = envelope.setdefault(name, {key: 0.0 for key in limits})
                    for key, limit in limits.items():
                        previous[key] = max(previous[key], limit)
        print(f"CLOVER_GENERATION_FREEZE call={call['key']} bounds={json.dumps(bounds)} reference_only=true", flush=True)
    inputs = {}
    for call in canonical:
        key = call["key"]
        with safe_open(directories[0] / (key + ".safetensors"), framework="pt", device="cpu") as source:
            inputs[key + ".tokens"] = source.get_tensor("tokens").clone()
        if call["preview"]:
            with safe_open(directories[0] / (key + ".policy.safetensors"), framework="pt", device="cpu") as source:
                inputs[key + ".picks"] = source.get_tensor("picks").clone()
                inputs[key + ".confidence"] = source.get_tensor("confidence").clone()
                policy_values = {name: source.get_tensor(name) for name in source.keys()}
                with safe_open(directories[0] / (key + ".safetensors"), framework="pt", device="cpu") as raw_source:
                    raw = raw_source.get_tensor("logits")
                ranked = generation_unpack(policy_values, raw)["penalized"].topk(2, dim=-1).values
                inputs[key + ".top2_gap"] = ranked[:, 0] - ranked[:, 1]
            with safe_open(directories[0] / (key + ".fill.safetensors"), framework="pt", device="cpu") as source:
                for name in ("counts", "block", "pending"):
                    inputs[key + "." + name] = source.get_tensor(name).clone()
    args.output.mkdir(parents=True)
    save_file(inputs, args.output / "inputs.safetensors")
    for name in ("thresholds.json", "sampling.safetensors"):
        shutil.copyfile(args.policy_reference / name, args.output / name)
    with (args.output / "FROZEN.json").open("x") as handle:
        json.dump(frozen, handle, indent=2); handle.write("\n")
    print(f"CLOVER_GENERATION_FREEZE PASS calls={len(canonical)} arms=4 noise_legs=5 candidate_observations=0 r2_changed=false", flush=True)


@torch.inference_mode()
def commit_operands(args):
    check(args.model is not None and args.replay is not None
          and args.policy_reference is not None, "commit operands require weights, native request and frozen bounds")
    check(not args.output.exists(), "commit operand output must not exist")
    frozen = json.loads((args.policy_reference / "FROZEN.json").read_text())
    capture = json.loads((args.replay / "capture.json").read_text())
    result = json.loads((args.replay / "result.json").read_text())
    check(frozen["revision"] == REVISION and frozen["keys"] == capture["keys"] == 813
          and frozen["candidate_observations"] == 0 and frozen["structural_tolerance"] == 0
          and frozen["r2_changed"] is False and result["mode"] == "native"
          and result["case"] == "p33_n33" and capture["preview_commit_control"]
          and not capture["defects"], "commit operand input identity")
    commits = [call for call in capture["calls"] if not call["preview"]
               and call["prefix"] == 33 and call["rows"] == 32]
    anchors = [call for call in frozen["calls"] if not call["preview"]
               and call["case"] == result["case"] and call["prefix"] == 33 and call["rows"] == 32]
    check(len(commits) == len(anchors) == 1, "commit operand boundary coverage")
    target, anchor = commits[0], anchors[0]
    bounds = {"state_" + name: anchor["bounds"]["cache.layer2." + name] for name in ("conv", "ssm")}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    args.output.mkdir(parents=True)
    tensors, accumulated, pre = {}, {}, {}

    def save(name, value):
        tensors[name] = value.detach().cpu().contiguous().clone()

    with mapped_model(args) as (model, _definition, _staged, placement):
        for dtype in (torch.bfloat16, torch.float32):
            generation_dtype(model, dtype, placement)
            validate_mapped(model, args, dtype)
            precision = "bf" if dtype == torch.bfloat16 else "fp"
            cache = None
            for call in capture["calls"]:
                if call["preview"]:
                    check(call["committed_unchanged"], "skipped preview changed committed state")
                    continue
                with safe_open(args.replay / (call["key"] + ".safetensors"), framework="pt", device="cpu") as source:
                    tokens = source.get_tensor("tokens").to(device="cuda", dtype=torch.int64)
                check((0 if cache is None else cache.attention_mask.shape[1]) == call["prefix"],
                      "commit operand causal position")
                hook = None
                if call is target and dtype == torch.float32:
                    def incoming(_module, values):
                        check(len(values) == 6 and values[5] is False
                              and set(values[1]) == {"conv", "ssm"}, "layer2 commit hook operands")
                        pre["x"] = values[0].detach().cpu().clone()
                        pre.update({name: value.detach().cpu().clone() for name, value in values[1].items()})
                    hook = model.model.layers[2].register_forward_pre_hook(incoming)
                try:
                    cache = model.model(tokens, past_key_values=cache, use_cache=True).past_key_values
                finally:
                    if hook is not None:
                        hook.remove()
                if call is target:
                    accumulated[precision] = {name: value.detach().cpu().clone()
                                              for name, value in cache.layers[2].items()}
                    break
            check(precision in accumulated, "target commit not replayed")
        layer = model.model.layers[2]
        wide_norm = copy.deepcopy(layer.input_layernorm)
        wide_mixer = copy.deepcopy(layer.linear_attn)
        for name, value in layer.state_dict().items():
            if not name.startswith("mlp."):
                check(torch.equal(value.float(), value.bfloat16().float()), "published BF16 weight changed")
                save("layer2.weight." + name, value.bfloat16())
        placement[0]()
        model.to("meta")
    del model, layer, cache, placement
    torch.cuda.empty_cache()
    check(pre["x"].shape == (1, 32, 2048) and pre["x"].dtype == torch.float32
          and pre["conv"].shape == (1, 8192, 3) and pre["conv"].dtype == torch.float32
          and pre["ssm"].shape == (1, 32, 128, 128) and pre["ssm"].dtype == torch.float32,
          "FP32 pre-commit operand layout")
    x = pre["x"].bfloat16()
    matched = {"conv": pre["conv"].bfloat16().float(), "ssm": pre["ssm"].clone()}
    tag = "layer2.commit"
    save(tag + ".x", x)
    for name, value in matched.items():
        save(tag + ".cache." + name, value)
    for name, value in pre.items():
        save(tag + ".original_fp32." + name, value)
    references = {}
    valid = torch.ones((1, 32), device="cuda", dtype=torch.bool)
    for precision, dtype in (("bf", torch.bfloat16), ("fp", torch.float32)):
        norm = copy.deepcopy(wide_norm).to(dtype=dtype)
        mixer = copy.deepcopy(wide_mixer).to(dtype=dtype)
        inputs = {name: value.to(device="cuda", dtype=dtype if name == "conv" else torch.float32)
                  for name, value in matched.items()}
        normalized = norm(x.to(device="cuda", dtype=dtype))
        _output, state = mixer(normalized, inputs, valid, False)
        references[precision] = {name: value.detach().cpu().clone() for name, value in state.items()}
        for name, value in references[precision].items():
            save(f"{tag}.{precision}.state_{name}", value)
        del norm, mixer, inputs, normalized, _output, state
    rows = []
    with safe_open(args.replay / (target["key"] + ".safetensors"), framework="pt", device="cpu") as native:
        for name in ("conv", "ssm"):
            value = native.get_tensor("cache.layer2." + name).clone()
            row = {"metric": "state_" + name, "bf_vs_fp": distance(accumulated["bf"][name], accumulated["fp"][name]),
                   "native_vs_bf": distance(value, accumulated["bf"][name]),
                   "native_vs_fp": distance(value, accumulated["fp"][name]), "limits": bounds["state_" + name]}
            rows.append(row)
            print("CLOVER_COMMIT_ACCUMULATED_ROW " + json.dumps(row), flush=True)
    rounding = {"x": distance(x.float(), pre["x"]), "conv": distance(matched["conv"], pre["conv"]),
                "ssm": distance(matched["ssm"], pre["ssm"])}
    table = {"revision": REVISION, "schema_keys": 813, "scope": "matched_fp32_commit_operands",
             "layer_cases": [{"tag": tag, "layer": 2, "prefix": 33, "rows": 32, "preview": False, "bounds": bounds}],
             "native_case": result["case"], "native_call": target["key"], "input_rounding": rounding,
             "accumulated_rows": rows, "diagnostic_only": True, "acceptance_rescore": False, "r2_changed": False}
    save_file(tensors, args.output / "layers.safetensors")
    with (args.output / "thresholds.json").open("x") as handle:
        json.dump(table, handle, indent=2); handle.write("\n")
    print("CLOVER_COMMIT_OPERAND_REFERENCE PASS layer=2 prefix=33 rows=32 precision_inputs=fp32 "
          "native_activation=bf16 native_history=fp32_bf16_values native_ssm=fp32 bounds=unchanged "
          "rounding=" + json.dumps(rounding), flush=True)


@torch.inference_mode()
def generation_replay(args):
    check(not args.output.exists(), "generation replay output must not exist")
    check(args.policy_reference is not None and args.replay is not None, "generation replay requires native capture and frozen trajectory")
    frozen = json.loads((args.policy_reference / "FROZEN.json").read_text())
    check(frozen["revision"] == REVISION and frozen["keys"] == 813
          and frozen["candidate_observations"] == 0 and frozen["r2_changed"] is False,
          "generation replay freeze identity")
    args.output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    records, defects = [], []

    def record(kind, valid, **values):
        entry = {"kind": kind, "valid": bool(valid), **values}
        records.append(entry)
        if not valid:
            defects.append(entry)
        print(f"CLOVER_GENERATION_WITNESS {'PASS' if valid else 'DEFECT'} {json.dumps(entry)}", flush=True)

    def load(directory, name):
        with safe_open(directory / name, framework="pt", device="cpu") as handle:
            return {key: handle.get_tensor(key).to("cuda") for key in handle.keys()}

    with mapped_model(args) as (model, definition, staged, placement), safe_open(
            args.policy_reference / "inputs.safetensors", framework="pt", device="cpu") as canonical:
        gen = model.generation_config
        mask, vocab = model.config.mask_token_id, model.config.vocab_size
        stops = sorted(set(gen.eos_token_id) | set(gen.stop_token_ids))
        paths = sorted(args.replay.glob("request*/result.json"))
        check(len(paths) == 5, "generation replay requires three free runs and two canonical replays")
        for dtype in (torch.bfloat16, torch.float32):
            generation_dtype(model, dtype, placement)
            validate_mapped(model, args, dtype)
            precision = "bf" if dtype == torch.bfloat16 else "fp"
            for result_path in paths:
                result = json.loads(result_path.read_text())
                directory = result_path.parent
                capture = json.loads((directory / "capture.json").read_text())
                case = next(case for case in frozen["cases"] if case["tag"] == result["case"])
                native_mode = result["mode"] == "native"
                check(result["mode"] in ("native", "canonical"), "generation replay mode")
                cache = None
                context = []
                history = torch.zeros(vocab, dtype=torch.int64, device="cuda")
                response, pending = [], torch.ones(32, dtype=torch.bool, device="cuda")
                block = torch.full((1, 32), mask, dtype=torch.int64, device="cuda")
                expect_replay, stopped = None, False
                for call in capture["calls"]:
                    key = call["key"]
                    source = load(directory, key + ".safetensors")
                    tokens = source["tokens"].long()
                    check(tokens.ndim == 2 and tokens.shape == (1, call["rows"])
                          and bool(((tokens >= 0) & (tokens < vocab)).all()), "native replay token layout/IDs")
                    prefix = 0 if cache is None else cache.attention_mask.shape[1]
                    record("position", prefix == call["prefix"], request=directory.name, call=key, precision=precision,
                           reference=prefix, native=call["prefix"])
                    if not call["preview"]:
                        if native_mode and precision == "bf" and expect_replay is not None:
                            record("causal_replay_ids", torch.equal(tokens, expect_replay), request=directory.name, call=key)
                            expect_replay = None
                        cache = model.model(tokens, past_key_values=cache, use_cache=True).past_key_values
                        context.extend(tokens[0].tolist())
                        end = cache.attention_mask.shape[1]
                        anchors = [entry for entry in frozen["calls"] if not entry["preview"]
                                   and entry["case"] == case["tag"] and entry["prefix"] + entry["rows"] == end]
                        if anchors:
                            check(len(anchors) == 1, "ambiguous causal reference boundary")
                            anchor = anchors[0]
                            for name, value in cache_tensors(cache).items():
                                if name == "attention_mask":
                                    continue
                                field = "cache." + name
                                actual = source[field]
                                check(actual.shape == value.shape, "native causal cache layout")
                                measured = distance(actual, value)
                                limits = anchor["bounds"][field]
                                record("cache", all(measured[k] <= limits[k] for k in measured), request=directory.name,
                                       call=key, precision=precision, field=field, measured=measured, limits=limits)
                        else:
                            check(end == 32 and len(case["prompt"]) == 33, "uncovered causal boundary")
                            print(f"CLOVER_GENERATION_COVERAGE request={directory.name} prefix32=existing_r2_gate intermediate_prompt_chunk=true", flush=True)
                        continue
                    snapshots = {name: value.clone() for name, value in cache_tensors(cache).items()}
                    output = model(tokens, past_key_values=cache, use_cache=True, preview=True)
                    check(output.past_key_values is cache
                          and all(torch.equal(v, cache_tensors(cache)[k]) for k, v in snapshots.items()),
                          "model definition preview changed committed state")
                    group = f"{case['tag']}.prefix{prefix}"
                    check(group in frozen["prefix_bounds"], "native preview prefix outside frozen geometry")
                    limits = frozen["prefix_bounds"][group]
                    measured = distance(source["logits"], output.logits[0])
                    record("logits", all(measured[k] <= limits["logits"][k] for k in measured), request=directory.name,
                           call=key, precision=precision, measured=measured, limits=limits["logits"], own_inputs=True)
                    if precision != "bf":
                        continue
                    canonical_calls = matching_generation_calls(context, tokens, case, frozen["calls"], canonical)
                    record("canonical_context", True, request=directory.name, call=key,
                           matched=len(canonical_calls) == 1, prefix_tokens=len(context), prefix=prefix)
                    if native_mode:
                        record("canvas_input", not stopped and torch.equal(tokens, block), request=directory.name, call=key)
                        record("response_length", call["length"] == len(response), request=directory.name, call=key,
                               native=call["length"], expected=len(response))
                    penalties = definition.CloverPenaltyState(gen, vocab, "cuda", stops + [gen.pad_token_id])
                    if native_mode:
                        penalties.counts, penalties.length = history, len(response)
                    else:
                        check(len(canonical_calls) == 1, "canonical replay did not use frozen canvas")
                        anchor_key = canonical_calls[0]["key"]
                        penalties.counts = canonical.get_tensor(anchor_key + ".counts").to("cuda")
                        penalties.length = canonical_calls[0]["length"]
                    penalized = source["logits"].float().clone()
                    penalized[:, mask] = -float("inf")
                    penalties.apply(penalized, tokens, mask)
                    prediction, confidence = definition.predict_with_confidence(penalized, gen)
                    ranked = penalized.topk(2, dim=-1).values
                    top_gap = ranked[:, 0] - ranked[:, 1]
                    if canonical_calls:
                        check(len(canonical_calls) == 1, "duplicate canonical canvas")
                        anchor = canonical_calls[0]
                        target = canonical.get_tensor(anchor["key"] + ".picks").to("cuda").long()
                        gap = canonical.get_tensor(anchor["key"] + ".top2_gap").to("cuda")
                        reference_confidence = canonical.get_tensor(anchor["key"] + ".confidence").to("cuda")
                        logit_bound = anchor["bounds"]["penalized"]["max_abs"]
                        probability_bound = anchor["bounds"]["confidence"]["max_abs"]
                        robust = gap > 2 * logit_bound
                        high = reference_confidence > 0.95 + probability_bound
                        low = reference_confidence < 0.95 - probability_bound
                        record("robust_policy", bool((prediction[robust] == target[robust]).all())
                               and bool((confidence[high] > 0.95).all()) and bool((confidence[low] < 0.95).all()),
                               request=directory.name, call=key, robust_ids=int(robust.sum()), robust_high=int(high.sum()),
                               robust_low=int(low.sum()), canonical_top2_gap=gap.tolist(),
                               canonical_threshold_margin=(reference_confidence - 0.95).tolist(),
                               native_top2_gap=top_gap.tolist(), native_threshold_margin=(confidence - 0.95).tolist())
                        active = tokens[0] == mask
                        scores = reference_confidence.masked_fill(~active, -float("inf"))
                        ordered = scores.topk(2).values
                        forced_robust = bool(low[active].all()) and bool(ordered[0] - ordered[1] > 2 * probability_bound)
                        if forced_robust:
                            record("robust_forced_progress", int(confidence.masked_fill(~active, -float("inf")).argmax()) == int(scores.argmax()),
                                   request=directory.name, call=key, active=int(active.sum()),
                                   margin=float(ordered[0] - ordered[1]) if int(active.sum()) > 1 else None)
                    if not native_mode:
                        continue
                    policy = generation_unpack(load(directory, key + ".policy.safetensors"), source["logits"])
                    fill = load(directory, key + ".fill.safetensors")
                    record("penalty_bits", torch.equal(penalized.view(torch.int32), policy["penalized"].view(torch.int32)),
                           request=directory.name, call=key)
                    record("argmax", torch.equal(prediction.int(), policy["picks"]), request=directory.name, call=key,
                           top2_gap=top_gap.tolist(), confidence_error=float((confidence - policy["confidence"]).abs().max()))
                    transfer = (confidence > 0.95) & pending
                    if not bool(transfer.any()):
                        transfer[confidence.masked_fill(~pending, -float("inf")).argmax()] = True
                    block[0, transfer] = prediction[transfer]
                    pending[transfer] = False
                    state = [len(response), 0, 0, int(pending.sum())]
                    record("fill", torch.equal(block.int(), fill["block"]) and torch.equal(pending[None].int(), fill["pending"])
                           and torch.equal(history, fill["counts"].long()) and call["state"] == state,
                           request=directory.name, call=key, remaining=int(pending.sum()), state=call["state"], expected=state)
                    if "commit" in call:
                        remaining = min(32, case["maximum"] - len(response))
                        new = block[0, :remaining].tolist()
                        stop_position = next((i for i, token in enumerate(new) if token in stops), None)
                        stopped = stop_position is not None
                        if stopped:
                            new = new[:stop_position + 1]
                        response.extend(new)
                        values = torch.tensor(new, device="cuda", dtype=torch.int64)
                        history.index_add_(0, values, torch.ones_like(values))
                        terminal = stopped or len(response) == case["maximum"]
                        expected = [len(response), len(new), int(stopped), 0]
                        commit = call["commit"]
                        record("commit", not bool(pending.any()) and commit["returned"] == new
                               and commit["state"] == expected and commit["produced"] == len(response)
                               and commit["terminal"] == terminal, request=directory.name, call=key, expected=expected, actual=commit)
                        if terminal:
                            record("terminal_replay", call is capture["calls"][-1], request=directory.name, call=key)
                        else:
                            expect_replay = block.clone()
                            block.fill_(mask)
                            pending.fill_(True)
                if native_mode and precision == "bf":
                    record("response", response == result["response"] and expect_replay is None,
                           request=directory.name, returned=len(response), native=result["response"], expected=response)
        summary = {"records": records, "defects": defects, "r2_changed": False,
                   "expert_slice_copies": staged[0], "expert_source_bytes": staged[1], "expert_cache_hits": staged[2]}
    with (args.output / "witness.json").open("x") as handle:
        json.dump(summary, handle, indent=2); handle.write("\n")
    print(f"CLOVER_GENERATION_REPLAY {'DEFECT' if defects else 'PASS'} requests=5 precisions=2 defects={len(defects)} r2_changed=false", flush=True)
    check(not defects, f"Clover generation candidate DEFECTs={len(defects)}")


def schema(model_dir, definition):
    for name in ("config.json", "configuration_clover_moe.py", "modeling_clover_moe.py",
                 "sampling_clover_moe.py", "model.safetensors.index.json"):
        check((model_dir / name).read_bytes() == (definition / name).read_bytes(),
              f"pinned definition mismatch: {name}")
    model = declared_model(definition)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    expected = model.state_dict()
    check(set(index) == set(expected) and len(index) == 813, "checkpoint key consumption")
    seen = set()
    for shard in sorted(set(index.values())):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                check(name not in seen and index.get(name) == shard, f"shard ownership: {name}")
                tensor = handle.get_slice(name)
                check(tuple(tensor.get_shape()) == tuple(expected[name].shape)
                      and tensor.get_dtype() == "BF16", f"header layout: {name}")
                seen.add(name)
    check(seen == set(expected), "missing shard keys")
    return model, index


def load_mixer(model, directory, index, layer):
    module = model.model.layers[layer].linear_attn
    prefix = f"model.layers.{layer}.linear_attn."
    weights = {}
    for name in module.state_dict():
        key = prefix + name
        with safe_open(directory / index[key], framework="pt", device="cpu") as handle:
            weights[name] = handle.get_tensor(key).to("cuda")
    report = module.load_state_dict(weights, strict=True, assign=True)
    check(not report.missing_keys and not report.unexpected_keys, "unclean mixer value load")
    check(all(not p.is_meta for p in module.parameters()), "unbound mixer parameter")
    module.eval()
    print(f"MIXER_LOAD layer={layer} keys={len(weights)} missing=0 unexpected=0 meta=0 PASS", flush=True)
    return module


def check_equations(definition):
    q = torch.tensor([[[[0.3, -0.7, 0.9, 0.2]], [[0.1, 0.2, 0.3, 0.4]], [[-0.6, 0.8, 0.1, 0.5]]]])
    k = q.flip(-1).contiguous()
    v = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4) / 17 - 0.5
    state = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4) / 19 - 0.7
    g = torch.tensor([[[-0.4, -0.1], [-0.8, -0.2], [-0.3, -0.7]]])
    valid = torch.tensor([[True, False, True]])
    scales, biases, coefficients = (torch.tensor(values) for values in ([3.0, 7.0], [1.0, 0.2], [0.8, -0.3]))
    actual = definition.bd_gdn_local_memory(q, k, v, g, clean_ssm_states=state,
        cache_indices=torch.tensor([0]), active_valid_mask=valid.flatten(), scale=0.5,
        softmax_logit_scale=scales, softmax_history_bias=biases, softmax_history_c=coefficients)
    q64 = q.double() / (q.double().square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k64 = k.double() / (k.double().square().sum(-1, keepdim=True) + 1e-6).sqrt()
    expected = torch.zeros_like(v, dtype=torch.float64)
    for row in (0, 2):
        for head in range(2):
            scores = torch.stack([scales[head].double() * torch.dot(q64[0, row, 0], k64[0, key, 0]) for key in (0, 2)]
                + [biases[head].double() + coefficients[head].double() * torch.tensor(2 / 32, dtype=torch.float64).log()])
            probabilities = scores.exp() / scores.exp().sum()
            history = (state[0, head].double() @ (q64[0, row, 0] * 0.5)) * g[0, row, head].double().exp()
            expected[0, row, head] = probabilities[0] * v[0, 0, head].double() + probabilities[1] * v[0, 2, head].double() + probabilities[2] * history
    beta = torch.full((1, 3, 2), 0.4)
    actual_commit, actual_state = definition.recurrent_commit_torch(q, k, v, g, beta, state, valid)
    oracle_state = state.double().clone()
    expected_commit = torch.zeros_like(expected)
    for row in (0, 2):
        for head in range(2):
            old = oracle_state[0, head] * g[0, row, head].double().exp()
            key = k64[0, row, 0]
            innovation = (v[0, row, head].double() - old @ key) * beta[0, row, head].double()
            oracle_state[0, head] = old + torch.outer(innovation, key)
            expected_commit[0, row, head] = oracle_state[0, head] @ (q64[0, row, 0] * 0.5)
    for name, got, oracle in (("preview", actual, expected), ("commit", actual_commit, expected_commit), ("state", actual_state, oracle_state)):
        error = (got.double() - oracle).abs().max().item()
        bound = 64 * torch.finfo(torch.float32).eps * (1 + oracle.abs().max().item())
        check(error <= bound, f"FP64 equation control {name}: {error} > {bound}")
        print(f"GDN_FP64_EQUATION {name} max_abs={error} bound={bound} PASS", flush=True)


def run(module, definition, x, cache, valid, preview, fixed=None):
    captured = {}
    handles = []
    for name in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z"):
        def hook(_module, _inputs, output, name=name):
            if fixed is not None:
                output = fixed[name].to(x.dtype).repeat(x.shape[0], 1, 1)
            captured[name] = output.detach().clone()
            return output
        handles.append(getattr(module, name).register_forward_hook(hook))

    def before_norm(_module, inputs):
        captured["core"] = inputs[0].detach().clone()

    def after_norm(_module, _inputs, output):
        captured["norm"] = output.detach().clone()

    handles += [module.norm.register_forward_pre_hook(before_norm),
                module.norm.register_forward_hook(after_norm)]
    original_recurrent = definition.recurrent_commit_torch

    def capture_recurrent(q, k, v, *args):
        captured["conv"] = torch.cat((q.flatten(2), k.flatten(2), v.flatten(2)), -1)
        return original_recurrent(q, k, v, *args)

    try:
        with patch.object(definition, "recurrent_commit_torch", capture_recurrent):
            _, next_cache = module(x, cache, valid, preview)
    finally:
        for handle in handles:
            handle.remove()
    captured.update({"state": next_cache["ssm"], "history": next_cache["conv"]})
    raw = captured["in_proj_qkv"]
    if preview:
        captured["conv"] = definition.bd_gdn_conv1d(
            raw.reshape(-1, 8192), clean_conv_states=cache["conv"],
            cache_indices=torch.arange(x.shape[0], device=x.device),
            conv_weights=module.conv1d.weight, block_size=x.shape[1],
            active_valid_mask=valid.flatten(), conv_right_weight=module.conv_right_weight,
        ).reshape(x.shape[0], x.shape[1], 8192)
    return captured


@torch.inference_mode()
def fixtures(args):
    check(not args.output.exists(), "output must not exist; frozen tables are create-exclusive")
    args.output.mkdir(parents=True)
    torch.manual_seed(47)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model, index = schema(args.model, args.definition)
    definition = importlib.import_module(type(model).__module__)
    check_equations(definition)
    table = {"revision": REVISION, "torch": torch.__version__,
             "transformers": transformers.__version__, "device": torch.cuda.get_device_name(),
             "schema_keys": 813, "scope": "real GDN weights; fixed BF16 projection boundaries",
             "cross_gpu": "NOT RUN", "cases": []}
    tensors = {}
    cases = [(0, 0, 32, 47, "all"), (0, 1, 1, 71, "all"),
             (20, 31, 2, 47, "all"), (20, 32, 8, 71, "holes"),
             (38, 33, 16, 47, "holes"), (38, 33, 32, 71, "all"),
             (38, 0, 32, 71, "none"), (38, 65, 32, 47, "all")]
    continuation = None
    for case_id, (layer, prefix, rows, seed, mask) in enumerate(cases):
        module = load_mixer(model, args.model, index, layer)
        wide = copy.deepcopy(module).float()
        hidden = pattern((1, rows, 2048), seed).bfloat16()
        valid = torch.ones((1, rows), dtype=torch.bool, device="cuda")
        if mask == "holes":
            valid[:, 1::3] = False
        elif mask == "none":
            valid.zero_()
        if prefix == 65:
            check(continuation is not None, "missing first committed block")
            cache = {key: value.clone() for key, value in continuation.items()}
        elif prefix:
            prompt = pattern((1, prefix, 2048), seed + 3).bfloat16()
            _, cache = module(prompt, None, torch.ones((1, prefix), dtype=torch.bool, device="cuda"), False)
            cache["ssm"] += pattern((1, 32, 128, 128), seed + 5, 0.25)
        else:
            cache = {"conv": torch.zeros((1, 8192, 3), device="cuda", dtype=torch.bfloat16),
                     "ssm": torch.zeros((1, 32, 128, 128), device="cuda")}
        for preview in (True, False):
            tag = f"case{case_id}.{'preview' if preview else 'commit'}"
            original = {key: value.clone() for key, value in cache.items()}
            bf = run(module, definition, hidden, cache, valid, preview)
            if case_id == 5 and not preview:
                continuation = {"conv": bf["history"].clone(), "ssm": bf["state"].clone()}
            fixed = {key: value for key, value in bf.items() if key.startswith("in_proj_")}
            fp = run(wide, definition, hidden.float(),
                     {key: value.float() for key, value in cache.items()}, valid, preview, fixed)
            repeat = run(module, definition, hidden, cache, valid, preview, fixed)
            batch = run(module, definition, hidden.repeat(2, 1, 1),
                        {key: value.repeat(2, 1, 1) if key == "conv" else value.repeat(2, 1, 1, 1)
                         for key, value in cache.items()}, valid.repeat(2, 1), preview, fixed)
            check(all(torch.equal(cache[key], original[key]) for key in cache), "input cache mutated")
            if preview:
                check(torch.equal(bf["state"], cache["ssm"]) and torch.equal(bf["history"], cache["conv"]),
                      "preview cache changed")
            bounds, reference, noise = {}, {}, {}
            for name in ("conv", "core", "norm", "state", "history"):
                reference[name] = distance(bf[name], fp[name])
                noise[name] = {metric: max(distance(repeat[name], bf[name])[metric],
                                          distance(batch[name][:1], bf[name])[metric])
                               for metric in ("max_abs", "relative_l2")}
                bounds[name] = {metric: 2 * max(reference[name][metric], noise[name][metric])
                                for metric in ("max_abs", "relative_l2")}
                tensors[f"{tag}.fp.{name}"] = fp[name].float().cpu().contiguous()
                tensors[f"{tag}.bf.{name}"] = bf[name].float().cpu().contiguous()
            for name, value in fixed.items():
                tensors[f"{tag}.{name}"] = value.cpu().contiguous()
            tensors[f"{tag}.valid"] = valid.int().cpu().contiguous()
            tensors[f"{tag}.history"] = cache["conv"].float().cpu().contiguous()
            tensors[f"{tag}.state"] = cache["ssm"].cpu().contiguous()
            for name in ("conv1d.weight", "conv_right_weight", "A_log", "dt_bias",
                         "bd_softmax_logit_scale", "bd_softmax_history_bias", "bd_softmax_history_c", "norm.weight"):
                tensors[f"{tag}.weight.{name}"] = module.state_dict()[name].cpu().contiguous()
            table["cases"].append({"tag": tag, "layer": layer, "prefix": prefix, "rows": rows,
                                   "preview": preview, "mask": mask, "seed": seed,
                                   "bounds": bounds, "reference": reference, "noise": noise})
            print(f"REFERENCE {tag} rows={rows} prefix={prefix} mask={mask} {json.dumps(bounds)} PASS", flush=True)
        del wide, module
        model.model.layers[layer].linear_attn.to("meta")
        torch.cuda.empty_cache()
    save_file(tensors, args.output / "gdn.safetensors")
    with (args.output / "thresholds.json").open("x") as handle:
        json.dump(table, handle, indent=2)
        handle.write("\n")
    print(f"CLOVER_REFERENCE PASS keys=813 cases={len(table['cases'])} frozen=true "
          f"max_memory_allocated={torch.cuda.max_memory_allocated()}", flush=True)


class FixedProjection(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return self.value.to(x.dtype).repeat(x.shape[0], 1, 1)


@torch.inference_mode()
def replay(args):
    check(not args.output.exists(), "replay output must not exist")
    args.output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model = declared_model(args.definition)
    definition = importlib.import_module(type(model).__module__)
    check_equations(definition)
    table = json.loads((args.replay / "thresholds.json").read_text())
    check(table["revision"] == REVISION and table["schema_keys"] == 813, "replay reference identity")
    results = {"revision": REVISION, "device": torch.cuda.get_device_name(), "cases": []}
    with safe_open(args.replay / "gdn.safetensors", framework="pt", device="cpu") as handle:
        for case in table["cases"]:
            tag, rows, preview = case["tag"], case["rows"], case["preview"]
            read = lambda name: handle.get_tensor(f"{tag}.{name}").cuda()
            module = copy.deepcopy(model.model.layers[case["layer"]].linear_attn)
            for name in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z"):
                value = torch.zeros((1, rows, 32), device="cuda", dtype=torch.bfloat16) if preview and name == "in_proj_b" else read(name)
                setattr(module, name, FixedProjection(value))
            module.out_proj = torch.nn.Identity()
            weights = {name: read(f"weight.{name}") for name in module.state_dict()}
            loaded = module.load_state_dict(weights, strict=True, assign=True)
            check(not loaded.missing_keys and not loaded.unexpected_keys
                  and all(not p.is_meta for p in module.parameters()), "replay mixer value load")
            cache = {"conv": read("history").bfloat16(), "ssm": read("state")}
            valid = read("valid").bool()
            x = torch.zeros((1, rows, 2048), device="cuda", dtype=torch.bfloat16)
            bf = run(module, definition, x, cache, valid, preview)
            wide = copy.deepcopy(module).float()
            fp = run(wide, definition, x.float(), {key: v.float() for key, v in cache.items()}, valid, preview)
            entry = {"tag": tag, "distance_to_local_fp32": {}, "distance_to_local_bf16": {},
                     "distance_to_peer_fp32": {}, "rounding_outside": {}}
            for name in ("conv", "core", "norm", "state", "history"):
                entry["distance_to_local_fp32"][name] = distance(bf[name], read(f"fp.{name}"))
                entry["distance_to_local_bf16"][name] = distance(bf[name], read(f"bf.{name}"))
                entry["distance_to_peer_fp32"][name] = distance(bf[name], fp[name])
            isolated = module.norm(read("bf.core").bfloat16(), read("in_proj_z").reshape(1, rows, 32, 128))
            for name, value, reference in (("conv", bf["conv"], read("bf.conv")), ("norm_isolated", isolated, read("bf.norm"))):
                def rank(tensor):
                    bits = tensor.bfloat16().view(torch.int16).int()
                    return torch.where(bits < 0, -32768 - bits, bits)
                entry["rounding_outside"][name] = int(((rank(value) - rank(reference)).abs() > 1).sum())
            results["cases"].append(entry)
            print(f"REFERENCE_REPLAY {json.dumps(entry)}", flush=True)
            del module, wide, bf, fp, cache, weights
    with (args.output / "replay.json").open("x") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    print(f"CLOVER_REFERENCE_REPLAY PASS cases={len(results['cases'])} "
          f"max_memory_allocated={torch.cuda.max_memory_allocated()}", flush=True)



def moe_routes(definition, logits, valid, top_p):
    masked = logits if top_p is None else definition.coreset_torch(logits, valid, 8, top_p)
    ids = masked.float().argsort(dim=-1, descending=True, stable=True)[..., :8]
    weights = masked.float().softmax(-1).gather(-1, ids)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    allowed = torch.ones(logits.shape[-1], device=logits.device, dtype=torch.int32)
    if bool(valid.any()):
        allowed = torch.isfinite(masked[valid][0]).int()
    return ids, weights, allowed


@torch.inference_mode()
def full_route_replay(args):
    check(not args.output.exists(), "route replay output must not exist")
    model = declared_model(args.definition)
    definition = importlib.import_module(type(model).__module__)
    torch.set_num_threads(2)
    records, results, weights = [], [], set()
    for line in (args.replay / "raw.log").read_text().splitlines():
        if line.startswith("CLOVER_FULL_ROUTE_DATA "):
            records.append(json.loads(line.removeprefix("CLOVER_FULL_ROUTE_DATA ")))
        elif line.startswith("CLOVER_FULL_NATIVE_RESULT "):
            results.append(json.loads(line.removeprefix("CLOVER_FULL_NATIVE_RESULT ")))
        elif line.startswith("CLOVER_FULL_MATCHED_ROUTE_WEIGHTS "):
            weights.add(line.split("tag=", 1)[1].split()[0])
    expected = {f"{phase}.layer{layer}" for phase in ("prefill", "preview", "commit") for layer in range(40)}
    check(len(records) == 120 and {r["tag"] for r in records} == weights == expected
          and len(results) == 1 and results[0]["cache_control"]
          and results[0]["acceptance"] == "frozen_union_r2", "incomplete final native route records")
    separated = -torch.arange(256, dtype=torch.float32)[None, None].expand(1, 32, -1).clone()
    valid = torch.ones((1, 32), dtype=torch.bool)
    wrong = torch.arange(8)[None, None].expand(1, 32, -1).clone()
    wrong[..., -1] = 255
    control_ids, _, _ = moe_routes(definition, separated, valid, None)
    check(not torch.equal(wrong, control_ids), "unreachable route control escaped")
    args.output.mkdir(parents=True)
    tensors, verdicts = {}, []
    for record in records:
        check(record["revision"] == REVISION and record["schema_keys"] == 813
              and record["acceptance"] == "frozen_union_r2", "route record identity")
        tag = record["tag"]
        logits = torch.tensor(record["reference_logits"], dtype=torch.float32).reshape(1, 32, 256)
        native = torch.tensor(record["native_logits"], dtype=torch.float32).reshape_as(logits)
        ids = torch.tensor(record["native_ids"], dtype=torch.int64).reshape(1, 32, 8)
        allowed = torch.tensor(record["native_allowed"], dtype=torch.int32)
        check(allowed.shape == (256,) and bool(((allowed == 0) | (allowed == 1)).all())
              and bool(torch.isfinite(logits).all()) and bool(torch.isfinite(native).all()), "route record shape/nonfinite")
        bound = record["bounds"]["route_logits"]["max_abs"]
        check(bound >= 0 and bool(torch.isfinite(torch.tensor(bound))), "route witness bound")
        top_p = 0.9 if record["preview"] else None
        identity_ids, _, _ = moe_routes(definition, logits, valid, top_p)
        reference_ids = torch.tensor(record["reference_ids"], dtype=torch.int64).reshape(1, 32, 8)
        check(torch.equal(identity_ids, reference_ids), "reference identity control failed: " + tag)
        # 2026-10-31 17:33:15Z: exact native logits are the bounded constructive witness.
        chosen, _, kept = moe_routes(definition, native, valid, top_p)
        maximum = float((native.double() - logits.double()).abs().max())
        bad_rows = torch.where((chosen != ids).any(-1)[0])[0].tolist()
        mask_bad = int((kept != allowed).sum())
        passes = maximum <= bound and not bad_rows and mask_bad == 0
        item = {"tag": tag, "pass": passes, "native_logit_max_abs": maximum, "frozen_logit_bound": bound,
                "id_bad_rows": bad_rows, "allowed_bad": mask_bad,
                "witness_max_abs": maximum, "reference_identity_exact": True,
                "native_allowed": torch.where(allowed)[0].tolist(), "witness_allowed": torch.where(kept)[0].tolist()}
        verdicts.append(item)
        tensors[tag + ".logits"] = native.contiguous()
        tensors[tag + ".ids"] = chosen.int().contiguous()
        tensors[tag + ".allowed"] = kept.contiguous()
        print(f"CLOVER_ROUTE_WITNESS {'PASS' if passes else 'DEFECT'} {json.dumps(item)}", flush=True)
    save_file(tensors, args.output / "witnesses.safetensors")
    defects = sum(not item["pass"] for item in verdicts)
    summary = {"native": results[0], "witness_case_defects": defects, "cases": verdicts,
               "criterion_utc": "2026-10-31T17:33:15Z", "bounds_changed": False, "control_detected": True,
               "reference_identity_cases": 120}
    with (args.output / "verdicts.json").open("x") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    passed = defects == 0 and results[0]["native_defects"] == 0
    print(f"CLOVER_FULL_ACCEPTANCE {'PASS' if passed else 'DEFECT'} layers=40 phases=3 "
          f"native_defects={results[0]['native_defects']} witness_case_defects={defects} "
          "records=120 reference_identity_cases=120 controls=1/1 bounds=unchanged_r2", flush=True)
    if not passed:
        raise SystemExit(1)


def moe_run(module, definition, x, valid, preview, logits):
    handle = module.gate.register_forward_hook(lambda _m, _i, _o: logits.to(x.dtype).repeat(x.shape[0], 1, 1))
    captures = {}
    def inputs(_m, values):
        captures["ids"], captures["weights"] = values[1:]
        captures["contributions"] = torch.zeros((*values[1].shape, x.shape[-1]), device=x.device, dtype=x.dtype)
    def output(_m, _i, value):
        captures["routed"] = value.reshape_as(x).clone()
    hooks = [module.experts.register_forward_pre_hook(inputs), module.experts.register_forward_hook(output)]
    original = definition.linear_fp32_output
    position = 0
    def linear(a, w):
        nonlocal position
        value = original(a, w)
        expert = captures["ids"].unique(sorted=True)[position]
        row, slot = torch.where(captures["ids"] == expert)
        captures["contributions"][row, slot] = (value * captures["weights"][row, slot, None]).to(x.dtype)
        position += 1
        return value
    try:
        with patch.object(definition, "linear_fp32_output", linear):
            captures["output"] = module(x, valid, preview)
    finally:
        handle.remove()
        for hook in hooks:
            hook.remove()
    return captures


@torch.inference_mode()
def moe_fixtures(args):
    check(not args.output.exists(), "MoE output must not exist")
    args.output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, index = schema(args.model, args.definition)
    definition = importlib.import_module(type(model).__module__)
    tensors = {}
    table = {"revision": REVISION, "schema_keys": 813, "torch": torch.__version__,
             "route_atol": 8 * torch.finfo(torch.float32).eps,
             "route_rtol": 32 * torch.finfo(torch.float32).eps,
             "route_cases": [], "moe_cases": []}
    def save(name, value):
        tensors[name] = value.detach().cpu().contiguous().clone()
    def route_case(tag, logits, valid, top_p):
        ids, weights, allowed = moe_routes(definition, logits, valid, top_p)
        again = moe_routes(definition, logits, valid, top_p)
        check(all(torch.equal(a, b) for a, b in zip((ids, weights, allowed), again)), "route repeat noise")
        shape = moe_routes(definition, logits.repeat(2, 1, 1), valid.repeat(2, 1), top_p)
        check(torch.equal(ids, shape[0][:1]) and torch.equal(weights, shape[1][:1]), "route batch-shape noise")
        for name, value in (("logits", logits), ("valid", valid.int()), ("ids", ids.int()), ("weights", weights), ("allowed", allowed)):
            save(f"{tag}.{name}", value)
        mask = logits.double()
        if top_p is not None:
            mask = mask.masked_fill(valid[..., None] & ~allowed.bool()[None, None], -float("inf"))
        reference = mask.softmax(-1).gather(-1, ids)
        reference /= reference.sum(-1, keepdim=True)
        noise = distance(weights, reference)
        entry = {"tag": tag, "rows": logits.shape[1], "top_p": top_p, "noise": noise,
                 "allowed_count": int(allowed.sum()), "union": ids.unique().tolist(),
                 "zero_fillers": int((weights == 0).sum())}
        table["route_cases"].append(entry)
        print(f"CLOVER_ROUTE_REFERENCE {json.dumps(entry)} PASS", flush=True)
        return ids, weights
    for tag, rows, top_p, kind in [("tie", 32, 0.5, "all"), ("crossing", 32, 0.51, "all"),
            ("resolved", 32, 0.9, "split"), ("holes", 32, 0.9, "holes"),
            ("invalid", 32, 0.9, "none"), ("short", 1, 0.01, "all"),
            ("causal", 32, None, "split")]:
        logits = torch.zeros((1, rows, 256), device="cuda", dtype=torch.bfloat16)
        valid = torch.ones((1, rows), device="cuda", dtype=torch.bool)
        if kind in ("split", "holes"):
            logits[:, :16, 16:24] = 3
            logits[:, 16:, 96:104] = 3
        if kind == "holes":
            valid[:, ::5] = False
        if kind == "none":
            valid.zero_()
        route_case(tag, logits, valid, top_p)
    for layer in (0, 20):
        module = model.model.layers[layer].mlp
        prefix = f"model.layers.{layer}.mlp."
        weights = {}
        for name in module.state_dict():
            key = prefix + name
            with safe_open(args.model / index[key], framework="pt", device="cpu") as handle:
                weights[name] = handle.get_tensor(key).to("cuda")
        report = module.load_state_dict(weights, strict=True, assign=True)
        check(not report.missing_keys and not report.unexpected_keys and all(not p.is_meta for p in module.parameters()), "unclean MoE value load")
        module.eval()
        print(f"MOE_LOAD layer={layer} keys={len(weights)} missing=0 unexpected=0 meta=0 PASS", flush=True)
        x = (pattern((1, 1, 2048), 47 + layer, 1.75) + pattern((1, 32, 2048), 71, 0.01)).bfloat16()
        valid = torch.ones((1, 32), device="cuda", dtype=torch.bool)
        logits = module.gate(x)
        weight_tag = f"layer{layer}"
        selected = set()
        for label, top_p in (("preview", 0.9), ("sparse", 0.01)):
            tag = f"{weight_tag}.{label}"
            module.config.moe_block_coreset_top_p = top_p
            ids, routing = route_case(tag, logits, valid, top_p)
            selected.update(ids.unique(sorted=True).tolist())
            save(f"{tag}.x", x)
            bf = moe_run(module, definition, x, valid, True, logits)
            repeat = moe_run(module, definition, x, valid, True, logits)
            shape = moe_run(module, definition, x.repeat(2, 1, 1), valid.repeat(2, 1), True, logits)
            wide = copy.deepcopy(module).float()
            fp = moe_run(wide, definition, x.float(), valid, True, logits)
            bounds, noise = {}, {}
            for name in ("contributions", "routed", "output"):
                value = bf[name]
                same = repeat[name]
                varied = shape[name][:value.shape[0]]
                noise[name] = {"repeat": distance(value, same), "shape": distance(value, varied), "bf16_fp32": distance(value, fp[name])}
                bounds[name] = {key: 2 * max(item[key] for item in noise[name].values()) for key in ("max_abs", "relative_l2")}
                save(f"{tag}.bf.{name}", value)
                save(f"{tag}.fp.{name}", fp[name].float())
            table["moe_cases"].append({"tag": tag, "weights_tag": weight_tag, "layer": layer, "top_p": top_p,
                                     "experts": ids.unique(sorted=True).tolist(), "bounds": bounds, "noise": noise})
            print(f"CLOVER_MOE_REFERENCE tag={tag} union={len(ids.unique())} bounds={json.dumps(bounds)} PASS", flush=True)
            del wide, bf, fp, shape, repeat
        check(len(selected) <= 24, "real fixture expert union exceeds the small-fixture bound")
        for name, value in module.state_dict().items():
            if name in ("experts.gate_up_proj", "experts.down_proj"):
                for expert in sorted(selected):
                    save(f"{weight_tag}.expert{expert}.{name.split('.')[-1]}", value[expert])
            else:
                save(f"{weight_tag}.{name}", value)
        del weights
        module.to("meta")
        torch.cuda.empty_cache()
    save_file(tensors, args.output / "moe.safetensors")
    with (args.output / "thresholds.json").open("x") as handle:
        json.dump(table, handle, indent=2)
        handle.write("\n")
    print(f"CLOVER_MOE_REFERENCE PASS routes={len(table['route_cases'])} cases={len(table['moe_cases'])} frozen=true max_memory_allocated={torch.cuda.max_memory_allocated()}", flush=True)



@torch.inference_mode()
def moe_replay(args):
    check(not args.output.exists(), "MoE replay output must not exist")
    args.output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    table = json.loads((args.replay / "thresholds.json").read_text())
    check(table["revision"] == REVISION and table["schema_keys"] == 813, "MoE replay identity")
    model = declared_model(args.definition)
    definition = importlib.import_module(type(model).__module__)
    result = {"revision": REVISION, "device": torch.cuda.get_device_name(), "routes": [], "cases": []}
    with safe_open(args.replay / "moe.safetensors", framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name).clone() for name in handle.keys()}
    aligned = copy.deepcopy(table)
    aligned["device"] = result["device"]
    with safe_open(args.replay / "moe.safetensors", framework="pt", device="cuda") as handle:
        for case in table["route_cases"]:
            tag = case["tag"]
            ids, weights, allowed = moe_routes(definition, handle.get_tensor(f"{tag}.logits"), handle.get_tensor(f"{tag}.valid").bool(), case["top_p"])
            check(torch.equal(ids.int(), handle.get_tensor(f"{tag}.ids")) and torch.equal(allowed, handle.get_tensor(f"{tag}.allowed")), "cross-device route ids/allowed")
            result["routes"].append({"tag": tag, "noise": distance(weights, handle.get_tensor(f"{tag}.weights"))})
        for case in table["moe_cases"]:
            tag, weight_tag = case["tag"], case["weights_tag"]
            selected = sorted(int(name.split(".expert")[1].split(".")[0]) for name in tensors
                              if name.startswith(f"{weight_tag}.expert") and name.endswith(".gate_up_proj"))
            with torch.device("meta"):
                module = definition.CloverMoeBlock(copy.deepcopy(model.config))
                compact = copy.deepcopy(model.config)
                compact.num_experts = len(selected)
                module.experts = definition.CloverExperts(compact)
            weights = {"experts.gate_up_proj": torch.stack([handle.get_tensor(f"{weight_tag}.expert{e}.gate_up_proj") for e in selected]),
                       "experts.down_proj": torch.stack([handle.get_tensor(f"{weight_tag}.expert{e}.down_proj") for e in selected])}
            for name in module.state_dict():
                if not name.startswith("experts."):
                    weights[name] = handle.get_tensor(f"{weight_tag}.{name}")
            loaded = module.load_state_dict(weights, strict=True, assign=True)
            check(not loaded.missing_keys and not loaded.unexpected_keys and all(not p.is_meta for p in module.parameters()), "MoE replay selected parameter load")
            remap = torch.full((256,), -1, device="cuda", dtype=torch.int64)
            remap[torch.tensor(selected, device="cuda")] = torch.arange(len(selected), device="cuda")
            def remap_ids(_module, values):
                mapped = remap[values[1]]
                check(bool((mapped >= 0).all()), "MoE replay unbound expert")
                return values[0], mapped, values[2]
            module.experts.register_forward_pre_hook(remap_ids)
            module.config.moe_block_coreset_top_p = case["top_p"]
            module.eval()
            x = handle.get_tensor(f"{tag}.x")
            valid = handle.get_tensor(f"{tag}.valid").bool()
            logits = handle.get_tensor(f"{tag}.logits")
            bf = moe_run(module, definition, x, valid, True, logits)
            wide = copy.deepcopy(module).float()
            fp = moe_run(wide, definition, x.float(), valid, True, logits)
            entry = {"tag": tag, "noise": {}}
            for name in ("contributions", "routed", "output"):
                entry["noise"][name] = {"cross_bf16": distance(bf[name], handle.get_tensor(f"{tag}.bf.{name}")),
                    "peer_to_local_fp32": distance(bf[name], handle.get_tensor(f"{tag}.fp.{name}")),
                    "peer_to_peer_fp32": distance(bf[name], fp[name])}
            result["cases"].append(entry)
            print(f"CLOVER_MOE_REPLAY_CASE {json.dumps(entry)} PASS", flush=True)
            native_logits = module.gate(x)
            repeated_logits = module.gate(x)
            varied_logits = module.gate(x.repeat(2, 1, 1))[:1]
            native_ids, native_weights, native_allowed = moe_routes(definition, native_logits, valid, case["top_p"])
            native_bf = moe_run(module, definition, x, valid, True, native_logits)
            native_fp = moe_run(wide, definition, x.float(), valid, True, native_logits)
            entry["native_router"] = {"cross_logits": distance(native_logits, logits),
                "repeat_logits": distance(native_logits, repeated_logits),
                "shape_logits": distance(native_logits, varied_logits),
                "ids_changed": int((native_ids != handle.get_tensor(f"{tag}.ids")).sum()),
                "union": native_ids.unique(sorted=True).tolist()}
            for name, value in (("logits", native_logits), ("ids", native_ids.int()),
                                ("weights", native_weights), ("allowed", native_allowed)):
                tensors[f"{tag}.{name}"] = value.detach().cpu().contiguous().clone()
            for name in ("contributions", "routed", "output"):
                tensors[f"{tag}.bf.{name}"] = native_bf[name].detach().cpu().contiguous().clone()
                tensors[f"{tag}.fp.{name}"] = native_fp[name].float().detach().cpu().contiguous().clone()
            for aligned_case in aligned["moe_cases"]:
                if aligned_case["tag"] == tag:
                    aligned_case["experts"] = entry["native_router"]["union"]
            for aligned_case in aligned["route_cases"]:
                if aligned_case["tag"] == tag:
                    aligned_case["union"] = entry["native_router"]["union"]
                    aligned_case["allowed_count"] = int(native_allowed.sum())
                    aligned_case["zero_fillers"] = int((native_weights == 0).sum())
            print(f"CLOVER_MOE_NATIVE_ROUTER {json.dumps(entry['native_router'])} tag={tag} PASS", flush=True)
            del native_bf, native_fp
            del module, wide, weights, bf, fp
    save_file(tensors, args.output / "moe.safetensors")
    with (args.output / "thresholds.json").open("x") as handle:
        json.dump(aligned, handle, indent=2)
        handle.write("\n")
    with (args.output / "replay.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"CLOVER_MOE_REPLAY PASS routes={len(result['routes'])} cases={len(result['cases'])} max_memory_allocated={torch.cuda.max_memory_allocated()}", flush=True)


def layer_run(module, x, cache, prefix, preview):
    rows, batch = x.shape[1], x.shape[0]
    valid = torch.ones((batch, rows), device=x.device, dtype=torch.bool)
    positions = torch.arange(prefix, prefix + rows, device=x.device)[None].expand(batch, -1)
    captures, hooks = {}, []
    names = {"input_layernorm": "input_norm", "post_attention_layernorm": "moe_input",
             "mlp.gate": "route_logits", "mlp": "moe"}
    names["linear_attn" if hasattr(module, "linear_attn") else "self_attn"] = "mixer"
    for name, tag in names.items():
        def capture(_module, _inputs, output, tag=tag):
            captures[tag] = (output[0] if isinstance(output, tuple) else output).clone()
        hooks.append(module.get_submodule(name).register_forward_hook(capture))
    def routes(_module, values):
        captures["ids"], captures["weights"] = values[1].clone(), values[2].clone()
    hooks.append(module.mlp.experts.register_forward_pre_hook(routes, prepend=True))
    try:
        output, state = module(x, cache, valid,
            torch.ones((batch, prefix), device=x.device, dtype=torch.bool), positions, preview)
    finally:
        for hook in hooks:
            hook.remove()
    captures["output"] = output
    captures["residual"] = (x + captures["mixer"]).to(x.dtype)
    captures.update({f"state_{name}": value for name, value in state.items()})
    return captures, state


@torch.inference_mode()
def layer_fixtures(args):
    check(not args.output.exists(), "layer output must not exist")
    args.output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, index = schema(args.model, args.definition)
    tensors = {}
    table = {"revision": REVISION, "schema_keys": 813, "torch": torch.__version__,
             "attention_backend": model.config.attention_backend,
             "device": torch.cuda.get_device_name(), "layer_cases": []}
    def save(name, value):
        tensors[name] = value.detach().cpu().contiguous().clone()
    def repeat_cache(cache):
        return {name: value.repeat((2,) + (1,) * (value.ndim - 1)) for name, value in cache.items()}
    for layer in (0, 3):
        module = model.model.layers[layer]
        prefix_key = f"model.layers.{layer}."
        weights = {}
        for name in module.state_dict():
            key = prefix_key + name
            with safe_open(args.model / index[key], framework="pt", device="cpu") as handle:
                weights[name] = handle.get_tensor(key).to("cuda")
        loaded = module.load_state_dict(weights, strict=True, assign=True)
        check(not loaded.missing_keys and not loaded.unexpected_keys
              and all(not p.is_meta for p in module.parameters()), "unclean layer value load")
        module.eval()
        print(f"CLOVER_LAYER_LOAD layer={layer} keys={len(weights)} missing=0 unexpected=0 meta=0 PASS", flush=True)
        wide = copy.deepcopy(module).float()
        cache = ({"conv": pattern((1, 8192, 3), 41, 0.03).bfloat16(),
                  "ssm": pattern((1, 32, 128, 128), 43, 0.01)} if layer == 0 else
                 {"key": pattern((1, 2, 7, 256), 41, 0.1).bfloat16(),
                  "value": pattern((1, 2, 7, 256), 43, 0.1).bfloat16()})
        fp_cache = {name: value.float().clone() for name, value in cache.items()}
        selected = set()
        for block in range(2):
            prefix = 7 + block * 32
            x = (pattern((1, 1, 2048), 47 + layer + block, 1.75)
                 + pattern((1, 32, 2048), 71 + block, 0.01)).bfloat16()
            for preview in (True, False):
                tag = f"layer{layer}.{'preview' if preview else 'commit'}{block}"
                saved = {name: value.clone() for name, value in cache.items()}
                bf, next_cache = layer_run(module, x, cache, prefix, preview)
                check(all(torch.equal(value, cache[name]) for name, value in saved.items()), "layer mutated its input cache")
                again, _ = layer_run(module, x, cache, prefix, preview)
                shape, _ = layer_run(module, x.repeat(2, 1, 1), repeat_cache(cache), prefix, preview)
                fp, next_fp_cache = layer_run(wide, x.float(), fp_cache, prefix, preview)
                noise, bounds = {}, {}
                save(f"{tag}.x", x)
                for name, value in cache.items():
                    save(f"{tag}.cache.{name}", value)
                for name, value in bf.items():
                    save(f"{tag}.bf.{name}", value.int() if name == "ids" else value)
                    save(f"{tag}.fp.{name}", fp[name].int() if name == "ids" else fp[name].float())
                    if name == "ids":
                        continue
                    varied = shape[name][:value.shape[0]]
                    noise[name] = {"repeat": distance(value, again[name]), "shape": distance(value, varied),
                                   "bf16_fp32": distance(value, fp[name])}
                    bounds[name] = {key: 2 * max(item[key] for item in noise[name].values())
                                    for key in ("max_abs", "relative_l2")}
                for output in (bf, again, shape, fp):
                    selected.update(output["ids"].unique().tolist())
                ranked = bf["route_logits"].float().sort(descending=True).values
                gap = ranked[..., 7] - ranked[..., 8]
                case = {"tag": tag, "layer": layer, "block": block, "preview": preview,
                        "prefix": prefix, "rows": 32, "bounds": bounds, "noise": noise,
                        "route_gap_min": gap.min().item(), "route_ties": int((gap == 0).sum()),
                        "experts": bf["ids"].unique(sorted=True).tolist(),
                        "zero_fillers": int((bf["weights"] == 0).sum())}
                table["layer_cases"].append(case)
                print(f"CLOVER_LAYER_REFERENCE {json.dumps(case)} PASS", flush=True)
                if not preview:
                    cache, fp_cache = next_cache, next_fp_cache
                del bf, again, shape, fp, saved, next_cache, next_fp_cache
            check(len(selected) <= 64, "layer fixture expert union exceeds64")
        for name, value in weights.items():
            if name.startswith("mlp.experts."):
                for expert in sorted(selected):
                    save(f"layer{layer}.expert{expert}.{name.split('.')[-1]}", value[expert])
            else:
                save(f"layer{layer}.weight.{name}", value)
        for case in table["layer_cases"]:
            if case["layer"] == layer:
                case["available_experts"] = sorted(selected)
        del wide, weights, cache, fp_cache
        module.to("meta")
        torch.cuda.empty_cache()
    save_file(tensors, args.output / "layers.safetensors")
    with (args.output / "thresholds.json").open("x") as handle:
        json.dump(table, handle, indent=2)
        handle.write("\n")
    print(f"CLOVER_LAYER_REFERENCE PASS cases={len(table['layer_cases'])} "
          f"stage=local_noise max_memory_allocated={torch.cuda.max_memory_allocated()}", flush=True)


@torch.inference_mode()
def layer_replay(args):
    check(not args.output.exists(), "layer replay output must not exist")
    args.output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    table = json.loads((args.replay / "thresholds.json").read_text())
    check(table["revision"] == REVISION and table["schema_keys"] == 813,
          "layer replay identity")
    model = declared_model(args.definition)
    definition = importlib.import_module(type(model).__module__)
    check(model.config.attention_backend == table["attention_backend"],
          "layer attention backend mismatch")
    with safe_open(args.replay / "layers.safetensors", framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name).clone() for name in handle.keys()}
    aligned = copy.deepcopy(table)
    aligned["device"] = torch.cuda.get_device_name()
    aligned["cross_device"] = table["device"]
    with safe_open(args.replay / "layers.safetensors", framework="pt", device="cuda") as handle:
        for layer in (0, 3):
            cases = [case for case in aligned["layer_cases"] if case["layer"] == layer]
            selected = cases[0]["available_experts"]
            module = model.model.layers[layer]
            compact = copy.deepcopy(model.config)
            compact.num_experts = len(selected)
            with torch.device("meta"):
                module.mlp.experts = definition.CloverExperts(compact)
            weights = {}
            for name in module.state_dict():
                if name.startswith("mlp.experts."):
                    weights[name] = torch.stack([handle.get_tensor(
                        f"layer{layer}.expert{expert}.{name.split('.')[-1]}")
                        for expert in selected])
                else:
                    weights[name] = handle.get_tensor(f"layer{layer}.weight.{name}")
            loaded = module.load_state_dict(weights, strict=True, assign=True)
            check(not loaded.missing_keys and not loaded.unexpected_keys
                  and all(not p.is_meta for p in module.parameters()),
                  "layer replay selected parameter load")
            remap = torch.full((256,), -1, device="cuda", dtype=torch.int64)
            remap[torch.tensor(selected, device="cuda")] = torch.arange(len(selected), device="cuda")
            def remap_ids(_module, values):
                mapped = remap[values[1]]
                check(bool((mapped >= 0).all()), "layer replay unbound expert")
                return values[0], mapped, values[2]
            module.mlp.experts.register_forward_pre_hook(remap_ids)
            module.eval()
            wide = copy.deepcopy(module).float()
            first = cases[0]["tag"]
            cache_names = ("conv", "ssm") if layer == 0 else ("key", "value")
            cache = {name: handle.get_tensor(f"{first}.cache.{name}") for name in cache_names}
            fp_cache = {name: value.float().clone() for name, value in cache.items()}
            for case in cases:
                tag, prefix, preview = case["tag"], case["prefix"], case["preview"]
                x = handle.get_tensor(f"{tag}.x")
                saved = {name: value.clone() for name, value in cache.items()}
                bf, next_cache = layer_run(module, x, cache, prefix, preview)
                check(all(torch.equal(value, cache[name]) for name, value in saved.items()),
                      "layer replay mutated input cache")
                again, _ = layer_run(module, x, cache, prefix, preview)
                expanded = {name: value.repeat((2,) + (1,) * (value.ndim - 1))
                            for name, value in cache.items()}
                varied, _ = layer_run(module, x.repeat(2, 1, 1), expanded, prefix, preview)
                fp, next_fp_cache = layer_run(wide, x.float(), fp_cache, prefix, preview)
                for name, value in cache.items():
                    tensors[f"{tag}.cache.{name}"] = value.detach().cpu().contiguous().clone()
                case["cross_ids_changed"] = int((bf["ids"].int() != handle.get_tensor(f"{tag}.bf.ids")).sum())
                for name, value in bf.items():
                    if name != "ids":
                        noise = case["noise"][name]
                        noise.update({"peer_repeat": distance(value, again[name]),
                            "peer_shape": distance(value, varied[name][:value.shape[0]]),
                            "peer_fp32": distance(value, fp[name]),
                            "cross_bf16": distance(value, handle.get_tensor(f"{tag}.bf.{name}"))})
                        case["bounds"][name] = {key: 2 * max(item[key] for item in noise.values())
                                                for key in ("max_abs", "relative_l2")}
                    tensors[f"{tag}.bf.{name}"] = (value.int() if name == "ids" else value).detach().cpu().contiguous().clone()
                    tensors[f"{tag}.fp.{name}"] = (fp[name].int() if name == "ids" else fp[name].float()).detach().cpu().contiguous().clone()
                case["experts"] = bf["ids"].unique(sorted=True).tolist()
                case["zero_fillers"] = int((bf["weights"] == 0).sum())
                ranked = bf["route_logits"].float().sort(descending=True).values
                gap = ranked[..., 7] - ranked[..., 8]
                case["route_gap_min"], case["route_ties"] = gap.min().item(), int((gap == 0).sum())
                print(f"CLOVER_LAYER_REPLAY {json.dumps(case)} PASS", flush=True)
                if not preview:
                    cache, fp_cache = next_cache, next_fp_cache
                del bf, again, varied, fp, saved, expanded, next_cache, next_fp_cache
            del wide, weights, cache, fp_cache
            module.to("meta")
            torch.cuda.empty_cache()
    save_file(tensors, args.output / "layers.safetensors")
    with (args.output / "thresholds.json").open("x") as handle:
        json.dump(aligned, handle, indent=2)
        handle.write("\n")
    print(f"CLOVER_LAYER_REPLAY PASS cases={len(aligned['layer_cases'])} "
          f"frozen=true max_memory_allocated={torch.cuda.max_memory_allocated()}", flush=True)


@torch.inference_mode()
def full_boundaries(args):
    check(args.model is not None and args.replay is not None, "boundary oracle requires weights and full reference")
    check(not args.output.exists(), "boundary output must not exist")
    args.output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model, index = schema(args.model, args.definition)
    source = json.loads((args.replay / "thresholds.json").read_text())
    check(source["revision"] == REVISION and source["consumed_keys"] == 813
          and source["allow_bf16_reduced_precision_reduction"] is False, "strict full reference identity")
    tensors = {}
    table = {"revision": REVISION, "schema_keys": 813, "device": torch.cuda.get_device_name(),
             "scope": "matched-input GDN20/36 and matched-QKV causal attention19/35; diagnostic only",
             "allow_bf16_reduced_precision_reduction": False, "layer_cases": [], "attention_cases": []}

    def save(name, value):
        tensors[name] = value.detach().cpu().contiguous().clone()

    def load(module, prefix):
        weights = {}
        for name in module.state_dict():
            key = prefix + name
            with safe_open(args.model / index[key], framework="pt", device="cpu") as handle:
                weights[name] = handle.get_tensor(key).to("cuda")
        report = module.load_state_dict(weights, strict=True, assign=True)
        check(not report.missing_keys and not report.unexpected_keys
              and all(not p.is_meta for p in module.parameters()), "unclean boundary module load")
        module.eval()
        print(f"CLOVER_BOUNDARY_LOAD prefix={prefix} keys={len(weights)} missing=0 unexpected=0 meta=0", flush=True)
        return weights

    def record(tag, values):
        bf, again, shape, fp = values
        noise = {"repeat": distance(bf, again), "shape": distance(bf, shape[:1]), "bf16_fp32": distance(bf, fp)}
        save(tag + ".bf", bf)
        save(tag + ".fp", fp.float())
        return {key: 2 * max(v[key] for v in noise.values()) for key in ("max_abs", "relative_l2")}

    with safe_open(args.replay / "full.safetensors", framework="pt", device="cpu") as handle:
        for layer in (20, 36):
            module = model.model.layers[layer]
            weights = load(module, f"model.layers.{layer}.")
            x = handle.get_tensor(f"prefill.layer{layer - 1}.bf.output").to("cuda")
            tag = f"prefill.layer{layer}"
            save(tag + ".x", x)
            save(tag + ".cache.ssm", torch.zeros((1, 32, 128, 128), device="cuda"))
            save(tag + ".cache.conv", torch.zeros((1, 8192, 3), device="cuda", dtype=torch.bfloat16))
            wide = copy.deepcopy(module).float()
            values = [layer_run(m, v, None, 0, False)[0] for m, v in
                      ((module, x), (module, x), (module, x.repeat(2, 1, 1)), (wide, x.float()))]
            bounds = {}
            for name in ("input_norm", "mixer", "residual", "moe_input", "route_logits", "state_ssm", "state_conv"):
                bounds[name] = record(f"{tag}.{name}", [v[name] for v in values])
            table["layer_cases"].append({"tag": tag, "layer": layer, "prefix": 0, "preview": False, "bounds": bounds})
            for name, weight in weights.items():
                if not name.startswith("mlp.") or name == "mlp.gate.weight":
                    save(f"layer{layer}.weight.{name}", weight)
            print(f"CLOVER_BOUNDARY_REFERENCE layer={layer} input=reference_bf16 cache=zero bounds={json.dumps(bounds)}", flush=True)
            del wide, weights, values, x
            module.to("meta")
            torch.cuda.empty_cache()
        for layer in (19, 35):
            block = model.model.layers[layer]
            load(block.input_layernorm, f"model.layers.{layer}.input_layernorm.")
            load(block.self_attn, f"model.layers.{layer}.self_attn.")
            x = handle.get_tensor(f"prefill.layer{layer - 1}.bf.output").to("cuda")
            captured = {}
            original = torch.nn.functional.scaled_dot_product_attention
            def capture(q, k, v, **kwargs):
                captured.update(q=q.clone(), k=k.clone(), v=v.clone(), kwargs=kwargs)
                result = original(q, k, v, **kwargs)
                captured["bf"] = result.clone()
                return result
            valid = torch.ones((1, 32), device="cuda", dtype=torch.bool)
            with patch.object(torch.nn.functional, "scaled_dot_product_attention", capture):
                block.self_attn(block.input_layernorm(x), None, valid, valid[:, :0],
                                torch.arange(32, device="cuda")[None], False)
            q, k, v = (captured[name] for name in ("q", "k", "v"))
            check(torch.equal(k, k[:, ::8].repeat_interleave(8, 1))
                  and torch.equal(v, v[:, ::8].repeat_interleave(8, 1)), "attention grouped-KV layout")
            again = original(q, k, v, **captured["kwargs"])
            shape = original(q.repeat(2, 1, 1, 1), k.repeat(2, 1, 1, 1),
                             v.repeat(2, 1, 1, 1), **captured["kwargs"])
            fp = original(q.float(), k.float(), v.float(), **captured["kwargs"])
            tag = f"prefill.layer{layer}.attention"
            save(tag + ".q", q.transpose(1, 2))
            save(tag + ".k", k[:, ::8])
            save(tag + ".v", v[:, ::8])
            bound = record(tag + ".output", [value.transpose(1, 2) for value in (captured["bf"], again, shape, fp)])
            table["attention_cases"].append({"tag": tag, "layer": layer, "rows": 32,
                "q_heads": 16, "kv_heads": 2, "head_dim": 256, "prefix": 0, "bounds": bound})
            print(f"CLOVER_QKV_REFERENCE tag={tag} backend=sdpa operands=identical bounds={json.dumps(bound)}", flush=True)
            block.input_layernorm.to("meta")
            block.self_attn.to("meta")
            del x, q, k, v, again, shape, fp, captured
            torch.cuda.empty_cache()
    save_file(tensors, args.output / "layers.safetensors")
    with (args.output / "thresholds.json").open("x") as handle:
        json.dump(table, handle, indent=2)
        handle.write("\n")
    print(f"CLOVER_BOUNDARY_REFERENCE PASS layers=2 attention_cases=2 candidate_observations=0 max_memory_allocated={torch.cuda.max_memory_allocated()}", flush=True)


def native_trajectory(args, source):
    check(args.trajectory_reference is not None, "native trajectory requires four preserved reference arms")
    records = []
    with ExitStack() as stack:
        native = stack.enter_context(safe_open(args.replay / "native.safetensors", framework="pt", device="cpu"))
        arms = []
        for directory in args.trajectory_reference:
            table = json.loads((directory / "thresholds.json").read_text())
            check(table["revision"] == REVISION and table["consumed_keys"] == 813,
                  "trajectory reference identity")
            arms.append((directory.name, stack.enter_context(safe_open(
                directory / "full.safetensors", framework="pt", device="cpu"))))
        check(len({name for name, _ in arms}) == 4, "trajectory reference arms must be distinct")
        for layer in source["layers"]:
            x = native.get_tensor(f"layer{layer}.input")
            previous = native.get_tensor(f"layer{layer - 1}.output") if layer else arms[-1][1].get_tensor("prefill.input")
            check(torch.equal(x, previous), "native trajectory input continuity")
            names = ["output", "route_logits"] + [name.rsplit(".", 1)[1] for name in native.keys()
                                                   if name.startswith(f"layer{layer}.state_")]
            for name in names:
                values = [(f"{label}.{precision}", handle.get_tensor(f"prefill.layer{layer}.{precision}.{name}"))
                          for label, handle in arms for precision in ("bf", "fp")]
                actual = native.get_tensor(f"layer{layer}.{name}")
                distances, pairs = {}, []
                for label, oracle in values:
                    distances[label] = distance(actual, oracle)
                    for other, value in values:
                        if label != other:
                            pairs.append({"value": other, "oracle": label, **distance(value, oracle)})
                radii = {label: {key: max(pair[key] for pair in pairs if pair["oracle"] == label)
                                 for key in ("max_abs", "relative_l2")} for label, _ in values}
                record = {"layer": layer, "metric": name, "native_distances": distances,
                          "reference_pairs": pairs, "reference_radii": radii,
                          "outside_any": {key: any(distances[label][key] > radii[label][key] for label, _ in values)
                                          for key in ("max_abs", "relative_l2")},
                          "outside_all": {key: all(distances[label][key] > radii[label][key] for label, _ in values)
                                          for key in ("max_abs", "relative_l2")}}
                records.append(record)
                print(f"CLOVER_NATIVE_TRAJECTORY {json.dumps(record)}", flush=True)
    with (args.output / "trajectory.json").open("x") as handle:
        json.dump({"revision": REVISION, "records": records, "reference_arms": [name for name, _ in arms],
                   "diagnostic_only": True, "r2_changed": False}, handle, indent=2)
        handle.write("\n")
    print(f"CLOVER_NATIVE_TRAJECTORY PASS layers=21 reference_arms=4 precisions=2 records={len(records)} "
          "diagnostic_only=true r2_changed=false", flush=True)


@torch.inference_mode()
def native_operands(args):
    check(args.model is not None and args.replay is not None, "native operand replay requires weights and capture")
    check(not args.output.exists(), "native replay output must not exist")
    source = json.loads((args.replay / "capture.json").read_text())
    check(source["revision"] == REVISION and source["schema_keys"] == 813
          and source["scope"] == "actual_native_operands" and source["layers"] == list(range(21))
          and source["rows"] == 32 and source["prefix"] == 0 and source["phase"] == "prefill"
          and source["tensors"] == 189 and source["r2_changed"] is False
          and source["acceptance_rescore"] is False, "native capture identity")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    args.output.mkdir(parents=True)
    native_trajectory(args, source)
    model, index = schema(args.model, args.definition)
    records, tensors, controls = [], {}, 0
    boundaries = ("output", "input_norm", "mixer", "moe_input", "moe", "route_logits")
    with safe_open(args.replay / "native.safetensors", framework="pt", device="cpu") as handle:
        expected = {f"layer{layer}.{name}" for layer in source["layers"] for name in ("input", *boundaries)}
        expected.update(f"layer{layer}.state_{name}" for layer in source["layers"]
                        for name in (("ssm", "conv") if hasattr(model.model.layers[layer], "linear_attn")
                                     else ("key", "value")))
        check(set(handle.keys()) == expected, "native capture key coverage")
        for layer in source["layers"]:
            module = model.model.layers[layer]
            names = (*boundaries, *(["state_ssm", "state_conv"] if hasattr(module, "linear_attn")
                                   else ["state_key", "state_value"]))
            weights = {}
            for name in module.state_dict():
                key = f"model.layers.{layer}.{name}"
                with safe_open(args.model / index[key], framework="pt", device="cpu") as shard:
                    weights[name] = shard.get_tensor(key).to("cuda")
            report = module.load_state_dict(weights, strict=True, assign=True)
            check(not report.missing_keys and not report.unexpected_keys
                  and all(not p.is_meta for p in module.parameters()), "unclean native operand module load")
            module.eval()
            print(f"CLOVER_NATIVE_LOAD layer={layer} keys={len(weights)} missing=0 unexpected=0 meta=0", flush=True)
            x = handle.get_tensor(f"layer{layer}.input").to("cuda")
            check(x.shape == (1, 32, 2048) and x.dtype == torch.bfloat16
                  and bool(torch.isfinite(x).all()), "native input layout/nonfinite")
            snapshot = x.clone()
            bf, _ = layer_run(module, x, None, 0, False)
            again, _ = layer_run(module, x, None, 0, False)
            check(torch.equal(x, snapshot), "definition mutated native input")
            wide = copy.deepcopy(module).float()
            fp, _ = layer_run(wide, x.float(), None, 0, False)
            for name in names:
                native = handle.get_tensor(f"layer{layer}.{name}").to("cuda")
                check(native.shape == bf[name].shape and native.dtype == bf[name].dtype,
                      "native output layout/dtype: " + name)
                record = {"layer": layer, "metric": name, "native_vs_bf": distance(native, bf[name]),
                          "native_vs_fp": distance(native, fp[name]),
                          "bf_vs_fp": distance(bf[name], fp[name]), "repeat": distance(bf[name], again[name])}
                records.append(record)
                print(f"CLOVER_NATIVE_OPERAND_TRACK {json.dumps(record)}", flush=True)
                if name in ("output", "state_ssm"):
                    for label, result in (("bf", bf), ("fp", fp)):
                        tensors[f"layer{layer}.{name}.{label}"] = result[name].cpu().contiguous().clone()
            if layer == 0:
                changed = x.clone()
                coordinate = int(changed[0, 0].abs().argmax())
                check(float(changed[0, 0, coordinate]) != 0.0, "input control has zero signal")
                changed[0, 0, coordinate] = -changed[0, 0, coordinate]
                control, _ = layer_run(module, changed, None, 0, False)
                detected = not torch.equal(control["output"], bf["output"]) and not torch.equal(control["state_ssm"], bf["state_ssm"])
                check(detected, "native input mutation control escaped")
                controls += int(detected)
                print(f"CLOVER_NATIVE_OPERAND_CONTROL layer={layer} row=0 channel={coordinate} detected={detected}", flush=True)
                del changed, control
            del weights, x, snapshot, bf, again, fp, wide, native, result
            module.to("meta")
            torch.cuda.empty_cache()
    save_file(tensors, args.output / "replay.safetensors")
    with (args.output / "distances.json").open("x") as handle:
        json.dump({"revision": REVISION, "records": records, "controls": controls,
                   "diagnostic_only": True, "r2_changed": False}, handle, indent=2)
        handle.write("\n")
    check(controls == 1, "native operand control coverage")
    print(f"CLOVER_NATIVE_OPERAND_REPLAY PASS layers=21 metrics={len(records)} controls={controls}/1 "
          f"diagnostic_only=true r2_changed=false max_memory_allocated={torch.cuda.max_memory_allocated()}", flush=True)


@torch.inference_mode()
def full_fixtures(args):
    check(args.model is not None, "full oracle requires the published checkpoint")
    check(not args.output.exists(), "full output must not exist")
    args.output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    previous_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    print(f"CLOVER_FULL_ACCUMULATION previous_reduced_precision={previous_reduction} "
          "allow_bf16_reduced_precision_reduction=False", flush=True)
    model, index = schema(args.model, args.definition)
    consumed, tensors = set(), {}
    table = {"revision": REVISION, "schema_keys": 813, "torch": torch.__version__,
             "device": torch.cuda.get_device_name(), "rows": 32,
             "scope": "all40 layers; causal prefill, one preview, causal replay",
             "acceptance": "reference_only",
             "allow_bf16_reduced_precision_reduction": False,
             "route_acceptance": "2026-10-31 matched-input IDs with margin exemptions; ID-aligned weights",
             "cross_device": "NOT RUN", "full_cases": [], "head_cases": []}

    def load(module, prefix):
        weights = {}
        for name in module.state_dict():
            key = prefix + name
            with safe_open(args.model / index[key], framework="pt", device="cpu") as handle:
                weights[name] = handle.get_tensor(key).to("cuda")
            check(key not in consumed, "duplicate full-model weight consumption: " + key)
            consumed.add(key)
        report = module.load_state_dict(weights, strict=True, assign=True)
        check(not report.missing_keys and not report.unexpected_keys
              and all(not p.is_meta for p in module.parameters()), "unclean full-model module load")
        module.eval()
        return module

    def save(name, value):
        tensors[name] = value.detach().cpu().contiguous().clone()

    def observe(tag, captures, metadata, destination):
        bf, again, shape, fp = captures
        noise, bounds = {}, {}
        for name, value in bf.items():
            if name not in ("output", "route_logits", "ids", "weights") and not name.startswith("state_"):
                continue
            save(f"{tag}.bf.{name}", value.int() if name == "ids" else value)
            save(f"{tag}.fp.{name}", fp[name].int() if name == "ids" else fp[name].float())
            if name == "ids":
                continue
            noise[name] = {"repeat": distance(value, again[name]),
                           "shape": distance(value, shape[name][:value.shape[0]]),
                           "bf16_fp32": distance(value, fp[name])}
            bounds[name] = {key: 2 * max(item[key] for item in noise[name].values())
                            for key in ("max_abs", "relative_l2")}
        entry = {"tag": tag, **metadata, "noise": noise, "bounds": bounds}
        if "ids" in bf:
            ranked = bf["route_logits"].float().sort(descending=True).values
            gap = ranked[..., 7] - ranked[..., 8]
            entry.update({"experts": bf["ids"].unique(sorted=True).tolist(),
                          "zero_fillers": int((bf["weights"] == 0).sum()),
                          "route_gap_min": float(gap.min()),
                          "route_ties": int((gap == 0).sum()),
                          "wide_route_changes": int((bf["ids"] != fp["ids"]).sum())})
        destination.append(entry)
        print(f"CLOVER_FULL_REFERENCE_CASE {json.dumps(entry)} PASS", flush=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True)
    text = "Explain how sunlight, water and carbon dioxide help a plant grow. Give a clear scientific explanation. "
    ids = tokenizer.encode(text * 6, add_special_tokens=False)
    check(len(ids) >= 64, "teacher-forced prompt too short")
    prompt = torch.tensor([ids[:32]], device="cuda")
    commit = torch.tensor([ids[32:64]], device="cuda")
    preview = commit.clone()
    preview[:, torch.arange(32, device="cuda") % 3 != 0] = model.config.mask_token_id
    phases = [("prefill", prompt, 0, False), ("preview", preview, 32, True),
              ("commit", commit, 32, False)]
    embedding = load(model.model.embed_tokens, "model.embed_tokens.")
    hidden = {}
    for name, tokens, _, _ in phases:
        save(name + ".tokens", tokens.int())
        value = embedding(tokens)
        save(name + ".input", value)
        hidden[name] = [value, value.clone(), value.repeat(2, 1, 1), value.float()]
    embedding.to("meta")
    torch.cuda.empty_cache()
    for layer, module in enumerate(model.model.layers):
        load(module, f"model.layers.{layer}.")
        wide = copy.deepcopy(module).float()
        caches = [None] * 4
        for name, _tokens, prefix, preview_mode in phases:
            outputs, next_caches = [], []
            for arm, x in enumerate(hidden[name]):
                previous = caches[arm]
                snapshots = None if previous is None else {k: v.clone() for k, v in previous.items()}
                result, next_cache = layer_run(wide if arm == 3 else module, x, previous, prefix, preview_mode)
                if snapshots is not None:
                    check(all(torch.equal(v, previous[k]) for k, v in snapshots.items()),
                          "full reference mutated input cache")
                    if preview_mode:
                        if hasattr(module, "linear_attn"):
                            check(all(torch.equal(v, next_cache[k]) for k, v in snapshots.items()),
                                  "full reference preview changed GDN state")
                        else:
                            check(all(torch.equal(v, next_cache[k][:, :, :prefix])
                                      for k, v in snapshots.items()),
                                  "full reference preview changed attention prefix")
                outputs.append(result)
                next_caches.append(next_cache)
            observe(f"{name}.layer{layer}", outputs,
                    {"layer": layer, "phase": name, "prefix": prefix, "preview": preview_mode},
                    table["full_cases"])
            hidden[name] = [result["output"] for result in outputs]
            if name == "prefill":
                caches = next_caches
            del outputs, next_caches, result, previous, snapshots
        del wide, caches
        module.to("meta")
        torch.cuda.empty_cache()
        print(f"CLOVER_FULL_LAYER_LOAD layer={layer} clean=true consumed={len(consumed)}", flush=True)
    norm = load(model.model.norm, "model.norm.")
    head = load(model.lm_head, "lm_head.")
    wide_norm, wide_head = copy.deepcopy(norm).float(), copy.deepcopy(head).float()
    for name, _tokens, prefix, preview_mode in phases:
        results = []
        for arm, x in enumerate(hidden[name]):
            value = wide_head(wide_norm(x)) if arm == 3 else head(norm(x))
            results.append({"output": value.float()})
        observe(name + ".logits", results, {"phase": name, "prefix": prefix, "preview": preview_mode},
                table["head_cases"])
    check(consumed == set(index), "full model has unconsumed published weights")
    table["consumed_keys"] = len(consumed)
    save_file(tensors, args.output / "full.safetensors")
    with (args.output / "thresholds.json").open("x") as handle:
        json.dump(table, handle, indent=2); handle.write("\n")
    print(f"CLOVER_FULL_REFERENCE PASS layers=40 phases=3 keys={len(consumed)} missing=0 unexpected=0 "
          f"candidate_observations=0 max_memory_allocated={torch.cuda.max_memory_allocated()}", flush=True)


def sampling_fixtures(args):
    import inspect
    import sys
    from types import SimpleNamespace

    args.output.mkdir(parents=True, exist_ok=False)
    if args.model is not None:
        model, _ = schema(args.model, args.definition)
        generation = json.loads((args.model / "generation_config.json").read_text())
    else:
        model = declared_model(args.definition)
        reference_table = json.loads((args.replay / "thresholds.json").read_text())
        check(reference_table["revision"] == REVISION and len(model.state_dict()) == 813,
              "sampling replay definition identity")
        generation = reference_table["generation"]
    module = importlib.import_module(type(model).__module__)
    sampling = importlib.import_module(module.__package__ + ".sampling_clover_moe")
    required = ("presence_penalty", "frequency_penalty", "repetition_penalty",
                "dllm_length_penalty", "dllm_length_penalty_start", "dllm_length_penalty_window",
                "dllm_think_end_bias", "dllm_think_end_start", "dllm_think_end_window",
                "dllm_think_end_token_id", "pad_token_id", "eos_token_id")
    check(all(key in generation for key in required), "publisher generation parameters missing")
    gen = transformers.GenerationConfig.from_dict(generation)
    sampling.validate_penalties(gen, model.config.vocab_size)
    rows, vocab, mask = model.config.block_size, model.config.vocab_size, model.config.mask_token_id
    stops = sorted(set(gen.eos_token_id) | set(gen.stop_token_ids))
    tensors = {}
    table = {"revision": REVISION, "schema_keys": 813, "generation": generation,
             "device": torch.cuda.get_device_name(), "bounds": {"logit_bits": 0, "ids": 0,
             "pending": 0, "counts": 0, "returned": 0}, "penalty_cases": [], "policy_cases": []}
    columns = torch.arange(vocab, device="cuda", dtype=torch.int64)
    raw = (((columns * 73 + 31) % 1009).float() / 257 - 18).repeat(rows, 1)
    for r in range(rows):
        raw[r, 10000 + r * 37] = 12 if r % 3 == 0 else 4
        raw[r, 15000 + r * 41] = raw[r, 10000 + r * 37] - (8 if r % 3 == 0 else 0)
    raw[:, mask] = 50
    tensors["raw"] = raw.cpu()
    pool = torch.tensor([10000 + r * 37 for r in range(rows)] + [17, 19, 37, 43, 69, 101, 103, 137, 197], device="cuda")

    def sparse_counts(tag, counts):
        ids = (counts != 0).nonzero().flatten()
        tensors[tag + ".ids"] = ids.int().cpu()
        tensors[tag + ".values"] = counts[ids].int().cpu()

    cases = [("history", 127, False, False), ("empty", 0, False, False),
             ("close_edge", 4095, False, False), ("close_after", 4096, False, False),
             ("stop_edge", 8191, False, False), ("stop_after", 8192, False, False),
             ("stop_end", 16383, False, False), ("saturated", 16384, False, False),
             ("close_visible", 8192, True, False), ("close_committed", 8192, False, True)]
    for tag, length, visible, committed in cases:
        state = sampling.CloverPenaltyState(gen, vocab, "cuda", stops + [gen.pad_token_id])
        history = pool[(torch.arange(length, device="cuda") * 73) % pool.numel()]
        if committed:
            history[-1] = gen.dllm_think_end_token_id
        state.commit(history)
        block = torch.full((1, rows), mask, device="cuda", dtype=torch.int64)
        if visible:
            block[0, 17] = gen.dllm_think_end_token_id
        block[0, 4] = 211
        pending = block[0] == mask
        logits = raw.clone()
        logits[:, mask] = -float("inf")
        state.apply(logits, block, mask)
        prediction, confidence = sampling.predict_with_confidence(logits, gen)
        repeat = raw.clone(); repeat[:, mask] = -float("inf")
        state.apply(repeat, block, mask)
        check(torch.equal(logits.view(torch.int32), repeat.view(torch.int32)), "penalty repeat bits")
        narrow = raw[:3].clone(); narrow[:, mask] = -float("inf")
        state.apply(narrow, block, mask)
        check(torch.equal(logits[:3].view(torch.int32), narrow.view(torch.int32)), "penalty shape bits")
        changed = (logits.view(torch.int32).flatten() != raw.view(torch.int32).flatten()).nonzero().flatten()
        tensors[tag + ".patch_ids"] = changed.int().cpu()
        tensors[tag + ".patch_values"] = logits.flatten()[changed].cpu()
        tensors[tag + ".block"] = block.flatten().int().cpu()
        tensors[tag + ".pending"] = pending.int().cpu()
        tensors[tag + ".picks"] = prediction.int().cpu()
        tensors[tag + ".confidence"] = confidence.cpu()
        sparse_counts(tag + ".counts", state.counts)
        ranked = logits.topk(2, dim=-1).values
        controls = []
        if tag == "history":
            controls = [{"name": "presence_off", "field": "presence_penalty", "value": 0.0},
                        {"name": "frequency_on", "field": "frequency_penalty", "value": 0.25},
                        {"name": "repetition_off", "field": "repetition_penalty", "value": 1.0},
                        {"name": "uncommitted_counted", "count_id": 211}]
        if tag == "empty":
            controls = [{"name": "prompt_counted", "count_id": 17}, {"name": "mask_unblocked", "mask": 151670}]
        if tag == "close_edge":
            controls = [{"name": "close_ramp_shift", "field": "dllm_think_end_start", "value": 4097}]
        if tag == "stop_edge":
            controls = [{"name": "stop_ramp_shift", "field": "dllm_length_penalty_start", "value": 8193}]
        if visible:
            controls = [{"name": "visible_close_ignored", "visible": False}]
        if committed:
            controls = [{"name": "committed_close_ignored", "clear_close": True}]
        for control in controls:
            mutated = copy.deepcopy(state)
            altered_block = block.clone()
            altered_mask = control.get("mask", mask)
            if "field" in control:
                setattr(mutated.gen, control["field"], control["value"])
            if "count_id" in control:
                mutated.counts[control["count_id"]] += 1
            if control.get("clear_close"):
                mutated.counts[gen.dllm_think_end_token_id] = 0
            if "visible" in control:
                altered_block[altered_block == gen.dllm_think_end_token_id] = mask
            altered = raw.clone(); altered[:, altered_mask] = -float("inf")
            mutated.apply(altered, altered_block, altered_mask)
            control["different_bits"] = int((altered.view(torch.int32) != logits.view(torch.int32)).sum())
            check(control["different_bits"] > 0, "penalty control escaped: " + control["name"])
        table["penalty_cases"].append({"tag": tag, "length": length, "controls": controls,
            "top2_margin": float((ranked[:, 0] - ranked[:, 1]).min()),
            "confidence_margin": float((confidence - 0.95).abs().min())})

    function = inspect.unwrap(type(model).generate)
    source, first = inspect.getsourcelines(function)
    fill_line = first + next(i for i, line in enumerate(source) if "if not bool(pending.any()):" in line)
    commit_line = first + next(i for i, line in enumerate(source) if "if len(response) < maximum and not stopped:" in line)
    modes = [("strict", 32), ("force_tie", 32), ("early_eos", 32),
             ("limit1", 1), ("limit31", 31), ("limit32", 32), ("limit33", 33)]
    threshold = torch.tensor(0.95, device="cuda")
    for tag, maximum in modes:
        events = []
        calls = 0
        prompt_calls = 0
        prompt = torch.tensor([[mask, 17, 17]], device="cuda")
        def mixer_forward(_module, ids, **_kwargs):
            nonlocal prompt_calls
            prompt_calls += 1
            if prompt_calls == 1:
                check(torch.equal(ids, prompt), "literal mask prompt changed")
            return SimpleNamespace(past_key_values=None)
        def forward(_module, ids, **_kwargs):
            return SimpleNamespace(logits=raw.unsqueeze(0).clone())
        def predictions(_logits, _gen):
            nonlocal calls
            ids = 10000 + torch.arange(rows, device="cuda") * 37
            confidence = torch.ones(rows, device="cuda")
            if tag == "strict" and calls == 0:
                confidence[:3] = torch.stack([threshold, torch.nextafter(threshold, torch.tensor(1., device="cuda")),
                                              torch.nextafter(threshold, torch.tensor(0., device="cuda"))])
            if tag == "force_tie" and calls < 2:
                confidence.fill_(0.6); confidence[1:4] = 0.7
                if calls == 1:
                    confidence[1] = 1.0
            if tag == "early_eos":
                ids[17] = stops[0]
                if calls == 0:
                    confidence.fill_(0.4); confidence[17] = 1.0
            calls += 1
            tensors[f"{tag}.step{calls}.picks"] = ids.int().cpu()
            tensors[f"{tag}.step{calls}.confidence"] = confidence.cpu()
            return ids, confidence
        def trace(frame, event, _arg):
            if frame.f_code is function.__code__ and event == "line":
                values = frame.f_locals
                if frame.f_lineno == fill_line:
                    key = f"{tag}.step{calls}"
                    tensors[key + ".block"] = values["block"].flatten().int().cpu().clone()
                    tensors[key + ".pending"] = values["pending"].flatten().int().cpu().clone()
                    events.append({"kind": "fill", "key": key, "length": values["penalties"].length})
                if frame.f_lineno == commit_line:
                    key = f"{tag}.commit{len([e for e in events if e['kind'] == 'commit'])}"
                    state = values["penalties"]
                    sparse_counts(key + ".counts", state.counts)
                    events.append({"kind": "commit", "key": key, "remaining": values["remaining"],
                                   "length": state.length, "returned": values["i"] + 1,
                                   "stopped": values["stopped"]})
            return trace
        with patch.object(type(model.model), "forward", mixer_forward), patch.object(type(model), "forward", forward), \
             patch.object(module, "predict_with_confidence", predictions):
            sys.settrace(trace)
            try:
                result = model.generate(prompt, generation_config=gen, max_new_tokens=maximum)
            finally:
                sys.settrace(None)
        check(events and events[-1]["kind"] == "commit", "definition policy trace missing")
        check(prompt_calls >= 1, "definition prompt prefill not observed")
        tensors[tag + ".returned"] = result[0, prompt.numel():].int().cpu()
        table["policy_cases"].append({"tag": tag, "maximum": maximum, "events": events,
                                       "prompt_literal_mask": True, "forward": "fixed logits; confidence injection"})
    if args.replay is not None:
        with safe_open(args.replay / "sampling.safetensors", framework="pt", device="cpu") as handle:
            check(set(handle.keys()) == set(tensors), "sampling peer fixture keys differ")
            for name, value in tensors.items():
                expected = handle.get_tensor(name)
                if name.endswith(".confidence"):
                    print(f"CLOVER_SAMPLE_NOISE {name} max_abs={(value - expected).abs().max().item()}", flush=True)
                else:
                    check(torch.equal(value.view(torch.uint8), expected.view(torch.uint8)),
                          "sampling peer reference bits: " + name)
        table["cross_device"] = reference_table["device"]
    save_file(tensors, args.output / "sampling.safetensors")
    with (args.output / "thresholds.json").open("x") as handle:
        json.dump(table, handle, indent=2); handle.write("\n")
    print(f"CLOVER_SAMPLE_REFERENCE PASS penalty_cases={len(table['penalty_cases'])} "
          f"policy_cases={len(table['policy_cases'])} bounds=exact candidate_observations=0", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--layer", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--generation-replay", action="store_true")
    parser.add_argument("--commit-operands", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--policy-reference", type=Path)
    parser.add_argument("--device-budget-gib", type=int)
    parser.add_argument("--reduction", choices=("original", "strict"), default="strict")
    parser.add_argument("--boundaries", action="store_true")
    parser.add_argument("--native-operands", action="store_true")
    parser.add_argument("--trajectory-reference", type=Path, nargs=4)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    check(not arguments.resume or (arguments.generate and arguments.trajectory_reference is None),
          "resume requires the generation reference recorder")
    check(arguments.commit_operands or arguments.generation_replay or arguments.generate or arguments.boundaries or arguments.native_operands or (arguments.model is None) != (arguments.replay is None), "select model fixtures or boundary replay")
    check(sum((arguments.moe, arguments.layer, arguments.sample, arguments.full, arguments.generate, arguments.generation_replay, arguments.commit_operands, arguments.boundaries, arguments.native_operands)) <= 1, "select one oracle boundary")
    if arguments.commit_operands:
        action = commit_operands
    elif arguments.generation_replay:
        action = generation_replay
    elif arguments.generate:
        action = generation_bounds if arguments.trajectory_reference is not None else generation_fixtures
    elif arguments.native_operands:
        action = native_operands
    elif arguments.boundaries:
        action = full_boundaries
    elif arguments.full:
        action = full_fixtures if arguments.model is not None else full_route_replay
    elif arguments.sample:
        action = sampling_fixtures
    elif arguments.layer:
        action = layer_fixtures if arguments.model is not None else layer_replay
    else:
        action = (moe_fixtures if arguments.model is not None else moe_replay) if arguments.moe else (fixtures if arguments.model is not None else replay)
    action(arguments)
