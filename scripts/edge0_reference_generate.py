#!/usr/bin/env python3
"""Greedy generation through the Python reference layers; emits token ids
for the Rust token-id comparison (the operational numeric acceptance)."""

import json
import math
import sys

import torch

sys.argv = ["x"]
exec(open("scripts/edge0_layer_reference.py").read().split("def main()")[0])

LAYERS = CFG["num_hidden_layers"]
EPS = CFG["rms_norm_eps"]
TOP_K = 4


class GdnState:
    def __init__(self):
        conv_dim = 2 * CFG["linear_num_key_heads"] * CFG["linear_key_head_dim"] + CFG["linear_num_value_heads"] * CFG["linear_value_head_dim"]
        self.conv = torch.zeros(conv_dim, 3)
        self.s = torch.zeros(CFG["linear_num_value_heads"], CFG["linear_key_head_dim"], CFG["linear_value_head_dim"])


def kind(layer):
    return "gdn" if (layer + 1) % CFG["full_attention_interval"] != 0 else "attn"


def gdn_step(state, x, layer):
    prefix = f"{P}.layers.{layer}.linear_attn"
    qkv = proj(f"{prefix}.in_proj_qkv", x)
    conv_w = bf16(f"{prefix}.conv1d.weight").reshape(-1, 4)
    conv_in = torch.cat([state.conv, qkv.unsqueeze(1)], dim=1)
    state.conv = conv_in[:, 1:].clone()
    conv_out = silu((conv_w * conv_in).sum(dim=1))
    key_dim = CFG["linear_num_key_heads"] * CFG["linear_key_head_dim"]
    num_v, num_k = CFG["linear_num_value_heads"], CFG["linear_num_key_heads"]
    dk, dv = CFG["linear_key_head_dim"], CFG["linear_value_head_dim"]
    q_all, k_all, v_all = conv_out[:key_dim], conv_out[key_dim:2 * key_dim], conv_out[2 * key_dim:]
    z = proj(f"{prefix}.in_proj_z", x)
    b = proj(f"{prefix}.in_proj_b", x)
    a = proj(f"{prefix}.in_proj_a", x)
    dt_bias = bf16(f"{prefix}.dt_bias")
    a_log = bf16(f"{prefix}.A_log")
    norm_w = bf16(f"{prefix}.norm.weight")
    out = torch.zeros(num_v * dv)
    scale = 1.0 / math.sqrt(dk)
    for head in range(num_v):
        k_head = head // (num_v // num_k)
        q = torch.nn.functional.normalize(q_all[k_head * dk:(k_head + 1) * dk], dim=0, eps=1e-6)
        k = torch.nn.functional.normalize(k_all[k_head * dk:(k_head + 1) * dk], dim=0, eps=1e-6)
        v = v_all[head * dv:(head + 1) * dv]
        g = -a_log[head].exp() * torch.nn.functional.softplus(a[head] + dt_bias[head])
        beta = torch.sigmoid(b[head])
        state.s[head] *= g.exp()
        kv_mem = state.s[head].t() @ k
        delta = (v - kv_mem) * beta
        state.s[head] += k.unsqueeze(1) * delta.unsqueeze(0)
        head_out = state.s[head].t() @ (q * scale)
        z_h = z[head * dv:(head + 1) * dv]
        normed = head_out * torch.rsqrt(head_out.pow(2).mean() + EPS) * norm_w
        out[head * dv:(head + 1) * dv] = normed * silu(z_h)
    return proj(f"{prefix}.out_proj", out)


class AttnState:
    def __init__(self):
        # Per-KV-head caches: head h attends only over its own kv head's
        # history, appended once per token (not once per query head).
        self.keys = [[] for _ in range(CFG["num_key_value_heads"])]
        self.values = [[] for _ in range(CFG["num_key_value_heads"])]


def attn_step(state, x, layer, position):
    prefix = f"{P}.layers.{layer}.self_attn"
    heads, kv_heads, head_dim = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"]
    rotary_dim = int(head_dim * CFG["partial_rotary_factor"])
    theta = CFG["rope_parameters"]["rope_theta"]
    q_raw = proj(f"{prefix}.q_proj", x)
    k_raw = proj(f"{prefix}.k_proj", x)
    v_raw = proj(f"{prefix}.v_proj", x)
    q_norm = bf16(f"{prefix}.q_norm.weight")
    k_norm = bf16(f"{prefix}.k_norm.weight")
    query = torch.zeros(heads * head_dim)
    gate = torch.zeros(heads * head_dim)
    for h in range(heads):
        base = h * head_dim * 2
        query[h * head_dim:(h + 1) * head_dim] = q_raw[base:base + head_dim]
        gate[h * head_dim:(h + 1) * head_dim] = q_raw[base + head_dim:base + 2 * head_dim]
    out = torch.zeros(heads * head_dim)
    for h in range(heads):
        kv_h = h // (heads // kv_heads)
        q = rmsnorm(query[h * head_dim:(h + 1) * head_dim], q_norm, EPS)
        k = rmsnorm(k_raw[kv_h * head_dim:(kv_h + 1) * head_dim], k_norm, EPS)
        v = v_raw[kv_h * head_dim:(kv_h + 1) * head_dim]
        half = rotary_dim // 2
        for i in range(half):
            freq = theta ** (-(2 * i) / rotary_dim)
            angle = position * freq
            c, s = math.cos(angle), math.sin(angle)
            q1, q2 = q[i].item(), q[i + half].item()
            k1, k2 = k[i].item(), k[i + half].item()
            q[i], q[i + half] = q1 * c - q2 * s, q2 * c + q1 * s
            k[i], k[i + half] = k1 * c - k2 * s, k2 * c + k1 * s
        if h % (heads // kv_heads) == 0:
            state.keys[kv_h].append(k)
            state.values[kv_h].append(v)
        scores = [(q @ kk).item() / math.sqrt(head_dim) for kk in state.keys[kv_h]]
        mx = max(scores)
        exps = [math.exp(s - mx) for s in scores]
        total = sum(exps)
        acc = torch.zeros(head_dim)
        for wgt, vv in zip(exps, state.values[kv_h]):
            acc += (wgt / total) * vv
        out[h * head_dim:(h + 1) * head_dim] = acc
    out = out * torch.sigmoid(gate)
    return proj(f"{prefix}.o_proj", out)


def moe_step(x, layer):
    return moe_layer(x, layer, TOP_K)


def embed_row(token):
    name = f"{P}.embed_tokens"
    groups = T[name + ".scales"][3][-1]
    scales = bf16(name + ".scales").reshape(-1, groups)[token]
    biases = bf16(name + ".biases").reshape(-1, groups)[token]
    words = torch.from_numpy(
        np.frombuffer(raw(name + ".weight", 1024, offset=token * 1024), dtype="<u4").astype(np.int64)
    )
    elems = []
    for j in range(8):
        elems.append(((words >> (4 * j)) & 0xF).float())
    q = torch.stack(elems, dim=1).reshape(2048)
    return scales.repeat_interleave(64) * q + biases.repeat_interleave(64)


def logits(hidden):
    name = "language_model.lm_head"
    groups = T[name + ".scales"][3][-1]
    rows = T[name + ".weight"][3][0]
    scales = bf16(name + ".scales").reshape(rows, groups)
    biases = bf16(name + ".biases").reshape(rows, groups)
    out = torch.zeros(rows)
    # Chunked to keep memory sane: dequant 4096 rows at a time.
    for start in range(0, rows, 4096):
        stop = min(start + 4096, rows)
        block = torch.from_numpy(
            np.frombuffer(raw(name + ".weight", (stop - start) * 1024, offset=start * 1024), dtype="<u4").astype(np.int64)
        ).reshape(stop - start, 256)
        elems = []
        for j in range(8):
            elems.append(((block >> (4 * j)) & 0xF).float())
        q = torch.stack(elems, dim=2).reshape(stop - start, 2048)
        w = scales[start:stop].repeat_interleave(64, dim=1) * q + biases[start:stop].repeat_interleave(64, dim=1)
        out[start:stop] = w @ hidden
    return out


def main():
    import os
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("models/Edge0/Edge0-35B-A3B-preview")
    prompt = os.environ.get(
        "EDGE0_REF_PROMPT", "Explain paging to a systems programmer.")
    steps = int(os.environ.get("EDGE0_REF_STEPS", "8"))
    templated = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok.encode(templated, add_special_tokens=False)
    print("prompt ids:", ids, file=sys.stderr)

    gdn_states = [GdnState() if kind(l) == "gdn" else None for l in range(LAYERS)]
    attn_states = [AttnState() if kind(l) == "attn" else None for l in range(LAYERS)]
    final_norm = bf16(f"{P}.norm.weight")

    hidden = None
    for position, t in enumerate(ids):
        hidden = embed_row(t)
        for layer in range(LAYERS):
            prefix = f"{P}.layers.{layer}"
            normed = rmsnorm(hidden, bf16(f"{prefix}.input_layernorm.weight"), EPS)
            if kind(layer) == "gdn":
                attn_out = gdn_step(gdn_states[layer], normed, layer)
            else:
                attn_out = attn_step(attn_states[layer], normed, layer, position)
            hidden = hidden + attn_out
            moe_in = rmsnorm(hidden, bf16(f"{prefix}.post_attention_layernorm.weight"), EPS)
            hidden = hidden + moe_step(moe_in, layer)
    generated = []
    step = 0
    while step < steps:
        values = logits(rmsnorm(hidden, final_norm, EPS))
        best = int(torch.argmax(values).item())
        generated.append(best)
        print(f"py token {step}: {best} ({tok.decode([best])})", file=sys.stderr)
        hidden = embed_row(best)
        position = len(ids) + step
        for layer in range(LAYERS):
            prefix = f"{P}.layers.{layer}"
            normed = rmsnorm(hidden, bf16(f"{prefix}.input_layernorm.weight"), EPS)
            if kind(layer) == "gdn":
                attn_out = gdn_step(gdn_states[layer], normed, layer)
            else:
                attn_out = attn_step(attn_states[layer], normed, layer, position)
            hidden = hidden + attn_out
            moe_in = rmsnorm(hidden, bf16(f"{prefix}.post_attention_layernorm.weight"), EPS)
            hidden = hidden + moe_step(moe_in, layer)
        step += 1
    print(json.dumps({"prompt_ids": ids, "generated": generated}))


if __name__ == "__main__":
    main()
