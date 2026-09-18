#!/usr/bin/env python3
"""Single-layer reference harness: official-op GDN/attention/MoE over
byte-verified dequantized weights, diffed against the Rust layer dump."""

import json
import math
import struct

import numpy as np
import torch

ROOT = "models/Edge0/Edge0-35B-A3B-preview"
P = "language_model.model"
CFG = json.load(open(f"{ROOT}/config.json"))["text_config"]


def load_headers():
    tensors = {}
    for i in range(1, 5):
        path = f"{ROOT}/model-0000{i}-of-00004.safetensors"
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            base = 8 + n
        for k, v in hdr.items():
            if k != "__metadata__":
                tensors[k] = (path, base + v["data_offsets"][0], v["dtype"], v["shape"])
    return tensors


T = load_headers()


def raw(name, count=None, offset=0):
    path, off, _, _ = T[name]
    with open(path, "rb") as f:
        f.seek(off + offset)
        return f.read(count)


def bf16(name):
    count = 1
    for d in T[name][3]:
        count *= d
    data = raw(name, count * 2)
    u32 = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).copy()
    return torch.from_numpy(u32).view(torch.float32)


def dequant(name):
    base = name[:-7] if name.endswith(".weight") else name
    _, _, _, shape = T[name]
    out_dim = shape[0]
    groups = T[base + ".scales"][3][-1]
    in_dim = groups * 64
    bits = 8 if shape[-1] * 4 == in_dim else 4
    per_word = 32 // bits
    data = raw(name, out_dim * shape[-1] * 4)
    words = torch.from_numpy(np.frombuffer(data, dtype="<u4").astype(np.int64))
    words = words.reshape(out_dim, in_dim // per_word)
    elems = []
    shift = 4 if bits == 4 else 8
    for j in range(per_word):
        elems.append(((words >> (shift * j)) & ((1 << shift) - 1)).float())
    q = torch.stack(elems, dim=2).reshape(out_dim, in_dim)
    scales = cached_bf16(base + ".scales").reshape(out_dim, groups)
    biases = cached_bf16(base + ".biases").reshape(out_dim, groups)
    w = scales.repeat_interleave(64, dim=1) * q + biases.repeat_interleave(64, dim=1)
    return w.contiguous()


LORA = {}
with open(f"{ROOT}/lora_edge0_35b.safetensors", "rb") as f:
    _n = struct.unpack("<Q", f.read(8))[0]
    _hdr = json.loads(f.read(_n))
    _base = 8 + _n
    for _k, _v in _hdr.items():
        if _k == "__metadata__":
            continue
        _cnt = 1
        for _d in _v["shape"]:
            _cnt *= _d
        f.seek(_base + _v["data_offsets"][0])
        _data = f.read(_cnt * 2)
        _vals = torch.tensor(
            [struct.unpack("<e", _data[i * 2 : i * 2 + 2])[0] for i in range(_cnt)],
            dtype=torch.float32,
        )
        LORA[_k] = _vals.reshape(_v["shape"])


# Cache static projections and lm_head only (~8 GiB as f32); caching the
# expert pool too would expand 18.17 GiB int4 to ~145 GiB f32 — the first
# cache attempt was OOM-killed for exactly that. Expert tensors are small
# (512 rows) and re-dequantize per routing hit.
_CACHE = {}


def cached_dequant(name):
    if ".switch_mlp." in name:
        return dequant(name)
    if name not in _CACHE:
        _CACHE[name] = dequant(name)
    return _CACHE[name]


_BF16_CACHE = {}


def cached_bf16(name):
    if name not in _BF16_CACHE:
        _BF16_CACHE[name] = bf16(name)
    return _BF16_CACHE[name]


def proj(name, x):
    """Dequantized projection + on-the-fly LoRA, matching the Rust path."""
    out = cached_dequant(f"{name}.weight") @ x
    if f"{name}.lora_A" in LORA:
        out = out + LORA[f"{name}.lora_B"] @ (LORA[f"{name}.lora_A"] @ x)
    return out


def rmsnorm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def silu(x):
    return x / (1 + torch.exp(-x))


def gdn_layer(x, layer):
    num_v = CFG["linear_num_value_heads"]
    num_k = CFG["linear_num_key_heads"]
    dk, dv = CFG["linear_key_head_dim"], CFG["linear_value_head_dim"]
    key_dim, value_dim = num_k * dk, num_v * dv
    conv_dim = 2 * key_dim + value_dim
    prefix = f"{P}.layers.{layer}.linear_attn"
    qkv = proj(f"{prefix}.in_proj_qkv", x)
    # conv kernel 4 with zero state: only the last tap sees this token
    conv_w = cached_bf16(f"{prefix}.conv1d.weight").reshape(conv_dim, 4)
    conv_out = conv_w[:, -1] * qkv  # single token, state zero -> last tap only
    conv_out = silu(conv_out)
    q_all, k_all, v_all = conv_out[:key_dim], conv_out[key_dim:2 * key_dim], conv_out[2 * key_dim:]
    z = proj(f"{prefix}.in_proj_z", x)
    b = proj(f"{prefix}.in_proj_b", x)
    a = proj(f"{prefix}.in_proj_a", x)
    dt_bias = cached_bf16(f"{prefix}.dt_bias")
    a_log = cached_bf16(f"{prefix}.A_log")
    norm_w = cached_bf16(f"{prefix}.norm.weight")
    out = torch.zeros(value_dim)
    scale = 1.0 / math.sqrt(dk)
    S = torch.zeros(num_v, dk, dv)
    for head in range(num_v):
        k_head = head // (num_v // num_k)
        q = torch.nn.functional.normalize(q_all[k_head * dk:(k_head + 1) * dk], dim=0, eps=1e-6)
        k = torch.nn.functional.normalize(k_all[k_head * dk:(k_head + 1) * dk], dim=0, eps=1e-6)
        v = v_all[head * dv:(head + 1) * dv]
        g = -a_log[head].exp() * torch.nn.functional.softplus(a[head] + dt_bias[head])
        beta = torch.sigmoid(b[head])
        kv_mem = S[head].t() @ k
        delta = (v - kv_mem) * beta
        S[head] += k.unsqueeze(1) * delta.unsqueeze(0)
        out[head * dv:(head + 1) * dv] = S[head].t() @ (q * scale)
        head_out = out[head * dv:(head + 1) * dv]
        z_h = z[head * dv:(head + 1) * dv]
        normed = head_out * torch.rsqrt(head_out.pow(2).mean() + 1e-6) * norm_w
        out[head * dv:(head + 1) * dv] = normed * silu(z_h)
    return proj(f"{prefix}.out_proj", out)


def attention_layer(x, layer, position=0):
    prefix = f"{P}.layers.{layer}.self_attn"
    heads, kv_heads, head_dim = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"]
    rotary_dim = int(head_dim * CFG["partial_rotary_factor"])
    theta = CFG["rope_parameters"]["rope_theta"]
    q_raw = proj(f"{prefix}.q_proj", x)
    k_raw = proj(f"{prefix}.k_proj", x)
    v_raw = proj(f"{prefix}.v_proj", x)
    q_norm = cached_bf16(f"{prefix}.q_norm.weight")
    k_norm = cached_bf16(f"{prefix}.k_norm.weight")
    query = torch.zeros(heads * head_dim)
    gate = torch.zeros(heads * head_dim)
    for h in range(heads):
        base = h * head_dim * 2
        query[h * head_dim:(h + 1) * head_dim] = q_raw[base:base + head_dim]
        gate[h * head_dim:(h + 1) * head_dim] = q_raw[base + head_dim:base + 2 * head_dim]
    out = torch.zeros(heads * head_dim)
    for h in range(heads):
        kv_h = h // (heads // kv_heads)
        q = rmsnorm(query[h * head_dim:(h + 1) * head_dim], q_norm)
        k = rmsnorm(k_raw[kv_h * head_dim:(kv_h + 1) * head_dim], k_norm)
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
        score = (q @ k) / math.sqrt(head_dim)
        del score  # single position: softmax of one logit is exactly 1
        out[h * head_dim:(h + 1) * head_dim] = v
    out = out * torch.sigmoid(gate)
    return proj(f"{prefix}.o_proj", out)


def moe_layer(x, layer, top_k=4):
    prefix = f"{P}.layers.{layer}.mlp"
    router = dequant(f"{prefix}.gate.weight") @ x
    top = torch.topk(router, top_k)
    weights = torch.softmax(top.values, dim=-1)
    out = torch.zeros(x.shape[0])
    for slot, expert in enumerate(top.indices.tolist()):
        base = f"{P}.layers.{layer}.mlp.switch_mlp"
        gate_w = dequant_stacked(base, "gate_proj", expert)
        up_w = dequant_stacked(base, "up_proj", expert)
        down_w = dequant_stacked(base, "down_proj", expert)
        g = gate_w @ x
        u = up_w @ x
        inner = silu(g) * u
        out += weights[slot] * (down_w @ inner)
    shared_gate = proj(f"{prefix}.shared_expert_gate", x)
    sg = proj(f"{prefix}.shared_expert.gate_proj", x)
    su = proj(f"{prefix}.shared_expert.up_proj", x)
    shared = proj(f"{prefix}.shared_expert.down_proj", silu(sg) * su)
    return out + torch.sigmoid(shared_gate) * shared


# Expert dequants stay uncached (30720 tensors x 4 MiB f32 = ~120 GiB if
# all cached); each is a few tensor ops on 512x256 words.
def dequant_stacked(base, proj, expert):
    return _dequant_stacked_impl(base, proj, expert)


def _dequant_stacked_impl(base, proj, expert):
    name = f"{base}.{proj}.weight"
    _, _, _, shape = T[name]
    rows, packed_cols = shape[1], shape[2]
    in_dim = packed_cols * 8
    groups = in_dim // 64
    word_bytes = rows * packed_cols * 4
    words = torch.from_numpy(
        np.frombuffer(raw(name, word_bytes, offset=expert * word_bytes), dtype="<u4").astype(np.int64)
    ).reshape(rows, packed_cols)
    elems = []
    for j in range(8):
        elems.append(((words >> (4 * j)) & 0xF).float())
    q = torch.stack(elems, dim=2).reshape(rows, in_dim)
    scales = cached_bf16(f"{base}.{proj}.scales")[expert * rows * groups:(expert + 1) * rows * groups].reshape(rows, groups)
    biases = cached_bf16(f"{base}.{proj}.biases")[expert * rows * groups:(expert + 1) * rows * groups].reshape(rows, groups)
    w = scales.repeat_interleave(64, dim=1) * q + biases.repeat_interleave(64, dim=1)
    return w.contiguous()


def main():
    dump = json.load(open("output/edge0-layer-dump.json"))
    state = 12345
    x = torch.zeros(CFG["hidden_size"])
    for i in range(len(x)):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        x[i] = ((state >> 33) / (2**32 - 1) - 0.5) * 0.2
    assert torch.allclose(x, torch.tensor(dump["input"], dtype=torch.float32), atol=1e-7)
    eps = CFG["rms_norm_eps"]
    for layer in (0, 1, 2, 3):
        entry = dump[f"layer{layer}"]
        prefix = f"{P}.layers.{layer}"
        normed = rmsnorm(x, cached_bf16(f"{prefix}.input_layernorm.weight"), eps)
        rust_normed = torch.tensor(entry["normed"])
        print(f"layer {layer} [{entry['kind']}] normed maxdiff: {(normed - rust_normed).abs().max():.6f}")
        block = gdn_layer(normed, layer) if entry["kind"] == "gdn" else attention_layer(normed, layer)
        rust_block = torch.tensor(entry["attn"])
        diff = (block - rust_block).abs().max()
        rel = diff / block.abs().max().clamp(min=1e-9)
        print(f"  attn maxdiff {diff:.6f} (rel {rel:.4f})  |py|={block.abs().max():.3f} |rs|={rust_block.abs().max():.3f}")
        residual = x + block
        moe_in = rmsnorm(residual, cached_bf16(f"{prefix}.post_attention_layernorm.weight"), eps)
        moe = moe_layer(moe_in, layer)
        print(f"  moe  |py|={moe.abs().max():.3f}")


if __name__ == "__main__":
    main()
