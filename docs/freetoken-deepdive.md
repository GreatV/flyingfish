# FreeToken implementation deepdive (code-survey round)

This document is the implementation survey of FreeToken that follows `docs/speculative-token-tech.md` §1. The previous round read the paper; this one reads the Apache-2.0 code under `https://github.com/FlashML-org/FreeToken`. **Snapshot: commit `cc1f5c2c91855f2cc7787ad6b909f7e46a5d5825` (2026-09-18, `fix(engine): preserve greedy sampling in mixed batches (#471)`).** Every claim below is anchored to a file and line; if a feature is not in the code, that is stated explicitly with the marker **未在代码中找到**.

Path convention used in this document: relative to the FreeToken repo root, prefixed with `python/` for code under the `python/` directory (which is where the main package lives) or with the bare path for top-level files such as `pyproject.toml`. The full URL anchor for a path `python/freetoken/engine/engine.py:42` is `github.com/FlashML-org/FreeToken/blob/cc1f5c2c91855f2cc7787ad6b909f7e46a5d5825/python/freetoken/engine/engine.py#L42`. No local-machine absolute paths appear in this document.

---

## §0 Module layout — Python vs the `[accel]` extension

The repo is split into two PyPI wheels:

- **Runtime wheel `freetoken`** — Python source under `python/freetoken/`, plus a small set of C++/CUDA sources under `python/freetoken/kernel/csrc/` that are compiled into the wheel by `setup.py`. Optional `[accel]` extra pulls two pre-built native kernels from PyPI.
- **Kernel-cache wheel `freetoken-kernel-cache`** — prebuilt TVM-FFI `.so` kernels, shipped as a separate wheel because PyPI cannot host `+local`-versioned files (see `release.yml:124-129`).

### §0.1 `accel` extra — what is C++/CUDA, what is Python

The `[accel]` extra resolves to two PyPI packages (no first-party C++ here):

```toml
# pyproject.toml:78-82
fi  = ["flashinfer-python[cu13]>=0.6,<0.7"]
sgl = ["sglang-kernel==0.4.5"]
accel = ["freetoken[fi,sgl]"]
```

First-party C++/CUDA (compiled by `setup.py`, present in the runtime wheel):

| File | LOC | Role |
|---|---|---|
| `python/freetoken/kernel/csrc/pinned_tensor.cpp` | 128 | `cudaHostAlloc` + `cudaHostRegister` wrappers, `cudaMallocHost`-based exact-size pinned tensors, Windows/WDDM `host_device_ptr` translation. |
| `python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp` | 2160 | CPU-side MoE compute (BF16 / NVFP4 / MXFP4 / Q4_0). |
| `python/freetoken/kernel/csrc/src/pynccl.cu` | (cu) | PyNCCL bindings for tensor-parallel comms. |
| `python/freetoken/kernel/csrc/jit/{store,index}.cu` | 123 / 172 | TVM-FFI JIT examples (also shipped as `.so` via the kernel-cache wheel). |
| `python/freetoken/kernel/csrc/gguf/gguf_kernel.cu` | (cu) | GGUF-format MoE kernel used by llama.cpp-style loads. |
| `python/freetoken/kernel/csrc/ple_store/ple_store_ext.cpp` | (cpp) | Qwen4-Exp PLE table (pinned host table backing the linear-attention state). |
| `python/freetoken/kernel/csrc/src/{radix.cpp, tensor.cpp}` | (cpp) | Radix-cache and tensor helpers. |

Triton kernels (Python, JIT-compiled with `nvcc` from PTX at first use, see `freetoken-kernel-cache/README.md:1-13`):

- `python/freetoken/kernel/triton/` — 40+ Triton files, one per MoE expert path (`fused_moe.py`, `fp8_blockscale_moe.py`, `mxfp4_moe.py`, `nvfp4_fused_moe.py`, `decode_moe.py`, …), per-architecture attention (`minimax_m3_sparse.py`, `qsa/`, `dsv4/`), `attention.py`, `rope.py`, `norm.py`, `sampling.py`, `moe_router.py`, etc.
- `python/freetoken/kernel/fla/` — flash-linear-attention chunk kernels (`chunk_delta_h.py`, `chunk_o.py`, `cumsum.py`, `solve_tril.py`, `wy_fast.py`, …) inherited from `flash-linear-attention`.

The `accel` extra's two packages are loaded lazily; the engine falls back to the pure-Triton path when either is missing (`python/freetoken/kernel/backend.py:16-23`):

```python
def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False
@functools.cache
def is_flashinfer_installed() -> bool:
    return _importable("flashinfer")
@functools.cache
def is_sgl_kernel_installed() -> bool:
    return _importable("sgl_kernel")
```

The `Engine._validate_attention_backend_choice` rejects a misconfigured backend before weights load (`engine/engine.py:206-217`).

### §0.2 Where the dispatcher sits

Three sibling dispatchers live in `python/freetoken/`:

1. **Attention backend dispatcher** — `python/freetoken/attention/__init__.py:38-148` (registry). Each backend declares a `BackendInfo(supported_types, requires_flashinfer, requires_sgl_kernel, requires_sm100, page_sizes, consumes_attn_spec)` (`attention/__init__.py:20-35`). Eight registered backends: `trtllm`, `fi`, `fa`, `triton`, `dsv4_sparse`, `dsa`, `m3_sparse`, `qsa_sparse` (`attention/__init__.py:41-148`). Comma-separated `"prefill,decode"` is parsed at `attention/__init__.py:176-184`. Engine instantiation: `engine/engine.py:458`.

2. **MoE backend dispatcher** — `python/freetoken/moe/__init__.py:9-14`. The CLI `--moe-strategy` accepts `fused`, `offload`, `cpu`, `hybrid`, `auto` (`server/args.py:588-599`). The offload family resolves at runtime to one of three `OffloadMoeCache` decode modes (`offload_cache.py:124-143`):
   - `gpu` — pure GPU offload (PCIe-streamed misses into slot cache).
   - `cpu` — all misses computed on the CPU executor.
   - `hybrid` — capped fetch over PCIe + CPU overflow, overlapped.

3. **KV-pool family dispatcher** — `python/freetoken/kvcache/__init__.py` exposes `create_kv_pool` / `resolve_pool_class` (called from `engine/engine.py:422` and `engine/engine.py:346`).

The boundary rule: Python owns policy (LRU pick, fetch-fraction, stream/event scheduling, KV index structure); Triton kernels own GEMM/GEMV/attention; C++/CUDA owns **host-pinning primitives** (`pinned_tensor.cpp`), **CPU MoE compute** (`cpu_moe_ext.cpp`), and **NCCL bindings** (`pynccl.cu`). The TVM-FFI `jit/` files and the optional `[accel]` wheel provide attention/MoE kernels that the rest of the engine calls into through the dispatcher.

---

## §1 The `q★ = m·B_P / B_H` policy

The paper's closed-form `q★` is **not literally in the code**. The code uses the same shape but with a half-step refinement (cost-balanced rounding) and stores the result as a 16-bit fixed-point fraction rather than a scalar.

### §1.1 Where the formula is computed

The CPU reference kernel (bit-identical to the GPU Triton kernel, used as the test oracle) — `python/freetoken/moe/offload_kernels.py:156-166`:

```python
if frac_q16 > 0:
    m, q = len(missing), 1 << 16
    lo = (m * frac_q16) >> 16
    cost = lambda f: max(f * (q - frac_q16), (m - f) * frac_q16)  # noqa: E731
    max_fetch = lo if cost(lo) <= cost(lo + 1) else lo + 1
```

The mapping `q★ → frac_q16` is the band-width ratio itself, computed at calibration time in `python/freetoken/moe/bench_profile.py:165-191` (`load_hybrid_fetch_fraction`):

```python
cpu_ov, pcie_ov = entry.get("cpu_moe_overlap_gbs"), entry.get("pcie_gather_overlap_gbs")
if cpu_ov and pcie_ov:
    return min(1.0, pcie_ov / (pcie_ov + cpu_ov))
cpu, pcie = entry.get("cpu_moe_gbs"), entry.get("pcie_gather_gbs")
if cpu and pcie:
    return min(1.0, pcie / cpu)
```

This is exactly the paper's `q★ = B_P / (B_P + B_C)` with two refinements:
- the *overlapped* `cpu_moe_overlap_gbs` / `pcie_gather_overlap_gbs` are the per-step regime-relevant bandwidths (both kernels running concurrently — the real contention regime);
- the older *standalone* `cpu_moe_gbs` / `pcie_gather_gbs` is the fallback when the overlapped pair was not measured (full-DRAM-contention model: `cpu_eff = cpu − pcie`, which reduces to `pcie/cpu`).

The formula `lo = (m * frac_q16) >> 16` is the truncated `m · frac_q16`. The cost function picks whichever of `lo` or `lo+1` minimizes `max(fetch_time, overflow_compute_time)`.

### §1.2 When is `q★` recomputed?

**Once at calibration.** The fraction lives in a JSON profile at `$XDG_CACHE_HOME/freetoken/benchbw/<gpu-uuid>.json` (`bench_profile.py:34-46`):

```python
def _cache_dir() -> str:
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(cache, "freetoken")

def default_profile_path(gpu_uuid: str | None = None) -> str | None:
    if gpu_uuid:
        return os.path.join(_cache_dir(), "benchbw", f"{gpu_uuid}.json")
    return os.path.join(_cache_dir(), "benchbw.json")
```

The CLI command that writes this profile is `ft bench bw` (`python/freetoken/cli.py:69-71` and `benchmarks/README.md:39-42`: "For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw` instead").

At engine boot, the profile is loaded once and resolved against the chosen `quant_format` (`bench_profile.py:114-153`). The fraction is then stored in `cache.hybrid_fetch_fraction: float` (`offload_cache.py:143`) and passed unchanged to every `ensure_experts_hybrid` call:

```python
# engine/engine.py, via OffloadMoeCache construction
# offload_cache.py:855-874
def ensure_experts_hybrid(self, layer_id, expert_ids):
    ensure_experts_hybrid(
        self, layer_id, expert_ids,
        self.hybrid_max_fetch, self.hybrid_fetch_fraction
    )
```

**Per-step behavior**: the fraction is *applied* every decode step (i.e. the integer cap `max_fetch` is recomputed against this step's miss count `m`), but the *fraction itself* is constant between `ft bench bw` runs. This is the "bandwidth-adaptive" part: the policy is calibrated once per `(format, hardware)` pair and then amortized across all decode steps. No per-request or periodic recalculation exists.

### §1.3 Where `m`, `B_P`, `B_H` are stored

- `m` (the step's miss count): `cache.num_missing_full[0]` — written by the Triton kernel, read by `record_decode_stats_hybrid` (`offload_cache.py:916`).
- `B_P`, `B_H`: not stored as scalars. They live in the JSON profile (`benchbw/<gpu-uuid>.json`) under keys `cpu_moe_overlap_gbs`, `pcie_gather_overlap_gbs`, `cpu_moe_gbs`, `pcie_gather_gbs` — written by `python/freetoken/moe/benchbw.py` (not surveyed line-by-line). The driver is `cli.py:69-71`.

### §1.4 Triggering condition (literal)

There is no `if cond: recompute()` style trigger — the fraction is loaded once at boot. Recalibration is manual: run `ft bench bw` again. The `_usable_profile` (`bench_profile.py:80-111`) silently discards a profile whose GPU name does not match the running device (logged as a warning), so swapping cards never reuses a stale ratio.

---

## §2 Double-buffered prefill (producer/consumer pipeline)

### §2.1 Where pinned host buffers are allocated

Two layers of "pinned":

1. **Pinned host banks** (MoE expert source, large) — `OffloadMoeCache.set_bank_sources` (`offload_cache.py:292-365`) stores `bank_sources[name]: list[num_layers]` of `torch.Tensor[num_experts, ...]` per layer; these are pinned+device-mapped via `python/freetoken/kernel/pinned.py:42-45` calling `cudaHostAlloc(Portable | Mapped)` in `kernel/csrc/pinned_tensor.cpp:71-92`:

```cpp
// python/freetoken/kernel/csrc/pinned_tensor.cpp:71-79
void *data_ptr = nullptr;
const cudaError_t alloc_err = cudaHostAlloc(
    &data_ptr, alloc_nbytes, cudaHostAllocPortable | cudaHostAllocMapped);
```

The Python entry point is `kernel/pinned.py:42`:

```python
def alloc_pinned_tensor(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate an exact-size, uninitialized pinned host tensor via cudaHostAlloc."""
    return _load_pinned_extension().alloc_pinned_tensor(list(shape), dtype)
```

2. **GPU double-buffer slot region** — `offload_cache.py:606-633`:

```python
def _init_prefill_overlap_buffers(self) -> None:
    assert self.banks, "set_bank_sources must register the banks first"
    self._prefill_buffer_layer = [None, None]
    self._prefill_buffer_released = [True, True]
    self._prefill_buffer_has_release_event = [False, False]
    # The double buffers borrow the slot cache's first 2 * num_experts slots
    # (one full expert layer per buffer), one view per registered bank.
    self.prefill_bank_buffers = [
        cache[: 2 * self.num_experts].view(2, self.num_experts, *cache.shape[1:])
        for _, cache in self.banks
    ]
    if self.device.type == "cuda":
        self.prefill_copy_stream = torch.cuda.Stream(device=self.device)
        self.prefill_ready_events = [torch.cuda.Event() for _ in range(2)]
        self.prefill_release_events = [torch.cuda.Event() for _ in range(2)]
        self.prefill_begin_event = torch.cuda.Event()
```

The pre-condition `cache_size >= 2 * num_experts` is enforced at `offload_cache.py:163-167` and again at `engine/cache_budget.py:69-74`:

```python
# engine/cache_budget.py:69-74
hi = min(total_experts, max_slots)
overlap = prefill_overlap and hi >= 2 * num_experts
lo = 2 * num_experts if overlap else num_experts
assert hi >= lo, f"slot cap {hi} below the minimum {lo} slots"
```

### §2.2 Which CUDA streams / `cudaMemcpyAsync` calls overlap which stage

- **Compute stream** — implicit (`torch.cuda.current_stream(self.device)`).
- **Copy stream** — `prefill_copy_stream = torch.cuda.Stream(device=self.device)`, declared once at `offload_cache.py:618`.
- **Pre-fill dispatch** — `prefetch_prefill_layer` (`offload_cache.py:668-701`) selects `buffer_id = layer_id % 2` (the ring) and issues the copy on `prefill_copy_stream`:

```python
# offload_cache.py:684-701
def copy() -> None:
    self._invalidate_prefill_buffer(buffer_id)
    for (per_layer, _), buffer in zip(self.banks, self.prefill_bank_buffers):
        buffer[buffer_id].copy_(per_layer[layer_id], non_blocking=True)

if self._prefill_hit_d2d_active:
    self._prefetch_split(layer_id, buffer_id)
elif self.prefill_copy_stream is None:
    copy()
else:
    with torch.cuda.stream(self.prefill_copy_stream):
        if self._prefill_buffer_has_release_event[buffer_id]:
            self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
        copy()
        self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)
```

- **Fence ordering** — `begin_prefill` (`offload_cache.py:645-666`) records `prefill_begin_event` on the compute stream, then makes `prefill_copy_stream.wait_event(prefill_begin_event)` so the copy side cannot stomp bytes the running decode is reading:

```python
# offload_cache.py:657-658
self.prefill_begin_event.record(torch.cuda.current_stream(self.device))
self.prefill_copy_stream.wait_event(self.prefill_begin_event)
```

- **Consumer wait** — `wait_prefill_layer` (`offload_cache.py:818-830`) makes the compute stream wait on `prefill_ready_events[buffer_id]`:

```python
if self.prefill_ready_events:
    torch.cuda.current_stream(self.device).wait_event(self.prefill_ready_events[buffer_id])
return tuple(buffer[buffer_id] for buffer in self.prefill_bank_buffers)
```

- **Producer release** — `release_prefill_layer` (`offload_cache.py:832-841`) records `prefill_release_events[buffer_id]` on the compute stream when the layer's GEMMs are done.

- **Optional hit-D2D split** — when `prefill_hit_d2d` is on and `cudaMemcpyBatchAsync` resolves (`offload_cache.py:622-633`, `735-744`), one prefill launches a hit gather on the compute stream and a single coalesced miss batch on the copy stream (`offload_cache.py:746-816`). This is the paper's "avoiding redundant context recomputation" path:

```python
# offload_cache.py:809-816
if dst:
    self._batch_memcpy(
        torch.tensor(dst, dtype=torch.int64),
        torch.tensor(src, dtype=torch.int64),
        torch.tensor(nbytes, dtype=torch.int64),
        torch.cuda.current_stream(self.device).cuda_stream,
    )
self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)
```

The class-level comments note the empirical hazard: `cudaMemcpyBatchAsync` silently degrades to synchronous when a batch mixes large entries with sub-256 KB entries on registered host memory (H100 + CUDA 13.0) — `offload_cache.py:17-26`.

### §2.3 Producer/consumer queue — explicit or implicit

**Explicit.** Two `torch.cuda.Stream` objects, three `torch.cuda.Event`s per buffer (`begin`, `ready`, `release`). No asyncio, no thread, no `c10::cuda::CUDAStream` switch beyond the PyTorch wrappers. The ring is `[buffer_id] in {0, 1}` selected by `buffer_id = layer_id % 2` (`offload_cache.py:676`).

### §2.4 Where the ringbuffer is maintained

`offload_cache.py:276-278`:

```python
self._prefill_buffer_layer: list[int | None] = [None, None]
self._prefill_buffer_released: list[bool] = [True, True]
self._prefill_buffer_has_release_event: list[bool] = [False, False]
```

The slot map for the borrowed buffers (slots `[0, 2*num_experts)`) is invalidated on every prefill dispatch by `_invalidate_prefill_buffer(buffer_id)` (`offload_cache.py:635-643`), which both frees the `id_of_slot` reverse-map entries and zeroes `usage` so the slots are picked first as `argmin(usage)` victims on the next `ensure_experts`.

---

## §3 Routing-locality weighted LRU

### §3.1 Data structure

The LRU is a **flat GPU-side tensor pool, not a Python dict**. Three tensors hold the LRU state (`offload_cache.py:168-201`):

```python
self.slot_for_id = torch.full(        # [num_layers, num_experts] int32, -1 = absent
    (self.num_layers, self.num_experts), -1, dtype=torch.int32, device=self.device)
self.id_of_slot = torch.full(         # [cache_size] int32, flat id = layer*E + expert
    (self.cache_size,), -1, dtype=torch.int32, device=self.device)
self.usage = torch.zeros(             # [cache_size] int64, last-active step
    (self.cache_size,), dtype=torch.int64, device=self.device)
self.step = torch.zeros((),           # [] int64, monotonic counter
    dtype=torch.int64, device=self.device)
self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
self.num_indices = torch.zeros((1,), dtype=torch.int64, device=self.device)
```

The eviction logic itself lives in `flashlib.kernels.slot_cache.lru_ensure` (third-party, pinned at `flashlib==0.3.0` in `pyproject.toml:42`). The non-hybrid path delegates the entire slot allocation / eviction in one Triton launch — `python/freetoken/moe/offload_kernels.py:28-40`:

```python
lru_ensure(
    expert_ids,
    cache.slot_for_id.view(-1),
    cache.id_of_slot,
    cache.usage,
    cache.step,
    expert_ids,
    cache.src_indices,
    cache.evict_slots,
    cache.num_indices,
    stats=cache.lru_stats[layer_id] if cache.collect_stats else None,
    id_base=layer_id * cache.num_experts,
)
```

The CPU reference mirror — `python/freetoken/moe/offload_kernels.py:167-180` — is what every test oracle pins against:

```python
usage = cache.usage.tolist()
for idx in range(num_fetch):
    expert = missing[idx]
    victim = min(range(cache.cache_size), key=lambda s: (usage[s], s))
    old_id = int(cache.id_of_slot[victim].item())
    if old_id >= 0:
        cache.slot_for_id.view(-1)[old_id] = -1
    cache.id_of_slot[victim] = layer_id * cache.num_experts + expert
    cache.slot_for_id[layer_id, expert] = victim
    cache.usage[victim] = step
    usage[victim] = step
    cache.evict_slots[idx] = victim
    cache.src_indices[idx] = expert  # layer-local row
```

### §3.2 The key

`id == layer_id * num_experts + expert` (a flat packing). The comment at `offload_cache.py:175-177` makes the design choice explicit:

```python
# Reverse map, in the flat id space flashlib's slot_cache works in:
# id == layer_id * num_experts + expert, so one array replaces the (layer,
# expert) pair and evicting a slot needs no decode.
```

So `key = (layer_id, expert_id)` packed, never `(model_id, expert_id)`. Per-layer is the only granularity; cross-model shared slots are **未在代码中找到** (no model_id in any slot key).

### §3.3 The weight

**Pure recency, not frequency.** `usage[slot]` is set to `step` on insertion (`offload_kernels.py:176`) and on every hit (`offload_kernels.py:147-151`). `step` is incremented on every `ensure_experts` call:

```python
# offload_kernels.py:142-143
step = int(cache.step.item()) + 1
cache.step.fill_(step)
```

The victim is `argmin(usage, slot_id)` (`offload_kernels.py:170`) — lowest `usage` wins, with `slot_id` as a deterministic tiebreaker. The `lru_stats[num_layers, N_STATS]` tensor (`offload_cache.py:228-230`) accumulates miss/active counters per layer for the `decode_miss_stats` reporting path (`offload_cache.py:928-949`), but it is not consulted for eviction.

### §3.4 Routing-locality bias

The "weighted" part of the LRU is **the hybrid fetch set selection, not the eviction policy.** `offload_cache.py:199-201`:

```python
# hybrid only: per-(layer, expert) last-active decode step (LRU on the expert), -1
# if never active. The hybrid ensure kernel reads it to pick which capped misses to
# fetch (most-recently active first) and bumps it for every active expert.
self.expert_recency = torch.full(
    (self.num_layers, self.num_experts), -1, dtype=torch.int64, device=self.device
)
```

The hybrid CPU reference (`offload_kernels.py:153-157`) is:

```python
if _HYBRID_FETCH_BY_RECENCY:
    rec = cache.expert_recency[layer_id].tolist()
    missing.sort(key=lambda e: (-rec[e], e))
else:
    missing.sort()
```

The env-var knob toggling the routing bias on/off (`offload_kernels.py:14-16`):

```python
_HYBRID_FETCH_BY_RECENCY = (
    os.getenv("FREETOKEN_HYBRID_FETCH", "recency").strip().lower() != "lowest_id"
)
```

So the *eviction* policy is plain LRU on `usage`, but the *fetch set* under capped hybrid decode is LRU-by-expert (with id tiebreak) — that's the routing-locality weighting the paper describes. Frequency/EMA-style weight is **未在代码中找到** (no exponential moving average, no counter-based weight).

---

## §4 Semantic boundary checkpoints (thinking / tool-call detection)

### §4.1 What is detected

The implementation names this "special-token checkpoint". The CLI knob is `--enable-special-token-ckpt` — `python/freetoken/server/args.py:739-752`:

```python
parser.add_argument(
    "--enable-special-token-ckpt",
    action="store_true",
    dest="special_token_ckpt",
    default=ServerArgs.special_token_ckpt,
    help=(
        "Checkpoint decode state at special tokens (currently the tool-call opener). "
        "When a GDN-hybrid or SWA model samples its tool-call opener token, the "
        "scheduler preserves a reuse point just after it (GDN: a state snapshot "
        "donated to the prefix cache; SWA: the trailing window is kept resumable), so "
        "a client that rewrites the echoed tool call only invalidates the call body, "
        "not the turn."
    ),
)
```

Only the **tool-call opener** is a checkpoint trigger; the thinking boundary is *not* a checkpoint trigger. The reasoning parser splits `…` (`server/reasoning_parser.py:30-33`):

```python
BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
THINK_START_TOKEN = ""
THINK_END_TOKEN = ""
```

The tool-call markers it knows about (`server/function_call_parser.py:69-79`):

```python
"<|tool_call>",
"<|tool_call_begin|>",
"<minimax:tool_call>",
"]<]minimax[>[<tool_call>",
"<｜DSML｜tool_calls>",
```

### §4.2 What is saved at the boundary

- **GDN-hybrid path** — the linear-state pool snapshot at the anchor position. Scheduler captures the anchor length on the sampled token (`scheduler/scheduler.py:368-372`):

```python
if (
    next_token == self.toolcall_anchor_id
    and req.toolcall_anchor_len is None
):
    req.toolcall_anchor_len = req.input_ids.numel()
```

Then `snapshot_toolcall_anchor` freezes the GDN state into the ping-pong slot that is idle during decode (`scheduler/cache.py:148-173`):

```python
def snapshot_toolcall_anchor(self, reqs: List[Req]) -> None:
    if not self.is_hybrid:
        return
    pool = self.linear_state_pool
    for r in reqs:
        a = r.toolcall_anchor_len
        if (
            a is None
            or r.mamba_ping_pong is None
            or r.mamba_last_track_seqlen is not None
            or r.cached_len != a
            or align_down(a, self.page_size) != a
        ):
            continue
        dst = r.mamba_ping_pong[r.mamba_next_track_idx]
        pool.copy_from(r.linear_slot_idx, dst)
        r.mamba_last_track_seqlen = a
        r.mamba_next_track_idx = 1 - r.mamba_next_track_idx
```

This state is later **donated into the radix prefix cache** at finish, where it can be reused if a future prompt replays the same prefix (the comment at `core.py:60-62`: "reaches it (`snapshot_toolcall_anchor`) and donated at finish").

- **SWA path** — the trailing sliding window is kept resumable. `maybe_free_swa_out_of_window` caps proactive eviction at `anchor − window − gap` so the `[anchor − window, anchor)` span stays alive even when normally out of window (`scheduler/cache.py:191-208`):

```python
if req.toolcall_anchor_len is not None:
    cap = req.toolcall_anchor_len - window - _SWA_RETAIN_GAP
    if threshold - cap > window + _SWA_RETAIN_GAP:
        req.toolcall_anchor_len = None
    else:
        threshold = min(threshold, cap)
```

The cap can be dropped ("drop the anchor and let normal eviction resume") if the request runs far past the anchor and the cap would grow the live SWA unbounded (`scheduler/cache.py:198-205`).

- **KV cache + prompt prefix** — always present via the radix tree; the checkpoint mechanism is an *additive* retention on top.

### §4.3 The anchor token resolution

`scheduler/scheduler.py:125-134`:

```python
self.toolcall_anchor_id = None
if config.special_token_ckpt and (
    self.cache_manager.is_hybrid or self.cache_manager.is_swa
):
    from freetoken.server.function_call_parser import toolcall_opener_for
    self.toolcall_anchor_id = load_toolcall_anchor_id(
        self.tokenizer,
        toolcall_opener_for(getattr(config, "tool_call_parser", "")),
    )
```

`toolcall_opener_for` (`function_call_parser.py:3709-3713`):

```python
def toolcall_opener_for(tool_call_parser: str) -> str | None:
    detector = FunctionCallParser.ToolCallParserEnum.get(tool_call_parser)
    return detector.toolcall_opener if detector is not None else None
```

The decoder mapping is the `toolcall_opener` attribute on each detector (e.g. `BaseFormatDetector.toolcall_opener` per-format).

The boundary detection point itself is inside the decode loop (`scheduler/scheduler.py:907-908`):

```python
if self.toolcall_anchor_id is not None and not batch.is_prefill:
    self.cache_manager.snapshot_toolcall_anchor(batch.reqs)
```

### §4.4 Thinking-boundary checkpoint

**未在代码中找到.** The reasoning parser (`server/reasoning_parser.py`) splits the text into `reasoning_content` / `content`, and the responses API folds reasoning items back into the assistant turn (`server/responses_api.py:291-309`), but there is no GDN/SWA retention hook fired by the thinking closer `` or by `<think>…</think>`. The only boundary trigger is the tool-call opener.

---

## §5 GPU cache capacity runtime adjustment

### §5.1 Is the per-step expert budget tunable at runtime?

**Yes.** Two layers:

1. **Static CLI knobs** — `python/freetoken/server/args.py:650-685`:
   - `--moe-cache-size` (int)
   - `--moe-cache-rate` (float, fraction of all experts)
   - `--moe-cache-auto` (bool, derive from free VRAM with KV floor)
   - `--kv-reserve-tokens` (floor for KV before auto fills experts)
   - `--moe-cache-policy` (choices: `["lru"]` only — `args.py:683`)
   - `--moe-hybrid-max-fetch` (int; `-1` = use benched fraction; `0` = never fetch; large = plain offload) — `args.py:714-725`
   - `--disable-moe-prefill-overlap` / `--moe-prefill-hit-d2d` — `args.py:727-765`

2. **Live rebuild endpoint** — `Engine.rebuild_runtime_cache` (`engine/engine.py:874-953`) resizes the MoE slot cache, KV page pool, GDN state pool, and/or SWA window pool *in place*, without reloading weights or host expert banks. The docstring at `engine/engine.py:882-885`:

```python
"""Idle-only in-place resize of the MoE slot cache, KV page pool, GDN (mamba) state pool,
and/or the window pool (num_swa_pages: an absolute pinned window), followed by CUDA-graph
re-capture. Does NOT reload weights or host expert banks. The caller (scheduler) must
guarantee no in-flight prefill/decode."""
```

The scheduler-facing wrapper (`scheduler/scheduler.py:160-201`):

```python
def rebuild_cache(
    self,
    *,
    moe_cache_size: int | None = None,
    num_pages: int | None = None,
    num_mamba_slots: int | None = None,
    num_swa_pages: int | None = None,
) -> None:
    """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
    pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, ..."""
    ...
    self.engine.rebuild_runtime_cache(
        moe_cache_size=moe_cache_size, num_pages=num_pages,
        num_mamba_slots=num_mamba_slots, num_swa_pages=num_swa_pages,
    )
```

The CLI client is `ft ctl cache rebuild --moe <slots> --kv <tokens> --mamba <slots> --swa <tokens>` (`control_cli.py:18-23`, `312-342`):

```python
# control_cli.py:18-23
CACHE_TARGETS = (
    ("moe_cache_size", "moe", "moe_cache_size"),
    ("kv_tokens", "kv", "num_pages"),
    ("num_mamba_slots", "mamba", "num_mamba_slots"),
    ("swa_tokens", "swa", "num_swa_pages"),
)
```

Units: `moe`/`mamba` are slot counts; `kv`/`swa` are token counts, rounded up to the pool's page size against the live geometry (`control_cli.py:412-438`).

### §5.2 PowerInfer-style `--cpu-offload-gb` equivalent

**未在代码中找到** a `--cpu-offload-gb` analog. FreeToken uses slot counts and token counts, not byte budgets, for its user-facing knobs. The byte budget for engine-internal planning is `memory_ratio` (`server/args.py`; the cache-budget planner at `engine/cache_budget.py:31-38`):

```python
def net_cache_budget_bytes(
    memory_ratio: float, baseline_free: int, weights_bytes: int, fixed_cache_size: int
) -> int:
    return int(memory_ratio * baseline_free) - weights_bytes - fixed_cache_size
```

The WSL-specific pin budget (`engine/engine.py:1317-1351`) does have a `FREETOKEN_PIN_BUDGET_GB` env-var override and the `WSL → 40% of RAM` fallback, but that is for the *host pin budget* of the MoE expert banks, not for the GPU cache geometry.

---

## §6 System dependencies

### §6.1 CUDA features (from `setup.py`, `pyproject.toml`, CI)

- **Toolkit**: CUDA 13 with `nvcc` on `PATH`. `docs/install.md:5`: "Linux x86_64, NVIDIA GPU, driver r580+ (CUDA 13)". `CONTRIBUTING.md:49`: "CUDA kernels are JIT-compiled on first use and need a CUDA 13 toolkit with `nvcc` on `PATH`".
- **PyTorch**: `>=2.11,<2.12`, `pyproject.toml:6`, sourced from a dedicated CUDA-13.0 index (`pyproject.toml:94-100`):

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu130" }
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
```

- **Triton**: pinned `triton==3.6.0` on Linux only (`pyproject.toml:62`): `"triton==3.6.0; platform_system == 'Linux'"`.
- **Native extensions**: `flashinfer-python[cu13]>=0.6,<0.7` and `sglang-kernel==0.4.5` (both pulled by `accel`; `pyproject.toml:79-81`). vLLM Marlin path requires `vllm>=0.14,<0.15` in a separate env (`pyproject.toml:83-87`).
- **Compute capabilities**: kernels in the repo span sm_80 (Marlin path) through sm_120 (RTX 5090). Examples:
  - `python/freetoken/kernel/triton/dsv4/sparse_attn.py:45` "sm_120, e.g. RTX 5090".
  - `python/freetoken/kernel/triton/nvfp4_linear.py:73` "RTX 5090 / GB202, 170 SMs".
  - `python/freetoken/kernel/fla/utils.py:312` "ADA = 101376 // RTX 4090".
  - `trtllm` backend gates on `is_sm100_family()` (`engine/engine.py:218-223`).
- **Driver probe**: `kernel/backend.py:50-65` resolves the max CUDA version via `freetoken.kernel._pinned_tensor.driver_cuda_version()`.

### §6.2 OS support: Linux only, or Windows too?

**Linux is the reference platform; Windows + WSL are second-class but supported.** No macOS, no Metal.

- `pyproject.toml:24` classifier: `"Operating System :: POSIX :: Linux"`. The README does *not* add a Windows classifier.
- `install.sh:3` is named "Linux, NVIDIA CUDA".
- `AGENTS.md:46` "Linux x86_64 with an NVIDIA GPU" is the only stated dev environment.
- The Python runtime path tolerates Windows + WSL via WDDM-aware helpers:
  - `kernel/csrc/pinned_tensor.cpp:62-67` notes that on Windows/WDDM `cudaHostRegister`'d memory maps to a different device VA, so zero-copy consumers must go through `host_device_ptr` (`kernel/pinned.py:59-68`):

```python
def device_ptr(t: torch.Tensor) -> int:
    """Base address of t as the GPU must dereference it.
    ...
    Host tensors must be pinned+mapped."""
    if t.is_cuda or _host_ptr_identity():
        return t.data_ptr()
    return _load_pinned_extension().host_device_ptr(t.data_ptr())
```

  - `engine/engine.py:1317-1351` detects WSL by kernel tag (`os.uname().release`) and defaults the pin budget to 40% of RAM because WDDM caps CUDA pinning at ~half of RAM.
  - `server/args.py:697-712` accepts `--moe-cpu-layers auto` for Windows/WSL where pinning is quota-capped:

```python
"'auto' is for Windows/WSL only, where CUDA pinned memory is capped: it locks just enough
head+tail layers for the banks over the pin budget. Any value, 'auto' included, commits to
CPU decode before the model is built, so the expert format must have a CPU executor path
(bf16, nvfp4, mxfp4); do not pass it on Linux."
```

  - `kernel/fla/utils.py:45-48`: "Detected Windows operating system. Triton does not have an official Windows release, thus FLA will not be adapted for Windows, and any potential errors will not be fixed." So even where Windows binary wheels exist, the FLA kernels are explicitly unsupported on Windows.
  - `kernel/triton/fp8_pertensor_linear.py:42` mentions a Windows-specific fallback path.

### §6.3 Why no Metal?

**No Apple Silicon / Metal / MPS / `darwin` references found anywhere in the codebase** (`grep -r "Apple\|Metal\|MPS\|darwin\|M1\|M2\|M3\|M4\|arm64\|aarch"` over `.py / .md / .toml / .yml` returns zero hits). The architectural reasons are visible in the code:

- All custom GEMM/GEMV kernels (Triton in `kernel/triton/`) assume CUDA shared memory + tensor cores + warp-level primitives.
- `accel` extra pins CUDA-only native kernels (`flashinfer-python[cu13]`, `sglang-kernel==0.4.5`).
- The MoE offload cache requires `cudaHostAlloc(Portable | Mapped)` to keep the host banks zero-copy on the GPU (`kernel/csrc/pinned_tensor.cpp:71-79`). Metal has no equivalent.
- The pinned bank count is computed assuming a Linux/UVA identity (`kernel/pinned.py:59-68`), which Windows/WDDM already breaks. Metal would need a separate host-pinning primitive.
- The flashml.ai landing page does advertise Windows + Linux desktop, but the `flashml.ai` download is the closed-source desktop app, **not** this open-source engine — `README.md:28-30`: "Download FreeToken for Windows or Linux at flashml.ai. It sets the engine up for you and gives you a GUI". The OSS engine is the Linux/WSL story above; the desktop is a separate bundle.

---

## §7 License + dependencies

### §7.1 License

Apache-2.0 confirmed. `LICENSE:1-2`:

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
```

`pyproject.toml:16-17` declares `license = "Apache-2.0"`, `license-files = ["LICENSE"]`. `README.md:85` references the Apache-2.0 link.

### §7.2 Required Python packages (from `pyproject.toml:36-64`)

Core runtime pins:

| Package | Version pin | Note |
|---|---|---|
| `torch` | `>=2.11,<2.12` | Build-time CUDA-13.0 toolchain link. |
| `torchvision` | `>=0.26,<0.27` | |
| `triton` | `==3.6.0` (Linux only) | Pinned to one version. |
| `transformers` | `>=5.16,<5.17` | |
| `safetensors` | `>=0.6,<1` | |
| `huggingface_hub` | `>=1.5,<2` | |
| `modelscope` | `>=1.37,<2` | |
| `gguf` | `>=0.19,<1` | llama.cpp-format. |
| `apache-tvm-ffi` | `==0.1.13.post3` | "pin it". |
| `flashlib` | `==0.3.0` | "slot_cache" LRU kernel. |
| `openai` | `>=2.0,<3` | OpenAI-compatible API surface. |
| `fastapi` | `>=0.115,<1` | |
| `uvicorn` | `>=0.30,<1` | |
| `pydantic` | `>=2.9,<3` | |
| `pyzmq` | `>=27,<28` | Tokenizer / daemon comms. |
| `partial-json-parser` | `>=0.2,<1` | Streaming tool-call JSON. |
| `pillow` | `>=10,<13` | |
| `numpy` | `>=2.0,<2.5` | `flashlib` (numba) needs <2.5. |
| `prompt_toolkit` | `>=3.0,<4` | Shell TUI. |
| `tqdm` / `einops` / `msgpack` | various | |

`[accel]` extra (`pyproject.toml:78-82`): `flashinfer-python[cu13]>=0.6,<0.7` + `sglang-kernel==0.4.5`.

`[dev]` extra: `pytest>=6.0`.

**vLLM Marlin** (`pyproject.toml:83-87`) is intentionally *not* in `[accel]` because vLLM pins `transformers>=4.56,<5`, which conflicts with the core `transformers>=5.16` requirement. The comment: "It is therefore not a lockable extra and is left out of the default resolution — install it separately (`pip install 'vllm>=0.14,<0.15'`) in a dedicated environment when you specifically need the Marlin path."

### §7.3 Model-specific dependencies

The model families listed in `docs/models.md:9-22` and exercised in `python/freetoken/models/`:

- **DeepSeek-V4-Flash** (`deepseek_v4/`, including `moe.py`, `compress.py`, `model.py`, `weight.py`) — sparse-attention + ds_fp4 expert format.
- **GLM-5.3-Flash / GLM-5.2 / GLM-4.7** (`glm5_next/`, `glm_moe_dsa/`, `glm4_moe/`) — ViT-tower streaming, linear-attention hybrid (kda.py), NVFP4 expert format.
- **Qwen3.6 / Qwen3.5 MoE** (`qwen3_5_moe/` — gdn.py, gdn_kernels.py, gdn_reference.py) — GatedDeltaNet hybrid state.
- **Qwen3.8-Flash-Next** (`qwen4_exp/` — gdn.py, ple.py, ple_disk.py, hc.py, attention.py, moe.py) — PLE (pinned linear-extension) table backing the linear-attention state.
- **Qwen3-MoE** (`qwen3_moe/`), **Qwen3-VL** (`qwen3_vl/` with vision tower), **Qwen2/3 dense** (`qwen2/`, `qwen3/`).
- **gpt-oss-120b / 20b** (`gpt_oss/`) — MXFP4 expert format with bias bank (`offload_cache.py:67-74` lists 6 banks for `mxfp4_triton`).
- **Gemma-4 26B-A4B / 12B / 31B** (`gemma4/`) — gemma4_unified variant uses a linear patch embedder; GGUF loader (`gemma4/gguf.py`).
- **MiniMax-M2.5 / MiniMax-M3** (`minimax_m2/`, `minimax_m3/`) — CLIP-style ViT tower, NVFP4 expert format; `minimax_m3_sparse.py` is the sparse-attention Triton kernel.
- **Muse-Glimmer-30B** (`muse_glimmer/`) — windowed ViT tower.
- **LLaMA / Mistral** (`llama/`, `mistral/`) — generic dense reference path.

Each MoE family may need its own format entry in `_BANK_SCHEMAS` (`offload_cache.py:36-78`): `bf16`, `fp8_block`, `q4_0`, `nvfp4`, `nvfp4_marlin`, `nvfp4_b12x`, `mxfp4_triton`, `ds_fp4`.

---

## §8 Test surface

### §8.1 Tests under `tests/`

`find tests -name "test_*.py" | wc -l` reports **122** test files. First five:

```
tests/mm/test_processor.py
tests/mm/test_encoder_cache.py
tests/server/test_mm_input.py
tests/server/test_streaming_model_matrix.py
tests/server/test_reasoning_parser_dsv4.py
```

Subsystems mirrored as subdirs of `tests/`:

```
attention/  checkpoint/  daemon/  dsv4/  e2e/  engine/  kernels/  kvcache/  layers/  mm/  models/  moe/  scheduler/  server/  tokenizer/
```

`tests/README.md` defines two markers (`pyproject.toml:118-121`):

```ini
markers = [
    "slow: takes tens of seconds (big kernel sweeps, real-checkpoint reads); deselect with -m 'not slow'",
    "needs_weights: needs a real local checkpoint, gated behind an env var (see tests/README.md)",
]
```

### §8.2 CI workflows under `.github/workflows/`

**4** workflows:

```
issue-labels.yml
nightly-wheels.yml
promote-beta.yml
release.yml
```

Plus 4 issue templates (`.github/ISSUE_TEMPLATE/`):

```
bug_desktop.yml
bug_engine.yml
config.yml
feature_request.yml
model_checkpoint.yml
```

`release.yml:33-37` triggers only on `push: tags: ["v*"]`; nightly wheels target `nightly` prerelease channel; the test matrix is `cp310 cp311 cp312 cp313` built inside a manylinux container. The CI does **not** run a smoke test matrix on every commit — only the wheel-build lane is exercised.

---

## §9 Hardware proof points from the repo itself

### §9.1 README GPU mentions

- `README.md:24`: "Scales across consumer laptops, gaming desktops, and workstation GPUs, with native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs."
- `README.md:30`: "Download FreeToken for Windows or Linux at flashml.ai".

### §9.2 Code-level GPU mentions

- `python/freetoken/kernel/triton/dsv4/sparse_attn.py:45`: "consumer-Blackwell (sm_120, e.g. RTX 5090) budget. (BLOCK_T=64 would need ~103KB.)"
- `python/freetoken/kernel/triton/nvfp4_linear.py:73`: "with ~100 KB smem/SM (e.g. RTX 5090 / GB202, 170 SMs) are smem-bound at 2 -> wave 340."
- `python/freetoken/kernel/triton/nvfp4_fused_moe.py:19`: "MiniMax-M2 decode toward the RTX 5090 read-bandwidth ceiling (~87%); the residual gap is FP4".
- `python/freetoken/kernel/fla/utils.py:312`: "ADA = 101376 # RTX 4090".
- `docs/cli.md:53-58`: example output shows an RTX 3060 Ti + RTX 5090 mixed system.

### §9.3 Where the paper's headline numbers live

The previous survey (`docs/speculative-token-tech.md` §1) recorded:

- 1.3-2.1× decode vs strongest baseline
- 1.19-1.22 s for 8K prefill
- <44 s tail TTFT

These three numbers are **not stated in the README, in any `docs/` page, in any test, or in any code comment** of this OSS repo. `grep -r "1\.19\|1\.22\|<44\|1\.3×\|1\.3x\|2\.1×\|2\.1x\|8K prefill\|tail.*TTFT\|44 s"` over `.md / .py / .txt / .toml / .yml` returns no hits. The numbers live only in the paper PDF (arXiv 2608.16157).

What *is* in the code are small, model-specific microbenchmarks:

- `python/freetoken/moe/cpu_executor.py:308-309`: "Measured on DeepSeek-V4-Flash bs=1 decode: 12.85 -> 15.65 tok/s, output bit-identical (tests/moe/test_dsfp4_prequant.py)."
- `tests/moe/test_hybrid_fetch.py:38`: "1.24 -> fetching 2 makes the PCIe side ~1.6x slower than balance; keep it at 1."
- `python/freetoken/kernel/triton/dsv4/fused_moe.py:97`: "constant over each 16-byte run): ~16x fewer SFU ops -> ~1.6x throughput."
- `benchmarks/bench_decode_moe.py:88`: `p.add_argument("--decode", type=int, default=256, help="decode tokens to measure (D)")`.

The harness to reproduce the paper numbers is `benchmarks/bench_decode_moe.py` (spawns `ft serve` per backend, times streamed `/v1/chat/completions`) plus `benchmarks/bench_offload_cache_copy.py` (synthetic micro-sweep). Neither writes the paper's headline numbers anywhere.

---

## §10 Paper-vs-code discrepancies

| Paper claim | Code reality | File:line |
|---|---|---|
| `q★ = m·B_P / B_H` literal closed form | Implemented as 16-bit fixed-point `frac_q16 = min(1, pcie / cpu)` plus `cost(lo) vs cost(lo+1)` rounding. The closed form `m · frac_q16` is the truncated version; the code adds a one-step cost-minimization to avoid the worst of the rounding error. | `moe/offload_kernels.py:158-162`, `moe/bench_profile.py:165-191` |
| "Global LRU" on `(model_id, expert_id)` | LRU on `(layer_id, expert_id)` packed as `layer*E + expert`. No `model_id` exists in any slot key. | `moe/offload_cache.py:175-177` |
| "Routing-locality weighted" | Eviction is *pure* recency LRU; the *routing-locality* part is the hybrid fetch-set selection (sort misses by per-expert `expert_recency`), not the eviction policy. | `moe/offload_cache.py:199-201`, `moe/offload_kernels.py:153-157` |
| Semantic anchor at *thinking* boundaries | Anchor fires only on the tool-call opener (`toolcall_anchor_id`); the thinking closer `` is a parser split (`reasoning_parser.py:32-33`) but no GDN/SWA retention hook is wired to it. | `server/args.py:739-752`, `scheduler/scheduler.py:368-372` |
| Per-request bandwidth calibration | Calibration is one-shot per `(format, hardware)` via `ft bench bw`; the fraction is loaded at engine boot and amortized across all subsequent decode steps. No per-request or periodic recalculation. | `moe/bench_profile.py:114-153`, `moe/offload_cache.py:143` |
| Hardware coverage "Linux + Windows desktop" | The OSS engine targets Linux reference; Windows + WSL is second-class via WDDM-aware pin primitives and `--moe-cpu-layers auto`. No macOS / Metal path. | `install.sh:3`, `AGENTS.md:46`, `engine/engine.py:1317-1351`, `kernel/pinned.py:59-68` |
| "1.3-2.1× / 1.19-1.22 s / <44 s" headline numbers | Not stated anywhere in the repo; reproducible only via `benchmarks/bench_decode_moe.py` against a model the user brings. | `benchmarks/README.md:1-39` |

---

## §11 Pointer map (where to read first)

For a reader who wants to follow the code top-down, this is the intended entry order:

1. `python/freetoken/engine/engine.py` — the dispatcher and lifecycle (`Engine.__init__`, `_validate_attention_backend_choice`, `rebuild_runtime_cache`).
2. `python/freetoken/moe/offload_cache.py` — `OffloadMoeCache` (LRU, double-buffer, prefill overlap, hit-D2D split).
3. `python/freetoken/moe/offload_kernels.py` — the Triton kernels for `ensure_experts`, `ensure_experts_hybrid`, `prefill_hit_compact`, plus the CPU reference oracle.
4. `python/freetoken/moe/bench_profile.py` — `q★` (as `pcie/(pcie+cpu)`) and the benchbw JSON profile loader.
5. `python/freetoken/engine/cache_budget.py` — startup + runtime cache-budget planner (`plan_cache_budget`, `resolve_moe_cache_auto`).
6. `python/freetoken/scheduler/scheduler.py` — sampling-loop glue; `--enable-special-token-ckpt` integration; `rebuild_cache` idle dispatcher.
7. `python/freetoken/scheduler/cache.py` — `snapshot_toolcall_anchor`, `maybe_free_swa_out_of_window`, `_cache_req_hybrid`.
8. `python/freetoken/kernel/pinned.py` + `python/freetoken/kernel/csrc/pinned_tensor.cpp` — the only first-party C++ the Python layer *must* have to function.
9. `python/freetoken/attention/__init__.py` — attention backend registry.
10. `python/freetoken/control_cli.py` + `python/freetoken/server/control_api.py` — the runtime cache-rebuild CLI surface (`ft ctl cache rebuild`).
11. `python/freetoken/server/args.py` — every user-facing flag, with descriptions that often quote the implementation comment.

The kernel-level MoE GEMMs and per-model attention live in `python/freetoken/kernel/triton/` and `python/freetoken/kernel/fla/`, but they are leaves of the dispatch tree and not necessary for a paper-level survey.

---

## §12 Theoretical foundations for dense-model edge inference (Round 2)

This section complements the implementation survey above. Where §1–§11 read the FreeToken code, §12 reads the literature that motivates it. Five theoretical topics drive the dense-model edge-inference regime flyingfish operates in: per-tier tensor placement as a linear program, the formal lower bound for compute-transfer overlap, the activation-power-law claim about FFN sparsity, the theoretical foundations of KV cache compression, and the granularity trade-off between per-expert and per-tensor transfer units. Each subsection follows the structure: principle (formal or near-formal), primary citation (URL + date), applicability to flyingfish (with `file:line` where relevant), and a verdict. Every quoted number carries the source URL and the date the source was last verified; items the literature does not pin down are recorded as **未能验证** rather than estimated.

### §12.1 FlexGen's LP offload formulation

**Primary citation**: Sheng, Zheng, Yuan, Li, Ryabinin, Fu, Xie, Chen, Barrett, Gonzalez, Liang, Ré, Stoica, Zhang. *FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU*. arXiv:2303.06865 (cs.LG); v1 2023-03-13, v2 2023-06-12; accepted **ICML 2023**. URL `https://arxiv.org/abs/2303.06865` (verified 2026-09-21). Code `github.com/FMInference/FlexLLMGen` (Apache-2.0).

**Principle (formal)**. FlexGen models inference as a graph traversal over a 2-D grid of (layer × output-token) squares; the LLM forwards as `T = T_pre · l + T_gen · (n − 1) · l`, with `l` layers and `n` tokens. The cost model assumes perfect overlap, so for each step
> `T_gen = max(c→g, g→c, d→c, c→d, comp)` and `T_pre = max(...)` analogously,
with one `c→g` / `g→c` / `d→c` / `c→d` term per (weights, activations, KV cache) movement divided by its measured bandwidth. The decision variables are **11** policy variables per block: `bls` (block size), `gbs` (GPU batch size), and nine percentages `wg/wc/wd` (weight on GPU/CPU/disk), `cg/cc/cd` (KV cache placement), `hg/hc/hd` (activation placement) with the partition constraints `wg + wc + wd = 1`, `cg + cc + cd = 1`, `hg + hc + hd = 1`. The objective is
> `min  T / bls   s.t.  gpu_peak_mem < gpu_mem_cap ∧ cpu_peak_mem < cpu_mem_cap ∧ disk_peak_mem < disk_mem_cap`,
which is the throughput reciprocal (Eq. 1, §A.3 of the arXiv PDF). The algorithm is two-level: **enumerate** `(bls, gbs)` (typically `gbs ∈ {4, 8, …}` and `bls < 20`), then **solve a 9-variable LP** for the placement `p`. The LP relaxes percentages to real values in `[0, 1]` and is solved cheaply per enumeration. **Theorem 4.1** (App. A.2): the zig-zag block schedule attains I/O complexity within **2×** of optimal for the block-scheduled regime.

**What `ff bench io --profile local-interconnect` solves vs ignores** (just a one-shot calibration): it measures `B_P`, `B_H`, the `B_P/B_H` ratio, and reports the `q★ = m · B_P / B_H` share (`src/host_profile.rs:79-94`, `src/cli/calibrate_io.rs:235`, `src/cli/glm.rs:30-43`; cross-ref `docs/speculative-token-tech.md` §1.5). What FlexGen has that the bulk of `ff` does **not** yet: (a) a **per-step** rather than per-run placement decision; (b) **co-residency** of weights, activations and KV cache on different tiers (we pick tiers independently per tensor class); (c) the **block size** axis (`bls`) that lets FlexGen amortise weight I/O over a column of the computation graph — we amortise only over the expert-cache LFU hit/miss ratio (`docs/models.md:43` records 2831 hits / 38140 misses / 26906 evictions across three runs). What the LP *cannot* capture that we already capture: arithmetic intensity (the LP assumes compute and IO are concurrent enough that `max(...)` holds; small per-step compute can violate this — see §12.2), and routing locality (which FreeToken captures in the per-step hybrid fetch-set, but FlexGen does not).

**Can the same LP serve as a per-step scheduler for H3's host-cache tier?** Yes in principle, but **two adapter items are missing**. (1) FlexGen's `T_gen` equation assumes a dense Transformer; H3's transformer block is conditioned by `--attention-projection-chunk-size 4096 --ffn-token-chunk-size 1024 --output-token-chunk-size 1024` (`docs/models.md:78-79`), so the `bls` axis needs to be aligned with the **denoise step** and the per-chunk token dimension, not with `n`. (2) FlexGen's host cache is a byte budget; H3's host cache is a **shard-level** structure (`crates/ff-h3/src/resources.rs` is the per-resource allocator). The LP would need a shard-placement variable analogous to `cg/cc/cd` — but shards, unlike KV pages, are not equally reusable across adjacent chunks, so a per-shard reuse-distance histogram would have to feed back into the LP cost model. **Verdict: needs adapter work.** The current `q★` is a single scalar; FlexGen's formulation lets us pull from a much wider policy space but only after the shard-cost model is in place.

### §12.2 Prefetch / pipeline overlap theoretical lower bound

**Primary citations**: (a) Narayanan et al., *PipeDream: Generalized Pipeline Parallelism for DNN Training*, SOSP 2019, arXiv:1806.03377 (verified via `proceedings.mlsys.org` Pipedream-derived follow-on papers; date confirmed 2019-11-06). (b) Huang et al., *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism*, NeurIPS 2019, arXiv:1811.06965. (c) For the LLM-inference modern formulation: *Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching*, arXiv:2504.06319 (2025-04-08), and *Architecting Long-Context LLM Acceleration with Packing-Prefetch Scheduler*, arXiv:2508.8457 (2025-08, IEEE MICRO submission). Cross-ref: *Flat GEMM Optimization via Double Buffering* (Hong et al., 2023) for the GEMM-overlap math, and `inferencex.semianalysis.com/glossary/double-buffering` for the canonical glossary statement.

**Principle (formal)**. The **sufficient condition** for a transfer to be fully hidden behind a compute step is
> `t_move ≤ t_compute` ⇒ `t_step = max(t_compute, t_move)`,
otherwise `t_step = t_compute + t_move` (serialised). The canonical second-form expression is `T_overlapped = max(T_compute, T_move)`; for a sequence of `m` micro-batches across `d` pipeline stages, GPipe-style **bubble fraction** is `(d − 1) / (m + d − 1)`; PipeDream-style 1F1B improves memory but the bubble remains until `m ≫ d`. The packing-prefetch paper quotes the same identity explicitly and reports an **8.06×** decode-throughput gain on Llama-3.1-8B (arXiv:2508.8457, headline number, author-measured on a workstation GPU, 2025-08). For the GEMM case, `t_compute ≈ (2 · M · N · K) / FLOPS` and `t_move ≈ (M·K + K·N) / σ`, so `t_move ≤ t_compute ⇔ arithmetic intensity ≥ (M·K + K·N) / (2·M·N) = (1/N + 1/M) / 2`, which is the textbook roofline "ridge point" — the bound becomes *easier* to satisfy as `M, N` grow.

**Apply to flyingfish H3 transformer-chunk sizes** (`docs/models.md:78-79`, `--attention-projection-chunk-size 4096 --ffn-token-chunk-size 1024 --output-token-chunk-size 1024`). The implicit pipeline is: **(i)** attention projections over a 4096-token block, **(ii)** FFN over a 1024-token block, **(iii)** output projection over a 1024-token block — three pipeline stages with chunk-size ratio 4:1:1. The H3 host-cache reads shards at this granularity (`crates/ff-h3/src/text_encoder.rs` is the per-encoder entry; `crates/ff-h3/src/video_vae.rs` and the denoise loop consume the per-chunk outputs). The pipeline is **sized for FL2VA memory budget** (the FL2VA / Ref2VA modules are the reason `attention-projection-chunk-size` is 4× the FFN chunk), **not** for transfer overlap. Whether `t_move ≤ t_compute` per stage depends on whether the host shard is already in `crates/ff-h3`'s host cache; if it is, `t_move ≈ 0` and the bound is trivially satisfied; if it is not, `t_move` is bounded by `S_shard / B_H` where `S_shard` is the per-shard byte size — and that is where the chunk sizes stop being tight. A chunk-size search that minimises `max(t_compute, t_move)` rather than memory would shrink the 4096 chunk on host-cold paths.

**FreeToken's double-buffered prefill asserts a similar lower bound** (`docs/speculative-token-tech.md` §1.5 cites `1.19–1.22 s` 8-k transfer at `64.4 / 52.7 GB/s PCIe 5.0 ×16`). The FreeToken implementation (§2 above) explicitly models the producer/consumer queue, so its `t_step = max(t_compute, t_move)` is the design constraint; ours is implicit. **Tightness**: ours is loose whenever a chunk crosses a shard boundary, because the implicit overlap assumes the shard is resident. Adding a per-step host-shard residency check to the chunk-size dispatcher would tighten the bound to `max(t_compute, t_move · 𝟙[miss])` and shrink the 30 m 53.57 s H3 wall time (`docs/models.md:11`). **Verdict: applicable**, with adapter work in `crates/ff-h3` to add the residency check.

### §12.3 Dense activation sparsity — PowerInfer's power-law claim

**Primary citations**: (a) Song, Mi, Xie, Chen, *PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU*, arXiv:2312.12456, v2 2024-12-12, accepted **SOSP 2024**, URL `https://arxiv.org/abs/2312.12456` (verified 2026-09-21). (b) Xue, Song, Mi, Zheng, Xia, Chen, *PowerInfer-2: Fast Large Language Model Inference on a Smartphone*, arXiv:2406.06282, v3 2024-12-12, URL `https://arxiv.org/abs/2406.06282`; project page `http://www.powerinfer.ai/v2`. (c) *TurboSparse: A Simple, Effective, and Efficient Large Language Model*, arXiv:2401.14209 (2024-01-25); EleutherAI explainer `medium.com/@EleutherAI/turbosparse-explained-26ba1ef6f0ce`. (d) For the power-law-adjacent observation in MoE: *Not All Experts are Equal*, arXiv:2402.18656.

**Principle (formal)**. PowerInfer v1's central empirical claim (arXiv:2312.12456 abstract, retrieved 2026-09-21): "the high locality inherent in LLM inference, characterized by a power-law distribution in neuron activation. This distribution indicates that a small subset of neurons, termed hot neurons, are consistently activated across inputs, while the majority, cold neurons, vary based on specific inputs." The mechanism is a per-neuron **adaptive predictor** that scores activation before the FFN executes, plus **neuron-aware sparse operators** that materialise only the predicted-hot neurons on the GPU. Reported speedup: up to **11.69×** over `llama.cpp` on OPT-175B / RTX 4090 (single 4090, 24 GiB VRAM) and **82 %** of A100 generation rate on OPT-30B / RTX 4090 (author-measured, 2024-12). PowerInfer-2 (2406.06282) reframes the engine as a **polymorphic neuron engine** with neuron-level vs cluster-level indexing, an `Hot/Cold` cache split, and a 5-stage PRED/GIO/GC/UDIO/UDC pipeline that overlaps I/O with compute; reported up to **29.2×** over `llama.cpp` on smartphones. **TurboSparse** (2401.14209) is a model, not an inference engine — it activates only **1–2 experts per token** in a MoE, achieving comparable perplexity with 8× fewer active experts than typical MoE.

**Apply to flyingfish dense int4 paths**. The applicability depends on the activation family, not the quant scheme. (a) **`ff qwen35` (dense groupwise-int4, all projections resident on device at decode, `docs/models.md:121-128`)** uses RMSNorm + **SwiGLU** in the FFN. SwiGLU is `Silu(gate(x)) · up(x)`, where `Silu(x) = x · σ(x)` — strictly non-negative but **smooth** through `x = 0`, not piecewise-linear like ReLU. PowerInfer v1's predictor is trained on per-neuron activations; for a SwiGLU gate, the predictor would have to score on `Silu(gate(x))` rather than `gate(x) > 0`, which costs one extra multiplication but does not break the locality hypothesis — **partial applicability**. (b) **`ff edge0` (hybrid GDN/full-attention MoE with groupwise-int4, `docs/models.md:105-119`)** uses the same SwiGLU at the FFN, plus a routed expert gate whose activation is **softmax over `E` experts** — softmax is positive everywhere, so the predictor would have to score on the routed probability, not on "is this neuron active". The routed probability is highly skewed (top-1 dominates in MoE inference), but the sparsity is at the **expert** axis, not the **neuron** axis — PowerInfer-1's per-neuron mechanism does not apply. (c) **`ff glm` (FP8 MoE, `docs/models.md:37-65`)** likewise has SwiGLU experts — same gating-axis story as `ff edge0`. (d) **H3 / DiT path (`docs/models.md:67-86`)** uses AdaLN-Zero + GELU; GELU is smooth and bounded below by a tiny negative number, **not** a ReLU. **Plain verdict: PowerInfer-1's claim does *not* apply cleanly to any of our dense or MoE paths at the character level.** The closest adaptation is PowerInfer-2's polymorphism, where the cluster-level indexing can be repurposed over SwiGLU `Silu(gate)·up` activations; that is an adapter item, not a free win. **Cross-references**: PowerInfer-2 changes the applicability analysis by adding a neuron-cluster mechanism (2406.06282 §4.1) that does not require ReLU — it is the most plausible port target for `ff qwen35`'s SwiGLU dense path, but the predictor would have to be trained per checkpoint and is not in any public release (PowerInfer-2 ships code only for the FP4 MoE path). TurboSparse is **not applicable** at the inference engine layer — it is a training-time MoE design.

**Verdict: not directly applicable; needs adapter work if pursued.** The most plausible port is PowerInfer-2's neuron-cluster indexing into `crates/ff-qwen35/src/`, conditioned on a per-checkpoint activation-sparsity audit that does not exist today. The current `ff` dense path gains nothing from this line of literature without the audit.

### §12.4 Dense KV cache theory — sinks, heavy hitters, query-aware Top-K

This subsection is **deliberately orthogonal to `docs/speculative-token-tech.md` §3**, which is implementation-level. Here we state the *theoretical* claims; the file:line anchors point to the paper PDFs.

**Primary citations** (all retrieved 2026-09-21). (a) Xiao, Tian, Chen, Han, Lewis, *Efficient Streaming Language Models with Attention Sinks*, arXiv:2309.17453, v4 2024-04-07, accepted **ICLR 2024**; code `https://github.com/mit-han-lab/streaming-llm`. (b) Zhang, Sheng, Zhou, Chen, Zheng, Cai, Song, Tian, Ré, Barrett, Wang, Chen, *H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*, arXiv:2306.14048, v3 2023-12-18, accepted **NeurIPS 2023**; code `https://github.com/FMInference/H2O`. (c) Tang, Zhao, Zhu, Xiao, Kasikci, Han, *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference*, arXiv:2406.10774, v2 2024-08-26, accepted **ICML 2024**; code `https://github.com/mit-han-lab/Quest`.

**Attention sink — formal claim** (Xiao et al. 2023, §3.1, Equation 1 of the ICLR 2024 paper PDF):
> `SoftMax(x)_i = e^{x_i} / (e^{x_1} + Σ_{j=2..N} e^{x_j}),   x_1 ≫ x_j, j ∈ {2, …, N}`.
The argument: `SoftMax` requires its outputs to sum to 1; when a query has no strong semantic match, the model "dumps" the residual probability mass onto a small set of tokens. Empirically (Table 2 of the paper), **four initial tokens** are sufficient to restore Llama-2 perplexity from `5158.07` (window-only) to `5.40` (sink + window), and replacing those four initial tokens with `"\n"` linebreak tokens reproduces the effect (PPL `5.60`) — proving the sink is a **positional** rather than **semantic** phenomenon. The model trains initial-token representations as a sink because initial tokens are visible to all subsequent tokens (autoregressive masking makes the first token the only one with full visibility). This is an existence-and-mechanism claim, not an optimality theorem; the **up-to-22.2×** speedup number is vs the sliding-window-with-recomputation baseline (author-measured on Llama-2-13B on PG-19).

**Heavy hitter oracle — formal claim** (Zhang et al. 2023, §3.2, Definitions 2.1 / 2.2 / 4.1 / 4.3, Theorem 4.4). Per-row execution model:
> `S_i = (S_{i−1} ∪ {i}) \ {u},   u = arg max_{v ∈ S_{i−1} ∪ {i}} F_score((S_{i−1} ∪ {i}) \ {v})`,
where `F_score(T) = Σ_{s ∈ T} o_s` is the cumulative attention score (Algorithm 1 of the paper). Empirical claim (Observation §3.2): cumulative attention scores follow a **power-law distribution**, with sparsity **> 95 %** in nearly every layer (Figure 2(a), OPT family). Theoretical guarantee (Theorem 4.4): under the **dynamic-submodular** assumption (`f(X ∪ {x}) − f(X) ≥ f(Y ∪ {x}) − f(Y)` for `Z ⊂ X ⊂ Y`, `x ∉ Y`), the greedy H₂O eviction satisfies
> `f(S̃_i) ≥ (1 − α)(1 − 1/e) · max_{|S| = k} f(S) − β`,
i.e., within a constant factor of the offline optimum with an `α, β`-perturbation. The constant `(1 − 1/e) ≈ 0.632` is the classical submodular greedy bound; `α, β > 0` capture the relaxations. The reported 29×/3× throughput gains over DeepSpeed / Accelerate / FlexGen are author-measured on OPT-6.7B / OPT-30B with **20 % heavy hitters**.

**Top-K query-aware bound — formal claim** (Tang et al. 2024, §3.4, Algorithm 1). Quest maintains per-page **min/max** metadata: `M_i = max(M_i, k_i)` and `m_i = min(m_i, k_i)` per dimension `i`. The criticality estimator is
> `score = Σ_{i=1..dim} max(q_i · m_i, q_i · M_i) = Σ_{i=1..dim} U_i`,
and the **upper-bound theorem** (the paper's §3.4 claim, derived from `k_i ∈ [m_i, M_i]` for every key in the page) is that `U_i ≥ q_i · k_i` for every key in the page, hence `Σ U_i` is an upper bound on the sum of attention weights across all keys in the page. Top-K pages by `Σ U_i` are loaded for the real attention pass. Reported numbers: **7.03×** self-attention kernel speedup and **2.23×** end-to-end on Llama-2-7B at 32 K context, on an **RTX 4090** (the same GPU flyingfish uses).

**Apply to flyingfish dense int4 paths with bf16 KV** (`docs/models.md:123` describes `ff qwen35` as "dense groupwise-int4 checkpoint … int4 weight + bf16 KV"). The theoretical claims are about `Q · K^T / √d` and the resulting softmax distribution — the KV **data type** (bf16, fp16, int8, fp4) does not appear in any of the three formal statements above, so the theory applies identically. What is **architecturally specific**: H₂O and Quest are validated on OPT, LLaMA, Falcon, MPT, Pythia — i.e., the SwiGLU / RMSNorm family, **the same family `ff qwen35` uses**. The first-two-layers caveat in Quest (§3.4 of the paper — "due to the low sparsity ratio for the first two layers, we only apply Quest and all baselines on later layers") maps onto `ff-qwen35`'s first-two-layers embedding/prefix-LM layers; the rest of the network is fair game. For **`ff edge0`** (MoE), the routing path has different attention dynamics and Quest-style page-level selection would have to be tested per-checkpoint — **pure research watch item**. **Verdict: applicable; needs adapter work** — the smallest port is Quest because it operates at page granularity (matches `candle`'s contiguous-tensor allocation); H₂O requires a cumulative-score path that runs in `O(L^2)` and is **incompatible** with FlashAttention unless an offline-score pass is added.

### §12.5 Granularity theory — per-expert vs per-layer vs per-tensor

**Primary citations**: (a) FlexGen — §4.2 / App. A.2 of arXiv:2303.06865 (per-tensor-page transfer within the block schedule). (b) KTransformers (`github.com/kvcache-ai/ktransformers`) — accessed 2026-09-21 via web-search citations; the per-expert transfer unit is the size of one routed-expert matrix, as documented in `docs/comparable-products.md` §2. (c) FreeToken §1.5 of `docs/speculative-token-tech.md` — per-expert transfer with a routing-locality-weighted LRU. (d) The granularity taxonomy itself: no canonical paper — it is a *systems* taxonomy that emerges from MoE vs dense inference literature.

**Principle (formal)**. Define a **transfer unit** as the smallest tensor chunk that the host/device scheduler moves as one DMA operation. Three regimes:

| Regime | Unit | Reuse key | Index overhead | When it fits |
|---|---|---|---|---|
| Per-expert (MoE) | one routed-expert matrix (gate + up + down, ~50–500 MiB at FP8) | `(layer_id, expert_id)` routed by the gate; reuse within one forward | small (E entries per layer) | MoE where experts are the natural *scheduling quantum* |
| Per-layer (dense, "row") | one transformer block's projections | `(layer_id)`; reuse across the batch dimension | trivial (one entry per layer) | dense prefill with long blocks; row-by-row schedules (the FlexGen row-by-row schedule in §3(a)) |
| Per-tensor-page (FlexGen, "column") | a column slice of one projection (`M/N` rows of one matrix) | `(layer_id, tensor_id, page_id)`; reuse across the column | larger (`N / page_rows` entries per layer × number of projections) | dense decode with high tensor-cache hit rates; FlexGen's zig-zag block schedule in §3(b) |

The trade-off is **standard cache theory**: finer granularity ⇒ larger effective capacity (more orthogonal units to fill the cache) but **larger index structures** and **higher per-fetch metadata overhead**. FlexGen quantifies this in its I/O-optimality proof (Theorem 4.1, App. A.2): the zig-zag block schedule's I/O complexity is **within 2×** of optimal, and the constant drops to 1 as the unit shrinks toward per-element. Below the page granularity, the **DMA-setup cost** dominates and the bound inverts — there is a sweet spot that is checkpoint- and bandwidth-specific.

**When would `ff` use per-tensor page transfer vs per-expert?** The decision is governed by three axes:

1. **Routing axis** — does the workload have a per-token routing decision? MoE: yes (gate output) ⇒ per-expert wins because the routing is the cache key. Dense: no ⇒ per-tensor wins because there is no routing signal to exploit.
2. **Reuse axis** — how often is a unit accessed per forward pass? MoE: each expert is accessed by the routed tokens only; a 256-expert MoE layer with top-2 routing reuses 2 experts per token ⇒ per-expert cache key captures reuse directly. Dense: every projection is accessed once per token ⇒ there is no temporal reuse to exploit beyond the batch dimension.
3. **Metadata axis** — how much RAM can the host cache spend on index structures? MoE: `E` per layer (~256) is small. Dense per-tensor: `L · P · (M / page_rows)` where `P` is projections per layer and `M` is row count — for Qwen3.8-27B (`docs/models.md:121`), `L ≈ 40`, `P ≈ 7`, `M / page_rows ≈ 32` ⇒ ~9000 entries, ~tens of KiB, affordable.

**Apply to flyingfish**. `ff glm` (FP8 MoE) — **per-expert already** (`docs/models.md:39-65` describes `expert-cache-layout shared-pool`, `expert-cache-replacement lfu`, `expert-cache-mib 4096`). `ff edge0` (hybrid GDN/full-attention MoE, `docs/models.md:105-119`) — **per-expert for the routed MoE part, per-tensor for the static projections** (`--resident-experts` makes the static part resident, so it does not need a host tier at all). **`ff qwen35` (dense groupwise-int4, all projections resident on device at decode, `docs/models.md:123-128`)** — currently *all-resident*, so granularity is moot; **if we ever add host offload for the dense path**, per-tensor page transfer à la FlexGen is the right unit, and the existing `q★ = m · B_P / B_H` formulation in `src/host_profile.rs:79-94` is exactly the right first cut — the FlexGen generalisation extends `q★` from a single scalar to a per-proplacement-percentage vector, but the *cost model* in §12.1 has to be filled in. **Verdict: applicable as a per-path decision** — MoE paths stay per-expert (they are correct today); if the dense path ever offloads, the FlexGen per-tensor LP is the right template.

### §12.6 Consolidated relevance matrix

| Topic | Principle | `ff` integration today | Integration cost | Verdict |
|---|---|---|---|---|
| §12.1 FlexGen LP offload | per-tier placement LP, 9-var, 2× optimal | one-shot `q★` scalar in `src/host_profile.rs:79-94` | high (shard-cost model + per-step LP) | needs adapter work |
| §12.2 Prefetch overlap lower bound | `T = max(T_compute, T_move)` + pipeline bubble | implicit in H3 chunk sizes; explicit in FreeToken | moderate (residency check in `crates/ff-h3`) | applicable |
| §12.3 Activation power-law | hot-neuron locality, ReLU-family | not used (we are SwiGLU/softmax) | very high (per-checkpoint predictor) | not applicable at character level; PowerInfer-2 may partially apply with adapter work |
| §12.4 KV cache theory | sink / heavy hitter / query-aware Top-K | not used (KV cache currently unbounded for typical workloads) | moderate (Quest page-granularity matches `candle`) | applicable for `ff qwen35` (same SwiGLU family) |
| §12.5 Granularity | per-expert vs per-layer vs per-tensor-page | per-expert for MoE, all-resident for dense | moderate if dense path ever offloads | applicable as policy |

### §12.7 Data limitations — 未能验证

| Item | Why |
|---|---|
| The exact `bls`, `gbs`, and percentage values that the FlexGen LP returns for Qwen3.8-27B on an RTX 4090 | requires running the published FlexGen LP on the specific `(L, h1, h2, s, n)` tuple; not computed in this round |
| Whether H3's `4096 / 1024 / 1024` chunk sizing satisfies `t_move ≤ t_compute` per stage on the host-cold path | requires per-shard size measurement not present in `crates/ff-h3` today |
| The exact cumulative-attention-score distribution for `ff qwen35` on the 128-token prompt in `docs/models.md` | would require running H₂O's profiling pass against the checkpoint |
| PowerInfer-2 neuron-cluster port to SwiGLU | no published checkpoint-specific results; would need a per-checkpoint activation audit |
| Quest hardware numbers on RTX 4090 with **bf16 KV** rather than the paper's fp16 KV | the paper does not separate the bf16 case; unlikely to differ but not directly verified |
| Whether TurboSparse (the *model*, not the engine) reuses any of PowerInfer-2's inference-engine code | the TurboSparse arXiv is training-only; no inference-engine release located |

---

## §13 The 62 GiB gap — what transfers when the expert pool does not fit in host RAM

This section is the one that §10 of the task brief required. FreeToken — the system surveyed end-to-end in §0–§11 above — **assumes the expert pool (and KV cache, and partial activations) fit in host RAM**. The hardware it was measured against in §1 ranges from "RTX 4060 laptop, 8 GB VRAM + the host's full DRAM" up to "RTX PRO 6000 workstation, 96 GB VRAM + ~512 GB DDR5"; the smallest GLM-5.2 753B-A40B workload listed in the paper is `~433 GB` of weights, which assumes `~500 GB` host RAM is available. Our flyingfish baseline is **62 GiB RAM + 8 GiB swap on an RTX 4090 (24 GiB)** — a hardware posture that is roughly **8–16× smaller** than FreeToken's smallest measured tier, **with a checkpoint (GLM-5.3-Flash FP8 at 306 GiB, `docs/models.md:11`) that already occupies the entire 62 GiB host before any KV cache is taken into account**. The 306 GiB checkpoint is therefore **disk-streamed through `mmap`** rather than RAM-resident from the moment the engine starts.

This is not a footnote. Every assumption in §1–§12 inherits a "host RAM fits the workload" premise that we violate on a per-workload basis. The bullet-by-bullet answer below states which FreeToken mechanisms transfer unchanged, which transfer with shape changes, and which fail outright; the closing matrix maps each to a `crates/` location in `ff` that would absorb (or refuse) the transfer.

### §13.1 What "host RAM fits the expert pool" actually buys FreeToken

To make the rest of the section concrete, list the four mechanisms in §1–§11 that *depend* on the RAM-resident-expert-pool assumption:

| FreeToken mechanism | Source | RAM-footprint requirement | Documented file:line |
|---|---|---|---|
| **Routing-locality weighted LRU** | §3 above | the LRU holds roughly the working-set of experts per layer — FreeToken tests at 16–39 % miss at cache sizes calibrated per consumer tier (§1.5, paper Tables) | `python/freetoken/moe/offload_cache.py:168-201`, `python/freetoken/moe/offload_kernels.py:28-40` |
| **Double-buffered prefill** | §2 above | two host-side pinned buffers per layer (or per expert group); the producer fills the back buffer while the consumer drains the front, on pinned memory obtained via `cudaHostAlloc(Portable|Mapped)` | `python/freetoken/kernel/pinned.py:42`, `python/freetoken/kernel/csrc/pinned_tensor.cpp:71-92` |
| **Bandwidth-adaptive policy `q★ = m·B_P / B_H`** | §1 above | measured bandwidths are **PCIe-gen-N ×16** and **CPU-FMA ceiling** — both are host-RAM-tier measurements, not disk-tier | `python/freetoken/moe/bench_profile.py:165-191`, `python/freetoken/moe/offload_kernels.py:158-162` |
| **Semantic-boundary checkpoint (anchor capture)** | §4 above | the anchor frozen into the radix tree at a tool-call opener, plus SWA eviction read off the anchor, presupposes that the cache line holding the linear state **is** in RAM | `python/freetoken/scheduler/cache.py:148-208` |

When the expert pool exceeds host RAM, mechanism #1 turns into a disk-cached hot-set (the OS page cache plays a role analogous to FreeToken's LRU but invisible to the application). Mechanism #2 becomes a "disk → RAM → GPU" two-stage path with no chance of host-side double-buffering the *whole* pipeline. Mechanism #3's `B_H` number is no longer the right one — the effective bandwidth is now the **disk read rate**, and `B_P` is no longer the PCIe peer-to-peer rate but the cache-hit re-fetch. Mechanism #4's anchor needs the cache line that holds the linear state to be RAM-resident, which it will not be for a 306 GiB model on 62 GiB host if the engine has read other parts of the checkpoint first.

### §13.2 Mechanism-by-mechanism verdict for the 62 GiB / 306 GiB workload

#### §13.2.1 The `q★` policy itself → **transfers**; the **two-bandwidth formula** it consumes must change

The *form* of `q★ = m · frac_q16` is bandwidth-ratio-shaped and is the right way to ask "what fraction of missing experts should run on the GPU vs CPU for this step?". Its operands (`m`, `B_P`, `B_H`) come from `python/freetoken/moe/bench_profile.py:165-191`, and the calibration reads **PCIe ×16** and **CPU FMA** numbers — neither of which exists in our disk-streaming regime.

For our workload, the right `q★` adds a third term to the bandwidth pool. Crucially, the bandwidth terms enter **harmonically**, not arithmetically: serial byte services sum their *times*, so the effective host-side bandwidth is `B_H_eff = (α_R / B_H + (1 − α_R) / B_D)⁻¹`, and the shape becomes
> `q★ ≈ m · B_P / B_H_eff = m · B_P · (α_R / B_H + (1 − α_R) / B_D)`,
where `B_P` is the PCIe peer-to-peer rate (still applies once a host shard has been DMA'd into host RAM), `B_H` is the host compute rate (CPU MoE), `B_D` is the **disk sequential-read rate** (NVMe-class), and `α_R ∈ [0, 1]` is the **fraction of the layer that is currently RAM-resident** (a "host cache hit ratio" semantic — the inverse of FreeToken's per-tier miss). The formula **reduces correctly**: when `α_R → 1`, `B_H_eff → B_H` and `q★ → m · B_P / B_H`, matching FreeToken's published scalar. When `α_R → 0`, `B_H_eff → B_D` and `q★ → m · B_P / B_D`, the disk-bound limit. The choice of harmonic (rather than arithmetic-arithmetic, as a draft of this section had it) matters numerically: a naive arithmetic average `(B_H·α_R + B_D·(1−α_R))` systematically **over-estimates** the host-side effective bandwidth because serial byte-service latencies add, and the over-estimate gets worse as `B_D ≪ B_H`. **Use harmonic.** The difference is that FreeToken's calibration is one-shot because `α_R ≈ 1` for them; ours is per-step because `α_R` is the central moving part. Our existing `ff bench io --profile local-interconnect` (`src/host_profile.rs:79-94` per `docs/comparable-products.md` §1.5 and `docs/models.md:41`) reports `B_P = 18.798 GiB/s`, `B_H = 31.248 GiB/s`, `B_P/B_H = 0.60`, plus `host_expert_share_per_mille = 398`. **The first three are a direct fit for the `q★` formula — the framing already exists in our code; only the disk term is missing.**

**Verdict: §13.2.1 transfers** with the amendment that the one-shot `frac_q16` becomes a per-step value `frac_q16 ≈ B_P / B_H_eff = B_P · (α_R / B_H + (1 − α_R) / B_D)`. The free piece of work is the disk-side `B_D` band on `ff bench io`'s profile output and the per-step `α_R` reading off `mincore(2)`-style residency counters — both straightforward.

> *Self-check on the harmonic mean (line 998 form):* the form is `q★ = m · B_P · (α_R / B_H + (1 − α_R) / B_D)`, dimensionally `(GiB/s) · (ratio / (GiB/s)) = ratio`, which matches `q★`'s rate-fraction reading. Serial byte-service times sum: the effective host-side throughput is bounded by the slower of the two legs (`B_H` when RAM-resident, `B_D` when disk-resident), with α_R weighting how often we are in each. The naive arithmetic mean denominator `(B_H · α_R + B_D · (1 − α_R))` (which an earlier draft had at line 998 before this round) is a *time-weighted average* — the correct one for serial byte services but in the wrong direction (it would treat `B_H` and `B_D` as **additive seconds** rather than **additive inverse-seconds**) — and produces a denominator that is **larger than the harmonic one** when `B_D ≪ B_H`, hence a `q★` that is **systematically smaller** than the bound. The harmonic form above is the right one for a host-side bandwidth bound; it gives the correct reduction at both extremes (`α_R = 1 → B_P/B_H`; `α_R = 0 → B_P/B_D`).

#### §13.2.2 Routing-locality weighted LRU → **does not transfer as a primary cache**; the **OS page cache plays the role** for our tier

FreeToken's LRU (`offload_cache.py:168-201`) tracks `(layer, expert)` pairs on **GPU-managed pinned host memory**. The whole point is the application controls eviction — most-frequently-accessed experts stay resident in pinned RAM and the cold ones are evicted. For our workload the application has nothing to track at all: the OS maps the 306 GiB mmap into the virtual address space, the kernel's **page cache** holds whatever 62 GiB (or 70 GiB with swap) of pages the disk readahead has left in RAM after the recent accesses, and the kernel's `LRU`/`TwoQ` algorithm evicts cold pages without our input. Our **observed cache hit rate** is precisely the OS page-cache hit rate. There is no place to insert an application-level LRU that the kernel hasn't already done for us, because the LRU IS the kernel's job once the data path is `mmap`. (When the working set exceeds RAM, the kernel **evicts** file-backed clean mmap pages — *drop-and-refetch from disk on next access* — and writes dirty file-backed pages back to disk before reclaiming them. **Swap applies only to *anonymous* pages** (stack, heap, anonymous mmap), not to the file-backed mmap that holds the checkpoint. For our workload the 8 GiB swap (`docs/models.md:5` "62 GiB RAM with 8 GiB swap") handles our own heap and the temporary staging buffers, *not* the 306 GiB checkpoint — the checkpoint's resident set is bounded by `MIN(62 GiB host RAM, kernel page-cache pressure)`, which is well below the checkpoint size and is the *real* working-set constraint.)

**The FreeToken implication is two-part.** (a) The OS page cache is what FreeToken's LRU would have been; we do not write it, we inherit it. (b) The OS's LRU decision is **per-page (4 KiB)**, and FreeToken's is per-expert (50–500 MiB). When the page-cache hit rate drops below the expert-cache hit rate (because pages are finer-grained than experts and individual pages may be kicked out before the expert is reused), **our effective cache hit rate is lower than FreeToken's calibration would predict** — the GLM `expert-cache-mib 4096` (`docs/models.md:60`) budget and the OS page cache's working-set budget are different cost models, and `q★` between them should be `B_P · (kernel_readahead) / (B_H · α_R + B_D · (1 − α_R))` rather than a flat per-checkpoint measurement. **Verdict: §13.2.2 does not transfer as the primary cache**; we instead observe the same phenomenon (access-locality → cache hit) **through the OS page cache**, and `α_R` becomes the empirically observed hit ratio.

#### §13.2.3 Double-buffered prefill → **partially transfers as a no-op** — pinned buffers exist but the producer stage is disk

`pinned_tensor.cpp:71-92` allocates pinned host buffers via `cudaHostAlloc(Portable|Mapped)` (`python/freetoken/kernel/pinned.py:42`). The pinned buffer is a small (per-layer) **destination**; the **producer** that fills it is the next call to `cudaMemcpyAsync` from the device side or a `cudaMemcpy` from the host's mmap. For our disk-streaming regime the producer is no longer RAM — the producer is the **disk read into the pinned buffer**, and that read cannot be a producer because disk read latency is the dominant cost, not PCIe latency. The double-buffering trick only helps when the producer's median-latency distribution is dominated by PCIe (or some slower but bounded source); for disk, the producer itself is the bottleneck.

This does **not** kill the technique. It narrows where it helps. The pinned-buffer + async-copy pattern is still useful for the **GPU↔RAM** leg of an explicit two-stage pipeline — i.e., after the OS page cache has already pulled the shard from disk into RAM, the GPU↔RAM copy can be double-buffered. So the FreeToken double-buffer applies **between RAM and GPU**, not **between disk and GPU**. The net gain from double-buffering that single stage on our workload is **bounded by the GPU↔RAM PCIe rate (≈ 25 GB/s on PCIe 4.0 ×16) over the per-layer tensor size** — a fraction of the wall time **per access**, but the cumulative gain across the 49 evaluations of H3 (`docs/models.md:11`) is small unless the per-step host residency is high. In our regime, the dominant gain is therefore at the **disk→RAM** step, where the OS page cache + our `mmap`-with-readahead (`crates/ff-h3/src/weights.rs`) does the equivalent of double-buffering transparently.

**Verdict: §13.2.3 transfers only for the GPU↔RAM leg.** The disk→RAM leg is supplied by the OS, with its own (less obvious) batch-ahead logic.

#### §13.2.4 Semantic-boundary checkpoint (`<|tool_call|>` anchor) → **does not transfer as written**; the *insight* transfers — trigger persistence at the boundary, don't pin GBs in RAM

`python/freetoken/scheduler/cache.py:148-208` snapshots the radix-tree anchor and the SWA linear state at the tool-call opener. The anchor is stored in **RAM** (the working set the LRU just admitted). For our disk-streamed regime the snapshot must persist across a kernel page-eviction that may evict the linear state. The naive first reading is "**mlock** the anchor's bytes (or `MAP_POPULATE` the backing pages for the lifetime of the GDN/SWA window)". That reading is **wrong for `ff`**: we already have an *out-of-RAM* persistence path that does the right thing — the H3 evaluation **durable checkpoint** machinery (the `DenoiseCheckpointEvent + recoverable_state` surface in `crates/ff-h3/src/pipeline.rs`, `crates/ff-h3/src/h3_conditioning.rs`, and `crates/ff-h3/src/video_vae.rs`, sitting on the disk-backed `ArtifactStaging` primitives in `src/durable_fs.rs`, with end-to-end recovery wired at `ff video generate` resume per `README.md`'s "Reusing the output directory resumes an interrupted run") — that already survives process restart across arbitrary working-set pressure. `mlock`-ing linear state into a 62 GiB host is also impossible **physically** for the upcoming `ff-dsv41` 189 GiB engram tables, so the `mlock` framing would never even have been the right path for the harder case.

The **transferable insight** from FreeToken's `<|tool_call|>` anchor is **not** "pin this cache line in RAM" — it is "trigger a persistence boundary at this token so the engine can recover from the disk on the other side". That is a model-level invariant, not a memory-level guarantee, and it is *already* the contract `ff-h3`'s durable checkpoint machinery exposes (`DenoiseCheckpointEvent` + `recoverable_state`). What is **not** yet wired is the *trigger*: H3 currently emits a `DenoiseCheckpointEvent` at every denoise step (49 per run), not specifically at a tool-call opener or a thinking boundary. The transferable FreeToken finding is therefore an **adapter task** on the trigger side, not a memory-pinning task.

This plugs straight into the three-way open decision for the `ff-dsv41` adapter's host-tier design:
1. **(a)** *mmap + OS page cache* (the current `ff-h3` default for the host tier — the kernel owns eviction).
2. **(b)** *bounded hash → slice LRU* (the application-owned-cache tier; FreeToken's direction, with the routing-locality weighting from §3.2).
3. **(c)** *admission rejection* (refuse prompts whose working set provably exceeds the bound; already in `ff-glm`'s admission reserve and `docs/...` commit `00b2fb1` "Persists structured capacity shortfalls on candidate rejections").

The FreeToken semantic-boundary insight is a **fourth axis that crosses (a)/(b)/(c)**: *at a semantic boundary, snapshot the relevant state to durable storage and let eviction proceed*. Under (a) the snapshot is the host's existing checkpoint file; under (b) the snapshot is keyed per-expert and refetchable on demand; under (c) the snapshot itself is the rejection log. **All three** preserve the FreeToken guarantee without any 60+ GiB `mlock`. The thinking-boundary case (which §4 noted is parsed but **未在代码中找到** in FreeToken) maps cleanly onto the same trigger-engine — but again, via *persistence-on-boundary*, not mlock.

**Verdict: §13.2.4 does not transfer as written, but the insight does.** The mlock framing is wrong; the durable boundary trigger is right and is already on `ff-dsv41`'s open-decisions list as a sub-question of the (a)/(b)/(c) three-way decision.

### §13.3 If we wanted to port FreeToken-end-to-end: how much host RAM would we need?

`docs/models.md:39-65` records `host_expert_share_per_mille: 398` for GLM-5.3-Flash under `--expert-cache-mib 4096` (4 GiB host cache) on a **306 GiB FP8 checkpoint**. Per expert set, `expert_cache_replace.py` reports **38140 misses / 256 entries** across the three runs — that is **149 misses per expert** over the 32-token prompt evaluated in the row. Each miss is a full expert read at the cache's I/O cost. For a 306 GiB checkpoint on a 62 GiB host, the only conclusion that makes sense is **disk-streaming with an in-RAM working set** that fits in `--expert-cache-mib 4096`, rather than the full expert pool in RAM.

To match **FreeToken's "expert pool fits in RAM" assumption** for GLM-5.3-Flash we would need approximately the **whole 306 GiB checkpoint plus a working-set slack** in host RAM — i.e., a workstation-class ~384–512 GB DDR5 host. Our 62 GiB is **6–8× short of that**. We are not FreeToken's target tier for this workload. The honest framing is:

> flyingfish on 62 GiB host is the **edge of edge** tier — the layer below any published "edge-native MoE serving" claim that assumes RAM-resident expert pools.

If we ever migrate to RAM-resident expert pools, the work is a host-RAM capacity *and* a checkpoint fabric redesign:

1. **Host-RAM capacity**: physical machine upgrade (or a workstation-class host with 384+ GB DDR5 and a single RTX PRO 6000 / 8-way H200 node); not a software task.
2. **Checkpoint fabric redesign**: replace `mmap` (kernel page cache, page-level granularity, `4 KiB`) with a checkpoint format that provides **per-expert pages aligned to FreeToken's `(layer, expert)` key** so the application-level LRU has actual ownership of the cache and the kernel page cache does not evict underneath it. This is comparable to FlexGen's per-tensor-page column-slicing (§12.1) and would require a re-shard step on the model.
3. **Two-stage bandwidth model**: harmonic form per §13.2.1 — `B_H_eff = (α_R / B_H + (1 − α_R) / B_D)⁻¹`, `q★ = m · B_P / B_H_eff`, with `α_R` measured live from disk-read vs page-cache-hit counters, replacing FreeToken's static `frac_q16`.

None of these is an `ff` task before the OS-side (or any future hardware-side) changes happen. The current-day engineering limit is **physical**, not algorithmic.

### §13.4 What we *can* lift today (closed list of FreeToken mechanisms that don't require RAM-resident checkpoints)

For completeness, list the **non-RAM-bound** mechanics from §0–§11 that we can adopt as-is or with shape changes:

| Mechanism | Source | Today's situation in `ff` | Verdict |
|---|---|---|---|
| `ft ctl cache rebuild` (§5) | `engine/engine.py:874-953` | `ff` already exposes runtime cache rebuild through `--expert-cache-mib`, `--expert-cache-replacement`, `--expert-cache-readmit`, `--expert-cache-min-mib` (`docs/models.md:60`) | **already present**; no port needed |
| Hybrid fetch-set selection (§3.2) | `offload_cache.py:199-201` + `offload_kernels.py:153-157` | our expert cache is LFU (`shared-pool`), not routing-locality weighted | **liftable**: replace LFU with routing-locality weighted LRU in `crates/ff-glm/src/expert_cache.rs` (low risk; weighting function is straightforward) |
| MoE-strategy auto-detect | `python/freetoken/moe/__init__.py:9-14` | our `--expert-cache-layout` is already auto-detected (`docs/models.md:43` references "explicit zero and false" tests for the layout enum) | **already present** for the cache layout; the strategy-level auto (gpu/cpu/hybrid) is **liftable** as a follow-up to §13.2.1 |
| Calibrated `q★` (`ft bench bw`) | `bench_profile.py:185-187` | `ff bench io --profile local-interconnect` (`docs/models.md:48`) | **already present** in equivalent form; the disk-term addition from §13.2.1 is the only new piece |
| Token-routing gating | `function_call_parser.py:3709-3713` | none | not applicable — text models (GLM, Edge0, Qwen3.5, MiniCPM, V4.1) parse tool calls inside their respective encoders, not from a shared scheduler hook |

The routability items that we **cannot** lift today without checkpoints-in-RAM:

| Mechanism | Why blocked | When becomes liftable |
|---|---|---|
| Tracking the `α_R` measured value | OS page cache is read-only and unexposed; we cannot ask for the hit ratio | only if we re-shard the checkpoint to per-expert pages (§13.3 item 2) |
| Routing-locality weighted LRU **as primary cache** | needs control over cache lines, which the kernel owns | only with the re-shard |
| Pinned `mlock`-style anchor survival | mmap'd state is evictable | only with explicit `mlock` or a custom allocator |
| FreeToken's exact `tf` (tool-call) anchor flow | depends on the per-step cache line | only after the above |

### §13.5 Cross-cutting observation

The matrix above is what makes the FreeToken story **not directly applicable** to `ff`'s published workloads but **directly applicable as research + instrumentation**. We are **below FreeToken's smallest measured tier by an order of magnitude**; this section is the engineering acknowledgement that the order-of-magnitude difference is *fundamental*, not optimisable. Where we *can* adopt FreeToken-equivalent mechanisms — the calibration discipline (`ff bench io`), the hybrid fetch-set selection at the cache-hit level, the runtime cache rebuild hooks — those map onto `ff` code paths already.

This closes the §13 requirement from the brief. The honest reading is that *the parts of FreeToken worth porting to a 62 GiB host are the parts that don't depend on RAM-resident expert pools*. Those parts are good — but the headline 1.3-2.1× decode / 1.19-1.22 s prefill / <44 s TTFT claims (in §9) are **measured against a 192–512 GB host RAM**, and cannot be replicated without that RAM tier. Closest literature comparator on our tier: KTransformers V0.2's 13.69 tok/s decode on a Xeon SPR dual-socket + 1 TB DDR5 + 4090D (`docs/comparable-products.md` §2.2) — a server-class tier ~16× our host RAM. The gap between us and that workload is **RAM tier, not algorithmic**.
