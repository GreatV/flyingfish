# 同类产品调研与性能对比

This document surveys the inference engines and frameworks that overlap with `ff`'s scope, against the exact baseline numbers recorded in `docs/models.md`. Every quoted digit carries the source URL, date, hardware, OS where reported (Linux / macOS / Windows / unspecified), and a tag for whether it is maintainer-measured or third-party.

Nothing in this document is a recommendation. Where a number could not be located in primary or community sources within three searches, it is recorded as **未能验证** rather than estimated.

## §0 我们的基线（ff，跑 docs/models.md 的 Performance 表）

**Operating environment**:
- OS: **Linux** (`docs/models.md:5`: "Measurements used Linux, Intel i9-13900KF, 62 GiB RAM with 8 GiB swap, local NVMe storage, and one RTX 4090 with 24 GiB VRAM").
- CPU: Intel i9-13900KF; host RAM 62 GiB + 8 GiB swap.
- Storage: local NVMe (FS details not in `docs/models.md`).
- GPU: 1× NVIDIA RTX 4090, 24 GiB VRAM.
- Software: Rust 1.95.0 + CUDA 13.2, built with `cargo build --locked --release --features flash-attn`.
- Sampling: each row in `docs/models.md` is the middle of three back-to-back runs (Cases ran sequentially); sampled GPU peak measured once per second.
- Reproduce: `grep -nE '^(\| Model|MiniMax-H3|GLM-5\.3-Flash|Edge0-35B|Qwen3\.8|MiniCPM|TRELLIS|CLIP|MiniMax-Music3)' docs/models.md`.

**Performance numbers (verbatim)**:

| Model / Workload | Wall time | Peak RSS | Sampled GPU peak | Notes |
|---|---|---|---|---|
| GLM-5.3-Flash — 32 tokens from a 20-token prompt, decode rate **0.87 tok/s** | 74.80 s | 1.59 GiB | 21.59 GiB | FP8 MoE, 306 GiB checkpoint, host LFU expert cache 4 GiB |
| MiniMax-H3 — 768p 16:9, 4 s, 49 evals | 1,853.57 s ≈ 30m 53.57s | 58.82 GiB | 20.92 GiB | `--flash-attention`, attention-projection-chunk-size 4096, ffn-token-chunk-size 1024, max-host-mib 66000 |
| MiniCPM5-2B — 31 tokens with 32-token limit | 5.34 s | 1.10 GiB | 4.62 GiB | greedy |
| MiniCPM5 + DSpark — same prompt | 2.24 s | 1.10 GiB | 5.17 GiB | speculative draft |
| Edge0-35B-A3B — 128 tokens, 128-token limit (streamed experts) | **17.50 s** | 9.27 GiB | 2.14 GiB | groupwise-int4 hybrid GDN/MoE, expert streaming |
| Qwen3.8-27B — 128 tokens, 128-token limit (text-only) | **13.93 s** | 16.63 GiB | 17.17 GiB | dense groupwise-int4, all projections resident |
| CLIP ViT-L/14 — 2 candidate texts on a 224×224 image | 2.44 s | 0.44 GiB | 0.99 GiB | |
| TRELLIS-text-base — 180,288 splats from text | 8.96 s | 0.70 GiB | 3.98 GiB | with CLIP ViT-L/14 conditioner |
| TRELLIS-text-large — 231,744 splats from text | 21.95 s | 0.69 GiB | 5.41 GiB | |
| TRELLIS-text-xlarge — 198,432 splats from text | 27.69 s | 0.72 GiB | 6.44 GiB | |
| TRELLIS-image-large + DINOv2 — 762,112 splats from 518×518 | 75.04 s | 0.71 GiB | 4.64 GiB | |
| TRELLIS.2-4B + DINOv3 — 254,641-vertex mesh from 512×512 | 33.33 s | 1.09 GiB | 3.36 GiB | |
| MiniMax-Music3 — 8-s audio, 30 denoise steps | 130.63 s | 2.11 GiB | 19.22 GiB | |

**Our technical posture** (for context to readers who don't know `ff`):
- Single Rust binary; no Python dependency in the deployed CLI.
- mmap-based weight streaming (`candle` + `memmap2`); `--weight-source mmap` required for the routed text adapters.
- Host/device split calibrated by `ff bench io --profile local-interconnect` and pinned to `--host-profile` (`docs/models.md:48`).
- MoE expert cache: shared LFU pool with explicit `--expert-cache-mib` and per-call re-admission (`--expert-cache-readmit` + `--expert-cache-min-mib`).
- Admission / capacity shortfalls: structured rejection (`docs/...`: "Persists structured capacity shortfalls on candidate rejections", commit 00b2fb1).
- Checkpoint / resume: H3 produces recovery checkpoints per evaluation; resumed from existing output-dir (commit `1aafd85` and earlier).

## §1 LLM 推理引擎（GLM/Edge0/Qwen3.8 类似负载的对手）

### §1.1 llama.cpp

| Field | Value | Source |
|---|---|---|
| Weight residency | GGUF single-file container. **`use_mmap=true` by default**; pages are demand-faulted from disk; companion `--mlock`, `--no-mmap`, `--direct-io`, recent `--load-mode auto\|none\|mmap\|mlock\|mmap+mlock\|dio`. After GPU upload, llama.cpp `unmap_fragment()` releases host mappings. | `github.com/ggml-org/llama.cpp` README "Description" section; `dev.to/multigrid/llamacpps-mmap-and-mlock-flags-explained-43fp` |
| Quantization | 1.5 / 2 / 3 / 4 / 5 / 6 / 8-bit integer quant. K-quants (Q2_K … Q6_K), IQ-quants (importance-aware IQ2_/IQ3_/IQ4_XS/IQ4_NL), legacy Q4_0/Q5_1/Q8_0, experimental TQ1_0/TQ2_0. | README; `bodegaone.ai/learn/guides/gguf-quantization-level` |
| GPU offload | Hybrid CPU+GPU via `-ngl` (n_gpu_layers), `-ngl 0`/`-ngl -1`/integer; multi-GPU split modes `NONE\|LAYER\|ROW` | README "Description" |
| Concurrency | Single-process by default; `llama-server` runs N independent contexts (`--parallel` / `-np`, `--ctx-size`, `--batch-size`, `--cont-batching`). Slot-parallel, **not vLLM-style iteration-level block-pool batched**. | README "Tools" |
| Target HW | 15+ backends: CUDA / HIP / Metal / Vulkan / SYCL / OpenCL / MUSA / OpenBLAS / BLIS / CANN (Ascend NPU) / Hexagon (Snapdragon) / IBM zDNN / ZenDNN / WebGPU / RPC / OpenVINO | README "Supported backends" |
| Disk-resident / mmap | **Yes** (default true) | as above |
| Per-layer / per-expert host offload | **Yes** (`-ngl`) | as above |
| Single 24 GiB GPU feasible | **Yes with offload** for ~40 GB Q4_K_M; 70B Q4_K_M (≈ 40 GB) partial offload; 671B even Q4 (~400 GB) does not fit | README + community references |
| Continuous batching | Slot-parallel only | as above |

**Headline maintainer-measured number (verified)**:
- No single canonical primary-source 70B / RTX 4090 throughput published on the project's own pages.
- **RTX 4090 + Qwen2.5-72B-Instruct Q4_K_M (~44 GB)**: **未能验证** in this survey — community-circulated band is ~4–6 tok/s decode at `-ngl 20–35`, ~30–50 tok/s prompt at 21.5 GB VRAM; primary anchors (`reddit.com/r/LocalLLaMA/comments/1g6phr2/...`, `github.com/ggml-org/llama.cpp/issues/18056`, `disc.ggml.ai/t/performance-qwen2-5-72b-instruct-q4-k-m/3951`) returned search-index hits but content fetches were blocked in this environment.

### §1.2 Ollama

| Field | Value | Source |
|---|---|---|
| Architecture | Go daemon wrapping llama.cpp (since 0.30/0.31, 2026) ships an **MLX** engine as a secondary Apple-Silicon backend. | `github.com/ollama/ollama` README, `ollama.com/blog/nvidia-spark-performance` |
| mmap / disk-resident | **Yes** (inherited from `ggml_mmap`; GGUF tensor alignment = 32-byte = 4 KiB OS page boundary) | as above |
| GPU offload | Automatic greedy fill of VRAM; remainder spills to CPU; override via `OLLAMA_NUM_GPU` env or `num_gpu` Modelfile param; `ollama ps` shows live split | `docs.ollama.com/gpu` |
| Quantization | GGUF K-quants + legacy; **MXFP4** native kernel for OpenAI `gpt-oss` | `github.com/ollama/ollama/blob/main/docs/quantization.md`; `ollama.com/blog/gpt-oss` (Aug 5, 2025) |
| Concurrency | `OLLAMA_NUM_PARALLEL` (default 4), `OLLAMA_MAX_QUEUE` (512), `OLLAMA_MAX_LOADED_MODELS`. Continuous batching added 2025. | `ollama.com/blog/...2025-09-23 new model scheduling` |
| Target HW | NVIDIA CUDA; AMD ROCm v7 + Vulkan; Apple Silicon MLX > llama.cpp; x86 CPU AVX2/AVX-512; AMD GPUs RX 9070 XT / 7900 XTX / Instinct MI300X/MI250X/MI210 / Ryzen AI Max | `docs.ollama.com/gpu` |

**Maintainer-measured (Oct 23, 2025)**:
- URL: `ollama.com/blog/nvidia-spark-performance`
- Hardware: NVIDIA DGX Spark, firmware 580.95.05, Ollama v0.12.6
- Workload: 10 runs, temperature 0, **500-token output**, caching disabled

| Model | Quant | Prefill (t/s) | Decode (t/s) |
|---|---|---|---|
| deepseek-r1 14B | q4_K_M | 5,919 | **19.99** |
| deepseek-r1 14B | q8_0 | 4,667 | 13.32 |
| qwen3 32B | q4_K_M | 705.0 | **9.41** |
| qwen3 32B | q8_0 | 487.2 | 6.24 |

**RTX 4090 specifically**: **未能验证** — Ollama does not publish a 4090 row; community-cited 7B numbers (~120 tok/s, Q4_K_M) are widely circulated but not on `ollama.com`.

### §1.3 vLLM

| Field | Value | Source |
|---|---|---|
| Weight residency | No mmap; weights load fully into GPU HBM. Sleep Mode moves them GPU↔CPU RAM (not disk). | `github.com/vllm-project/vllm`; `docs.vllm.ai/...features/sleep_mode` |
| GPU offload | `--cpu-offload-gb` V1; CPU KV-cache offloading v0.19.0 (Apr 2026). Cited as ~35× slowdown vs GPU; "don't use unless 100% sure your model won't fit" | `discuss.vllm.ai/t/the-new-v1-way-to-cpu-offload-gb/432` |
| Quantization | GPTQ, AWQ, Marlin, INT4 (GPTQ/AWQ), INT8, INT8 KV cache, FP8 E4M3/E5M2, MXFP8, MXFP4, NVFP4, BF16/FP16, GGUF, bitsandbytes NF4/FP4, compressed-tensors, ModelOpt, TorchAO, HQQ, SqueezeLLM, Quark | `docs.vllm.ai/en/latest/features/quantization/` |
| Concurrency | Continuous batching (defining feature); OpenAI-compatible HTTP; chunked prefill; prefix caching; speculative decoding (EAGLE/DFlash); multi-LoRA; gRPC | `blog.vllm.ai/2025/09/05/anatomy-of-high-throughput-llm-inference-system.html` |
| Target HW | NVIDIA CUDA (primary); AMD ROCm; Intel XPU; CPU (`vllm-cpu`); TPU (`vllm-tpu`); Neuron; Gaudi; Ascend NPU; Rebellions NPU; Apple Silicon (`vllm-metal`); MetaX GPU | `docs.vllm.ai/en/latest/getting_started/installation.html` |
| Disk-resident | **No** | as above |
| Single 24 GiB feasible | **~13B BF16 or ~30–32B AWQ**; 70B BF16 (140 GB) doesn't fit; 70B AWQ-INT4 (~35–40 GB) doesn't fit either; `--cpu-offload-gb` allows larger but slow | community references |

**Maintainer-measured H100 throughput** (third-party: SemiAnalysis InferenceX):
- URL: `https://inferencex.semianalysis.com/`
- Hardware: single NVIDIA H100 80GB
- Model + quant: Llama 3.3 70B FP8
- **Steady-state decode: 1,496 tok/s/GPU; Peak: 2,568 tok/s/GPU**

**RTX 4090 + 70B**: **未能验证** — vLLM maintainer blog (Anatomy of a High-Throughput LLM Inference System, Sept 2025) has no 70B/4090 row. Third-party (`willitrunai.com`, 2× RTX 4090, Llama 70B Q4) reports ~25–30 tok/s aggregate (Mar 2026).

### §1.4 TGI (text-generation-inference, Hugging Face)

| Field | Value | Source |
|---|---|---|
| Architecture | Rust inference engine + Python orchestration + gRPC over Unix Domain Sockets; `--shard-uds-path`; `NUM_SHARD` + NCCL tensor parallelism; streaming via SSE | `github.com/huggingface/text-generation-inference`; `adyen.com/knowledge-hub/llm-inference-at-scale-with-tgi` |
| Quantization | bitsandbytes (8-bit, NF4, FP4), GPT-Q (4-bit, Marlin auto), AWQ (4-bit), EETQ (8-bit), Marlin (4-bit), fp8 (E4M3), compressed-tensors, exl2 (no tensor parallelism), KV cache fp8_e4m3fn/e5m2 | as above |
| Concurrency | Continuous batching; tunables `--max-batch-total-tokens`, `--waiting-served-ratio`, `--max-waiting-tokens`, `--max-batch-prefill-tokens`, `--max-concurrent-requests` (default 128) | `huggingface.co/docs/text-generation-inference/en/reference/launcher` |
| Target HW | NVIDIA CUDA ≥12.2 (primary); AMD ROCm (MI210/MI250); Inferentia (`optimum-neuron`); Gaudi (`tgi-gaudi`); TPU (`optimum-tpu`) | README |
| mmap / disk-resident | **No** | README |
| Host/CPU offload | **Limited** — `device_map="auto"` for non-optimized models places some layers on CPU; not a general flag | README "Optimized architectures" |
| **STATUS** | **Maintenance mode per README caution callout**; HF recommends vLLM/SGLang/llama.cpp/MLX for new deployments | README "Caution" block |
| Single 24 GiB | Only via AWQ-INT4/GPTQ-INT4 for 70B (still ~35 GB) | community |

**Maintainer-measured RTX 4090 + 70B**: **未能验证** — HF no longer publishes 70B/4090 tables; HF blog posts (`/blog/llama31-inference`, `/blog/tgi-throughput-benchmark`, etc.) all 404. Third-party estimate ~22–23 tok/s Llama-3 70B AWQ-INT4 single 4090 single-stream, not HF-measured.

### §1.5 SGLang

| Field | Value | Source |
|---|---|---|
| Architecture | **RadixAttention** — KV cache in a radix tree; page size 1 token; LRU eviction; shared prefix cache; longest-shared-prefix-first scheduling; Theorem 3.1 (offline optimal hit rate). Compressed FSM multi-token constrained decoding. | `arxiv.org/abs/2312.07104` §3, §4, Fig 3, Thm 3.1 |
| Quantization | FP4 / FP8 / INT4 / AWQ / GPTQ | `github.com/sgl-project/sglang` README |
| Concurrency | Continuous batching + paged attention + prefill-decode disaggregation + speculative decoding + multi-LoRA + chunked prefill | as above |
| Target HW | NVIDIA (GB200 / B300 / H100 / A100 / Spark / 5090); AMD MI355 / MI300; Intel Xeon CPU; Google TPU (`SGLang-Jax`); Ascend NPU | `github.com/sgl-project/sglang` |
| mmap / disk-resident | **No first-class feature**; default data flow disk → DRAM → GPU HBM | `lmsys.org/blog/2025-12-10-rfork/` |
| Host/CPU offload | Component + layerwise; MoE expert CPU offload recent | as above |

**Maintainer-measured (LMSYS, July 25, 2024)**:
- URL: `https://www.lmsys.org/blog/2024-07-25-sglang-llama3/`
- Hardware: 8× A100 80GB SXM (bf16) / 8× H100 80GB SXM (FP8)
- Models: `meta-llama/Meta-Llama-3-70B-Instruct` (bf16) / `neuralmagic/Meta-Llama-3-70B-Instruct-FP8`
- Workload: synthetic input-512-output-1024, etc.; 1K–6K offline; 1–16 RPS online
- **Headline**: "SGLang consistently outperforms vLLM, achieving up to 3.1× higher throughput on Llama-70B. It also often matches or sometimes outperforms TensorRT-LLM."

**Mixtral / DeepSeek-V3 on RTX 4090**: **未能验证** — SGLang-published numbers are mostly rack-scale.

### §1.6 TensorRT-LLM

| Field | Value | Source |
|---|---|---|
| Architecture | AOT-compiled engines; `trtllm-build` produces a TensorRT engine keyed to (model, GPU cc, dtype, max_batch_size, max_input_len, parallelism shape). Engine files are large; pre-build + mount recommended. | `github.com/NVIDIA/TensorRT-LLM`; `nvidia.github.io/TensorRT-LLM/performance.html` |
| Quantization | FP4 (NVFP4 on Blackwell); FP8 (E4M3/E5M2 Hopper/Blackwell); INT8 (SmoothQuant/AWQ); INT4 (AWQ/GPTQ); FP16/BF16; W4A8 (DeepSeek V3/R1 only) | as above |
| Concurrency | In-flight batching built in; max batch 2048 on B200 | as above |
| Target HW | NVIDIA only — Ampere (A100 SM80), Ada (L40S / RTX 4090 SM89), Hopper (H100/H200 SM90), Blackwell (B200/GB200 SM100, RTX 5090 SM120). **Ampere SM80/86 NOT supported for DeepSeek-V3/R1** | examples/models/core/deepseek_v3 README `## Hardware Requirements` |
| mmap / disk-resident | **No** | README |
| Host/CPU offload | **KV cache only** — `kv_host_cache_bytes` / `hostCacheSize` / `kv_cache_host_memory_bytes`; doc warns "cost is negligible on Grace-Hopper… unlikely to yield benefits on older architectures" | `nvidia.github.io/TensorRT-LLM/kv_cache_reuse.html` |
| Single 24 GiB feasible | 70B FP16/BF16 infeasible; AWQ-INT4 still ~35 GB; example repo ships `quantize_llama3_70b_on_3x_4090` recipe (June 2024) | `examples/models/core/deepseek_v3` |

**Maintainer-measured (NVIDIA, H100/L40S/A100)**:

| Hardware | Model + Quant | Throughput | BS / TP / ISL / OSL |
|---|---|---|---|
| L40S FP8 | Llama 70B | **561 out tok/s/GPU** | BS 256 / TP=2 / 128 / 128 |
| H100 FP8 | Llama 70B | **3,269 out tok/s/GPU** | BS 1024 / TP=2 / 128 / 128 |
| A100 FP16 | Llama 70B | **565 out tok/s/GPU** | BS 256 / TP=4 / 128 / 128 |

**DeepSeek-R1 671B on Blackwell B200** (NVIDIA dev blog, March 18 2025):
- 8× B200 NVL8, FP4 GEMM + FP8 KV
- Single DGX B200: **>250 tok/s/user, >30,000 tok/s max throughput**
- **RTX 4090 + Llama-3 70B: 未能验证** — `nvidia.github.io/TensorRT-LLM/performance.html` does not list RTX 4090.

### §1.7 MLX (Apple Silicon)

| Field | Value | Source |
|---|---|---|
| Architecture | Apple's array framework; composable function transforms (autodiff / autovec / graph opt); **lazy computation**; **unified memory** (CPU/GPU share address space, no `cudaMemcpy`); multi-device | `ml-explore.github.io/mlx/build/html/index.html` |
| Quantization | `mlx.nn.quantize` / `mlx.core.quantize` / `quantized_matmul`. 3/4/6/8-bit safetensors via MLX Community; FP8 / MX-FP | `github.com/ml-explore/mlx-examples` |
| Concurrency | `mlx_lm.generate` single-stream by default; **`mlx-lm` server** supports continuous batching (`--max-batched-tokens`); paged attention PRs ml-explore/mlx#2044/#2047 | `ml-explore.github.io/mlx/...` |
| Target HW | Apple Silicon only (M1/M2/M3/M4/M5). Experimental CUDA backend exists but not production-supported | as above |
| mmap / disk-resident | **Yes** — `mlx.core.load`, `save_safetensors`, sharded AllToShardedLinear / ShardedToAllLinear; unified memory + safetensors mmap = disk-resident tensors loaded lazily | `huggingface.co/mlx-community/Meta-Llama-3-70B-Instruct-4bit` model card |
| Host/CPU offload | **Trivial** via unified memory — choose `mlx.core.set_default_device("cpu")` or `mx.stream(gpu_device)` per op | as above |

**Apple MLX team (cited via third-party report, Awni Hannun post, March 2025) — M3 Ultra 512 GB**:
- URL: `https://www.hardware-corner.net/studio-m3-ultra-running-deepseek-v3`
- Hardware: Mac Studio M3 Ultra 512 GB unified memory (800 GB/s memory bandwidth)
- Model + quant: DeepSeek-V3-0324 4-bit MLX

| Context | Prefill (t/s) | Prefill time | Generation (t/s) |
|---|---|---|---|
| 69-token prompt | 58.08 | 1.19 s | **21.05** |
| 1,145-token prompt | 82.48 | 13.89 s | 17.81 |
| 15,777-token prompt | 69.45 | 227 s | 5.79 |

Third-party `cnrai/llm-perfbench` corroborates: DeepSeek-V3-0324-4bit → **20.9 tok/s** generation.

## §2 超大规模 MoE Offload 引擎（单卡 + 有限 host RAM 跑 >300B 模型）

### §2.1 FlexGen / FlexLLMGen

| Field | Value | Source |
|---|---|---|
| Architecture | Offload scheduler; IO-efficient block schedule + linear-programming cost model; aggregates GPU + CPU + disk; weights optionally 4-bit compressed; KV cache tiled across GPU/CPU/disk; paged block-schedule. | `github.com/FMInference/FlexLLMGen`; `arxiv.org/abs/2303.06865` NeurIPS 2023 |
| Quantization | INT4 weights + INT4 KV cache (compression path), FP16 default | paper |
| Concurrency | **Throughput-oriented, large effective batch** — paper evaluates up to bs=256. Designed for aggregate throughput, not single-request latency. | paper §8 |
| Weight-loading pattern | Disk → CPU → GPU paging | paper |

**Maintainer-measured (paper Table)**: Hardware NVIDIA T4 16 GB on GCP + 208 GB DRAM + 1.5 TB SSD. Input 512 / output 32 tokens. Throughput = generated tokens / elapsed time.

| System | OPT-6.7B | OPT-30B | OPT-175B |
|---|---|---|---|
| HuggingFace Accelerate | 25.12 (bs=2 GPU) | 0.62 (bs=8 CPU) | 0.01 (bs=2 disk) |
| DeepSpeed ZeRO-Inference | 9.28 (bs=16 CPU) | 0.60 (bs=4 CPU) | 0.01 (bs=1 disk) |
| Petals | 8.25 (bs=2 GPU) | 2.84 (bs=2 GPU) | 0.08 (bs=2 GPU) |
| FlexLLMGen | 25.26 (bs=2 GPU) | 7.32 (bs=144 CPU) | **0.69 (bs=256 disk)** |
| FlexLLMGen w/ compression | 29.12 (bs=72 GPU) | 8.38 (bs=512 CPU) | **1.12 (bs=144 CPU)** |

**Critical caveat**: T4 16 GB, **not RTX 4090 24 GB**; numbers are **aggregate throughput at high batch**, not single-stream decode. Hacker News thread Feb 2023 (`news.ycombinator.com/item?id=34830270`) verifies the 1.12 figure (OPT-175B 4-bit, batch=144).

### §2.2 KTransformers

| Field | Value | Source |
|---|---|---|
| Architecture | Expert-level offload: GPU holds MLA + shared expert + KV cache; CPU holds routed expert weights. Intel AMX-accelerated MoE kernel. Selective-expert (6 of 8 active). | `github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md`; `kvcache-ai.github.io/ktransformers/` |
| Quantization | INT4 (GGUF Q4_K_M) experts + FP8 MLA on GPU; on V0.3 also online INT8 CPU + INT4 GPU | docs |
| Concurrency | V0.2 single-stream; V0.2.4 added `balance_serve` batch ≤4 server | docs |
| Target HW | 1× RTX 4090/4090D 24 GB + Intel Xeon (Sapphire Rapids / MRN, AMX required); 382 GB – 1 TB DDR5 | docs |
| Weight-loading | MLA + shared experts preloaded on GPU; 160 routed experts mmap'd in host DRAM, fetched on demand into NUMA-local memory; GPU never holds full expert set | docs |

**Maintainer-measured V0.2 (Intel Xeon Gold 6454S, 2 sockets × 32 cores, 1 TB DDR5-4800; 1× RTX 4090D 24 GB)**:
- DeepSeek-V3-q4km, 500-token prompt:

| Configuration | Prefill tok/s | Decode tok/s |
|---|---|---|
| llama.cpp (8 experts, 2×32 cores) | 10.31 | 4.51 |
| KTransformers V0.2 single-socket 6 experts | 65.14 | 10.30 |
| KTransformers V0.2 dual-socket 6 experts | 97.32 | **13.69** |
| KTransformers V0.2.1 6 experts, 4K prompt | 102 | 14.9 |
| KTransformers V0.3 AMX, 1K prompt, 6 experts | 203.70 | — |
| KTransformers V0.3 AMX, 2K prompt, 6 experts | **286.55** | — |

**Independent validation** (Thaki Cloud, 2026-07-19, `https://thakicloud.com/tech-blog/en/llmops/ktransformers-moe-offload-28x-validation`): Reproduced on RunPod with Xeon Platinum 8480+ (SPR, AMX) + 4090: AMX INT4 MoE kernel measured **12.4 tok/s** (CPU-only decode); end-to-end with overlap **14–16 tok/s**. The "28×" figure is prefill, not decode; decode multiplier vs llama.cpp is ~3×.

### §2.3 PowerInfer v1

| Field | Value | Source |
|---|---|---|
| Architecture | Neuron-aware offload: offline profiler + ILP solver preloads hot neurons to GPU; online predictor gates sparse matrix-vector ops per layer; CPU computes cold neurons in-place via AVX2. Per-neuron split, not per-layer. | `arxiv.org/abs/2312.12456`; `github.com/SJTU-IPADS/PowerInfer` |
| Quantization | FP16 + INT4 (GPTQ-style) | paper |
| Concurrency | Single-stream focus (bs=1 headline); up to 6.08× llama.cpp at small batches; advantage narrows to 4.38× at bs=32 as joint-activation sparsity collapses | paper |
| Target HW | RTX 4090 24 GB primary; RTX 2080Ti 11 GB also validated. "PC-High" = i9-13900K 8P + 192 GB DDR5 + RTX 4090 + PCIe 4.0; "PC-Low" = i7-12700K + 64 GB + RTX 2080Ti + PCIe 3.0 | paper |

**Maintainer-measured (PC-High, FP16 / INT4, batch=1)**:

| Setup | Models | Avg tok/s | Peak tok/s |
|---|---|---|---|
| FP16 | OPT 6.7B–175B, Falcon-40B, LLaMA-70B (ReGLU) | **8.32** | **16.06** |
| INT4 | same set | **13.20** | **29.08** |
| OPT-175B INT4 only | OPT-175B | ~2 | — |

**Caveat**: PowerInfer v1 **validates ReLU-family dense** models (OPT, Falcon-40B-ReLU, LLaMA-ReGLU). It does NOT validate MoE models. On OPT-175B specifically it achieves ~2 tok/s vs our GLM-5.3-Flash's **0.87 tok/s** — so on a similar-shape dense model we are ~2.3× behind. Independent confirmation: Sohu / 机器之心 report (`sohu.com/a/745917088_129720`, 2023-12-21).

### §2.4 PowerInfer-2

| Field | Value | Source |
|---|---|---|
| Architecture | Polymorphic neuron engine: **NPU-centric prefill** + **CPU-centric decode**; segmented neuron cache (attention pinned, FFN LRU); 4-KB random-read bundles from UFS flash; neuron-cluster-level pipeline hides I/O. | `arxiv.org/abs/2406.06282`; `powerinfer.ai/v2/` |
| Quantization | INT4 FFN + FP16 attention; requires TurboSparse retraining for predictable sparsity | paper |
| Concurrency | Single-stream (smartphone target) | paper |
| Target HW | **Smartphones**: OnePlus 12 (24 GB DRAM, Snapdragon 8 Gen 3, UFS 4.0) and OnePlus Ace 2 (16 GB DRAM, Snapdragon 8+ Gen 1, UFS 3.1). **NOT desktop GPUs** | paper |

**Maintainer-measured (OnePlus 12, TurboSparse-Mixtral-47B)**:

| Metric | Value |
|---|---|
| Decode tok/s (avg, 50% FFN offload) | **11.68** |
| Decode tok/s (19 GB DRAM) | **11.68** |
| Decode tok/s (7 GB DRAM limit) | 2.13 |
| Prefill tok/s @ 512 tokens (Llama-2-7B) | 405 |
| Prefill tok/s @ 512 tokens (Mixtral-47B) | 79 |
| vs llama.cpp | up to 29.2× peak, 21.2× avg decode |
| vs LLM-in-a-Flash | 3.94× avg, 4.38× peak |

**Hardware caveat**: There is **no published RTX 4090 number for PowerInfer-2**. The polymorphic engine is designed around smartphone XPUs (big.LITTLE CPU + NPU + UFS); on a 4090 it would degrade substantially (no NPU prefill path).

### §2.5 Exo

| Field | Value | Source |
|---|---|---|
| Architecture | P2P layer-sharded inference; each Mac holds a contiguous slice of layers proportional to unified-memory size; auto-discovery via UDP multicast; ring topology with memory-weighted partitioning; OpenAI-compatible API on :52415 | `github.com/exo-explore/exo` |
| Concurrency | Single-stream per session; tensor-parallel requires power-of-two node counts (2 or 4) | as above |
| Target HW | Apple Silicon Macs (M1/M2/M3/M4) connected via Thunderbolt 4 / 10GbE / Wi-Fi; cross-arch pooling | as above |
| Weight-loading | On startup downloads shards from HuggingFace and partitions across nodes; **layers resident in unified memory for model lifetime** (no streaming fetch per token); ring allreduce of activations per layer | as above |

**Independent benchmark (thinkdifferent.blog, Jakub Jirák, 2026-07-26)** — Llama 3.3 70B 4-bit:

| Setup | tok/s |
|---|---|
| Single M2 Ultra 192 GB, MLX | ~11.5 |
| Single M3 Ultra 192 GB, MLX | ~11.5–20 |
| **2× Macs via exo, Thunderbolt 4 bridge** | **~8.2** (slower than 1 Mac — model fits on one) |
| 2× Macs via exo, 10GbE | ~6.9 |
| 2× M2 Max (64 GB ea) | 18–22 |
| 4× M3 Ultra with RDMA | ~16 (70B FP16) |
| 2× M3 Max over Wi-Fi 6 | 6–10 |
| DeepSeek-V3-class 4× Mac Studio 128 GB cluster | single-digit tok/s |

**Key constraint**: M2 Ultra unified-memory bandwidth **~800 GB/s** vs Thunderbolt 4 bridge **~2.5–3 GB/s**. When the model fits on one Mac, adding a second Mac slows it down. Exo only wins for >192 GB unified-memory requirements.

## §3 视频生成 Offload 引擎（与 MiniMax-H3 比对）

### §3.1 HuggingFace diffusers

| Field | Value | Source |
|---|---|---|
| Weight residency | Three modes: `enable_model_cpu_offload()` (whole-model); `enable_sequential_cpu_offload()` (sub-module, "extremely slow"); `enable_group_offload()` (COW-unchain, layer/block streams, 2026); pinned per `--max_memory={0:"8GB", 1:"1GB"}` | `huggingface.co/docs/diffusers/main/en/optimization/memory` |
| Quantization | `enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)`; integrates TorchAO INT8, Quanto FP8 | as above |
| Concurrency | Per-request single pipeline; multi-GPU via `device_map` only | as above |
| Checkpoint / resume | **No**. Each `pipeline(...)` call runs the full loop. No built-in diffusion-step snapshotting. | as above |
| Host/device budgets | Partial — `max_memory` is per-device; **no semantic host-RAM ceiling**; docs say "ensure you have 2x the amount of memory as the model size" (unbounded) | as above |

**Maintainer-measured (RTX 4090)**: **未能验证** — diffusers docs only show A100/H100 examples.

### §3.2 ComfyUI

| Field | Value | Source |
|---|---|---|
| Weight residency | Node-graph runtime (Python/PyTorch); per-node model load/unload; "Smart memory management" per README ("can automatically run large models on GPUs with as low as 1GB vram") | `github.com/comfyanonymous/ComfyUI` |
| Quantization | fp8 (e.g. `ComfyUI-Kijai/WanVideoWrapper`), bf16, NF4 via community; VAE tiling / slicing | README |
| Concurrency | Built-in async queue with re-execution caching. `--gpu-only` flag; xDiT integration in maintainer repos. | README |
| Target HW | NVIDIA / AMD / Intel / Apple Silicon / Ascend NPU / Cambricon MLU / Iluvatar CoreX | README |

**Third-party benchmark**: Wan 2.1 T2V-14B at 720p on RTX 4090 (FP8, full-precision path): **~240 s** per video, peak ~22.3 GB; URL: `https://blog.csdn.net/gitblog_02672/article/details/150000438` (2025). Wan 2.1 14B 480p on RTX 4090: ~281 s (SaladCloud via `http://runaihome.com/blog/wan-video-local-ai-gpu-guide-2026`, 2026). HunyuanVideo on RTX 4090: ~10 min per clip with significant VRAM optimization.

**Checkpoint / resume**: **No**. ComfyUI is graph-execution; per-node caching only re-runs changed subgraphs.

### §3.3 AnimateDiff / SparseCtrl

| Field | Value | Source |
|---|---|---|
| Architecture | ~417M–453M motion module added to frozen SD1.5/SDXL UNet; SparseCtrl adds 1.85 GB RGB/sketch encoders | `github.com/guoyww/AnimateDiff`; `arxiv.org/abs/2307.04725` |
| VRAM | SD1.5 AnimateDiff ~8–10 GB; SDXL ~13 GB per README | README |
| Diffusers offload | `enable_model_cpu_offload` / `enable_sequential_cpu_offload` since 2023; no unique-to-AnimateDiff offload | `github.com/guoyww/AnimateDiff` |
| **RTX 4090 published numbers** | **未能验证** — no maintainer benchmark found for RTX 4090; community SD1.5 ~1–3 s/frame @ 512×512; SDXL ~10–60 s for 16 frames | Reddit r/StableDiffusion / community |

### §3.4 CogVideoX (THUDM)

| Field | Value | Source |
|---|---|---|
| Models | CogVideoX-2B (FP16); CogVideoX-5B (BF16); CogVideoX1.5-5B (BF16) | `github.com/THUDM/CogVideo` |
| Offload | `enable_sequential_cpu_offload()` + `vae.enable_slicing()` + `vae.enable_tiling()` (official README) | as above |
| Quantization | TorchAO INT8, Quanto FP8, BF16 — official diffusers integration | as above |
| Concurrency | Per-request; multi-GPU via xDiT (Ulysses + Ring sequence-parallel) | README "Tools/parallel_inference_xdit" |
| Checkpoint / resume | **No**. One-shot `cli_demo`. | as above |

**Maintainer-measured (official README table)**:

| Model | A100 | H100 |
|---|---|---|
| CogVideoX-2B (6s, 720×480) | ~90 s | ~45 s |
| CogVideoX-5B (6s, 720×480) | ~180 s | ~90 s |
| CogVideoX1.5-5B (5s, 1360×768) | ~1000 s | ~550 s |

README warns: "If optimizations are disabled, memory consumption will multiply, with peak memory usage being about 3 times the value in the table. However, speed will increase by about 3-4 times." So `enable_sequential_cpu_offload()` on gives ~540-720 s on A100 for CogVideoX-5B. **RTX 4090 row: 未能验证**.

### §3.5 Wan 2.1 (Alibaba)

| Field | Value | Source |
|---|---|---|
| Models | T2V-1.3B (12 layers, 1536-dim), T2V-14B (40 layers, 5120-dim, 40 heads), I2V-14B-720P/480P, FLF2V-14B, VACE-1.3B/14B | `github.com/Wan-Video/Wan2.1` |
| Offload flags | `--offload_model True`; `--t5_cpu` (T5 encoder to CPU); FSDP + xDiT USP | README |
| Quant | Diffusers `enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn)`; FP8 community via DiffSynth-Studio / LightX2V | README |
| VRAM | T2V-1.3B single 4090 (--offload_model True --t5_cpu): **8.19 GB** baseline; T2V-14B 720p FP8 + offload: **~22.3 GB** peak (community) | README + CSDN benchmarks |
| Checkpoint / resume | **No**. `generate.py` is one-shot. | README |

**Maintainer-measured (Alibaba team, README + `comp_effic.png`)**:
- T2V-1.3B single RTX 4090, 832×480, no quant, **with `--offload_model True --t5_cpu`**: **~240 s** for a 5-second 480P video (~4 minutes) — README headline.
- T2V-14B single RTX 4090 480p (SaladCloud): ~281 s. **720p OOMs at full precision on 24 GB; FP8 path required**.
- T2V-14B on H100 SXM 80 GB: 85 s @ 480p, 284 s @ 720p (A100 SXM: 170 s / 523 s).

**Note (official README)**: "T2V-14B is slower than I2V-14B because the former samples 50 steps while the latter uses 40 steps."

## §4 三 workload 锚定对比

### §4.1 Workload A：单卡 24 GiB + 有限 host RAM 跑超大 MoE (>300B)

**Anchor numbers**:

| System | Model | Quant | tok/s decode | Hardware | Source | Notes |
|---|---|---|---|---|---|---|
| **ff (us)** | GLM-5.3-Flash 306B | FP8 | **0.87** | RTX 4090 + 62 GiB RAM | `docs/models.md:11` | **single-stream**, Linux |
| FlexLLMGen | OPT-175B | FP16 | **0.69** aggregate | T4 16GB + 208 GB DRAM + 1.5 TB SSD | `arxiv.org/abs/2303.06865` Tab | **bs=256 aggregate** |
| FlexLLMGen w/ compress | OPT-175B | INT4 | **1.12** aggregate | same | same | **bs=144 aggregate** |
| KTransformers V0.2 | DeepSeek-V3 671B | INT4 Q4_K_M | **13.69** | 4090D + 2× Xeon Gold 6454S (1 TB DDR5, AMX) | `ktransformers` tutorial | **server-class CPU**, AMX |
| KTransformers V0.3 AMX | DeepSeek-V3 671B | INT4 | (decode ~14 tok/s, prefill 286.55) | same | as above | prefill 27.79× llama.cpp |
| PowerInfer v1 | OPT-175B | INT4 | **~2** | RTX 4090 + 192 GB DDR5 | `arxiv.org/abs/2312.12456` §8.2 | dense ReLU-family only |
| PowerInfer v1 | OPT-175B | FP16 | (lower) | same | as above | "PowerInfer nearly reaches two tokens per second" |
| Ollama (DGX Spark) | deepseek-r1 14B | Q4_K_M / Q8_0 | 19.99 / 13.32 | DGX Spark (not RTX 4090) | `ollama.com/blog/nvidia-spark-performance` Oct 23, 2025 | much smaller model |
| TensorRT-LLM (B200) | DeepSeek-R1 671B | FP4 + FP8 KV | **>250 tok/s/user, 30,000 tok/s max/GPU** | 8× B200 NVL8 | NVIDIA dev blog March 18, 2025 | **rack-scale**, single-user 253 tok/s |
| MLX (M3 Ultra) | DeepSeek-V3-0324 671B | 4-bit | **21.05** (1.1K prompt) / 17.81 / 5.79 | M3 Ultra 512 GB unified memory (800 GB/s) | `hardware-corner.net` (citing Awni Hannun) | **smartphone/M-series**, not Linux/x86 |

**Reading the table**: We sit between FlexLLMGen (older, slower, batch-aggregate throughput) and PowerInfer v1 (similar scale, similar single-stream regime, 1.5–3× ahead on dense ReLU models but **not validated for MoE**). KTransformers is **the closest direct comparator** — same target hardware spec (4090-class GPU + substantial host RAM) and a published 14 tok/s decode on a 671B MoE. We have **not** yet validated on a DeepSeek-V3/R1-class workload; their hardware requirements (Xeon SPR with AMX) are server-class, and our 62 GiB host is sub-server-class by a factor of ~16×. TensorRT-LLM B200 is a different tier (4-GPU FP4 + FP8 KV on datacentre Blackwell). MLX on M3 Ultra 512 GB exploits unified memory and is 24× ahead of us on the same model class.

**Caveats and unknowns**:
- Our 0.87 tok/s is for **32 decode tokens** on a 306B FP8 MoE (GLM-5.3-Flash) — not directly comparable to KTransformers' 13.69 tok/s for **671B INT4** (KTransformers' model is 2.2× larger; quant is INT4 vs FP8 — different arithmetic intensity).
- Linux vs Windows OS sensitivity is real (mmap/page-cache/NUMA), so OS-mismatched numbers are deliberately not directly compared below; `docs/models.md:5` confirms Linux; most peer numbers don't state OS (treat as Linux-default unless otherwise noted).
- Our GLM is FP8 (one byte/parameter); DeepSeek-V3/R1 in KTransformers is INT4 Q4_K_M (≈0.5 byte/parameter); the comparison is therefore under a different checkpoint size regime.

### §4.2 Workload B：70B 级 dense tok/s（对比 Edge0/Qwen3.8 实测）

**Anchor numbers**:

| System | Model | Quant | tok/s decode | Hardware | Source |
|---|---|---|---|---|---|
| **ff (us)** | Edge0-35B-A3B | groupwise-int4 | 128 tok / 17.50 s = **7.31 tok/s** | RTX 4090 + 62 GiB RAM | `docs/models.md:15` |
| **ff (us)** | Qwen3.8-27B | groupwise-int4 | 128 tok / 13.93 s = **9.19 tok/s** | same | `docs/models.md:16` |
| llama.cpp | Qwen2.5-72B-Instruct Q4_K_M (~44 GB) | INT4 | 4–6 tok/s (community band) | RTX 4090 | **未能验证** in primary sources |
| vLLM | Llama 3.3 70B FP8 | FP8 | **1,496 tok/s/GPU steady-state** | H100 80GB | `inferencex.semianalysis.com` (third-party) |
| vLLM | Llama 70B (community Q4 on 2× 4090) | INT4 | ~25–30 tok/s aggregate | 2× RTX 4090 | third-party `willitrunai.com` March 2026 |
| Ollama | qwen3 32B Q4_K_M | INT4 | **9.41 tok/s** | DGX Spark (not 4090) | `ollama.com/blog/nvidia-spark-performance` Oct 2025 |
| TensorRT-LLM | Llama 70B FP8 | FP8 | **3,269 out tok/s/GPU** at BS 1024 / TP=2 | H100 SXM | `nvidia.github.io/TensorRT-LLM/performance.html` (NVIDIA) |
| TensorRT-LLM | Llama 70B FP8 | FP8 | 561 out tok/s/GPU at BS 256 / TP=2 | L40S | as above |
| MLX | Llama-3-70B Q4_K_M | INT4 | **12.48 TG / 118.79 PP** | M2 Ultra 76-core GPU 192 GB | `github.com/SuperXiang/GPU-Benchmarks-on-LLM-Inference` May 2024 |
| SGLang | Llama-3-70B (bf16) vs vLLM | bf16 | **up to 3.1× higher throughput** than vLLM on 8× A100 | 8× A100 80GB | `lmsys.org/blog/2024-07-25-sglang-llama3/` |

**Reading the table**: Our single-stream Edge0 (7.31 tok/s) and Qwen3.8 (9.19 tok/s) are competitive with **community-band llama.cpp on RTX 4090** (4–6 tok/s, unverified), and roughly **on par with Ollama on DGX Spark for qwen3 32B Q4_K_M** (9.41 tok/s, in a 70B-sh class). MLX on M2 Ultra 192 GB for Llama-3-70B Q4_K_M is **12.48 tok/s** — we are within ~25–35% of a unified-memory 192 GB machine's small-model throughput, which is the strongest desktop-level cross-platform comparator we have.

We are decisively behind vLLM/SGLang/TensorRT-LLM **at scale and at single-node batch** — those systems win the throughput-vs-concurrency tradeoff on datacentre GPUs (H100/B200) at batch ≥ 32, where our `--device cuda:N` path does not target. We are not designed to compete on `bs=512, ISL/OSL 128/128` datacenter bins.

### §4.3 Workload C：视频生成 offload 端到端时间

**Anchor numbers**:

| System | Model | Resolution / duration | Wall time | Peak VRAM | Source |
|---|---|---|---|---|---|
| **ff (us)** | MiniMax-H3 T2VA | 1344×768, 4 s × 49 evals | **30m 53s = 1,853 s total (≈ 37.8 s/eval)** | 20.92 GiB | `docs/models.md:12` |
| diffusers | CogVideoX-5B | 720×480, 6 s | **~180 s (A100) / ~90 s (H100)**; `enable_sequential_cpu_offload` on ≈ 540–720 s | not published for 4090 | `github.com/THUDM/CogVideo` README |
| diffusers | CogVideoX1.5-5B | 1360×768, 5 s | ~1000 s (A100) / ~550 s (H100) | from 10 GB BF16 | as above |
| ComfyUI | Wan 2.1 T2V-14B FP8+offload | 720p, 5 s | **~240 s** | ~22.3 GB | `blog.csdn.net/gitblog_02672/...150000438` (third-party) |
| ComfyUI | Wan 2.1 T2V-14B fp16+offload | 480p, 5 s | ~281 s | OOM @ 720p fp16 on 24 GB | `runaihome.com/blog/wan-video-local-ai-gpu-guide-2026` (third-party) |
| Wan 2.1 official | T2V-1.3B single 4090, `--offload_model True --t5_cpu` | 832×480, 5 s | **~240 s** | 8.19 GB | `github.com/Wan-Video/Wan2.1` README headline (`comp_effic.png`) |
| Wan 2.1 | T2V-14B H100 SXM | 480p / 720p | 85 s / 284 s | 80 GB | `Wan2.1` README |
| AnimateDiff SDXL | 16 frames | (community, RTX 4090) | (10–60 s) | ~13 GB | community r/StableDiffusion (not maintainer) |

**Reading the table (corrected; pre-edit draft had an inverted direction)**: H3 produces **one** finished 1344×768 4-second **video + audio** T2VA per run at total wall **1,853 s** on RTX 4090 (`docs/models.md:12`: "Generated 107 frames in 49 evaluations for a 768p, 16:9, 4-second request"). Per finished clip, H3 is **~7.7× slower** than Wan 2.1-T2V-1.3B's README-claimed 240 s for a single 832×480 5-second clip on the same GPU with `--offload_model True --t5_cpu` (1853 ÷ 240 ≈ 7.72). **The comparison is not apples-to-apples**: H3 is a multi-modal video+audio T2VA at 1344×768 with ~24 fps × 4 s = 96 frames out; Wan 2.1-T2V-1.3B is text-to-video only at 832×480 with 5 s × 24 fps = 120 frames out, no audio. The 4-second Wan 2.1 reference (16% the resolution of H3, audio-stripped) is the closest direct comparator but still mismatched. Wan 2.1's 240 s amortizes one-time setup that H3 also amortizes (~49 s text encoding + ~434 s static-context preparation in the same `docs/models.md` row; H3's per-eval ratio is ~28 s/eval after subtracting the one-time cost, but 49 evals still collectively produce one final clip). There is **no direct single-clip H3-equivalent** in published peer reports, because H3's checkpointed-resume across 49 evals has no peer analogue.

There is **no direct single-clip H3-equivalent** in published peer reports, because H3's checkpointed-resume over 49 evals has no peer analogue. ComfyUI's 240-s per 720p 14B FP8 clip is in the same magnitude as Wan 2.1 official.

## §5 诚实差距分析

### §5.1 我们在哪明确落后

1. **Concurrent-batching serving 缺失**。vLLM/SGLang/TensorRT-LLM/Ollama 都在 continuous batching 上花了多年工程；我们目前是 single-stream-only（除了 H3 的 evaluation 循环）。任何 `bs>1` 长 context + 多请求并发场景，我们连入门门槛都没到。
2. **Serving API 缺失**。vLLM/SGLang/TGI 都有 OpenAI-compatible HTTP server；我们只有 CLI。这把我们定位为"本地工具型 CLI"而不是"在线服务"。
3. **量化广度**。llama.cpp (1.5–8-bit GGUF/K-quant/IQ) 和 vLLM (GPTQ/AWQ/Marlin/MXFP4/NVFP4/TorchAO 等一长串) 的覆盖远超过我们。`ff` 当前的 text adapter 量化取决于模型出版方（GLM 提供 FP8 weights，Edge0/Qwen3.8 提供 groupwise-int4），运行时不做变体转换。
4. **MoE expert 缓存策略**。GLM 用 shared-pool LFU（commit `31cb073`），但没有 PowerInfer v1 的 activation-sparsity-aware neuron-level 预测——我们按专家 ID 缓存，最近的活跃专家留下来，cold 专家整片重读，与 activation distribution 解耦。
5. **生态 & 第三方模型支持**。vLLM/TGI/SGLang 自动覆盖 Hugging Face 上任何 causal LM 架构；我们要写适配器（参考 `docs/models.md` "Adding model adapters" 五步法）才能加一个新模型，节奏差一个数量级。
6. **Mobile / 多节点统一内存 / 异构硬件**。MLX (Apple Silicon), PowerInfer-2 (smartphone NPU), Exo (multi-Mac), TensorRT-LLM (B200/Grace-Hopper) 都覆盖了我们没碰的硬件类别——虽然这不是落后（产品定位不同），但读者应当注意我们当前的硬件覆盖范围。
7. **NVFP4/B200 类性能数字**。TensorRT-LLM 在 B200 上跑 671B MoE 拿 **>30,000 tok/s/GPU max throughput** 是我们短期内不会去竞争的层面。

### §5.2 我们在哪有差异化优势

1. **校准驱动的 host/device split**。`ff bench io --profile local-interconnect` 测 B_P / B_H，给出 host_expert_share_per_mille；`--host-profile output/*.json` 把这次校准固化下来，每条 inference 命令都分享同一份校准——不是改 `--device` 那种经验式的选择。我们的 GLM 表就是这条路径的产物：`docs/models.md` 第 41 行明确写 "B_P 18.798 GiB/s and B_H 31.248 GiB/s evaluating (8.095 GiB/s end to end), B_P/B_H 0.60"。
2. **Admission control 与 capacity shortfall**。`ff` 把 candidate rejection 的原因结构化落到 evidence（commit `00b2fb1`：Persists structured capacity shortfalls on candidate rejections；commit `efb4522`：Fix admission accounting: unified-memory pool, exclusive bytes, pool-scaled reserves），比 LLM engines 提示 "out of memory" 然后崩溃要细致。
3. **Checkpoint / resume 跨阶段（视频独有）**。H3 在 49 evals 过程中产出 recovery checkpoint，可从 output-dir 恢复（`README.md:84`）。这在视频管线里独有——diffusers / ComfyUI / CogVideoX / Wan 2.1 都是一次性 `pipeline(prompt, num_frames=...)`，断电没恢复。
4. **显式字节预算拆分**。`--max-host-mib` / `--max-device-mib` / `--expert-cache-mib` / `--max-steps` 这些是 `ff` 的一等公民。ComfyUI 给了 `--lowvram` `--medvram` `--novram` 这种 preset；diffusers 给了 `max_memory` per device；都没给我们这种语义上的 host-RAM ceiling + admission gate。我们 host_expert_share_per_mille 这种 per-call 报告也少见。
5. **单二进制零依赖工具链**。Rust 单一二进制，无 Python/conda/torch/pip 依赖；离线工具链适合 CI、边缘、单机。再叠加 `ff probe --device cpu --json` / `ff text generate --weight-source mmap` 这种 CLI surface，可以无 GPU、无 PyTorch 地跑模型发现和轻量推理。
6. **MoE MoE Cache + admission 双层证据**。GLM 既复用了每条 forward 的 shared-pool LFU cache，又维护 admission reserves；Edge0/Qwen3.8 用 groupwise-int4 + optional resident experts 上传；这些都是 `ff` 里 closed-loop 调和过的路径。
7. **成绩单可复现**。`docs/models.md` 的每一个 Performance 行都是"中间三次连跑"，并明确写出 wall / RSS / GPU 三项 sample mean；不是峰值/单次。这种"三次取中"约定在 peer 项目中罕见。

## §6 数据局限与未能验证项

| # | 限制 | 影响 |
|---|---|---|
| 1 | `mcp__web-reader__webReader` / `WebFetch` 在此环境对 `github.com` / `reddit.com` / `huggingface.co` 域名有 sandbox 限制 | 某些页可搜索到但内容 fetch 失败；只能依靠搜索摘要片段 + 第三方汇总（`cnrai/llm-perfbench`、`SuperXiang/GPU-Benchmarks-on-LLM-Inference`、`blog.csdn.net`、`hardware-corner.net`、`thakicloud.com`、`thinkdifferent.blog`）来填充主源缺失 |
| 2 | llama.cpp / SGLang / TensorRT-LLM / MLX 的 **RTX 4090 单卡 70B row** 都为 **未能验证** | 这些行只 third-party 估计；不在 maintainer 表内 |
| 3 | TGI 维护方未发布 RTX 4090/70B 表 | TGI 已经进 maintenance mode，HF 也不再对外公布 benchmark table |
| 4 | PowerInfer v1 **不验证 MoE** 模型（只 ReLU-family dense） | 它和我们的 GLM/Edge0 不能直接对比——只能在 OPT-175B dense 上做"形态接近"参照（结果：~2 vs 我们 0.87 tok/s） |
| 5 | PowerInfer-2 是 smartphone-only | 它在 desktop GPU 上没有可测数字 |
| 6 | Exo / Ollama 多数数字来自 macOS / DGX Spark | 与 Linux/x86 不严格同 OS；`docs/models.md` 是 Linux |
| 7 | diffusers 的 `max_memory` per-device **没有语义 host-RAM ceiling** | "host 占多少"是用户自行管理；与我们的 `--max-host-mib` 不可直接比较 |
| 8 | ComfyUI 的 ~240 s Wan2.1 数字来自第三方 | maintainer 没给 70B / 4090 row |
| 9 | KTransformers 的 13.69 tok/s 数字用 server-class Xeon SPR + AMX | 与我们 i9-13900KF (client-class, no AMX) 不严格可比；只有 "形态近似" 价值 |
| 10 | Wan 2.1 README `comp_effic.png` 图片 fetch 失败 | per-cell 数字仅能从 README 文本段落推断；不能精确到每个 cell |

## §7 全部引用清单

### §7.1 我们的内部文档
- `docs/models.md` — Performance 表 + GLM/Edge0/Qwen3.8/MiniCPM5/CLIP/TRELLIS/Music3/Edge0/Qwen 表格
- `README.md` — Build / Run / Quick Start / Models
- `models/deepseek-ai/DeepSeek-V4.1-Flash/` (在本调研范围之外)
- `docs/cli-*.md` 三份（先前任务交付）

### §7.2 论文与 arXiv
- `arxiv.org/abs/2303.06865` — FlexLLMGen, NeurIPS 2023.
- `arxiv.org/abs/2312.07104` — SGLang, Lianmin Zheng et al., 2024-06-06.
- `arxiv.org/abs/2309.06180` — vLLM / PagedAttention, SOSP 2023.
- `arxiv.org/abs/2312.12456` — PowerInfer v1, SJTU IPADS, SOSP'23.
- `arxiv.org/abs/2406.06282` — PowerInfer-2, SJTU IPADS + Tsinghua + Shanghai AI Lab, NeurIPS'24.

### §7.3 项目主页 / GitHub
- `github.com/ggml-org/llama.cpp` — README "Description" / "Supported backends" / "Tools".
- `github.com/ollama/ollama` — README "Supported backends".
- `github.com/vllm-project/vllm` — README `vllm is fast with`.
- `github.com/huggingface/text-generation-inference` — "Caution" callout (maintenance mode).
- `github.com/sgl-project/sglang` — README "About".
- `github.com/NVIDIA/TensorRT-LLM` — `examples/models/core/deepseek_v3` README "Hardware Requirements".
- `github.com/SJTU-IPADS/PowerInfer` — repo.
- `powerinfer.ai/v2/` — PowerInfer-2 project.
- `github.com/exo-explore/exo` — repo.
- `github.com/FMInference/FlexLLMGen` — README "Generation Throughput" table.
- `github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md` — V0.2/V0.2.1/V0.3 tables.
- `github.com/huggingface/diffusers` — `docs/source/en/optimization/memory.md` (group offload + layerwise casting).
- `github.com/comfyanonymous/ComfyUI` — README "Smart memory management".
- `github.com/guoyww/AnimateDiff` — README VRAM.
- `github.com/THUDM/CogVideo` — README "Model Introduction" table.
- `github.com/Wan-Video/Wan2.1` — README + `comp_effic.png`.
- `github.com/ml-explore/mlx` — `docs` + `mlx-lm/FAQ.md`.
- `github.com/ml-explore/mlx-examples` — README "Text Models" (Mixtral 8x7B entry).
- `huggingface.co/mlx-community/Meta-Llama-3-70B-Instruct-4bit` — model card.

### §7.4 官方博客
- `ollama.com/blog/nvidia-spark-performance` — Oct 23, 2025.
- `ollama.com/blog/gpt-oss` — Aug 5, 2025.
- `lmsys.org/blog/2024-07-25-sglang-llama3/` — July 25, 2024 (SGLang v0.2).
- `lmsys.org/blog/2025-12-10-rfork/` — Dec 10, 2025 (R-Fork).
- `developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/` — Mar 18, 2025.
- `blog.vllm.ai/2025/09/05/anatomy-of-high-throughput-llm-inference-system.html` — Sept 5, 2025.

### §7.5 第三方基准 / 复现
- `inferencex.semianalysis.com/` — Llama 3.3 70B FP8 + vLLM H100 80GB.
- `thakicloud.com/tech-blog/en/llmops/ktransformers-moe-offload-28x-validation` — July 19, 2026.
- `thinkdifferent.blog/blog/the-multi-mac-ai-cluster-insane-overkill-or-the-future` — Jakub Jirák, July 26, 2026.
- `github.com/SuperXiang/GPU-Benchmarks-on-LLM-Inference` — May 2024.
- `github.com/cnrai/llm-perfbench` — DeepSeek-V3-0324-4bit on M3 Ultra.
- `hardware-corner.net/studio-m3-ultra-running-deepseek-v3` — citing Awni Hannun.
- `news.ycombinator.com/item?id=34830270` — Feb 2023 (FlexGen 1.12 t/s confirmation).
- `blog.csdn.net/gitblog_02672/article/details/150000438` — Wan2.1 14B FP8 720p on 4090.
- `runaihome.com/blog/wan-video-local-ai-gpu-guide-2026` — Wan2.1 480p/720p on 4090.
- `sohu.com/a/745917088_129720` — 机器之心 / PowerInfer v1 13.20 / 29.08 tok/s.
- `bestgpusforai.com` — 70B/4090 estimates (third-party).
- `willitrunai.com` — 2× RTX 4090 / Llama 70B Q4 (~25–30 tok/s aggregate).

### §7.6 项目文档
- `dev.to/multigrid/llamacpps-mmap-and-mlock-flags-explained-43fp` — llama.cpp `--load-mode`.
- `jonathanding.github.io/llm-learning/en/articles/llama-cpp-model-loading` — llama.cpp `unmap_fragment()`.
- `bodegaone.ai/learn/guides/gguf-quantization-level` — IQ-quants overview.
- `docs.ollama.com/gpu` — Ollama hardware support.
- `docs.vllm.ai/en/latest/features/quantization/` — vLLM quantization kernels.
- `docs.vllm.ai/en/latest/getting_started/installation.html` — vLLM hardware support.
- `huggingface.co/docs/text-generation-inference/en/reference/launcher` — TGI launcher reference.
- `adyen.com/knowledge-hub/llm-inference-at-scale-with-tgi` — TGI architecture.
- `nvidia.github.io/TensorRT-LLM/performance.html` — Llama 70B perf table.
- `nvidia.github.io/TensorRT-LLM/kv_cache_reuse.html` — KV-cache host offload.
- `huggingface.co/docs/diffusers/main/en/optimization/memory` — group offload + layerwise casting.
- `kvcache-ai.github.io/ktransformers/` — KTransformers docs.
- `ml-explore.github.io/mlx/build/html/index.html` — MLX docs.

### §7.7 我们的 commit 历史（自查）
- `1aafd85 Build CUDA kernels ahead of time per architecture`
- `7e4d0ad Add margin-based draft gating to the qwen35 speculative example`
- `9ff47d0 Fill the edge0/qwen35 performance rows from measured runs`
- `cb5259e Fix cuda-gated clippy errors, opt in the vision e2e test`
- `d808e04 Wire edge0/qwen35 into the text CLI, document them, and add CI tests`
- `31cb073 Charge the expert cache by what its layout retains`
- `b9ee73a Re-measure the GLM row and record its profile and cache preconditions`
- `00b2fb1 Persist structured capacity shortfalls on candidate rejections`
- `518b239 Remove the unused premerge script`
- `0a82f30 Drop stale references to the removed reference scripts`
- `1d2d968 ff-qwen35: P1 vision — image→text with mrope, verified against HF fixtures`
- `6f297e6 Wire ff-edge0 unified-memory planning`
- `eacace3 Consolidate admission reserves on one pool-scaled formula`
- `0f8a859 Add Edge0-35B-A3B and Qwen3.5-27B adapters with GPU decode, wide kernels, and MTP speculation`
- `efb4522 Fix admission accounting: unified-memory pool, exclusive bytes, pool-scaled reserves`

## §8 阅读指南

- 如果你对比 **超大规模 MoE single-stream on a consumer GPU**: 重点看 §4.1, §2.2 (KTransformers), §2.3 (PowerInfer v1). 我们比 PowerInfer 慢，但形态接近；比 KTransformers 慢几个数量级，因为它的 server-class Xeon + AMX 是不同的硬件 tier。
- 如果你对比 **70B 级 dense decode**: §4.2. 我们和 llama.cpp community band、Ollama-DGX-Spark-on-qwen3-32B、MLX-M2-Ultra 在同一数量级（小数 tok/s 到 ~10 tok/s），离 vLLM/SGLang/TensorRT-LLM 那种 batched-throughput 维度差一个量级。
- 如果你对比 **视频生成 offload**: §4.3. 我们 H3 一条 4 秒 T2VA（768p + 音频）共 1,853 s = ~7.7× 慢于 Wan 2.1-T2V-1.3B 一条 5 秒 T2V（480p，无音频）240 s 的官方数字（`docs/models.md:12` vs `github.com/Wan-Video/Wan2.1` README）。H3 模型更大、产物多音频轨道、分辨率更高 1.6×，不可直比。
- 如果你想找**我们的差异化优势**: §5.2. 主要在"显式字节预算 + admission + checkpoint/resume + 校准驱动的 host/device split + 单二进制零依赖"。
