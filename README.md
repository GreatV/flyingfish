# flyingfish

Run large published models on machines that cannot hold them.

`ff` is a Rust CLI for video, text, music and 3D inference. It streams model weights from disk and exposes configurable host and device caches, allowing inference without keeping the entire checkpoint resident in RAM or VRAM.

Active weights, activations and execution workspace still need to fit the available memory. Storage bandwidth and cache budgets affect throughput.

## Tasks and models

Commands are organized by task. `--model` identifies a checkpoint directory; its configuration selects the model adapter, independent of the directory name. Replace angle-bracket placeholders below with your own inputs.

Each adapter supplies its supported operations and options. Use `ff text generate --help` for shared options, or `ff text generate --adapter glm --help` for a model's full options without loading a checkpoint. Unsupported model/task combinations and options are rejected before inference.

| Command | Model | What it does |
|---|---|---|
| `ff video` | [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | Text, first/last-frame and reference conditioning for video and audio, checkpointed and resumable |
| `ff text` | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | Disk-streamed MoE text generation, single or multi-GPU |
| `ff text` | [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B), [DSpark](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark) | Greedy text generation with an optional speculative draft |
| `ff text` | Edge0-35B-A3B | Groupwise-int4 hybrid GDN/MoE text generation with optional CUDA expert residency |
| `ff text` | Qwen3.8-27B | Groupwise-int4 dense text generation with an optional PNG image input |
| `ff music` | [MiniMax Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) | Lyrics and a caption to a stereo WAV |
| `ff 3d` | [TRELLIS-1](https://huggingface.co/microsoft/TRELLIS-text-large), [TRELLIS.2](https://huggingface.co/microsoft/TRELLIS.2-4B) | TRELLIS-1 text/image to Gaussian splats; TRELLIS.2 image to a colored mesh |
| `ff similarity` | [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) | Image/text similarity scoring |

TRELLIS generation requires a separate `--conditioner` checkpoint. Image inputs must be prepared PNGs: 518×518 for TRELLIS-1, or 512×512 with `--resolution 512` for TRELLIS.2.

## Build and install

Rust 1.95.0 is pinned in `rust-toolchain.toml`, with Clippy and rustfmt. Run these commands from the repository root:

```bash
cargo build --locked --release                         # CPU
cargo build --locked --release --features cuda         # NVIDIA CUDA
cargo build --locked --release --features flash-attn   # CUDA with H3 FlashAttention
cargo build --locked --release --features metal        # Metal on macOS
```

The executable is `target/release/ff` (`ff.exe` on Windows). To install it into Cargo's binary directory:

```bash
cargo install --locked --path .
```

Add the same `--features` option to the install command for a GPU build. The examples below assume Cargo's binary directory is on `PATH`.

CUDA builds require the CUDA toolkit, `nvcc` on `PATH`, and a host C++ compiler. At runtime, the selected CUDA paths need the NVIDIA driver and CUDA libraries, including cuBLAS/cuBLASLt; H3's cuDNN paths also need cuDNN. For H3 FlashAttention, build with `flash-attn` and pass `--flash-attention` when running.

When building without a visible GPU, set `CUDA_COMPUTE_CAP` to the target GPU's compute capability, such as `80` for 8.0.

Each in-repo CUDA kernel ships as one `compute_80` PTX — the sealed numerical contract — plus cubins `ptxas` translates it to ahead of time. The runtime loads a cubin only on an exact architecture match and falls back to the PTX otherwise; a cubin is the same instructions pre-translated, so the choice never changes numerics, but only a cubin loads on a driver whose PTX ISA predates the build toolkit's. `FF_CUDA_ARCHS=80,86,89,90,100,120` overrides the translated set; the default covers the supported fleet. The cudarc bindings are pinned to CUDA 13.2 (`cuda-13020`): the pin selects the FFI binding set, not the machine's toolkit, so any 11.8-or-newer toolkit builds and any same-major driver runs.

Select a device with `--device cpu`, `--device cuda:N` or `--device metal:0`. In a CUDA build, `auto` attempts `cuda:0`; builds without CUDA use CPU for `auto`. Select Metal explicitly. GLM generation defaults to `cuda:0`.

## Quick start

Inspect the CPU and available host memory, then generate a short response with MiniCPM:

```bash
ff probe --device cpu --json

ff text generate \
    --model "<minicpm-checkpoint>" \
    --prompt 'Explain paging to a systems programmer.' \
    --device cpu \
    --max-new-tokens 32
```

Use `--device cuda:0` or `--device metal:0` with the corresponding build.

## Model guide

See [Model guide](docs/models.md) for per-model commands, measured performance and adapter development. See [Topology-aware configuration](docs/topology-aware-config.md) for how performance settings are derived from the machine profile.

## License

Apache-2.0. See [LICENSE](LICENSE), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the upstream implementations these adapters follow.
