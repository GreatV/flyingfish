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

Select a device with `--device cpu`, `--device cuda:0` or `--device metal:0`. In a CUDA build, `auto` attempts `cuda:0`; builds without CUDA use CPU for `auto`. Select Metal explicitly. GLM generation defaults to `cuda:0`.

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

Use `--device cuda:0` or `--device metal:0` with the corresponding build. Add `--draft-model "<dspark-checkpoint>"` to enable speculative decoding.

### GLM with explicit cache budgets

Generate with explicit cache budgets; parameter and memory admission checks run before inference:

```bash
ff text generate \
    --model "<glm-checkpoint>" \
    --prompt 'Explain paging to a systems programmer.' \
    --device cuda:0 \
    --max-new-tokens 32 \
    --weight-source mmap \
    --host-cache-mib 4096 --host-cache-granularity tensor \
    --expert-cache-mib 2048 --expert-cache-replacement lru \
    --json
```

Adjust the cache budgets for the machine. The current GLM text profile handles one request at a time with at most 2,048 prompt-plus-generated tokens, and requires `--weight-source mmap`. Use `ff text generate-multi --help` for CUDA layer partitioning across devices.

### Resumable H3 generation

```bash
mkdir -p output
ff video generate \
    --model "<h3-checkpoint>" \
    --prompt 'A fishing boat crossing a calm lake at sunrise.' \
    --device cuda:0 \
    --short-edge 768 --aspect-ratio 16:9 --duration-seconds 4 \
    --output-dir output/boat
```

A successful run writes PNGs under `frames/`, `generated.wav` and recovery checkpoints in the run directory. Repeating the same command with the same output directory resumes an interrupted run.

Use `ff video --help` for first/last-frame and reference-conditioning workflows, and `ff music generate --help`, `ff 3d generate --help` or `ff similarity score --help` for the other adapters.

### 3D generation

Generate a colored mesh with TRELLIS.2 and its DINOv3 conditioner. The input is a prepared 512×512 PNG:

```bash
ff 3d generate \
    --model "<trellis2-checkpoint>" \
    --conditioner "<dinov3-checkpoint>" \
    --models-root "<models-root>" \
    --image "<prepared-image.png>" \
    --resolution 512 \
    --output "<mesh.ply>"
```

TRELLIS-1 uses the same task entry point with a compatible text or image conditioner. Its text path accepts `--prompt`; its image path accepts a prepared 518×518 PNG. Use `ff 3d --help` for generation and decoding operations.

See [Adding model adapters](docs/adding-models.md) for the CLI extension interface.

## License

Apache-2.0. See [LICENSE](LICENSE), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the upstream implementations these adapters follow.
