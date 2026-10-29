# flyingfish

LLM inference engine in Rust, targeting near-roofline decode and prefill on consumer and workstation GPUs.

- **Hardware-adaptive.** At setup the engine reads GPU memory, SM count, PCIe bandwidth, host RAM and cgroup limits, and derives the execution plan from them: chunk sizes, graph shapes and expert placement.
- **MoE expert tiering.** Experts live across GPU slots, pinned host memory and NVMe, with a CPU expert path for host-resident experts.
- **Speculative decoding.** MTP heads and DSpark draft models, with verification on the target model.
- **Published weights only.** Checkpoints are computed in their published format (BF16, FP8). Nothing is requantized or converted.

## Models

| Model | Architecture | Weights | Notes |
|---|---|---|---|
| Qwen3.8-27B | `Qwen3_5ForConditionalGeneration` | BF16, FP8 | MTP up to depth 4 |
| MiniCPM-V-4.7 | `MiniCPMV4_7ForConditionalGeneration` | BF16, FP8 | MoE expert tiering, DSpark draft, image input |
| MiniCPM5-2B | `LlamaForCausalLM` | BF16 | DSpark draft |
| Flint | `FlintForCausalLM` | BF16 | MoE expert tiering, sliding-window attention with sinks |

In progress: GLM-5.3-Flash and DeepSeek-V4.1-Flash.

## Platforms

| Platform | Backend | Status |
|---|---|---|
| NVIDIA Ampere / Ada (sm_86, sm_89) | CUDA | Supported |
| NVIDIA Blackwell (sm_120) | CUDA | In progress |
| Apple Silicon | Metal | Planned |
| AMD | ROCm / HIP | Planned |

Multi-GPU expert placement (layer split across devices) is in progress.

## Build

Requirements:
- the Rust toolchain pinned in `rust-toolchain.toml`;
- a CUDA toolkit with `nvcc` on `PATH` or under `CUDA_HOME`;
- an NVIDIA driver.

CUDA builds also require the device-only `nightly-2026-10-06` toolchain, with
`rust-src`, `llvm-tools`, `llvm-bitcode-linker`, and the `nvptx64-nvidia-cuda`
target. Install it and populate its core-build dependencies before the first
host build:

```sh
rustup toolchain install nightly-2026-10-06 --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker,rustfmt,clippy --target nvptx64-nvidia-cuda
RUSTFLAGS='-C target-cpu=sm_86' cargo +nightly-2026-10-06 build --manifest-path device/Cargo.toml --target nvptx64-nvidia-cuda -Z build-std=core --release
```

The host build uses the stable pin and builds native attention cubins for each
selected architecture with the device pin. Its device subprocess is offline;
it does not install toolchains or download dependencies. No-CUDA builds do not
invoke the device compiler.

```sh
cargo build --release
```

This builds `flyingfish` and `ff`. `ff` is a short alias: it runs the `flyingfish` binary in the same directory, so keep the two together.

Kernels are compiled for the GPUs that `nvidia-smi` reports. To build without a GPU visible, set the target explicitly, for example `CUDA_ARCH_LIST=89`.

`cargo build --no-default-features` builds the host-side code without CUDA. It has no GPU inference path.

## Usage

```sh
# One-time per machine and model: measure kernels and transfer rates.
flyingfish --model <model-dir> --runtime-dir <dir> calibrate

# Generate from a prompt.
flyingfish --model <model-dir> --runtime-dir <dir> generate --prompt "Hello" --max-new-tokens 64 --graph

# OpenAI-compatible server: /v1/completions and /v1/chat/completions.
flyingfish --model <model-dir> --runtime-dir <dir> serve --port 8080
```

Common options:
- `--runtime-dir` (or `FLYINGFISH_RUNTIME_DIR`) holds per-machine calibration. MoE models require it. It must be outside any model directory.
- `--draft-model <dir>` enables a DSpark draft model.
- `--mtp <0-4>` sets the MTP depth. It defaults to 4 when the checkpoint has MTP weights for an architecture with MTP decode (Qwen3.8); 0 selects plain decoding.
- `--kv-format affine8` stores an 8-bit KV cache for long contexts. The default is `bf16`.
- `--device <n>` selects the GPU.

`ff` takes the same commands and options as `flyingfish`, and uses the same calibration. `flyingfish --help` lists them all.

## Development

Every change passes:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --no-default-features -- -D warnings
cargo test --workspace
cargo test --workspace --no-default-features
```

GPU tests are `#[ignore]`d. Run them on a free GPU with `-- --include-ignored --test-threads=1`.
