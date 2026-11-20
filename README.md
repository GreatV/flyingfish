# flyingfish

Inference engine in Rust for language, multimodal, speech, diffusion, media-generation and image-to-3D models, targeting near-roofline decode and prefill on consumer and workstation GPUs.

- **Hardware-adaptive.** At setup the engine reads GPU memory, SM count, PCIe bandwidth, host RAM and cgroup limits, and derives the execution plan from them: chunk sizes, graph shapes, dense-layer streaming and expert placement.
- **MoE expert tiering.** Experts live across GPU slots, pinned host memory and NVMe, with a CPU expert path for host-resident experts whose arithmetic matches the GPU reduction. GLM-5.3-Flash and DeepSeek-V4.1-Flash read NVMe experts with direct IO and admit decode misses into the host tier by prefill expert rank.
- **Speculative decoding.** MTP heads and DSpark draft models, with verification on the target model.
- **Published weights only.** Checkpoints are computed in their published format (BF16, F16, F32, FP8, E2M1 experts, ModelOpt NVFP4, GGUF Q4_0 for the MiniMax-H3 diffusion transformer). Nothing is requantized or converted.
- **One kernel source for two vendors.** The CUDA C++ kernels compile with `nvcc` for NVIDIA and with `hipcc` for AMD; the host engine is shared and the device layer is selected at build time.

## Documentation

- [Models](docs/models.md): supported models, published formats and backends.
- [CLI usage](docs/usage.md): commands, required inputs, options, defaults and output formats for each model.
- [Library](docs/library.md): calling the engine from Rust, with the entry point, request and output of each model family.

## Platforms

| Platform | Backend | Status |
|---|---|---|
| NVIDIA Ampere and Ada (sm_86, sm_89) | CUDA | Supported |
| AMD RDNA3 (gfx1100) | ROCm and HIP | Supported for the models whose backends in [docs/models.md](docs/models.md) include ROCm |
| NVIDIA Blackwell (sm_120) | CUDA | In progress |
| Apple Silicon | Metal | Planned |

Two-GPU MoE expert residency with peer compute is supported.

## Build

### NVIDIA (CUDA, the default)

Requirements:
- the stable Rust toolchain pinned in the repository's `rust-toolchain.toml`;
- the device nightly toolchain below;
- a CUDA toolkit with `nvcc` on `PATH` or under `CUDA_HOME`;
- an NVIDIA driver;
- ffmpeg on `PATH` for MP4 output. Y4M+WAV output does not require it.

The device crate, which compiles the Rust GPU kernels to PTX, needs the pinned `nightly-2026-10-06` toolchain with `rust-src`, `llvm-tools`, `llvm-bitcode-linker` and the `nvptx64-nvidia-cuda` target: the `ptx-kernel` ABI, inline assembly on NVPTX and `-Z build-std=core` are not available on stable Rust. Install it and build the device crate once to populate its core-build dependencies before the first host build. `<device-manifest>` is the device crate's `Cargo.toml`:

```sh
rustup toolchain install nightly-2026-10-06 --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker,rustfmt,clippy --target nvptx64-nvidia-cuda
RUSTFLAGS='-C target-cpu=sm_86' cargo +nightly-2026-10-06 build --manifest-path <device-manifest> --target nvptx64-nvidia-cuda -Z build-std=core --release
```

Then build the engine with the stable toolchain:

```sh
cargo build --release
```

The host build compiles native attention cubins for each selected architecture with the device toolchain. That step runs offline; it does not install toolchains or download dependencies. Kernels are compiled for the GPUs that `nvidia-smi` reports; to build without a GPU visible, set the targets explicitly, for example `CUDA_ARCH_LIST=89`.

The build produces `flyingfish` and `ff`. `ff` is a short alias that runs the `flyingfish` binary in the same directory, so keep the two together.

### AMD (ROCm)

```sh
cargo build --release --no-default-features --features rocm
```

It needs `hipcc` (found through `HIP_HOME`, `ROCM_PATH` or `PATH`) and the hipBLAS and hipBLASLt libraries; it does not use the device nightly toolchain. Kernels are compiled for the architectures that `rocminfo` reports; set `ROCM_ARCH_LIST` (for example `gfx1100`) to build without a GPU visible.

### Host-only

`cargo build --no-default-features` builds the host-side code without a GPU backend and does not invoke the device compiler. It has no GPU inference path.

## Usage

Common commands; [docs/usage.md](docs/usage.md) documents every command and option.

```sh
# One-time per machine and model: measure kernels and transfer rates.
ff --model <model-dir> calibrate

# Generate from a prompt.
ff --model <model-dir> generate --prompt "Hello" --max-new-tokens 64 --graph

# Transcribe a 16 kHz WAV (PCM16, PCM24 or F32) with duration-derived SRT cues.
ff --model <parakeet-dir> transcribe <clip.wav> --format srt

# Write speaker segments as RTTM.
ff --model <diarization-dir> diarize <meeting.wav> -o <meeting.rttm>

# Embed raw text as a 768-value JSON vector.
ff --model <EmbeddingGemma-2-dir> embed --text "a red cube"

# OpenAI-compatible server: http://<host>:<port>/v1/completions and http://<host>:<port>/v1/chat/completions.
ff --model <model-dir> serve --port 8080

# Generate MiniMax-H3 video and audio from a keyframe.
ff --model <MiniMax-H3-dir> video --prompt "A boat crosses a quiet lake" --image <keyframe.png> --frames 136 --seed 42 --lora <minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors> -o <out.mp4>

# Raw output: <out.y4m> and <out.wav>. Add --json to print output metadata as JSON.
ff --model <MiniMax-H3-dir> video --prompt "A boat crosses a quiet lake" --image <keyframe.png> --frames 136 --seed 42 --lora <minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors> -o <out.y4m> --json

# Generate a MiniMax-H3-Image PNG, optionally with --expand or repeated --edit references.
ff --model <MiniMax-H3-Image-dir> image --prompt "A red cube on a wooden table" --size 1024x1024 --seed 42 -o <cube.png> --json

# Synthesize Breeze-TTS-2 speech with the published sampling configuration.
ff --model <Breeze-TTS-2-dir> speak "Welcome aboard. Your journey begins now." --seed 42 -o <speech.wav>

# Generate MiniMax-Music3 stereo audio from a caption and lyrics.
ff --model <MiniMax-Music3-dir> music --prompt "A warm acoustic pop song" --lyrics "[Verse] Morning light" --seed 7 --frames 25 -o <song.wav>

# Generate HyperFlow video and audio; either keyframe option can be omitted.
ff --model <HyperFlow-H3-Flash-dir> video --base-model <MiniMax-H3-dir> --prompt "A boat crosses a quiet lake" --image <first.png> --last-image <last.png> --frames 124 --seed 42 -o <out.mp4>
```

`--model` goes before the command; other shared options may also follow it. `--device <n>` selects the GPU, `--runtime-dir` (default `~/.flyingfish`, or `FLYINGFISH_RUNTIME_DIR`) holds per-machine calibration and must be outside any model directory, `--verbose` prints placement, Lt and decode-plan diagnostics, and `--draft-model`, `--mtp` and `--kv-format` control speculative decoding and the KV cache. Each model's required inputs, defaults and output formats are in [docs/usage.md](docs/usage.md).

`ff` takes the same commands and options as `flyingfish` and uses the same calibration. `ff --help` lists them all.

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
