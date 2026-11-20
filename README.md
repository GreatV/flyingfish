# flyingfish

Inference engine in Rust for language, multimodal, speech, diffusion, media-generation and image-to-3D models, targeting near-roofline decode and prefill on consumer and workstation GPUs.

- **Hardware-adaptive.** At setup the engine reads GPU memory, SM count, PCIe bandwidth, host RAM and cgroup limits, and derives the execution plan from them: chunk sizes, graph shapes, dense-layer streaming and expert placement.
- **MoE expert tiering.** Experts live across GPU slots, pinned host memory and NVMe, with a CPU expert path for host-resident experts whose arithmetic matches the GPU reduction. GLM-5.3-Flash and DeepSeek-V4.1-Flash read NVMe experts with direct IO and admit decode misses into the host tier by prefill expert rank.
- **Speculative decoding.** MTP heads and DSpark draft models, with verification on the target model.
- **Published weights only.** Checkpoints are computed in their published format (BF16, F16, F32, FP8, E2M1 experts, ModelOpt NVFP4, GGUF Q4_0 for the MiniMax-H3 diffusion transformer). Nothing is requantized or converted.
- **One kernel source for two vendors.** The CUDA C++ kernels compile with `nvcc` for NVIDIA and with `hipcc` for AMD; the host engine is shared and the device layer is selected at build time.

## Models

Supported models, their formats and backends are listed in [docs/models.md](docs/models.md).

## Library entry points

Library callers use two entry layers:

| Layer | Public surface | Backends | Resource contract |
|---|---|---|---|
| Model request | `RunOptions` with `minimax_h3::run_minimax_h3_i2va`; family types such as `minimax_h3::ImageRunOptions` and `music3::Music3Request` | CUDA for these examples; other families follow the model tables above | The caller supplies request inputs, device ordinal and runtime directory through the entry's arguments. The entry constructs and manages the stages for that request. |
| Stage | `backend::BackendDevice` and the encoder, flow, decoder and workspace types exported by `backend` | `BackendDevice` is exported with CUDA or ROCm; individual stages retain their model and backend restrictions | The caller constructs the device and stage objects, supplies stage inputs, and owns the runtime and workspace lifetimes. The device carries GPU identity, context, execution stream and upload stream. |

The generic `backend::Backend` loader and the autoregressive `model::layer_program` walker have these execution-domain errors:

| `Architecture` | Entry | Exact error |
|---|---|---|
| `Qwen3` | `Backend` load | `Qwen3 AR execution is not bound` |
| `Qwen3Vl` | `Backend` load and AR layer walker | `Qwen3-VL runs only as the MiniMax-H3 conditioner` |
| `Clover` | AR layer walker | `Clover canvas execution is not bound` |
| `DiffusionGemma` | AR layer walker | `DiffusionGemma layer program is not bound yet` |

Clover and DiffusionGemma load on the separate eager canvas path. Their graph capture returns `canvas graph execution is not bound`.

## Platforms

| Platform | Backend | Status |
|---|---|---|
| NVIDIA Ampere and Ada (sm_86, sm_89) | CUDA | Supported |
| AMD RDNA3 (gfx1100) | ROCm and HIP | Supported for the models marked ROCm above |
| NVIDIA Blackwell (sm_120) | CUDA | In progress |
| Apple Silicon | Metal | Planned |

Two-GPU MoE expert residency with peer compute is supported.

## Build

Requirements:
- the Rust toolchain pinned in `<rust-toolchain.toml>`;
- a CUDA toolkit with `nvcc` on `PATH` or under `CUDA_HOME`;
- an NVIDIA driver;
- ffmpeg on `PATH` for MP4 output. Y4M+WAV output does not require it.

CUDA builds also require the device-only `nightly-2026-10-06` toolchain, with
`rust-src`, `llvm-tools`, `llvm-bitcode-linker`, and the `nvptx64-nvidia-cuda`
target. Install it and populate its core-build dependencies before the first
host build. Replace `<device-manifest>` with the device crate's `<Cargo.toml>` file:

```sh
rustup toolchain install nightly-2026-10-06 --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker,rustfmt,clippy --target nvptx64-nvidia-cuda
RUSTFLAGS='-C target-cpu=sm_86' cargo +nightly-2026-10-06 build --manifest-path <device-manifest> --target nvptx64-nvidia-cuda -Z build-std=core --release
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

For AMD GPUs, build the ROCm backend instead of CUDA:

```sh
cargo build --release --no-default-features --features rocm
```

It needs `hipcc` (found through `HIP_HOME`, `ROCM_PATH` or `PATH`) and the hipBLAS and hipBLASLt libraries. Kernels are compiled for the architectures that `rocminfo` reports; set `ROCM_ARCH_LIST` (for example `gfx1100`) to build without a GPU visible.

## Usage

See [CLI usage by model](https://github.com/GreatV/flyingfish/blob/main/docs/usage.md) for required inputs, options, defaults and output formats.

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
ff --model <MiniMax-Music3-dir> music --prompt "A warm acoustic pop song" --lyrics "[Verse]\nMorning light" --seed 7 --frames 25 -o <song.wav>

# Generate HyperFlow video and audio; either keyframe option can be omitted.
ff --model <HyperFlow-H3-Flash-dir> video --base-model <MiniMax-H3-dir> --prompt "A boat crosses a quiet lake" --image <first.png> --last-image <last.png> --frames 124 --seed 42 -o <out.mp4>
```

Common options:
- MiniMax-H3 video requires a keyframe, a frame count and a seed. Requested frames round up to `17n+5`, so 136 resolves to 141; the published duration is 5-15 seconds at 24 fps. No documented frame-count or seed defaults exist.
- HyperFlow uses the published `fl2va_int8_convrot` component and MiniMax-H3 companion files supplied by `--base-model`, without an external LoRA or GGUF replacement. It supports text-only, first, last and first+last keyframes; ref2va media inputs are not supported. Frames round up to `17n+5`, within 124-362 aligned frames at 24 fps. The sampler uses the published six-step grid with video and audio shifts 12 and 3, respectively.
- MiniMax-H3-Image `image` requires a prompt, `WIDTHxHEIGHT` size and seed. `--expand <dir>` replaces the prompt through the expander; repeated `--edit <image>` supplies upright RGB8 references. Output is PNG at the requested size, and `--json` reports size, seed and reference count.
- MiniMax-Music3 `music` requires a caption, lyrics and seed. `--frames` defaults to 1500 (60 seconds at 25 fps); output is stereo 44.1 kHz PCM16 WAV, or WAV bytes to stdout when `-o` is omitted.
- Media commands accept `--verbose` for placement, Lt and decode-plan diagnostics. Video, image, speak, music, transcribe, diarize and embed default stderr contains progress and errors only, with timing fields hidden unless verbose is enabled; other commands retain their existing diagnostics.
- Breeze-TTS-2 `speak` uses the plain `[S0]` prompt and reads `<generation_config.json>`: temperature 0.9 for backbone and depth, top-k 50 and top-p 1.0 from the omitted library defaults, repetition penalty 1.1 on generated backbone codebook-0 tokens, at most 750 frames, and backbone EOS 2051. The default seed is 42. Output is mono 24 kHz PCM16 WAV; omit `-o` to write WAV bytes to stdout.
- The runtime directory defaults to `~/.flyingfish`. `--runtime-dir` or `FLYINGFISH_RUNTIME_DIR` overrides it. It must be outside any model directory, and no path component is named `models`.
- `--draft-model <dir>` enables a DSpark draft model.
- `--mtp <0-4>` sets the MTP depth. It defaults to 4 when the checkpoint has MTP weights for an architecture with MTP decode (Qwen3.8, GLM-5.3-Flash); 0 selects plain decoding. GLM-5.3-Flash drafts one token per round at any nonzero depth. MTP decode is CUDA-only, so ROCm builds need `--mtp 0` for such checkpoints.
- `--kv-format affine8` stores an 8-bit KV cache for long contexts. The default is `bf16`.
- `--device <n>` selects the GPU.

`ff` takes the same commands and options as `flyingfish`, and uses the same calibration. `ff --help` lists them all.

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
