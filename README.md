# flyingfish

Inference engine in Rust for language, multimodal, speech, diffusion, media-generation and image-to-3D models, targeting near-roofline decode and prefill on consumer and workstation GPUs.

- **Hardware-adaptive.** At setup the engine reads GPU memory, SM count, PCIe bandwidth, host RAM and cgroup limits, and derives the execution plan from them: chunk sizes, graph shapes, dense-layer streaming and expert placement.
- **MoE expert tiering.** Experts live across GPU slots, pinned host memory and NVMe, with a CPU expert path for host-resident experts whose arithmetic matches the GPU reduction. GLM-5.3-Flash and DeepSeek-V4.1-Flash read NVMe experts with direct IO and admit decode misses into the host tier by prefill expert rank.
- **Speculative decoding.** MTP heads and DSpark draft models, with verification on the target model.
- **Published weights only.** Checkpoints are computed in their published format (BF16, F16, F32, FP8, E2M1 experts, ModelOpt NVFP4, GGUF Q4_0 for the MiniMax-H3 diffusion transformer). Nothing is requantized or converted.
- **One kernel source for two vendors.** The CUDA C++ kernels compile with `nvcc` for NVIDIA and with `hipcc` for AMD; the host engine is shared and the device layer is selected at build time.

## Models

Language and multimodal models:

| Model | Architecture | Notes | Backends |
|---|---|---|---|
| Qwen3.8-27B | `Qwen3_5ForConditionalGeneration` | BF16 and FP8; image input; MTP up to depth 4 on CUDA (ROCm runs it with `--mtp 0`) | CUDA, ROCm |
| MiniCPM-V-4.7 | `MiniCPMV4_7ForConditionalGeneration` | BF16 and FP8; MoE expert tiering, DSpark draft, image input | CUDA, ROCm |
| MiniCPM5-2B | `LlamaForCausalLM` | DSpark draft | CUDA, ROCm |
| Nanbeige4.3-33B | `LlamaForCausalLM` | BF16 dense layers streamed through device slots when they do not fit | CUDA |
| Flint | `FlintForCausalLM` | MoE expert tiering, sliding-window attention with sinks | CUDA, ROCm |
| LoopVL | `LoopVLForConditionalGeneration` | Recurrent text decode, image input with prefix attention | CUDA, ROCm |
| DeepSeek-V4.1-Flash | `DeepseekV41ForCausalLM` | Block FP8 projections, E2M1 experts, compressed latent attention, mHC; segmented CUDA-graph decode | CUDA |
| GLM-5.3-Flash | `Glm5NextForConditionalGeneration` | FP8 routed experts tiered across GPU, pinned host and NVMe; DSA and KDA attention, mHC; segmented CUDA-graph decode; MTP with one draft token per round | CUDA |
| Gemma 4 E2B, E4B, 12B, 31B | `Gemma4ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration` | Text for all sizes; image and audio input for E2B, E4B and 12B; image input for 31B; 12B and 31B stream dense layers when they do not fit | CUDA |
| EmbeddingGemma-2 | `EmbeddingGemma2Model` | Text embeddings through `embed` with JSON output; image and audio embeddings remain library-only | CUDA |
| clef-flash | `Qwen3_5ForConditionalGeneration` | Hybrid linear-attention and full-attention text tower with plain decode; the vision tower and MTP layers are not run; the joint schema head scoring path is gated, with no CLI or library entry | CUDA |
| MiniMax-H3-Image-PE | `Qwen3_5ForConditionalGeneration` | ModelOpt NVFP4 projections executed as W4A16 (weights decoded to BF16, no FP4 tensor-core path); image input | CUDA |

Diffusion language models:

| Model | Architecture | Notes | Backends |
|---|---|---|---|
| Clover-35B-A3B | `CloverMoeModelLM` | Canvas inference with tiered experts | CUDA |
| DiffusionGemma-26B-A4B | `DiffusionGemmaForBlockDiffusion` | Canvas inference | CUDA |

Speech (library entry points and CLI):

| Model | Notes | Backends |
|---|---|---|
| Parakeet TDT | Log-mel frontend, 24-layer FastConformer encoder, greedy TDT decode with one CUDA graph per decode step; `flyingfish::parakeet::ParakeetTdt` and `transcribe`; txt/json/srt/vtt output with duration-derived cues | CUDA |
| Canary-1B-v2 | FastConformer encoder, AED decoder, greedy decode with cached attention and one CUDA graph per decode step; `flyingfish::canary::Canary` and `transcribe` with published language/punctuation prompts; Canary subtitles are whole-clip only | CUDA |
| Nemotron 3.5 ASR streaming 0.6B | FastConformer encoder over a full clip and as a chunked stream with cached context; the RNNT decoder and joint are not run; `flyingfish::parakeet::ParakeetEncoder` | CUDA |
| parakeet realtime EOU 120M | Reads the `.nemo` container in place; offline greedy RNNT decode and cache-aware streaming with a reset on end-of-utterance; `flyingfish::eou::Eou` and `transcribe`; offline subtitles are whole-clip only, streaming subtitles are per utterance | CUDA |
| Nemotron-3-Diarization | Offline and streaming speaker diarization with an arrival-order speaker cache; `flyingfish::diarization::Diarization` and `diarize`; RTTM output with overlaps preserved | CUDA |
| Multitalker Parakeet streaming 0.6B | Reads the `.nemo` container in place; 1-4 speaker lanes through the pre-encode, layer-0 speaker and background kernels, and 24 conformer layers with block-anchored attention windows; greedy RNNT decode over speaker-keyed hypotheses with per-lane prediction state; `flyingfish::multitalker::Multitalker` | CUDA |
| Breeze-TTS-2 | Text encoder, backbone and depth decoder; greedy decode over supplied audio-code frames with per-layer KV caches and CUDA-graph replay; audio tokenizer decode to a 24 kHz waveform; `flyingfish::backend::{Breeze, BreezeTokenizer}`. No sampling and no free-running text-to-speech entry | CUDA |

Media generation (library entry points and CLI):

| Model | Notes | Backends |
|---|---|---|
| MiniMax-H3 | Image-to-video with audio: Qwen3-VL conditioner, streamed diffusion transformer with Turbo LoRA, video and audio decoders, native prompt, canvas and keyframe producers, noise drawn from the request seed; the diffusion transformer also loads from a GGUF Q4_0 file; `flyingfish::minimax_h3::run_minimax_h3_i2va`; the `video` command writes MP4 or Y4M+WAV | CUDA |
| MiniMax-H3-Image | Prompt-to-image is usable. A prompt and optional upright RGB8 references become RGB8 through the Qwen3-VL conditioner, native VAE encode, six-step Euler under a derived placement, native VAE decode, truncation toward zero and Lanczos-3; one Philox draw from the request seed; `flyingfish::minimax_h3::image_latents`, `flyingfish::minimax_h3::image_encode`, `flyingfish::minimax_h3::image_decode` and the `image` command writes PNG. `image_sample` runs the six evaluations on supplied embeddings and latent rows | CUDA |
| MiniMax-Music3 | Autoregressive stage and synthesis stage (condition encoder, flow transformer with guidance, vocoder) in one entry from token ids and a seed to stereo PCM; `flyingfish::music3::generate` | CUDA |

Image to 3D (library entry points and CLI):

| Model | Notes | Backends |
|---|---|---|
| Idol | DINOv3 and FLUX.2 VAE encoders, latent flow denoiser, octree and Gaussian decoders: a preprocessed 1024 px image to a Gaussian splat and a camera; `flyingfish::backend::IdolPipelineRuntime` | CUDA |
| Pixal3D | Single-view and multiview: RGBA views with camera parameters and supplied noise, through sparse-structure, shape and texture flows, to a raw mesh with six material channels; `flyingfish::backend::PixalGeneration` | CUDA |
| Hi3D | DINOv2 and CLIP image conditioning, sparse flow and sparse VAE decoder at 512, 1024 or 1536, refining a supplied coarse mesh; raw mesh or a hole-filled, simplified and normalized mesh; `flyingfish::backend::Hi3DGeneration` | CUDA |
| TRELLIS.2-4B | Image to a textured mesh on the default 1024-cascade pipeline: DINOv3 conditioning, sparse-structure flow with Euler interval-CFG sampling, cascade shape and texture structured-latent flows, FlexiDualGrid mesh and SparseUnet texture decoders, seeded stage noise; raw mesh only (no fill, simplify, UV, bake or render), RGBA input with the background already removed, and the decoder and structure-latent outputs differ from the reference at F16 quanta; `flyingfish::backend::TrellisTo3d`, CLI `flyingfish --model <dir> to-3d <image.png> --dino <dir> --seed 42 -o mesh.ply` | CUDA |

In progress: TRELLIS.2 image-to-3D (flow sampling and the structured-latent, mesh and texture stages landed; remaining: render, GLB and bake, the shape and texture encoders, and closing the decoder and structure-latent differences from the reference).

## Platforms

| Platform | Backend | Status |
|---|---|---|
| NVIDIA Ampere / Ada (sm_86, sm_89) | CUDA | Supported |
| AMD RDNA3 (gfx1100) | ROCm / HIP | Supported for the models marked ROCm above |
| NVIDIA Blackwell (sm_120) | CUDA | In progress |
| Apple Silicon | Metal | Planned |

Two-GPU MoE expert residency with peer compute is supported.

## Build

Requirements:
- the Rust toolchain pinned in `rust-toolchain.toml`;
- a CUDA toolkit with `nvcc` on `PATH` or under `CUDA_HOME`;
- an NVIDIA driver;
- ffmpeg on `PATH` for MP4 output. Y4M+WAV output does not require it.

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

For AMD GPUs, build the ROCm backend instead of CUDA:

```sh
cargo build --release --no-default-features --features rocm
```

It needs `hipcc` (found through `HIP_HOME`, `ROCM_PATH` or `PATH`) and the hipBLAS and hipBLASLt libraries. Kernels are compiled for the architectures that `rocminfo` reports; set `ROCM_ARCH_LIST` (for example `gfx1100`) to build without a GPU visible.

## Usage

```sh
# One-time per machine and model: measure kernels and transfer rates.
flyingfish --model <model-dir> --runtime-dir <dir> calibrate

# Generate from a prompt.
flyingfish --model <model-dir> --runtime-dir <dir> generate --prompt "Hello" --max-new-tokens 64 --graph

# Transcribe a 16 kHz WAV (PCM16, PCM24 or F32) with duration-derived SRT cues.
flyingfish --model <parakeet-dir> --runtime-dir <dir> transcribe clip.wav --format srt

# Write speaker segments as RTTM.
flyingfish --model <diarization-dir> --runtime-dir <dir> diarize meeting.wav -o meeting.rttm

# Embed raw text as a 768-value JSON vector.
flyingfish --model <EmbeddingGemma-2-dir> --runtime-dir <dir> embed --text "a red cube"

# OpenAI-compatible server: /v1/completions and /v1/chat/completions.
flyingfish --model <model-dir> --runtime-dir <dir> serve --port 8080

# Generate MiniMax-H3 video and audio from a keyframe.
flyingfish --model <MiniMax-H3-dir> --runtime-dir <dir> video --prompt "A boat crosses a quiet lake" --image keyframe.png --frames 136 --seed 42 --lora minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors -o out.mp4

# Raw output: out.y4m and out.wav. Add --json to print output metadata as JSON.
flyingfish --model <MiniMax-H3-dir> --runtime-dir <dir> video --prompt "A boat crosses a quiet lake" --image keyframe.png --frames 136 --seed 42 --lora minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors -o out.y4m --json

# Generate a MiniMax-H3-Image PNG, optionally with --expand or repeated --edit references.
flyingfish --model <MiniMax-H3-Image-dir> --runtime-dir <dir> image --prompt "A red cube on a wooden table" --size 1024x1024 --seed 42 -o cube.png --json
```

Common options:
- MiniMax-H3 video requires a keyframe, a frame count and a seed. Requested frames round up to `17n+5`, so 136 resolves to 141; the published duration is 5-15 seconds at 24 fps. No documented frame-count or seed defaults exist.
- MiniMax-H3-Image `image` requires a prompt, `WIDTHxHEIGHT` size and seed. `--expand <dir>` replaces the prompt through the expander; repeated `--edit <image>` supplies upright RGB8 references. Output is PNG at the requested size, and `--json` reports size, seed and reference count.
- Media commands accept `--verbose` for placement, Lt and decode-plan diagnostics. Video, image, transcribe, diarize and embed default stderr contains progress and errors only, with timing fields hidden unless verbose is enabled; other commands retain their existing diagnostics.
- `--runtime-dir` (or `FLYINGFISH_RUNTIME_DIR`) holds per-machine calibration. MoE models require it. It must be outside any model directory.
- `--draft-model <dir>` enables a DSpark draft model.
- `--mtp <0-4>` sets the MTP depth. It defaults to 4 when the checkpoint has MTP weights for an architecture with MTP decode (Qwen3.8, GLM-5.3-Flash); 0 selects plain decoding. GLM-5.3-Flash drafts one token per round at any nonzero depth. MTP decode is CUDA-only, so ROCm builds need `--mtp 0` for such checkpoints.
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
