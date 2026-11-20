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
| Parakeet TDT | Log-mel frontend, 24-layer FastConformer encoder, greedy TDT decode with one CUDA graph per decode step; `flyingfish::parakeet::ParakeetTdt` and `transcribe`; txt, json, srt and vtt output with duration-derived cues | CUDA |
| Canary-1B-v2 | FastConformer encoder, AED decoder, greedy decode with cached attention and one CUDA graph per decode step; `flyingfish::canary::Canary` and `transcribe` with published language and punctuation prompts; Canary subtitles are whole-clip only | CUDA |
| Nemotron 3.5 ASR streaming 0.6B | Full-clip and cache-aware chunked-stream FastConformer encoding with shared greedy RNNT prediction and joint decode; blank 13087 and at most 10 symbols per step; `transcribe` outputs txt and json and whole-clip srt and vtt cues; `flyingfish::nemotron::transcribe` | CUDA |
| parakeet realtime EOU 120M | Reads the `.nemo` container in place; offline greedy RNNT decode and cache-aware streaming with a reset on end-of-utterance; `flyingfish::eou::Eou` and `transcribe`; offline subtitles are whole-clip only, streaming subtitles are per utterance | CUDA |
| Nemotron-3-Diarization | Offline and streaming speaker diarization with an arrival-order speaker cache; `flyingfish::diarization::Diarization` and `diarize`; RTTM output with overlaps preserved | CUDA |
| Multitalker Parakeet streaming 0.6B | Reads the `.nemo` container in place; 1-4 speaker lanes through the pre-encode, layer-0 speaker and background kernels, and 24 conformer layers with block-anchored attention windows; greedy RNNT decode over speaker-keyed hypotheses with per-lane prediction state; `transcribe --diar-model` runs it end to end (Nemotron-3-Diarization arrival-order channels 0-3 as the lanes, speaker kernels by default, `--masked` for lane masking, per-speaker text with word timestamps); a fifth active diarization channel errors instead of being dropped; kernel-mode free running diverges from the published wrapper's batch-1 pre-encode (measured 4.272e-4); `flyingfish::multitalker::Multitalker` | CUDA |
| Breeze-TTS-2 | Text encoder, backbone and depth decoder; greedy decode over supplied audio-code frames with per-layer KV caches and CUDA-graph replay; published-config sampling for free-running text-to-speech with seeded Philox draws; audio tokenizer decode to a 24 kHz waveform; `flyingfish::breeze::speak` and the `speak` command | CUDA |

Media generation (library entry points and CLI):

| Model | Notes | Backends |
|---|---|---|
| MiniMax-H3 | Image-to-video with audio: Qwen3-VL conditioner, streamed diffusion transformer with Turbo LoRA, video and audio decoders, native prompt, canvas and keyframe producers, noise drawn from the request seed; the diffusion transformer also loads from a GGUF Q4_0 file; `flyingfish::minimax_h3::run_minimax_h3_i2va`; the `video` command writes MP4 or Y4M+WAV | CUDA |
| HyperFlow H3 Flash | Usable through `video`: published fl2va INT8 convrot and low-rank diffusion transformer; six-step text-to-video with audio, optional first or last keyframes, MiniMax-H3 companion conditioner and VAEs; `flyingfish::minimax_h3::run_hyperflow_h3_video`; the `video` command writes MP4 or Y4M+WAV | CUDA |
| MiniMax-H3-Image | Prompt-to-image is usable. A prompt and optional upright RGB8 references become RGB8 through the Qwen3-VL conditioner, native VAE encode, six-step Euler under a derived placement, native VAE decode, truncation toward zero and Lanczos-3; one Philox draw from the request seed; `flyingfish::minimax_h3::image_latents`, `flyingfish::minimax_h3::image_encode`, `flyingfish::minimax_h3::image_decode` and the `image` command writes PNG. `image_sample` runs the six evaluations on supplied embeddings and latent rows | CUDA |
| MiniMax-Music3 | Autoregressive stage and synthesis stage (condition encoder, flow transformer with guidance, vocoder) in one entry from token ids and a seed to stereo PCM; published caption and lyrics prompt construction with paired conditional and unconditional ids and the `music` command writing 44.1 kHz stereo WAV; `flyingfish::music3::generate` | CUDA |

Image to 3D (library entry points and CLI):

| Model | Notes | Backends |
|---|---|---|
| Idol | DINOv3 and FLUX.2 VAE encoders, latent flow denoiser, octree and Gaussian decoders: a preprocessed 1024 px image to a Gaussian splat and a camera; `flyingfish::backend::IdolPipelineRuntime` | CUDA |
| Pixal3D | Single-view and multiview: RGBA views with camera parameters, through sparse-structure, shape and texture flows, to a raw mesh with six material channels; CLI `ff --model <dir> to-3d <image.png> --dino <dir> --naf <dir> --camera <cam.json> [--view <image2.png> --camera <cam2.json> ...] [--resolution 1024|1536] --seed 42 -o <mesh.ply>`; the camera JSON carries camera_angle_x, mesh_scale and a world_to_camera extrinsic (flat 12 floats or the 4x4 nested form; required for multiview, where a shared look-at would project every view from one direction; without it the single view uses the look-at at (0,0,distance)); MoGe-2 estimation is not integrated and a missing camera errors; the four stage noise draws are flyingfish's philox counter-stream convention (streams 1-4), not the publisher's torch RNG; `flyingfish::backend::PixalGeneration` and `PixalSeededRequest` | CUDA |
| Hi3D | DINOv2 and CLIP image conditioning, sparse flow and sparse VAE decoder at 512, 1024 or 1536, refining a supplied coarse mesh; raw mesh or a hole-filled, simplified and normalized mesh; same-seed reruns are bit-stable; CLI `ff --model <dir> to-3d <image.png> --dino <dir> --clip <dir> --mesh <coarse.ply> [--resolution 512|1024|1536] [--orientation 1] --seed 42 -o <mesh.ply>`; `flyingfish::backend::Hi3DGeneration` | CUDA |
| TRELLIS.2-4B | Image to a textured mesh on the default 1024-cascade pipeline: DINOv3 conditioning, sparse-structure flow with Euler interval-CFG sampling, cascade shape and texture structured-latent flows, FlexiDualGrid mesh and SparseUnet texture decoders, seeded stage noise; raw mesh only (no fill, simplify, UV, bake or render), RGBA input with the background already removed, and the decoder and structure-latent outputs differ from the reference at F16 quanta; `flyingfish::backend::TrellisTo3d`, CLI `ff --model <dir> to-3d <image.png> --dino <dir> --seed 42 -o <mesh.ply>` | CUDA |

In progress: TRELLIS.2 image-to-3D (flow sampling and the structured-latent, mesh and texture stages landed; remaining: render, GLB and bake, the shape and texture encoders, and closing the decoder and structure-latent differences from the reference).

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
ff --model <model-dir> --runtime-dir <dir> calibrate

# Generate from a prompt.
ff --model <model-dir> --runtime-dir <dir> generate --prompt "Hello" --max-new-tokens 64 --graph

# Transcribe a 16 kHz WAV (PCM16, PCM24 or F32) with duration-derived SRT cues.
ff --model <parakeet-dir> --runtime-dir <dir> transcribe <clip.wav> --format srt

# Write speaker segments as RTTM.
ff --model <diarization-dir> --runtime-dir <dir> diarize <meeting.wav> -o <meeting.rttm>

# Embed raw text as a 768-value JSON vector.
ff --model <EmbeddingGemma-2-dir> --runtime-dir <dir> embed --text "a red cube"

# OpenAI-compatible server: http://<host>:<port>/v1/completions and http://<host>:<port>/v1/chat/completions.
ff --model <model-dir> --runtime-dir <dir> serve --port 8080

# Generate MiniMax-H3 video and audio from a keyframe.
ff --model <MiniMax-H3-dir> --runtime-dir <dir> video --prompt "A boat crosses a quiet lake" --image <keyframe.png> --frames 136 --seed 42 --lora <minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors> -o <out.mp4>

# Raw output: <out.y4m> and <out.wav>. Add --json to print output metadata as JSON.
ff --model <MiniMax-H3-dir> --runtime-dir <dir> video --prompt "A boat crosses a quiet lake" --image <keyframe.png> --frames 136 --seed 42 --lora <minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors> -o <out.y4m> --json

# Generate a MiniMax-H3-Image PNG, optionally with --expand or repeated --edit references.
ff --model <MiniMax-H3-Image-dir> --runtime-dir <dir> image --prompt "A red cube on a wooden table" --size 1024x1024 --seed 42 -o <cube.png> --json

# Synthesize Breeze-TTS-2 speech with the published sampling configuration.
ff --model <Breeze-TTS-2-dir> --runtime-dir <dir> speak "Welcome aboard. Your journey begins now." --seed 42 -o <speech.wav>

# Generate MiniMax-Music3 stereo audio from a caption and lyrics.
ff --model <MiniMax-Music3-dir> --runtime-dir <dir> music --prompt "A warm acoustic pop song" --lyrics "[Verse]\nMorning light" --seed 7 --frames 25 -o <song.wav>

# Generate HyperFlow video and audio; either keyframe option can be omitted.
ff --model <HyperFlow-H3-Flash-dir> --runtime-dir <dir> video --base-model <MiniMax-H3-dir> --prompt "A boat crosses a quiet lake" --image <first.png> --last-image <last.png> --frames 124 --seed 42 -o <out.mp4>
```

Common options:
- MiniMax-H3 video requires a keyframe, a frame count and a seed. Requested frames round up to `17n+5`, so 136 resolves to 141; the published duration is 5-15 seconds at 24 fps. No documented frame-count or seed defaults exist.
- HyperFlow uses the published `fl2va_int8_convrot` component and MiniMax-H3 companion files supplied by `--base-model`, without an external LoRA or GGUF replacement. It supports text-only, first, last and first+last keyframes; ref2va media inputs are not supported. Frames round up to `17n+5`, within 124-362 aligned frames at 24 fps. The sampler uses the published six-step grid with video and audio shifts 12 and 3, respectively.
- MiniMax-H3-Image `image` requires a prompt, `WIDTHxHEIGHT` size and seed. `--expand <dir>` replaces the prompt through the expander; repeated `--edit <image>` supplies upright RGB8 references. Output is PNG at the requested size, and `--json` reports size, seed and reference count.
- MiniMax-Music3 `music` requires a caption, lyrics and seed. `--frames` defaults to 1500 (60 seconds at 25 fps); output is stereo 44.1 kHz PCM16 WAV, or WAV bytes to stdout when `-o` is omitted.
- Media commands accept `--verbose` for placement, Lt and decode-plan diagnostics. Video, image, speak, music, transcribe, diarize and embed default stderr contains progress and errors only, with timing fields hidden unless verbose is enabled; other commands retain their existing diagnostics.
- Breeze-TTS-2 `speak` uses the plain `[S0]` prompt and reads `<generation_config.json>`: temperature 0.9 for backbone and depth, top-k 50 and top-p 1.0 from the omitted library defaults, repetition penalty 1.1 on generated backbone codebook-0 tokens, at most 750 frames, and backbone EOS 2051. The default seed is 42. Output is mono 24 kHz PCM16 WAV; omit `-o` to write WAV bytes to stdout.
- `--runtime-dir` (or `FLYINGFISH_RUNTIME_DIR`) holds per-machine calibration. MoE models require it. It must be outside any model directory.
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
