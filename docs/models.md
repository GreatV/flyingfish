# Models

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
| Idol | DINOv3 and FLUX.2 VAE encoders, latent flow denoiser, octree and Gaussian decoders: a preprocessed 1024 px image to a Gaussian splat and a camera; `flyingfish::backend::IdolPipelineRuntime` is exported but cannot yet be opened from outside the crate, and there is no CLI command | CUDA |
| Pixal3D | Single-view and multiview: RGBA views with camera parameters, through sparse-structure, shape and texture flows, to a raw mesh with six material channels; CLI `ff --model <dir> to-3d <image.png> --dino <dir> --naf <dir> --camera <cam.json> [--view <image2.png> --camera <cam2.json> ...] [--resolution 1024|1536] --seed 42 -o <mesh.ply>`; the camera JSON carries camera_angle_x, mesh_scale and a world_to_camera extrinsic (flat 12 floats or the 4x4 nested form; required for multiview, where a shared look-at would project every view from one direction; without it the single view uses the look-at at (0,0,distance)); MoGe-2 estimation is not integrated and a missing camera errors; the four stage noise draws are flyingfish's philox counter-stream convention (streams 1-4), not the publisher's torch RNG; `flyingfish::backend::PixalGeneration` and `PixalSeededRequest` | CUDA |
| Hi3D | DINOv2 and CLIP image conditioning, sparse flow and sparse VAE decoder at 512, 1024 or 1536, refining a supplied coarse mesh; raw mesh or a hole-filled, simplified and normalized mesh; same-seed reruns are bit-stable; CLI `ff --model <dir> to-3d <image.png> --dino <dir> --clip <dir> --mesh <coarse.ply> [--resolution 512|1024|1536] [--orientation 1] --seed 42 -o <mesh.ply>`; `flyingfish::backend::Hi3DGeneration` | CUDA |
| TRELLIS.2-4B | Image to a textured mesh on the default 1024-cascade pipeline: DINOv3 conditioning, sparse-structure flow with Euler interval-CFG sampling, cascade shape and texture structured-latent flows, FlexiDualGrid mesh and SparseUnet texture decoders, seeded stage noise; raw mesh only (no fill, simplify, UV, bake or render), RGBA input with the background already removed, and the decoder and structure-latent outputs differ from the reference at F16 quanta; `flyingfish::backend::TrellisTo3d`, CLI `ff --model <dir> to-3d <image.png> --dino <dir> --seed 42 -o <mesh.ply>` | CUDA |

In progress: TRELLIS.2 image-to-3D (flow sampling and the structured-latent, mesh and texture stages landed; remaining: render, GLB and bake, the shape and texture encoders, and closing the decoder and structure-latent differences from the reference).
