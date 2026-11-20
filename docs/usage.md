# CLI usage by model

`flyingfish` and `ff` take the same commands and options. Keep both binaries in the same directory. [models.md](models.md) lists supported models, weight formats and backends; commands below use the published model directories without converting their weights.

Replace `<model-dir>` and the other placeholders with your files and directories. Put `--model` before the command. EOU and Multitalker transcription take the actual `<model.nemo>` archive instead of a directory. Keep runtime data and output files outside model directories; output parent directories must exist, and media and text output files must not already exist.

```sh
ff --help
ff help generate
ff generate --help
ff --version
```

## Shared options and calibration

```sh
ff --model <model-dir> calibrate
```

`calibrate` measures kernels and transfer rates for a machine and model and writes them into the runtime directory. DeepSeek-V4.1-Flash uses fixed execution plans.

### Runtime directory

The default is `~/.flyingfish`, created on first use. Override it with `--runtime-dir` or `FLYINGFISH_RUNTIME_DIR`. It must be outside any model directory, and no path component is named `models`. It holds per-machine calibration and measured plans keyed by model config and build. H3-Image, MiniMax-H3, HyperFlow, speech, embedding and image-to-3D commands fill it on their first run with no separate step. MoE language models other than DeepSeek-V4.1-Flash need one `ff --model <model-dir> calibrate` first, otherwise generate stops and prints that command. One directory per machine, not copied between machines. Delete it or re-run `calibrate` to re-measure.

| Option | Default and accepted values | Use |
|---|---|---|
| `--model <MODEL>` | Required, or `FLYINGFISH_MODEL` | Published model directory, or EOU and Multitalker `.nemo` archive; before the command |
| `--runtime-dir <RUNTIME_DIR>` | `~/.flyingfish`; `--runtime-dir` or `FLYINGFISH_RUNTIME_DIR` | Per-machine calibration |
| `--device <DEVICE>` | `0` | GPU index |
| `--verbose` | Flag | Print placement, Lt and decode-plan diagnostics |
| `--capacity <CAPACITY>` | `4096` | Token workspace and KV capacity |
| `--chunk <CHUNK>` | Derived from VRAM budget and tile shapes | Prefill chunk override |
| `--device-budget-gib <DEVICE_BUDGET_GIB>` | Unset senses free device memory at setup | Declared device budget in GiB for expert-tier derivation |
| `--draft-model <DRAFT_MODEL>` | Unset, or `FLYINGFISH_DRAFT_MODEL` | DSpark draft directory |
| `--tree-builder <TREE_BUILDER>` | `waves`; `serial`, `waves` | Speculative tree construction |
| `--spec-budget <SPEC_BUDGET>` | `tree16`; `chain`, `tree16`, `tree32`, `tree64` | Speculative verification budget |
| `--spec-graph <SPEC_GRAPH>` | `true`; `true`, `false` | Speculative graph execution |
| `--prefix-cache <PREFIX_CACHE>` | `false`; `true`, `false` | Prefix cache |
| `--kv-format <KV_FORMAT>` | `bf16`; `bf16`, `affine8` | `affine8` stores an 8-bit KV cache for long contexts |
| `--dram-gb-s <DRAM_GB_S>` | `958` | DRAM reference denominator in GB per second; default is an RTX 4090 reference value |
| `--gemm-tflops <GEMM_TFLOPS>` | `166.06` | GEMM reference denominator in TFLOP per second; default is an RTX 4090 reference value |
| `-h`, `--help` | Flag | Print help |
| `-V`, `--version` | Root flag | Print version |

Shared options other than `--model` can also follow the command. The CLI exposes generation and speculation options globally; media and speech commands use their family-specific request paths.

## Language and multimodal models

### Text generation

Use the directory for the named model in each command:

```sh
ff --model <Qwen3.8-27B-dir> generate --prompt "Hello" --max-new-tokens 64
ff --model <MiniCPM-V-4.7-dir> generate --prompt "Hello" --max-new-tokens 64
ff --model <MiniCPM5-2B-dir> generate --prompt "Hello" --max-new-tokens 64
ff --model <Nanbeige4.3-33B-dir> generate --prompt "Hello" --max-new-tokens 64
ff --model <Flint-dir> generate --prompt "Hello" --max-new-tokens 64
ff --model <LoopVL-dir> generate --prompt "Hello" --max-new-tokens 64
ff --model <DeepSeek-V4.1-Flash-dir> generate --prompt "Hello" --max-new-tokens 64
ff --model <GLM-5.3-Flash-dir> generate --prompt "Hello" --max-new-tokens 64
ff --model <Gemma-4-dir> generate --prompt "Hello" --max-new-tokens 64
ff --model <clef-flash-dir> generate --prompt "Hello" --max-new-tokens 64
ff --model <MiniMax-H3-Image-PE-dir> generate --prompt "Hello" --max-new-tokens 64
```

Exactly one input is required: `--prompt <PROMPT>` for raw text, `--tokens <TOKENS>` for comma-separated token IDs, or `--tokens-file <TOKENS_FILE>` for JSON. A tokens file accepts an ID array, an `input_ids` array, or the first row of nested `input_ids`. Generation prints JSON with input IDs, generated IDs and decoded text.

| Generate option | Default and requirement |
|---|---|
| `--max-new-tokens <MAX_NEW_TOKENS>` | `32` |
| `--graph` | Enable graph decode |
| `--mtp <MTP>` | Depth `0-4`; defaults to `4` when the checkpoint contains MTP weights for an architecture with MTP decode; `0` selects plain generation |
| `--image-tensors <IMAGE_TENSORS>` | Processor-packed image input saved by the serve input trace; use matching token IDs |
| `--dump <DUMP>` | Operator trace output; cannot be combined with `--graph` |
| `--trace-layer <TRACE_LAYER>` | Zero-based decoder layer for operator trace; requires `--dump` |

Qwen3.8 and GLM-5.3-Flash have MTP decode. GLM-5.3-Flash drafts one token per round at any nonzero depth. MTP decode is CUDA-only, so ROCm builds need `--mtp 0` for such checkpoints.

DSpark is available for MiniCPM-V-4.7 and MiniCPM5-2B:

```sh
ff --model <MiniCPM-V-4.7-dir> --draft-model <DSpark-dir> generate --prompt "Hello" --max-new-tokens 64
ff --model <MiniCPM5-2B-dir> --draft-model <DSpark-dir> generate --prompt "Hello" --max-new-tokens 64
```

DSpark `--dump` requires `--spec-graph false`; DSpark does not support `--trace-layer`.

| Model | Requirements and limits |
|---|---|
| Qwen3.8-27B | BF16 and FP8; image input; MTP up to depth 4 on CUDA (ROCm runs it with `--mtp 0`) |
| MiniCPM-V-4.7 | BF16 and FP8; MoE expert tiering, DSpark draft, image input |
| MiniCPM5-2B | DSpark draft |
| Nanbeige4.3-33B | BF16 dense layers streamed through device slots when they do not fit |
| Flint | MoE expert tiering, sliding-window attention with sinks |
| LoopVL | Recurrent text decode, image input with prefix attention |
| DeepSeek-V4.1-Flash | Block FP8 projections, E2M1 experts, compressed latent attention, mHC; segmented CUDA-graph decode |
| GLM-5.3-Flash | FP8 routed experts tiered across GPU, pinned host and NVMe; DSA and KDA attention, mHC; segmented CUDA-graph decode; MTP with one draft token per round |
| Gemma 4 E2B, E4B, 12B, 31B | Text for all sizes; image and audio input for E2B, E4B and 12B; image input for 31B; 12B and 31B stream dense layers when they do not fit |
| clef-flash | Hybrid linear-attention and full-attention text tower with plain decode; the vision tower and MTP layers are not run; the joint schema head scoring path is gated, with no CLI or library entry |
| MiniMax-H3-Image-PE | ModelOpt NVFP4 projections executed as W4A16 (weights decoded to BF16, no FP4 tensor-core path); image input |

### DeepSeek-V4.1-Flash runtime setup

DeepSeek derives its Engram token map, hash multipliers and bucket primes at load from the `config.json` in the checkpoint's `inference` directory and the checkpoint's `tokenizer.json`. The inferred geometry must match `config.json`. Generation and serving require no external hash record or Python setup step; the runtime directory can start empty.

### Serving and multimodal input

```sh
ff --model <model-dir> serve --port 8080
ff --model <MiniCPM-V-4.7-dir> --draft-model <DSpark-dir> serve --port 8080
```

`serve` exposes `http://<host>:<port>/v1/completions` and `http://<host>:<port>/v1/chat/completions`. `--port <PORT>` defaults to `8080`. `--mtp <MTP>` has the same checkpoint-dependent default as generation; `0` selects plain serving.

For image-capable models, chat messages accept `image_url` content parts with a base64 PNG or JPEG data URL. Raw image filenames are not `generate --prompt` inputs; `generate --image-tensors` consumes a processor-packed trace. The model table's image and audio capabilities include library entry points; they do not imply a CLI flag for each modality.

### EmbeddingGemma-2

```sh
ff --model <EmbeddingGemma-2-dir> embed --text "a red cube" -o <embedding.json>
ff --model <EmbeddingGemma-2-dir> embed --input <text-file> -o <embedding.json>
```

Exactly one of `--text <TEXT>` and `--input <INPUT>` is required. File input is UTF-8; `--input -` reads stdin. Output is JSON with a 768-value `embedding` and `dimension`. `-o, --output <OUTPUT>` selects a `.json` file; omitted output writes JSON to stdout. Image and audio embeddings remain library-only.

## Diffusion language models

```sh
ff --model <Clover-35B-A3B-dir> generate --prompt "Hello" --max-new-tokens 64
ff --model <DiffusionGemma-26B-A4B-dir> generate --prompt "Hello" --max-new-tokens 64
```

Clover uses canvas inference with tiered experts; DiffusionGemma uses canvas inference. Both run on CUDA through the eager canvas path. Use the same input choices and JSON output as text generation. Omit `--graph`: canvas graph execution is not bound. Their canvas paths are separate from the autoregressive layer walker.

## Speech

All speech commands below use CUDA. Transcription and diarization input is mono 16 kHz WAV, PCM16, PCM24 or F32; `-` as the positional input reads WAV bytes from stdin. `speak` takes text. Output goes to stdout unless `-o, --output <OUTPUT>` names a file with the matching extension.

| Transcription model | `--model` input |
|---|---|
| Parakeet TDT, Canary-1B-v2, Nemotron 3.5 ASR | Published directory containing `<config.json>` and safetensors |
| Parakeet realtime EOU, Multitalker Parakeet | Actual `<model.nemo>` archive file; its containing directory is not accepted |

### Parakeet TDT

```sh
ff --model <Parakeet-TDT-dir> transcribe <audio.wav> --format srt -o <transcript.srt>
```

`--format <FORMAT>` defaults to `txt`; accepted values are `txt`, `json`, `srt`, `vtt`. Parakeet produces greedy TDT text and duration-derived subtitle cues. This path is offline; `--stream` is not supported for Parakeet TDT.

### Canary-1B-v2

```sh
ff --model <Canary-1B-v2-dir> transcribe <audio.wav> --source-lang en --target-lang en --pnc --format json -o <transcript.json>
```

`--source-lang <SOURCE_LANG>` and `--target-lang <TARGET_LANG>` both default to `en`. `--pnc` and `--no-pnc` are mutually exclusive; punctuation and capitalization are enabled unless `--no-pnc` is given. Canary uses published language and punctuation prompts; Canary subtitles are whole-clip only. `--stream` is not supported for Canary.

### Nemotron 3.5 ASR streaming 0.6B

```sh
ff --model <Nemotron-3.5-ASR-dir> transcribe <audio.wav> --format txt -o <transcript.txt>
ff --model <Nemotron-3.5-ASR-dir> transcribe <audio.wav> --stream --format json -o <transcript.json>
```

`--stream` selects cache-aware chunked-stream FastConformer encoding with shared greedy RNNT prediction and joint decode; omitted, it encodes the full clip. Both modes consume the supplied WAV. `--format <FORMAT>` defaults to `txt`; accepted values are `txt`, `json`, `srt`, `vtt`. Subtitle cues are whole-clip in both modes. The decoder uses blank 13087 and at most 10 symbols per step.

### Parakeet realtime EOU 120M

```sh
ff --model <model.nemo> transcribe <audio.wav> --format txt -o <transcript.txt>
ff --model <model.nemo> transcribe <audio.wav> --stream --format vtt -o <transcript.vtt>
```

Reads the `.nemo` container in place; offline greedy RNNT decode and cache-aware streaming with a reset on end-of-utterance. Offline subtitles are whole-clip only, streaming subtitles are per utterance. `--stream` selects the streaming path over the supplied clip.

### Nemotron-3-Diarization

```sh
ff --model <Nemotron-3-Diarization-dir> diarize <meeting.wav> -o <meeting.rttm>
ff --model <Nemotron-3-Diarization-dir> diarize <meeting.wav> --stream --threshold 0.5 -o <meeting.rttm>
```

`--threshold <THRESHOLD>` defaults to `0.5`; `--stream` selects streaming speaker diarization. RTTM output preserves overlaps and uses arrival-order speaker channels.

### Multitalker Parakeet streaming 0.6B

```sh
ff --model <model.nemo> transcribe <meeting.wav> --diar-model <Nemotron-3-Diarization-dir> --format json -o <speakers.json>
```

`--diar-model <DIAR_MODEL>` is required to select the multitalker path. The `.nemo` container is read in place. Nemotron-3-Diarization arrival-order channels 0-3 are the lanes; speaker kernels run by default. `--masked` selects masked-ASR lane masking and requires `--diar-model`. `--stream` selects streaming diarization for the masks. A fifth active diarization channel errors instead of being dropped.

Output contains per-speaker text with word timestamps; `txt`, `json`, `srt` and `vtt` are accepted, with speaker-prefixed word cues for subtitles. Kernel-mode free running diverges from the published wrapper's batch-1 pre-encode (measured 4.272e-4).

### Breeze-TTS-2

```sh
ff --model <Breeze-TTS-2-dir> speak "Hello, world." --seed 42 -o <speech.wav>
ff --model <Breeze-TTS-2-dir> speak - -o <speech.wav> < <text.txt>
ff --model <Breeze-TTS-2-dir> speak "Hello, world." > <speech.wav>
```

The positional `<TEXT>` is required; `-` reads UTF-8 text from stdin. `--seed <SEED>` defaults to `42`. `-o, --output <OUTPUT>` writes mono 24 kHz PCM16 WAV; omitted output writes WAV bytes to stdout. The command runs free-running text-to-speech with published-config sampling and seeded Philox draws, followed by audio tokenizer decode. Greedy decode over supplied audio-code frames remains available through the library.

## Media generation

These commands use CUDA. Default stderr contains progress and errors only, with timing fields hidden unless `--verbose` is enabled. `--verbose` prints placement, Lt and decode-plan diagnostics.

### MiniMax-H3 image-to-video with audio

```sh
ff --model <MiniMax-H3-dir> video --prompt "A boat crosses a quiet lake" --image <keyframe.png> --frames 136 --seed 42 --lora <turbo-lora.safetensors> -o <out.mp4>
ff --model <MiniMax-H3-dir> video --prompt "A boat crosses a quiet lake" --image <keyframe.png> --frames 136 --seed 42 --lora <turbo-lora.safetensors> --gguf <transformer.gguf> -o <out.y4m> --json
```

MiniMax-H3 video requires a keyframe, a frame count and a seed. Requested frames round up to `17n+5`, so 136 resolves to 141; the published duration is 5-15 seconds at 24 fps. No documented frame-count or seed defaults exist.

| Video option | Required input and behavior |
|---|---|
| `--prompt <PROMPT>` | Required video description |
| `--image <IMAGE>` | Keyframe image, required for MiniMax-H3 |
| `--frames <FRAMES>` | Required frame count; aligned as above |
| `--seed <SEED>` | Required noise seed |
| `--lora <LORA>` | Required Turbo LoRA safetensors, for example `<minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors>` |
| `--gguf <GGUF>` | Optional GGUF Q4_0 transformer instead of transformer safetensors; the model directory still supplies the other components |
| `-o, --output <OUTPUT>` | Required `.mp4`, or `.y4m` with a sibling `.wav` |
| `--json` | Print resolved output metadata, including aligned frame count, as JSON |

MP4 output requires ffmpeg on `PATH`. Y4M+WAV output does not require it. Without `--json`, stdout lists the written output paths.

### HyperFlow H3 Flash video with audio

```sh
ff --model <HyperFlow-H3-Flash-dir> video --base-model <MiniMax-H3-dir> --prompt "A boat crosses a quiet lake" --frames 124 --seed 42 -o <out.mp4>
ff --model <HyperFlow-H3-Flash-dir> video --base-model <MiniMax-H3-dir> --prompt "A boat crosses a quiet lake" --image <first.png> --last-image <last.png> --frames 124 --seed 42 -o <out.y4m> --json
```

HyperFlow uses the published `fl2va_int8_convrot` component and MiniMax-H3 companion files supplied by `--base-model`, without an external LoRA or GGUF replacement. It supports text-only, first, last and first+last keyframes; ref2va media inputs are not supported. Frames round up to `17n+5`, within 124-362 aligned frames at 24 fps. The sampler uses the published six-step grid with video and audio shifts 12 and 3, respectively.

| HyperFlow video option | Required input and behavior |
|---|---|
| `--prompt <PROMPT>` | Required video description |
| `--base-model <BASE_MODEL>` | Required MiniMax-H3 companion checkpoint for the text encoder and VAEs |
| `--frames <FRAMES>` | Required frame count; no default |
| `--seed <SEED>` | Required noise seed; no documented default |
| `--image <IMAGE>` | Optional first keyframe |
| `--last-image <LAST_IMAGE>` | Optional final keyframe, alone or with `--image` |
| `-o, --output <OUTPUT>` | Required `.mp4`, or `.y4m` with a sibling `.wav` |
| `--json` | Print resolved output metadata, including aligned frame count, as JSON |

Omit both keyframes for text-to-video; provide either one for a single anchor, or both for first+last conditioning. `--lora` and `--gguf` are invalid for HyperFlow. `--base-model` and `--last-image` are HyperFlow-only; MiniMax-H3 still requires its first keyframe and Turbo LoRA. MP4 requires ffmpeg on `PATH`; Y4M+WAV does not.

### MiniMax-H3-Image

```sh
ff --model <MiniMax-H3-Image-dir> image --prompt "A red cube on a wooden table" --size 1024x1024 --seed 42 -o <cube.png> --json
ff --model <MiniMax-H3-Image-dir> image --prompt "A red cube on a wooden table" --size 1024x1024 --seed 42 --expand <prompt-expander-dir> --edit <reference.png> --edit <second-reference.png> -o <edited.png>
```

`image` requires `--prompt <PROMPT>`, `--size <SIZE>` in `WIDTHxHEIGHT` form, `--seed <SEED>` and `-o, --output <OUTPUT>`. No seed default is documented. `--expand <EXPAND>` replaces the prompt through the expander; repeated `--edit <EDIT>` supplies upright RGB8 references. Output is PNG at the requested size, and `--json` reports size, seed and reference count. Without it, stdout prints the output path.

Prompt-to-image is usable. Native generation uses six-step Euler under a derived placement, native VAE decode, truncation toward zero and Lanczos-3; one Philox draw from the request seed.

### MiniMax-Music3

```sh
ff --model <MiniMax-Music3-dir> music --prompt "A warm acoustic pop song" --lyrics "[Verse] Morning light" --seed 7 --frames 25 -o <song.wav>
ff --model <MiniMax-Music3-dir> music --prompt "A warm acoustic pop song" --lyrics - --seed 7 -o <song.wav> < <lyrics.txt>
```

`music` requires `--prompt <PROMPT>` (caption), `--lyrics <LYRICS>` and `--seed <SEED>`. Lyrics accept optional section tags; `--lyrics -` reads stdin. `--frames <FRAMES>` defaults to `1500` (60 seconds at 25 fps). `-o, --output <OUTPUT>` writes stereo 44.1 kHz PCM16 WAV; omitted output writes WAV bytes to stdout. The command uses published caption and lyrics prompt construction with paired conditional and unconditional IDs, the autoregressive stage and synthesis stage.

## Image to 3D

`to-3d` uses CUDA and requires an input PNG with alpha; the background must already be removed (RMBG-2.0 integration is pending). Output is a raw `.ply` or `.obj` mesh selected with required `-o, --output <OUTPUT>`. GLB, texture bake and render are out of scope. PLY carries the material channels; OBJ writes geometry and RGB, without the extra per-vertex material channels.

### Pixal3D

```sh
ff --model <Pixal3D-dir> to-3d <image.png> --dino <DINOv3-dir> --naf <NAF-dir> --camera <camera.json> --resolution 1536 --seed 42 -o <mesh.ply>
ff --model <Pixal3D-MV-dir> to-3d <front.png> --view <back.png> --dino <DINOv3-dir> --naf <NAF-dir> --camera <front-camera.json> --camera <back-camera.json> --resolution 1024 --seed 42 -o <mesh.ply>
```

Pixal3D requires `--dino <DINO>`, `--naf <NAF>` and one `--camera <JSON>` per view. Repeated `--view <IMAGE>` adds RGBA views; cameras are paired in input order, starting with the positional image. `--resolution <RESOLUTION>` defaults to `0`, selecting the model default (Pixal3D `1536`); Pixal3D accepts `1024` or `1536`. `--seed <SEED>` defaults to `0`.

The camera JSON carries `camera_angle_x`, `mesh_scale` and a `world_to_camera` extrinsic (flat 12 floats or the 4x4 nested form; required for multiview, where a shared look-at would project every view from one direction; without it the single view uses the look-at at `(0,0,distance)`). For example:

```json
{"camera_angle_x": 0.7, "mesh_scale": 1.0, "distance": 3.0, "world_to_camera": [1,0,0,0,0,1,0,0,0,0,1,3]}
```

MoGe-2 estimation is not integrated and a missing camera errors. The four stage noise draws are flyingfish's philox counter-stream convention (streams 1-4), not the publisher's torch RNG. The result is a raw mesh with six material channels.

### TRELLIS.2-4B

```sh
ff --model <TRELLIS.2-4B-dir> to-3d <image.png> --dino <DINOv3-dir> --seed 42 -o <mesh.ply>
```

TRELLIS.2 requires `--dino <DINO>`. `--seed <SEED>` defaults to `0`. It uses the default 1024-cascade pipeline; `--view`, `--camera`, `--naf` and a nonzero `--resolution` are Pixal3D options and are rejected for TRELLIS.2.

Raw mesh only (no fill, simplify, UV, bake or render), RGBA input with the background already removed, and the decoder and structure-latent outputs differ from the reference at F16 quanta. Remaining work includes render, GLB and bake, the shape and texture encoders, and closing those differences.

### Hi3D

```sh
ff --model <Hi3D-dir> to-3d <image.png> --dino <DINOv2-dir> --clip <CLIP-dir> --mesh <coarse.ply> --resolution 512 --seed 42 -o <mesh.ply>
```

Hi3D requires `--dino <DINOv2>`, `--clip <CLIP>` and `--mesh <PLY>`, a coarse mesh to refine. `--orientation <N>` defaults to `1`; `--resolution <RESOLUTION>` defaults to `0`, selecting the model default (Hi3D `512`); Hi3D accepts `512`, `1024` or `1536`. `--seed <SEED>` defaults to `0`. `--view`, `--camera` and `--naf` are Pixal3D options and are rejected for Hi3D.

The result is a hole-filled, simplified and normalized mesh, and same-seed reruns are bit-stable.

### Library-only image-to-3D entries

Idol uses `flyingfish::backend::IdolPipelineRuntime`: a preprocessed 1024 px image to a Gaussian splat and a camera. It is not selected by the current `to-3d` command.

## Inspection, verification and profiling commands

These commands share the model and runtime options above. In the command lines below, prepend `ff --model <model-dir>`. Input-based commands accept exactly one of `--prompt`, `--tokens` and `--tokens-file`, as in generation. Fixture, plan and limits files must match the model and the corresponding acceptance gate.

| Command and required arguments | Optional arguments and defaults | Input and output |
|---|---|---|
| `inspect` | Shared options | Reads model config; prints the parsed config |
| `tokenize --prompt "Hello"` | Shared options | Raw text to a JSON token-ID array |
| `bench --prompt "Hello"` | `--steps 256`, `--warmup 32`, `--runs 3`, `--graph` | Generation benchmark |
| `profile --prompt "Hello"` | `--phase decode` (`prefill`, `decode`), `--warmup 32`, `--graph` | Stage profiling |
| `verify --fixture <fixture> --hidden-limits <limits.json> --logit-limits <limits.json>` | `--graph` | Model-definition fixture and thresholds; verification report |
| `compare-prefill --fixture <fixture> --dump <dump> --hidden-limits <limits.json> --logit-limits <limits.json>` | Shared options | Saved trace comparison |
| `compare-log --fixture <fixture> --log <log> --hidden-limits <limits.json> --logit-limits <limits.json>` | Shared options | Saved log comparison |
| `profile-spec --prompt "Hello"` | Shared options | DSpark profiling |
| `bench-spec --prompt "Hello"` | `--steps 256`, `--runs 3` | DSpark benchmark |
| `verify-spec --prompt "Hello" --strict` | `--steps 64`, `--reference <reference>`; replace `--strict` with `--logit-limits <limits.json> --prompt-name <name>` for bounded comparison | DSpark greedy comparison |
| `replay-draft --plan <plan> --thresholds <thresholds>` | Shared options | Supplied draft replay plan and thresholds |
| `compare-spec-graph --prompt "Hello"` | `--steps 64` | Eager and graph DSpark comparison |
| `compare-tree-padding --prompt "Hello"` | `--nodes 9` | Tree padding comparison |
| `compare-tree-builders --prompt "Hello"` | `--steps 256` | Tree builder comparison |
| `replay-tree --plan <plan> --dump <dump> --commit-policy fixture --reset-each-round` | Shared options | Replay fixture commits and write a dump |
| `check-tree-greedy --prompt "Hello" --report <report> --strict --ignore-eos` | `--steps 256` | Write strict greedy tree verification report |

All DSpark profiling, replay and tree commands in this table require `--draft-model <draft-dir>`. `verify-spec --strict` conflicts with `--logit-limits`; `--prompt-name` and `--logit-limits` require each other. `replay-tree` requires the `fixture` commit policy and `--reset-each-round`; `check-tree-greedy` requires both `--strict` and `--ignore-eos`.

`help` prints the command list; `help <command>` and `<command> --help` print a command's options.
