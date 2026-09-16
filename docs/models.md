# Model guide

## Performance

Measurements used Linux, Intel i9-13900KF, 62 GiB RAM with 8 GiB swap, local NVMe storage, and one RTX 4090 with 24 GiB VRAM. The binary was built with Rust 1.95.0, CUDA 13.2 and `cargo build --locked --release --features flash-attn`.

Each row is one run; cases ran sequentially with OS caches retained. Wall time includes loading, initialization, inference and output writing. Peak RSS measures resident process memory and excludes the OS page cache. GPU usage is sampled across the device once per second, so brief peaks may be missed. Timings depend on inputs, cache state and hardware.

| Model | Workload and output | Wall time | Peak RSS | Sampled GPU peak |
|---|---|---:|---:|---:|
| [GLM-5.3-Flash](#glm-53-flash) | Generated 32 tokens from a 20-token prompt | 74.98 s | 1.59 GiB | 21.35 GiB |
| [MiniMax-H3](#minimax-h3) | Generated 107 frames in 49 evaluations for a 768p, 16:9, 4-second request | 1,853.57 s | 58.82 GiB | 20.92 GiB |
| [MiniCPM5-2B](#minicpm5-2b-and-dspark) | Generated 31 tokens with a 32-token limit | 5.34 s | 1.10 GiB | 4.62 GiB |
| [MiniCPM5-2B + DSpark](#minicpm5-2b-and-dspark) | Generated 31 tokens with a 32-token limit | 2.24 s | 1.10 GiB | 5.17 GiB |
| [CLIP ViT-L/14](#clip) | Scored two candidate texts against a 224×224 image | 2.44 s | 0.44 GiB | 0.99 GiB |
| [TRELLIS-text-base](#trellis-1) | Generated 180,288 Gaussian splats from a text prompt | 8.96 s | 0.70 GiB | 3.98 GiB |
| [TRELLIS-text-large](#trellis-1) | Generated 231,744 Gaussian splats from a text prompt | 21.95 s | 0.69 GiB | 5.41 GiB |
| [TRELLIS-text-xlarge](#trellis-1) | Generated 198,432 Gaussian splats from a text prompt | 27.69 s | 0.72 GiB | 6.44 GiB |
| [TRELLIS-image-large + DINOv2](#trellis-1) | Generated 762,112 Gaussian splats from a 518×518 image | 75.04 s | 0.71 GiB | 4.64 GiB |
| [TRELLIS.2-4B + DINOv3](#trellis2) | Generated a mesh with 254,641 vertices from a 512×512 image | 33.33 s | 1.09 GiB | 3.36 GiB |
| [MiniMax-Music3](#minimax-music3) | Generated 8 s of audio in 30 denoise steps | 130.63 s | 2.11 GiB | 19.22 GiB |

Image tests used a fixed procedural chair pattern at each required resolution. DINOv2/v3 are included in the corresponding TRELLIS pipeline timings. RMBG-2.0/BiRefNet currently has no inference adapter.

## Usage

From the repository root, install the CUDA CLI with `cargo install --locked --path . --features flash-attn`. The examples assume `ff` is on `PATH`.

Replace checkpoint and image placeholders with your own paths. Use a fresh output path for each timed run. MiniCPM, Music3 and TRELLIS use automatic device residency; H3 and GLM use the cache settings shown below. Size memory budgets for your machine.

```bash
mkdir -p output
```

### GLM-5.3-Flash

The measured run achieved **1.065 tokens/s**: 32 tokens in 30.042 s of decode, with 37.575 s of prefill. This uses pinned FP8 transfers, a 4 GiB shared LFU expert cache and a host/device split from a local hardware profile.

Create the profile on the machine running inference; calibration is separate from the generation timing:

```bash
ff bench io --profile local-interconnect \
    --model "<glm-checkpoint>" \
    --device cuda:0 \
    --output output/glm-host-profile.json

FF_GLM_DIRECT_FILL=0 FF_GLM_HOST_THREADS=6 FF_GLM_FILL_AHEAD=1 \
ff text generate \
    --model "<glm-checkpoint>" \
    --prompt 'Explain paging to a systems programmer.' \
    --device cuda:0 --max-new-tokens 32 --temperature 0 --seed 42 \
    --weight-source mmap \
    --expert-cache-layout shared-pool --expert-cache-replacement lfu \
    --expert-cache-mib 4096 \
    --host-profile output/glm-host-profile.json --pinned-fp8-transfer \
    --json
```

These settings use buffered reads, six host expert workers and one fill-ahead worker. The current GLM text profile handles one request at a time with at most 2,048 prompt-plus-generated tokens and requires `--weight-source mmap`. Use `ff text generate-multi --help` for CUDA layer partitioning across devices.

### MiniMax-H3

The official example target is **768, 16:9, 4 s with 49 model evaluations**. The measured execution uses FlashAttention, larger compute chunks and a host cache that retains the transformer shards. Set `--max-host-mib` to a memory budget your host can sustain; the values here describe the measured machine.

```bash
ff video generate \
    --model "<h3-checkpoint>" \
    --prompt 'A fishing boat crossing a calm lake at sunrise.' \
    --device cuda:0 \
    --short-edge 768 --aspect-ratio 16:9 --duration-seconds 4 \
    --sigma-points 50 --seed 42 --flash-attention \
    --attention-projection-chunk-size 4096 \
    --ffn-token-chunk-size 1024 --output-token-chunk-size 1024 \
    --weight-source memory --host-cache-mib 64700 --max-host-mib 66000 \
    --output-dir output/h3-example
```

The target resolves to **1344×768, 107 frames (4.458 s)**. Total time was **30m 53.57s**, with a steady denoise rate of **24.2 s/evaluation**. Text encoding took 49.06 s, static-context preparation including weight loading took 434.99 s, and overlapping video/audio decode took 152.96 s. Total time includes these stages.

A successful run writes PNGs under `frames/`, `generated.wav` and recovery checkpoints. Reusing the output directory resumes an interrupted run. Use `ff video --help` for first/last-frame and reference-conditioning workflows.

### MiniCPM5-2B and DSpark

Generate with MiniCPM, or add the DSpark draft for speculative decoding. Both measured runs produced the same 31-token greedy text under a 32-token limit.

```bash
ff text generate \
    --model "<minicpm-checkpoint>" \
    --prompt 'Explain paging to a systems programmer.' \
    --device cuda:0 --max-new-tokens 32

ff text generate \
    --model "<minicpm-checkpoint>" \
    --draft-model "<dspark-checkpoint>" \
    --prompt 'Explain paging to a systems programmer.' \
    --device cuda:0 --max-new-tokens 32
```

### TRELLIS-1

Use a base, large or xlarge text checkpoint with the CLIP ViT-L/14 conditioner. The measured runs used checkpoint-default sampling steps and seed 0.

```bash
ff 3d generate \
    --model "<trellis-text-checkpoint>" \
    --conditioner "<clip-vit-large-patch14-checkpoint>" \
    --models-root "<models-root>" \
    --prompt 'a brightly colored wooden chair' \
    --device cuda:0 --seed 0 \
    --output output/trellis-text.ply
```

The image-large checkpoint uses DINOv2 with registers and a prepared 518×518 RGB PNG:

```bash
ff 3d generate \
    --model "<trellis-image-large-checkpoint>" \
    --conditioner "<dinov2-with-registers-large-checkpoint>" \
    --models-root "<models-root>" \
    --image "<prepared-518x518.png>" \
    --device cuda:0 --seed 0 \
    --output output/trellis-image.ply
```

### TRELLIS.2

Generate a colored mesh with TRELLIS.2-4B, the DINOv3 ViT-L/16 conditioner and a prepared 512×512 RGB PNG. The measured run used checkpoint-default sampling steps and seed 0.

```bash
ff 3d generate \
    --model "<trellis2-checkpoint>" \
    --conditioner "<dinov3-vitl16-checkpoint>" \
    --models-root "<models-root>" \
    --image "<prepared-512x512.png>" --resolution 512 \
    --device cuda:0 --seed 0 \
    --output output/trellis2.ply
```

Use `ff 3d --help` for generation and decoding operations.

### MiniMax-Music3

Supply both a style prompt and lyrics. This example produces approximately 8 s of 44.1 kHz stereo audio with 30 denoise steps:

```bash
ff music generate \
    --model "<music3-checkpoint>" \
    --prompt 'gentle solo piano' \
    --lyrics 'soft light on the water, a quiet afternoon' \
    --duration 8 --steps 30 --seed 0 \
    --device cuda:0 \
    --output output/music3.wav
```

### CLIP

Score candidate texts against an RGB PNG prepared at the checkpoint's input size, 224×224 for ViT-L/14:

```bash
ff similarity score \
    --model "<clip-vit-large-patch14-checkpoint>" \
    --image "<prepared-224x224.png>" \
    --text 'a wooden chair' --text 'a fishing boat' \
    --device cuda:0
```

## Adding model adapters

The public CLI is organized by task: `text`, `video`, `music`, `3d` and `similarity`. A checkpoint's architecture metadata selects an adapter within that task. Checkpoint directory names do not select implementations.

Each registration in `src/cli/adapters` supplies four things:

| Field | Responsibility |
|---|---|
| `id` and `task` | Name the adapter and the task it implements |
| `recognizes` | Recognize supported architecture metadata without reading weights |
| `command` | Build the adapter's operations, typed options and defaults with Clap |
| `run` | Parse the selected operation and call its implementation |

To add another model for an existing task:

1. Implement its inference engine in a model crate, using `ff-core` for shared weight loading, caching and telemetry where applicable.
2. Add an adapter module with its metadata recognizer, command schema and execution handler. Reuse shared argument groups for common cache controls.
3. Register its descriptor in `BUILTINS` in `src/cli/adapters/mod.rs`.
4. Add checkpoint recognition and component inventory support in `src/models.rs` if the model should also be available to `ff models list` and `inspect`.
5. Test metadata routing, invalid options, and execution with a small fixture.

The top-level command enum and task router do not need another model branch. The registry test `adding_a_registered_model_requires_no_task_router_changes` demonstrates registration and dispatch of another text model.

An adapter can recognize multiple compatible architectures. The TRELLIS adapter recognizes both TRELLIS-1 and TRELLIS.2 under `ff 3d generate`; their execution code validates the different conditioning and resolution requirements.

General help shows options shared by models that provide the same operation. Once a model is identified, its complete schema handles parsing, including defaults, conflicts and required arguments. Flags belonging to another adapter are rejected. Use `--adapter <name> --help` to inspect a particular schema before downloading weights. An explicit adapter also resolves overlapping recognizers; it must still support the supplied checkpoint metadata.

Parameter validation and memory admission are part of normal execution. Keep those checks before weight materialization and output publication. They should not become an alternate execution mode.
