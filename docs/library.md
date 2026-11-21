# Library

The `flyingfish` crate exposes the same model entries the `ff` command uses. Library callers use two entry layers:

| Layer | Public surface | Backends | Resource contract |
|---|---|---|---|
| Model request | `RunOptions` with `minimax_h3::run_minimax_h3_i2va` or `backend::IdolPipelineRuntime::open`; family types such as `minimax_h3::ImageRunOptions` and `music3::Music3Request` | CUDA for video and music; CUDA and ROCm for `MiniMax-H3-Image`; other families follow [docs/models.md](models.md) | The caller supplies request inputs, device ordinal and runtime directory through the entry's arguments. The entry manages its stages. The Idol runtime can serve successive requests on the same device. |
| Stage | `backend::BackendDevice` and the encoder, flow, decoder and workspace types exported by `backend` | `BackendDevice` is exported with CUDA or ROCm; individual stages retain their model and backend restrictions | The caller constructs the device and stage objects, supplies stage inputs, and owns the runtime and workspace lifetimes. The device carries GPU identity, context, execution stream and upload stream. The speech, Breeze-TTS-2 and MiniMax-Music3 stages can be constructed this way; the vision and image-to-3D stages need a `Blas`, which only the crate constructs. |

[docs/models.md](models.md) lists the backends each model runs on. [docs/usage.md](usage.md) documents the CLI for the same models.

## Dependency and features

The crate is `flyingfish`, edition 2024, built with the toolchain pinned in `rust-toolchain.toml`. Add it as a path or git dependency:

```toml
[dependencies]
flyingfish = { path = "<flyingfish-checkout>" }
anyhow = "1"
```

| Feature | Selection | Effect |
|---|---|---|
| `cuda` | Default | CUDA backend through `cudarc`; the build compiles the device crate and native attention cubins (see the README's Build section for the toolchains, `CUDA_ARCH_LIST` and `nvcc`) |
| `rocm` | `default-features = false, features = ["rocm"]` | HIP backend; needs `hipcc` and the hipBLAS and hipBLASLt libraries; `ROCM_ARCH_LIST` selects kernel targets |
| none | `default-features = false` | Host-side code only; `Model::load` returns `no inference backend available: CUDA backend is missing; rebuild with --features cuda` |

`cuda` and `rocm` are mutually exclusive; enabling both is a compile error (``features `cuda` and `rocm` are mutually exclusive``).

Items by feature gate:

| Gate | Items |
|---|---|
| Always compiled | `RunOptions`; `model::{Model, GenerationOutput, generate_plain, generate_multimodal, embed_text, embedding_json, check_generation_capacity}`; `serve::{run, run_mtp}`; `spec::{Options, generate}`; `tokenizer::{encode, decode}`; `media` readers and writers; `sparse::SparseMesh`; the request and spec types of `minimax_h3` and `music3` (`MiniMaxH3I2vaRequest`, `HyperFlowH3Request`, `H3Output`, `ImageRunOptions`, `ImageRequest`, `ImageReference`, `ImageLatents`, `Music3Request`, `Music3Output`, `music3::prompt_ids`, `music3::SynSpec`) |
| `cuda` or `rocm` | `backend::{BackendDevice, Backend, with_verbose}`, `backend::gpu`; `minimax_h3::{run_minimax_h3_i2va, run_hyperflow_h3_video, generate, image_latents, image_encode, image_decode, image_sample, ImageSession}`; `backend::{DinoV3Encoder, DinoV3Output, DinoV3Workspace, Flux2VaeConfig, Flux2VaeEncoder, Flux2VaeOutput, Flux2VaeWorkspace, VisionEncoderConfig}`; the DSA and mHC types in `backend` |
| `cuda` | `minimax_h3::VideoSession`; `parakeet::{transcribe, tdt_transcription, ParakeetTdt, ParakeetEncoder, NemotronStream, StreamChunk, TdtTrajectory}`; `canary::{transcribe, Canary}`; `nemotron::transcribe`; `eou::{transcribe, Eou, EouUtterance}`; `diarization::{diarize, speaker_masks, SpeakerMasks, Diarization}`; `multitalker::{transcribe, Multitalker, MultitalkerTranscription, SpeakerTranscription}`; the `breeze` module; `music3::{generate, Runtime}`; `backend::{Hi3DGeneration, PixalGeneration, PixalSeededRequest, PixalView, PixalOutput, ProjectionCamera, TrellisTo3d, TrellisMesh, IdolPipelineRuntime}` and the other Idol, Pixal3D, Hi3D and TRELLIS.2 stage types; `backend::{Breeze, BreezeTokenizer, Music3Syn}` |
| `rocm` | `backend::hip` |

## Run options and the runtime directory

`flyingfish::RunOptions`:

```rust
#[derive(Clone, Debug)]
pub struct RunOptions {
    pub device: usize,
    pub runtime_dir: std::path::PathBuf,
    pub verbose: bool,
}
```

- `device`: GPU ordinal.
- `runtime_dir`: per-machine directory for calibration and measured plans, keyed by model config and build.
- `verbose`: diagnostics switch; `run_minimax_h3_i2va`, `run_hyperflow_h3_video` and `IdolPipelineRuntime::open` apply it.

Fields each `RunOptions` entry reads:

| Entry | `device` | `runtime_dir` | `verbose` |
|---|---|---|---|
| `parakeet::transcribe`, `canary::transcribe`, `nemotron::transcribe`, `eou::transcribe`, `diarization::diarize`, `diarization::speaker_masks`, `multitalker::transcribe` | Yes | No | No |
| `breeze::speak`, `model::embed_text` | Yes | Yes | No |
| `minimax_h3::run_minimax_h3_i2va`, `minimax_h3::run_hyperflow_h3_video`, `backend::IdolPipelineRuntime::open` | Yes | Yes | Yes |

Diagnostics are on by default on each thread. `backend::with_verbose(verbose: bool, run: impl FnOnce() -> T) -> T` sets them for the duration of `run` and restores the previous value; the CLI wraps every media, speech and embedding command in it.

The library has no default runtime directory. The CLI resolves `--runtime-dir`, then `FLYINGFISH_RUNTIME_DIR`, then `$HOME/.flyingfish`, and errors with `HOME is unset or empty` when none applies; that resolution lives in the `ff` binary. A library caller passes the directory explicitly: in `RunOptions::runtime_dir`, `ImageRunOptions::runtime_dir`, the `runtime_dir` argument of `music3::generate`, the `runtime` field of the image-to-3D requests, or `Settings::runtime_dir` for language models. `Settings::runtime_dir` is an `Option`; `None` fails language-model setup with `CUDA setup requires --runtime-dir or FLYINGFISH_RUNTIME_DIR`. The directory must lie outside the model directory and no path component may be named `models` (`runtime directory must be outside all readonly models directories`). It is created on first use. Use one directory per machine.

Calibration:

- H3-Image, MiniMax-H3, HyperFlow, speech, embedding and image-to-3D entries fill the runtime directory on their first run with no separate step.
- MoE language models other than DeepSeek-V4.1-Flash need one calibration per machine and model before `Model::load` succeeds. Without it, load fails and the error quotes the equivalent `calibrate` command line. From Rust, run `Model::calibrate` once with the same directory, device, capacity, chunk and settings as the later load.
- Calibration measures kernels and transfer rates (for MoE models, expert load service and host-device copy rates) and writes them into the runtime directory. Delete the directory or calibrate again to re-measure.

```rust
pub fn calibrate(
    dir: &Path,
    ordinal: usize,
    capacity: usize,
    chunk: Option<usize>,
    settings: Settings,
) -> Result<Option<Self>>
```

`Model::calibrate` returns `Some(model)` only for Flint, `None` for every other architecture.

## Speech

All speech entries are CUDA-only and take a mono waveform as `&[f32]` at 16 kHz. `media::read_wav(reader: impl Read, expected_rate: u32) -> Result<Vec<f32>>` reads mono PCM16, PCM24 or F32 WAV and errors when the rate differs from `expected_rate`.

Transcription entries return `media::Transcription`:

```rust
pub struct Transcription {
    pub text: String,
    pub segments: Vec<TextSegment>,
    pub words: Vec<TextSegment>,
}
pub struct TextSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}
```

`start` and `end` are seconds. `media::write_srt`, `media::write_vtt` (`(writer: impl Write, segments: &[TextSegment]) -> Result<()>`) and `media::write_rttm(writer: impl Write, file_id: &str, segments: &[TextSegment]) -> Result<()>` write subtitle and RTTM files from segments.

### Parakeet TDT

```rust
pub fn transcribe(
    directory: &Path,
    waveform: &[f32],
    options: crate::RunOptions,
) -> Result<crate::media::Transcription>
```

`parakeet::transcribe` runs the log-mel frontend, the FastConformer encoder and greedy TDT decode, and returns text with duration-derived segments and words. `parakeet::ParakeetTdt::load(directory: &Path, device: &Device) -> Result<Self>` and `ParakeetTdt::decode(&mut self, input: &[f32], valid: usize) -> Result<TdtTrajectory>` are the stage form; `parakeet::tdt_transcription(directory: &Path, trajectory: &TdtTrajectory, blank: u32) -> Result<crate::media::Transcription>` turns a trajectory into a transcription.

```rust
use flyingfish::{RunOptions, media, parakeet};
use std::{fs::File, path::{Path, PathBuf}};

fn main() -> anyhow::Result<()> {
    let options = RunOptions { device: 0, runtime_dir: PathBuf::from("<runtime-dir>"), verbose: false };
    let waveform = media::read_wav(File::open("<audio.wav>")?, 16000)?;
    let result = parakeet::transcribe(Path::new("<Parakeet-TDT-dir>"), &waveform, options)?;
    println!("{}", result.text);
    media::write_srt(File::create("<out.srt>")?, &result.segments)?;
    Ok(())
}
```

### Canary-1B-v2

```rust
pub fn transcribe(
    directory: &Path,
    waveform: &[f32],
    source_language: &str,
    target_language: &str,
    punctuation: bool,
    options: crate::RunOptions,
) -> Result<crate::media::Transcription>
```

`canary::transcribe` builds the published prompt from the source language, target language and punctuation flag (`canary::canary_prompt` with the same arguments returns the prompt ids), runs greedy AED decode and returns whole-clip text: one segment spanning the clip and no words. The stage form is `canary::Canary::load(directory: &Path, device: &Device) -> Result<Self>`.

```rust
let result = flyingfish::canary::transcribe(Path::new("<Canary-1B-v2-dir>"), &waveform, "en", "en", true, options)?;
```

### Nemotron 3.5 ASR streaming 0.6B

```rust
pub fn transcribe(
    directory: &Path,
    waveform: &[f32],
    streaming: bool,
    options: crate::RunOptions,
) -> Result<crate::media::Transcription>
```

`nemotron::transcribe` encodes the full clip (`streaming = false`) or as a cache-aware chunked stream (`streaming = true`), runs greedy RNNT decode and returns whole-clip text.

### Parakeet realtime EOU 120M

```rust
pub fn transcribe(
    nemo: &Path,
    waveform: &[f32],
    stream: bool,
    options: crate::RunOptions,
) -> Result<crate::media::Transcription>
```

`eou::transcribe` reads the `.nemo` archive in place. With `stream = false` it runs offline greedy RNNT decode and returns whole-clip text. With `stream = true` it runs cache-aware streaming with a reset at each end of utterance and returns one segment per utterance; `text` joins the utterances with spaces. The stage form is `eou::Eou::load(nemo: &Path, device: &Device) -> Result<Self>`; `eou::eou_decode(nemo: &Path, ids: &[u32]) -> Result<String>` decodes token ids with the archive's tokenizer.

### Nemotron-3-Diarization

```rust
pub fn diarize(
    directory: &Path,
    waveform: &[f32],
    stream: bool,
    threshold: f64,
    options: crate::RunOptions,
) -> Result<Vec<crate::media::TextSegment>>
```

`diarization::diarize` runs offline (`stream = false`) or streaming (`stream = true`, the published `low_latency` mode) diarization with an arrival-order speaker cache. Each returned segment is one speaker turn: `text` is `speaker_<n>`, times are seconds rounded to 0.01, overlaps are kept, and segments are sorted by start then speaker. `threshold` is the sigmoid activity threshold in `[0, 1]`; the CLI default is `diarization::DIARIZATION_THRESHOLD` (`0.5`).

```rust
pub fn speaker_masks(
    directory: &Path,
    waveform: &[f32],
    stream: bool,
    threshold: f64,
    options: crate::RunOptions,
) -> Result<SpeakerMasks>
```

`diarization::speaker_masks` returns the binarized activity on the encoder frame grid that Multitalker consumes:

```rust
pub struct SpeakerMasks {
    pub frames: usize,
    pub speakers: usize,
    pub frame_seconds: f64,
    pub active: Vec<bool>,
    pub pooled: Vec<f32>,
}
```

`active` and `pooled` are `[frame][speaker]`; `pooled` holds the pooled speaker probabilities that `active` binarizes.

```rust
let segments = flyingfish::diarization::diarize(Path::new("<Nemotron-3-Diarization-dir>"), &waveform, false, flyingfish::diarization::DIARIZATION_THRESHOLD, options)?;
flyingfish::media::write_rttm(File::create("<out.rttm>")?, "<file-id>", &segments)?;
```

### Multitalker Parakeet streaming 0.6B

```rust
pub fn transcribe(
    nemo: &Path,
    waveform: &[f32],
    masks: &crate::diarization::SpeakerMasks,
    masked_asr: bool,
    options: crate::RunOptions,
) -> Result<MultitalkerTranscription>
```

`multitalker::transcribe` reads the `.nemo` archive in place and runs 1-4 speaker lanes selected by `masks` (arrival-order channels 0-3). `masked_asr = false` uses the speaker kernels; `true` selects lane masking. A diarization channel at index 4 or above that is ever active is an error. The mask frame period must equal the ASR frame period.

```rust
pub struct MultitalkerTranscription {
    pub speakers: Vec<SpeakerTranscription>,
}
pub struct SpeakerTranscription {
    pub speaker: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub words: Vec<crate::media::TextSegment>,
    pub tokens: Vec<u32>,
    pub timestamps: Vec<u32>,
}
```

`words` carry per-word times in seconds on the 80 ms frame grid. Both types implement `serde::Serialize`; `tokens` and `timestamps` are skipped in serialized output.

```rust
let masks = flyingfish::diarization::speaker_masks(Path::new("<Nemotron-3-Diarization-dir>"), &waveform, true, flyingfish::diarization::DIARIZATION_THRESHOLD, options.clone())?;
let result = flyingfish::multitalker::transcribe(Path::new("<multitalker.nemo>"), &waveform, &masks, false, options)?;
for speaker in &result.speakers {
    println!("speaker_{} [{:.2}-{:.2}] {}", speaker.speaker, speaker.start, speaker.end, speaker.text);
}
```

### Breeze-TTS-2

```rust
pub fn speak(
    directory: &Path,
    text: &str,
    seed: u64,
    options: crate::RunOptions,
) -> Result<BreezeSpeech>
```

`breeze::speak` runs free-running text-to-speech with the published `generation_config.json` sampling and seeded Philox draws, then decodes the audio codes with the audio tokenizer. The `breeze` module is compiled only with `cuda`.

```rust
#[derive(Clone, Debug)]
pub struct BreezeSpeech {
    pub samples: Vec<f32>,
    pub sampling_rate: usize,
}
```

`samples` is mono; `sampling_rate` is 24000.

```rust
let speech = flyingfish::breeze::speak(Path::new("<Breeze-TTS-2-dir>"), "Hello", 7, options)?;
flyingfish::media::write_wav(File::create("<out.wav>")?, speech.sampling_rate, 1, &speech.samples)?;
```

## Media generation

### MiniMax-H3 image-to-video with audio

```rust
pub fn run_minimax_h3_i2va(
    request: &MiniMaxH3I2vaRequest,
    options: crate::RunOptions,
) -> Result<H3Output>
```

```rust
#[derive(Clone, Debug)]
pub struct MiniMaxH3I2vaRequest {
    pub model: PathBuf,
    pub lora: PathBuf,
    pub keyframe: PathBuf,
    pub prompt: String,
    pub frames: usize,
    pub seed: u64,
    pub gguf: Option<PathBuf>,
}
```

- `model`: MiniMax-H3 model directory.
- `lora`: Turbo LoRA safetensors file.
- `keyframe`: PNG or JPEG keyframe file.
- `prompt`: video description.
- `frames`: requested frame count; it rounds up to `17n+5` and the aligned count must be 5-15 seconds at `minimax_h3::FPS` (24).
- `seed`: seed of the four native noise draws.
- `gguf`: optional GGUF Q4_0 transformer file that replaces the transformer safetensors; the model directory supplies the other components.

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct H3Output {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub video: Vec<u8>,
    pub samples: usize,
    pub sampling_rate: usize,
    pub audio: Vec<f32>,
}
```

`video` is RGB8 `[frames, height, width, 3]`. `audio` is stereo F32 `[2, samples]`, channel-major, at `sampling_rate`. `frames` is the aligned frame count.

```rust
use flyingfish::{RunOptions, media, minimax_h3};
use std::{fs::File, path::PathBuf};

fn main() -> anyhow::Result<()> {
    let options = RunOptions { device: 0, runtime_dir: PathBuf::from("<runtime-dir>"), verbose: false };
    let request = minimax_h3::MiniMaxH3I2vaRequest {
        model: PathBuf::from("<MiniMax-H3-dir>"),
        lora: PathBuf::from("<turbo-lora.safetensors>"),
        keyframe: PathBuf::from("<keyframe.png>"),
        prompt: "A boat crosses a quiet lake".into(),
        frames: 136,
        seed: 42,
        gguf: None,
    };
    let out = minimax_h3::run_minimax_h3_i2va(&request, options)?;
    let stereo: Vec<f32> = (0..out.samples).flat_map(|i| [out.audio[i], out.audio[out.samples + i]]).collect();
    media::write_y4m(File::create("<out.y4m>")?, out.width, out.height, (minimax_h3::FPS, 1), out.frames, &out.video)?;
    media::write_wav(File::create("<out.wav>")?, out.sampling_rate, 2, &stereo)?;
    Ok(())
}
```

`media::mux_mp4(output: &Path, video: Y4mVideo, audio: Option<(usize, usize, &[f32])>) -> Result<()>` writes MP4 through `ffmpeg` on `PATH`; `audio` is `(sampling_rate, channels, interleaved samples)`. `media::Y4mVideo` has the fields `width`, `height`, `fps: (u32, u32)`, `frames` and `video: Vec<u8>`.

`minimax_h3::generate(model: &Path, lora: &Path, request: &crate::minimax_h3::H3Request, (device, runtime): (usize, &Path), gguf: Option<&Path>) -> Result<crate::minimax_h3::H3Output>` runs a prepared `H3Request` (packed token ids and tags, image grids and pixel values, keyframes, canvas, frames, seed, evaluations and shifts) for callers that build the request themselves.

### HyperFlow H3 Flash

```rust
pub fn run_hyperflow_h3_video(
    request: &HyperFlowH3Request,
    options: crate::RunOptions,
) -> Result<H3Output>
```

```rust
#[derive(Clone, Debug)]
pub struct HyperFlowH3Request {
    pub model: PathBuf,
    pub base_model: PathBuf,
    pub first_frame: Option<PathBuf>,
    pub last_frame: Option<PathBuf>,
    pub prompt: String,
    pub frames: usize,
    pub seed: u64,
}
```

- `model`: HyperFlow H3 Flash directory. CUDA builds require `fl2va_int8_convrot/config.json`; ROCm builds use `fl2va_bf16` or `fl2va_int8_convrot`, whichever is present; when both are present they select `fl2va_bf16` if the detected GPU has equal BF16 and INT8 tensor-core rates and the BF16 checkpoint plus activations fit in device memory.
- `base_model`: MiniMax-H3 companion directory for the text encoder and VAEs.
- `first_frame`, `last_frame`: optional keyframe files; neither gives text-to-video, either one gives a single anchor, both give first+last conditioning.
- `frames`: rounds up to `17n+5`; the aligned count must lie within 5 seconds and 362 frames at 24 fps.
- `seed`: noise seed.

`convrot::ProductKernel::MFastStriped { tiles, stripes, one_barrier }` exposes the panel schedule. Setup sets `one_barrier` only for the sm120 INT8 gate/up product with 37,717 rows, 5,376 input columns, 28,672 output columns and five stripes of up to 59 row tiles each. A false value selects the two-barrier schedule.

The output is `H3Output` as above. `minimax_h3::is_hyperflow_model(model: &Path) -> Result<bool>` tells a HyperFlow directory from a MiniMax-H3 directory by its published transformer component.

```rust
let request = minimax_h3::HyperFlowH3Request {
    model: PathBuf::from("<HyperFlow-H3-Flash-dir>"),
    base_model: PathBuf::from("<MiniMax-H3-dir>"),
    first_frame: None,
    last_frame: None,
    prompt: "A boat crosses a quiet lake".into(),
    frames: 124,
    seed: 42,
};
let out = minimax_h3::run_hyperflow_h3_video(&request, options)?;
```

### Resident MiniMax-H3 and HyperFlow video

On CUDA, `minimax_h3::VideoSession` keeps the conditioner, transformer and video and audio decoders ready for successive requests. Call `prepare()` on a `MiniMaxH3I2vaRequest` or `HyperFlowH3Request` to build each `H3Request`.

```rust
pub fn open(
    (model, companion): (&Path, Option<&Path>),
    (lora, gguf): (Option<&Path>, Option<&Path>),
    options: &RunOptions,
    requests: impl IntoIterator<Item = Result<H3Request>>,
) -> Result<VideoSession>
pub fn render(&mut self, request: &H3Request) -> Result<H3Output>
```

For MiniMax-H3, pass `(&request.model, None)` and `(Some(&request.lora), request.gguf.as_deref())`. For HyperFlow, pass `(&request.model, Some(&request.base_model))` and `(None, None)`.

`open` consumes a nonempty iterator of prepared requests to select a placement and decoder tile group. `render` accepts new seeds, prompt tokens and keyframe pixels when the canvas, frame count, packed token layout, image grids, keyframe positions and sampling schedule match. A shape mismatch returns an error. Each call returns the same RGB8 and stereo F32 layout as the one-shot API.

### MiniMax-H3-Image

```rust
pub fn image_latents(
    model: &Path,
    options: &crate::minimax_h3::ImageRunOptions,
    request: &crate::minimax_h3::ImageRequest,
) -> Result<crate::minimax_h3::ImageLatents>
```

```rust
#[derive(Clone, Debug)]
pub struct ImageRunOptions {
    pub device: usize,
    pub runtime_dir: PathBuf,
}

pub struct ImageReference {
    pub image: crate::serve::image::RgbImage,
}

pub struct ImageRequest {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    pub min_generation_area: usize,
    pub expander: Option<PathBuf>,
    pub references: Vec<ImageReference>,
}
```

- `ImageRequest::new(prompt: impl Into<String>, width: usize, height: usize, seed: u64) -> Self` sets `min_generation_area` to `IMAGE_MIN_GENERATION_AREA` (`2048 * 2048`), no expander and no references.
- `width`, `height`: size of the returned image.
- `seed`: one Philox draw from this seed supplies every noise value of the request.
- `min_generation_area`: smallest generation canvas area in pixels, within `[1024, 4096 * 4096]`; the canvas sides scale by `max(1, sqrt(min_area / (width * height)))` and snap to the latent patch multiple.
- `expander`: optional MiniMax-H3-Image-PE directory; its output replaces `prompt` and does not change the size.
- `references`: upright RGB8 edit references. `serve::image::decode_image(bytes: &[u8], mime: Option<&str>) -> Result<RgbImage>` decodes PNG or JPEG into `RgbImage { width, height, pixels }`.

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct ImageLatents {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub values: Vec<f32>,
    pub canvas_height: usize,
    pub canvas_width: usize,
    pub rgb: Vec<u8>,
}
```

`values` is the normalized generated latent, NCHW `[1, channels, height, width]`, for a `canvas_height` x `canvas_width` canvas. `rgb` is HWC RGB8 at the requested `width` x `height`: clamp, truncation toward zero, then Lanczos-3 of the decoder canvas.

```rust
use flyingfish::{media, minimax_h3, serve};
use std::{fs::File, path::{Path, PathBuf}};

fn main() -> anyhow::Result<()> {
    let model = Path::new("<MiniMax-H3-Image-dir>");
    let options = minimax_h3::ImageRunOptions { device: 0, runtime_dir: PathBuf::from("<runtime-dir>") };
    let mut request = minimax_h3::ImageRequest::new("A red cube on a wooden table", 1024, 1024, 42);
    request.references.push(minimax_h3::ImageReference {
        image: serve::image::decode_image(&std::fs::read("<reference.png>")?, None)?,
    });
    let image = minimax_h3::image_latents(model, &options, &request)?;
    media::write_png(File::create("<out.png>")?, request.width, request.height, &image.rgb)?;
    Ok(())
}
```

`minimax_h3::ImageSession` keeps one set of weights loaded across requests that share one packed geometry:

```rust
pub fn open(
    model: &Path,
    options: &ImageRunOptions,
    requests: impl IntoIterator<Item = Result<ImageRequest>>,
) -> Result<Self>
pub fn render(&mut self, request: &ImageRequest) -> Result<ImageLatents>
```

`open` reads every planned request and errors when the list is empty or a request's shape differs from the first; `render` errors on a request whose shape differs from the session's.

```rust
let prompts = ["A red cube", "A blue sphere"];
let request = |index: usize| minimax_h3::ImageRequest::new(prompts[index], 1024, 1024, index as u64);
let mut session = minimax_h3::ImageSession::open(model, &options, (0..prompts.len()).map(|index| Ok(request(index))))?;
for index in 0..prompts.len() {
    let image = session.render(&request(index))?;
    media::write_png(File::create(format!("<out-{index}.png>"))?, 1024, 1024, &image.rgb)?;
}
```

Stage entries of the same pipeline:

```rust
pub fn image_encode(
    model: &Path,
    (ordinal, runtime): (usize, &Path),
    extent: [usize; 2],
    rgb: &[f32],
    epsilon: &[f32],
) -> Result<(Vec<f32>, [usize; 2])>
pub fn image_sample(
    model: &Path,
    layout: &ImageLayout,
    (device, runtime): (usize, &Path),
    rows: &[f32],
    text: &[bf16],
) -> Result<Vec<f32>>
pub fn image_decode(
    model: &Path,
    (ordinal, runtime): (usize, &Path),
    extent: [usize; 2],
    normalized: &[f32],
) -> Result<Vec<f32>>
```

- `image_encode`: F32 RGB HWC in `[0, 255]` at the pixel `extent` `[height, width]` and the caller's posterior noise to normalized F32 NCHW latents and their latent extent.
- `image_sample`: the six published Euler evaluations on `rows` (`[image rows, 96]`, references first) with `text` conditioner rows (`half::bf16`); returns the image rows after the six updates, with reference rows unchanged. `minimax_h3::image_layout(spec: &ImageSpec, text_tags: &[u8], references: &[(usize, usize)], latent_height: usize, latent_width: usize) -> Result<ImageLayout>` builds the packed layout.
- `image_decode`: normalized F32 NCHW latents at the latent `extent` `[height, width]` to F32 RGB HWC before clamp and quantization.

### MiniMax-Music3

```rust
pub fn generate(
    model: &Path,
    request: &crate::music3::Music3Request,
    runtime: &Path,
) -> Result<crate::music3::Music3Output>
```

```rust
pub struct Music3Request {
    pub prompt_ids: Vec<u32>,
    pub frames: usize,
    pub seed: u64,
    pub device: usize,
}
pub struct Music3Output {
    pub pcm: Vec<f32>,
}
```

- `prompt_ids`: paired conditional and unconditional prompt ids, `[2, prompt_rows]`, row 0 conditional; the length must be even. `music3::prompt_ids(directory: &Path, prompt: &str, lyrics: &str) -> Result<Vec<u32>>` builds them from a caption and lyrics with the published prompt construction.
- `frames`: audio frame count; the `music` command defaults to 1500 (60 seconds).
- `seed`: drives the autoregressive draws and the initial latent draw.
- `device`: GPU ordinal.

`music3::generate` runs the autoregressive stage, releases it, then runs the condition encoder, flow transformer and vocoder. `pcm` is stereo `[2, n]`, channel-major; `Music3Output::interleaved(&self) -> Result<Vec<f32>>` returns interleaved samples. The output rate is `music3::SynSpec::read(dir)?.output_rate` (44.1 kHz for the published model). `music3::generate` and `music3::Runtime` are compiled only with `cuda`.

```rust
use flyingfish::{media, music3};
use std::{fs::File, path::Path};

fn main() -> anyhow::Result<()> {
    let model = Path::new("<MiniMax-Music3-dir>");
    let prompt_ids = music3::prompt_ids(model, "A warm acoustic pop song", "[Verse] Morning light")?;
    let rate = music3::SynSpec::read(model)?.output_rate;
    let request = music3::Music3Request { prompt_ids, frames: 1500, seed: 7, device: 0 };
    let output = music3::generate(model, &request, Path::new("<runtime-dir>"))?;
    media::write_wav(File::create("<song.wav>")?, rate as usize, 2, &output.interleaved()?)?;
    Ok(())
}
```

## Image to 3D

All image-to-3D entries are CUDA-only and live in `backend`. Inputs are RGBA with the background already removed; `serve::image::decode_image_rgba(bytes: &[u8], mime: Option<&str>) -> Result<(Vec<u8>, usize, usize)>` decodes a PNG to `(rgba, width, height)`. Meshes are `sparse::SparseMesh { vertices: Vec<f32>, faces: Vec<i32> }`: row-major `[vertices, 3]` world coordinates and `[triangles, 3]` indices with winding and multiplicity retained. `media::write_ply(writer: impl Write, mesh: &MeshWrite<'_>) -> Result<()>` and `media::write_obj` write a `MeshWrite { positions, triangles, rgb, extras }`; PLY carries the named `extras` channels and OBJ does not. `media::read_mesh(bytes: &[u8], extension: &str) -> Result<MeshRead>` reads `ply` or `obj` into `MeshRead { positions, triangles }`.

### Pixal3D

```rust
pub struct PixalSeededRequest<'a> {
    pub model: &'a Path,
    pub dino: &'a Path,
    pub naf: &'a Path,
    pub runtime: &'a Path,
    pub views: &'a [PixalView<'a>],
    pub multiview: bool,
    pub requested_resolution: u32,
}
pub fn run(&self, seed: u64, device: &Device) -> Result<PixalOutput>
```

```rust
pub struct PixalView<'a> {
    pub rgba: &'a [u8],
    pub dimensions: [usize; 2],
    pub camera: crate::backend::ProjectionCamera,
}
pub struct ProjectionCamera {
    pub world_to_camera: [f32; 12],
    pub horizontal_fov: f32,
    pub mesh_scale: f32,
}
```

- `model`, `dino`, `naf`: Pixal3D, DINOv3 and NAF directories; `runtime`: runtime directory.
- `views`: one view for single-view; several with `multiview = true`, which reads `pipeline_mv.json` instead of `pipeline.json`. `dimensions` is `[height, width]`.
- `camera`: row-major 3x4 world-to-camera matrix on the rotated XYZ grid, horizontal field of view in radians, and mesh scale. Multiview requires a distinct extrinsic per view.
- `requested_resolution`: 1024 or 1536.
- `seed`: the four stage noise draws (structure, low-resolution shape, high-resolution shape, texture) come from Philox counter streams `backend::PIXAL_NOISE_STREAMS` (`[1, 2, 3, 4]`), not the publisher's torch RNG.

`PixalGeneration<'a>` has the same fields plus `noise: [CoordinateNoise<'a>; 4]` for caller-supplied noise, and `PixalGeneration::run(&self, device: &Device) -> Result<PixalOutput>`.

```rust
pub struct PixalOutput {
    pub mesh: crate::sparse::SparseMesh,
    pub material: SparseDecoderOutput,
    pub resolution: u32,
    pub origin: [f32; 3],
    pub voxel_size: f32,
}
pub struct SparseDecoderOutput {
    pub coordinates: Vec<[i32; 4]>,
    pub features: Vec<f32>,
    pub channels: usize,
}
```

`material.features` is `[rows, channels]` with R, G, B, metallic, roughness and alpha at the final voxel coordinates. `mesh` is raw.

```rust
use flyingfish::backend::{BackendDevice, PixalSeededRequest, PixalView, ProjectionCamera};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let (rgba, width, height) = flyingfish::serve::image::decode_image_rgba(&std::fs::read("<image.png>")?, None)?;
    let camera = ProjectionCamera {
        world_to_camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0],
        horizontal_fov: 0.7,
        mesh_scale: 1.0,
    };
    let views = [PixalView { rgba: &rgba, dimensions: [height, width], camera }];
    let device = BackendDevice::new(0)?;
    let output = PixalSeededRequest {
        model: Path::new("<Pixal3D-dir>"),
        dino: Path::new("<DINOv3-dir>"),
        naf: Path::new("<NAF-dir>"),
        runtime: Path::new("<runtime-dir>"),
        views: &views,
        multiview: false,
        requested_resolution: 1536,
    }
    .run(42, &device)?;
    let mesh = flyingfish::media::MeshWrite { positions: &output.mesh.vertices, triangles: &output.mesh.faces, rgb: None, extras: &[] };
    flyingfish::media::write_ply(std::io::BufWriter::new(std::fs::File::create("<mesh.ply>")?), &mesh)?;
    Ok(())
}
```

### Hi3D

```rust
pub struct Hi3DGeneration<'a> {
    pub model: &'a Path,
    pub dino: &'a Path,
    pub clip: &'a Path,
    pub runtime: &'a Path,
    pub rgba: &'a [u8],
    pub dimensions: [usize; 2],
    pub orientation: u8,
    pub mesh: &'a crate::sparse::SparseMesh,
    pub requested_resolution: u32,
    pub seed: u64,
}
pub fn run_processed(&self, device: &Device) -> Result<(crate::sparse::SparseMesh, u32)>
pub fn run(&self, device: &Device) -> Result<(crate::sparse::SparseMesh, u32)>
```

- `dino`, `clip`: DINOv2 and CLIP directories.
- `dimensions`: `[height, width]` of `rgba`.
- `orientation`: the CLI default is 1.
- `mesh`: coarse mesh to refine; it is voxelized before any boundary fill.
- `requested_resolution`: 512, 1024 or 1536.

`run_processed` returns the hole-filled, component-filtered, simplified and unit-radius mesh with its resolution; `run` returns the raw mesh before those steps. Same-seed reruns are bit-stable.

```rust
let coarse = flyingfish::media::read_mesh(&std::fs::read("<coarse.ply>")?, "ply")?;
let coarse = flyingfish::sparse::SparseMesh { vertices: coarse.positions, faces: coarse.triangles };
let (rgba, width, height) = flyingfish::serve::image::decode_image_rgba(&std::fs::read("<image.png>")?, None)?;
let device = flyingfish::backend::BackendDevice::new(0)?;
let (mesh, resolution) = flyingfish::backend::Hi3DGeneration {
    model: Path::new("<Hi3D-dir>"),
    dino: Path::new("<DINOv2-dir>"),
    clip: Path::new("<CLIP-dir>"),
    runtime: Path::new("<runtime-dir>"),
    rgba: &rgba,
    dimensions: [height, width],
    orientation: 1,
    mesh: &coarse,
    requested_resolution: 512,
    seed: 42,
}
.run_processed(&device)?;
```

### TRELLIS.2-4B

```rust
pub struct TrellisTo3d<'a> {
    pub model: &'a std::path::Path,
    pub dino: &'a std::path::Path,
    pub runtime: &'a std::path::Path,
    pub rgba: &'a [u8],
    pub dimensions: [usize; 2],
    pub seed: u64,
    pub device: usize,
}
pub fn run(&self) -> Result<TrellisMesh>
```

`TrellisTo3d` takes a device ordinal and creates its own device. `dino` is the DINOv3 directory; `dimensions` is `[height, width]`. It runs the default 1024-cascade pipeline.

```rust
pub struct TrellisMesh {
    pub vertices: Vec<f32>,
    pub faces: Vec<i32>,
    pub material: Vec<f32>,
    pub resolution: u32,
}
```

`vertices` is world-space `[rows, 3]`, `faces` is `[faces, 3]` with winding retained, and `material` is per-vertex `[rows, 6]` (R, G, B, metallic, roughness, alpha) in the pipeline's output space. The mesh is raw: no fill, simplify, UV, bake or render.

```rust
let (rgba, width, height) = flyingfish::serve::image::decode_image_rgba(&std::fs::read("<image.png>")?, None)?;
let mesh = flyingfish::backend::TrellisTo3d {
    model: Path::new("<TRELLIS.2-4B-dir>"),
    dino: Path::new("<DINOv3-dir>"),
    runtime: Path::new("<runtime-dir>"),
    rgba: &rgba,
    dimensions: [height, width],
    seed: 42,
    device: 0,
}
.run()?;
```

### Idol

On CUDA, `backend::IdolPipelineRuntime::open` takes the Idol checkpoint directory, the DINOv3 ViT-H/16+ directory, the FLUX.2-dev `vae` subdirectory and `&RunOptions`. Open once on the selected device and reuse the runtime with a prepared image and a seed for each request. The result contains a Gaussian splat and a camera with yaw and pitch in radians, radius, and field of view in degrees.

```rust
use flyingfish::{RunOptions, backend::{IdolCamera, IdolPipelineRuntime}};
use std::path::Path;

pub fn generate_idol(
    idol: &Path, dinov3: &Path, vae: &Path, options: &RunOptions,
    prepared_rgb: &[f32], seed: u64, output: &Path,
) -> anyhow::Result<IdolCamera> {
    let mut runtime = IdolPipelineRuntime::open(idol, dinov3, vae, options)?;
    let (splat, camera) = runtime.generate_from_rgb(prepared_rgb, seed)?;
    splat.save_ply(output)?;
    Ok(camera)
}
```

`prepared_rgb` contains three contiguous planes (red, green, blue), each 1024 × 1024 pixels, with finite F32 values in [0, 1]. Callers can use the publisher's saved preprocessed RGB image directly by dividing each channel byte by 255 and arranging the values in that order.

Raw-image background removal and framing are caller-side steps. To match the Idol model definition, prepare the image as follows.

1. Resize with Lanczos, preserving aspect ratio and rounding each scaled dimension to the nearest integer so the short side is 1024. Use the resized alpha channel if any pixel has alpha below 255. Otherwise obtain an alpha matte from the publisher's BiRefNet background remover on the RGB image.
2. Erode alpha with a 3 × 3 minimum filter. Find the inclusive bounds of its nonzero pixels. An empty mask is an error. Center a square on the midpoint of those bounds, with side equal to the longer coordinate span (max minus min) times the crop margin. The default margin is 1.2 for a synthetic image or 1.7 for a real photo. Truncate each crop edge toward zero and pad beyond the image with transparent black.
3. Resize the RGBA crop to 1024 × 1024 with Lanczos and composite onto black using the resized alpha mask. Convert the RGB bytes to planar F32 as above. The runtime applies DINOv3's ImageNet mean and standard deviation and maps the FLUX.2 VAE input to [-1, 1] internally.

## Language models

Language and diffusion language models load through `model::Model`, a wrapper over `backend::Backend`. Loading and decoding need `cuda` or `rocm`; MTP decode is CUDA-only.

```rust
pub fn load(
    dir: &Path,
    ordinal: usize,
    capacity: usize,
    chunk: Option<usize>,
    settings: Settings,
) -> Result<Self>
```

- `dir`: model directory.
- `ordinal`: GPU ordinal.
- `capacity`: token workspace and KV capacity (CLI default 4096).
- `chunk`: prefill chunk override; `None` derives it from the VRAM budget and tile shapes.
- `settings`: `backend::Settings`.

```rust
pub struct Settings {
    pub tree_builder: TreeBuilder,
    pub spec_budget: SpecBudget,
    pub spec_graph: bool,
    pub draft_model: Option<std::path::PathBuf>,
    pub prefix_cache: bool,
    pub kv_format: crate::kv::KvFormat,
    pub runtime_dir: Option<std::path::PathBuf>,
    pub device_budget_gib: Option<u64>,
}
```

| Field | CLI default | Meaning |
|---|---|---|
| `tree_builder` | `TreeBuilder::Waves` | Speculative tree construction (`Serial`, `Waves`) |
| `spec_budget` | `SpecBudget::Tree16` with a draft model, `SpecBudget::Chain` without | Speculative verification budget (`Chain`, `Tree16`, `Tree32`, `Tree64`) |
| `spec_graph` | `true` | Speculative graph execution |
| `draft_model` | `None` | DSpark draft directory; `Settings::with_draft(&self, path: &Path) -> Self` sets it |
| `prefix_cache` | `false` | Prefix cache |
| `kv_format` | `KvFormat::Bf16` | `Affine8` stores an 8-bit KV cache |
| `runtime_dir` | Resolved by the CLI | Runtime directory; required |
| `device_budget_gib` | `None` | Declared device budget in GiB for the expert-tier derivation; `None` senses free device memory at setup |

Generation:

```rust
pub fn generate_plain(
    model: &mut Model,
    ids: &[u32],
    count: usize,
    ignore_eos: bool,
    graph: bool,
    trace: Option<&mut Trace>,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<GenerationOutput>
pub fn generate_multimodal(
    model: &mut Model,
    input: &crate::vision_input::MultimodalInput<'_>,
    count: usize,
    ignore_eos: bool,
    graph: bool,
    trace: Option<&mut Trace>,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<GenerationOutput>
pub struct GenerationOutput {
    pub ids: Vec<u32>,
    pub prefill_ms: f64,
    pub capture_ms: f64,
    pub decode_ms: f64,
}
```

- `count`: maximum new tokens, positive. `model::check_generation_capacity(prompt: usize, count: usize, capacity: usize) -> Result<()>` checks that the prompt and generation fit the KV capacity.
- `ignore_eos`: `false` stops at the model config's `eos_token_id`.
- `graph`: capture and replay a decode graph; Clover and DiffusionGemma require `false`.
- `trace`: operator trace, or `None`.
- `on_token`: called with each new token id; returning `false` stops generation.
- `ids` in the output holds the generated tokens, starting with the token prefill emits.

Decoding is greedy: `Model::prefill(&mut self, ids: &[u32], trace: Option<&mut Trace>) -> Result<u32>` and `Model::decode(&mut self, graph: bool, trace: Option<(&mut Trace, &str)>) -> Result<u32>` return the argmax token id, and `Model::decode_logits(&self) -> Result<Vec<f32>>` returns the decode logits. `Model::reset(&mut self) -> Result<()>` clears the request state. `sampling::Sampling` serves Breeze-TTS-2 only. Text goes in and out through `tokenizer::encode(dir: &Path, text: &str) -> Result<Vec<u32>>` and `tokenizer::decode(dir: &Path, ids: &[u32]) -> Result<String>`, which read `tokenizer.json` from the directory and apply no chat template.

```rust
use flyingfish::backend::{Settings, SpecBudget, TreeBuilder};
use flyingfish::model::{self, Model};
use std::path::{Path, PathBuf};

fn main() -> anyhow::Result<()> {
    let dir = Path::new("<model-dir>");
    let settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        draft_model: None,
        prefix_cache: false,
        kv_format: flyingfish::kv::KvFormat::Bf16,
        runtime_dir: Some(PathBuf::from("<runtime-dir>")),
        device_budget_gib: None,
    };
    let ids = flyingfish::tokenizer::encode(dir, "Hello")?;
    model::check_generation_capacity(ids.len(), 64, 4096)?;
    let mut model = Model::load(dir, 0, 4096, None, settings)?;
    let output = model::generate_plain(&mut model, &ids, 64, false, true, None, &mut |_| true)?;
    println!("{}", flyingfish::tokenizer::decode(dir, &output.ids)?);
    Ok(())
}
```

`MultimodalInput` carries processor-packed input:

```rust
pub struct MultimodalInput<'a> {
    pub input_ids: &'a [u32],
    pub modality_ids: &'a [u8],
    pub pixel_values: &'a [f32],
    pub grids: &'a [ImageGrid],
    pub audio: &'a [AudioClip<'a>],
}
```

`modality_ids` has one entry per token: 0 text, 1 image, 2 video, 3 audio. `serve::image::PreparedInput::read(spec: &ModelSpec, path: &Path, ids: &[u32]) -> Result<Self>` loads a processor-packed image trace and `PreparedInput::input(&self, ids)` returns its `MultimodalInput`.

Speculative decoding:

- MTP: `Model::generate_mtp(dir: &Path, ids: &[u32], options: &crate::spec::Options, depth: usize, policy: crate::mtp::VerifyPolicy, graph: bool) -> Result<crate::mtp::Output>` loads the checkpoint's MTP component and generates one request; `mtp::Output` has `ids`, `load_ms`, `prefill_ms` and `decode_ms`. `checkpoint::has_mtp_component(dir: &Path) -> Result<bool>` reports whether a checkpoint carries MTP weights. `Model::load_mtp` and `Model::generate_mtp_reuse` keep one MTP model across requests. ROCm builds return `HIP generate_mtp is unsupported`.
- DSpark: `spec::generate(target: &Path, draft: &Path, ids: &[u32], media: Option<&crate::vision_input::MultimodalInput<'_>>, options: &Options, dump: Option<&Path>) -> Result<()>` runs one drafted request and prints per-round statistics and the result as JSON lines on stdout.

`spec::Options` groups the load arguments:

```rust
pub struct Options {
    pub device: usize,
    pub capacity: usize,
    pub chunk: Option<usize>,
    pub backend: Settings,
    pub count: usize,
    pub ignore_eos: bool,
}
```

### OpenAI-compatible server

```rust
pub fn run(
    target: PathBuf,
    draft: Option<PathBuf>,
    options: SpecOptions,
    tokenizer_dir: PathBuf,
    port: u16,
) -> Result<()>
pub fn run_mtp(
    target: PathBuf,
    options: SpecOptions,
    tokenizer_dir: PathBuf,
    port: u16,
    depth: usize,
) -> Result<()>
```

`serve::run` loads the model (and the DSpark draft when `draft` is `Some`) on an engine thread, binds `127.0.0.1:<port>`, serves `/v1/completions` and `/v1/chat/completions`, and returns after Ctrl-C shuts the server down. `serve::run_mtp` serves with MTP at `depth`. `SpecOptions` is `spec::Options`; the CLI passes `count: 256` and `ignore_eos: false`. A request's `max_tokens` or `max_completion_tokens` sets its token count, default 256. Chat messages of image-capable models accept `image_url` parts with a base64 PNG or JPEG data URL.

```rust
let dir = PathBuf::from("<model-dir>");
let options = flyingfish::spec::Options { device: 0, capacity: 4096, chunk: None, backend: settings, count: 256, ignore_eos: false };
flyingfish::serve::run(dir.clone(), None, options, dir, 8080)?;
```

## Embeddings

```rust
pub fn embed_text(directory: &Path, text: &str, options: crate::RunOptions) -> Result<Vec<f32>>
pub fn embedding_json(values: &[f32]) -> Result<serde_json::Value>
```

`model::embed_text` tokenizes `text`, loads EmbeddingGemma-2 with a workspace of `max(tokens, 1024)` rows, and returns 768 finite values; an input longer than the model context is an error. `model::embedding_json` returns `{"embedding": [...], "dimension": 768}`. The CLI checks the architecture before calling it; `embed_text` does not. Image and audio embeddings go through `Model::embed(&mut self, input: &crate::vision_input::MultimodalInput<'_>, mask: &[bool], trace: Option<&mut Trace>) -> Result<Vec<f32>>` with processor-packed `pixel_values` and `grids` or `audio` clips and one mask entry per token. `Model::prefill` rejects EmbeddingGemma-2 with `EmbeddingGemma-2 requires the embedding entry point`.

```rust
let values = flyingfish::model::embed_text(Path::new("<EmbeddingGemma-2-dir>"), "a red cube", options)?;
println!("{}", flyingfish::model::embedding_json(&values)?);
```

## Stage layer

`backend::BackendDevice` is exported with `cuda` or `rocm`. Its public fields and methods:

```rust
pub info: DeviceInfo,
pub ctx: Arc<CudaContext>,
pub stream: Arc<CudaStream>,
pub upload: Arc<CudaStream>,

pub fn new(ordinal: usize) -> Result<Self>
pub fn finish_upload(&self) -> Result<()>
```

`new` opens GPU `ordinal` with its context, an execution stream and an upload stream; with `cuda` it first checks that the CUDA driver, cuBLAS and cuBLASLt libraries load and names the one that does not. `finish_upload` makes the execution stream wait for the upload stream and synchronizes it. `backend::gpu` re-exports the context, stream, slice and launch types for the selected backend.

The caller owns every object it constructs: the device, each stage runtime and each workspace, and their drop order. Exported stage types include:

- Speech: `parakeet::{ParakeetEncoder, ParakeetTdt}`, `canary::Canary`, `eou::Eou`, `diarization::Diarization`, `multitalker::Multitalker`, each with `load(<directory or .nemo>: &Path, device: &Device) -> Result<Self>`.
- Breeze-TTS-2: `backend::Breeze::load_for_generation(weights: &Path, runtime: &Path, ordinal: usize, text_rows: usize, max_new_tokens: usize) -> Result<Self>` and `backend::BreezeTokenizer::load_for_frames(path: &Path, device: &Device, capacity: usize) -> Result<Self>`.
- MiniMax-Music3: `music3::Runtime::load(model: &Path, runtime: &Path, prompt: usize, frames: usize, device_limit: usize, ordinal: usize) -> Result<Self>` for the autoregressive stage and `backend::Music3Syn::load(dir: &Path, d: &Device, max_length: usize) -> Result<Self>` for synthesis.
- Vision and image-to-3D: `DinoV3Encoder`, `DinoV3Workspace`, `Flux2VaeEncoder`, `Flux2VaeWorkspace`, `IdolVisionEncoderRuntime`, `IdolDenoiser`, `IdolDenoiserWorkspace`, `IdolOctreeDecoder`, `IdolOctreeWorkspace`, `IdolGaussianDecoder`, `IdolGaussianWorkspace`, `Hi3DImageConditioner`, `TrellisImageConditioner`, `TrellisSsFlow`, `PixalConditioningRuntime`, `PixalDecoder` and `Flow`.

`DinoV3Encoder::load`, `Flux2VaeEncoder::load`, `IdolVisionEncoderRuntime::load`, `Hi3DImageConditioner::load`, `TrellisImageConditioner::load`, `PixalConditioningRuntime::load`, and `TrellisSsFlow::load` take `&mut Blas`, which has no public constructor; those stages are usable only through the request entries above.

## Errors

Every entry returns `anyhow::Result`. Setup and input checks fail with an error that names what failed (a missing file or component, an extent mismatch, a non-finite value, an unsupported architecture); there are no silent fallbacks.

The generic `backend::Backend` loader and the autoregressive `model::layer_program` walker have these execution-domain errors:

| `Architecture` | Entry | Exact error |
|---|---|---|
| `Qwen3` | `Backend` load | `Qwen3 AR execution is not bound` |
| `Qwen3Vl` | `Backend` load and AR layer walker | `Qwen3-VL runs only as the MiniMax-H3 conditioner` |
| `Clover` | AR layer walker | `Clover canvas execution is not bound` |
| `DiffusionGemma` | AR layer walker | `DiffusionGemma layer program is not bound yet` |

Clover and DiffusionGemma load on the separate eager canvas path. Their graph capture returns `canvas graph execution is not bound`.
