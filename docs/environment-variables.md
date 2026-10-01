# Environment variables

One reference for every environment variable the committed code reads: what it controls, the accepted values, what happens when it is unset (the derived default), and the reading site. This is the single place env-var behavior is explained; other docs keep at most an inline mention with a link. Numeric overrides go through one shared reader: an invalid, non-Unicode, zero or out-of-range value stops the run with a named error naming the variable (crates/ff-core/src/probe.rs:1122-1166). Presence flags accept any value and are marked as such. The README build section additionally shows `CUDA_COMPUTE_CAP` and `FF_CUDA_ARCHS` in its build steps.

## Build

| Variable | Controls | Values | Unset | Site |
|---|---|---|---|---|
| `FF_CUDA_ARCHS` | The architectures translated ahead of time into cubins (the README build section shows its use in build steps) | comma list of dot-less compute capabilities such as `80,86,89,90,100,120`; an invalid entry fails the build | union of the fleet list, `CUDA_COMPUTE_CAP` and the build host's GPUs | crates/ff-cuda/src/build.rs:19, :109-128 |
| `CUDA_COMPUTE_CAP` | Target capability for a GPU-less build (README build section) | a positive dot-less capability such as `86`/`120`, or `major.minor` such as `8.6`/`12.0` with minor 0..=9; an unparseable value fails the build with `invalid CUDA_COMPUTE_CAP=...` | not added to the set | crates/ff-cuda/src/build.rs:110-111, :145 |
| `NVCC` | The CUDA compiler executable for kernel builds | path to an `nvcc` | `nvcc` from `PATH` | crates/ff-cuda/src/build.rs:355; also third_party/candle-kernels/build.rs:45 |
| `PTXAS` | The PTX assembler for ahead-of-time translation | path to a `ptxas` | the toolkit's `ptxas` | crates/ff-cuda/src/build.rs:89 |
| `FLYINGFISH_CUDA_NVCC_VERSION` | Compile-time record of the nvcc version the H3 kernels were built with, frozen into the hardware profile; exported by the build script, not an operator input | set by the build | build fails (the constant is compiled in) | crates/ff-h3/build.rs:38, crates/ff-h3/src/cuda/profile.rs:17 |
| `CARGO_MANIFEST_DIR`, `OUT_DIR`, `TARGET`, `CARGO_FEATURE_CUDA` | Build-system inputs set by Cargo itself, not operator switches | — | — | e.g. crates/ff-cuda/src/build.rs:349, :357; crates/ff-qwen35/build.rs:21, :24; third_party/candle-kernels/build.rs:14, :16 |

## Devices and CUDA

| Variable | Controls | Values | Unset | Site |
|---|---|---|---|---|
| `CUDA_VISIBLE_DEVICES` | CUDA-standard visible-device filter and reorder; the topology fingerprint treats a remap as a different machine, so host-profile records do not transfer across it | CUDA device-ordinal list | all devices in driver order | crates/ff-core/src/topology.rs:224-229 |

## Core

| Variable | Controls | Values | Unset | Site |
|---|---|---|---|---|
| `FF_MODELS_DIR` | Root for checkpoint-relative directory resolution | path | no root: callers skip or fail rather than guessing one | crates/ff-core/src/paths.rs:7-13 |
| `FF_WEIGHT_LOAD_THREADS` | Weight-loading worker count | integer 1 or larger; an invalid or out-of-range value is a named error (`{name} must be in {min}..={max}; supply a valid value and rerun ff`) | 32 | crates/ff-core/src/weights.rs:662-666, crates/ff-core/src/probe.rs:1150-1166 |
| `FF_WEIGHT_WARM_THREADS` | Weight-warming worker count (unix) | integer 1 or larger; invalid values are named errors as above | 8 | crates/ff-core/src/weights/cuda_allocation.rs:91-95, crates/ff-core/src/probe.rs:1150-1166 |
| `FF_COLD_CACHE_EVIDENCE` | Names the JSON with measured cold-file evidence that cold-cache gates consume | path to a JSON report | claimed nonzero residency is an error naming this variable | crates/ff-core/src/cold_cache.rs:74, :95 |
| `FF_TOPOLOGY_PROFILE` | The measured topology record H3 planning reads | path to a profile; an invalid record is refused with `regenerate it with ff probe --json` | the topology is captured live from the device | src/cli/generate.rs:1767-1776 |

## Qwen3.8-27B

| Variable | Controls | Values | Unset | Site |
|---|---|---|---|---|
| `FF_GROUP4_BODY` | Explicit group4 kernel body — the alternative to the persisted `--host-profile` record that CUDA int4 runs of qwen35 and Edge0 require (the requirement, including the 16-bit exception and the column-2 speculative record, is in the [model guide](models.md#qwen38-27b)) | `stock` or `xr16`; any other value is refused with `expected stock or xr16` | the persisted record is required for CUDA int4 runs; `ff bench group4` refuses to run while this is set, so a calibration always measures both bodies | crates/ff-edge0/src/gpu.rs:339-346, src/cli/calibrate_io.rs:342-344 |
| `QWEN35_PREFILL` | Pins the MLP prefill backend; the derived default is packed int4, which preserves weight residency and falls back to GEMV when the packed workspace would reduce it (model guide) | `gemv`, `packed` or `mma`; any other value is refused with `must be gemv, packed or mma` | derived by the startup probe | crates/ff-qwen35/src/gpu.rs:652-658, crates/ff-qwen35/src/gpu.rs:794-800 |
| `QWEN35_GRAPH` | Captured decode-graph replay | `0` disables, `1` enables; anything else is refused with `must be 0 or 1` | 1 (enabled); the setting is recorded in the group4 program identity | crates/ff-qwen35/src/gpu.rs:141-147, :2463, :2591; src/cli/calibrate_io.rs:346-347 |
| `QWEN35_FORCE_STREAM` | Forces the streamed-layer plan in production constructors and reference checks | any value sets it | the residency planner decides from measured memory | crates/ff-qwen35/src/gpu.rs:130-134 |
| `QWEN35_FREE_OVERRIDE_MIB` | Replaces the free-memory readings feeding the residency planner | MiB as an integer; a value that overflows byte arithmetic is refused with a named error | the device's measured free memory | crates/ff-qwen35/src/gpu.rs:139-155, :565 |
| `QWEN35_CHECK_DEVICES` | Ordered device list for the `logit_dump` numerical corpus | comma list of ordinals such as `0,1`; an unparsable entry is an error | `0` | crates/ff-qwen35/examples/logit_dump.rs:21, :487-492 |

## Edge0-35B-A3B

| Variable | Controls | Values | Unset | Site |
|---|---|---|---|---|
| `EDGE0_MAX_CTX` | Context bound at GPU/KV setup | integer 1..=8192; anything else is a named error | 4096 | crates/ff-edge0/src/model.rs:22-27 |
| `EDGE0_MATVEC_THREADS` | CPU matvec worker count, resolved once at setup alongside `EDGE0_MATVEC_INLINE` | positive integer; anything else is a named error | `(out_dim / 512).clamp(1, cores)` per projection | crates/ff-edge0/src/int4.rs:208-212, :265-285 |
| `EDGE0_MATVEC_INLINE` | Forces inline CPU projection scheduling for small, parallel expert execution; resolved once at setup | any value sets it | the scheduling heuristic decides | crates/ff-edge0/src/int4.rs:208-212, :265-281 |
| `EDGE0_MEGA` | Selects the separate compiled MoE execution mode | any value sets it | the standard grouped path; the persisted program key records the setting either way | crates/ff-edge0/src/gpu.rs:2281, :4602, crates/ff-edge0/src/model.rs:916, :949 |

## GLM-5.3-Flash

| Variable | Controls | Values | Unset | Site |
|---|---|---|---|---|
| `FF_GLM_DIRECT_FILL` | Direct reads instead of buffered reads for pinned FP8 uploads (Linux), for covered aligned spans; allows device page warming to be skipped | `0` or `1`; `1` enables, anything else is a named error | 0 (buffered reads) | crates/ff-glm/src/fp8/staging.rs:580-585 |
| `FF_GLM_FILL_AHEAD` | Fill-ahead workers preparing pinned uploads ahead of use | integer 0..=8; out-of-range values are named errors, not clamped | 0 (off) | crates/ff-glm/src/model.rs:335-338 |
| `FF_GLM_HOST_SHARE` | Explicit per-mille share of each routed miss set evaluated on the host, recorded in the execution policy | integer per-mille; above 1000 is refused with `must be in 0..=1000`, not clamped | the host profile's measured `host_share()` | crates/ff-glm/src/model.rs:228-237, :1062 |
| `FF_GLM_HOST_THREADS` | Host expert workers | integer 1 or larger; anything else is a named error | 4 | crates/ff-glm/src/fp8.rs:296-300, crates/ff-glm/src/model.rs:240-242 |
| `FF_GLM_HOST_MATVEC_THREADS` | Per-matvec worker count dividing host expert CPU capacity | integer 1 or larger; anything else is a named error | `cores / FF_GLM_HOST_THREADS`, minimum 1 | crates/ff-glm/src/fp8.rs:296-303 |
| `FF_GLM_LOAD_LANES` | CUDA staging concurrency for FP8 uploads | integer 1..=8; out-of-range values are named errors, not clamped | 1 | crates/ff-glm/src/fp8/cuda.rs:28 |
| `FF_GLM_PREFETCH_THREADS` | Page-warming reader override. GLM requires an explicit `--host-profile` or **both** reader overrides set; with neither the command refuses with `missing --host-profile`. An unparsable value is refused with `invalid FF_GLM_PREFETCH_THREADS`. Required unset during `ff bench io` calibration | integer 1 or larger | the host profile's reader count | src/cli/glm.rs:143-162, crates/ff-glm/src/model.rs:371-379, :1066-1075 |
| `FF_GLM_PINNED_FILL_THREADS` | Pinned-fill reader override, same rules as the warming override (both must be set to replace a profile) | integer 1 or larger | the host profile's reader count | src/cli/glm.rs:143-162, crates/ff-glm/src/model.rs:371-379, :1066-1075 |
| `FF_GLM_IO_TRACE` | Adds process-wide I/O and page-fault deltas to GLM generation reports | `0` or `1`; anything else is a named error | 0 (off) | crates/ff-glm/src/io_trace.rs:35-36 |

Measured context for `FF_GLM_DIRECT_FILL`: controlled I/O experiments measured buffered reads ahead of direct reads on both test machines — 25.3% lower wall time with a 4.9% prefill cost on two A4000s, 19.6% lower wall time with a 5.8% prefill cost on the 2026-09-28 A4000 rerun, and 18% lower wall time on an RTX 4090 (decode 37% faster, prefill 14% slower). A transient latency episode on one NVMe drive of the local LVM set invalidated an earlier 4090 batch, so check per-disk read latency before trusting cold-read timings.

## MiniMax-H3

| Variable | Controls | Values | Unset | Site |
|---|---|---|---|---|
| `FF_H3_PREFETCH` | Prefetches each stage's weights on a second stream while the current stage computes, shortening cold runs by roughly a quarter | `0` disables, `1` enables; anything else is a named error | 1 (enabled) | crates/ff-h3/src/prefetch.rs:21-28 |
| `FF_CUDA_TUNED_KERNELS` | Disables the tuned-kernel composition; not a correctness escape, both compositions execute | `0` disables, `1` enables; anything else is a named error | 1 (tuned kernels used) | crates/ff-h3/src/cuda/profile.rs:315-321 |
| `FF_H3_TIMING` | Destination for H3 stage timing | `0` or empty disables, `1` writes to stderr, any other value is a file path | disabled | crates/ff-h3/src/timing.rs:19, :28-34, :208 |
| `CUBLAS_WORKSPACE_CONFIG`, `CUBLASLT_WORKSPACE_SIZE`, `TORCH_CUBLASLT_UNIFIED_WORKSPACE` | Read and recorded as part of the reference-build identity; honored by the NVIDIA libraries themselves | NVIDIA-defined | NVIDIA defaults | crates/ff-h3/src/cuda/profile.rs:11-15, :277 |

## Ambient snapshot

These are read once and recorded — not interpreted — in resource-policy evidence, so a report shows the environment it ran under: `RAYON_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `CUBLAS_WORKSPACE_CONFIG`, `NVIDIA_TF32_OVERRIDE`, `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`, `CUDA_LAUNCH_BLOCKING` (src/resource_policy/evidence.rs:153-171).
