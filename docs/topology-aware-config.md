# Topology-aware configuration derivation

`ff video generate` and sibling commands derive their performance configuration from a measured machine profile instead of requiring operator flags. Explicit flags still win; the derivation fills whatever the operator left unset. `--explain-config` prints the derivation trace with the numbers behind every decision.

## Design

Three layers, each in its own module:

1. `ff-core::topology` — `TopologyProfile` captures the machine: device inventory (name, VRAM, compute capability per CUDA ordinal), host memory, interconnect level, and a storage-bandwidth slot fed from the I/O calibration cache. Profiles persist as JSON and reload only on the machine that produced them, keyed by the same `HardwareFingerprint` mechanism the calibration caches use.
2. `ff-core::configure` — `derive(ordinal, profile, requirement) -> DerivedConfig` applies the rules below. The consuming device's own VRAM (selected by ordinal) drives device-side rules. Every decision appends a provenance step with its inputs and outcome.
3. Adapters — each model architecture implements `ModelRequirement`: steady-state weight bytes re-read per evaluation, activation peak bytes, single-pass weight bytes (streamed once, e.g. text encoders), FLOPs per evaluation, full materialization bytes, and optionally a chunk-plan ladder search. Unmodeled quantities default to zero and the affected rules degrade to the static defaults.

## Rules

1. **Weight source** — `Memory` when full materialization plus an OS allowance fits in host RAM, otherwise `Mmap`. The criterion charges the full materialization (the resident-set reality of the memory source), not the steady re-read set; an earlier steady-based criterion admitted machines that later died to swap storms. Single-pass weights always stream through mmap with drop-behind regardless (measured 180-490s -> 48s on the H3 text encoder). When the derivation picks `Mmap`, the provenance reports the per-evaluation re-read cost, using measured storage bandwidth when available (FlexGen's fastest-tier placement principle). Under `Memory` the derivation also sizes the two host bounds that keep admission consistent: the cache ceiling (`min(host - reserve, materialization)`) and the admission host bound, recomputed with the same cache-charge function and cache policy the run itself uses, so a derived run cannot be refused by a bound it derived.
2. **Chunks** — pick the largest chunk plan whose activation peak leaves the admission reserve inside the selected device's VRAM (roofline: larger chunks raise MFU until the device wall). Architectures without a chunk model keep static defaults.
3. **Residency budget** — device VRAM minus activation peak minus the admission reserve, handed to the existing residency planner. Budget is an upper bound; the planner fills less when phase demand is sparser.
4. **Prefetch** — on CUDA with event tracking, stage-ahead weight prefetch on a second stream is on by default (`FF_H3_PREFETCH=0` disables). Steady-state effect is noise-level on fast hosts; cold starts improve by roughly a quarter.
5. **Multi-device** — recorded in the profile but not yet derived (P2). Parallel clips, one per device, is the measured winner below NVLink-class interconnect (~1.7x on 2x A4000); sequence-parallel only pays with fast P2P links.

## Validation

The derivation reproduces the hand-tuned optima measured on three machines: RTX 4090 24GB/62.6GiB (Memory, 4096/1024 chunks), RTX PRO 6000 96GB/1TiB (Memory), and RTX A4000 16GB/30GB (Mmap). Hosts whose RAM sits within roughly 1GiB of the model's materialization are knife-edge: the derivation admits them to `Memory` only when the OS allowance fits, and the H3 checkpoint at 61.7GiB wants a 64GB-or-larger host for comfortable operation.

## References

Roofline (Williams et al. 2009); FlexGen (Sheng et al. 2023) tiered placement; ZeRO-Inference pinned-buffer prefetch; Helix (2024) interconnect-aware placement; xDiT (2024) diffusion parallelism ablations; Ansor (2020) measurement-driven tuning.
