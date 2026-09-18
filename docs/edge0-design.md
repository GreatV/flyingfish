# Edge0-35B-A3B adapter design

Status: draft for review (implementer: 2d, reviewer: 79, per user 2026-09-16).
Worktree: `../flyingfish-edge0`, branch `edge0-support` off `main` (5008a5a).
Checkpoint: `models/Edge0/Edge0-35B-A3B-preview` (19 GiB, 4 shards) — all
facts below verified from `config.json` and safetensors headers on 2026-09-16.

## 1. What the model is

`Qwen3_5MoeForConditionalGeneration` (`qwen3_5_moe`), multimodal
(image + video token ids), text side:

- **40 hybrid layers**: 30 `linear_attention` (GDN/KDA family: `in_proj_qkv`
  + `in_proj_a/b/z`, `A_log`, `dt_bias`, `convNd` k=4, ssm state f32) +
  10 `full_attention` (`full_attention_interval: 4`; GQA 16 heads / 2 KV,
  head_dim 256, `attn_output_gate`, partial rotary 0.25, **interleaved mrope**
  sections [11,11,10]).
- **MoE per layer**: 256 routed experts (top-8, `moe_intermediate_size` 512),
  **fused** into `switch_mlp.{gate,up,down}_proj` stacked tensors, plus a
  shared expert (512) with its own scalar gate.
- hidden 2048, vocab 248320 (embed/lm_head untied), max_position 262144,
  **1 MTP layer** (`mtp_num_hidden_layers`).
- Vision: SigLIP-class ViT (depth 27, hidden 1152, patch 16, spatial_merge 2,
  temporal_patch 2, 2304 positions) + merger → 2048.

## 1b. Official reference facts (from the model README + edge0 source, 2026-09-16)

Edge0 is **built for the `edge0` streaming framework** (MLX backend, Apple):
base = Qwen3.5/3.6-MoE-35B-A3B, int4 + Recover-LoRA + prerouter, "SSD expert
offload" — the same architecture class as this repo. Reference performance
(Mac mini M4 Pro 24 GB): **14.9-17.7 tok/s decode, 113/140 tok/s prefill,
2.9 GiB peak active**. Three source-level findings:

- **Runtime routing width K=4 — EXTERNAL inference, not checkpoint-verified**
  (epistemic status deliberately different from the byte-level format facts:
  source is the upstream `models/edge0_35b` module and README "256 / 4",
  against the config's trained width of 8). Consequences if wrong are large
  (all per-token byte estimates halve at 4). Mitigations: (a) the effective
  K is recorded at run time in the execution policy/log line — never
  implicit; (b) once a forward pass runs, K is empirically checkable under
  `norm_topk_prob` (if the model effectively routes 4, gate weights at
  ranks 5-8 should sit clearly below ranks 1-4) — that test upgrades this
  entry from inference to measurement.
- **Perf goal is TWO-STAGE (79's constraint on an unverified assumption)**:
  P0 target = the official BASELINE form (no prerouter). The README's
  14.9-17.7 tok/s does not state whether it includes the prerouter's +59%;
  **if it does, the P0-reachable bound is 14.9-17.7 ÷ 1.59 ≈ 9.4-11.1 tok/s**
  — both numbers recorded so a shortfall is attributable, not chased
  structurally. The +59% belongs to prerouter-as-a-feature, evaluated when
  and if the prerouter weights ship.
- **Prerouter/MTP absence is a RELEASE INCOMPLETE, not a permanent fact**:
  README Contents promises `prerouter_edge0_35b.safetensors`; neither it
  nor any MTP weights (despite `mtp_num_hidden_layers: 1`) are in the
  checkpoint. If the official release ships them, P0's positioning is
  re-evaluated rather than assumed final.
- **CORRECTION (2026-09-17): `prerouter_edge0_35b.safetensors` IS shipped**
  (138 MB, 99 F16 tensors: per-layer fc1 [512,2560] / fc2 [256,512] /
  linear_init [256,2560], ~33 layers). The original "not in the checkpoint"
  claim verified one scope (the shard index weight_map) and asserted about
  another (the directory): the prerouter, like the LoRA, is a separate file
  the index does not cover. MTP weights remain genuinely absent.
  **The +59% is a StreamingExperts-regime number**: prerouter predicts
  next-step routing to PREFETCH expert weights, so it pays where experts
  are NOT resident (small-VRAM devices streaming over PCIe — the
  mode.rs StreamingExperts branch). Under FullResident (16.88 GiB resident)
  there is nothing to prefetch; do not quote +59% for the resident path.
  P0 proceeds without prerouter — the
  framework-level fallback (no prefetch prediction) is the baseline the
  reference's "+59% with prerouter" is measured against.
- **Nibble order settled: LOW nibble first** (little-endian nibbles, the
  MLX `mx.gather_qmm` convention the source declares bit-compatibility
  with). Evidence: adjacent-smoothness discriminator on real rows —
  lag-1 msse/var ≈ 0.05-0.075 (low) vs ≈ 2.0 (high, the iid level);
  30-40× separation, unambiguous. Test vectors carry both orders for the
  record.

## 2. Quantization format (new to this repo)

Groupwise affine, group_size 64, verified layouts:

| tensor class | bits | storage |
|---|---|---|
| body (switch_mlp, shared_expert, linear_attn projs, embed, lm_head, vision) | int4 | `U32 [.., in/8]`, last-dim packed 8×int4 |
| router `mlp.gate`, `shared_expert_gate` | int8 | `U32 [.., in/4]` (verified: gate `[256, 512]` = 2048/4) |
| per-group params | — | `scales`/`biases` BF16 `[out, in/64]` |
| norms, `dt_bias`, `A_log`, `convNd`, `k_norm` | bf16 | plain |

Dequant convention — **settled 2026-09-16 by three metadata discriminators**
(mean ratios → integer histograms at two window scales → bitwise authority
test), pending only byte-level confirmation:

**Unified format, both bit-widths: per-group integer zero-point z, with
scales and biases each independently bf16-rounded from exact values**
(b\* = −z·s\*). This single model explains all six observations: int4 ratios
100% integer at ±0.05 (0.6% = double-rounding bound), 91% at ±0.02; int8
78% at ±0.30 (same distribution scaled to r≈150), 100% under loose windows;
bitwise b == bf16(−z·s) matching only 58-97% (b derives from exact s, not
the stored one); the exact −128 cluster (z=128 groups). z distribution:
mode 8 (~62%), tail 9-12, and a z=0 mode at ~40% of switch_mlp groups
(symmetric-clamping — for these, bias must be EXACTLY 0.0, the one
tolerance-free assertion).

**Consequences:**
- **Kernel form `w = s·q + b` — decided on checkpoint grounds, not
  numerical ones.** Numerically both forms are approximations under the
  double-rounding model, and `s·(q−z)` is the slightly TIGHTER one
  (error s·a·(q−z), exactly zero at the zero point, vs s·(a·q−c·z),
  never zero; z-recovery itself is reliable at ±0.05≪0.5). The deciding
  reason: **z is not stored in the checkpoint** — `s·q + b` needs no
  recovery step, no lookup, no recovery-bug surface, which is why any
  implementation reading this file uses it. (Recorded so a future kernel
  optimizer does not "fix" the form on numerical grounds and add a
  z-recovery step for no gain.)
- **Reference-decode assertion: agreement within 1 ULP, not bit-exact.**
  Rust may contract `s·q+b` into an FMA (one rounding) while the Python
  reference multiplies then adds (two roundings) — a last-bit difference
  that is a property of the CHECK, not of the format; a bit-exact
  assertion would misreport it as misalignment. The 1-ULP bound and its
  FMA provenance go in the assertion's comment. The one tolerance-free
  assertion stays: z=0 groups (40% of switch_mlp) must carry bias
  exactly 0.0.
- Discriminator-design lesson recorded: integer-detection windows must be
  DERIVED from the encoder's rounding model (independent double bf16
  rounding → ±0.6% relative), never chosen as a bare absolute (distorted
  by r scaling) or bare relative (degenerates at large r). A wrong window
  lets one dataset support opposite conclusions — both peers hit one
  window error each on this line.

Fused expert geometry (layer 0): `up_proj U32 [256, 512, 256]` → logical
[256 experts, 512, 2048]; `down U32 [256, 2048, 64]` → [256, 2048, 512];
`biases BF16 [256, 512, 32]` (32 = 2048/64 groups).

**LoRA — ON-THE-FLY, not merged (decided 2026-09-16 on measurement):**
`lora_edge0_35b.safetensors`, rank 16, FP16, 620 tensors covering
shared_expert (40×3), linear_attn (30×5: qkv/a/b/z + out_proj) AND all 10
full-attention layers' q/k/v/o (discovered in the byte audit). Options were
(a) merge at materialization → forces **2.625 GiB** of full-precision
residency on exactly the capacity-bound machines this adapter targets, vs
(c) keep int4 and apply `y = Wx + B(Ax)` at use time → **40.4 MiB** of A/B
residency, +16/in ≈ 0.8% FLOPs. 66× memory difference decides: **(c)**;
the int4 fused kernels take an optional per-tensor LoRA pair. This is the
regime framework applied: capacity-bound targets trade negligible compute
for memory.

### 2b. SCALE_BIAS_OVERHEAD = 12.5% of packed bytes (named, geometry-fixed)

`group_size = 64`, scales and biases both BF16 `[rows, in/64]`: two values ×
2 bytes per 64 weights = 4 B per 32 B packed (64 int4 = 32 B) → **12.5%**.
Every scale/bias byte budget (per-token traffic AND VRAM residency) derives
from THIS constant, never re-derived by hand. (Two hand derivations landed
on 7.1% — counting only one of s/b — and on a GB/GiB slip; they cancelled
by luck in the total but were wrong at the component level.)

Corrected VRAM residency (bytes as GiB): experts packed 15.00 + expert s/b
1.88 + static projections + their s/b 1.35 + embed/lm_head 0.5 = **18.73
GiB** weights; peak with state/activations ≈ 19.7 GiB — fits 24 GiB only
with experts resident as PACKED INT4 (BF16 expansion = 4×, out).

### 2c. CPU-side decode decomposition (measured, 2026-09-16/17)

moe-compute 88 ms/token = **43** fused-AVX2 port-bound compute (independently
confirmed: 503M MAC / 11.8 GMAC/s) + **7** per-call fixed overhead
(14.2 µs × 480, line-fit intercept) + **6** cold-rotation L2 misses
(rotating probe: 116 vs 103 µs/call) + **~25 in orchestration code**
(probe v2: per-layer-rate 14.25 MiB line-granular reads interleaved at the
model's real granularity moved matvec-only time by 4.7 µs/call — eviction
is real but small; ~93% of the residual is routing/dispatch/indexing
between calls). Probe v1's "+1%" was VOID — a chunks() unit bug touched
8.9 KiB instead of 570 MiB (intervention absent; caught by 79's
arithmetic pre-check: a +1.0 µs delta sits 16× BELOW the 15.6 µs/call
theoretical floor of the intended read — "an intervention that costs
nothing is an intervention that did not happen"). CUDA implication stands
on launch-overhead grounds (480 launches × ~5-10 µs ≈ the same order as
the 7 ms per-call overhead alone), with the ~25 ms orchestration residual
as supporting evidence, not the premise: prefer one fused kernel per
layer (router + experts + shared) over ~480 individual launches.

## 3. Reuse map


- **ff-glm** (closest precedent — glm5_next is also hybrid-GDN MoE + MTP):
  GDN layer math, MoE streaming (expert cache/host split/routing trace),
  MTP, streamed-transformer skeleton, admission-breakdown pattern, expert
  cache with dequantized entries (int4 source feeds the same BF16-entry cache).
- **ff-trellis** (clip/dinov2/dinov3): ViT implementation precedent for the
  vision tower; merger precedent from CLIP projector.
- **New**: the int4-group dequant path (analog of `ff-glm/src/fp8.rs`),
  fused-stacked expert layout, interleaved mrope, 8-bit gates, LoRA apply.

### 3b. Performance-mode auto-sensing (user directive 2026-09-17)

The adapter SENSES the best performance mode from hardware and RE-DERIVES
whenever user configuration changes — `mode.rs`. Unset VRAM budget defaults
to all VRAM minus a reserve; a user budget re-runs the whole derivation
(FullResident / StreamingExperts / HostOnly) against it. Byte arithmetic
flows from `SCALE_BIAS_OVERHEAD` and config geometry (the 18.73 GiB budget
is an OUTPUT of `WeightSizes::from_config`, not a hand-typed constant), and
every decision appends a provenance line so a run's mode is attributable
after the fact. The planner is pure — same inputs, same plan — and tested
against synthetic hardware profiles including the no-VRAM and forced-host
paths. **2026-09-18: the planner is now wired into `enable_gpu`** — an
EDGE0_GPU=full request probes real VRAM (mem_get_info), runs plan_mode
against the weighed bucket bytes, prints the provenance, and refuses
full residency if the planner selects anything else (previously the
upload simply OOM'd mid-way on a too-small card).

### 3c. Handoff notes for the next session (2026-09-17)

Three traps, all bit someone this session:

1. **Count the denominator before dividing.** The same 45 ms was divided
   by 480, 680, and 280 across three messages, none counted; three reworks
   followed. FIX INSTRUMENTATION FIRST: a per-token `synchronize()` counter
   (one atomic in the GPU runtime) ends this entire line of errors. The
   moe-compute timer covers router + shared expert projections too — post-
   batching it is ~7 syncs/layer (router 1, shared 4, expert_layer 2).
2. **Per-width x buffers fail on multiple live inputs of the same width.**
   The v1 fix (interleaved upload/launch) only works while ONE input per
   width is live; merging router+shared into the layer batch makes that 5
   live 512-wide inputs. The durable fix is per-SLOT buffers (one buffer
   per slot, or one large buffer sliced by slot offset) — interleaving is
   a bridge, not the destination. The failure mode is silent garbage, not
   an error.
3. **Determinism rules for device-side accumulation** (79's four): no
   atomicAdd across experts (run-to-run nondeterminism flips near-tie
   argmaxes); fold router weights into `inner` (4× cheaper than scaling
   down outputs, folds into the SiLU·mul kernel); down_proj needs a
   `y[row] += total` variant; same-stream sequential launches are
   deterministic by construction.
4. **"Counted" already has an implementation — call it.** Sizing a cache
   or expansion is one multiplication away from `bucket_bytes` (packed ×8,
   s/b ×2 for f32); the OOM'd cache was designed without consulting the
   component built two steps earlier for exactly that question. The rule
   "count before placing = count before dividing" lands by CALLING the
   tool, not by re-forming the habit.
5. **`ulimit -v` caps virtual address space** — mmap'd weights count in
   full even when cold; a kill under `-v` may be a VA false positive with
   RSS far below the cap. Check peak RSS before blaming the cache;
   `systemd-run --scope -p MemoryMax=` is the residency-correct tool.
   And slice BEFORE dequantizing stacked expert tensors (byte-slice one
   expert's span first; dequant-then-slice does 256× the work and looks
   identical in Python).

**Five dead constants (all died the same way — measured total ÷ unverified
count; do NOT divide moe time by anything again):** 162 µs (46/280 sync),
96 µs (46/480 sync), 19 µs ("measured" 1.5/80 — confounded with a removed
kernel), 54 µs/kernel (26/480), and the kernel-count model itself (moe
stayed 26 ms at 480→120 kernels AND 3→1 syncs/layer — invariant across
both). Next-session order (79, final): (1) FIVE sub-timers inside one MoE
layer (gate/up enq+exec, 2×dtoh, host silu, htod, down) — they must sum to
650 µs/layer; the gap is information. (2) THEN wire GPU silu (one line;
kernel compiled, `batched_gate_y/up_y` → contiguous `batched_inner_y`
already exists). If moe drops to ~3 ms, done; if only to ~20 ms, STOP and
read the sub-timers — the midsection round trip may be a trigger (217
µs/round-trip implied is 10× PCIe-typical; pipeline stall is the suspect,
not transfer). Launch counter live; sync ≈ 0 measured.

**Reference generation DIAGNOSED AND KILLED (2026-09-17, 79's sampling).**
It was not computing — it was thrashing: `utime 28% / stime 72%` (pure
Python compute reads ~100% utime; the kernel share was page faults —
4.2·10⁹ minflt ≈ 17 TB first-touch, consistent with stime 5614 s), with
majflt 93k and VmSwap 2.1 GiB. Mechanism: the 8.2 GiB static cache's
cyclic working set exceeded available memory → hit rate pinned at 0 → every
"cache" hit re-materializes GBs and evicts the next victim — the cache
degenerated into a generator ("fits ≠ usable"; fourth cell of
slow → OOM → slice-layer → working-set). Top offender: **lm_head — one
248320×2048 f32 materialization is 2.03 GB used ONCE per token.** The fix
when the token-id comparison is resumed: drop lm_head from the cache
(chunk the matvec directly over the int4 payload, or dequant-chunk once),
shrinking the cache 8.2 → 6.2 GiB. Process killed and its 9.2 GiB RSS +
2.1 GiB swap returned (5e needs the host for D2).

**Diagnosis tool for the next session (replaces /proc/PID/stack, which
needs CAP_SYS_ADMIN and reads the KERNEL stack — useless for user-space
Python):** `awk '{print $10,$12,$14,$15}' /proc/PID/stat` gives
minflt/majflt/utime/stime, zero-dependency and unprivileged — **the
utime:stime ratio separates "computing" (~100% utime) from "faulting"
(kernel-dominated).** Sampling produces information; waiting does not.

**Slotx trap (per-slot principle, instance #2):** down inputs differ per
expert — a shared-x kernel arg is correct ONLY for gate/up; the down
variant (`edge0_batched_gemv4_slotx`) indexes x by slot from a contiguous
[slots,512] scratch. Instance #1 was four inners aliasing one per-width
buffer. Both are "reuse-by-width/share fails with multiple live
same-width inputs" — contiguous per-slot scratch is the terminal form.

Remaining sequence after the above: layer glue 11.4 ms (allocation
counter first — it is the signature of alloc/copy, not compute, at 300×
lower op density than the SIMD recurrence) → GDN state resident on device →
30 tok/s needs only 28.5 ms/token (batched GEMV at its promised 6.5 ms
plus today's floor reaches it with margin). Also pending: token-id
comparison vs the cached Python reference generation (running in
background), and the reference-decode acceptance hook uses `s·q+b` in f32
(never `q−z`; the biases tensor is authoritative — z is not stored).

## 4. Plan

- **P0 text-only** (crate `ff-edge0`, `--adapter edge0` under `ff text`):
  config.rs, int4.rs (CPU fused dequant+matvec first — same shape as
  `fused_block_fp8_matvec`, plus the optional LoRA pair), model.rs
  (GDN/attn/MoE/MTP), greedy decode, `ff models inspect` compatibility.
  **mrope: the position-index layout and rope interface are designed
  MULTIMODAL from P0 (image/video position segments reserved, empty in
  P0) — position encoding is the text/vision shared layer; a text-only
  rope now is a P1 rewrite later (79's review constraint, accepted).**
  CUDA = dequant-to-BF16 entries (no new kernels required for P0: reuse
  the GLM entry-cache pattern; LoRA applies to entries at cache-fill time
  on CUDA, on-the-fly on CPU).
- **P1 vision**: ViT tower + merger + mrope positions, image→text.
- **P2**: video tokens, CUDA fused int4 kernels (only if P0's dequant-to-BF16
  path shows as a wall), speculative draft if a draft model exists.

Estimated size P0+P1: 6–8k lines (about half of ff-glm; the skeleton,
streaming and cache machinery are adaptations, not inventions).

## 5. Verification hooks (decided before code, per house rules)

- int4 path: unit-test one full projection row against a hand-decoded
  reference from the checkpoint bytes (pick a tensor, decode in Python
  once, assert Rust matches bit-wise on bf16).
- routing/1-token smoke: same-input determinism across runs (established
  control tool), plus `--routing-trace` parity with the trace format so
  the existing offline analysis tooling applies.
- acceptance P0: greedy decode produces stable output for a fixed prompt;
  admission uses `host_device_memory_is_unified`-aware paths from day one
  (this branch is off `main`, so A's merged-pool logic lands via rebase
  before any UMA testing — no split-axis re-introduction).

**Acceptance re-run (2026-09-18, post-review-fixes): CLOSED, green.**
Oracle regenerated after its KV-cache and final_norm fixes (per-kv-head
append, final_norm applied before lm_head): prompt 1's 8 ids reproduced
unchanged; prompt 2 diverges from the pre-fix archive at token 16 (the
archive was accumulated under the KV bug — discarded, json re-baked).
With final_norm corrected to plain RMSNorm on both Rust paths (the
checkpoint ships unshifted norm weights — no MTP, conv1d [dim,k,1] — per
the official sanitize()), the extended gate passes end to end: LoRA dump
worst max_rel 4.9e-6, 8/8 and 32/32 token ids exact vs the fixed oracle,
CPU/GPU ids identical.

## 6. Mega-kernel design (per-layer persistent CTAs) — 2026-09-17, for review

**Problem.** Post-fusion the decode step is ~16 kernels/layer, every kernel
4-56 us isolated, sum ~7 ms/token — but the pipeline measures ~15 ms. The
2x is dependency serialization: on one stream each kernel waits for FULL
completion of its predecessor (tail drain + launch-to-launch gap), and no
launch-count reduction removes it. Graph replay hits the same per-node
floor (~9 us/node measured).

**Shape.** One kernel per layer type — `gdn_mega`, `attn_mega`, `moe_mega`
— each a persistent-CTA loop over the layer's phases with a grid-wide
barrier between phases. Decode step = 3 launches/layer = ~120 nodes/token;
expected wall = traffic-bound ~31 us/layer + barrier overhead ≈ 2.4-3.6
ms/token (~250-330 tok/s), lm_head+argmax stays separate (~0.45 ms).

**Barrier strategy — arrival counters, NOT cooperative launch.**
`grid.sync()` requires cudaLaunchCooperativeKernel and a grid sized to
guaranteed co-residency (occupancy-bounded; a register-pressure change
silently breaks launch). Instead: a global arrival counter per phase —
each CTA atomically increments on phase completion and spins until the
counter reaches gridDim (with `__threadfence()` before arrival, acquire
loads in the spin). Costs ~2-4 us per barrier on a 128-SM part at our grid
sizes (~128-256 CTAs), needs no special launch mode, works inside graphs,
and degrades gracefully (if the grid exceeds residency, the spin still
makes forward progress only if all CTAs are resident — SO the grid MUST be
sized ≤ co-residency anyway; the counter just avoids the launch-mode
fragility and lets us overlap phases later). Grid sizing is asserted at
upload from cudaOccupancyMaxActiveBlocksPerMultiprocessor, not assumed.

**Determinism (§3c(3) survives).** Every phase keeps its current
reduction shape: same partials layout, same fixed-order group sums, same
thread-0 row finalize. The barrier only orders phases, exactly as stream
order does today — the arithmetic per phase is bit-identical to the
current kernels. The router's data-dependent expert ids do not branch the
kernel shape (same work, different row indices), so all CTAs traverse the
same barrier sequence unconditionally.

**Phase maps.** `gdn_mega`: [rmsnorm | group-GEMV(qkv,z,b,a)+lora | conv |
heads | out_proj group | add+norm] — 6 barriers. `attn_mega`: [rmsnorm |
group(q,k,v) | qk-norm+rope+KV append | scores+softmax+gate | o_proj |
add+norm] — 5. `moe_mega`: [router+topk | batched gate/up | slotx_silu
down | shared silu-down | combine] — 4. Work per phase is assigned by CTA
range (blockIdx-strided over the phase's row space, same as today's
grids); CTAs idle through phases with no rows (barrier-only).

**Fallbacks if barrier cost disappoints:** (a) merge only the MoE phases
(highest phase count per byte moved); (b) warp-specialized
producer/consumer within one kernel (no grid barrier; complexity ~2x);
(c) keep the current kernels but overlap tail-drain via two streams +
events — rejected as primary: reintroduces the event machinery that broke
capture isolation.

**Open items for review:** barrier spin cost on contended L2 (measure
with a microbench before committing to the layout); whether lm_head
joins the last layer's mega-kernel (saves one more node, costs an
occupancy-constrained 248K-row phase inside a resident grid — probably
not); MTP interplay (draft layer wants the same mega treatment).

### 6.1 Measured scaffolding results (2026-09-17; archived values — the one-off harness is not in the tree)

| item | measured |
|---|---|
| barrier, 64/128 CTAs | 1.0 us |
| barrier, 256 CTAs | 2.1 us |
| barrier, 768 CTAs (full residency) | 3.1 us |
| stub register union (3 phase families) | 40 regs |
| stub shared union (~3 KB) | 6 blocks/SM |
| co-residency cap (128 SMs) | 768 CTAs |
| streaming phase inside 256-CTA resident grid | 3.2 TB/s (L2-resident bench buffer; DRAM-bound weights bench 817 GB/s standalone) |

Implications: at ~5 barriers/layer the barrier tax is ~0.1-0.3 ms/token —
noise against the 2.4-3.6 ms estimate. The footprint union does not bind
occupancy (768-CTA headroom vs 128-256-CTA grids). The spin cap (50M
iterations then __trap) is in place and validated at exactly-full
residency; keep it env-gated (`cap > 0`) for production, always on in
development. Remaining §3c(3) note: the `y[row] += total` down variant is
absorbed by moe_mega's combine phase (routed downs write slot-local y,
combine accumulates into hidden — the +=-into-shared-hidden form only
matters if combine itself is fused away, which the phase map does not
require).

### 6.2 moe_mega landed + the attribution open question (2026-09-17)

moe_mega is in place (EDGE0_MEGA=1; moe_closed remains the default
fallback): one launch for router|ss|sg|su -> top-k -> routed gate/up ->
slotx_silu down + shared down -> combine, four barriers, row math ported
verbatim. Acceptance 8/8 green; GDN warp-reduction divergence 4.5e-5 (archived measurement).

Measured with 20 whole decode steps enqueued with NO intermediate syncs,
one sync at the end (archived; the one-off harness is not in the tree): **14.4 ms/token of GPU execution; ~2 us per
launch CPU submit**. The wall is invariant to: kernel count (640 -> 440
changed nothing; MoE 6 -> 1 kernels changed nothing), clocks (boost
throughout), L2 residency (cold-L2 flush changed nothing), and allocation
scatter (512 scattered 1 MiB reads = 689 GB/s, same as one slab — TLB
hypothesis falsified). Isolated per-kernel benches sum to ~7 ms/token but
the same kernels take 14.4 in sequence.

The per-boundary cost between DIFFERENT kernels in sequence is real but
none of the host-side instruments can decompose it further — every sync
absorbs upstream drain, every isolated bench changes the regime. Next
instrument: CUPTI/nsight per-kernel timing on a benchmark window. The
structural fix proceeds regardless: gdn_mega and attn_mega next (3
launches/layer total), then re-measure — if the wall is boundary-cost,
it must move; if it doesn't, the cost is inside the phases and CUPTI
names the phase.
