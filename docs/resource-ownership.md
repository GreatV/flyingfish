# Resource reserve and residency ownership map

Working document for the A7 admission-calibration work on branch
`unified-memory-admission`. It maps every reserve/residency charge to its
single owning ledger, so that admission changes stop the
three-homes-two-families whack-a-mole we hit while landing the unified-memory
fold (phase stamp → estimate allowance → sizing headroom took three rounds to
align).

## Reserves (bytes deliberately kept free)

| reserve | protects | owning ledger (target) | current homes (legacy) |
|---|---|---|---|
| GLM admission safety | allocator slack against modelled-peak error | **scaled: min(1 GiB, axis_total/20)** — device total on CUDA, host total on CPU, min of both totals when unified; recorded in the sidecar. Crossover: 20 GiB pools, below which the reserve relaxes (deliberate, pinned) | `GLM_ADMISSION_SAFETY_BYTES` (declared contract constant, identity) |
| GLM host promotion reserve | headroom for weight-source promotion | **alive and gating** promotions (`resource_policy/glm.rs` capacity check); **scaled the same way** (min(1 GiB, pool_total/20)) — the pre-A7 `max(1 GiB, pool/20)` never shrank and grew with the pool | the *field* `host_promotion_reserve_bytes` is still write-only (A3) |
| H3/generic device residency reserve | activation/workspace slack around retained weights | **still fixed 1 GiB** — intentionally NOT scaled yet: the generic adapters (minicpm/music/trellis) and H3 keep `DEVICE_RESIDENCY_RESERVE_BYTES` flat. This is a *known two-family inconsistency* introduced by scaling GLM only; **trigger to fix: any sub-20 GiB pool running those adapters**, where the flat GiB decides admission | cli/resource.rs allowance + phase stamp + `automatic_device_cache` headroom |
| FA backend workspace floor | cuDNN/FlashAttention workspace | **deferred**: scale with sequence geometry, but no ground truth exists (FA never ran on the Orin pool) — deferred deliberately rather than guessed | `DEFAULT_FLASH_BACKEND_WORKSPACE_MIB` |
| non-FA backend workspace | attention score workspaces | geometry-scaled | `DEFAULT_NON_FLASH_BACKEND_WORKSPACE_MIB` (1536 MiB) |

## Residency (bytes held, by reclaimability)

| class | reclaimable? | charged for fit? | field |
|---|---|---|---|
| process allocations, pinned buffers, device tensors | no | yes | `required_*` / `optional_*` |
| mmap'd shard residency (page cache) | yes (kernel drops under pressure) | **no** — telemetry only (A7-2) | `reclaimable_host_bytes` |
| H3 mapped weight residency | yes | same rule (pending) | `mapped_weight_residency_bytes` (assumption axis) |

## Follow-ups with triggers (no dates)

- **FA backend workspace geometry scaling** — trigger: a machine where
  flash-attention actually runs on a small pool (it never ran on the Orin, so
  no ground truth exists to calibrate a scaled constant).
- **Generic adapters' flat 1 GiB device residency reserve** (H3 included) —
  trigger: any sub-20 GiB pool running minicpm/music/trellis, where the flat
  GiB decides admission. On such pools GLM now reserves 5% while they reserve
  1 GiB of the same physical pool.
- **Discrete-path axis selection for reserve scaling is code- and unit-test
  covered only, not live-measured** — trigger: a machine whose host and device
  totals straddle the 20 GiB crossover (e.g. a 16 GiB card with a 62 GiB
  host), where the two axes would give different reserves. On the 4090 both
  axes clamp to 1 GiB, so the local baseline cannot tell them apart.
- **The unified-fold retention guard fix (`b3d3647`) has no observable effect
  on our only unified machine** — its justification is correctness (the flag
  conflated "no device" with "device shares host memory"), not performance:
  zero retention was actually faster in one measurement (1393 s vs 1555 s,
  budgets moved between runs so wall clocks are not directly comparable). Do
  not cite it as a performance fix.

## Worked claim, with its arithmetic

"Under the final accounting this pool cannot afford device weight retention."
Numbers from the Orin tip run (h3-smoke, `20260917-orin-final-h3`): measured
device free at selection 5.53 GiB; folded host requirement 4.70 GiB plus the
1 GiB residency reserve = 5.70 GiB; 5.70 > 5.53, so the automatic device
cache sized to zero. This conclusion previously appeared as an artifact of a
double-charged reserve (fixed in `821f51b`); it now stands with single
charging, but any future reader should re-check the arithmetic rather than
trust the sentence.

## Rule

Admission answers "will it OOM", not "will it thrash". Reclaimable residency
is reported (it dominates delivered supply when evicted), never charged. A
fixed-byte reserve must declare what it protects and scale with the relevant
quantity (pool size or request working set), and the scaled value is recorded
in the selection sidecar. Reserves scale with the pool's TOTAL bytes, never
the instantaneous available view — a busier machine does not get a smaller
margin, and recorded reserves stay comparable across runs. Exception: legacy
records and unprobed captures lack totals, so the fallbacks reach the
available views there; reserves recorded against those fallbacks are NOT
comparable across runs, which is exactly the data needing calibration.
