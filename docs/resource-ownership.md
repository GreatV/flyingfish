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
| GLM admission safety | allocator slack against modelled-peak error | pool-proportional, charged once in the phase model | `GLM_ADMISSION_SAFETY_BYTES` baked into every phase's device reserve / CPU safety term |
| GLM host promotion reserve | headroom for weight-source promotion | (dead — recorded, never charged; A3) | `host_promotion_reserve_bytes`, set in `resource_policy/glm.rs`, read nowhere |
| H3 device residency reserve | activation/workspace slack around retained weights | the request's `additional_host_allowance_bytes` under the unified fold; the phase `device_reserve_bytes` stamp on discrete | cli/resource.rs allowance + phase stamp + `automatic_device_cache` headroom |
| FA backend workspace floor | cuDNN/FlashAttention workspace | should scale with sequence geometry, not a flat 3584 MiB | `DEFAULT_FLASH_BACKEND_WORKSPACE_MIB` |
| non-FA backend workspace | attention score workspaces | geometry-scaled | `DEFAULT_NON_FLASH_BACKEND_WORKSPACE_MIB` (1536 MiB) |

## Residency (bytes held, by reclaimability)

| class | reclaimable? | charged for fit? | field |
|---|---|---|---|
| process allocations, pinned buffers, device tensors | no | yes | `required_*` / `optional_*` |
| mmap'd shard residency (page cache) | yes (kernel drops under pressure) | **no** — telemetry only (A7-2) | `reclaimable_host_bytes` |
| H3 mapped weight residency | yes | same rule (pending) | `mapped_weight_residency_bytes` (assumption axis) |

## Rule

Admission answers "will it OOM", not "will it thrash". Reclaimable residency
is reported (it dominates delivered supply when evicted), never charged. A
fixed-byte reserve must declare what it protects and scale with the relevant
quantity (pool size or request working set), and the scaled value is recorded
in the selection sidecar.
