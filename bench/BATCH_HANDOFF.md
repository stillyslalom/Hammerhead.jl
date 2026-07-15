# Handoff: portable batch-memory architecture for the CUDA backend

## Goal

Port the generic batch-memory lifecycle now implemented and hardware-validated
in `ext/HammerheadAMDGPUExt.jl` to `ext/HammerheadCUDAExt.jl`, then validate it
on the CUDA development box. Do not copy AMD-specific rocFFT workarounds or HIP
pool behavior verbatim.

The CUDA extension currently has a per-engine memory cap. A multipass
`PIVWorkspace`, however, retains one engine per window configuration, so sizing
each engine independently can overcommit the aggregate cache. Plain
`run_piv(...; workspace = nothing)` calls also leave stage-engine cleanup to
Julia GC. The portable architecture fixes both cases while preserving large
batches when the workload and GPU have room.

## What is already in core

`src/pipeline.jl` now defines the backend hook:

```julia
release_piv_engines!(::_AbstractHammerheadBackend, engines, workspace) = nothing
```

`piv_pass` calls it after correlation, validation, and any UQ sweep. AMDGPU
implements the hook by releasing every stage batch when no workspace is
supplied. CUDA can add its own extension method without another core change.

The AMDGPU extension is the architectural reference for:

- one cache manager shared by all window configurations in a workspace;
- exact per-window byte accounting for `CA + CB + Rt + uqdcs_d`;
- batch targets bounded by actual `njobs`;
- fair sharing after the workspace discovers multiple configurations;
- retaining all configurations when their aggregate fits;
- least-recently-used eviction when it does not;
- immediate release of batch arrays and FFT plans before replacement;
- explicit stage cleanup when no reusable workspace owns the engine.

The policy is workload-driven. It does not assume three passes, a particular
precision, image size, card model, or VRAM capacity.

## CUDA-specific findings

### Keep `inv(engine.fwd)`

Do **not** copy AMDGPU's direct `plan_bfft!` workaround. CUDA.jl 6.2's
`plan_inv(::CuFFTPlan)` constructs the inverse plan from stored dimensions and
does not allocate a full batch-sized `CuArray`. The AMDGPU implementation did,
which was the source of a monotonic VRAM staircase on the RX 6800 XT.

The existing CUDA code is therefore appropriate:

```julia
engine.fwd = plan_fft!(engine.CA, (1, 2))
engine.bwd = inv(engine.fwd)
```

Re-check this on the CUDA.jl 5 version in the compatibility matrix, but change
it only if source inspection or hardware memory tracing demonstrates a real
temporary allocation.

### Avoid default `CUDA.reclaim()` in the hot path

CUDA.jl 6.2's default `CUDA.reclaim()` uses `RECLAIM_DROP`: it drops task-local
library state, runs full Julia GC, purges handle caches, synchronizes, and trims
the pool. Calling it for every LRU eviction would defeat plan/buffer reuse.

Prefer:

1. synchronize before releasing buffers that may still be in flight;
2. call `CUDA.unsafe_free!` on evicted `CuArray`s;
3. call `CUDA.unsafe_free!` on the forward plan and on `engine.bwd.p` (the
   inverse is an `AbstractFFTs.ScaledPlan`);
4. let CUDA's stream-ordered pool retain freed blocks for fast reuse;
5. use a heavier reclaim level only as an allocation-failure fallback or an
   explicitly measured requirement.

If synchronization-only reclamation is needed, CUDA.jl 6.2 exposes
`CUDA.reclaim(CUDA.RECLAIM_SYNC)`. Verify availability before relying on it
because Hammerhead supports CUDA.jl 5 and 6.

### Include reusable CUDA pool slack in the budget

On stream-ordered CUDA allocators, physical `CUDA.free_memory()` excludes
memory reserved by CUDA's pool even when part of that pool is currently free
and immediately reusable. CUDA.jl 6.2 exposes:

```julia
CUDA.cached_memory()  # bytes reserved by the pool
CUDA.used_memory()    # reserved bytes currently live
```

The allocatable baseline should therefore include pool slack:

```julia
pool_slack = CUDA.cached_memory() - CUDA.used_memory()
allocatable = CUDA.free_memory() + pool_slack
```

When deciding whether cached Hammerhead engines can be replaced, add their
tracked batch bytes as reclaimable. Continue reserving headroom for image
staging, deformation contexts, masks, external allocations, and driver/library
state. The current 70% aggregate fraction is a starting policy, not a
per-engine fraction.

CUDA.jl may return `missing` for pool metrics on a non-stream-ordered allocator.
CUDA.jl 5 may expose a different surface. Add a small compatibility helper that
falls back to zero pool slack when the API or metric is unavailable; do not
make CUDA 6.2 APIs an unconditional extension-load requirement.

## Recommended implementation sequence

1. Add a CUDA cache manager and per-engine `cache`/LRU-stamp fields, mirroring
   the AMDGPU structure but using CUDA types and APIs.
2. Change `_cuda_batch_cap` to accept `njobs` and return the final stage target
   directly. Update all four call sites:
   `process_windows!`, `uncertainty_sweep!`, `accumulate_planes!`, and
   `ensemble_analyze!`.
3. Preserve `_CUDA_BATCH_DEFAULT = 8192`, `_CUDA_MIN_BATCH = 256`, and the 512
   quantum. These remain occupancy/anti-thrash bounds; the byte budget chooses
   the safe target beneath them.
4. Compute one aggregate cache budget from total/free VRAM, CUDA pool slack,
   and tracked reclaimable engine bytes.
5. Fair-share that budget across the window configurations discovered in the
   workspace. Always clamp by real job count. Small/single-pass jobs should
   retain their full useful batch.
6. Evict cold batch buffers only when active buffers plus the requested target
   exceed the aggregate budget. Release the current engine first when its
   target changes so replacement allocation never overlaps the old batch.
7. Implement `release_piv_engines!(::_CUDABackend, engines, workspace)` so a
   no-workspace pass synchronizes, releases its batch arrays/plans, and returns
   without waiting for GC. Do not perform default `CUDA.reclaim()` here.
8. Keep `HAMMERHEAD_CUDA_BATCH` as a per-stage benchmarking/debug override, but
   allow cache eviction around that target. An override larger than one stage
   can physically fit may still spill or fail; that is intentional diagnostic
   behavior.
9. Keep AMDGPU and CUDA policy behavior aligned, while allowing the memory
   queries, pool accounting, release calls, plan construction, and reclaim
   behavior to remain vendor-specific.

## Release helper checklist

The CUDA release helper should free and reset every batch-dependent device
array currently allocated in `_ensure_batch!`:

- `CA`, `CB`, `Rt`;
- `meanA_d`, `meanB_d`;
- `vals_d`, `locs_d`, `out_d`;
- `uqstats_d`, `uqmeans_d`, `uqdcs_d`;
- `origins_d`;
- `fwd` and the underlying inverse plan `bwd.p`.

Reset host batch scratch (`out_host`, `uqstats_host`, `origins_host`), set plans
to `nothing`, and set `bs = 0`. Do not release fixed configuration arrays
(`apod_d`, `gain_d`, `phase_filter_d`) or image staging buffers during ordinary
LRU batch eviction; they are much smaller or tied to the reusable engine's
configuration/image size.

## Compatibility risks to check

- Hammerhead declares CUDA compat `"5, 6"`. Verify `CUDA.unsafe_free!`,
  `CUDA.cached_memory`, `CUDA.used_memory`, reclaim levels, and plan wrapper
  structure on both supported majors.
- In CUDA.jl 6.2, `engine.bwd` is a `ScaledPlan` and the releasable plan is
  `engine.bwd.p`. Confirm CUDA.jl 5 uses the same wrapper.
- A stream-ordered free may be asynchronous. Synchronize before releasing
  buffers last used by a completed stage; do not assume assignment plus Julia
  GC is timely enough.
- CUDA's cuFFT handle cache is useful. Explicit plan release should return
  handles to that cache; aggressive reclaim/purge may destroy them.
- The byte model intentionally omits small per-window scalar buffers and relies
  on memory-fraction headroom. Validate the low-water mark rather than treating
  the estimate as exact total process memory.
- Ensemble plane accumulators and deformation/image staging are not batch-cache
  entries, but their live memory reduces `free_memory()` and must therefore
  influence the available aggregate budget.

## Validation on the CUDA development box

Use `bench/CUDA` with the local Hammerhead checkout developed into it.

### 1. Precompile and correctness

```bash
julia --project=bench/CUDA -t 4 bench/gpu_validate.jl cuda
```

Expected tolerances from prior RTX 2000 Ada validation are approximately
`1e-15` for Float64 vectors, `1e-8` or better for UQ, and bitwise equality for
ensemble UQ. Exercise single-pass, multipass, masks, phase correlation,
ensemble, device UQ, and hybrid UQ.

### 2. Production-size automatic policy

The sweep accepts `auto` as a batch specification:

```bash
julia --project=bench/CUDA -t auto bench/gpu_batch_sweep.jl cuda 4096 Float64 auto high 3
julia --project=bench/CUDA -t auto bench/gpu_batch_sweep.jl cuda 4096 Float32 auto high 3
```

Observe dedicated and shared memory externally as well as the reported
low-water mark. Memory should converge after configuration discovery instead
of increasing monotonically across repeated calls.

### 3. Fixed-batch comparison

```bash
julia --project=bench/CUDA -t auto bench/gpu_batch_sweep.jl cuda 4096 Float64 8192,4096,2048,1024 high 3
julia --project=bench/CUDA -t auto bench/gpu_batch_sweep.jl cuda 4096 Float32 8192,4096,2048,1024 high 3
```

The old RTX 2000 Ada measurements were:

| fixed batch | Float64 4096² | Float32 4096² |
|-------------|----------------|----------------|
| 8192 | 33.3 s, spilling | 2.98 s |
| 4096 | 13.0 s | 3.63 s |
| 2048 | 11.3 s | 3.71 s |
| 1024 | 7.47 s | 3.85 s |

These are comparison points, not constants to encode. The automatic cache
should adapt to the device, precision, FFT sizes, job counts, pass schedule,
other live Hammerhead buffers, and external VRAM pressure.

### 4. Workspace lifecycle cases

Verify both ownership modes explicitly:

- repeated `run_piv` calls with one `piv_workspace(; backend = :cuda)` should
  converge to stable batches and VRAM usage, then reuse them;
- repeated multipass calls with `workspace = nothing` should release stage
  batches at each pass boundary and return close to baseline VRAM after the
  call;
- single-pass/small-window workloads should retain large batches instead of
  being penalized by multipass-oriented constants;
- schedules with two, three, and more distinct window configurations should
  remain within the aggregate budget without assuming a fixed pass count;
- changing image size or precision should clear the workspace cache safely.

### 5. Performance interpretation

Separate first-run discovery/plan compilation from steady-state sequence
performance. Report at least:

- selected batch per configuration after convergence;
- live/reserved/free VRAM;
- warmup/discovery time;
- steady-state time with workspace reuse;
- no-workspace time;
- CPU speedup and numerical deltas.

If LRU eviction occurs every pair, inspect whether fair sharing is converging
after all configurations are discovered. Do not add a card-specific ceiling
until the generic budget, pool accounting, and cache lifecycle have been
verified.

## AMDGPU reference results

The generic architecture was validated on an RX 6800 XT at 4096² high effort.
After configuration discovery, Float64 converged to 1024/4608/8192 batches,
used 7.99 GiB live, left 7.80 GiB dedicated VRAM free, and ran in 4.99 s
steady-state. Float32 ran in 3.29 s with 6.62 GiB free. Plain no-workspace
multipass calls returned close to baseline VRAM after each call.

The full AMDGPU validation remained within about `8e-15` of CPU for Float64
vectors and `5e-15` for UQ; ensemble UQ was bitwise equal. These results prove
the architecture, not the CUDA implementation.
