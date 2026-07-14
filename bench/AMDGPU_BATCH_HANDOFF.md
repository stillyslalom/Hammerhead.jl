# Handoff: memory-aware batch sizing for the AMDGPU backend

**Goal:** port the memory-aware sub-batch cap that landed in the CUDA extension
to `ext/HammerheadAMDGPUExt.jl`, then validate + tune it on the RX 6800 XT.
This work was scoped and validated on an NVIDIA RTX 2000 Ada (16 GiB); AMDGPU
is not installed on that box, so the AMDGPU half was deliberately deferred to a
machine with the hardware.

## Why (the bug being fixed)

The device sub-batch size was a fixed constant (`_AMDGPU_BATCH = 8192`,
`_CUDA_BATCH = 8192`). On a 16 GiB card at 4096² `effort = :high` in **Float64**
this **spills VRAM to shared/system memory** and is catastrophic. Measured on
the RTX 2000 Ada:

| batch | FP64 4096² time | FP32 4096² time |
|-------|-----------------|-----------------|
| 8192 (old fixed) | **33.3 s** (spilling, free VRAM → 0) | 2.98 s (fits, best) |
| 4096 | 13.0 s | 3.63 s |
| 2048 | 11.3 s | 3.71 s |
| 1024 | **7.47 s** (optimum) | 3.85 s |

FP32 and FP64 want **opposite** batch sizes, so a single constant can't win.

**Root cause (important — not what I first assumed):** the spill is the
correlation buffers (`CA + CB + Rt`) at the **first pass's `fft_size`**, *not*
cuFFT/rocFFT plan workspace. `effort = :high` runs windows 128→64→32 and
`padding = true` doubles each, so the first pass uses a **256×256 FFT** — 16× the
pixels of the final 32×32 window's 64×64 FFT. In Float64 that first pass alone
is ~20 GiB of buffers at bs=8192, over a 16 GiB card. cuFFT workspace for our
power-of-two Z2Z/C2C sizes is ~0 (verified via `cufftGetSize` → 0 MiB);
**confirm the rocFFT equivalent is also ~0** (see step 4) — if rocFFT allocates
real workspace, the byte model below must add it.

## What shipped on CUDA (mirror this exactly)

`ext/HammerheadCUDAExt.jl` — the pattern to copy:

1. Constants (replace the single `_AMDGPU_BATCH = 8192`):
   ```julia
   const _AMDGPU_BATCH_DEFAULT = 8192
   const _AMDGPU_MEM_FRACTION = 0.7    # of free VRAM budgeted for batch buffers
   const _AMDGPU_MIN_BATCH = 256       # floor: keep enough windows for occupancy
   const _AMDGPU_BATCH_QUANTUM = 512   # quantize the cap so free-VRAM wobble
                                       # doesn't thrash _ensure_batch!
   ```

2. Two helpers, placed **after** the `_AMDGPUCorrelationEngine` struct
   definition (they reference the struct type in their signatures — placing
   them before it fails precompile with `UndefVarError`, which is exactly how
   the CUDA port broke on the first try). In the CUDA ext they sit right after
   `_correlation_apod(engine) = engine.apod_d`:
   ```julia
   _amdgpu_bytes_per_window(engine::_AMDGPUCorrelationEngine{T}) where {T} =
       prod(engine.fft_size) * (2 * sizeof(Complex{T}) + sizeof(T)) +
       2 * max(engine.wsize...)^2 * sizeof(T)

   function _amdgpu_batch_cap(engine::_AMDGPUCorrelationEngine{T}) where {T}
       ov = get(ENV, "HAMMERHEAD_AMDGPU_BATCH", "")
       isempty(ov) || return parse(Int, ov)
       bpw = _amdgpu_bytes_per_window(engine)
       budget = (AMDGPU.free_memory() + engine.bs * bpw) * _AMDGPU_MEM_FRACTION
       cap = floor(Int, budget / bpw)
       cap = (cap ÷ _AMDGPU_BATCH_QUANTUM) * _AMDGPU_BATCH_QUANTUM
       return clamp(cap, _AMDGPU_MIN_BATCH, _AMDGPU_BATCH_DEFAULT)
   end
   ```
   - `bytes_per_window` must match what `_ensure_batch!` actually allocates
     per batch element. In CUDA that's `CA + CB` (`Complex{T}`), `Rt` (`T`), and
     the UQ ΔC cache `uqdcs_d` (`2·mm²·T`, `mm = max(wsize...)`). **Re-derive
     from the AMDGPU `_ensure_batch!`** in case the buffer set differs — it
     should be identical (the two exts are line-for-line mirrors), but verify.
   - The `+ engine.bs * bpw` term adds back this engine's own currently-resident
     buffers, because `_ensure_batch!` frees and reallocates them; without it,
     repeated calls across a sequence ratchet the cap down. On the first call
     `engine.bs == 0`, so the term is correctly zero.

3. Replace the four `bs = min(_AMDGPU_BATCH, …)` call sites (currently at
   `ext/HammerheadAMDGPUExt.jl` lines **281, 368, 439, 499** —
   `process_windows!`, `uncertainty_sweep!`, `accumulate_planes!`,
   `ensemble_analyze!`) with `bs = min(_amdgpu_batch_cap(engine), njobs)` /
   `… length(jobvec))`. `engine` is in scope at all four.

## API to verify (I could NOT check these — AMDGPU wasn't installed)

- **`AMDGPU.free_memory()` / `AMDGPU.total_memory()`** — confirm these exist and
  return bytes. On CUDA.jl 6.2 the names are `CUDA.free_memory` /
  `CUDA.total_memory` (NOT `available_memory` — that doesn't exist). AMDGPU.jl
  may spell it differently (possibly under `AMDGPU.Runtime.Mem` or via HIP
  `AMDGPU.Mem` / `hipMemGetInfo`). Fix the two call sites in the helper and in
  `bench/gpu_batch_sweep.jl` (the `amdgpu` branch already assumes
  `AMDGPU.free_memory` / `AMDGPU.total_memory` — adjust if wrong).
- **rocFFT workspace size** — confirm ~0 for our sizes (step 4). If not, add it
  to `_amdgpu_bytes_per_window`.

## Validation steps (on the RX 6800 XT)

1. **Precompile / smoke:** `julia --project=bench/CUDA` won't work (that env is
   CUDA-only). Use whatever project has AMDGPU + dev Hammerhead (mirror
   `bench/CUDA/Project.toml` as e.g. `bench/AMDGPU/Project.toml` with
   `AMDGPU` + `Hammerhead`). Load `using AMDGPU, Hammerhead` and run a small
   `run_piv(A, B, passes; backend = :amdgpu)` to confirm it compiles and the
   helpers resolve.
2. **Correctness unchanged:** `run_piv` `:amdgpu` vs `:cpu` must still match to
   ~1e-15 (Float64) / ~1e-4 (Float32). `bench/gpu_validate.jl amdgpu` is the
   existing harness.
3. **Spill fixed + tune the optimum:** run
   `julia --project=<amdgpu env> -t auto bench/gpu_batch_sweep.jl amdgpu 4096 Float64`
   (and `Float32`). This sweeps batch = 8192/4096/2048/1024, printing wall-time
   and free-VRAM low-water. Expect the fixed-8192 FP64 spill to be gone under
   the auto cap (unset env). **RDNA2 has full-rate FP64? No — it's 1/16; still
   memory behavior, not ALU, is the spill lever here.** Note the timing-optimal
   batch for FP64; if the pure memory cap leaves time on the table (it did on
   Ada: cap chose ~3584, optimum was ~1024), that's the **occupancy plateau** —
   a separate limit the byte-budget can't see. Decision on Ada was to ship the
   memory cap and tune the occupancy ceiling later per-GPU; do the same here or
   pin `HAMMERHEAD_AMDGPU_BATCH` to the measured optimum.
4. **rocFFT workspace check:** the CUDA probe was
   `CUDA.CUFFT.cufftGetSize(plan.handle, ref)` on a `plan_fft!`'d
   `CuArray{Complex{T},3}`. Find the rocFFT analogue (AMDGPU.jl `rocFFT`
   wrappers) and confirm workspace ≈ 0 for a 256×256 × 8192 plan; if nonzero,
   fold it into `_amdgpu_bytes_per_window`.

## Context / provenance

- The env-override escape hatch (`HAMMERHEAD_CUDA_BATCH`, and the proposed
  `HAMMERHEAD_AMDGPU_BATCH`) is intentional: it pins a fixed batch for
  benchmarking/debugging and is what `bench/gpu_batch_sweep.jl` drives.
- `bench/gpu_batch_sweep.jl` is committed and already has the `amdgpu` branch;
  it's the reusable per-GPU tuning tool going forward.
- CLAUDE.md's GPU section documents the backends as "line-for-line mirrors" —
  keep them so. Until this lands, the two exts diverge (CUDA has the cap,
  AMDGPU has the old fixed const).
- Watch the file-encoding trap: edit these files with an editor that preserves
  UTF-8 (the source has `²`, `→`, `ΔC`, `÷`). A PowerShell
  `Get-Content | Set-Content` round-trip corrupts them to mojibake and breaks
  the parser.
