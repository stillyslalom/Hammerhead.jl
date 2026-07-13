# GPU Extension Plan

## Goal

Accelerate Hammerhead's heavy workloads while preserving the current CPU
behavior by default:

- publication-quality production PIV runs;
- high-effort iterative deformation;
- uncertainty quantification;
- large ensemble correlation runs;
- stereo runs that include dewarping plus per-camera PIV.

GPU support must stay cross-platform. CUDA should be the first proving backend,
but the architecture must leave room for AMDGPU/ROCm, Metal, oneAPI, and future
JuliaGPU backends.

## Current Hot Path

The current implementation is CPU-first:

- core imports FFTW in `src/Hammerhead.jl`;
- correlators store `Matrix` buffers and FFTW plans in `src/correlators.jl`;
- `PIVWorkspace` reuses CPU interpolant, deformation, and correlator buffers in
  `src/pipeline.jl`;
- `process_windows!` correlates each interrogation window independently;
- `deform_images` performs full-image cubic B-spline deformation;
- `run_piv_ensemble` repeatedly accumulates per-window correlation planes;
- UQ statistics are accumulated from deformed windows in Float64.

The best acceleration targets are:

1. Batched interrogation-window correlation.
2. Multipass image deformation.
3. Ensemble plane accumulation.
4. UQ statistics.
5. Dewarping and selected preprocessing.

## Architecture

Use an execution backend abstraction. Do not put CUDA-specific dependencies in
the core package.

Possible future driver shape:

```julia
run_piv(imgA, imgB; effort = :high, backend = :cpu, kwargs...)
run_piv_sequence(pairs; backend = :cpu, kwargs...)
run_piv_ensemble(pairs; backend = :cpu, kwargs...)
piv_workspace(; backend = :cpu)
```

`PIVParameters` should remain algorithmic. Backend/device choice belongs in the
driver and workspace layer, not in result semantics.

Backend types are internal implementation details. Do not export names like
`CPUBackend`; those names are likely to collide with other offloading packages.
Public backend selection should use documented selectors such as `backend =
:cpu`, without requiring users to import generic backend type names.

GPU execution must be designed around launch and transfer latency. Hammerhead's
hot loops contain many small logical operations, but the GPU implementation
should not mirror the CPU structure as one launch per window or one transfer per
stage. Work must be grouped into coarse device-resident batches so kernel launch
overhead and host-to-device/device-to-host transfers are amortized over many
windows, images, or pairs.

Latency rules:

- keep source images, deformed images, window batches, FFT buffers,
  accumulators, and UQ scratch on the device for the duration of a pass;
- copy final vector-grid outputs back to the host, not per-window planes;
- process windows in memory-bounded batches large enough to saturate the device;
- fuse cheap per-window kernels when it reduces launch count without making
  kernels too complex to optimize;
- avoid GPU offload for small images or low-effort runs unless profiling shows
  transfer plus launch overhead is still paid back;
- stream sequence/ensemble pairs through persistent device buffers, overlapping
  host loading/preprocessing with GPU work where possible;
- consider backend-specific graph or command-buffer replay later for iterative
  final passes, but do not make the portable first implementation depend on it.

Implementation options:

- keep the default CPU backend internal to core;
- put GPU code in a weak extension or sibling package such as
  `HammerheadGPU.jl`;
- use KernelAbstractions.jl for portable custom kernels;
- use backend-specific array and FFT packages where needed:
  - CUDA.jl/cuFFT for NVIDIA;
  - AMDGPU.jl/rocFFT for AMD;
  - Metal.jl/Metal Performance Shaders where available;
  - oneAPI.jl/oneMKL where available.

Fallback rule:

- unsupported GPU features fall back to CPU with a clear warning;
- `strict_gpu = true` should turn fallback into an error;
- CPU default behavior must remain bitwise identical.

## Phase 0: Benchmarks And Profiling

Add benchmarks before implementation changes:

- 1024, 2048, and 4096 px synthetic pairs;
- `effort = :high`;
- final-pass `max_iterations = 2:5`;
- `uncertainty = true`;
- `run_piv_ensemble` with 10, 100, and 1000 pairs;
- stereo path with dewarp plus per-camera high-effort PIV.

Record separate timing buckets:

- loading/preprocess;
- image deformation;
- window correlation;
- validation/replacement;
- UQ;
- result construction;
- host/device transfer once GPU exists.

Acceptance:

- benchmark script runs without optional GPU packages;
- CPU timings remain comparable to the current benchmark style;
- output identifies the top bottlenecks for high-effort workloads.

## Phase 1: Backend Plumbing

Introduce internal execution types and capability checks:

```julia
abstract type _AbstractHammerheadBackend end
struct _CPUBackend <: _AbstractHammerheadBackend end
```

Possible capability predicates:

```julia
supports_fft(::_AbstractHammerheadBackend)
supports_batched_fft(::_AbstractHammerheadBackend)
supports_fp64(::_AbstractHammerheadBackend)
supports_unified_memory(::_AbstractHammerheadBackend)
```

Thread `backend` through:

- `run_piv`;
- `run_piv_sequence`;
- `run_piv_ensemble`;
- `run_piv_stereo`;
- `piv_workspace` / `PIVWorkspace`.

Acceptance:

- the default CPU backend is internal and not exported;
- public driver/workspace keywords use `backend = :cpu`;
- existing call sites still work;
- full test suite remains bitwise identical on CPU;
- docs include any documented backend internals in the internals reference, not
  as exported public API.

## Phase 2: Batched GPU Correlation

Replace per-window FFTW calls with a batched GPU correlation engine.

Pipeline:

1. Gather windows into device buffers shaped like `(fft_rows, fft_cols, batch)`.
2. In a portable kernel, perform mean subtraction, mask handling, apodization,
   and zero padding.
3. Run batched forward FFTs.
4. In a portable kernel, compute cross-power spectrum.
5. Run batched inverse FFTs.
6. In a portable kernel, apply `fftshift_abs!` and overlap gain.
7. In a portable kernel, find peaks and run `gauss3`/`gauss9` subpixel
   refinement.
8. Copy only grid outputs back to the host.

Initial scope:

- `correlation_method = :cross`;
- `subpixel_method in (:gauss3, :gauss9)`;
- `padding = true/false`;
- `apodization in (:none, :gauss)`;
- `keep_correlation_planes = false`.

CPU fallback:

- `correlation_method = :phase`;
- `subpixel_method = :gauss2d`;
- `keep_correlation_planes = true`;
- backends without usable FFT support.

Implementation note:

Window batching is required. Full materialization of all overlapping windows on
large images can exceed device memory.

Latency note:

The correlation path must launch per batch, not per interrogation window. For a
batch, gather/load, spectrum formation, normalization, peak finding, and result
writeback should run as a small fixed sequence of coarse kernels around batched
FFT calls. The host should see one batch-level operation, not thousands of tiny
operations.

Acceptance:

- CUDA prototype passes existing accuracy tests with tolerance-based GPU
  comparisons;
- CPU default remains unchanged;
- GPU path can run a single-pass synthetic PIV case end to end.

## Phase 3: GPU Multipass Deformation

Accelerate `apply_predictor` / `deform_images`.

Conservative first implementation:

- keep the existing cubic B-spline prefilter on CPU;
- copy spline coefficients to device;
- evaluate cubic samples in a KernelAbstractions kernel;
- initially keep predictor-grid interpolation on CPU because the vector grid is
  small;
- move predictor interpolation later only if profiling shows it matters.

Do not silently change the interpolation model. The current publication-quality
path depends on cubic B-spline resampling.

Latency note:

Once image coefficients are copied to the device, all deformation sweeps for a
pass should reuse those resident coefficients and output buffers. Do not copy
warped images back to the host between deformation and correlation.

Acceptance:

- multipass high-effort synthetic cases match CPU scientific tolerances;
- iterative deformation converges on the same scenarios as CPU;
- GPU path handles workspace reuse across calls.

## Phase 4: Ensemble And UQ

Ensemble correlation should keep accumulators on the device:

- stream each image pair to the GPU;
- deform if a predictor exists;
- correlate each window batch;
- accumulate planes on device;
- transfer only final vector-grid outputs.

The ensemble path is especially sensitive to transfer strategy. Repeatedly
copying summed planes or per-pair per-window outputs back to the host would
erase much of the GPU gain. The intended dataflow is image pair in, device-side
accumulate, final result out.

UQ should be capability gated:

- preserve Float64 accumulation semantics by default;
- GPU UQ only when `supports_fp64(backend)` is true and performance is
  worthwhile;
- on weak-Float64 backends, especially many Apple/consumer GPUs, keep UQ on CPU
  unless an explicit approximate mode is introduced later.

### Phase 4b: Device UQ

Implemented for the KernelAbstractions CPU proving backend, CUDA, and AMDGPU.
The portable kernel computes the Wieneke (2015) per-window statistics in
Float64 from the already gathered, mean-subtracted, apodized windows. Fused
single-sweep UQ returns only these packed scalars; iterative passes perform one
device sweep over the final warped images. Ensemble statistics remain on the
device and add across pairs before a single final transfer. The existing CPU
finalizer remains the correctness reference and preserves result semantics.

Acceptance:

- ensemble GPU path reduces time for large pair counts;
- UQ results are within documented tolerances;
- CPU UQ remains the correctness reference.

## Phase 5: Dewarping, Preprocessing, And Validation

Accelerate only after correlation and deformation are working:

- `dewarp!` is a good portable-kernel candidate because source coordinates are
  precomputed;
- background subtraction and highpass filtering may help sequence workloads;
- CLAHE is lower priority because histogram kernels are more complex;
- UOD, replacement, and smoothing likely stay CPU-first until very dense final
  grids prove otherwise.

Acceptance:

- stereo high-effort workloads avoid unnecessary host/device ping-pong;
- dewarp masks and sign conventions remain unchanged.

## Testing Strategy

Required test tiers:

- CPU default tests: exact current behavior.
- KernelAbstractions CPU backend tests: validate portable kernels without GPU
  hardware.
- CUDA smoke/accuracy tests where hardware is available.
- AMDGPU and Metal smoke tests where CI or local hardware exists.

Golden scenarios:

- uniform shift;
- vortex midpoint truth;
- masked interrogation windows;
- padded + Gaussian apodized correlation;
- final-pass UQ median sanity;
- ensemble uncertainty shrinkage with pair count;
- stereo dewarp plus high-effort per-camera PIV.

GPU comparisons should use scientific tolerances, not bitwise equality, because
FFT and reduction order will differ from FFTW.

## Risks

- Batched FFT portability is the central risk.
- Device memory can be exhausted without window batching.
- GPU floating-point order changes results.
- UQ's Float64 convention conflicts with weak-Float64 GPUs.
- Optional GPU dependencies must not burden normal CPU users or registration.
- `keep_correlation_planes = true` can dominate memory and should remain
  CPU-first until there is a concrete use case.

## First Implementation Slice

Implement Phase 0 and a minimal Phase 1 skeleton:

1. Add benchmark instrumentation for high-effort production workloads.
2. Add internal backend plumbing for the default CPU path.
3. Thread an internal CPU backend default through `run_piv`,
   `run_piv_sequence`, `run_piv_ensemble`, and `run_piv_stereo` without
   exporting backend type names.
4. Keep all CPU execution paths unchanged.
5. Add tests that prove the internal CPU backend path matches default behavior.
6. Update docs without exporting backend types; documented internals belong in
   the internals reference.

Do not add CUDA/AMDGPU/Metal dependencies in this first slice.
