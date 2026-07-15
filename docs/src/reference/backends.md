```@meta
CurrentModule = Hammerhead
```

# KernelAbstractions and GPU backends

This page describes the internal execution-backend interface and the shared
KernelAbstractions (KA) implementation. It is intended for contributors who
are changing kernels, maintaining the CUDA/AMDGPU extensions, or adding a new
device backend. The backend API is private and may change between releases.
For installation and user-facing backend selection, see
[Run PIV on a GPU](../howto/gpu.md).

## Architecture

Backend selection is deliberately separate from [`PIVParameters`](@ref) and
from result types. A public symbol is resolved once, at the driver boundary,
to a private dispatch object:

| Selector | Dispatch type | Correlation implementation |
|---|---|---|
| `:cpu` | `_CPUBackend` | FFTW correlators and CPU analysis |
| `:ka` | `_KABackend` | Shared KA kernels on `KernelAbstractions.CPU()` |
| `:cuda` | `_CUDABackend` | Shared KA kernels, CUDA arrays, and cuFFT |
| `:amdgpu` | `_AMDGPUBackend` | Shared KA kernels, ROC arrays, and rocFFT |

The core owns `_AbstractHammerheadBackend`, symbol resolution, pipeline
orchestration, portable kernels, and the hardware-free `:ka` implementation.
CUDA.jl and AMDGPU.jl remain weak dependencies. Loading either package
activates its extension, whose `_resolve_backend(::Val{...})` method registers
the corresponding selector.

The three KA-family implementations share numerical kernels but not engine
types. Each device extension owns the array allocation, transfer, FFT-plan,
batching, and synchronization code appropriate to its device package. This
division keeps device APIs out of the core while giving all three paths the
same gather, correlation post-processing, peak analysis, deformation, and
uncertainty mathematics.

## Pipeline dispatch points

The backend interface is a collection of private multiple-dispatch hooks, not
an abstract type with a fixed list of methods. The principal hooks are:

| Hook | Responsibility |
|---|---|
| `_resolve_backend(::Val)` | Register a selector and construct its backend object |
| `_check_backend_params` | Reject unsupported options before expensive work begins |
| `_engine_nchunks` | Choose host-level fan-out; device backends use one logical chunk |
| `piv_correlation_engines` | Create or retrieve engines for a pass configuration |
| `process_windows!` | Correlate one pair and return packed per-window results |
| `_deform_context` | Stage interpolation coefficients and allocate resident warp buffers |
| `apply_predictor` | Deform a pair and evaluate the predictor on the vector grid |
| `_plane_accumulator` | Allocate storage for ensemble sum-of-correlation planes |
| `accumulate_planes!` | Add one pair to the ensemble accumulator |
| `ensemble_analyze!` | Analyze the completed accumulator and scatter the result grid |

Capability predicates such as `_supports_fft`, `_supports_batched_fft`, and
`_supports_fp64` default to `false`; a backend must opt in. The KA-family
scope guard currently permits cross- or filtered phase correlation,
`:gauss3` or `:gauss9`
subpixel estimation, and optional Float64 uncertainty statistics. It rejects
enlarged `search_area_size`, `:gauss2d`, and retained correlation planes. A new backend
should fail unsupported configurations in `_check_backend_params`, rather
than falling back to the CPU in the middle of a pass.

[`PIVWorkspace`](@ref) owns a backend object and a private engine pool keyed
by backend, precision, correlation method, window configuration, and peak
count. An engine owns
mutable scratch arrays and FFT plans. Consequently a workspace is
backend-specific and must not be shared by concurrent `run_piv` calls.

## Shared KA correlation path

For each device tile, `process_windows!` follows this sequence:

1. `_ka_window_means!` computes masked window means.
2. `_ka_gather!` copies mean-subtracted, apodized interrogation windows into
   the complex FFT batches.
3. The backend's batched forward FFT plans transform both batches.
4. `_ka_crosspower!` forms the cross spectrum in place, or
   `_ka_phasepower!` forms an epsilon-guarded normalized spectrum with the
   same Gaussian frequency filter as the CPU `PhaseCorrelator`.
5. The backend's inverse FFT plan transforms the correlation batch.
6. `_ka_shiftgain!` centers each plane and applies overlap-gain correction
   when padding is enabled.
7. `_ka_analyze!` locates peaks and computes subpixel displacement, peak
   ratio, correlation moment, and alternative peaks.
8. Only the packed scalar output is copied to the host and scattered into the
   result arrays.

Correlation planes use the plane-major layout `R[i, j, k]`: adjacent threads
within a workgroup traverse pixels of one plane. `_ka_analyze!` launches one
workgroup per window and uses `_KA_TPW` threads cooperatively. Its workgroup
size is part of the kernel contract: the launch size and `Val{TPW}` argument
must agree.

The trailing window dimension of scratch and accumulator arrays is padded in
the GPU extensions. Do not remove this apparently unused element: an exact
power-of-two plane stride caused severe device-memory channel conflicts on
validated hardware. The UQ scratch uses a similarly padded leading window
dimension.

## Deformation and residency

Cubic B-spline prefiltering stays on the CPU. After that one-time step,
`_deform_context` copies the padded coefficient arrays to the selected KA
device and allocates device-local warp outputs. Each iterative sweep transfers
only the coarse predictor grid; `_ka_deform!` evaluates the predictor and the
cubic B-spline interpolation on the device. The correlation engine recognizes
those resident warp arrays and consumes them directly.

A pass without deformation instead stages the source images in the engine.
Masks are materialized as dense `Array{Bool}` before upload because a
`BitMatrix` or arbitrary view cannot be transferred with the same direct copy
path. Predictor-node interpolation and validation/replacement remain on the
host because they operate on the much smaller vector grid.

## Ensemble and uncertainty paths

Ensemble correlation does not bring each pair's planes back to the host.
`_KAPlaneAccumulator` holds one plane-major sum for the complete vector grid.
For every pair, the engine runs through the inverse FFT and
`_ka_shiftgain_accum!` adds the live tile into that sum. After the last pair,
`ensemble_analyze!` analyzes accumulator views in device-sized tiles and
returns packed scalars.

Correlation-statistics uncertainty has a distinct additive accumulator.
`_ka_uq_fill!` materializes the smoothed correlation-difference field once,
including its mean, and `_ka_uq_stats!` reduces it into Float64 statistics.
Single-pair analysis copies those statistics back for finalization. Ensemble
analysis keeps them device-resident across pairs and transfers them only when
the accumulated field is finalized. A backend that cannot provide Float64
device accumulation must reject uncertainty in its scope check.

## CUDA and AMDGPU extension responsibilities

The CUDA and AMDGPU extensions intentionally mirror one another. Each one:

- registers its selector and capability methods;
- defines a correlation engine containing device images, FFT batches,
  correlation and UQ scratch, packed-output buffers, origins, and FFT plans;
- pools engines through `pooled_engines` using a backend-specific key;
- implements device staging and a bounded sub-batch loop;
- launches the shared KA kernels with its native KA backend;
- applies in-place FFT plans with `plan * array`; and
- implements resident deformation and ensemble hooks.

Device-specific differences should remain in allocation, array/view
behavior, FFT integration, synchronization, and tuning constants. Numerical
changes belong in `src/ka_backend.jl` so `:ka`, CUDA, and AMDGPU exercise the
same implementation. When changing an extension, compare its corresponding
method in the other extension; accidental drift between these nearly parallel
files is a common source of backend-only bugs.

GPU kernels must avoid operations that lower to host runtime calls. In
particular, throwing conversions and `round(Int, x)` have caused allocation
hostcalls on GPU compilers. Guard the range explicitly and use non-throwing
conversion operations inside kernels. All cross-work-item state that survives
a barrier must be in `@localmem`; ordinary locals do not provide that contract,
including on the KA CPU backend.

## Changing or adding a backend

For a numerical-kernel change, make the change in `src/ka_backend.jl` and use
`backend = :ka` as the first proving tier. Preserve the array layouts,
precision policy, sign convention, and column-major tie ordering in peak
selection. Then validate both single-pair and ensemble paths, with and without
deformation and uncertainty.

For a new device backend, keep the device package in `[weakdeps]`, register a
package extension, and implement the dispatch points above. Start by adapting
one existing GPU extension rather than copying the built-in `:ka` engine: the
GPU engines contain the necessary staging, tiling, residency, and accumulator
behavior that the CPU proving tier does not need to expose through a device
API.

The hardware-free regression suite is `test/test_ka.jl`. Hardware validation
and workload benchmarks are provided by `bench/gpu_validate.jl` and
`bench/gpu_benchmarks.jl`; pass `cuda` or `amdgpu` as the script argument.
Correctness comparisons should use scientific tolerances because device FFTs
and reduction order need not be bitwise identical to FFTW.
