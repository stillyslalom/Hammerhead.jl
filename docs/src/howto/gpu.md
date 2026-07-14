# Run PIV on a GPU

**Goal:** run particle image velocimetry (PIV) on a graphics processing unit
(GPU), select and validate the backend, and understand which work stays on
the device, and avoid configurations where transfer or Float64 cost outweighs
the acceleration.

Hammerhead's default `backend = :cpu` uses the central processing unit (CPU)
and remains the complete reference
implementation. GPU support is optional: loading AMDGPU.jl or CUDA.jl
activates a package extension without adding either device stack to
Hammerhead's normal installation.

## Install and select a backend

Add the package for the device in the environment where Hammerhead runs:

```julia
using Pkg
Pkg.add("AMDGPU")  # AMD/ROCm
# Pkg.add("CUDA")  # NVIDIA/CUDA
```

Load that package before selecting its backend:

```julia
using Hammerhead
using AMDGPU

AMDGPU.functional() || error("AMDGPU cannot use the current ROCm installation")

result = run_piv(imgA, imgB; effort = :high, backend = :amdgpu)
```

For NVIDIA, use `using CUDA`, check `CUDA.functional()`, and pass
`backend = :cuda`. The built-in `backend = :ka` runs the same portable
KernelAbstractions kernels on the CPU. It is a correctness and development
tier, not a faster replacement for the FFTW-backed `:cpu` path.

The available selectors are:

| Selector | Provider | Intended use |
|---|---|---|
| `:cpu` | Hammerhead | Default, complete FFTW reference path |
| `:ka` | Hammerhead | Hardware-free test of the portable kernels |
| `:amdgpu` | AMDGPU.jl extension | AMD GPUs through ROCm |
| `:cuda` | CUDA.jl extension | NVIDIA GPUs through CUDA |

Hammerhead currently supports AMDGPU.jl 2 and CUDA.jl 5 or 6. Driver and
runtime compatibility is controlled by the device package. Follow the
[AMDGPU.jl setup guide](https://amdgpu.juliagpu.org/stable/) or
[CUDA.jl installation guide](https://cuda.juliagpu.org/stable/installation/overview/)
when `functional()` is false. The AMD path is hardware-validated
on an RX 6800 XT with ROCm 6.4; the CUDA path is hardware-validated on an
RTX 2000 Ada with CUDA.jl 6.2.

## Use the backend across drivers

The selector is accepted by the planar, sequence, ensemble, and stereo
drivers:

```julia
single = run_piv(imgA, imgB, passes; backend = :amdgpu)

series = run_piv_sequence(pairs, passes;
    backend = :amdgpu,
    output = "recording.jld2",
)

mean_field = run_piv_ensemble(pairs, passes;
    backend = :amdgpu,
    progress = false,
)

stereo = run_piv_stereo(A1, B1, A2, B2, dw1, dw2, passes;
    backend = :amdgpu,
)
```

[`run_piv_sequence`](@ref) and [`run_piv_ensemble`](@ref) create a matching
[`PIVWorkspace`](@ref) and reuse device buffers and fast Fourier transform
(FFT) plans across pairs.
For a hand-written loop, do the same explicitly:

```julia
workspace = piv_workspace(backend = :amdgpu)
results = [run_piv(a, b, passes; backend = :amdgpu, workspace)
           for (a, b) in pairs]
```

A workspace is backend-specific. Passing a CPU workspace to a GPU run, or the
reverse, is an error.

## Check that the configuration is supported

The KernelAbstractions (KA) family of backends (`:ka`, `:amdgpu`, and `:cuda`)
currently implements:

| Feature | `:cpu` | KA-family |
|---|:---:|:---:|
| Cross-correlation | yes | yes |
| Phase correlation | yes | yes, filtered normalized spectrum |
| `:gauss3` / `:gauss9` subpixel fit | yes | yes |
| `:gauss2d` subpixel fit | yes | no |
| Padding and Gaussian apodization | yes | yes |
| Multi-pass and iterative deformation | yes | yes |
| Correlation-statistics uncertainty | yes | yes, Float64 statistics |
| Masked interrogation windows | yes | yes |
| Ensemble correlation | yes | yes, device-resident accumulator |
| `keep_correlation_planes = true` | yes | no |

Unsupported combinations raise an `ArgumentError` before analysis rather
than silently switching algorithms or falling back to the CPU. Use
`backend = :cpu` when `:gauss2d` or retained correlation
planes are required.

Stereo forwards the backend to both per-camera PIV analyses. Dewarping and
three-component (3C) reconstruction remain on the CPU. Loading, preprocessing,
validation,
outlier replacement, smoothing, and result construction are also CPU work.

## Understand device residency

For a non-deforming pass, each source pair is uploaded once. For deforming
passes, Hammerhead performs the cubic B-spline prefilter on the CPU, stages
the coefficients once, and keeps warped images on the device between
deformation and correlation. Each correlation batch performs window gather,
FFT, cross-power, shift/gain, peak analysis, and subpixel refinement on the
device. Only packed per-window outputs return to the host.

Uncertainty statistics use Float64 even when images are Float32. An iterative
pass computes them once from the last converged deformation. Ensemble runs
accumulate both correlation planes and uncertainty statistics on the device
across all pairs, then transfer final packed values.

Device correlation is internally tiled to bound FFT scratch memory. Ensemble
planes, however, cover the complete vector grid for the duration of a pass.
Their approximate footprint is:

```text
number of windows * correlation-plane rows * correlation-plane columns * sizeof(T)
```

Padding doubles each plane dimension and therefore quadruples this part of
the footprint. For example, a 2048x2048 image with 32-pixel windows,
16-pixel overlap, padding, and Float64 data has about 16,129 windows and a
64x64 plane per window, or roughly 504 MiB for the ensemble accumulator alone.
Pair count increases runtime but not accumulator size.

## Decide whether GPU execution pays off

GPU execution is intended for large images, dense window grids, iterative
deformation, and large ensembles. The CPU is usually faster for small images
or low-effort single passes because device setup, launches, and transfers do
not amortize. `threaded` controls CPU work; device backends process one logical
window batch and do not fan it out over Julia host threads.

Float32 reduces image, FFT, and correlation-plane memory and is often much
faster on consumer GPUs:

```julia
result = run_piv_sequence(pairs, passes;
    backend = :amdgpu,
    image_type = Float32,
)
```

The returned fields follow the image precision, while uncertainty's internal
statistics remain Float64. On GPUs with weak Float64 throughput, uncertainty
may erase the speedup from correlation. The RX 6800 XT development system is
one such case: use the CPU reference or benchmark the complete workload when
uncertainty quantification (UQ) dominates.

## Validate a device and benchmark the workload

The repository includes a correctness script covering single-pass,
multi-pass, masks, ensemble accumulation, and uncertainty:

```bash
julia --project=/path/to/gpu-env -t 4 bench/gpu_validate.jl amdgpu
julia --project=/path/to/gpu-env -t 4 bench/gpu_benchmarks.jl amdgpu
```

Pass `cuda` for NVIDIA. GPU comparisons use scientific tolerances because
device FFT and intrinsic order can differ from FFTW. Validate after changing
drivers, device-package versions, or Julia versions, then benchmark the same
image size, precision, pass schedule, mask, and pair count used in production.

## Troubleshoot common failures

- **Unsupported backend selector:** load `AMDGPU` or `CUDA` in the same Julia
  process before calling Hammerhead. Installing the package alone does not
  activate its extension.
- **`functional()` is false:** repair the driver/runtime installation using
  the device package's diagnostics before running Hammerhead.
- **Unsupported option error:** switch the option or use `backend = :cpu`;
  Hammerhead does not silently change the requested analysis.
- **Workspace backend mismatch:** rebuild it with
  `piv_workspace(backend = backend)`.
- **Out of device memory:** use Float32, reduce overlap, avoid retained full
  grids where possible, or process a smaller region. For ensembles, padding
  and vector-grid density are the main accumulator-memory multipliers.
- **GPU slower than CPU:** benchmark a warmed run, reuse a workspace, and test
  a production-sized workload. Small transfers and weak device Float64 are
  common causes.
