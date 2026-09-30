# Hammerhead.jl benchmarks

Performance benchmarks for regression checking. From the package root:

```bash
julia --project=. --threads=auto bench/run_benchmarks.jl
```

The suite covers:

- **Correlation**: per-window cost of `CrossCorrelator` (plain and
  padded/apodized) and `PhaseCorrelator` at 16–64 px window sizes.
- **Full pipeline**: single-pass and multi-pass schedules on synthetic
  vortex images, with serial and threaded execution.
- **High effort**: configurable image sizes, final-pass iterations,
  ensemble pair counts, and stereo dewarping and reconstruction.
- **Validation**: per-validator cost on a 128×128 vector field.

Most timings report the minimum across several samples. The larger
high-effort workloads default to one sample; control them with
`HAMMERHEAD_BENCH_SIZES`, `HAMMERHEAD_BENCH_ENSEMBLES`, and
`HAMMERHEAD_BENCH_SAMPLES`. Single-sample results can include compilation or
allocation effects, so repeat a comparison before attributing a difference
to an implementation change.

For regression checks, use an existing baseline checkout and the same Julia
version, thread count, input sizes, and environment settings for both runs:

```bash
julia --project=../Hammerhead-baseline --threads=4 ../Hammerhead-baseline/bench/run_benchmarks.jl > baseline.log
julia --project=. --threads=4 bench/run_benchmarks.jl > new.log
```

Keep the machine otherwise idle and compare repeated runs. A percentage
change alone does not distinguish a regression from scheduling, compilation,
or thermal effects; inspect the affected workloads and their variability.

## GPU backends

`bench/gpu_validate.jl` checks a device backend against the CPU reference
(single-pass, multipass, masked, ensemble, and uncertainty paths) and
`bench/gpu_benchmarks.jl` times it (see
the headers for the required environment; both take the backend selector as
ARGS[1]; the benchmark optionally takes `exclusion` or `regionalmax` as
ARGS[2], defaulting to the package's `regionalmax`). Record the device,
driver, package versions, CPU thread count, image size, numeric type, and
peak-finding mode with the results. CPU agreement checks implementation
consistency; it does not measure error against the true particle displacement.

`bench/gpu_profile_uq.jl <backend>` (CUDA only; it uses `CUDA.@profile`)
prints the per-kernel device-time breakdown of a UQ-enabled multipass run.
Use it to identify which kernels dominate on your device. Device kernel time
does not include the full cost of file loading, preprocessing, or transfers.

The user-facing setup, support matrix, memory sizing, and troubleshooting
guide is [`docs/src/howto/gpu.md`](../docs/src/howto/gpu.md).

## Allocation/GC profiling

For a portable CPU/device/hybrid comparison on a target machine, run:

```bash
julia --project=<gpuenv> -t auto bench/gpu_configurations.jl amdgpu 2048 Float32 high 3
```

Use `cuda` or `ka` instead of `amdgpu` as appropriate. The script compares
all-CPU, all-device, and device-correlation plus threaded-CPU uncertainty.
For real data or a custom pass schedule, call `benchmark_piv_configurations`
directly with a representative loaded image pair.

For batch memory work, `gc_profile.jl` reads a directory of camera images.
The default points to the full Case E sequence under `cases/`, which is not
included in the repository. Supply your recording explicitly:

```bash
julia --project=. --threads=4 bench/gc_profile.jl --camera-dir=path/to/camera --pairs=5
```

The script uses:
`image_pairs(frames; mode=:chained)`, `multipass_parameters([128, 64, 32, 16])`,
and `run_piv_sequence(...; progress=false)`. It writes a concise GC/allocation
summary plus `Profile.Allocs` flat/tree reports under `bench/profile-output/`
(gitignored). Use `--progress=true` only when you specifically want terminal
progress-lock overhead included in the profile. For quick smoke runs, add
`--stdlib-reports=false` to skip the verbose stdlib reports and write only the
custom summary.
