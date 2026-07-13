# Hammerhead.jl benchmarks

Performance benchmarks for regression checking. From the package root:

```bash
julia --project=. --threads=auto bench/run_benchmarks.jl
```

The suite covers:

- **Correlation**: per-window cost of `CrossCorrelator` (plain and
  padded/apodized) and `PhaseCorrelator` at 16–64 px window sizes.
- **Full pipeline**: `run_piv` on a 512×512 synthetic vortex pair —
  single-pass, padded, and multi-pass schedules, serial vs. threaded.
- **Validation**: per-validator cost on a 128×128 vector field.

Timings are minima over several samples (no BenchmarkTools dependency, so the
script runs with the package's own environment). For regression checks, run on
a quiet machine and compare against a baseline log from `main`:

```bash
git stash && julia --project=. bench/run_benchmarks.jl > baseline.log && git stash pop
julia --project=. bench/run_benchmarks.jl > new.log
```

Treat >20 % changes in the minima as signal; smaller differences are usually
noise.

## GPU backends

`bench/gpu_validate.jl` checks a device backend against the CPU reference
(single-pass, multipass, masked, ensemble, and uncertainty paths) and
`bench/gpu_benchmarks.jl` times it (see
the headers for the required environment; both take the backend selector as
ARGS[1]). On the dev box's RX 6800 XT (ROCm 6.4, 4 CPU threads), the
`:amdgpu` backend matches `:cpu` to ~7e-15 (Float64) and runs a
padded+apodized 32/16 single pass at 1.3–2.4× the threaded CPU speed
(1024²–2048², Float64/Float32). The `:cuda` backend is likewise validated on
an RTX 2000 Ada (CUDA.jl 6.2, driver CUDA 12.8): every path matches `:cpu` to
~1e-15 (Float64) and runs 1.7–2.8× the threaded CPU speed (1024²–2048²,
Float64/Float32; multipass-deformation included).

The user-facing setup, support matrix, memory sizing, and troubleshooting
guide is [`docs/src/howto/gpu.md`](../docs/src/howto/gpu.md).

## Allocation/GC profiling

For batch memory work, profile the real Case E sequence workload with:

```bash
julia --project=. --threads=4 bench/gc_profile.jl --pairs=5
```

The script defaults to the committed workflow shape:
`image_pairs(frames; mode=:chained)`, `multipass_parameters([128, 64, 32, 16])`,
and `run_piv_sequence(...; progress=false)`. It writes a concise GC/allocation
summary plus `Profile.Allocs` flat/tree reports under `bench/profile-output/`
(gitignored). Use `--progress=true` only when you specifically want terminal
progress-lock overhead included in the profile. For quick smoke runs, add
`--stdlib-reports=false` to skip the verbose stdlib reports and write only the
custom summary.
