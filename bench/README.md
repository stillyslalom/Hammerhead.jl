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
