```@meta
CurrentModule = Hammerhead
```

# Ensemble and statistics

Use [`run_piv_ensemble`](@ref) to estimate a representative displacement when individual
image pairs have weak correlation peaks and the flow is statistically
stationary. The peak of summed correlations need not equal the arithmetic
mean of individual vectors. Use [`field_statistics`](@ref) on a sequence of individually
measured fields to obtain pointwise means, fluctuation RMS, and valid-sample
counts. For recordings larger than RAM, feed each completed field to a
[`FieldStatisticsAccumulator`](@ref), then finalize it with
`field_statistics(acc)`. Its storage depends on the grid size rather than
the recording length; snapshots share the vector API's validity and
population-moment conventions. This page also covers temporal validation, reference-error
diagnostics, and spectra. The [sequence tutorial](../tutorials/sequence_statistics.md)
compares the two approaches; the [ensemble guide](../howto/ensemble.md)
covers setup and limitations.

For spectra, provide a field-sampling `dt` or explicit `sample_times`, separately
from each image pair's delay. Explicit times undergo exact interval and global
grid-residual checks with zero-default tolerances. The
[spectrum timing guide](../howto/spectrum_timing.md) covers Float64 range roundoff,
units, scale compatibility and the opt-in detached timing report. Neither API
discovers timestamps or resamples irregular recordings.

```@index
Pages = ["ensemble.md"]
```

## Ensemble correlation

```@autodocs
Modules = [Hammerhead]
Pages = ["ensemble.jl"]
Private = false
```

## Time-series statistics and diagnostics

```@autodocs
Modules = [Hammerhead]
Pages = ["statistics.jl"]
Private = false
```
