```@meta
CurrentModule = Hammerhead
```

# Ensemble and statistics

Ensemble (sum-of-correlation) particle image velocimetry (PIV) for recordings
with low signal-to-noise ratio (SNR), time-series
statistics over planar and stereo result sequences, temporal validation,
accuracy diagnostics,
and spectra. See the [ensemble how-to](../howto/ensemble.md). GPU backends
keep summed planes and uncertainty statistics device-resident; setup and
memory sizing are covered in [Run PIV on a GPU](../howto/gpu.md).

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
