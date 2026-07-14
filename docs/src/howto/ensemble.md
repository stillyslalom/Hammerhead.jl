# Ensemble correlation for low signal-to-noise ratio

**Goal:** extract a mean displacement field from recordings whose
individual pairs are too noisy for reliable peaks — micro-scale particle
image velocimetry (micro-PIV), weak seeding, or low laser power (for example,
PIV Challenge case 4A). This is the low signal-to-noise ratio (SNR) regime.

## When it applies

Ensemble (sum-of-correlation) PIV [Meinhart2000](@cite) averages each
interrogation window's *correlation planes* across many pairs before
locating the peak once. Random noise peaks average out; the displacement
peak reinforces. The catch: it assumes **statistically stationary flow** —
the result is the ensemble-mean field, and pair-to-pair fluctuation is
averaged away, not measured.

## Basic use

[`run_piv_ensemble`](@ref) takes the same pair list as
[`run_piv_sequence`](@ref):

```julia
using Hammerhead

files = readdir("micropiv"; join = true)
pairs = image_pairs(files)                 # (1,2), (3,4), ... double-frame
passes = multipass_parameters([64, 32, 16]; padding = true, apodization = :gauss)

result = run_piv_ensemble(pairs, passes; preprocess = img -> highpass_filter!(img, sigma = 6))
```

Multi-pass works as in the single-pair engine, with one difference: each
pass's ensemble field acts as a *shared* deformation predictor for every
pair of the next pass.

## Keep large ensembles on a GPU

Pass `backend = :amdgpu` or `:cuda` to accumulate the summed planes on the
device across every pair:

```julia
using AMDGPU
result = run_piv_ensemble(pairs, passes;
    backend = :amdgpu,
    preprocess = img -> highpass_filter!(img, sigma = 6),
)
```

Loading and `preprocess` remain CPU work. The correlation accumulator and,
when enabled, Float64 uncertainty statistics remain device-resident until
the final field is built. Accumulator memory scales with vector-grid density
and correlation-plane area, not pair count; padding quadruples the plane
footprint. Use the sizing formula and validation workflow in
[Run PIV on a GPU](gpu.md) before a large production ensemble.

## Practical notes

- **More pairs beat bigger windows.** The whole point is that window size
  no longer has to compensate for noise; keep windows sized to the flow
  structure and add pairs until the field is clean.
- **`peak_ratio` describes the ensemble planes**, so it improves with pair
  count — a rising ensemble peak ratio is your convergence indicator.
- **File paths are reloaded once per pass.** For many passes over slow
  storage, load frames into memory first and pass matrices.
- **Preprocessing** (`preprocess`, `image_type`) and **masking** (`mask`,
  one static mask for all pairs) work exactly as in the batch driver.

## Uncertainty of the ensemble mean

With `uncertainty = true` (final pass repeated for convergence — see
[uncertainty quantification](../explanation/uncertainty.md)), the
correlation-statistics estimator pools its sums across all pairs, so
`uncertainty_u`/`uncertainty_v` describe the noise-driven uncertainty of
the ensemble-mean vector and shrink as pairs are added:

```julia
passes = multipass_parameters([64, 32, 16, 16];
    padding = true, apodization = :gauss, uncertainty = true)
result = run_piv_ensemble(pairs, passes)
```

This does **not** include genuine flow fluctuation. If the flow is not
perfectly stationary, quantify the fluctuation separately with
[`field_statistics`](@ref) over single-pair results — when the per-pair
SNR allows it — and treat the ensemble uncertainty as a lower bound.
