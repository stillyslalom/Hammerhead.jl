# Ensemble correlation for low signal-to-noise ratio

**Goal:** extract a mean displacement field from recordings whose
individual pairs are too noisy for reliable peaks, as in micro-scale particle
image velocimetry (micro-PIV), weak seeding, or low laser power (for example,
PIV Challenge case 4A). This is the low signal-to-noise ratio (SNR) regime.

## When it applies

The [sequence tutorial](../tutorials/sequence_statistics.md) compares an
ensemble result with the mean of separately measured vector fields.

Ensemble (sum-of-correlation) PIV [Meinhart2000](@cite) averages each
interrogation window's *correlation planes* across many pairs before
locating the peak once. Random noise peaks average out; the displacement
peak reinforces. Use it when the flow is **statistically stationary**: the
combined peak estimates a representative displacement field. Stationarity
does not require every pair to have the same displacement. If displacements
fluctuate, the combined peak need not lie at the arithmetic mean of the
individual vectors. Use individually measured fields for fluctuation
statistics, and inspect the ensemble's sensitivity to the selected pairs.

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

## Stereo ensembles

For a statistically stationary low-SNR stereo recording, compose one
ensemble field per camera followed by calibrated 3C reconstruction:

```julia
mean3c = run_piv_stereo_ensemble(cam1_pairs, cam2_pairs, dw1, dw2, passes;
    preprocess = (preprocess_cam1, preprocess_cam2),
)
```

Raw frames are loaded and dewarped lazily on each pass, so the driver does not
retain the entire dewarped recording. As in planar ensemble PIV, the result is
one stationary mean field. Use `run_piv_stereo_sequence` followed by
`field_statistics` when physical fluctuations and all six Reynolds-stress
terms are required.

## Practical notes

- **Add pairs before enlarging windows.** Keep windows small enough to
  resolve the flow structure, then add pairs until the field is stable.
- **Check `peak_ratio` and field stability as you add pairs.** The ratio
  describes the ensemble correlation plane; a clearer peak does not by
  itself show that the estimated field has stopped changing. Compare results
  from increasing counts or separate subsets of the recording.
- **File paths are reloaded once per pass.** For many passes over slow
  storage, load frames into memory first and pass matrices.
- **Preprocessing** (`preprocess`, `image_type`) and **masking** (`mask`,
  one static mask for all pairs) work exactly as in the batch driver.

## Uncertainty of the combined estimate

For `uncertainty = true`, repeat the final window size to reduce its residual
(see [uncertainty quantification](../explanation/uncertainty.md)). The
correlation-statistics estimator then pools its sums across pairs. The reported
`uncertainty_u`/`uncertainty_v` describe the noise-driven uncertainty of the
combined correlation estimate under a shared-displacement assumption. More
pairs may reduce that estimate, but the change need not be monotonic:

```julia
passes = multipass_parameters([64, 32, 16, 16];
    padding = true, apodization = :gauss, uncertainty = true)
result = run_piv_ensemble(pairs, passes)
```

This does **not** include genuine flow fluctuation or changes in the flow
during the recording. When individual pairs have sufficient signal, use
[`field_statistics`](@ref) on their results to describe fluctuation and
compare it with the ensemble field. Do not interpret the pooled uncertainty
as total error when displacements vary between pairs.
