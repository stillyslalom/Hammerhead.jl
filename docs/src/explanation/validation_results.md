# How accurate are the measurements?

These results come from synthetic image studies with known particle motion,
run on the CPU during development. Use them to choose window schedules and to
decide how much weight to give the stored uncertainty estimates. The study
scripts were removed from `bench/` to keep the repository small; recover them
from git history at commit `88a4bda` (`bench/validation_*.jl`,
`bench/*_uncertainty.jl`, `bench/spatial_transfer.jl`) to rerun them.

All PIV studies used 128 × 128 or 256 × 256 px synthetic pairs with 3 px
Gaussian particles, padded cross correlation with Gaussian apodization, and a
`[32, 16, 16]` multipass schedule unless noted. Errors are measured minus
true displacement at the trajectory midpoint.

## Displacement error

With `padding = true` and `apodization = :gauss`, the random error on uniform
translation and linear shear is about **0.02–0.03 px RMS** over accepted
vectors. Plain circular correlation without padding is biased toward zero by
about 0.15 px; see [correlation accuracy](correlation.md).

An image pair with no particles produces no accepted vectors: every window is
reported as unavailable (`NaN`) rather than as a zero displacement. See
[windows without displacement information](noninformative_windows.md).

## Stored uncertainty is too small

The correlation-statistics uncertainty ([Wieneke2015](@citet); see
[uncertainty quantification](uncertainty.md)) under-covers the true error on
these clean synthetic images:

| Population | 1σ coverage u / v | 2σ coverage u / v |
|:--|--:|--:|
| Baseline translation, 3 seeds | 18% / 15% | 46% / 54% |
| Baseline translation, 8 seeds | 18% / 17% | 45% / 56% |
| Gaussian reference | 68% | 95% |

Two effects contribute:

- About 12% of windows report **σ = 0**. The covariance sum entering the
  estimate comes out negative for those windows and is clamped to zero. With
  repeated independent noise on the same scene, those windows show clearly
  positive error variance.
- For windows with σ > 0, the estimate is still small compared with the
  observed error. Adding noise (uniform half-width 0.03) raises 2σ coverage to
  about 68%, so the shortfall is largest on clean images.

A Bartlett-weighted covariance sum (all lags to ±4 with triangular weights,
which cannot go negative) covered noticeably better in paired repeated-noise
tests: 2σ coverage of paired differences rose from about 90% to about 97%.
Changing the production estimator in `src/uncertainty.jl` is an open
investigation.

Until then, treat stored σ as a relative indicator for comparing windows and
recipes, and use a median over accepted vectors when comparing it with an
independent error estimate. Rendering particles by pixel-area integration
instead of point sampling lowered the error and lowered coverage further, so a
more realistic renderer widens the gap rather than closing it.

## Spatial response

A sinusoidal shear of amplitude 0.5 px tested how much of a spatial variation
two schedules keep. Both started with 64 px and 32 px passes and ended either
with two 16 px passes (8 px overlap) or two 32 px passes (24 px overlap), giving
the same 8 px grid spacing.

| Wavelength | Response, 16 px final windows | Response, 32 px final windows | Accepted vectors |
|--:|--:|--:|:--|
| 128 px | 1.03 | 1.03 | all |
| 64 px | 1.05 | 1.00–1.02 | all |
| 32 px | 0.75–0.76 | 0.52–0.59 | 30–56% |

Variations of 64 px and longer are measured at full amplitude. At 32 px the
amplitude is reduced and validation rejects half or more of the vectors, so
resolve features near that scale with smaller final windows or with
[PTV](../tutorials/ptv.md).

## Particle tracking

Eight-frame annotated synthetic clips (25 particles each) tested
[`run_ptv`](@ref) and [`track_particles`](@ref) with default settings.

- **Clean clips:** every particle was detected, matched and tracked.
- **Gaps:** particles hidden for one frame were relinked with `max_gap ≥ 1`,
  and for two frames with `max_gap = 2`.
- **Close encounters, noise and clutter:** accepted matches had 100%
  precision and about 89% recall (170/190 and 168/190 on two seeds), with no
  identity switches within a returned track.

On the independent VSJ301 standard images (frames 000–007), the listed
particle positions line up with Hammerhead's detections only after adding
**+0.5 px** to both source coordinates: 12,097 of the listed positions were
localized with that offset, against about 2,000 without it. Check the pixel-origin
convention of any external reference before comparing positions.
