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

## How well the stored uncertainty covers the error

The correlation-statistics uncertainty ([Wieneke2015](@citet); see
[uncertainty quantification](uncertainty.md)) estimates the random part of
the error. These measurements come from a repeated-noise study made for
release 0.2. To measure that part directly, each scene was processed with 8
independent realizations of Gaussian pixel noise (standard deviation 3% of
the particle peak intensity) on 256 × 256 px pairs with 0.02 particles per
pixel and 3 px particle images (``e^{-2}`` diameter). The spread of a
vector's error over the realizations is its random error; the mean is its
systematic error.

| Final windows | Random error within 1σ / 2σ | RMS σ / RMS random error | Windows with σ = 0 |
|:--|--:|--:|--:|
| 32 px, `[64, 32, 32]` | 60–62% / 92–93% | 0.92–0.94 | none |
| 16 px, `[32, 16, 16]` | 53–55% / 89–90% | 0.90–0.91 | none |
| Gaussian reference | 68% / 95% | 1 | |

The ranges span the u and v components. Gentle shear with midpoint truth,
higher noise (10% of peak) and 5 px particles give the same picture. The
average of σ over the 8 realizations covers each window's random error at
60–65% / 91–95%, so the estimate is right on average for a given window. A
single σ comes from the few particles in one window and scatters by 30–40%
between realizations; that scatter, not a bias, lowers the coverage of
individual vectors, in the way a t-statistic with three to six degrees of
freedom covers less than a Gaussian. For 16 px windows σ is also about 10%
low, which matches the underestimation [Wieneke2015](@citet) reports for
small windows.

## Systematic error on clean images

Without image noise, the error of these 3 px particle images is almost
entirely systematic: 0.010–0.014 px, nearly the same in every window. It
depends on the fractional part of half the displacement (it vanishes for
even-integer displacements) and falls quickly with particle size, to
0.004–0.006 px for 4 px and below 0.003 px for 5 px particle images, which
identifies it as interpolation error of the image deformation on
under-sampled particle images. The uncertainty estimate does not see
systematic error: with `[32, 16, 16]` on 128 × 128 px pairs its median σ is
0.005–0.009 px, so only 18–28% of total errors fall within 1σ and 59–72%
within 2σ. Check systematic error with
[`peak_locking`](@ref) or a ground-truthed [`error_statistics`](@ref), and use
particle images of 4 px or more when errors at the 0.01 px level matter.

## How the estimate is formed

Two details of the covariance sum decide whether σ covers the random error.

- The covariance sums use raw products of the smoothed correlation-difference
  field. The zero-mean condition of the method is the converged correlation
  peak itself. Centring each window's field on its own mean, as releases
  before 0.2 did, forces the sum over all lags to zero and biased the
  truncated ±4 px sum low by about the number of summed lags divided by the
  window's pixel count, a third of the variance for 16 px windows.
- The variance is never taken below its zero-lag term, the independent-pixel
  limit of the method. A truncated sum below that bound is sampling noise in
  a small window. Before 0.2 such sums were clamped to zero and 5–12% of
  16 px windows (1–2% of 32 px windows) reported σ = 0 although their errors
  varied between noise realizations.

Together these raised the 16 px coverage from 42–44% / 74–75% to the values
in the table above. A Bartlett-weighted (triangular) covariance sum, which
also cannot go negative, was evaluated as an alternative; it gave σ 17–20%
below the measured random error and no better coverage.

When comparing σ with an independent error estimate, report the
valid-vector selection and use a median over accepted vectors, since a few
near-outlier windows report very large σ.

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
