# Measure a specified pipeline's spatial response

From a checkout with dependencies installed, run the fixed clean-image study:

```sh
julia --project=. bench/spatial_transfer.jl --output=bench/profile-output/spatial-transfer
```

The command makes sixteen CPU PIV calls: two fixed 256 × 256 px particle scenes
(seeds 7321 and 7322), a translation control plus three sinusoidal transverse
shears, and two complete processing schedules. Each image has particle density
0.04, Gaussian diameter 3 px (4σ), peak intensity 1, no noise and no dropout.
The production point-sampling renderer and its clipped rounded square bbox are
unchanged. At σ=0.75 the actual radius is `ceil(3σ)=3` px, with rounded square
support. Each pixel receives the point-sampled intensity at its center, and
intensities retain their rendered values.

Particles in the sinusoidal cases move by the prescribed launch displacement

```math
u_\mathrm{launch}=2.25+0.5\sin\left(2\pi\frac{y_\mathrm{launch}-128.5}{\lambda}+\pi/4\right),
\qquad v=-1.5,
```

at wavelengths 128, 64 and 32 px. Each particle takes one discrete displacement
step evaluated at its launch position.
Frame B is rendered from displaced particle positions, rather than warping
frame A. The constant transverse displacement gives an exact midpoint reference:
`y_launch = y_vector - v/2`. Analytic truth is evaluated at every returned
coordinate from the prescribed motion. The processing pipeline builds its own
predictors from the measured images.

Both schedules start with 64 px windows/32 px overlap and 32 px windows/16 px
overlap. Their last two windows are respectively `[16,16]` or `[32,32]`, with
overlaps 8 or 24 px. Thus both terminal grids have an 8 px stride; the 32 px
wavelength has four grid samples per period, rather than being at the spatial
Nyquist limit. All passes use padded cross correlation, Gaussian apodization,
one peak, no outlier replacement and one sweep. The final pass enables stored
random uncertainty. Predictor smoothing and the default validation settings
remain part of processing. Every `PIVParameters` field and driver setting is
recorded. The measured response belongs to these complete terminal-window
schedules, including deformation, smoothing and validation.

The descriptive least-squares model is
`intercept + s*sin(θ) + c*cos(θ)`, with θ from the known midpoint reference.
For prescribed amplitude A=0.5, report `s/A` as signed in-phase response, `c/A`
as quadrature response, `hypot(s,c)/A` as amplitude response and `atan(c,s)`
as phase in radians (positive means a phase lead). The same v coefficients,
normalized by the prescribed **u** amplitude, describe the v output's coupling
to the imposed u harmonic. The translation control has A=0:
absolute fitted coefficients and harmonic leakage are reported at all three
analysis wavelengths, while normalized gain and response phase are unavailable.

Fits retain every selected sample and fail explicitly for too few observations,
nonfinite arithmetic, deficient rank or poor conditioning. The relative SVD
rank threshold is `max(N,3)*eps(Float64)*largest_singular_value`; the condition
cap is `1e8`. Scaling the output values before solving avoids unnecessary
overflow, and coefficient/residual checks still gate availability. Fit,
normalization and phase availability remain separate: a zero harmonic has no
defined phase, and failed fits retain an unavailable gain. The report records
rank, conditioning, sample counts and fit residuals. These are descriptive
summaries of the stated, spatially correlated sample populations.

Original full-grid error/yield/UQ populations remain independent of fits.
Primary origin is checked against the actual final recipe: one peak, no final
replacement, one sweep and stored UQ. Report counts retain masked, flagged and
nonfinite vectors. Yield uses all unmasked nodes as its denominator; error
metrics and fits select finite, unmasked, unflagged primary vectors.
Zero sigma and component unavailability are retained. Coverage uses the full
truth error, including its original bias and harmonic response.

A fixed interior retains coordinates `64 ≤ x,y ≤ 192`. The two terminal grids
share 256 candidate interior coordinates exactly. Own-primary fits and metrics
are reported separately from paired common-primary fits; shared candidates and
losses to the intersection are explicit. Exact common coordinates determine the
intersection; sigma availability is assessed separately for coverage. Excluded
analysis nodes are represented by a temporary metric mask and labeled as
analysis exclusions; the stored result remains unchanged. Full-grid boundary
errors are still visible.

The TOML report stores image/particle hashes, exact motion/renderer conventions,
full recipes, actual diagnostics, complete populations and source/environment
identity. Markdown keeps full-error/yield/coverage alongside the paired response
and control leakage. Run in a fresh Julia process with frozen sources. Output
guards protect sources, fixtures and unrelated existing report files. Use a
dedicated output directory for each run. The report contains scientific response
and population metrics.

The study measures finite-amplitude response at three spatial frequencies,
using two particle placements, one phase and one orientation. Particle support,
finite-frame/interpolation effects, predictor smoothing and validation all
contribute to the response of each complete pipeline. Selection can vary with
frequency, so read each response with its accepted population and yield losses.
The original [scorecard](validation_scorecard.md),
[uncertainty sweep](validation_uncertainty.md) and
[rendering contrast](rendering_uncertainty.md) provide complementary error and
image-formation observations.

The frozen Windows/Julia 1.11.4 evidence run is stored at
`bench/profile-output/spatial-transfer-final/spatial_transfer.{toml,md}`. The
focused suite passed 209 checks; independent inspection of the final TOML
passed 2366 population, fit, recipe, image-identity and source-stability checks.
After a separate core report-validation guard changed source identities, a
fresh 16-call regeneration exactly reproduced all eight scientific groups,
including every original metric and fitted response. The numerical estimator
and production defaults remained unchanged. These results describe the recorded Windows
configuration.

The signed u in-phase responses below are conditional on the stated common
interior populations. Full-grid primary yields retain their different schedule
denominators:

| Seed | Wavelength px | Full primary W16 | Full primary W32 | Common interior | Interior primary lost W16/W32 | Signed u response W16/W32 |
|---|---:|---:|---:|---:|---:|---:|
| 7321 | 128 | 961/961 | 841/841 | 256/256 | 0/0 | 1.0312/1.0300 |
| 7322 | 128 | 961/961 | 841/841 | 256/256 | 0/0 | 1.0324/1.0315 |
| 7321 | 64 | 958/961 | 841/841 | 256/256 | 0/0 | 1.0474/0.9981 |
| 7322 | 64 | 961/961 | 841/841 | 256/256 | 0/0 | 1.0551/1.0164 |
| 7321 | 32 | 289/961 | 423/841 | **43/256** | **15/76** | 0.7608/0.5949 |
| 7322 | 32 | 325/961 | 472/841 | **59/256** | **10/72** | 0.7490/0.5243 |

At 32 px, own-interior primary counts were 58/256 versus 119/256 for seed 7321,
and 69/256 versus 131/256 for seed 7322. The common sets therefore discard
substantial populations before fitting. Each gain describes the surviving
common nodes: 43 for seed 7321 and 59 for seed 7322. Fit conditioning remained
acceptable on those subsets. All common component sigmas were available in this
run; fit selection was based on primary measurements.

Responses slightly above one at longer wavelengths are retained as observations.
The control preserves full-grid errors and zero-sigma coverage separately from
its small absolute harmonic leakage. Both quantities are reported as measured.
At the shortest wavelength, the response reflects the finite-amplitude pipeline
and its substantial validation losses together. Comparing full-grid yield,
own-interior fits and common-interior fits shows which populations contribute
to that observation.
