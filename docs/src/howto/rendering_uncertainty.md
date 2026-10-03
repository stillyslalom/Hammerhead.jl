# Investigate rendering and interpolation sensitivity

From the repository root, run the fixed clean-image diagnostic:

```sh
julia --project=. bench/rendering_uncertainty.jl --output=bench/profile-output/rendering-uncertainty
```

The command makes twelve CPU PIV calls: two fixed particle placements (seeds
7321 and 7322), two constant translations (the original `(2.25,-1.5)` and an
integer-half-warp sentinel `(2,-2)`), and three rendering policies. Each call
evaluates a clean image pair with the existing production uncertainty estimator.
The scientific recipe remains Float64, `[32,16,16]`, primary-only final output
with stored random uncertainty and final retained-window reproduction checks.

The unchanged production policy samples Gaussian intensity at pixel centers.
Its nominal support parameter is 3σ, but its actual radius is `ceil(3σ)` and
its square bounds use `round(Int, center ± radius)`, clipped to the image. It is
a rounded square support in image coordinates. The wider point policy keeps the identical
point formula on common pixels and uses `ceil(6σ)`. The third policy integrates
the same Gaussian over each unit pixel area on that wider rounded bbox.
Continuous σ is 0.75 px for all policies: renderer diameter 3 px means 4σ,
whereas the paper's particle-size convention is 2σ. Pixel integration changes
the pixel response while preserving the prescribed particle displacement and
continuous Gaussian width. Intensities and flux retain their rendered values,
including finite-frame and support losses.

Pixel-area integration is separable Gauss–Legendre quadrature, generated using
stdlib `LinearAlgebra`. A fixed 16-node rule is compared with 32 nodes before
each area-rendering accuracy call; a maximum image difference above `1e-10`
refuses the report. Tests
check polynomial moments, positivity/symmetry, Gaussian scalar integrals against
an independent BigFloat composite-Simpson reference, full-support particle flux,
transpose symmetry and integer translation. Together these checks assess
quadrature convergence and the implemented geometry. The quadrature construction
uses public stdlib operations.

Each renderer retains its original full-truth-error/stored-sigma populations,
including zero sigma, unavailable components and coverage denominators. A
separately named common-primary intersection selects nodes accepted by all three
renderers, independently of sigma availability; per-renderer losses and component
availability are explicit. On this intersection the study measures deterministic
field differences directly, using the fixed common particle placement. The original-renderer baseline images
and complete scientific rows remain unchanged. Algebraically derived predictor
error is calculated as `full truth error - observed primary residual`. It is an
algebraic quantity derived from those two recorded displacement observations.

The separate oracle-deformation lane uses the **known constant translation**
only for diagnostics. Production cubic B-spline interpolation evaluates
`A(r-dv/2,c-du/2)` and `B(r+dv/2,c+du/2)`. The analytic comparison rerenders
particle centers at the midpoint `+(du/2,dv/2)`. Sign/row/column conventions are
tested independently with affine fields. Full-support sampled Gaussian images
and their finite-bbox counterparts are compared against direct midpoint fields.
The oracle lane evaluates image interpolation separately; PIV calls continue
to use their estimated production predictors.

Oracle errors retain both full-frame and fixed-interior populations: the interior
removes 16 pixels from each edge, so the default denominators are 16384 and 9216.
Finite frames, zero extrapolation and spline prefilter boundaries affect even
full-support sampled fields. Comparing full-frame and interior errors shows
the effect of those boundary conditions. `(2,-2)` makes the known oracle
half-warps integer; estimated PIV predictors retain their measured, potentially
fractional values.

Reports contain complete recipes, particle-placement and image hashes, quadrature
nodes/weights, support policies, retained-window traces and source/environment
identity. Use a fresh process with frozen sources. Output guards protect sources,
fixtures and unrelated existing reports. Use a dedicated output directory
for each study.

Read the results as deterministic sensitivity to support, pixel response and
interpolation within this specified clean-image pipeline. The
[conditional noise study](conditional_uncertainty.md) examines repeated random
perturbations, and the [retained-window audit](diagnostic_uncertainty.md) exposes
estimator arithmetic. Together they retain the original scorecard populations
and the random-uncertainty conventions of [Wieneke2015](@citet).

The frozen Windows/Julia 1.11.4 evidence run wrote
`bench/profile-output/rendering-uncertainty-final/rendering_uncertainty.{toml,md}`.
All twelve retained-window reproductions passed, 16/32-node image differences
were at most `2.89e-15`, and the complete original-shift production scientific
rows exactly matched both earlier diagnostic and conditional clean controls.
The dedicated suite passed 219 checks; the final report passed 552 population,
source-identity, quadrature and baseline-equality checks. These results establish
the recorded Windows/Julia configuration and its specified inputs.

For the original translation, all renderers shared 224/225 primary nodes for
each scene, with no additional intersection losses. The RMS of the deterministic
renderer difference was approximately `1.25e-6` to `2.03e-6` px for wider point
support, and `0.00301` to `0.00347` px for area versus wider point sampling. This
is the RMS of a difference, distinct from the change in full-truth-error RMS.
Area sampling reduced full-error RMS in these four components but decreased
their stored-sigma full-error coverage:

| Scene | Component | Production full-error 2σ | Area full-error 2σ |
|---|---|---:|---:|
| 7321 | u | 108/224 | 81/224 |
| 7321 | v | 130/224 | 92/224 |
| 7322 | u | 97/223 | 79/224 |
| 7322 | v | 120/224 | 93/224 |

The production component has 223 available sigmas, while its common-primary
error population retains all 224 nodes.
Integer-shift full-support oracle interior RMS was approximately `1e-16`, while
original fractional-shift interior RMS was `0.0102`–`0.0108` for point sampling
and `0.00677`–`0.00718` for area sampling. These oracle residuals are image
intensity differences. The displacement-error metrics use pixels as their units
and retain their own coverage observations. Full-frame oracle errors remained
nonzero, making the boundary/interior contrast visible even for the integer
sentinel.
