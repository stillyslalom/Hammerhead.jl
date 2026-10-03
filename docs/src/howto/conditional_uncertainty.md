# Investigate uncertainty at fixed clean scenes

Run the bench-only conditional study from the repository root:

```sh
julia --project=. bench/conditional_uncertainty.jl --output=bench/profile-output/conditional-uncertainty
```

This performs 26 CPU processing calls without timing samples: two clean controls
(particle-placement seeds 7321 and 7322), then two noise half widths (0.01 and
0.03), with six independent noise realizations for each fixed clean scene and
amplitude. The clean images, `[32,16,16]` pass schedule, Float64 precision, and
truth remain fixed. Noise is additive uniform noise with zero mean **in
distribution**, independent domain-separated A/B streams, and no clipping or
finite-image recentering. Negative intensities are allowed. This differs from
the clipped noise in the original validation scorecard.

The TOML report retains each run's original full-truth-error/stored-random-sigma
metrics, component availability, zero sigma, classifications, complete scientific
recipe, clean/noisy image digests and noise stream identities. It recomputes
production statistics from final retained CPU windows and verifies exact stored
uncertainty reproduction. The independent covariance and moving-block reference
implementations are tests, not a second numerical estimator used for that audit.
On-disk source/environment hashes are checked before and after processing; use a
fresh Julia process with frozen sources. Output guards refuse source/fixture
aliases and unrelated existing report files. Report publication is ordinary
file writing; there is no concurrent-writer or crash-durability guarantee.

For each node accepted as primary in **all six noisy realizations**, the study
reports its conditional mean error and sample variance (denominator five).
Primary selection does not depend on sigma availability. The report counts
nodes lost from every realization to this complete-case intersection, and
separately counts the intersection with accepted clean-control nodes. Clean
error and conditional-mean-minus-clean-error use only this latter population.
These differences are descriptive; noise can change processing, deformation and
acceptance, so they do not isolate a pure systematic term.

Three pre-specified disjoint pairs, `(1,2)`, `(3,4)` and `(5,6)`, compare error
differences against `hypot(sigma_i, sigma_j)`. A fixed conditional mean cancels
algebraically without fitting an empirical center. For independent identical
realizations, the unconditional difference variance is twice the conditional
error variance; acceptance can change that population. Full differences include
all common primary nodes. Sigma availability selects only the coverage subset,
whose denominator and losses remain visible. Zero sigmas remain in coverage.
Paired metrics supplement original full-error metrics and never replace them.

A separately named Bartlett comparator uses **all** covariance lags through
±4, independently of the production ring cutoff, with the pre-specified weight
`(1-abs(dr)/5)*(1-abs(dc)/5)`. On the zero-mean smoothed correlation-difference
field, this covariance sum equals the sum of squared zero-padded moving 5×5
block sums divided by 25, including boundary partial blocks. Tests verify that
identity on rectangular/transposed fields and the actual field used by
production statistics. The comparator uses the same peak-to-displacement
formula. A negative weighted sum is reported as unavailable, with an explicit
summation-only roundoff bound; it is never silently clamped. This bound does not
bound covariance-accumulation error or measurement uncertainty.

Six realizations, three pairs, two scenes and spatially overlapping vectors do
not establish error calibration. Sigma and error share input data; centered or
paired coverage alone cannot prove a calibrated estimator. A clamp accompanied
by positive conditional variance identifies a discrepancy worth investigating,
without establishing its cause or validating Bartlett as a replacement.
Renderer support/quadrature contrasts, sigma floors and production estimator
changes are outside this study. Interpret these diagnostics alongside
[the retained-window audit](diagnostic_uncertainty.md) and the random-uncertainty
conventions of [Wieneke2015](@citet).

The frozen-source default run recorded under the local ignored directory
`bench/profile-output/conditional-uncertainty-final` completed all 26 calls.
Both clean-control scientific rows exactly matched the prior retained-window
audit, and all primary components reproduced stored uncertainty. Complete-case
noisy primary populations were 224/225 and 223/225 nodes for scene 7321 at
half widths 0.01 and 0.03, and 224/225 and 222/225 for scene 7322. Paired
populations can differ from the six-way complete-case population; both remain
explicit in the report.

For scene 7321 at half width 0.01, conditional noise RMS was 0.00951470 px in
u and 0.00714488 px in v. Among its 224 complete nodes, 79 u nodes and 68 v
nodes had a production negative-variance clamp in at least one realization
and positive conditional sample variance across the six realizations. This
co-occurrence shows why a fixed mean alone cannot describe every observed
random discrepancy; it does not establish why sigma and conditional sample
variability differ.
Stored-sigma paired observed 2σ u coverage was 207/224, 200/224 and 198/224;
the Bartlett comparator gave 218/224, 216/224 and 217/224. The original clean
full-error coverage remained 108/224 for stored sigma and 103/224 for Bartlett.
The contrast preserves rather than replaces the original low coverage, and
neither higher paired coverage nor the pre-specified comparator establishes
calibration. The report contains all other scenes, amplitudes, components,
selection losses and original full-error populations.
