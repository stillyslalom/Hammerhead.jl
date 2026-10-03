# Investigate uncertainty at fixed clean scenes

Run the bench-only conditional study from the repository root:

```sh
julia --project=. bench/conditional_uncertainty.jl --output=bench/profile-output/conditional-uncertainty
```

This performs 26 CPU processing calls: two clean controls
(particle-placement seeds 7321 and 7322), then two noise half widths (0.01 and
0.03), with six independent noise realizations for each fixed clean scene and
amplitude. The clean images, `[32,16,16]` pass schedule, Float64 precision, and
truth remain fixed. Noise is additive uniform noise with zero mean **in
distribution**, using independent domain-separated A/B streams. Its realized
values, including negative intensities and finite-image mean deviations, are
retained exactly. This differs from
the clipped noise in the original validation scorecard.

The TOML report retains each run's original full-truth-error/stored-random-sigma
metrics, component availability, zero sigma, classifications, complete scientific
recipe, clean/noisy image digests and noise stream identities. It recomputes
production statistics from final retained CPU windows and verifies exact stored
uncertainty reproduction. The independent covariance and moving-block reference
implementations provide independent test oracles for the arithmetic.
On-disk source/environment hashes are checked before and after processing; use a
fresh Julia process with frozen sources. Output guards refuse source/fixture
aliases and unrelated existing report files. Use a dedicated output directory
for each study.

For each node accepted as primary in **all six noisy realizations**, the study
reports its conditional mean error and sample variance (denominator five).
Primary selection uses the measurement flags and finite components, independently
of sigma availability. The report counts
nodes lost from every realization to this complete-case intersection, and
separately counts the intersection with accepted clean-control nodes. Clean
error and conditional-mean-minus-clean-error use only this latter population.
These differences show the combined effect of noise on processing, deformation
and acceptance in the selected population.

Three pre-specified disjoint pairs, `(1,2)`, `(3,4)` and `(5,6)`, compare error
differences against `hypot(sigma_i, sigma_j)`. A fixed conditional mean cancels
algebraically, giving a comparison based directly on the paired errors. For independent identical
realizations, the unconditional difference variance is twice the conditional
error variance; acceptance can change that population. Full differences include
all common primary nodes. Sigma availability selects only the coverage subset,
whose denominator and losses remain visible. Zero sigmas remain in coverage.
Paired and original full-error metrics are reported together.

A separately named Bartlett comparator uses **all** covariance lags through
±4, independently of the production ring cutoff, with the pre-specified weight
`(1-abs(dr)/5)*(1-abs(dc)/5)`. On the zero-mean smoothed correlation-difference
field, this covariance sum equals the sum of squared zero-padded moving 5×5
block sums divided by 25, including boundary partial blocks. Tests verify that
identity on rectangular/transposed fields and the actual field used by
production statistics. The comparator uses the same peak-to-displacement
formula. A negative weighted sum is reported as unavailable, with an explicit
summation-only roundoff bound. That bound describes rounding in the weighted
sum; covariance accumulation and measurement uncertainty have separate errors.

The study describes six realizations, three pairs and two fixed scenes, with
spatially overlapping vectors. Sigma and error share the same input data, and
the complete-case rules select the reported populations. A production clamp
alongside positive conditional variance highlights a numerical estimate that
differs from observed repeated-noise variability. Bartlett supplies a second,
pre-specified covariance calculation for examining that discrepancy. Interpret
these diagnostics alongside
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
co-occurrence pairs a zero production estimate with positive repeated-noise
variance. The ring contributions and Bartlett calculation provide further
diagnostics of this difference.
Stored-sigma paired observed 2σ u coverage was 207/224, 200/224 and 198/224;
the Bartlett comparator gave 218/224, 216/224 and 217/224. The original clean
full-error coverage remained 108/224 for stored sigma and 103/224 for Bartlett.
Paired coverage and clean full-error coverage describe different questions and
populations. Keep both in view when interpreting the contrast. The report contains
all other scenes, amplitudes, components, selection losses and original full-error
populations.
