# Diagnose synthetic uncertainty estimates

Use this opt-in checkout bench when the uncertainty scorecard reports low
full-error coverage or zero uncertainty with nonzero truth error. It exposes
numerical estimator outcomes and processing sensitivity. It does not change
Hammerhead's random-uncertainty estimator or establish calibrated uncertainty.

From a checkout with its dependencies installed, run:

```bash
julia --project=. --threads=1 bench/diagnostic_uncertainty.jl
```

The default investigation processes baseline, window32 and noise 0.03 scenes
for seeds 7321, 7322, 7323. Five additional contrasts use **only seed 7321**:
three fixed final sweeps, a six-sweep budget with tolerance 0.001, and three
fractional-displacement changes. This is 14 processing calls, each run once.
There are no warmups, independent performance samples or reported timings.

It writes `diagnostic_uncertainty.toml` and `diagnostic_uncertainty.md` under
the ignored `bench/profile-output/diagnostic-uncertainty/`. The table retains
original full-error metrics, UQ availability and observed coverage. TOML adds
complete recipes and image identities, actual sweep/tolerance observations,
classification counts, selected scalar traces, and separately named
supplementary calculations.

Use a fresh Julia process with the checkout unchanged throughout. On-disk
source/environment identities are captured before processing and checked
afterward; an unstable report is refused. These hashes describe files on disk,
not an attestation of previously loaded modules. Reports inside the checkout
must stay under `bench/profile-output`. Unrelated existing destinations,
source/fixture aliases, dangling symlinks and two output paths aliasing the same
file are refused before processing/writing. Concurrent external writers and
crash-safe publication of the two report files are not supported.

For a smaller run:

```bash
julia --project=. --threads=1 bench/diagnostic_uncertainty.jl --seeds=7321 --no-contrasts --output=bench/profile-output/diagnostic-uncertainty-small
```

One to eight distinct nonnegative integer seeds are accepted. A single seed
provides a reproducible case, not between-scene variability. The contrast rows
always use the first selected seed and are not presented as replicated results.

## Preserve the original population

The audit uses the same synthetic renderer and analytic displacement truth as
the [uncertainty scorecard](validation_uncertainty.md). Primary validity requires
both output components finite and the node unmasked and unflagged. The captured
measurement history additionally verifies that these outputs came from the
primary peak. This does not prove that the selected peak is correct.

Coverage always retains the original full truth error, including signed bias.
Each component's denominator includes every primary-valid node with finite,
nonnegative stored random sigma, including zero sigma. Zero sigma covers only
exactly zero error. Positive-sigma normalized errors keep their separate
population; no artificial sigma floor is applied. Nonfinite/negative sigma and
arithmetic-unavailable errors remain counted under the scorecard's conventions.

Centered RMS describes spread around the observed error mean. The supplementary
centered-coverage calculation subtracts the **same seed's UQ-subset mean**;
it is an in-sample sensitivity calculation, not an independently established
bias correction. It never replaces original coverage. Centered coverage and
normalized quantiles are reported per seed; pooled centered moments combine
original errors across seeds rather than averaging seed-centered coverage.
Overlapping vectors are correlated, so no vector-based binomial intervals or
Gaussian 68%/95% acceptance gates are supplied.

## Read the trace

For each selected component, the audit recomputes production statistics from
the retained final CPU windows, then checks the trace against both the
production finalizer and stored sigma. Separate tests use an independent
brute-force pixel-pair covariance reference. A mismatch throws rather than
writing unverified evidence. The trace exposes:

- Raw zero/plus/minus shift correlation sums and covariance offsets.
- The `0.05*S00` threshold, each square ring's maximum, signed contribution,
  inclusion decision and running variance.
- Variance before the production zero clamp, perturbed side values,
  Gaussian-fit curvature denominator and sigma before output conversion.
- An exclusive numerical classification: positive sigma, zero covariance,
  exact covariance cancellation, negative variance clamped to zero, rounded
  side perturbation, rounded logarithmic difference, output-type underflow,
  or an invalid-statistics/curvature/arithmetic outcome.

Classification counts conserve the entire primary component population.
Negative pre-clamp variance is also counted as a marginal observation; it can
coexist with an invalid peak that does not return zero sigma. Examples are the
first two column-major primary nodes per classification, not a representative
sample or a list of all affected nodes. Only scalar trace metadata is retained;
images, vector fields and history arrays are released after each scene.

## Interpret residual alternatives separately

Wieneke's equation 9 describes random correlation uncertainty. Section 3.1 also
discusses a residual equation 4 term added in quadrature for small iteration
oscillations [Wieneke2015](@cite). This bench supplies two **descriptive**
alternatives:

| Alternative | Calculation | Limitation |
|---|---|---|
| Primary residual quadrature | `hypot(stored_random_sigma, primary_residual)` | Residual of the actual last correlation sweep, not a calibrated error term |
| Raw equation4 quadrature | `hypot(stored_random_sigma, raw_eq4_residual)` | Raw side-sum fit differs from the overlap-normalized/magnitude FFT plane |

An alternative is available only when original sigma and its residual term
are available. Its availability counts and coverage denominator remain
explicit; unavailable residuals never silently inherit another population.
These alternatives are not production total uncertainty. A higher coverage
fraction does not establish that an alternative estimates error correctly.

The retained pair comes from the **last executed sweep's deformation
predictor**, before that sweep's residual is added to the displacement. It is
the pair used by UQ for a single final sweep, fixed-budget iterations and
tolerance early exit. Rewarping using the final returned displacement would
produce another pair and is deliberately excluded. A tolerance stop describes
the validated/filled predictor-field change, not certified per-node zero
residual or truth accuracy.

## Scope and conventions

The first audit slice supports CPU multipass cross correlation, equal window
and search sizes, `gauss3`, primary-only output and no replacement in the final
result. Intermediate predictor filling remains part of the recipe. It uses no
ROI, mask, preprocessing or physical scale. Single-pass nondeforming workspace
buffers are not audited, and KA/device deformation contexts are outside the
retained-window adapter. Tests compare the KA statistics kernels on the same
fixed signals, isolating covariance arithmetic from deformation differences.

The paper defines particle image size as 2sigma, whereas this renderer's
diameter is 4sigma: renderer diameter 3px corresponds to paper size 1.5px.
Pixel-center sampling, finite particle/window support, mean subtraction,
Gaussian weights on both inputs and the padded plane's overlap gain need to be
considered before comparing numerical studies. The paper's threshold wording
does not uniquely specify Hammerhead's whole-square-ring stopping rule. This
bench exposes that rule without choosing a replacement.

Quadrature renderers, a factorial study, spatial-resolution transfer, GPU
hardware and independent experimental displacement truth are deferred.
Neither a bias aggregate nor a clamp count alone identifies the cause of low
coverage. See the [uncertainty explanation](../explanation/uncertainty.md) for
the underlying estimator's assumptions.

## Recorded local investigation

The fresh Windows CPU run on 2026-10-02 completed all 14 cases with stable
source/environment identities and verified retained-window reproduction.
Ignored local evidence is under
`bench/profile-output/diagnostic-uncertainty-final/`. The accompanying
`shared_row_comparison.json` records hashes and comparisons with the existing
default/expanded uncertainty scorecards. All 15 shared-row comparisons match
input identities, complete scientific recipes and original metrics exactly:
six against the default report and nine against the expanded report. Workspace
and callback capture metadata are explicitly excluded from the recipe
comparison; no scientific pass setting is excluded.

| Three-seed population | Primary-valid / unmasked | Component | UQ available | Zero sigma from negative variance clamp | Original full-error 2sigma coverage |
|---|---:|---|---:|---:|---:|
| Baseline | 672/675 | u | 671 | 81 | 311/671 (0.46349) |
| Baseline | 672/675 | v | 670 | 71 | 361/670 (0.53881) |
| Window32 | 147/147 | u | 147 | 19 | 15/147 (0.10204) |
| Window32 | 147/147 | v | 147 | 22 | 6/147 (0.04082) |
| Noise0.03 | 671/675 | u | 671 | 33 | 465/671 (0.69300) |
| Noise0.03 | 671/675 | v | 669 | 40 | 456/669 (0.68161) |

Every available zero sigma in these three groups was produced by a negative
pre-clamp covariance sum. No zero covariance, exact cancellation or rounding
classification occurred. Invalid peak curvature accounts for the remaining
unavailable component estimates: baseline has one u and two v cases; noise has
two v cases. These observations identify how the numerical zeros arise in
these recipes, not why the covariance estimate is negative or which alternative
would estimate uncertainty correctly. The independent tests validate the
covariance arithmetic and classification, rather than select a covariance
truncation policy.

The first-seed three-sweep budget and six-sweep tolerance case both executed
three final sweeps and produced identical original error/UQ summaries; their
stop reasons were respectively `iteration_budget` and
`tolerance_condition_met`. The phase rows vary the same seed's displacement
without changing the renderer. Their differences describe sensitivity of this
renderer/processing combination and do not isolate subpixel fitting from
rendering, deformation, weighting or boundary effects. Centered and residual
alternatives remain separately named in TOML; their coverage does not replace
any fraction in the table above. No production estimator change follows from
this bounded investigation.

Run the focused covariance/classification and retained-window tests with:

```bash
julia --project=. --threads=1 test/test_diagnostic_uncertainty.jl
```
