# Evaluate uncertainty on seeded synthetic pairs

Run this command from a checkout with its dependencies installed:

```bash
julia --project=. -t 4 bench/validation_uncertainty.jl
```

It writes `uncertainty.toml` and `uncertainty.md` under the gitignored
`bench/profile-output/validation-uncertainty/`. The Markdown table shows pooled
counts, bias/RMS, and observed coverage. The TOML report also includes each
seed's metrics and quantiles, rendered-pixel hashes, every pass setting,
execution residual observations, software/environment identity, and local
processing timings. Use a fresh Julia process and keep the checkout unchanged
throughout. Sources, sizes, hashes, and software environment are captured before
evaluation and checked afterward; unstable evidence requires regeneration.

Coverage measures agreement between the stored random uncertainty and analytic
displacement truth for the specified renderer, conditions, and processing recipe.
The [validation scorecard](validation_scorecard.md) complements these seeded
tests with real A/4E processing observations. Those recordings supply images for
inspection and replay; experimental coverage evaluation requires an independent
displacement reference.

## Conditions and measurement origin

The default runs three fixed seeds for each of four 128 × 128 px conditions:
baseline translation, uniform noise of half-width 0.03, frame-B particle dropout
probability 0.35, and linear shear 0.025 px/px. It also reports one intentionally
empty scene. Baseline density is 0.02 particles/px², diameter is 3 px (`4σ` of
the rendered Gaussian), and displacement at the image center is
`u=2.25, v=-1.5` px. The existing scorecard's specified SplitMix64 stream,
Gaussian particle renderer, and clipping conventions are reused. Noise is
independent uniform additive noise followed by `max(0, intensity)`; its setting
specifies the uniform distribution's half-width. Dropout counts and both
image hashes identify the actual realized pair. Seeds denote scene replicates;
nodes within overlapping windows are correlated.

The recipe uses `[32,16,16]` padded Gaussian-apodized passes and enables final
uncertainty. The final pass uses `n_peaks=1`, `replace_outliers=false`, and
`max_iterations=1`. The final field therefore retains the primary-peak
measurement at each node, including values marked by built-in validators.
Intermediate predictor filling remains part of the processing recipe. The
metric helper evaluates trusted synthetic measurements or explicit test fixtures;
historical outputs require separate evidence of their measurement origin.

Each seed includes actual final primary residual summaries and stop/check
observations. Residuals describe the last correlation sweep, while stopping
observations describe the iteration rule. The residual population consists of
finite, unmasked primary residuals before validation; the error/UQ populations
below use their own selection rules. All eligible nodes enter those metrics
regardless of their residual magnitude.

Particles move in one forward-Euler step. With constant known `v=dv`, the exact
launch-to-midpoint inverse is `y_launch=y_vector-dv/2`; the reference is
`u=du+shear*(y_launch-center), v=dv`. This reference depends entirely on the
prescribed motion and returned coordinates. Component errors are measured minus
reference in px.
Bias is their signed population mean, and RMS includes bias.

## Denominators and unavailable values

Valid yield divides primary-valid nodes by **all unmasked grid nodes**, retaining
rejected and nonfinite measurements in its denominator. Primary validity requires
both displacement components finite and the node unmasked and unflagged. The
report partitions unmasked nodes into primary-valid, flagged finite, flagged
nonfinite, and unflagged nonfinite nodes. Flagged and nonfinite marginal counts
overlap; adding those two marginals would double-count nodes.

UQ populations are selected independently for each component, so usable `σv`
values contribute even at nodes where `σu` is unavailable:

| Metric | Population/denominator |
|---|---|
| Full primary error bias/RMS | All primary-valid nodes |
| UQ-subset error bias/RMS | Primary-valid nodes with that component's finite, nonnegative σ |
| Coverage at 1σ and 2σ | The same UQ subset, including σ=0 |
| Normalized error `z=(measured-truth)/σ` | Primary-valid nodes with that component's finite, strictly positive σ |

Both full and UQ-subset error moments appear together so changes in selection
remain visible. Primary-valid nodes partition into finite nonnegative σ,
nonfinite σ, and negative σ; available σ then partitions into zero and positive
σ. Zero σ is covered only when the error is exactly zero. Zero σ with zero error
and zero σ with nonzero error are counted separately; neither enters normalized
errors, including the undefined `0/0` case. The metrics use σ exactly as stored.
Finite σ above 0.3 px remains counted and carries the estimator's linearization
caution.

Finite measured/reference values can overflow during subtraction, and positive
σ can be too small for a representable normalized error. Such nodes remain in
their measurement/UQ denominators. Affected moments and quantiles are explicitly
unavailable instead of recomputed on a smaller finite subset. Subtraction
overflow also makes coverage unavailable. When a finite error divided by finite
positive σ overflows, that node is known to lie outside 1σ/2σ, while its
normalized moments remain unavailable. Scaled squares keep the moment
calculation representable for large finite errors.

Coverage compares the **full truth error**, including systematic bias, with a
random correlation-uncertainty estimate. Keeping bias in the error makes its
effect on observed 1σ/2σ coverage visible. Interpret the fractions on the stated
synthetic populations: Gaussian 68%/95% reference fractions depend on a Gaussian
error model, while these components and overlapping windows can be correlated.
The report therefore presents empirical coverage directly. See the
[uncertainty explanation](../explanation/uncertainty.md) for estimator assumptions
and the systematic contributions that random correlation uncertainty omits.

## Expand the conditions and repeat timings

```bash
julia --project=. -t 4 bench/validation_uncertainty.jl --expanded --samples=3
```

Expanded mode uses eight fixed seeds and twelve condition groups: the default
four plus densities 0.006/0.04, diameters 2/5 px, noise half-width 0.1,
dropout probability 0.6, shear 0.06, and a baseline `[64,32,32]` window
comparison. These are 96 seeded pairs plus one empty scene. The specified
condition sweep examines each listed perturbation and the window comparison
measures recipe sensitivity. For response across spatial frequencies, use the
[sinusoidal-shear study](spatial_transfer.md).

Pooled counts, coverage numerators, and stable moments stream across seeds with
one seed's field/node-error workspace at a time. Report metadata retains scalar
per-seed summaries. Quantiles are calculated separately for each seed, then
node errors are released. Quantiles describe individual seeds; pooled summaries
contain counts, coverage and moments. Pooled metrics weight nodes, so a seed's
contribution follows its valid-node count.

Each pair runs once for warmup and then `--samples` measured calls (1–10).
Warmed calls reuse the same pair to sample processing performance. Independent
scene replicates come from the distinct seeds. Timings include the loaded-image
CPU PIV call and its
diagnostic callback/source hashing; rendering, setup, metric reduction, report
writing, and the pre-call garbage collection are excluded. GC within the call
is included. Processing uses `backend=:cpu`, `threaded=false`; actual Julia,
FFTW, BLAS thread counts and CPU/OS identity are recorded.

Allocation samples count cumulative Julia-managed bytes allocated during a call;
peak live memory and native-library allocations require separate measurements.
Timing samples describe the recorded local CPU configuration.
`--output=directory` chooses another destination; output
inside the checkout must stay under `bench/profile-output`. Unrelated existing
files, source/fixture aliases, and dangling report symlinks are refused.

## Recorded local evaluation

The fresh default and expanded Windows CPU runs on 2026-10-02 completed with
stable source/environment identities, using Julia 1.11.4 on an Intel i9-12900K
and one warmed timing sample per pair. Their local artifacts are under
`bench/profile-output/validation-uncertainty-default/` and
`bench/profile-output/validation-uncertainty-expanded/`. All twelve shared
condition/seed rows have identical image identities, complete recipes, metrics,
and primary-residual diagnostics across the two runs. Both empty scenes report
`no_valid_measurements` (0/225), with unavailable error/coverage metrics.

| Baseline population | Primary-valid/unmasked | UQ subset u/v | Observed 1σ coverage u/v | Observed 2σ coverage u/v |
|---|---:|---:|---:|---:|
| Default, three seeds | 672/675 | 671/670 | 0.18331/0.15075 | 0.46349/0.53881 |
| Expanded, eight seeds | 1789/1800 | 1785/1782 | 0.18095/0.16554 | 0.45098/0.55780 |

The expanded baseline includes 221 zero-σ u estimates and 181 zero-σ v
estimates with nonzero truth error; those nodes remain in coverage denominators
and are explicitly absent from normalized-error populations. Low full-error
coverage motivates examining systematic terms, residual assumptions and
estimator/renderer sensitivity. The [retained-window audit](diagnostic_uncertainty.md)
exposes the numerical paths behind these estimates.

Selection matters across conditions. At density 0.006, the expanded run retains
1634/1800 primary-valid nodes and UQ subsets of 1500/1504 (u/v). Full-error RMS
is 0.027331/0.029056 px, while UQ-subset RMS is 0.021548/0.023530 px. These
are different populations: the smaller subset RMS describes the nodes for which
uncertainty was available. At dropout probability 0.6, yield falls to 919/1800; 63 u and 79 v
estimates exceed the 0.3 px linearization caution and remain counted. The 32 px
window row uses another grid/population and shows sensitivity to that recipe
change.

Across expanded seeded rows, the single warmed CPU-call samples span
0.069–0.120 s and approximately 41.0–46.6 MB of cumulative Julia allocations,
including diagnostic source hashing. The dedicated core/GUI/docs and Qt probe
processes were paused or finished during these sequential runs. Each pair has one
recorded timing sample; use repeated calls to examine runtime variability.
The allocation totals describe cumulative Julia-managed bytes for these calls.
