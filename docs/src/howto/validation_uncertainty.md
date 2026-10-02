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

This evaluates the specified synthetic renderer, conditions, and processing
recipe. It does not establish experimental accuracy or universally calibrated
uncertainty. The [separate validation scorecard](validation_scorecard.md) provides
real A/4E smoke observations; those fixtures have no supplied displacement truth
for uncertainty coverage. Known-motion recordings with independent motion
references, independent datasets beyond A/4E, stereo coverage, calibration and
timing uncertainty, peak memory, and vendor GPU hardware remain unavailable.

## Conditions and measurement origin

The default runs three fixed seeds for each of four 128 × 128 px conditions:
baseline translation, uniform noise of half-width 0.03, frame-B particle dropout
probability 0.35, and linear shear 0.025 px/px. It also reports one intentionally
empty scene. Baseline density is 0.02 particles/px², diameter is 3 px (`4σ` of
the rendered Gaussian), and displacement at the image center is
`u=2.25, v=-1.5` px. The existing scorecard's specified SplitMix64 stream,
Gaussian particle renderer, and clipping conventions are reused. Noise is
independent uniform additive noise followed by `max(0, intensity)`; its setting
is a half-width, not Gaussian standard deviation or SNR. Dropout counts and both
image hashes identify the actual realized pair. Seeds denote scene replicates;
nodes within overlapping windows are correlated.

The recipe uses `[32,16,16]` padded Gaussian-apodized passes and enables final
uncertainty. The final pass uses `n_peaks=1`, `replace_outliers=false`, and
`max_iterations=1`. These choices isolate the final primary measurement:
alternative peaks are not substituted and the final field is not filled from
neighbors. Intermediate predictor filling remains part of the specified
processing recipe. Only built-in validators that mark flags are accepted.
The metric helper assumes trusted synthetic measurements or explicit test
fixtures; persisted parameters alone do not certify the origin of arbitrary
historical results.

Repeating a final window does **not prove zero-residual convergence**. Actual
final primary residual summaries and stop/check observations accompany each
seed. Their population is finite, unmasked primary residuals before validation,
which differs from the error/UQ populations. No aggregate residual certifies
individual nodes, and this command imposes no hidden residual-based selection.

Particles move in one forward-Euler step. With constant known `v=dv`, the exact
launch-to-midpoint inverse is `y_launch=y_vector-dv/2`; the reference is
`u=du+shear*(y_launch-center), v=dv`. It never uses measured displacement to
construct the reference. Component errors are measured minus reference in px.
Bias is their signed population mean, and RMS includes bias.

## Denominators and unavailable values

Valid yield divides primary-valid nodes by **all unmasked grid nodes**, retaining
rejected and nonfinite measurements in its denominator. Primary validity requires
both displacement components finite and the node unmasked and unflagged. The
report partitions unmasked nodes into primary-valid, flagged finite, flagged
nonfinite, and unflagged nonfinite nodes. Flagged and nonfinite marginal counts
overlap; adding those two marginals would double-count nodes.

UQ populations are separate for each component. An unavailable `σu` does not
remove a usable `σv`:

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
errors, including the undefined `0/0` case. No artificial σ floor is applied.
Finite σ above 0.3 px is counted and retained with the estimator's linearization
caution, rather than silently removed.

Finite measured/reference values can overflow during subtraction, and positive
σ can be too small for a representable normalized error. Such nodes remain in
their measurement/UQ denominators. Affected moments and quantiles are explicitly
unavailable instead of recomputed on a smaller finite subset. Subtraction
overflow also makes coverage unavailable. When a finite error divided by finite
positive σ overflows, that node is known to lie outside 1σ/2σ, while its
normalized moments remain unavailable. Large representable errors use scaled
squares so squaring alone does not fabricate overflow.

Coverage compares the **full truth error**, including systematic bias, with a
random correlation-uncertainty estimate. Bias is not silently subtracted to
improve coverage. Reported 1σ/2σ fractions are observed synthetic coverage;
there is no Gaussian 68%/95% pass/fail target, independence assumption between
components, or binomial confidence interval treating overlapping vectors as
independent observations. See the [uncertainty explanation](../explanation/uncertainty.md)
for estimator assumptions and omitted systematic contributions.

## Expand the conditions and repeat timings

```bash
julia --project=. -t 4 bench/validation_uncertainty.jl --expanded --samples=3
```

Expanded mode uses eight fixed seeds and twelve condition groups: the default
four plus densities 0.006/0.04, diameters 2/5 px, noise half-width 0.1,
dropout probability 0.6, shear 0.06, and a baseline `[64,32,32]` window
comparison. These are 96 seeded pairs plus one empty scene. This is a specified
condition sweep, not a factorial study or an experimental validity envelope.
The window comparison describes processing sensitivity; linear shear does not
measure a spatial-resolution transfer function.

Pooled counts, coverage numerators, and stable moments stream across seeds with
one seed's field/node-error workspace at a time. Report metadata retains scalar
per-seed summaries. Quantiles are calculated separately for each seed, then
node errors are released. There are **no pooled quantiles** and per-seed
quantiles are not averaged into purported pooled quantiles. Pooled metrics
weight nodes; they do not give every seed equal weight when valid counts differ.

Each pair runs once for warmup and then `--samples` measured calls (1–10).
Warmed calls reuse that pair and are performance samples, **not independent
accuracy replicates**. Timings include the loaded-image CPU PIV call and its
diagnostic callback/source hashing; rendering, setup, metric reduction, report
writing, and the pre-call garbage collection are excluded. GC within the call
is included. Processing uses `backend=:cpu`, `threaded=false`; actual Julia,
FFTW, BLAS thread counts and CPU/OS identity are recorded.

Allocation samples are cumulative Julia bytes during a call, **not peak
host/device memory**; native-library allocations may be absent. Timing
variability is descriptive local evidence, not an accuracy gate or hardware
performance promise. `--output=directory` chooses another destination; output
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
coverage is an observation requiring further diagnosis. Omitted systematic
terms, residual assumptions, and estimator/renderer effects are candidates;
this evaluation does not decompose their contributions or identify one cause.

Selection matters across conditions. At density 0.006, the expanded run retains
1634/1800 primary-valid nodes and UQ subsets of 1500/1504 (u/v). Full-error RMS
is 0.027331/0.029056 px, while UQ-subset RMS is 0.021548/0.023530 px. These
are different populations, so the smaller subset RMS does not establish better
accuracy. At dropout probability 0.6, yield falls to 919/1800; 63 u and 79 v
estimates exceed the 0.3 px linearization caution and remain counted. The 32 px
window row uses another grid/population and describes recipe sensitivity, not a
spatial-resolution or universal uncertainty-calibration claim.

Across expanded seeded rows, the single warmed CPU-call samples span
0.069–0.120 s and approximately 41.0–46.6 MB of cumulative Julia allocations,
including diagnostic source hashing. The dedicated core/GUI/docs and Qt probe
processes were paused or finished during these sequential runs. One timing
sample per pair does not characterize runtime variability, and these allocation
totals are not peak memory or a production-scale hardware promise.
