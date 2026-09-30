# Tune validation

**Goal:** adjust outlier detection when plausible vectors are flagged or
spurious vectors survive. Compare flags with the particle images and with
the flow structures you expect to resolve.

## What runs by default

Every [`run_piv`](@ref) pass applies, in order:

1. **Universal outlier detection (UOD)** (normalized median test,
   [WesterweelScarano2005](@cite)) — `uod_threshold = 2.0`,
   `uod_neighborhood = 2` (a 5×5 neighborhood), denominator floor
   `epsilon = 0.1` px.
2. **Peak-ratio check** — disabled by default (`min_peak_ratio = 1.0`).
3. **Peak substitution** — flagged vectors are re-tested against their
   secondary/tertiary correlation peaks (`n_peaks = 3`); a locally
   consistent alternative is accepted as measured data and unflagged.
4. **Local-median replacement** — remaining flagged vectors are replaced
   (`replace_outliers = true`; intermediate multi-pass passes always
   replace). A pass with `max_iterations > 1` then *re-measures* replaced
   vectors, re-deforming by the corrected field and re-correlating until it
   converges or reaches `max_iterations`. See
   [iterative passes](../explanation/multipass.md#Convergence-sweeps-and-iterative-passes).

## If too many good vectors are flagged

- **Raise `uod_threshold`** (e.g. 2.0 → 3.0). Higher is less sensitive.
- **Keep `uod_neighborhood = 2`.** The 5×5 neighborhood exists because 3×3
  can falsely flag smooth gradients at field edges. Try adjusting the
  threshold before shrinking the neighborhood.
- **Keep `epsilon` near its 0.1 px default unless you have a reason to
  change it.** This floor stabilizes the normalized residual when neighboring
  vectors are nearly identical. Near zero, small measurement differences can
  produce large scores even in a uniform field.

## If spurious vectors survive

- **Try the peak-ratio check**: `min_peak_ratio = 1.3` is a starting value.
  Compare flagged vectors with their image windows; a low ratio can also
  occur where the flow has a steep gradient or few particle images.
- **Add validators** via the `validation` parameter, as `Symbol => value`
  specs or validator objects:

```julia
params = PIVParameters(
    min_peak_ratio = 1.3,
    validation = (
        :velocity_magnitude => (max = 12,),         # px per frame interval
        :correlation_moment => 4.0,                 # peak too broad
    ),
)
```

## The validators, in one place

Every validator can be given either as a validator object or as a
`Symbol => value` **pair spec** in the `validation` tuple. The full set:

| Validator | Flags a vector when… | Pair spec | Default / recommended |
|---|---|---|---|
| [`UniversalOutlierValidator`](@ref) | its normalized median residual vs. its neighbors exceeds `threshold` (the default UOD; runs even without a spec) | `:uod => (threshold = 2.0, neighborhood_size = 2, epsilon = 0.1)` (alias `:universal_outlier`; only `threshold` is required) | `threshold = 2.0`, `neighborhood_size = 2` (5×5), `epsilon = 0.1` px |
| [`PeakRatioValidator`](@ref) | its correlation peak ratio is below `threshold` (`NaN` too) | `:peak_ratio => threshold` | off (`1.0`); `1.3` to enable |
| [`CorrelationMomentValidator`](@ref) | its correlation peak is broader than `threshold` (`NaN` too) | `:correlation_moment => threshold` | data-dependent; e.g. `4.0` |
| [`VelocityMagnitudeValidator`](@ref) | its displacement magnitude is outside `[min, max]` px (`NaN` too) | `:velocity_magnitude => (min = 0, max = 50)` (`min` defaults to `0`) | set `max` to your physical limit |

The UOD and peak-ratio checks have dedicated `PIVParameters` keywords
(`uod_threshold`/`uod_neighborhood`, `min_peak_ratio`); the rest go in the
`validation` tuple. See [`validate_vectors!`](@ref) for how a pipeline is
applied.

For time-resolved sequences, [`validate_temporal!`](@ref) can flag a vector
that looks plausible spatially but disagrees with its time history. It needs
at least three valid samples at a grid point. Use it after processing and
before statistics; inspect newly flagged samples rather than assuming every
transient is an error.

## Keep or replace outliers?

`replace_outliers = false` (final pass) leaves flagged vectors holding
their measured values, with `result.outliers` telling you which they are —
appropriate when you want to apply your own replacement or reject them
outright. For a smoothness-based fill that also interpolates gaps, use
[`smoothn`](@ref) [Garcia2010](@cite) with the outlier/mask flags as
weights:

```julia
w = .!(result.outliers .| result.mask)
u_smooth = smoothn(result.u; weights = w).z
```

This produces an interpolated value at masked and flagged cells. Keep the
original `result.mask` and `result.outliers` flags alongside the smoothed
field so filled values are not mistaken for measurements.

## Judge the tuning

Count flags relative to windows that produced a measurement, then look at
*where* they occur:

```julia
eligible = count(.!result.mask)
flag_fraction = eligible == 0 ? NaN : count(result.outliers) / eligible
```

Look for flags concentrated where image quality is poor, such as reflections
or particle dropout. Compare settings on the same representative pairs and
record both the flag fraction and the retained flow features. If flags
follow shear layers or other plausible structures, inspect their image
windows before changing the threshold; smooth physical gradients can differ
from their neighbors without being bad measurements.
