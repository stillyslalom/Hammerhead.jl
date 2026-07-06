# Tune validation

**Goal:** adjust outlier detection when the defaults flag too much (smooth
fields peppered with false positives) or too little (spurious vectors
surviving into your statistics).

## What runs by default

Every [`run_piv`](@ref) pass applies, in order:

1. **Universal outlier detection** (normalized median test,
   [WesterweelScarano2005](@cite)) — `uod_threshold = 2.0`,
   `uod_neighborhood = 2` (a 5×5 neighborhood), noise floor
   `epsilon = 0.1` px.
2. **Peak-ratio check** — disabled by default (`min_peak_ratio = 1.0`).
3. **Peak substitution** — flagged vectors are re-tested against their
   secondary/tertiary correlation peaks (`n_peaks = 3`); a locally
   consistent alternative is accepted as measured data and unflagged.
4. **Local-median replacement** — remaining flagged vectors are replaced
   (`replace_outliers = true`; intermediate multi-pass passes always
   replace).

## If too many good vectors are flagged

- **Raise `uod_threshold`** (e.g. 2.0 → 3.0). Higher is less sensitive.
- **Keep `uod_neighborhood = 2`.** The 5×5 neighborhood exists because 3×3
  falsely flags smooth gradients at field edges — shrinking it is rarely
  the right fix.
- **Don't lower `epsilon` below ~0.1 px.** It represents the physical
  subpixel noise floor; with near-zero `epsilon`, a *uniform* flow field
  gets flagged wholesale because the neighbor residuals are pure noise.

## If spurious vectors survive

- **Enable the peak-ratio check**: `min_peak_ratio = 1.3` is a reasonable
  starting point (a peak barely taller than the noise peak is unreliable).
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

See [`validate_vectors!`](@ref) and the validator types
([`UniversalOutlierValidator`](@ref), [`PeakRatioValidator`](@ref),
[`CorrelationMomentValidator`](@ref),
[`VelocityMagnitudeValidator`](@ref)) for the accepted forms.

- **For time-resolved sequences**, add [`validate_temporal!`](@ref) after
  processing: a vector consistent with its spatial neighbors can still be
  exposed by the point's time history.

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

## Judge the tuning

Count flags, and look at *where* they are:

```julia
count(result.outliers) / length(result.outliers)   # flag fraction
```

A well-tuned setup flags a few percent, concentrated where the image data
is genuinely poor (edges, reflections, dropout), not tracing the contours
of real flow structures. If flags follow your shear layers, validation is
too tight.
