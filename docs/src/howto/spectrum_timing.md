# Analyze spectra with explicit sample times

An FFT spectrum requires uniformly spaced samples. Use `sample_times` to check
the times explicitly before calculating a spectrum. No acquisition timestamp is
discovered from a source, result or timing companion. The samples must already
represent one chosen time coordinate.

```@example spectrum_timing
using Hammerhead

n = 64
signal = sin.(2pi*8 .* ((0:n-1)./n))
epoch = big(typemax(Int64)) + 100
times = [epoch + k//4 for k in 0:n-1]
spectrum = power_spectrum(signal; sample_times=times, time_unit="s",
    window=:none, return_timing=true)
@assert spectrum.frequencies[argmax(spectrum.psd)] == 0.5
@assert spectrum.timing["uniformity"] == "exact"
@assert spectrum.timing["period"]["numerator"] == "1"
@assert spectrum.timing["period"]["denominator"] == "4"
spectrum.timing
```

Supported sample times are finite standard signed/unsigned integers including
BigInt, Float16/32/64, and rationals over standard integer types. Bool, BigFloat
and custom Real values are refused. There must be one time per signal sample and
at least two samples. Times must be strictly increasing. Exact subtraction
preserves large integer epochs and avoids native integer subtraction overflow.
The inferred FFT interval is converted to Float64 only after validation; an
unrepresentable interval, frequency spacing or PSD normalization is refused.

`dt` and `sample_times` are mutually exclusive. `power_spectrum(signal)` keeps
its legacy interval of 1.0. `result_spectrum` requires either `dt` or explicit
times. Default returns remain `(; frequencies, psd)`, including when times are
provided. `return_timing=true` adds a detached scalar timing report. Supported
timing-number types are also required for `dt` when requesting that report;
legacy `dt` arithmetic otherwise stays unchanged.

## Set an explicit uniformity tolerance

The exact reference period is

```math
\Delta t = \frac{t_N-t_1}{N-1}.
```

For each sample, both its preceding interval's deviation from this period and its
deviation from the regular grid `t₁ + (k-1)Δt` must be at most
`timing_atol + timing_rtol*Δt`. `timing_atol` has the chosen time-coordinate unit;
`timing_rtol` is dimensionless. Both default to zero. Tolerances must be finite,
nonnegative supported timing numbers. The bound never depends on epoch magnitude
or total recording duration. Checking the global grid prevents many individually
small interval deviations from accumulating into a large phase drift.

Ordinary Float64 decimal ranges can fail the exact default because their stored
differences are not mathematically equal. Supply a justified explicit tolerance:

```@example spectrum_timing
rounded = [0., 0.1, 0.2, 0.3]
approximate = power_spectrum([0., 1., 0., -1.]; sample_times=rounded,
    timing_atol=1e-15, time_unit="s", return_timing=true)
@assert approximate.timing["uniformity"] == "within_explicit_tolerance"
approximate.timing["max_interval_residual"]
```

Accepted tolerance means the supplied samples are treated as an approximate
regular FFT grid with the endpoint-average period. No resampling is performed.
It does not establish that the resulting spectral bias is negligible. Genuine
irregular sampling, missing time positions or accumulated drift outside the
bound are rejected. This API does not calculate irregular-sampling spectra.

The report records provided/default interval provenance or explicit sample-time
provenance, sample count, exact endpoints/reference period, the applied interval,
and provided/unknown time-unit labels. For explicit times it also records exact
tolerances and maximum interval/global-grid residuals. It retains no complete
timeline. An interval-only call assumes uniformity; it has no sample timestamps
to verify. Unit labels do not convert time coordinates or authenticate a clock.

## Combine compatible result fields

Use `result_spectrum(results, row, column; sample_times, ...)` for one stored
planar component. Coordinates, component/flag dimensions, attached scale factors
and unit labels must agree across all results. An absent scale differs from an
attached one. No result is automatically converted or interpolated spatially.

```julia
spectrum = result_spectrum(results, 12, 8; sample_times=field_times,
    component=:u, time_unit="s", return_timing=true)
```

Sampling times describe successive **field samples**, not the two-image delay
within a PIV pair. `PhysicalScale.dt` only describes displacement-to-velocity
conversion. It is never used to infer the spectrum cadence or its time unit.
For example, acquisition frames every 0.01 s grouped as `(1,2), (3,4), …` can
produce field samples every 0.02 s, even though each pair delay is 0.01 s.
Choose a defensible field-time attribution yourself; the spectrum API does not
assume that a pair midpoint is an exposure center.

If pair delays differ, explicitly convert compatible raw results to velocities
first, then analyze their stored values:

```julia
velocity_results = physical.(results)
spectrum = result_spectrum(velocity_results, 12, 8; sample_times=field_times,
    time_unit="s", return_timing=true)
```

Grid coordinates and resulting scale factors/labels must still agree after
conversion. Equality of scale metadata alone cannot prove the arrays are
physical velocities; results retain the established stored-value contract.
The chosen sampling-axis unit is independent of a component's velocity-unit
denominator. For example, an axis in milliseconds may describe a component
stored in mm/s: frequencies are cycles/ms and the PSD is in (mm/s)² per cycles/ms.
No label-based conversion occurs. Unknown sampling units remain unknown.

Masked, flagged or nonfinite values of the **selected component** are rejected by
default. `invalid=:mean` fills them with the valid mean. `invalid=:interpolate`
linearly fills interior gaps on the accepted regular FFT grid and holds the
nearest valid endpoint value. This uses ordinal grid positions even when an
explicit tolerance accepts slightly nonuniform provided times; it is not
actual-time interpolation or resampling. Original sample count and time positions
are preserved. At least one valid component sample is required; a bad unselected
component alone does not invalidate the selected one.

Only `return_timing=true` adds the original invalid count, fill policy,
component/index and copied attached-scale metadata. Filling does not recover
measured values, and spectra describe the resulting stored/filled signal without
an uncertainty estimate. The mean-removal, Hann/untapered windows and one-sided
PSD normalization are unchanged. See the
[statistics reference](../reference/ensemble.md) and
[derived analysis reference](../reference/derived.md).
