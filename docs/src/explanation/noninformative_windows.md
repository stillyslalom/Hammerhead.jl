# Windows without displacement information

A correlation plane needs a finite, positive peak and variation across lags to
provide a displacement measurement. Hammerhead rejects a completely flat plane,
a plane with no positive value, or a plane containing any nonfinite value. An
exactly empty or constant window presented to the correlator therefore cannot
produce a valid displacement merely because a peak finder chooses a deterministic
corner of the plane. This guarantee concerns correlation inputs, including
deformed inputs; it does not establish that a deformed window contains genuine
particle texture in the original images.

The check applies to single-pair and ensemble processing on CPU and the shared
KernelAbstractions kernels. It uses no absolute intensity or correlation-amplitude
cutoff: a finite, distinct positive peak can remain informative at very small
amplitude. This is a check for missing information, not a calibrated accuracy or
signal-quality guarantee. Peak-ratio filtering, outlier detection, and optional
validators still judge informative measurements.

Mean subtraction happens before apodization. When all valid pixels in a window
have exactly the same value in the processing precision, Hammerhead subtracts
that value exactly. Averaging repeated floating-point values can otherwise leave
a rounding residual; multiplying that residual by a Gaussian window creates an
artificial pattern. Masked pixels do not participate in the constant test or
mean. Frame A and an enlarged frame-B search area are centered independently.

Unmasked nodes without a measurement have `NaN` displacement and diagnostic
values and are flagged in `result.outliers`, even when optional validation is
disabled. Their uncertainties remain `NaN`. If vector replacement can fill a
node from enough valid neighbors, its displacement becomes finite but its
outlier flag remains set. Such a value is an estimate from neighboring vectors;
default field statistics exclude it. Masked nodes remain a separate category:
they hold `NaN`, are marked in `result.mask`, and are not counted as outliers.

Between passes, the predictor uses finite replacements where available. Missing
predictor values that cannot be filled from neighbors receive zero displacement
only in the predictor, without changing measured results or flags. An entirely
zero predictor skips image deformation, avoiding needless resampling roundoff.
Outlier detection excludes nonfinite vectors from neighboring medians. These
steps let repeated passes proceed through missing-data regions without treating
missing displacements as measurements.

There is a remaining precision limit for a constant patch beside moving texture.
A nonzero predictor can resample that patch with small floating-point intensity
variations. The resulting correlation input is no longer exactly constant, and
these checks cannot distinguish its numerical texture from real low-contrast
texture. A targeted two-pass test scene (constant `0.1` on the left half of a
64 × 64 image, moving particles on the right) demonstrates this away from both
the image and texture boundaries. Local Float64 CPU and KA runs accepted blank
interior nodes after deformation. Mask known unseeded regions when possible;
outlier detection is not guaranteed to reject these artifacts. Carrying original
source-support information through deformation and ensemble accumulation remains
open work. An arbitrary intensity threshold would also discard real weak signals
and is deliberately not used as a substitute.

An empty image pair contributes zero correlation to an ensemble. It cannot
produce a measurement by itself, but does not erase information contributed by
other pairs. A completely uninformative ensemble remains flagged. This does not
establish per-pair validity counts or normalize an ensemble by informative-pair
yield.

The generic [`find_peaks`](@ref) function retains its plateau and nonpositive
candidate behavior; finding a candidate does not establish a measurement. The
public [`correlate`](@ref) convenience function returns `NaN` displacement, peak,
and refined location for an uninformative plane, with integer `peakloc = (0, 0)`
as an absent-location sentinel. The returned correlation plane still aliases the
correlator's internal buffer.
