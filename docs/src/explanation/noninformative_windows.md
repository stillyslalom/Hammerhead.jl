# Windows without displacement information

A correlation plane needs a finite, positive peak and variation across lags to
provide a displacement measurement. Hammerhead rejects a completely flat plane,
a plane with no positive value, or a plane containing any nonfinite value. An
exactly empty or constant window presented to the correlator therefore cannot
produce a valid displacement merely because a peak finder chooses a deterministic
corner of the plane. For deformed windows, Hammerhead additionally checks for
contrast in the original images under the stencil convention below.

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

Nonzero deformation can introduce floating-point texture into an originally
constant patch. Before correlating a deformed window, the automatic original
stencil guard evaluates its predictor at every unmasked destination pixel. Frame
A samples at `(row - v/2, col - u/2)` and frame B at `(row + v/2, col + u/2)`.
For each in-image query, the guard examines the clipped 4 × 4 neighborhood of
original pixels around that cubic cell, using the images and mask supplied to
PIV after preprocessing and ROI cropping. It combines the eligible original values
from these neighborhoods; any exact difference establishes contrast. Differing
constant neighborhood levels count as contrast too. CPU and portable predictor
evaluation orders are both checked, so a rounding difference at a cell boundary
cannot remove real contrast present in either sampled stencil.

The comparison uses the processing precision: one representable intensity step
is enough, and no intensity threshold is applied. Originally masked pixels and
masked destination pixels do not supply evidence. Queries outside the image
supply no original evidence, so virtual extrapolated zeros cannot make a constant
source informative. With genuine original contrast and some outside-image
samples, correlation retains its existing zero-extrapolation behavior. The
user's mask remains static and its node geometry is unchanged.

A window is skipped if either frame supplies no original contrast. Its
displacement, ratio, moment, alternatives, and uncertainty are unavailable.
Vector replacement may fill its displacement but retains its outlier flag.
Retained correlation planes contain a logical zero contribution for skipped
windows; that zero is not a computed correlation diagnostic. Raw warped image
values are unchanged. The same guard runs against the original images on every
iteration, with the current predictor, and on every deformed ensemble pair.

This is a bounded scientific convention, not the mathematical support of the
cardinal B-spline interpolant. Its coefficient prefilter has nonlocal influence.
When every eligible original stencil is constant, Hammerhead treats distant
coefficient tails, ringing, and resampling roundoff as insufficient evidence of
local displacement. Contrast inside any sampled stencil survives regardless of
amplitude; an informative window's correlation calculation is unchanged. Near a
feature boundary, distant interpolant influence can be material, so this guard
should not be read as an accuracy guarantee. Known unseeded regions can still be
masked explicitly. A nonfinite original pair bypasses the stencil proof and
retains the existing nonfinite correlation-plane rejection instead of being
silently discarded from an ensemble.

Metadata is built lazily only when a nonzero predictor deforms the images. The
compact map stores one byte per cubic cell per frame, referencing original
intensities without duplicating them; a frame whose every stencil varies needs
no map. The metadata stays on the CPU for all backends. Device engines receive
only per-window skip markers through their existing origin upload.

An empty image pair, including a deformed window skipped by the original stencil
guard, contributes zero correlation and zero uncertainty statistics to an
ensemble. It cannot produce a measurement by itself, but does not erase
information contributed by other pairs. A completely uninformative ensemble
remains flagged. This does not establish per-pair validity counts or normalize an
ensemble by informative-pair yield.

The generic [`find_peaks`](@ref) function retains its plateau and nonpositive
candidate behavior; finding a candidate does not establish a measurement. The
public [`correlate`](@ref) convenience function returns `NaN` displacement, peak,
and refined location for an uninformative plane, with integer `peakloc = (0, 0)`
as an absent-location sentinel. The returned correlation plane still aliases the
correlator's internal buffer.
