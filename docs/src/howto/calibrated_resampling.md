# Register PIV and PLIF onto a shared planar grid

Fit each camera's calibration features to the same known planar coordinates,
then sample the PIV field and scalar image at an explicit common grid. The
position transform includes the offset; the displacement transform uses only
the linear matrix. Specify the PIV exposure-pair delay separately to obtain
velocity. A camera's pixel spacing and the requested output-grid spacing are
different quantities.

Use calibration images from the measurement plane with unchanged camera
geometry. Correspondences are ordered `(x, y)` pairs: x increases along image
columns and y along rows. [`calculate_manual_registration`](@ref) requires at
least three non-collinear finite pairs and returns a Float64 affine fit. Inspect
calibration residuals and feature coverage yourself; this validation does not
impose a scientific fit-quality threshold. Convert its matrix and offset into
a [`PlanarTransform`](@ref) for the resampling APIs.

The example uses known synthetic dot centers for two unequal-resolution
cameras. Both calibrations map to millimeters in the same plane; its positive
y axis deliberately follows image rows. An experimental fit may instead
include a reflection or rotation to define another physical basis.

```@example calibrated_resampling
using Hammerhead, Random, LinearAlgebra

world_dots = [(0., 0.), (12., 0.), (0., 12.), (12., 12.)]
piv_dots = [(1., 1.), (121., 1.), (1., 121.), (121., 121.)]
plif_dots = [(1., 1.), (241., 1.), (1., 181.), (241., 181.)]

piv_fit = calculate_manual_registration(piv_dots, world_dots)
plif_fit = calculate_manual_registration(plif_dots, world_dots)
piv_transform = PlanarTransform(piv_fit.A, piv_fit.b)
plif_transform = PlanarTransform(plif_fit.A, plif_fit.b)

dt = 0.2 # seconds between the two PIV images, not between successive fields
motion = (x, y, z, t) -> (10., 5., 0.) # pixel velocity for this synthetic renderer
image_a, image_b, _, _ = SyntheticData.generate_synthetic_piv_pair(
    motion, (128, 128), dt; particle_density=0.08,
    background_noise=0., rng=MersenneTwister(83))
piv_mask = falses(128, 128)
piv_mask[48:64, 48:64] .= true # true means excluded
pass = PIVParameters(window_size=(24, 24), search_area_size=(32, 32),
    overlap=(12, 12), uod_enable=false, replace_outliers=false)
raw = run_piv(image_a, image_b, pass; roi=ROI(17:112, 17:112),
    mask=piv_mask, threaded=false)
@assert raw.scale === nothing
# ROI output coordinates still refer to the original full image.
@assert first(raw.x) > 17 && first(raw.y) > 17

concentration(X, Y) = exp(-((X - 6)^2 + (Y - 6)^2) / 8)
plif = [concentration(plif_transform((Float64(c), Float64(r)))...)
        for r in 1:192, c in 1:256]
plif_mask = falses(size(plif))
plif_mask[100:115, 170:180] .= true

# Cropping a scalar image must retain its original source coordinates.
rows, cols = 13:180, 17:240
plif_crop = view(plif, rows, cols)
mask_crop = view(plif_mask, rows, cols)
common_x, common_y = collect(1.:0.25:11.), collect(1.:0.25:11.)
piv_common = resample_planar(raw, common_x, common_y;
    transform=piv_transform, length_unit="mm", dt, time_unit="s",
    coordinate_frame="synthetic_target")
plif_common = resample_image(plif_crop, common_x, common_y;
    transform=plif_transform, length_unit="mm", value_unit="relative_fluorescence",
    coordinate_frame="synthetic_target", mask=mask_crop,
    source_x=cols, source_y=rows)

joint_available = piv_common.available .& plif_common.available
@assert any(joint_available) && !all(joint_available)
@assert count(joint_available) < count(piv_common.available)
@assert size(piv_common.u) == size(plif_common.values) == size(joint_available)
(target_nodes=length(joint_available), joint_nodes=count(joint_available),
 piv_nodes=count(piv_common.available), scalar_nodes=count(plif_common.available))
```

The common vector components are expressed in the fitted planar basis and
divided by the supplied delay. This synthetic motion is nominally `(1, 0.5)`
mm/s under the PIV camera's 0.1 mm/pixel map; the resampled vectors carry
the measurement error of the PIV result they came from. The scalar values remain relative fluorescence, not an inferred
concentration calibration. Position units, scalar units and frame labels are
caller-supplied interpretation metadata, not automatic unit conversions.

Each output node is inverse-mapped to the source and bilinearly sampled.
Every corner with strictly positive weight must be usable. An invalid corner
with zero weight at an exact source node or edge does not invalidate that
sample. Out-of-domain samples, masked/nonfinite contributors and arithmetic
failures are unavailable; do not treat their NaN values as zero signal. Planar
sampling excludes flagged vectors by default. `include_invalid=true` explicitly
permits finite flagged vectors but does not unmask excluded nodes.

The returned arrays and availability/support diagnostics are separate from
the original result. They are not a new `PIVResult`, do not recreate correlation
measurements and do not propagate PIV uncertainty. Use `joint_available` when
combining the two modalities; the overlap in output coordinates alone does
not establish joint measurement availability or synchronized acquisition.

Pass raw, unscaled planar results. Attaching a `PhysicalScale` and also applying
the calibrated affine map would make the conversion ambiguous, so that input
is refused. Stripping the metadata with `with_scale(result, nothing)` does not
undo a previous `physical(result)` conversion: the arrays can still contain
physical positions and velocities. Use the original unconverted pixel result;
absence of scale metadata alone does not prove that basis. Omitting `dt` leaves
calibrated displacement; supplying it requires
an explicit time-unit label. A common output grid can have ascending or
descending axes, but its labels and orientation must agree between modalities.

A static affine fit does not correct transient turbulence-induced refractive
distortion, lens distortion or out-of-plane structure. This bilinear operation
is neither anti-alias filtering nor conservative area averaging. Before
downsampling a PLIF image, choose and document any needed filtering separately.
Equal grid nodes also do not turn PIV interrogation-window estimates and PLIF
pixels into measurements with equal spatial or temporal support.

Sampling geometry uses one applied Float64 copy of the calibration and axes.
Returned target coordinates describe those represented queries, including when
the requested axes have higher precision. Axes must remain finite and strictly
monotone after conversion; no tolerance expands coverage at boundaries. A
singleton source axis supports only its exact represented coordinate. Output
values promote the source, calibration, target-axis and explicit-delay types;
all-Float32 inputs can retain Float32 values. The returned target axes must
represent the applied geometry exactly in that output precision.
