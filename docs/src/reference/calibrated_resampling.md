```@meta
CurrentModule = Hammerhead
```

# Calibrated planar resampling

[`resample_planar`](@ref) samples a raw planar PIV field and transforms its
displacements into an explicit planar basis. [`resample_image`](@ref) samples
a scalar image in that same coordinate frame. Both use inverse affine mapping
and bilinear interpolation onto caller-supplied x/y axes; neither infers a
target grid, acquisition timing, calibration authenticity or units.

See the executable [PIV/PLIF workflow](../howto/calibrated_resampling.md) for two
manual dot-grid fits, unequal image resolutions, a PIV ROI, explicit scalar
crop coordinates, masks and joint availability. Input coordinates use x along
columns and y along rows; result arrays are indexed `[y, x]`.

Outputs carry separate arrays and support/availability diagnostics. Only
strictly positive interpolation weights require valid source contributors.
An exact-node query therefore need not depend on a masked zero-weight neighbor.
Unavailable output is NaN, not a filled measurement. No PIVResult or uncertainty
estimate is synthesized. Scalar units remain caller-supplied labels; an explicit
pair delay converts calibrated displacement to velocity without discovering
timestamps or applying attached `PhysicalScale` metadata.

Geometry is explicitly Float64: one applied calibration governs inverse queries
and vector components, and source/target axes must remain finite and strictly
monotone after that conversion. Returned target axes describe these applied
queries, not an unrounded higher-precision request. Values promote the numeric
input types, while target coordinates must represent the applied geometry
exactly in the promoted output type. A singleton supports only its exact
represented coordinate; there is no coverage tolerance or extrapolation.
Flags can overlap, and necessary positive-weight arithmetic underflow is an
arithmetic failure rather than permission to omit an invalid contributor.

Manual registration validates two finite representable coordinates per pair,
numerical affine rank of both point sets and finite invertible fitted
coefficients. The existing Float64 least-squares fit convention is preserved;
validation supplies no residual acceptance criterion or dynamic distortion
correction. The frame and unit labels are interpretation metadata, not proof
of a common physical calibration.

```@index
Pages = ["calibrated_resampling.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["src/calibrated_resampling.jl"]
Order = [:function]
```
