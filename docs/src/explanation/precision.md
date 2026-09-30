# Numeric precision policy

Hammerhead follows one rule: **precision follows the images**. The element
type of the analysis is

```julia
T = float(promote_type(eltype(imgA), eltype(imgB)))
```

and determines the precision of the correlators, image deformation, and
numeric arrays in the returned [`PIVResult`](@ref). The default
[`load_image`](@ref) returns `Float64`; `load_image(Float32, path)` or
`image_type = Float32` in a batch driver keeps the main image-processing
buffers in single precision.

## Choosing Float32 or Float64

Fast Fourier transforms (FFTs) over interrogation windows and B-spline
resampling over whole images account for substantial work. Float32 uses
half the storage per image element of Float64. If memory use matters, run a
representative pair at both precisions and compare vectors and diagnostics
at the tolerance your application requires. Choose the precision by loading
or converting the input images to the desired element type.

On graphics processing unit (GPU) backends, image deformation, FFTs, and
correlation planes still follow
the image type. Float32 therefore reduces the dominant device buffers. The
uncertainty exception below remains Float64 on the GPU as well; see
[Run PIV on a GPU](../howto/gpu.md) for the resulting memory and performance
tradeoffs.

## Computations that use Float64

Some computations use Float64 regardless of image type. Numeric fields in a
`PIVResult` are converted back to the result's element type where applicable;
camera models and precomputed maps retain their own precision:

- **Camera calibration and stereo geometry** — calibration fits
  ([`calibrate_camera`](@ref), [`detect_calibration_grid`](@ref)), precomputed
  dewarp coordinate maps, stereo reconstruction, and self-calibration.
- **Uncertainty statistics** — the Wieneke (2015) sums accumulate in
  Float64 per window to reduce rounding error during pooling and
  cancellation.
- **Robust statistics** — replacement medians, temporal validation buffers,
  and field statistics accumulate in Float64.
- **The iterative `:gauss2d` subpixel fit** (an LsqFit solve).

A `Float32` analysis returns `Float32` displacement and diagnostic arrays.
Precision choice and correlation settings address different sources of error;
see [Correlation accuracy](correlation.md) for the latter.
