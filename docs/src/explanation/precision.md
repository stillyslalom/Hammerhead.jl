# Numeric precision policy

Hammerhead follows one rule: **precision follows the images**. The element
type of the analysis is

```julia
T = float(promote_type(eltype(imgA), eltype(imgB)))
```

and it flows through the correlators, image deformation, and every field of
the returned [`PIVResult`](@ref)`{T}`. Feed `Float64` matrices (the
[`load_image`](@ref) default) and everything runs in double precision; feed
`Float32` matrices (`load_image(Float32, path)`, or
`image_type = Float32` in the batch drivers) and the whole hot path — FFTs,
window loads, resampling — runs in single precision, halving memory traffic
on large recordings.

## Choosing Float32 or Float64

Particle image velocimetry (PIV) is bandwidth-bound: the dominant costs are
fast Fourier transforms (FFTs) over interrogation
windows and B-spline resampling over whole images. Float32 is generally
sufficient for image data quantized to 8–16 bits and reduces memory use.
Choose precision by loading or converting the input images to the desired
element type.

On graphics processing unit (GPU) backends, image deformation, FFTs, and
correlation planes still follow
the image type. Float32 therefore reduces the dominant device buffers. The
uncertainty exception below remains Float64 on the GPU as well; see
[Run PIV on a GPU](../howto/gpu.md) for the resulting memory and performance
tradeoffs.

## Computations that use Float64

Some computations use Float64 regardless of image type. Their stored results
are converted back to `T`:

- **Camera calibration and stereo geometry** — offline, once-per-experiment
  fits ([`calibrate_camera`](@ref), [`detect_calibration_grid`](@ref)), the
  precomputed dewarp coordinate maps, per-vector stereo reconstruction, and
  self-calibration. These are O(points) or O(vector grid), not O(pixels).
- **Uncertainty statistics** — the Wieneke (2015) sums accumulate in
  Float64 per window so that pooling across thousands of ensemble pairs
  cannot lose precision to cancellation.
- **Robust statistics** — replacement medians, temporal validation buffers,
  and field statistics accumulate in Float64.
- **The iterative `:gauss2d` subpixel fit** (an LsqFit solve).

A `Float32` analysis returns `Float32` fields. For the accuracy configuration
described in [Correlation accuracy](correlation.md), measured error on
synthetic data is about 0.03 px root-mean-square. The Float64 computations
above preserve precision where sums or fits need it.
