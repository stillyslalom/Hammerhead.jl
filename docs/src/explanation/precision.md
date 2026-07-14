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

## Why not just always Float64?

Particle image velocimetry (PIV) is bandwidth-bound: the dominant costs are
fast Fourier transforms (FFTs) over interrogation
windows and B-spline resampling over whole images. Single precision is
plenty for image data quantized to 8–16 bits, and choosing precision at the
input keeps the choice in the caller's hands with zero configuration
surface — there is no `precision` parameter to document or misuse.

On graphics processing unit (GPU) backends, image deformation, FFTs, and
correlation planes still follow
the image type. Float32 therefore reduces the dominant device buffers. The
uncertainty exception below remains Float64 on the GPU as well; see
[Run PIV on a GPU](../howto/gpu.md) for the resulting memory and performance
tradeoffs.

## Deliberate Float64 islands

A few computations always run in double precision regardless of the image
type, because they are far from the per-pixel hot path and benefit from the
extra headroom. They convert back to `T` when stored:

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

The practical consequence: a `Float32` pipeline produces `Float32` results
whose *accuracy* is limited by the PIV method (≳ 0.03 px), not by the
arithmetic — the islands make sure of it.
