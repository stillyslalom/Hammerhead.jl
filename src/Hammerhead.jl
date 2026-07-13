module Hammerhead

using FFTW
using KernelAbstractions
using LinearAlgebra
using Interpolations
using LsqFit: curve_fit
using Statistics: median, median!, std, quantile, quantile!
using StaticArrays
using CoordinateTransformations: AffineMap
import FileIO
using ImageCore: Colorant, Gray
using ImageFiltering: imfilter, KernelFactors
using JLD2: jldopen
using ProgressMeter: Progress, next!

export PIVParameters, PIVResult, run_piv, multipass_parameters, PIVWorkspace, piv_workspace
export PhysicalScale, physical, with_scale
export load_image, image_pairs, save_results, load_results, run_piv_sequence, frame_index_strings
export polygon_mask, load_mask
export run_piv_ensemble, field_statistics, validate_temporal!, power_spectrum
export find_peaks, peak_locking, smoothn, error_statistics
export SyntheticData
export Correlator, CrossCorrelator, PhaseCorrelator, correlate, correlate_deformable
export AffineTransform, warp_image, calculate_manual_registration, transform_vector_field
export CameraCalibration, PinholeCamera, SoloffCamera, TransformedCamera, calibrate_camera
export world_to_pixel, pixel_to_world, reprojection_errors, calibration_quality
export self_calibrate, SelfCalibrationReport
export CalibrationGrid, detect_calibration_grid, calibration_points, render_calibration_target
export DewarpGrid, ImageDewarper, dewarp, dewarp!, common_dewarp_grid
export StereoPIVResult, run_piv_stereo
export Particles, detect_particles, PTVParameters, PTVResult, run_ptv, run_ptv_sequence, ptv_to_grid
export Trajectory, TrackingResult, track_particles, trajectory_velocities
export calculate_peak_ratio, calculate_correlation_moment, universal_outlier_detection
export PIVValidator, LocalValidator, NeighborhoodValidator
export PeakRatioValidator, CorrelationMomentValidator, VelocityMagnitudeValidator, UniversalOutlierValidator
export validate_vectors!, apply_validator!
export plot_vector_field, plot_vector_field!
export compute_background, subtract_background, intensity_cap, highpass_filter, clahe
export subtract_background!, intensity_cap!, highpass_filter!, clahe!

include("types.jl")
include("synthetic_data.jl")
include("preprocessing.jl")
include("localmax.jl")
include("correlators.jl")
include("uncertainty.jl")
include("transforms.jl")
include("calibration.jl")
include("target_detection.jl")
include("dewarp.jl")
include("quality.jl")
include("masking.jl")
include("pipeline.jl")
include("ka_backend.jl")
include("particles.jl")
include("ptv.jl")
include("tracking.jl")
include("stereo.jl")
include("scaling.jl")
include("io.jl")
include("ensemble.jl")
include("selfcal.jl")
include("statistics.jl")

# Auto arrow-length scale for plot_vector_field: the multiplier that maps the
# 0.99-quantile magnitude among the selected vectors to `target_length` (the
# plotted grid spacing), so the busiest arrows span roughly one grid cell.
# `valid` selects the vectors entering the quantile (e.g. non-masked,
# non-outlier); `nothing` selects all. Falls back to all finite vectors when
# the selection is empty, and returns `nothing` (meaning "leave lengths as
# measured") when there is nothing with positive magnitude to scale.
function arrow_lengthscale(u, v, valid, target_length::Real)
    mags = Float64[]
    for i in eachindex(u, v)
        (valid === nothing || valid[i]) || continue
        m = hypot(float(u[i]), float(v[i]))
        isfinite(m) && push!(mags, m)
    end
    if isempty(mags)
        for i in eachindex(u, v)
            m = hypot(float(u[i]), float(v[i]))
            isfinite(m) && push!(mags, m)
        end
    end
    isempty(mags) && return nothing
    q = quantile(mags, 0.99)
    (isfinite(q) && q > 0) || return nothing
    return target_length / q
end

# Spacing of a regular grid axis (used to size auto-scaled arrows).
grid_axis_step(x) = length(x) > 1 ? abs(float(x[2]) - float(x[1])) : 1.0

"""
    plot_vector_field(result::PIVResult; stride=1, lengthscale=:auto, show_replaced=true, replaced_color=:orangered, figure=(;), axis=(;), kwargs...)
    plot_vector_field(result::PTVResult; highlight_outliers=true, figure=(;), axis=(;), kwargs...)
    plot_vector_field(x, y, u, v; stride=1, lengthscale=:auto, figure=(;), axis=(;), kwargs...)

Plot a PIV or PTV vector field as arrows and return the `Makie.Figure`. The
y-axis is reversed to match image (row-down) coordinates. For a
[`PIVResult`](@ref) the arrows sit on the interrogation grid; for a
[`PTVResult`](@ref) they are scattered at the frame-A particle positions.

For grid fields, `stride` plots every `stride`-th vector in each direction
(inspect a high-resolution field without a solid mat of arrows), and
`lengthscale = :auto` scales the arrows so the 0.99-quantile of the valid
vectors' magnitudes spans one plotted grid cell (`stride ×` the grid spacing);
pass a `Real` for a manual multiplier. Flagged/replaced vectors
(`result.outliers`) are drawn in `replaced_color` when `show_replaced` is
true and omitted otherwise; masked (`NaN`) vectors are always skipped;
remaining `kwargs` are passed to `arrows2d!`.

A result with an attached [`PhysicalScale`](@ref) is plotted in physical
units (positions and velocities converted via [`physical`](@ref), axis
labels showing the length unit); to overlay arrows on the source image in
pixel coordinates instead, plot `with_scale(result, nothing)`. The bare
`x, y, u, v` method always plots the arrays as given with pixel labels.

Provided by a package extension: load a Makie backend first (e.g.
`using GLMakie` or `using CairoMakie`).
"""
function plot_vector_field end

"""
    plot_vector_field!(ax, result::PIVResult; stride=1, lengthscale=:auto, show_replaced=true, replaced_color=:orangered, kwargs...)
    plot_vector_field!(ax, result::PTVResult; highlight_outliers=true, kwargs...)
    plot_vector_field!(ax, x, y, u, v; stride=1, lengthscale=:auto, kwargs...)

Like [`plot_vector_field`](@ref), but draws into an existing `Makie.Axis`.
Scaled results are converted the same way, but the labels of an existing
axis are left alone.
"""
function plot_vector_field! end

const _MAKIE_HINT = "plot_vector_field requires Makie: load a backend first, e.g. `using GLMakie` or `using CairoMakie`."
plot_vector_field(args...; kwargs...) = error(_MAKIE_HINT)
plot_vector_field!(args...; kwargs...) = error(_MAKIE_HINT)

end # module Hammerhead
