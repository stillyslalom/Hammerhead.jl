module Hammerhead

using FFTW
using LinearAlgebra
using Interpolations
using LsqFit: curve_fit
using Statistics: median, median!, std
using StaticArrays
using CoordinateTransformations: AffineMap
import FileIO
using ImageCore: Colorant, Gray
using ImageFiltering: imfilter, KernelFactors
using JLD2: jldopen
using ProgressMeter: Progress, next!

export PIVParameters, PIVResult, run_piv, multipass_parameters
export load_image, image_pairs, save_results, load_results, run_piv_sequence
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
export DewarpGrid, ImageDewarper, dewarp, dewarp!
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
include("correlators.jl")
include("uncertainty.jl")
include("transforms.jl")
include("calibration.jl")
include("target_detection.jl")
include("dewarp.jl")
include("quality.jl")
include("masking.jl")
include("pipeline.jl")
include("particles.jl")
include("ptv.jl")
include("tracking.jl")
include("stereo.jl")
include("io.jl")
include("ensemble.jl")
include("selfcal.jl")
include("statistics.jl")

"""
    plot_vector_field(result::PIVResult; highlight_outliers=true, figure=(;), axis=(;), kwargs...)
    plot_vector_field(result::PTVResult; highlight_outliers=true, figure=(;), axis=(;), kwargs...)
    plot_vector_field(x, y, u, v; figure=(;), axis=(;), kwargs...)

Plot a PIV or PTV vector field as arrows and return the `Makie.Figure`. The
y-axis is reversed to match image (row-down) coordinates. For a
[`PIVResult`](@ref) the arrows sit on the interrogation grid; for a
[`PTVResult`](@ref) they are scattered at the frame-A particle positions.
Outliers are drawn in red when `highlight_outliers` is set; masked (`NaN`)
vectors are skipped; remaining `kwargs` are passed to `arrows2d!`.

Provided by a package extension: load a Makie backend first (e.g.
`using GLMakie` or `using CairoMakie`).
"""
function plot_vector_field end

"""
    plot_vector_field!(ax, result::PIVResult; highlight_outliers=true, kwargs...)
    plot_vector_field!(ax, result::PTVResult; highlight_outliers=true, kwargs...)
    plot_vector_field!(ax, x, y, u, v; kwargs...)

Like [`plot_vector_field`](@ref), but draws into an existing `Makie.Axis`.
"""
function plot_vector_field! end

const _MAKIE_HINT = "plot_vector_field requires Makie: load a backend first, e.g. `using GLMakie` or `using CairoMakie`."
plot_vector_field(args...; kwargs...) = error(_MAKIE_HINT)
plot_vector_field!(args...; kwargs...) = error(_MAKIE_HINT)

end # module Hammerhead
