module Hammerhead

using FFTW
using LinearAlgebra
using Interpolations
using LsqFit: curve_fit
using Statistics: median

export PIVParameters, PIVResult, run_piv
export Correlator, CrossCorrelator, PhaseCorrelator, correlate, correlate_deformable
export AffineTransform, warp_image, calculate_manual_registration, transform_vector_field
export calculate_peak_ratio, calculate_correlation_moment, universal_outlier_detection
export plot_vector_field, plot_vector_field!

include("types.jl")
include("correlators.jl")
include("transforms.jl")
include("quality.jl")
include("pipeline.jl")

"""
    plot_vector_field(result::PIVResult; highlight_outliers=true, figure=(;), axis=(;), kwargs...)
    plot_vector_field(x, y, u, v; figure=(;), axis=(;), kwargs...)

Plot a PIV vector field as arrows and return the `Makie.Figure`. The y-axis is
reversed to match image (row-down) coordinates. Outliers are drawn in red when
`highlight_outliers` is set; remaining `kwargs` are passed to `arrows2d!`.

Provided by a package extension: load a Makie backend first (e.g.
`using GLMakie` or `using CairoMakie`).
"""
function plot_vector_field end

"""
    plot_vector_field!(ax, result::PIVResult; highlight_outliers=true, kwargs...)
    plot_vector_field!(ax, x, y, u, v; kwargs...)

Like [`plot_vector_field`](@ref), but draws into an existing `Makie.Axis`.
"""
function plot_vector_field! end

const _MAKIE_HINT = "plot_vector_field requires Makie: load a backend first, e.g. `using GLMakie` or `using CairoMakie`."
plot_vector_field(args...; kwargs...) = error(_MAKIE_HINT)
plot_vector_field!(args...; kwargs...) = error(_MAKIE_HINT)

end # module Hammerhead
