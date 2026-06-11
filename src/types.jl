"""
    PIVParameters(; kwargs...)

Immutable, validated configuration for a PIV analysis.

# Keyword arguments
- `window_size = (32, 32)`: interrogation window size `(rows, cols)`; an `Int` is
  expanded to a square window.
- `overlap = (16, 16)`: window overlap `(rows, cols)`; must satisfy
  `0 ≤ overlap < window_size`. An `Int` is expanded likewise.
- `correlation_method = :cross`: `:cross` (standard FFT cross-correlation) or
  `:phase` (phase correlation).
- `subpixel_method = :gauss3`: `:gauss3` (3-point Gaussian fit), `:gauss2d`
  (least-squares 2D Gaussian fit), or `:none`.
- `deformation_iterations = 0`: number of iterative window-deformation passes;
  `0` disables deformation.
- `uod_enable = true`: run universal outlier detection on the result.
- `uod_threshold = 2.0`: UOD sensitivity (higher is less sensitive).
- `uod_neighborhood = 1`: UOD neighborhood layers (1 → 3×3, 2 → 5×5, ...).
"""
struct PIVParameters
    window_size::Tuple{Int,Int}
    overlap::Tuple{Int,Int}
    correlation_method::Symbol
    subpixel_method::Symbol
    deformation_iterations::Int
    uod_enable::Bool
    uod_threshold::Float64
    uod_neighborhood::Int

    function PIVParameters(;
        window_size::Union{Int,Tuple{Int,Int}} = (32, 32),
        overlap::Union{Int,Tuple{Int,Int}} = (16, 16),
        correlation_method::Symbol = :cross,
        subpixel_method::Symbol = :gauss3,
        deformation_iterations::Int = 0,
        uod_enable::Bool = true,
        uod_threshold::Real = 2.0,
        uod_neighborhood::Int = 1,
    )
        ws = window_size isa Int ? (window_size, window_size) : window_size
        ov = overlap isa Int ? (overlap, overlap) : overlap
        all(>=(4), ws) ||
            throw(ArgumentError("window_size must be at least 4 in each dimension, got $ws"))
        all(0 .<= ov .< ws) ||
            throw(ArgumentError("overlap must satisfy 0 ≤ overlap < window_size, got overlap=$ov for window_size=$ws"))
        correlation_method in (:cross, :phase) ||
            throw(ArgumentError("correlation_method must be :cross or :phase, got :$correlation_method"))
        subpixel_method in (:gauss3, :gauss2d, :none) ||
            throw(ArgumentError("subpixel_method must be :gauss3, :gauss2d, or :none, got :$subpixel_method"))
        deformation_iterations >= 0 ||
            throw(ArgumentError("deformation_iterations must be non-negative, got $deformation_iterations"))
        uod_threshold > 0 ||
            throw(ArgumentError("uod_threshold must be positive, got $uod_threshold"))
        uod_neighborhood >= 1 ||
            throw(ArgumentError("uod_neighborhood must be at least 1, got $uod_neighborhood"))
        new(ws, ov, correlation_method, subpixel_method, deformation_iterations,
            uod_enable, Float64(uod_threshold), uod_neighborhood)
    end
end

function Base.show(io::IO, p::PIVParameters)
    print(io, "PIVParameters(window_size=$(p.window_size), overlap=$(p.overlap), ",
        "correlation_method=:$(p.correlation_method), subpixel_method=:$(p.subpixel_method), ",
        "deformation_iterations=$(p.deformation_iterations), uod=",
        p.uod_enable ? "(threshold=$(p.uod_threshold), neighborhood=$(p.uod_neighborhood))" : "off",
        ")")
end

"""
    PIVResult

Result of [`run_piv`](@ref).

# Fields
- `x`, `y`: window-center coordinates of the interrogation grid (`x` along
  columns, `y` along rows, in pixels).
- `u`, `v`: displacement components on the `(length(y), length(x))` grid; `u` is
  the column (x) displacement and `v` the row (y) displacement, in pixels. A
  particle at `(row, col)` in the first image is found at `(row + v, col + u)`
  in the second.
- `peak_ratio`: primary-to-secondary correlation peak ratio per window (higher
  is more reliable).
- `correlation_moment`: second moment of the correlation peak per window (an
  uncertainty proxy; lower is sharper).
- `outliers`: `BitMatrix` from universal outlier detection (`true` = outlier);
  all `false` when UOD is disabled.
- `parameters`: the `PIVParameters` used.
"""
struct PIVResult
    x::Vector{Float64}
    y::Vector{Float64}
    u::Matrix{Float64}
    v::Matrix{Float64}
    peak_ratio::Matrix{Float64}
    correlation_moment::Matrix{Float64}
    outliers::BitMatrix
    parameters::PIVParameters
end

function Base.show(io::IO, r::PIVResult)
    ny, nx = size(r.u)
    print(io, "PIVResult($(nx)×$(ny) grid, $(sum(r.outliers)) outliers)")
end
