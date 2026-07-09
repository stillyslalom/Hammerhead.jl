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
- `padding = false`: zero-pad the correlation FFT to twice the window size
  (true linear correlation; removes the wrap-around bias at ~4× FFT cost).
- `apodization = :none`: `:gauss` applies a Gaussian window to each
  interrogation window before correlating.
- `subpixel_method = :gauss3`: `:gauss3` (two independent 3-point Gaussian
  fits), `:gauss9` (closed-form 2D Gaussian regression on the 3×3
  neighborhood — exact for rotated elliptical peaks, less peak locking, only
  marginally slower), `:gauss2d` (iterative least-squares 2D Gaussian fit),
  or `:none`.
- `n_peaks = 3`: number of correlation peaks located per window (primary +
  alternatives, ≥ 1). When > 1, vectors that fail validation are re-tested
  against their secondary/tertiary peak displacements and locally consistent
  alternatives are accepted before local-median replacement kicks in
  ("peak substitution"; accepted cells are unflagged since they hold measured
  data). The top two peaks are always located — the peak ratio needs both —
  so values up to 2 are free; each additional peak costs about one extra
  scan of the correlation plane, and alternatives are always refined with
  the cheap 3-point fit.
- `uncertainty = false`: estimate a per-vector measurement uncertainty from
  correlation statistics (Wieneke 2015) into the `uncertainty_u` /
  `uncertainty_v` fields of the result. The estimator analyzes the residual
  asymmetry of the correlation peak between the two deformed windows and
  assumes the peak sits at ~zero residual displacement, so it runs on the
  final pass only and is meaningful only after multi-pass deformation has
  converged — use a schedule that repeats the final window size (e.g.
  `multipass_parameters([32, 16, 16])`). It estimates the random error only
  (systematic bias such as peak locking is invisible to it) and is accurate
  for uncertainties up to ~0.3 px.
- `uod_enable = true`: validate vectors with universal outlier detection.
- `uod_threshold = 2.0`: UOD sensitivity (higher is less sensitive).
- `uod_neighborhood = 2`: UOD neighborhood layers (1 → 3×3, 2 → 5×5, ...).
  The 5×5 default tolerates smooth velocity gradients at the field edges,
  where 3×3 neighborhoods falsely flag (and replacement then corrupts) the
  outermost rows of e.g. a sheared field.
- `min_peak_ratio = 1.0`: vectors whose correlation peak ratio falls below this
  are flagged invalid; values ≤ 1 disable the check (the peak ratio is ≥ 1 by
  construction).
- `validation = ()`: tuple of additional validators applied after the UOD and
  peak-ratio checks. Entries are validator objects or `Symbol => value` specs,
  e.g. `(:peak_ratio => 1.3, :velocity_magnitude => (max = 50,))` — see
  [`validate_vectors!`](@ref). Specs are parsed (and rejected if malformed)
  at construction.
- `replace_outliers = true`: replace flagged vectors with the local median of
  valid neighbors. Intermediate passes of a multi-pass run always replace,
  regardless of this setting, to keep the predictor field well behaved.
- `keep_correlation_planes = false`: retain each window's full correlation
  plane in the result's `correlation_planes` field for inspection. **Opt-in
  and memory-heavy** — a 32² window on a 100×100 grid in `Float64` is
  ~800 MB — so intended for small regions or coarse grids. As
  [`run_piv`](@ref) returns the final pass, set it on the final pass only
  (pair it with the `final` keyword of [`multipass_parameters`](@ref)). See
  [`PIVResult`](@ref).

Multi-pass interrogation is configured with a vector of `PIVParameters` (one
per pass) — see [`run_piv`](@ref) and [`multipass_parameters`](@ref).
"""
struct PIVParameters
    window_size::Tuple{Int,Int}
    overlap::Tuple{Int,Int}
    correlation_method::Symbol
    padding::Bool
    apodization::Symbol
    subpixel_method::Symbol
    n_peaks::Int
    uncertainty::Bool
    uod_enable::Bool
    uod_threshold::Float64
    uod_neighborhood::Int
    min_peak_ratio::Float64
    validation::Tuple
    replace_outliers::Bool
    keep_correlation_planes::Bool

    function PIVParameters(;
        window_size::Union{Int,Tuple{Int,Int}} = (32, 32),
        overlap::Union{Int,Tuple{Int,Int}} = (16, 16),
        correlation_method::Symbol = :cross,
        padding::Bool = false,
        apodization::Symbol = :none,
        subpixel_method::Symbol = :gauss3,
        n_peaks::Int = 3,
        uncertainty::Bool = false,
        uod_enable::Bool = true,
        uod_threshold::Real = 2.0,
        uod_neighborhood::Int = 2,
        min_peak_ratio::Real = 1.0,
        validation::Tuple = (),
        replace_outliers::Bool = true,
        keep_correlation_planes::Bool = false,
    )
        ws = window_size isa Int ? (window_size, window_size) : window_size
        ov = overlap isa Int ? (overlap, overlap) : overlap
        all(>=(4), ws) ||
            throw(ArgumentError("window_size must be at least 4 in each dimension, got $ws"))
        all(0 .<= ov .< ws) ||
            throw(ArgumentError("overlap must satisfy 0 ≤ overlap < window_size, got overlap=$ov for window_size=$ws"))
        correlation_method in (:cross, :phase) ||
            throw(ArgumentError("correlation_method must be :cross or :phase, got :$correlation_method"))
        apodization in (:none, :gauss) ||
            throw(ArgumentError("apodization must be :none or :gauss, got :$apodization"))
        subpixel_method in (:gauss3, :gauss9, :gauss2d, :none) ||
            throw(ArgumentError("subpixel_method must be :gauss3, :gauss9, :gauss2d, or :none, got :$subpixel_method"))
        n_peaks >= 1 ||
            throw(ArgumentError("n_peaks must be at least 1, got $n_peaks"))
        uod_threshold > 0 ||
            throw(ArgumentError("uod_threshold must be positive, got $uod_threshold"))
        uod_neighborhood >= 1 ||
            throw(ArgumentError("uod_neighborhood must be at least 1, got $uod_neighborhood"))
        min_peak_ratio >= 0 ||
            throw(ArgumentError("min_peak_ratio must be non-negative, got $min_peak_ratio"))
        new(ws, ov, correlation_method, padding, apodization, subpixel_method,
            n_peaks, uncertainty, uod_enable, Float64(uod_threshold), uod_neighborhood,
            Float64(min_peak_ratio), map(parse_validator, validation), replace_outliers,
            keep_correlation_planes)
    end
end

function Base.show(io::IO, p::PIVParameters)
    print(io, "PIVParameters(window_size=$(p.window_size), overlap=$(p.overlap), ",
        "correlation_method=:$(p.correlation_method), padding=$(p.padding), ",
        "apodization=:$(p.apodization), subpixel_method=:$(p.subpixel_method), ",
        "n_peaks=$(p.n_peaks), ", p.uncertainty ? "uncertainty=true, " : "", "uod=",
        p.uod_enable ? "(threshold=$(p.uod_threshold), neighborhood=$(p.uod_neighborhood))" : "off",
        ", min_peak_ratio=$(p.min_peak_ratio)",
        isempty(p.validation) ? "" : ", validation=$(p.validation)",
        ", replace_outliers=$(p.replace_outliers)",
        p.keep_correlation_planes ? ", keep_correlation_planes=true" : "", ")")
end

"""
    PIVResult{T<:AbstractFloat}

Result of [`run_piv`](@ref). The numeric precision `T` follows the input
images: `float(promote_type(eltype(imgA), eltype(imgB)))`, e.g. `Float32`
images produce a `PIVResult{Float32}` and the whole pipeline runs in
single precision.

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
- `uncertainty_u`, `uncertainty_v`: per-vector measurement uncertainty (one
  standard deviation, in pixels) of `u` and `v`, estimated from correlation
  statistics (Wieneke 2015) when the `uncertainty` parameter is enabled.
  `NaN` when disabled, for masked windows, and where the estimate is
  undefined (no usable correlation signal, or noise beyond the ~0.3 px
  validity of the method). The estimate describes the correlation
  measurement at the window; it is not updated when validation replaces or
  substitutes the vector.
- `outliers`: `BitMatrix` marking vectors that failed validation (UOD,
  peak-ratio check, and/or the `validation` pipeline). When outlier
  replacement is active, the `u`/`v` entries at
  these positions hold the local-median replacement rather than the measured
  displacement.
- `mask`: `BitMatrix` marking interrogation windows dropped because they
  overlap the analysis mask (see `mask` in [`run_piv`](@ref)). Masked windows
  hold `NaN` in `u`/`v`/`peak_ratio`/`correlation_moment` and are never
  counted as outliers. All-false when no mask was supplied.
- `parameters`: the `PIVParameters` of the (final) pass.
- `correlation_planes`: `nothing` unless the pass's
  `keep_correlation_planes` was set, in which case a
  `Matrix{Union{Nothing,Matrix{T}}}` indexed like the vector grid, holding a
  copy of each window's full correlation plane (`nothing` for masked/dropped
  windows). Opt-in and memory-heavy — see [`PIVParameters`](@ref).
"""
struct PIVResult{T<:AbstractFloat}
    x::Vector{T}
    y::Vector{T}
    u::Matrix{T}
    v::Matrix{T}
    peak_ratio::Matrix{T}
    correlation_moment::Matrix{T}
    uncertainty_u::Matrix{T}
    uncertainty_v::Matrix{T}
    outliers::BitMatrix
    mask::BitMatrix
    parameters::PIVParameters
    correlation_planes::Union{Nothing,Matrix{Union{Nothing,Matrix{T}}}}
end

# Backward-compatible constructors: no correlation planes stored. Keep the
# many 11-argument call sites (tests, benchmarks, ensemble/stereo/ptv) valid,
# in both the inferred (`PIVResult(...)`) and explicit (`PIVResult{T}(...)`) forms.
PIVResult(x::AbstractVector, y::AbstractVector, u::AbstractMatrix, v::AbstractMatrix,
          peak_ratio::AbstractMatrix, correlation_moment::AbstractMatrix,
          uncertainty_u::AbstractMatrix, uncertainty_v::AbstractMatrix,
          outliers, mask, parameters::PIVParameters) =
    PIVResult(x, y, u, v, peak_ratio, correlation_moment, uncertainty_u,
              uncertainty_v, outliers, mask, parameters, nothing)

PIVResult{T}(x, y, u, v, peak_ratio, correlation_moment, uncertainty_u,
             uncertainty_v, outliers, mask, parameters) where {T} =
    PIVResult{T}(x, y, u, v, peak_ratio, correlation_moment, uncertainty_u,
                 uncertainty_v, outliers, mask, parameters, nothing)

function Base.show(io::IO, r::PIVResult{T}) where {T}
    ny, nx = size(r.u)
    print(io, "PIVResult{$T}($(nx)×$(ny) grid, $(sum(r.outliers)) outliers",
          any(r.mask) ? ", $(sum(r.mask)) masked)" : ")")
end
