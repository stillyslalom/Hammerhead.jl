"""
    PhysicalScale(; pixel_size = 1.0, dt = 1.0, length_unit = "px", time_unit = "frame")
    PhysicalScale(pixel_size::Unitful.Length, dt::Unitful.Time)

Attach spatial and temporal calibration to a result. `pixel_size` is the
physical length per image pixel, and `dt` is the time between the two images
in a pair. For stereo results, positions and displacements are already in
calibration-grid world units; leave `pixel_size = 1` and supply `dt`. Both
numeric factors must be positive and finite. `length_unit` and `time_unit`
are display labels, so choose numeric factors consistent with those labels.

Attach a scale with the `scale` keyword of the drivers ([`run_piv`](@ref),
[`run_piv_sequence`](@ref), [`run_piv_ensemble`](@ref),
[`run_piv_stereo`](@ref), [`run_ptv`](@ref), [`run_ptv_sequence`](@ref),
[`track_particles`](@ref)) or after the fact with [`with_scale`](@ref).
Attaching a scale leaves stored arrays in their measured units. Use
[`physical`](@ref) to convert positions and displacements.

The Unitful-quantity constructor is provided by a package extension: load
Unitful first (`using Unitful`), e.g. `PhysicalScale(20.0u"µm", 0.5u"ms")`.
"""
struct PhysicalScale
    pixel_size::Float64
    dt::Float64
    length_unit::String
    time_unit::String

    function PhysicalScale(pixel_size::Real, dt::Real,
                           length_unit::AbstractString, time_unit::AbstractString)
        isfinite(pixel_size) && pixel_size > 0 ||
            throw(ArgumentError("pixel_size must be positive and finite, got $pixel_size"))
        isfinite(dt) && dt > 0 ||
            throw(ArgumentError("dt must be positive and finite, got $dt"))
        new(Float64(pixel_size), Float64(dt), String(length_unit), String(time_unit))
    end
end

PhysicalScale(; pixel_size::Real = 1.0, dt::Real = 1.0,
              length_unit::AbstractString = "px", time_unit::AbstractString = "frame") =
    PhysicalScale(pixel_size, dt, length_unit, time_unit)

# Display label for velocities under scale `s`, e.g. "mm/s" ("px/frame" for
# the default construction).
velocity_unit(s::PhysicalScale) = string(s.length_unit, "/", s.time_unit)

# `true` when applying `s` multiplies every field by 1 — data that is already
# physical (the scale `physical` leaves behind) or unscaled defaults.
is_identity(s::PhysicalScale) = s.pixel_size == 1.0 && s.dt == 1.0

function Base.show(io::IO, s::PhysicalScale)
    if is_identity(s)
        print(io, "PhysicalScale(identity, ", velocity_unit(s), ")")
    else
        print(io, "PhysicalScale(pixel_size = ", s.pixel_size, " ", s.length_unit,
              "/px, dt = ", s.dt, " ", s.time_unit, "/frame)")
    end
end

# Internal execution-backend hook. Backend selection belongs at the
# driver/workspace layer, not in PIVParameters or result semantics. Keep the
# names private: generic backend type names are likely to collide with other
# offloading packages.
abstract type _AbstractHammerheadBackend end

# Internal default backend. GPU implementations should live in extensions or
# sibling packages so the core package needs no device packages.
struct _CPUBackend <: _AbstractHammerheadBackend end

const _DEFAULT_BACKEND = _CPUBackend()

# Backends are selected by a public Symbol (`backend = :cpu`, `:cuda`, …) and
# resolved to a private backend object through `Val`-dispatch. The core package
# registers `:cpu` and `:ka`; a GPU extension (loaded via its device package,
# e.g. `using CUDA`) adds a method like `_resolve_backend(::Val{:cuda}) =
# _CUDABackend()`. Resolution is a once-per-call driver-level step, not hot
# path, so the runtime `Val(::Symbol)` construction is cheap.
_resolve_backend(backend::Symbol) = _resolve_backend(Val(backend))
_resolve_backend(::Val{:cpu}) = _DEFAULT_BACKEND
_resolve_backend(::Val{B}) where {B} =
    throw(ArgumentError("unsupported Hammerhead backend :$B; the core package " *
                        "provides backend = :cpu and :ka. GPU backends live in " *
                        "extension packages — load the device package " *
                        "(`using AMDGPU` for :amdgpu, `using CUDA` for :cuda) " *
                        "to register them."))

_require_cpu_backend(backend::_CPUBackend) = backend
_require_cpu_backend(backend::_AbstractHammerheadBackend) =
    throw(ArgumentError("unsupported Hammerhead backend $(typeof(backend)); " *
                        "the core package currently supports the internal CPU backend only"))

# Private backend capability predicates. Keep the defaults conservative so
# extension backends opt in explicitly as they gain implementations.
_supports_fft(::_AbstractHammerheadBackend) = false
_supports_batched_fft(::_AbstractHammerheadBackend) = false
_supports_fp64(::_AbstractHammerheadBackend) = false
_supports_unified_memory(::_AbstractHammerheadBackend) = false

_supports_fft(::_CPUBackend) = true
_supports_batched_fft(::_CPUBackend) = false
_supports_fp64(::_CPUBackend) = true
_supports_unified_memory(::_CPUBackend) = false

# Backend hooks overridable by extensions. The default backend is fully capable
# and keeps the host-thread chunk count it is given; a device backend overrides
# `_check_backend_params` to reject options its engine does not implement yet,
# and `_engine_nchunks` to run the whole grid as one logical batch (device
# engines tile it internally) instead of fanning out across host threads.
_check_backend_params(::_AbstractHammerheadBackend, passes) = nothing
_engine_nchunks(::_AbstractHammerheadBackend, requested::Int) = requested

"""
    PIVParameters(; kwargs...)

Immutable, validated configuration for a PIV analysis.

# Keyword arguments
- `window_size = (32, 32)`: interrogation window size `(rows, cols)`; an `Int` is
  expanded to a square window.
- `search_area_size = window_size`: centered search-area size `(rows, cols)`.
  Each dimension must be at least the interrogation-window size and differ
  from it by an even number of pixels, so both areas share one pixel-grid
  center. A larger search area increases the measurable first-pass
  displacement without increasing the particle-sampling window.
- `overlap = (16, 16)`: window overlap `(rows, cols)`; must satisfy
  `0 ≤ overlap < window_size`. An `Int` is expanded likewise.
- `correlation_method = :cross`: `:cross` (standard FFT cross-correlation) or
  `:phase` (phase correlation).
- `padding = false`: set `true` to zero-pad each FFT dimension to twice the
  search-area size. This replaces circular with linear correlation and avoids
  wraparound contributions; the padded arrays have four times as many elements.
- `apodization = :none`: `:gauss` applies a Gaussian window to each
  interrogation window before correlating.
- `subpixel_method = :gauss3`: `:gauss3` fits three correlation samples along
  each axis; `:gauss9` fits a 2D Gaussian to the 3×3 neighborhood; `:gauss2d`
  uses an iterative 2D fit; `:none` returns the integer peak location.
- `n_peaks = 3`: number of correlation peaks located per window (primary +
  alternatives, ≥ 1). When > 1, vectors that fail validation are re-tested
  against their secondary/tertiary peak displacements and locally consistent
  alternatives are accepted before local-median replacement kicks in
  ("peak substitution"; accepted cells are unflagged since they hold measured
  data). The top two peaks are located even when `n_peaks = 1` so the result
  can report a peak ratio. Alternative peaks use the three-point subpixel fit.
- `peak_finder = :regionalmax`: how integer correlation peaks are selected.
  The default first restricts candidates to 8-connected local maxima, so a
  real nearby secondary peak can still contribute to peak-ratio validation
  and peak substitution. `:exclusion` instead repeatedly
  take the largest remaining value outside a fixed exclusion box around
  stronger peaks.
- `uncertainty = false`: estimate a per-vector measurement uncertainty from
  correlation statistics (Wieneke 2015) into the `uncertainty_u` /
  `uncertainty_v` fields of the result. The estimator analyzes the residual
  asymmetry of the correlation peak between the two deformed windows and
  assumes the peak sits at ~zero residual displacement, so it runs on the
  final pass only and is meaningful only after multi-pass deformation has
  converged. Iterate the final pass (`max_iterations`) or repeat its window
  size in the schedule (e.g. `multipass_parameters([32, 16, 16])`). It estimates
  random correlation error, not systematic bias such as peak locking. The
  linearization is intended for uncertainties up to about 0.3 px; estimates
  above that range are not automatically rejected.
- `uod_enable = true`: validate vectors with universal outlier detection.
- `uod_threshold = 2.0`: UOD sensitivity (higher is less sensitive).
- `uod_neighborhood = 2`: UOD neighborhood layers (1 → 3×3, 2 → 5×5, ...).
  A larger neighborhood can help distinguish smooth gradients from isolated
  vectors, including near field edges.
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
- `max_iterations = 1`: iteration budget for this pass. With the default 1
  the pass correlates once. With more, the pass *iterates*:
  its own validated field becomes the deformation predictor, the images are
  re-deformed, and the windows re-correlated, until the field converges (see
  `convergence_tol`) or the budget is spent. A flagged vector can be
  remeasured at the same window size before the next pass. Sweeps beyond the
  first always replace flagged vectors
  internally (the next predictor must be well behaved), but the returned
  field still honors `replace_outliers`.
- `convergence_tol = 0.05`: convergence threshold in pixels for the
  `max_iterations` loop. The pass stops early when the 95th percentile of
  per-vector changes over unmasked nodes falls below this value.
  `0` disables the early exit (the pass always runs `max_iterations`
  sweeps). Unused when `max_iterations == 1`.
- `keep_correlation_planes = false`: retain each window's full correlation
  plane in the result's `correlation_planes` field for inspection. This can
  use substantial memory: 32×32 planes on a 100×100 grid in `Float64` hold
  about 82 MB of plane values before array overhead (about 328 MB if padding
  makes each plane 64×64). As
  [`run_piv`](@ref) returns the final pass, set it on the final pass only
  (pair it with the `final` keyword of [`multipass_parameters`](@ref)). See
  [`PIVResult`](@ref).

Multi-pass interrogation is configured with a vector of `PIVParameters` (one
per pass) — see [`run_piv`](@ref) and [`multipass_parameters`](@ref).
"""
struct PIVParameters
    window_size::Tuple{Int,Int}
    search_area_size::Tuple{Int,Int}
    overlap::Tuple{Int,Int}
    correlation_method::Symbol
    padding::Bool
    apodization::Symbol
    subpixel_method::Symbol
    n_peaks::Int
    peak_finder::Symbol
    uncertainty::Bool
    uod_enable::Bool
    uod_threshold::Float64
    uod_neighborhood::Int
    min_peak_ratio::Float64
    validation::Tuple
    replace_outliers::Bool
    max_iterations::Int
    convergence_tol::Float64
    keep_correlation_planes::Bool

    function PIVParameters(;
        window_size::Union{Int,Tuple{Int,Int}} = (32, 32),
        search_area_size::Union{Nothing,Int,Tuple{Int,Int}} = nothing,
        overlap::Union{Int,Tuple{Int,Int}} = (16, 16),
        correlation_method::Symbol = :cross,
        padding::Bool = false,
        apodization::Symbol = :none,
        subpixel_method::Symbol = :gauss3,
        n_peaks::Int = 3,
        peak_finder::Symbol = :regionalmax,
        uncertainty::Bool = false,
        uod_enable::Bool = true,
        uod_threshold::Real = 2.0,
        uod_neighborhood::Int = 2,
        min_peak_ratio::Real = 1.0,
        validation::Tuple = (),
        replace_outliers::Bool = true,
        max_iterations::Int = 1,
        convergence_tol::Real = 0.05,
        keep_correlation_planes::Bool = false,
    )
        ws = window_size isa Int ? (window_size, window_size) : window_size
        ss = search_area_size === nothing ? ws :
             search_area_size isa Int ? (search_area_size, search_area_size) : search_area_size
        ov = overlap isa Int ? (overlap, overlap) : overlap
        all(>=(4), ws) ||
            throw(ArgumentError("window_size must be at least 4 in each dimension, got $ws"))
        all(ss .>= ws) ||
            throw(ArgumentError("search_area_size must be at least window_size in each dimension, got search_area_size=$ss for window_size=$ws"))
        all(iseven.(ss .- ws)) ||
            throw(ArgumentError("search_area_size - window_size must be even in each dimension so the areas share a center, got search_area_size=$ss for window_size=$ws"))
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
        peak_finder in (:exclusion, :regionalmax) ||
            throw(ArgumentError("peak_finder must be :exclusion or :regionalmax, got :$peak_finder"))
        uod_threshold > 0 ||
            throw(ArgumentError("uod_threshold must be positive, got $uod_threshold"))
        uod_neighborhood >= 1 ||
            throw(ArgumentError("uod_neighborhood must be at least 1, got $uod_neighborhood"))
        min_peak_ratio >= 0 ||
            throw(ArgumentError("min_peak_ratio must be non-negative, got $min_peak_ratio"))
        max_iterations >= 1 ||
            throw(ArgumentError("max_iterations must be at least 1, got $max_iterations"))
        convergence_tol >= 0 ||
            throw(ArgumentError("convergence_tol must be non-negative, got $convergence_tol"))
        new(ws, ss, ov, correlation_method, padding, apodization, subpixel_method,
            n_peaks, peak_finder, uncertainty, uod_enable, Float64(uod_threshold), uod_neighborhood,
            Float64(min_peak_ratio), map(parse_validator, validation), replace_outliers,
            max_iterations, Float64(convergence_tol), keep_correlation_planes)
    end
end

function Base.show(io::IO, p::PIVParameters)
    print(io, "PIVParameters(window_size=$(p.window_size)",
        p.search_area_size == p.window_size ? "" : ", search_area_size=$(p.search_area_size)",
        ", overlap=$(p.overlap), ",
        "correlation_method=:$(p.correlation_method), padding=$(p.padding), ",
        "apodization=:$(p.apodization), subpixel_method=:$(p.subpixel_method), ",
        "n_peaks=$(p.n_peaks), ",
        p.peak_finder === :regionalmax ? "" : "peak_finder=:$(p.peak_finder), ",
        p.uncertainty ? "uncertainty=true, " : "", "uod=",
        p.uod_enable ? "(threshold=$(p.uod_threshold), neighborhood=$(p.uod_neighborhood))" : "off",
        ", min_peak_ratio=$(p.min_peak_ratio)",
        isempty(p.validation) ? "" : ", validation=$(p.validation)",
        ", replace_outliers=$(p.replace_outliers)",
        p.max_iterations > 1 ?
            ", max_iterations=$(p.max_iterations), convergence_tol=$(p.convergence_tol)" : "",
        p.keep_correlation_planes ? ", keep_correlation_planes=true" : "", ")")
end

"""
    PIVResult{T<:AbstractFloat}

Vector field returned by [`run_piv`](@ref). Numeric arrays use
`T = float(promote_type(eltype(imgA), eltype(imgB)))`; for example, Float32
images produce a `PIVResult{Float32}`. Uncertainty statistics accumulate in
Float64 before being stored as `T`.

# Fields
- `x`, `y`: window-center coordinates of the interrogation grid (`x` along
  columns, `y` along rows, in pixels).
- `u`, `v`: displacement components on the `(length(y), length(x))` grid; `u` is
  the column (x) displacement and `v` the row (y) displacement, in pixels. A
  particle at `(row, col)` in the first image is found at `(row + v, col + u)`
  in the second.
- `peak_ratio`: primary-to-secondary correlation peak ratio per window.
  A higher ratio indicates a more distinct primary peak, not a calibrated
  probability that the vector is correct.
- `correlation_moment`: peak-spread diagnostic per window (lower is sharper).
  It is not a calibrated uncertainty.
- `uncertainty_u`, `uncertainty_v`: per-vector measurement uncertainty (one
  standard deviation, in pixels) of `u` and `v`, estimated from correlation
  statistics (Wieneke 2015) when the `uncertainty` parameter is enabled.
  `NaN` when disabled, for masked windows, or when the correlation statistics
  cannot yield an estimate. The estimate describes the original correlation
  measurement; validation does not update it after replacing or substituting
  a vector.
- `outliers`: `BitMatrix` marking vectors that failed validation (UOD,
  peak-ratio check, and/or the `validation` pipeline). When outlier
  replacement is active, the `u`/`v` entries at
  these positions hold the local-median replacement rather than the measured
  displacement.
- `mask`: `BitMatrix` marking nodes dropped because the masked fraction of
  their interrogation or search-area footprint reaches `mask_threshold` (see
  `mask` in [`run_piv`](@ref)). Masked windows
  hold `NaN` in `u`/`v`/`peak_ratio`/`correlation_moment` and are never
  counted as outliers. All-false when no mask was supplied.
- `parameters`: the `PIVParameters` of the (final) pass.
- `correlation_planes`: `nothing` unless the pass's
  `keep_correlation_planes` was set, in which case a
  `Matrix{Union{Nothing,Matrix{T}}}` indexed like the vector grid, holding a
  copy of each window's full correlation plane (`nothing` for masked/dropped
  windows). See [`PIVParameters`](@ref) for memory use.
- `scale`: the [`PhysicalScale`](@ref) attached via the `scale` keyword of
  [`run_piv`](@ref) or [`with_scale`](@ref); `nothing` when none was
  attached. The arrays above stay in pixels until
  [`physical`](@ref) converts them.
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
    scale::Union{Nothing,PhysicalScale}
end

# Backward-compatible constructors: no correlation planes and/or no physical
# scale stored. Keep the many 11- and 12-argument call sites (tests,
# benchmarks, pipeline/ensemble/stereo/ptv) valid, in both the inferred
# (`PIVResult(...)`) and explicit (`PIVResult{T}(...)`) forms.
PIVResult(x::AbstractVector, y::AbstractVector, u::AbstractMatrix, v::AbstractMatrix,
          peak_ratio::AbstractMatrix, correlation_moment::AbstractMatrix,
          uncertainty_u::AbstractMatrix, uncertainty_v::AbstractMatrix,
          outliers, mask, parameters::PIVParameters) =
    PIVResult(x, y, u, v, peak_ratio, correlation_moment, uncertainty_u,
              uncertainty_v, outliers, mask, parameters, nothing, nothing)

PIVResult(x::AbstractVector, y::AbstractVector, u::AbstractMatrix, v::AbstractMatrix,
          peak_ratio::AbstractMatrix, correlation_moment::AbstractMatrix,
          uncertainty_u::AbstractMatrix, uncertainty_v::AbstractMatrix,
          outliers, mask, parameters::PIVParameters, correlation_planes) =
    PIVResult(x, y, u, v, peak_ratio, correlation_moment, uncertainty_u,
              uncertainty_v, outliers, mask, parameters, correlation_planes, nothing)

PIVResult{T}(x, y, u, v, peak_ratio, correlation_moment, uncertainty_u,
             uncertainty_v, outliers, mask, parameters) where {T} =
    PIVResult{T}(x, y, u, v, peak_ratio, correlation_moment, uncertainty_u,
                 uncertainty_v, outliers, mask, parameters, nothing, nothing)

PIVResult{T}(x, y, u, v, peak_ratio, correlation_moment, uncertainty_u,
             uncertainty_v, outliers, mask, parameters, correlation_planes) where {T} =
    PIVResult{T}(x, y, u, v, peak_ratio, correlation_moment, uncertainty_u,
                 uncertainty_v, outliers, mask, parameters, correlation_planes, nothing)

function Base.show(io::IO, r::PIVResult{T}) where {T}
    ny, nx = size(r.u)
    print(io, "PIVResult{$T}($(nx)×$(ny) grid, $(sum(r.outliers)) outliers",
          any(r.mask) ? ", $(sum(r.mask)) masked)" : ")")
end
