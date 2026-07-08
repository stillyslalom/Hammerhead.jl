# Two-frame PTV (Phase 8): predictor-guided nearest-neighbor matching with
# global cost-sorted greedy one-to-one assignment ("super-resolution PIV",
# Keane, Adrian & Zhang 1995). With a decent predictor the match residual is
# ≲1–2 px — far below the mean particle spacing — so greedy assignment is
# near-optimal and O(n log n); no Hungarian or relaxation method (deferred).
#
# Position attribution is frame-A (x/y are the frame-A particle positions,
# u/v the displacement to frame B), which matches the SyntheticData
# forward-Euler contract exactly: each particle's true displacement is the
# velocity at its launch point, so ground-truth tests compare directly with no
# midpoint correction (unlike PIV's symmetric deformation).

"""
    PTVParameters(; kwargs...)

Immutable, validated configuration for particle detection, matching, and
scattered-vector validation in [`detect_particles`](@ref), [`run_ptv`](@ref),
and [`track_particles`](@ref).

# Keyword arguments
- `threshold = :auto`: detection intensity threshold — `:auto` for a robust
  floor `bg + threshold_k·MAD`, or an absolute intensity `Real`.
- `threshold_k = 4.0`: MAD multiplier for the `:auto` threshold.
- `min_separation = 2.0`: minimum spacing (px) between accepted peaks; the
  brighter of a closer pair is kept.
- `min_diameter = 1.0`, `max_diameter = 12.0`: accepted particle diameter
  range (px, 4σ). Particles with a `NaN` diameter (centroid fallback) pass.
- `search_radius = 3.0`: match search radius (px) around the predicted
  frame-B position of each frame-A particle.
- `uod_enable = true`: flag outliers with the scattered normalized median test
  (Duncan et al. 2010).
- `uod_threshold = 2.0`: scattered-UOD sensitivity (higher is less sensitive).
- `uod_neighbors = 8`: number of nearest matched neighbors used per vector.
- `uod_epsilon = 0.1`: px noise floor for the scattered UOD (same rationale as
  the gridded UOD `epsilon`).
"""
struct PTVParameters
    threshold::Union{Symbol,Float64}
    threshold_k::Float64
    min_separation::Float64
    min_diameter::Float64
    max_diameter::Float64
    search_radius::Float64
    uod_enable::Bool
    uod_threshold::Float64
    uod_neighbors::Int
    uod_epsilon::Float64

    function PTVParameters(;
        threshold::Union{Symbol,Real} = :auto,
        threshold_k::Real = 4.0,
        min_separation::Real = 2.0,
        min_diameter::Real = 1.0,
        max_diameter::Real = 12.0,
        search_radius::Real = 3.0,
        uod_enable::Bool = true,
        uod_threshold::Real = 2.0,
        uod_neighbors::Int = 8,
        uod_epsilon::Real = 0.1,
    )
        (threshold === :auto || threshold isa Real) ||
            throw(ArgumentError("threshold must be :auto or a Real, got $threshold"))
        threshold_k > 0 ||
            throw(ArgumentError("threshold_k must be positive, got $threshold_k"))
        min_separation > 0 ||
            throw(ArgumentError("min_separation must be positive, got $min_separation"))
        search_radius > 0 ||
            throw(ArgumentError("search_radius must be positive, got $search_radius"))
        uod_threshold > 0 ||
            throw(ArgumentError("uod_threshold must be positive, got $uod_threshold"))
        uod_epsilon > 0 ||
            throw(ArgumentError("uod_epsilon must be positive, got $uod_epsilon"))
        (0 < min_diameter < max_diameter) ||
            throw(ArgumentError("must satisfy 0 < min_diameter < max_diameter, got ($min_diameter, $max_diameter)"))
        uod_neighbors >= 3 ||
            throw(ArgumentError("uod_neighbors must be at least 3, got $uod_neighbors"))
        thr = threshold === :auto ? :auto : Float64(threshold)
        new(thr, Float64(threshold_k), Float64(min_separation), Float64(min_diameter),
            Float64(max_diameter), Float64(search_radius), uod_enable,
            Float64(uod_threshold), uod_neighbors, Float64(uod_epsilon))
    end
end

function Base.show(io::IO, p::PTVParameters)
    print(io, "PTVParameters(threshold=", p.threshold === :auto ? ":auto" : p.threshold,
        ", threshold_k=$(p.threshold_k), min_separation=$(p.min_separation), ",
        "diameter=($(p.min_diameter), $(p.max_diameter)), search_radius=$(p.search_radius), ",
        "uod=", p.uod_enable ?
            "(threshold=$(p.uod_threshold), neighbors=$(p.uod_neighbors), epsilon=$(p.uod_epsilon))" : "off",
        ")")
end

"""
    PTVResult{T<:AbstractFloat}

Result of [`run_ptv`](@ref). The numeric precision `T` follows the input
images, like [`PIVResult`](@ref).

# Fields
- `x`, `y`: frame-A particle positions of the matched pairs (`x` along columns,
  `y` along rows, in pixels).
- `u`, `v`: displacement to frame B (px, package sign convention: a particle at
  `(y, x)` in frame A is found at `(y + v, x + u)` in frame B).
- `match_residual`: `|found − predicted|` distance (px) — a match-quality proxy.
- `outliers`: `BitVector` of scattered-UOD flags. Matched vectors are never
  replaced (a track is a measurement of one particle), only flagged.
- `index_a`, `index_b`: the match `i` pairs `particles_a[index_a[i]]` with
  `particles_b[index_b[i]]`.
- `particles_a`, `particles_b`: all detections in each frame (unmatched ones are
  seeding dropout or new entries), kept for diagnostics.
- `parameters`: the `PTVParameters` used.
"""
struct PTVResult{T<:AbstractFloat}
    x::Vector{T}
    y::Vector{T}
    u::Vector{T}
    v::Vector{T}
    match_residual::Vector{T}
    outliers::BitVector
    index_a::Vector{Int}
    index_b::Vector{Int}
    particles_a::Particles{T}
    particles_b::Particles{T}
    parameters::PTVParameters
end

function Base.show(io::IO, r::PTVResult{T}) where {T}
    print(io, "PTVResult{$T}($(length(r.x)) matches of ",
          "$(length(r.particles_a))/$(length(r.particles_b)) particles, ",
          "$(sum(r.outliers)) outliers)")
end

# ---------------------------------------------------------------------------
# Matching (shared with tracking.jl)
# ---------------------------------------------------------------------------

# Greedy one-to-one match of A-particles at predicted positions
# `(pred_x, pred_y)` to the B-particles indexed by `cl_b`, within
# `search_radius`. Every candidate pair within range is scored by squared
# distance to the prediction, all candidates are sorted ascending by
# `(cost, i, j)` (deterministic), and pairs whose A- and B-index are both
# unused are accepted greedily. Returns `(index_a, index_b, residual)`.
function greedy_match(pred_x::AbstractVector{T}, pred_y::AbstractVector{T},
                      cl_b::CellList{T}, search_radius::Real) where {T}
    candidates = Tuple{T,Int,Int}[]
    buf = Int[]
    for i in eachindex(pred_x)
        within_radius!(buf, cl_b, pred_x[i], pred_y[i], search_radius)
        for j in buf
            cost = (cl_b.px[j] - pred_x[i])^2 + (cl_b.py[j] - pred_y[i])^2
            push!(candidates, (cost, i, j))
        end
    end
    sort!(candidates)                    # lexicographic: (cost, i, j)
    used_a = falses(length(pred_x))
    used_b = falses(cl_b.n)
    index_a = Int[]; index_b = Int[]; residual = T[]
    for (cost, i, j) in candidates
        (used_a[i] || used_b[j]) && continue
        used_a[i] = true
        used_b[j] = true
        push!(index_a, i); push!(index_b, j); push!(residual, sqrt(cost))
    end
    return index_a, index_b, residual
end

# ---------------------------------------------------------------------------
# Predictor resolution
# ---------------------------------------------------------------------------

# Turn the `predictor` option into a gridded (x, y, u, v) NamedTuple, or
# `nothing` for pure nearest-neighbor (zero displacement).
resolve_predictor(::Nothing, imgA, imgB, piv_passes, mask, ::Type) = nothing
resolve_predictor(p::PIVResult, imgA, imgB, piv_passes, mask, ::Type) =
    build_predictor(p, true)
resolve_predictor(p::NamedTuple, imgA, imgB, piv_passes, mask, ::Type) = begin
    all(k -> hasproperty(p, k), (:x, :y, :u, :v)) ||
        throw(ArgumentError("a NamedTuple predictor must have x, y, u, and v fields"))
    p
end
function resolve_predictor(p::Symbol, imgA, imgB, piv_passes, mask, ::Type)
    p === :piv ||
        throw(ArgumentError("predictor Symbol must be :piv, got :$p"))
    return build_predictor(run_piv(imgA, imgB, piv_passes; mask), true)
end
resolve_predictor(p, imgA, imgB, piv_passes, mask, ::Type) =
    throw(ArgumentError("predictor must be :piv, a PIVResult, a NamedTuple with " *
                        "x/y/u/v fields, or nothing, got $(typeof(p))"))

# Predicted frame-B positions of the frame-A particles: A position plus the
# predictor displacement interpolated there (Gridded linear + Flat, like
# apply_predictor). `nothing` predictor → zero displacement.
function predicted_positions(pa::Particles{T}, pred) where {T}
    px = Vector{T}(undef, length(pa))
    py = Vector{T}(undef, length(pa))
    if pred === nothing
        copyto!(px, pa.x)
        copyto!(py, pa.y)
        return px, py
    end
    itp_u = extrapolate(interpolate((pred.y, pred.x), pred.u, Gridded(Linear())), Flat())
    itp_v = extrapolate(interpolate((pred.y, pred.x), pred.v, Gridded(Linear())), Flat())
    for i in eachindex(pa.x)
        px[i] = pa.x[i] + T(itp_u(pa.y[i], pa.x[i]))
        py[i] = pa.y[i] + T(itp_v(pa.y[i], pa.x[i]))
    end
    return px, py
end

# ---------------------------------------------------------------------------
# Scattered universal outlier detection (Duncan et al. 2010)
# ---------------------------------------------------------------------------

# Cell size giving ~1 point per cell, for the scattered neighbor searches.
function spacing_cell_size(x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    n = length(x)
    n > 1 || return T(1)
    area = (maximum(x) - minimum(x)) * (maximum(y) - minimum(y))
    area > 0 || return T(1)
    return T(max(1.0, sqrt(area / n)))
end

# Normalized-median test on scattered matches: for each vector, compare it to
# the median of its `uod_neighbors` nearest matched neighbors, normalized by
# the median absolute neighbor residual plus `uod_epsilon`. Fewer than 3
# neighbors → left unflagged. Flags a `BitVector`; u/v are never changed.
function scattered_uod(x::Vector{T}, y::Vector{T}, u::Vector{T}, v::Vector{T},
                       params::PTVParameters) where {T}
    n = length(x)
    flags = falses(n)
    n >= 4 || return flags
    cl = build_cell_list(x, y, spacing_cell_size(x, y))
    thr = params.uod_threshold
    eps = params.uod_epsilon
    for i in 1:n
        nbrs = knn(cl, x[i], y[i], params.uod_neighbors; skip = i)
        length(nbrs) >= 3 || continue
        nu = @view u[nbrs]
        nv = @view v[nbrs]
        um = median(nu)
        vm = median(nv)
        ru = median(abs.(nu .- um))
        rv = median(abs.(nv .- vm))
        if abs(u[i] - um) / (ru + eps) > thr || abs(v[i] - vm) / (rv + eps) > thr
            flags[i] = true
        end
    end
    return flags
end

# ---------------------------------------------------------------------------
# run_ptv
# ---------------------------------------------------------------------------

"""
    run_ptv(imgA, imgB, params = PTVParameters();
            predictor = :piv, piv_passes = multipass_parameters([64, 32]),
            mask = nothing) -> PTVResult

Two-frame particle tracking velocimetry on an image pair. Particles are
detected in both frames ([`detect_particles`](@ref)), each frame-A particle is
matched to a frame-B particle by predictor-guided nearest neighbor with global
cost-sorted greedy one-to-one assignment [Keane1995](@cite), and the matched
displacements are validated with a scattered normalized median test
[Duncan2010](@cite) that flags — but never replaces — outliers.

The `predictor` supplies the displacement field that centers each match search:

- `:piv` (default) runs a coarse [`run_piv`](@ref) on the pair internally
  (`piv_passes`), so `run_ptv` works out of the box at realistic densities and
  displacements — the hybrid PIV-guided default.
- a [`PIVResult`](@ref) reuses an existing field (skips the redundant run).
- a NamedTuple with `x`, `y`, `u`, `v` fields is used as-is.
- `nothing` gives pure nearest neighbor (zero displacement); set
  `search_radius` accordingly.

`mask` is an optional image-sized `Bool` matrix marking excluded pixels
(`true` = excluded); it is applied to detection in both frames and forwarded to
the internal PIV run. Position attribution is frame-A: `x`/`y` are frame-A
positions and `u`/`v` the displacement to frame B (see [`PTVResult`](@ref)).
Empty frames or no matches yield a valid empty result rather than throwing.
"""
function run_ptv(imgA::AbstractMatrix{<:Real}, imgB::AbstractMatrix{<:Real},
                 params::PTVParameters = PTVParameters();
                 predictor = :piv,
                 piv_passes = multipass_parameters([64, 32]),
                 mask::Union{Nothing,AbstractMatrix{Bool}} = nothing)
    size(imgA) == size(imgB) ||
        throw(DimensionMismatch("images must have the same size, got $(size(imgA)) and $(size(imgB))"))
    mask === nothing || size(mask) == size(imgA) ||
        throw(DimensionMismatch("mask must have the same size as the images, got $(size(mask))"))
    T = float(promote_type(eltype(imgA), eltype(imgB)))

    pa = detect_particles(imgA, params; mask)
    pb = detect_particles(imgB, params; mask)
    # No detections in a frame means no matches: skip the (image-size-sensitive)
    # predictor entirely and return a valid empty result.
    if isempty(pa.x) || isempty(pb.x)
        return PTVResult{T}(T[], T[], T[], T[], T[], falses(0), Int[], Int[], pa, pb, params)
    end
    pred = resolve_predictor(predictor, imgA, imgB, piv_passes, mask, T)
    px, py = predicted_positions(pa, pred)

    cl_b = build_cell_list(pb.x, pb.y, params.search_radius)
    index_a, index_b, residual = greedy_match(px, py, cl_b, params.search_radius)

    n = length(index_a)
    x = T[pa.x[index_a[k]] for k in 1:n]
    y = T[pa.y[index_a[k]] for k in 1:n]
    u = T[pb.x[index_b[k]] - pa.x[index_a[k]] for k in 1:n]
    v = T[pb.y[index_b[k]] - pa.y[index_a[k]] for k in 1:n]

    outliers = params.uod_enable ? scattered_uod(x, y, u, v, params) : falses(n)
    return PTVResult{T}(x, y, u, v, residual, outliers, index_a, index_b, pa, pb, params)
end

# ---------------------------------------------------------------------------
# ptv_to_grid
# ---------------------------------------------------------------------------

"""
    ptv_to_grid(result::PTVResult, image_size;
                window_size = (32, 32), overlap = (16, 16),
                min_count = 3, include_outliers = false) -> PIVResult

Bin the matched PTV vectors onto the regular interrogation grid that
[`run_piv`](@ref) would use for `image_size`. Each vector contributes to the
single bin whose center is nearest its frame-A position, and each bin value is
the component-wise **median** of its vectors (robust to residual mismatches).
Bins with fewer than `min_count` vectors are masked (`mask = true`, `NaN`
fields), so the result plugs straight into the masked-result conventions
([`field_statistics`](@ref), [`build_predictor`](@ref), plotting). Outlier-
flagged vectors are excluded unless `include_outliers`.

This is a **binned PTV field**, not a correlation measurement: `peak_ratio`,
`correlation_moment`, and the uncertainty fields are `NaN`, and no vector is
flagged an outlier in the returned grid.
"""
function ptv_to_grid(result::PTVResult{T}, image_size::Tuple{Int,Int};
                     window_size = (32, 32), overlap = (16, 16),
                     min_count::Int = 3, include_outliers::Bool = false) where {T}
    keep = include_outliers ? eachindex(result.x) :
           [k for k in eachindex(result.x) if !result.outliers[k]]
    return bin_to_grid(T, result.x[keep], result.y[keep], result.u[keep], result.v[keep],
                       image_size, PIVParameters(; window_size, overlap), min_count)
end

# Bin scattered vectors `(mx, my, mu, mv)` onto the interrogation grid of
# `params` for `image_size`: each vector joins the single nearest-center bin
# (integer index arithmetic, not an O(n·bins) scan) and each bin holds the
# component-wise median; bins with < `min_count` vectors are masked. Shared by
# `ptv_to_grid` and the tracking predictor.
function bin_to_grid(::Type{T}, mx::AbstractVector, my::AbstractVector,
                     mu::AbstractVector, mv::AbstractVector,
                     image_size::Tuple{Int,Int}, params::PIVParameters,
                     min_count::Int) where {T}
    grid = pass_grid(T, image_size, params, nothing, 0.5)
    xs, ys = grid.x, grid.y
    nx, ny = length(xs), length(ys)
    (nx > 0 && ny > 0) ||
        throw(ArgumentError("window_size $(params.window_size) yields an empty grid for image_size $image_size"))
    stepc = params.window_size[2] - params.overlap[2]
    stepr = params.window_size[1] - params.overlap[1]
    x1, y1 = xs[1], ys[1]

    binu = [T[] for _ in 1:ny, _ in 1:nx]
    binv = [T[] for _ in 1:ny, _ in 1:nx]
    for k in eachindex(mx)
        gj = clamp(round(Int, (mx[k] - x1) / stepc) + 1, 1, nx)
        gi = clamp(round(Int, (my[k] - y1) / stepr) + 1, 1, ny)
        push!(binu[gi, gj], mu[k])
        push!(binv[gi, gj], mv[k])
    end

    u = fill(T(NaN), ny, nx)
    v = fill(T(NaN), ny, nx)
    nanfield() = fill(T(NaN), ny, nx)
    gmask = falses(ny, nx)
    for gj in 1:nx, gi in 1:ny
        if length(binu[gi, gj]) >= min_count
            u[gi, gj] = T(median(binu[gi, gj]))
            v[gi, gj] = T(median(binv[gi, gj]))
        else
            gmask[gi, gj] = true
        end
    end
    return PIVResult{T}(xs, ys, u, v, nanfield(), nanfield(), nanfield(), nanfield(),
                        falses(ny, nx), gmask, params)
end
