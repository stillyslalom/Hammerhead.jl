# Particle detection for PTV (Phase 8): local-maxima finding with 3-point
# log-Gaussian subpixel refinement. Detection is deliberately in-house rather
# than target_detection.jl's connected-component blobs or an ecosystem blob
# detector — PIV-density particle images are 2–4 px Gaussians that overlap, so
# labeling merges them and loses the subpixel fidelity PTV needs (the
# sanctioned exception to the JuliaImages ecosystem policy).
#
# The uniform-cell neighbor list defined here is shared by dedupe (this file),
# candidate matching, and scattered UOD (ptv.jl) — the no-new-dependencies
# rule rules out NearestNeighbors.jl.

# ---------------------------------------------------------------------------
# Uniform cell list for 2D neighbor queries
# ---------------------------------------------------------------------------

# Points are bucketed into square cells of side `cell_size`; a query scans the
# block of cells covering the search disc. Coordinates may be negative (a
# predicted position can fall outside the frame), so cells are stored in a Dict
# rather than a dense grid.
struct CellList{T<:AbstractFloat}
    cell_size::T
    px::Vector{T}          # column (x) positions of the indexed points
    py::Vector{T}          # row (y) positions
    cells::Dict{Tuple{Int,Int},Vector{Int}}
    n::Int
end

_cell(s::Real, x::Real, y::Real) = (floor(Int, x / s), floor(Int, y / s))

function build_cell_list(px::Vector{T}, py::Vector{T}, cell_size::Real) where {T<:AbstractFloat}
    length(px) == length(py) ||
        throw(ArgumentError("px and py must have the same length"))
    cell_size > 0 || throw(ArgumentError("cell_size must be positive, got $cell_size"))
    s = T(cell_size)
    cells = Dict{Tuple{Int,Int},Vector{Int}}()
    for i in eachindex(px)
        push!(get!(() -> Int[], cells, _cell(s, px[i], py[i])), i)
    end
    return CellList{T}(s, px, py, cells, length(px))
end

# All indexed points within Euclidean distance `radius` of `(qx, qy)`, appended
# to `out` (cleared first). `skip` (an index into the point set) is excluded.
function within_radius!(out::Vector{Int}, cl::CellList, qx::Real, qy::Real,
                        radius::Real; skip::Int = 0)
    empty!(out)
    span = ceil(Int, radius / cl.cell_size)
    cx, cy = _cell(cl.cell_size, qx, qy)
    r2 = radius^2
    for jx in (cx - span):(cx + span), jy in (cy - span):(cy + span)
        bucket = get(cl.cells, (jx, jy), nothing)
        bucket === nothing && continue
        for i in bucket
            i == skip && continue
            ((cl.px[i] - qx)^2 + (cl.py[i] - qy)^2 <= r2) && push!(out, i)
        end
    end
    return out
end

# Indices of the `k` nearest indexed points to `(qx, qy)`, nearest first,
# excluding `skip`. The cell span is grown until the k-th distance is provably
# covered (all points within distance D lie within ⌈D/cell_size⌉ cell rings);
# returns fewer than `k` only when the whole set is smaller.
function knn(cl::CellList, qx::Real, qy::Real, k::Int; skip::Int = 0)
    avail = skip == 0 ? cl.n : cl.n - 1
    k = min(k, avail)
    k <= 0 && return Int[]
    s = cl.cell_size
    cx, cy = _cell(s, qx, qy)
    cand = Int[]
    dist2(i) = (cl.px[i] - qx)^2 + (cl.py[i] - qy)^2
    span = 1
    while true
        empty!(cand)
        for jx in (cx - span):(cx + span), jy in (cy - span):(cy + span)
            bucket = get(cl.cells, (jx, jy), nothing)
            bucket === nothing && continue
            for i in bucket
                i == skip && continue
                push!(cand, i)
            end
        end
        if length(cand) >= k
            perm = sortperm(cand; by = dist2)
            dk = sqrt(dist2(cand[perm[k]]))
            if span >= ceil(Int, dk / s)
                return cand[perm[1:k]]
            end
        end
        # No cell outside the current span can hold a point once every point is
        # already collected: stop growing and return the best available.
        if length(cand) >= avail
            perm = sortperm(cand; by = dist2)
            return cand[perm[1:min(k, length(cand))]]
        end
        span += 1
    end
end

# ---------------------------------------------------------------------------
# Particles container
# ---------------------------------------------------------------------------

"""
    Particles{T<:AbstractFloat}

Detected particles from one frame (a struct-of-arrays), as produced by
[`detect_particles`](@ref). `x` is the subpixel column (x) position, `y` the
subpixel row (y) position, `intensity` the background-subtracted peak
amplitude, and `diameter` the 4σ estimate from the Gaussian fit (`NaN` when
both axes fell back to the centroid, e.g. a saturated particle).
"""
struct Particles{T<:AbstractFloat}
    x::Vector{T}
    y::Vector{T}
    intensity::Vector{T}
    diameter::Vector{T}
end

Base.length(p::Particles) = length(p.x)

Base.show(io::IO, p::Particles{T}) where {T} =
    print(io, "Particles{$T}($(length(p)) particles)")

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

"""
    detect_particles(img, params = PTVParameters(); mask = nothing) -> Particles

Detect particles in a single image as subpixel local maxima with Gaussian peak
fits. The numeric precision follows the image: `T = float(eltype(img))`.

The algorithm is:

1. **Threshold** from a robust intensity floor. `bg` is the median over valid
   (unmasked) pixels and `mad = 1.4826 · median(|img − bg|)`; with
   `params.threshold === :auto` the threshold is `bg + params.threshold_k·mad`,
   otherwise `params.threshold` is used as an absolute intensity.
2. **Local maxima** among interior pixels above threshold (deterministic single
   pick on flat plateaus; masked pixels are skipped).
3. **Subpixel refinement** by an independent 3-point log-Gaussian fit per axis
   on background-subtracted intensities, falling back to an intensity-weighted
   3×3 centroid on any axis where the fit is undefined.
4. **Diameter filtering** to `[params.min_diameter, params.max_diameter]`
   (particles with a `NaN` diameter from the centroid fallback are kept).
5. **Deduplication**: brighter particles win when two lie within
   `params.min_separation` px.

See [`PTVParameters`](@ref) for the detection parameters and [`run_ptv`](@ref)
for the full tracking pipeline.
"""
function detect_particles(img::AbstractMatrix{<:Real}, params = PTVParameters();
                          mask::Union{Nothing,AbstractMatrix{Bool}} = nothing)
    # `params` is intentionally unannotated: `PTVParameters` is defined in ptv.jl,
    # which is included after this file (its default-value expression is only
    # evaluated at call time, so the forward reference is fine).
    mask === nothing || size(mask) == size(img) ||
        throw(DimensionMismatch("mask must have the same size as the image, got $(size(mask))"))
    T = float(eltype(img))
    nr, nc = size(img)

    # 1. Robust background/threshold over valid pixels (Float64 scalars).
    valid = Float64[]
    for c in 1:nc, r in 1:nr
        (mask !== nothing && mask[r, c]) && continue
        push!(valid, Float64(img[r, c]))
    end
    if isempty(valid)
        return Particles(T[], T[], T[], T[])
    end
    bg = median(valid)
    mad = 1.4826 * median!(abs.(valid .- bg))
    if params.threshold === :auto
        thr = mad > 0 ? bg + params.threshold_k * mad :
              bg + 0.1 * (maximum(valid) - bg)
    else
        thr = Float64(params.threshold)
    end

    # 2. Local maxima (interior, above threshold). Plateau ties broken by
    # column-major linear order: `≥` earlier neighbors, `>` later ones.
    cx = Float64[]; cy = Float64[]; cint = Float64[]; cdia = Float64[]
    for c in 2:(nc - 1), r in 2:(nr - 1)
        (mask !== nothing && mask[r, c]) && continue
        I0 = Float64(img[r, c])
        I0 > thr || continue
        is_local_maximum(img, r, c) || continue

        # 3. Subpixel refinement on J = I − bg, per axis.
        δx, σx, okx = fit_axis(Float64(img[r, c - 1]) - bg, I0 - bg, Float64(img[r, c + 1]) - bg)
        δy, σy, oky = fit_axis(Float64(img[r - 1, c]) - bg, I0 - bg, Float64(img[r + 1, c]) - bg)
        if !(okx && oky)
            gx, gy = centroid3(img, r, c, bg)
            okx || (δx = gx - c)
            oky || (δy = gy - r)
        end
        posx = c + δx
        posy = r + δy
        # Diameter = 4σ, averaged over the axes that fit; NaN if neither did.
        if okx && oky
            dia = 4.0 * (σx + σy) / 2
        elseif okx
            dia = 4.0 * σx
        elseif oky
            dia = 4.0 * σy
        else
            dia = NaN
        end
        push!(cx, posx); push!(cy, posy); push!(cint, I0 - bg); push!(cdia, dia)
    end

    # 4. Diameter filter (NaN passes — saturated particles are real).
    keep = [isnan(cdia[i]) || (params.min_diameter <= cdia[i] <= params.max_diameter)
            for i in eachindex(cdia)]
    cx = cx[keep]; cy = cy[keep]; cint = cint[keep]; cdia = cdia[keep]

    # 5. Deduplicate: brightest first, reject any within min_separation.
    order = sortperm(cint; rev = true)
    acc = dedupe(cx, cy, order, params.min_separation)
    return Particles(T.(cx[acc]), T.(cy[acc]), T.(cint[acc]), T.(cdia[acc]))
end

# 3-point log-Gaussian fit on one axis: samples (Jm, J0, Jp) at offsets −1/0/+1.
# Returns (offset, σ, ok). Fails (ok=false) on non-positive samples, a
# non-negative curvature, a non-finite result, or |offset| ≥ 1.
function fit_axis(Jm::Float64, J0::Float64, Jp::Float64)
    (Jm > 0 && J0 > 0 && Jp > 0) || return (0.0, 0.0, false)
    denom = log(Jm) + log(Jp) - 2 * log(J0)
    denom < 0 || return (0.0, 0.0, false)     # not a peak (needs concave-down)
    δ = (log(Jm) - log(Jp)) / (2 * denom)
    σ = sqrt(-1 / denom)
    (isfinite(δ) && isfinite(σ) && abs(δ) < 1) || return (0.0, 0.0, false)
    return (δ, σ, true)
end

# Intensity-weighted centroid of the 3×3 neighborhood around (r, c) on
# background-subtracted intensities (negatives clipped). Returns (x, y); falls
# back to the pixel center when the weight sum vanishes.
function centroid3(img::AbstractMatrix, r::Int, c::Int, bg::Float64)
    sw = 0.0; swx = 0.0; swy = 0.0
    for dc in -1:1, dr in -1:1
        w = max(Float64(img[r + dr, c + dc]) - bg, 0.0)
        sw += w
        swx += w * (c + dc)
        swy += w * (r + dr)
    end
    sw > 0 || return (Float64(c), Float64(r))
    return (swx / sw, swy / sw)
end

# Greedy dedupe: walk `order` (brightest first), accepting a point only when no
# already-accepted point lies within `min_sep`. Cell size = min_sep, so a 3×3
# block of cells covers the rejection disc.
function dedupe(x::Vector{Float64}, y::Vector{Float64}, order::Vector{Int}, min_sep::Real)
    s = Float64(min_sep)
    cells = Dict{Tuple{Int,Int},Vector{Int}}()
    accepted = Int[]
    s2 = s^2
    for idx in order
        qx, qy = x[idx], y[idx]
        bx, by = _cell(s, qx, qy)
        ok = true
        for jx in (bx - 1):(bx + 1), jy in (by - 1):(by + 1)
            bucket = get(cells, (jx, jy), nothing)
            bucket === nothing && continue
            for a in bucket
                if (x[a] - qx)^2 + (y[a] - qy)^2 < s2
                    ok = false
                    break
                end
            end
            ok || break
        end
        if ok
            push!(accepted, idx)
            push!(get!(() -> Int[], cells, (bx, by)), idx)
        end
    end
    return accepted
end
