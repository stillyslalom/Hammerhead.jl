"""
    calculate_peak_ratio(R, peakloc::Tuple{Int,Int}; exclusion_radius=2)

Ratio of the primary correlation peak at `peakloc` to the largest value outside
a square exclusion zone of `±exclusion_radius` pixels around it. Returns `Inf`
when no positive secondary peak exists. Higher values indicate a more reliable
displacement estimate.
"""
function calculate_peak_ratio(R::AbstractMatrix{T}, peakloc::Tuple{Int,Int};
                              exclusion_radius::Int = 2) where {T<:AbstractFloat}
    nr, nc = size(R)
    pr, pc = peakloc
    (1 <= pr <= nr && 1 <= pc <= nc) ||
        throw(ArgumentError("peak location $peakloc is outside the $nr×$nc matrix"))
    P1 = R[pr, pc]
    P2 = T(-Inf)
    @inbounds for j in 1:nc, i in 1:nr
        (abs(i - pr) <= exclusion_radius && abs(j - pc) <= exclusion_radius) && continue
        R[i, j] > P2 && (P2 = R[i, j])
    end
    P2 > 0 || return P1 > 0 ? T(Inf) : T(NaN)
    return P1 / P2
end

const CORR_MOMENT_EPSILON = 1e-9

"""
    calculate_correlation_moment(R, peakloc::Tuple{Real,Real}; neighborhood_size=3)

Square root of the second moment of the correlation values in an odd-sized
square neighborhood around the (subpixel) peak location `(row, col)` — a
weighted standard deviation of the peak, usable as an uncertainty proxy. Lower
values indicate a sharper peak. Returns `NaN` when the neighborhood is empty or
its correlation sum is non-positive.
"""
function calculate_correlation_moment(R::AbstractMatrix{<:Real}, peakloc::Tuple{<:Real,<:Real};
                                      neighborhood_size::Int = 3)
    (neighborhood_size >= 1 && isodd(neighborhood_size)) ||
        throw(ArgumentError("neighborhood_size must be a positive odd integer, got $neighborhood_size"))
    nr, nc = size(R)
    pr, pc = peakloc
    (isfinite(pr) && isfinite(pc)) || return NaN
    half = neighborhood_size ÷ 2
    r0 = round(Int, pr)
    c0 = round(Int, pc)
    rows = max(1, r0 - half):min(nr, r0 + half)
    cols = max(1, c0 - half):min(nc, c0 + half)
    (isempty(rows) || isempty(cols)) && return NaN

    sumC = 0.0
    sum_dx2 = 0.0
    sum_dy2 = 0.0
    @inbounds for c in cols, r in rows
        w = max(Float64(R[r, c]), 0.0)
        sumC += w
        sum_dx2 += (c - pc)^2 * w
        sum_dy2 += (r - pr)^2 * w
    end
    sumC < CORR_MOMENT_EPSILON && return NaN
    return sqrt((sum_dx2 + sum_dy2) / sumC)
end

"""
    replace_vectors!(u, v, invalid::AbstractMatrix{Bool})

Replace each vector flagged in `invalid` with the component-wise median of the
*valid* vectors in a square neighborhood, growing the neighborhood until at
least 3 valid neighbors are found. Replacements are computed from the original
field, so the result does not depend on traversal order. Vectors with no valid
neighbors anywhere are left unchanged.
"""
function replace_vectors!(u::AbstractMatrix, v::AbstractMatrix, invalid::AbstractMatrix{Bool})
    size(u) == size(v) == size(invalid) ||
        throw(ArgumentError("u, v, and invalid must have the same dimensions"))
    any(invalid) || return u, v
    nr, nc = size(u)
    uref = copy(u)
    vref = copy(v)
    bufu = Float64[]
    bufv = Float64[]
    max_radius = max(nr, nc) - 1
    for c in 1:nc, r in 1:nr
        invalid[r, c] || continue
        for radius in 1:max_radius
            empty!(bufu)
            empty!(bufv)
            for c2 in max(1, c - radius):min(nc, c + radius),
                r2 in max(1, r - radius):min(nr, r + radius)
                invalid[r2, c2] && continue
                push!(bufu, uref[r2, c2])
                push!(bufv, vref[r2, c2])
            end
            if length(bufu) >= 3
                u[r, c] = median(bufu)
                v[r, c] = median(bufv)
                break
            end
        end
    end
    return u, v
end

"""
    smooth_field(f::AbstractMatrix) -> Matrix{float(eltype(f))}

3×3 binomial (separable [1 2 1]/4) smoothing with replicated edges, preserving
the field's floating-point precision. Used to condition the predictor field
between interrogation passes.
"""
function smooth_field(f::AbstractMatrix{<:Real})
    T = float(eltype(f))
    quarter = T(1) / 4
    nr, nc = size(f)
    tmp = Matrix{T}(undef, nr, nc)
    out = Matrix{T}(undef, nr, nc)
    @inbounds for c in 1:nc, r in 1:nr
        tmp[r, c] = quarter * (f[r, max(c - 1, 1)] + 2f[r, c] + f[r, min(c + 1, nc)])
    end
    @inbounds for c in 1:nc, r in 1:nr
        out[r, c] = quarter * (tmp[max(r - 1, 1), c] + 2tmp[r, c] + tmp[min(r + 1, nr), c])
    end
    return out
end

"""
    smoothn(y; s = nothing, weights = nothing, robust = false) -> (; z, s)

Penalized least-squares smoothing of a gridded field (Garcia, CSDA 2010),
solved in the DCT domain. The smoothing parameter `s` is chosen by
generalized cross-validation when not given; larger values smooth more.
`weights` (same size, ≥ 0) express per-point confidence — e.g.
`.!(result.outliers .| result.mask)` — and non-finite entries of `y`
automatically get weight 0 and are filled from the smooth surface.
`robust = true` adds bisquare reweighting passes that resist outliers not
captured by the weights. Returns the smoothed field `z` and the `s` used.
"""
function smoothn(y::AbstractMatrix{<:Real};
                 s::Union{Nothing,Real} = nothing,
                 weights::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
                 robust::Bool = false, maxiter::Int = 100, tol::Real = 1e-3)
    s === nothing || s > 0 || throw(ArgumentError("s must be positive, got $s"))
    nr, nc = size(y)
    W0 = weights === nothing ? ones(nr, nc) : Float64.(weights)
    size(W0) == size(y) ||
        throw(ArgumentError("weights must have the same size as y"))
    all(>=(0), W0) || throw(ArgumentError("weights must be non-negative"))
    y0 = Float64.(y)
    finite = isfinite.(y0)
    W0[.!finite] .= 0
    nvalid = count(>(0), W0)
    nvalid > 0 || throw(ArgumentError("y has no valid (finite, weighted) data"))
    y0[.!finite] .= 0

    # DCT-II eigenvalues of the discrete Laplacian, per dimension.
    lam = [(-2 + 2 * cospi((i - 1) / nr)) + (-2 + 2 * cospi((j - 1) / nc))
           for i in 1:nr, j in 1:nc]
    lam2 = lam .^ 2
    n = length(y0)

    # GCV score for a candidate s given the current working DCT.
    function gcv(scand, xdc)
        gamma = 1 ./ (1 .+ scand .* lam2)
        zc = idct(gamma .* xdc)
        rss = 0.0
        for i in eachindex(y0)
            W0[i] > 0 || continue
            rss += W0[i] * (y0[i] - zc[i])^2
        end
        return (rss / nvalid) / (1 - sum(gamma) / n)^2
    end

    valid_mean = sum(y0[W0 .> 0]) / nvalid
    z = [W0[i] > 0 ? y0[i] : valid_mean for i in eachindex(y0)]
    z = reshape(z, nr, nc)
    W = copy(W0)
    s_used = s === nothing ? 1.0 : Float64(s)
    weighted = any(<(1), W)
    for pass in 1:(robust ? 4 : 1)
        for it in 1:maxiter
            xdc = dct(W .* (y0 .- z) .+ z)
            if s === nothing && it == 1
                # Coarse log-grid GCV search; the score is smooth and flat
                # near its minimum, so a refinement step buys little.
                best = Inf
                for ls in range(-6, 8; length = 43)
                    g = gcv(exp10(ls), xdc)
                    g < best && (best = g; s_used = exp10(ls))
                end
            end
            znew = idct(xdc ./ (1 .+ s_used .* lam2))
            rel = sqrt(sum(abs2, znew .- z) / max(sum(abs2, znew), eps()))
            z = znew
            (rel < tol || !weighted) && break
        end
        (robust && pass < 4) || break
        # Bisquare reweighting from the studentized residuals.
        r = y0 .- z
        resid = [r[i] for i in eachindex(r) if W0[i] > 0]
        sigma = 1.4826 * median!(abs.(resid))
        sigma = max(sigma, eps())
        W = [W0[i] * (abs(r[i]) < 4.685 * sigma ?
                      (1 - (r[i] / (4.685 * sigma))^2)^2 : 0.0)
             for i in eachindex(r)]
        W = reshape(W, nr, nc)
        weighted = true
    end
    return (; z, s = s_used)
end

"""
    universal_outlier_detection(u, v, threshold;
                                neighborhood_size=1, epsilon=0.1,
                                exclude=nothing) -> BitMatrix

Universal outlier detection (normalized median test, Westerweel & Scarano 2005)
on a 2D vector field. For each vector, the residual relative to the median of
its neighbors is normalized by the median absolute neighbor residual plus
`epsilon`; the vector is flagged when either component's normalized residual
exceeds `threshold`.

`neighborhood_size` is the number of neighbor layers (1 → 3×3, 2 → 5×5).
`epsilon` is the assumed measurement noise level in the same units as `u` and
`v` (typically ≈ 0.1 px for PIV); it keeps ordinary subpixel noise in smooth
regions from being flagged.

Cells marked `true` in `exclude` (e.g. masked windows) are never flagged and
never enter a neighbor median, so `NaN` entries there cannot poison the test.

Returns a `BitMatrix` where `true` marks an outlier.
"""
function universal_outlier_detection(u::AbstractMatrix{T}, v::AbstractMatrix{T},
                                     threshold::Real;
                                     neighborhood_size::Int = 1, epsilon::Real = 0.1,
                                     exclude::Union{Nothing,AbstractMatrix{Bool}} = nothing) where {T<:Real}
    size(u) == size(v) || throw(ArgumentError("u and v fields must have the same dimensions"))
    exclude === nothing || size(exclude) == size(u) ||
        throw(ArgumentError("exclude must have the same dimensions as u and v"))
    neighborhood_size >= 1 ||
        throw(ArgumentError("neighborhood_size must be at least 1, got $neighborhood_size"))
    epsilon > 0 || throw(ArgumentError("epsilon must be positive, got $epsilon"))

    nr, nc = size(u)
    is_outlier = falses(nr, nc)
    max_neighbors = (2neighborhood_size + 1)^2 - 1
    nbr_u = Vector{T}(undef, max_neighbors)
    nbr_v = Vector{T}(undef, max_neighbors)
    res_u = Vector{T}(undef, max_neighbors)
    res_v = Vector{T}(undef, max_neighbors)

    for c in 1:nc, r in 1:nr
        exclude !== nothing && exclude[r, c] && continue
        n = 0
        for nc2 in max(1, c - neighborhood_size):min(nc, c + neighborhood_size),
            nr2 in max(1, r - neighborhood_size):min(nr, r + neighborhood_size)
            (nr2 == r && nc2 == c) && continue
            exclude !== nothing && exclude[nr2, nc2] && continue
            n += 1
            nbr_u[n] = u[nr2, nc2]
            nbr_v[n] = v[nr2, nc2]
        end
        n == 0 && continue

        # Scratch buffers are ours; sort in place (median!) to avoid the copy
        # `median` makes. Sorting nbr_u/nbr_v reorders them, but the residuals
        # below are computed over all n entries — an order-independent multiset
        # — so the result is unchanged.
        u_med = median!(view(nbr_u, 1:n))
        v_med = median!(view(nbr_v, 1:n))
        for i in 1:n
            res_u[i] = abs(nbr_u[i] - u_med)
            res_v[i] = abs(nbr_v[i] - v_med)
        end
        u_norm = abs(u[r, c] - u_med) / (median!(view(res_u, 1:n)) + epsilon)
        v_norm = abs(v[r, c] - v_med) / (median!(view(res_v, 1:n)) + epsilon)
        if u_norm > threshold || v_norm > threshold
            is_outlier[r, c] = true
        end
    end
    return is_outlier
end

"""
    substitute_alternatives!(result, alt_u, alt_v, params) -> n_substituted

Peak substitution: for each vector flagged in `result.outliers` (masked
windows excluded), test its alternative peak displacements — `alt_u`/`alt_v`
are `(ny, nx, m)` arrays ordered by peak strength, `NaN` where absent —
against the valid neighbors using the UOD criterion (median ± threshold ×
(MAD + 0.1 px)). The first consistent alternative replaces the vector and
clears its outlier flag: it is measured data, just not the tallest peak.
Acceptance is judged against a snapshot of the field, so the result does not
depend on traversal order.
"""
function substitute_alternatives!(result::PIVResult, alt_u::AbstractArray{<:Real,3},
                                  alt_v::AbstractArray{<:Real,3}, params::PIVParameters)
    any(result.outliers) || return 0
    nr, nc = size(result.u)
    er = params.uod_neighborhood
    thr = params.uod_threshold
    uref = copy(result.u)
    vref = copy(result.v)
    valid = .!(result.outliers .| result.mask)
    bufu = Float64[]
    bufv = Float64[]
    nsub = 0
    for c in 1:nc, r in 1:nr
        (result.outliers[r, c] && !result.mask[r, c]) || continue
        empty!(bufu)
        empty!(bufv)
        for c2 in max(1, c - er):min(nc, c + er), r2 in max(1, r - er):min(nr, r + er)
            (r2 == r && c2 == c) && continue
            valid[r2, c2] || continue
            push!(bufu, uref[r2, c2])
            push!(bufv, vref[r2, c2])
        end
        length(bufu) >= 3 || continue
        med_u = median(bufu)
        med_v = median(bufv)
        bufu .= abs.(bufu .- med_u)   # reuse buffers for the MADs
        bufv .= abs.(bufv .- med_v)
        mad_u = median!(bufu)
        mad_v = median!(bufv)
        for m in axes(alt_u, 3)
            au = alt_u[r, c, m]
            av = alt_v[r, c, m]
            isfinite(au) && isfinite(av) || break   # peaks are stored in order
            if abs(au - med_u) / (mad_u + 0.1) <= thr &&
               abs(av - med_v) / (mad_v + 0.1) <= thr
                result.u[r, c] = au
                result.v[r, c] = av
                result.outliers[r, c] = false
                nsub += 1
                break
            end
        end
    end
    return nsub
end

"""
    PIVValidator

Abstract supertype for vector validation criteria. A validator is applied with
[`apply_validator!`](@ref), which ORs the vectors it rejects into
`result.outliers`; [`validate_vectors!`](@ref) applies a whole pipeline.
"""
abstract type PIVValidator end

"""
    LocalValidator <: PIVValidator

Validators that judge each vector from its own properties alone.
"""
abstract type LocalValidator <: PIVValidator end

"""
    NeighborhoodValidator <: PIVValidator

Validators that judge each vector relative to its spatial neighbors.
"""
abstract type NeighborhoodValidator <: PIVValidator end

"""
    PeakRatioValidator(threshold)

Flag vectors whose correlation peak ratio falls below `threshold` (`NaN`
ratios are flagged too). Pair spec: `:peak_ratio => threshold`. See the
[validation how-to](../howto/validation.md) for the full list of validators.
"""
struct PeakRatioValidator <: LocalValidator
    threshold::Float64
end

"""
    CorrelationMomentValidator(threshold)

Flag vectors whose correlation moment exceeds `threshold`, i.e. whose
correlation peak is too broad (`NaN` moments are flagged too). Pair spec:
`:correlation_moment => threshold`. See the
[validation how-to](../howto/validation.md) for the full list of validators.
"""
struct CorrelationMomentValidator <: LocalValidator
    threshold::Float64
end

"""
    VelocityMagnitudeValidator(min, max)

Flag vectors whose displacement magnitude lies outside `[min, max]` pixels
(`NaN` magnitudes are flagged too). Pair spec:
`:velocity_magnitude => (min = 0, max = 50)`; `min` defaults to 0. See the
[validation how-to](../howto/validation.md) for the full list of validators.
"""
struct VelocityMagnitudeValidator <: LocalValidator
    min::Float64
    max::Float64
end

"""
    UniversalOutlierValidator(threshold; neighborhood_size = 2, epsilon = 0.1)

Flag vectors that fail the normalized median test — see
[`universal_outlier_detection`](@ref) for the parameters. Pair spec:
`:uod => (threshold = 2.0, neighborhood_size = 2, epsilon = 0.1)` (only
`threshold` is required; `:universal_outlier` is an alias). See the
[validation how-to](../howto/validation.md) for the full list of validators.
"""
struct UniversalOutlierValidator <: NeighborhoodValidator
    threshold::Float64
    neighborhood_size::Int
    epsilon::Float64
end

UniversalOutlierValidator(threshold::Real; neighborhood_size::Int = 2, epsilon::Real = 0.1) =
    UniversalOutlierValidator(Float64(threshold), neighborhood_size, Float64(epsilon))

"""
    parse_validator(spec) -> PIVValidator

Normalize a validator specification: `PIVValidator` objects pass through, and
`Symbol => value` pairs construct the corresponding validator (see the
validator docstrings for the accepted pair forms).
"""
parse_validator(v::PIVValidator) = v

function parse_validator(spec::Pair{Symbol, <:Real})
    name, threshold = spec
    name === :peak_ratio && return PeakRatioValidator(threshold)
    name === :correlation_moment && return CorrelationMomentValidator(threshold)
    throw(ArgumentError("unknown validator :$name (expected :peak_ratio or :correlation_moment)"))
end

function parse_validator(spec::Pair{Symbol, <:NamedTuple})
    name, config = spec
    if name === :uod || name === :universal_outlier
        return UniversalOutlierValidator(config.threshold;
            neighborhood_size = get(config, :neighborhood_size, 2),
            epsilon = get(config, :epsilon, 0.1))
    elseif name === :velocity_magnitude
        return VelocityMagnitudeValidator(get(config, :min, 0.0), config.max)
    end
    throw(ArgumentError("unknown validator :$name (expected :uod or :velocity_magnitude)"))
end

parse_validator(spec) = throw(ArgumentError("cannot interpret $spec as a validator"))

"""
    validate_vectors!(result::PIVResult, pipeline) -> PIVResult

Apply a validation pipeline — a single validator or specification, or a tuple
of them (see [`parse_validator`](@ref)) — accumulating rejected vectors into
`result.outliers`.
"""
function validate_vectors!(result::PIVResult, pipeline)
    for spec in (pipeline isa Tuple ? pipeline : (pipeline,))
        apply_validator!(result, parse_validator(spec))
    end
    return result
end

"""
    apply_validator!(result::PIVResult, v::PIVValidator) -> PIVResult

Apply a single validator, ORing the vectors it rejects into `result.outliers`.
"""
function apply_validator!(result::PIVResult, v::PeakRatioValidator)
    # NaN ratios fail the comparison and are flagged.
    @. result.outliers |= !(result.peak_ratio >= v.threshold)
    return result
end

function apply_validator!(result::PIVResult, v::CorrelationMomentValidator)
    @. result.outliers |= !(result.correlation_moment <= v.threshold)
    return result
end

function apply_validator!(result::PIVResult, v::VelocityMagnitudeValidator)
    @inbounds for i in eachindex(result.u)
        mag = hypot(result.u[i], result.v[i])
        (v.min <= mag <= v.max) || (result.outliers[i] = true)
    end
    return result
end

function apply_validator!(result::PIVResult, v::UniversalOutlierValidator)
    result.outliers .|= universal_outlier_detection(result.u, result.v, v.threshold;
        neighborhood_size = v.neighborhood_size, epsilon = v.epsilon,
        exclude = any(result.mask) ? result.mask : nothing)
    return result
end
