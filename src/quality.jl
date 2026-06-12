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
    smooth_field(f::AbstractMatrix) -> Matrix{Float64}

3×3 binomial (separable [1 2 1]/4) smoothing with replicated edges. Used to
condition the predictor field between interrogation passes.
"""
function smooth_field(f::AbstractMatrix{<:Real})
    nr, nc = size(f)
    tmp = Matrix{Float64}(undef, nr, nc)
    out = Matrix{Float64}(undef, nr, nc)
    @inbounds for c in 1:nc, r in 1:nr
        tmp[r, c] = 0.25 * (f[r, max(c - 1, 1)] + 2f[r, c] + f[r, min(c + 1, nc)])
    end
    @inbounds for c in 1:nc, r in 1:nr
        out[r, c] = 0.25 * (tmp[max(r - 1, 1), c] + 2tmp[r, c] + tmp[min(r + 1, nr), c])
    end
    return out
end

"""
    universal_outlier_detection(u, v, threshold;
                                neighborhood_size=1, epsilon=0.1) -> BitMatrix

Universal outlier detection (normalized median test, Westerweel & Scarano 2005)
on a 2D vector field. For each vector, the residual relative to the median of
its neighbors is normalized by the median absolute neighbor residual plus
`epsilon`; the vector is flagged when either component's normalized residual
exceeds `threshold`.

`neighborhood_size` is the number of neighbor layers (1 → 3×3, 2 → 5×5).
`epsilon` is the assumed measurement noise level in the same units as `u` and
`v` (typically ≈ 0.1 px for PIV); it keeps ordinary subpixel noise in smooth
regions from being flagged.

Returns a `BitMatrix` where `true` marks an outlier.
"""
function universal_outlier_detection(u::AbstractMatrix{T}, v::AbstractMatrix{T},
                                     threshold::Real;
                                     neighborhood_size::Int = 1, epsilon::Real = 0.1) where {T<:Real}
    size(u) == size(v) || throw(ArgumentError("u and v fields must have the same dimensions"))
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
        n = 0
        for nc2 in max(1, c - neighborhood_size):min(nc, c + neighborhood_size),
            nr2 in max(1, r - neighborhood_size):min(nr, r + neighborhood_size)
            (nr2 == r && nc2 == c) && continue
            n += 1
            nbr_u[n] = u[nr2, nc2]
            nbr_v[n] = v[nr2, nc2]
        end
        n == 0 && continue

        u_med = median(view(nbr_u, 1:n))
        v_med = median(view(nbr_v, 1:n))
        for i in 1:n
            res_u[i] = abs(nbr_u[i] - u_med)
            res_v[i] = abs(nbr_v[i] - v_med)
        end
        u_norm = abs(u[r, c] - u_med) / (median(view(res_u, 1:n)) + epsilon)
        v_norm = abs(v[r, c] - v_med) / (median(view(res_v, 1:n)) + epsilon)
        if u_norm > threshold || v_norm > threshold
            is_outlier[r, c] = true
        end
    end
    return is_outlier
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
ratios are flagged too). Pair spec: `:peak_ratio => threshold`.
"""
struct PeakRatioValidator <: LocalValidator
    threshold::Float64
end

"""
    CorrelationMomentValidator(threshold)

Flag vectors whose correlation moment exceeds `threshold`, i.e. whose
correlation peak is too broad (`NaN` moments are flagged too). Pair spec:
`:correlation_moment => threshold`.
"""
struct CorrelationMomentValidator <: LocalValidator
    threshold::Float64
end

"""
    VelocityMagnitudeValidator(min, max)

Flag vectors whose displacement magnitude lies outside `[min, max]` pixels
(`NaN` magnitudes are flagged too). Pair spec:
`:velocity_magnitude => (min = 0, max = 50)`; `min` defaults to 0.
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
`threshold` is required; `:universal_outlier` is an alias).
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
        neighborhood_size = v.neighborhood_size, epsilon = v.epsilon)
    return result
end
