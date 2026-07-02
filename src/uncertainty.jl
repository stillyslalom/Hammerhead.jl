# Per-vector uncertainty quantification from correlation statistics
# (Wieneke 2015, Meas. Sci. Technol. 26 074002). Works directly on the two
# deformed interrogation windows at convergence: each pixel contributes
# ΔC_i = A(x)B(x+1) − A(x+1)B(x) (eq 5) to the correlation-peak asymmetry
# ΔC = C(+1) − C(−1), which is ~zero once the multi-pass scheme has converged
# (eq 3). The ΔC_i add up in a random-walk fashion, but neighboring pixels are
# correlated over roughly a particle diameter, so the variance of ΔC is the
# sum of spatial covariance sums S_{δ} of the ΔC_i field (eqs 6–7), not the
# independence limit Σ ΔC_i² (eq 8). σ_ΔC is then converted to a displacement
# uncertainty with the same 3-point Gaussian fit an actual asymmetry would
# produce (eqs 4, 9): σ_u = f(C0, C± − σ_ΔC/2, C± + σ_ΔC/2), where
# C± = (C₊ + C₋)/2. The linearization is accurate for σ_u ≲ 0.3 px; the
# estimator captures the random error only (bias such as peak locking is
# invisible to it).

const UQ_MAX_OFFSET = 4

# Covariance offsets (δrow, δcol) covering one half-plane, origin excluded,
# ordered by square ring max(|δr|, |δc|) = 1, 2, 3, 4. The raw covariance
# sums satisfy S(−δ) = S(δ) exactly (identical set of pixel pairs), so the
# full ±4 square of eq (7) is S(0,0) + 2·Σ over this list, and the half-plane
# maximum of a ring equals its full-plane maximum.
const UQ_OFFSETS = let m = UQ_MAX_OFFSET
    offs = [(dr, dc) for dc in 0:m for dr in (dc == 0 ? (1:m) : (-m:m))]
    sort!(offs; by = o -> max(abs(o[1]), abs(o[2])))
end
const UQ_RINGS = [findall(o -> max(abs(o[1]), abs(o[2])) == r, UQ_OFFSETS)
                  for r in 1:UQ_MAX_OFFSET]

# Per-window statistics layout, one row per displacement component (1 = u,
# 2 = v): [C0, C+, C−, S(0,0), S(UQ_OFFSETS)...]. Accumulated in Float64 —
# a deliberate CPU-side island like the correlation-moment accumulation —
# and converted to the pipeline precision on store. The statistics are
# additive across image pairs, which is what the ensemble path pools.
const UQ_NSTATS = 4 + length(UQ_OFFSETS)

new_uncertainty_stats(njobs::Integer) = [zeros(2, UQ_NSTATS) for _ in 1:njobs]

# Window-sized scratch for one task/chunk: mean-subtracted apodized copies of
# the two windows plus the ΔC_i field and its smoothed version (square at the
# larger window dimension so the transposed v-component pass fits too).
function uncertainty_scratch(::Type{T}, wsize::Dims{2}) where {T}
    m = max(wsize...)
    return (wA = Matrix{T}(undef, wsize), wB = Matrix{T}(undef, wsize),
            dC = Matrix{T}(undef, m, m), dCs = Matrix{T}(undef, m, m))
end

# Mirror of load_windows!: windows are mean-subtracted over their valid
# pixels and apodized, masked pixels enter at zero — the uncertainty analysis
# sees exactly the signal the correlator saw.
function load_uncertainty_windows!(wA::AbstractMatrix{T}, wB::AbstractMatrix{T},
                                   subA::AbstractMatrix, subB::AbstractMatrix,
                                   submask::Union{Nothing,AbstractMatrix{Bool}},
                                   apod::AbstractMatrix) where {T}
    wr, wc = size(wA)
    local meanA::T, meanB::T
    if submask === nothing
        meanA = T(sum(subA) / length(subA))
        meanB = T(sum(subB) / length(subB))
    else
        sA = zero(T)
        sB = zero(T)
        n = 0
        @inbounds for j in 1:wc, i in 1:wr
            submask[i, j] && continue
            sA += T(subA[i, j])
            sB += T(subB[i, j])
            n += 1
        end
        meanA = n > 0 ? sA / n : zero(T)
        meanB = n > 0 ? sB / n : zero(T)
    end
    @inbounds for j in 1:wc, i in 1:wr
        if submask !== nothing && submask[i, j]
            wA[i, j] = zero(T)
            wB[i, j] = zero(T)
        else
            wA[i, j] = apod[i, j] * (T(subA[i, j]) - meanA)
            wB[i, j] = apod[i, j] * (T(subB[i, j]) - meanB)
        end
    end
    return nothing
end

# Add one image pair's correlation-difference statistics for both
# displacement components to `stats` (2 × UQ_NSTATS).
function accumulate_uncertainty!(stats::Matrix{Float64}, scratch,
                                 subA::AbstractMatrix, subB::AbstractMatrix,
                                 submask::Union{Nothing,AbstractMatrix{Bool}},
                                 apod::AbstractMatrix)
    load_uncertainty_windows!(scratch.wA, scratch.wB, subA, subB, submask, apod)
    uq_component!(view(stats, 1, :), scratch.dC, scratch.dCs, scratch.wA, scratch.wB)
    uq_component!(view(stats, 2, :), scratch.dC, scratch.dCs,
                  transpose(scratch.wA), transpose(scratch.wB))
    return stats
end

# Statistics of one displacement component, with the shift direction along
# the columns of A/B (the v component passes transposed views). Fills the
# leading (nr, nc−1) block of dC with ΔC_i (eq 5) and accumulates into `s`:
# C0/C+/C− over the same support, then the covariance sums S_{δ} (eq 7) of
# the zero-meaned ΔC_i field after (1,2,1)/4 smoothing along the shift
# direction (§2.1 — eliminates the negative pixel-noise covariance at δ = ±1).
function uq_component!(s::AbstractVector{Float64}, dC, dCs, A, B)
    nr, nc = size(A)
    m = nc - 1
    T = eltype(dC)
    C0 = 0.0
    Cp = 0.0
    Cm = 0.0
    @inbounds for c in 1:m, r in 1:nr
        a0 = T(A[r, c])
        a1 = T(A[r, c + 1])
        b0 = T(B[r, c])
        b1 = T(B[r, c + 1])
        C0 += Float64(a0 * b0)
        Cp += Float64(a0 * b1)
        Cm += Float64(a1 * b0)
        dC[r, c] = a0 * b1 - a1 * b0
    end
    s[1] += C0
    s[2] += Cp
    s[3] += Cm
    quarter = T(1) / 4
    @inbounds for c in 1:m, r in 1:nr
        cl = max(c - 1, 1)
        cr = min(c + 1, m)
        dCs[r, c] = quarter * (dC[r, cl] + 2 * dC[r, c] + dC[r, cr])
    end
    # Eq (6) requires the ΔC_i to have zero mean; subtract the window mean.
    μ = 0.0
    @inbounds for c in 1:m, r in 1:nr
        μ += Float64(dCs[r, c])
    end
    μ /= nr * m
    S00 = 0.0
    @inbounds for c in 1:m, r in 1:nr
        d = Float64(dCs[r, c]) - μ
        S00 += d * d
    end
    s[4] += S00
    for (k, (δr, δc)) in enumerate(UQ_OFFSETS)
        Sk = 0.0
        @inbounds for c in max(1, 1 - δc):min(m, m - δc),
                      r in max(1, 1 - δr):min(nr, nr - δr)
            Sk += (Float64(dCs[r, c]) - μ) * (Float64(dCs[r + δr, c + δc]) - μ)
        end
        s[4 + k] += Sk
    end
    return nothing
end

# Convert accumulated statistics into σ for one component: total variance
# from the covariance sums, summed ring by ring outward until the ring's
# maximum drops below 0.05·S00 ("only the inner values are summed up until
# S/S0,0 drops below 0.05", §2.1). Inner rings are taken whole — their
# negative members are real covariance (signal×noise anticorrelation) that
# largely cancels against S00; truncation only guards against outer terms
# that are pure sampling noise. Then eq (9). NaN when the window carries no
# usable correlation signal or the noise exceeds the peak curvature (beyond
# the ~0.3 px validity of eq 9).
function finalize_uncertainty(::Type{T}, s::AbstractVector{Float64}) where {T}
    C0, Cp, Cm, S00 = s[1], s[2], s[3], s[4]
    σ2 = S00
    for ring in UQ_RINGS
        maximum(k -> s[4 + k], ring) < 0.05 * S00 && break
        for k in ring
            σ2 += 2 * s[4 + k]
        end
    end
    σΔC = sqrt(max(σ2, 0.0))
    Cpm = (Cp + Cm) / 2
    lo = Cpm - σΔC / 2
    hi = Cpm + σΔC / 2
    (C0 > 0 && lo > 0) || return T(NaN)
    denom = 2 * log(C0) - log(hi) - log(lo)
    denom > 0 || return T(NaN)
    return T((log(hi) - log(lo)) / (2 * denom))
end
