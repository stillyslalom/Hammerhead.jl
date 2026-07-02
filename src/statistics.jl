# Time-series statistics and temporal validation over sequences of PIVResults
# sharing one interrogation grid (e.g. from run_piv_sequence).

function check_same_grid(results::AbstractVector{<:PIVResult})
    isempty(results) && throw(ArgumentError("results must not be empty"))
    r1 = results[1]
    for r in results
        (r.x == r1.x && r.y == r1.y) ||
            throw(ArgumentError("all results must share the same interrogation grid"))
    end
    return r1
end

# A sample enters the statistics when it is finite, unmasked, and (unless
# invalid vectors are requested) not an outlier.
sample_valid(r::PIVResult, i, include_invalid::Bool) =
    isfinite(r.u[i]) && isfinite(r.v[i]) && !r.mask[i] &&
    (include_invalid || !r.outliers[i])

"""
    field_statistics(results; include_invalid = false) -> NamedTuple

Pointwise temporal statistics over a sequence of same-grid `PIVResult`s:
returns `(x, y, mean_u, mean_v, rms_u, rms_v, reynolds_uv, count)`, where
`rms_u`/`rms_v` are the RMS of the fluctuating components and `reynolds_uv`
is the Reynolds shear-stress correlation `mean(u'v')` (denominator `n`, the
per-point valid-sample count in `count`). Outlier-flagged vectors, masked
windows, and non-finite entries are excluded unless `include_invalid`; points
with no valid samples are `NaN`.
"""
function field_statistics(results::AbstractVector{<:PIVResult};
                          include_invalid::Bool = false)
    r1 = check_same_grid(results)
    ny, nx = size(r1.u)
    su = zeros(ny, nx)
    sv = zeros(ny, nx)
    suu = zeros(ny, nx)
    svv = zeros(ny, nx)
    suv = zeros(ny, nx)
    count = zeros(Int, ny, nx)
    for r in results
        size(r.u) == (ny, nx) ||
            throw(ArgumentError("all results must share the same interrogation grid"))
        for i in eachindex(r.u)
            sample_valid(r, i, include_invalid) || continue
            ui, vi = Float64(r.u[i]), Float64(r.v[i])
            su[i] += ui
            sv[i] += vi
            suu[i] += ui^2
            svv[i] += vi^2
            suv[i] += ui * vi
            count[i] += 1
        end
    end
    stat(f) = [count[i] > 0 ? f(i) : NaN for i in eachindex(count)]
    mean_u = reshape(stat(i -> su[i] / count[i]), ny, nx)
    mean_v = reshape(stat(i -> sv[i] / count[i]), ny, nx)
    rms_u = reshape(stat(i -> sqrt(max(suu[i] / count[i] - (su[i] / count[i])^2, 0.0))), ny, nx)
    rms_v = reshape(stat(i -> sqrt(max(svv[i] / count[i] - (sv[i] / count[i])^2, 0.0))), ny, nx)
    reynolds_uv = reshape(stat(i -> suv[i] / count[i] - su[i] * sv[i] / count[i]^2), ny, nx)
    return (; x = r1.x, y = r1.y, mean_u, mean_v, rms_u, rms_v, reynolds_uv, count)
end

"""
    validate_temporal!(results; threshold = 3, epsilon = 0.1) -> results

Temporal normalized-median test across a sequence of same-grid `PIVResult`s:
at each grid point the median and median absolute deviation (MAD) of the
valid samples over time form a robust reference, and samples with
`|component − median| / (MAD + epsilon) > threshold` in either component are
flagged into their result's `outliers`. Points with fewer than 3 valid
samples, masked windows, and already-flagged vectors are left untouched.
Complements the spatial UOD: a vector consistent with its spatial neighbors
can still be exposed as an outlier by the point's time history.
"""
function validate_temporal!(results::AbstractVector{<:PIVResult};
                            threshold::Real = 3, epsilon::Real = 0.1)
    r1 = check_same_grid(results)
    threshold > 0 || throw(ArgumentError("threshold must be positive, got $threshold"))
    epsilon > 0 || throw(ArgumentError("epsilon must be positive, got $epsilon"))
    bufu = Float64[]
    bufv = Float64[]
    for i in eachindex(r1.u)
        empty!(bufu)
        empty!(bufv)
        for r in results
            sample_valid(r, i, false) || continue
            push!(bufu, Float64(r.u[i]))
            push!(bufv, Float64(r.v[i]))
        end
        length(bufu) >= 3 || continue
        med_u = median(bufu)
        med_v = median(bufv)
        mad_u = median(abs.(bufu .- med_u))
        mad_v = median(abs.(bufv .- med_v))
        for r in results
            sample_valid(r, i, false) || continue
            if abs(r.u[i] - med_u) / (mad_u + epsilon) > threshold ||
               abs(r.v[i] - med_v) / (mad_v + epsilon) > threshold
                r.outliers[i] = true
            end
        end
    end
    return results
end

"""
    peak_locking(displacements; nbins = 21) -> (fractions, counts, index)

Diagnose peak locking from the fractional parts `f = x − round(x) ∈
[−0.5, 0.5)` of a displacement sample (any array; non-finite entries are
skipped). Returns the histogram bin centers and counts, plus a locking
index comparing the sample density near integer displacements (`|f| ≤ 0.1`)
with the density near half-integers (`|f| ≥ 0.4`): 0 for a uniform (locking
free) distribution, → 1 as fractions pile up on integers, negative if they
avoid them. `NaN` when there are no valid samples.
"""
function peak_locking(displacements::AbstractArray{<:Real}; nbins::Int = 21)
    nbins >= 3 || throw(ArgumentError("nbins must be at least 3, got $nbins"))
    counts = zeros(Int, nbins)
    n_center = 0
    n_edge = 0
    nvalid = 0
    for x in displacements
        isfinite(x) || continue
        f = x - round(x)
        f >= 0.5 && (f -= 1.0)   # guard the round-half-even edge
        counts[clamp(1 + floor(Int, (f + 0.5) * nbins), 1, nbins)] += 1
        abs(f) <= 0.1 && (n_center += 1)
        abs(f) >= 0.4 && (n_edge += 1)
        nvalid += 1
    end
    fractions = [-0.5 + (b - 0.5) / nbins for b in 1:nbins]
    # Both bands cover a 0.2 width, so their raw counts are comparable.
    index = (n_center + n_edge) > 0 ?
            (n_center - n_edge) / (n_center + n_edge) : NaN
    nvalid == 0 && (index = NaN)
    return (; fractions, counts, index)
end

"""
    power_spectrum(signal; dt = 1.0, window = :hann) -> (frequencies, psd)

One-sided power spectral density of a uniformly sampled series (sampling
interval `dt`): the mean is removed, the taper applied (`:hann` or `:none`,
power-normalized), and `sum(psd) * Δf` recovers the signal variance.
Frequencies are in cycles per unit of `dt`. Extract a per-point velocity
time series from a sequence with e.g. `[r.u[i, j] for r in results]`.
"""
function power_spectrum(signal::AbstractVector{<:Real};
                        dt::Real = 1.0, window::Symbol = :hann)
    n = length(signal)
    n >= 2 || throw(ArgumentError("signal must have at least 2 samples, got $n"))
    dt > 0 || throw(ArgumentError("dt must be positive, got $dt"))
    x = Float64.(signal)
    x .-= sum(x) / n
    if window === :hann
        x .*= [0.5 * (1 - cospi(2 * (k - 1) / (n - 1))) for k in 1:n]
        wnorm = sum(abs2, 0.5 * (1 - cospi(2 * (k - 1) / (n - 1))) for k in 1:n)
    elseif window === :none
        wnorm = Float64(n)
    else
        throw(ArgumentError("window must be :hann or :none, got :$window"))
    end
    X = rfft(x)
    psd = abs2.(X) .* (2 * dt / wnorm)
    psd[1] /= 2                    # DC is not doubled
    iseven(n) && (psd[end] /= 2)   # nor is Nyquist
    return (frequencies = collect(rfftfreq(n, 1 / dt)), psd = psd)
end
