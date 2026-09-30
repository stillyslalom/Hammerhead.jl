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

function check_same_grid(results::AbstractVector{<:StereoPIVResult})
    isempty(results) && throw(ArgumentError("results must not be empty"))
    r1 = results[1]
    for r in results
        (r.x == r1.x && r.y == r1.y && r.z == r1.z) ||
            throw(ArgumentError("all results must share the same stereo interrogation grid"))
    end
    return r1
end

# A sample enters the statistics when it is finite, unmasked, and (unless
# invalid vectors are requested) not an outlier.
sample_valid(r::PIVResult, i, include_invalid::Bool) =
    isfinite(r.u[i]) && isfinite(r.v[i]) && !r.mask[i] &&
    (include_invalid || !r.outliers[i])

sample_valid(r::StereoPIVResult, i, include_invalid::Bool) =
    isfinite(r.u[i]) && isfinite(r.v[i]) && isfinite(r.w[i]) && !r.mask[i] &&
    (include_invalid || !r.outliers[i])

"""
    field_statistics(results; include_invalid = false) -> NamedTuple

Calculate pointwise statistics from a nonempty sequence of planar results
on the same grid. Returns `(x, y, mean_u, mean_v, rms_u, rms_v, reynolds_uv,
count)`. RMS describes fluctuations about the local mean; `reynolds_uv`
is `mean(u'v')`. Moments divide by the valid-sample count, without an
`n - 1` correction.

A sample contributes only when both components are finite and the cell is
unmasked. Flagged vectors are also excluded unless `include_invalid=true`.
Cells with no samples have `count == 0` and `NaN` statistics.

The stored arrays are used directly. To obtain velocity statistics, convert
each result with [`physical`](@ref) first and use consistent units across the
sequence. Fluctuation RMS includes measurement noise; it is not an uncertainty
estimate for the mean.
"""
function field_statistics(results::AbstractVector{<:PIVResult};
                          include_invalid::Bool = false)
    r1 = check_same_grid(results)
    ny, nx = size(r1.u)
    mu, mv, m2u, m2v, c2uv = (zeros(ny, nx) for _ in 1:5)
    count = zeros(Int, ny, nx)
    for r in results
        size(r.u) == (ny, nx) ||
            throw(ArgumentError("all results must share the same interrogation grid"))
        for i in eachindex(r.u)
            sample_valid(r, i, include_invalid) || continue
            ui, vi = Float64(r.u[i]), Float64(r.v[i])
            count[i] += 1
            n = count[i]
            du, dv = ui-mu[i], vi-mv[i]
            mu[i] += du/n; mv[i] += dv/n
            m2u[i] += du*(ui-mu[i]); m2v[i] += dv*(vi-mv[i])
            c2uv[i] += du*(vi-mv[i])
        end
    end
    stat(f) = [count[i] > 0 ? f(i) : NaN for i in eachindex(count)]
    mean_u = reshape(stat(i -> mu[i]), ny, nx)
    mean_v = reshape(stat(i -> mv[i]), ny, nx)
    rms_u = reshape(stat(i -> sqrt(max(m2u[i] / count[i], 0.0))), ny, nx)
    rms_v = reshape(stat(i -> sqrt(max(m2v[i] / count[i], 0.0))), ny, nx)
    reynolds_uv = reshape(stat(i -> c2uv[i] / count[i]), ny, nx)
    return (; x = r1.x, y = r1.y, mean_u, mean_v, rms_u, rms_v, reynolds_uv, count)
end


"""
    field_statistics(results::AbstractVector{<:StereoPIVResult};
                     include_invalid = false) -> NamedTuple

Calculate pointwise statistics from a nonempty, same-grid stereo sequence.
Returns coordinates, means and fluctuation RMS values for `u`, `v`, and `w`,
six Reynolds-stress terms (`reynolds_uu`, `reynolds_vv`, `reynolds_ww`,
`reynolds_uv`, `reynolds_uw`, `reynolds_vw`), and valid-sample `count`.
Normal stresses equal the corresponding RMS squared. Moments divide by the
valid-sample count, without an `n - 1` correction.

A sample contributes only when all three components are finite and the node
is unmasked. Flagged vectors are also excluded unless `include_invalid=true`.
Cells with no samples have `count == 0` and `NaN` statistics. Coordinates,
including `z`, must match across results. Values use the stored arrays' units;
attached scales are not applied automatically.
"""
function field_statistics(results::AbstractVector{<:StereoPIVResult};
                          include_invalid::Bool = false)
    r1 = check_same_grid(results)
    ny, nx = size(r1.u)
    moments = [zeros(ny, nx) for _ in 1:9]
    mu, mv, mw, m2u, m2v, m2w, c2uv, c2uw, c2vw = moments
    count = zeros(Int, ny, nx)
    for r in results
        size(r.u) == (ny, nx) ||
            throw(ArgumentError("all results must share the same stereo interrogation grid"))
        for i in eachindex(r.u)
            sample_valid(r, i, include_invalid) || continue
            ui, vi, wi = Float64(r.u[i]), Float64(r.v[i]), Float64(r.w[i])
            count[i] += 1
            n = count[i]
            du, dv, dw = ui-mu[i], vi-mv[i], wi-mw[i]
            mu[i] += du/n; mv[i] += dv/n; mw[i] += dw/n
            m2u[i] += du*(ui-mu[i]); m2v[i] += dv*(vi-mv[i]); m2w[i] += dw*(wi-mw[i])
            c2uv[i] += du*(vi-mv[i]); c2uw[i] += du*(wi-mw[i]); c2vw[i] += dv*(wi-mw[i])
        end
    end
    value(i, f) = count[i] > 0 ? f(count[i]) : NaN
    field(f) = reshape([value(i, n -> f(i, n)) for i in eachindex(count)], ny, nx)
    mean_u = field((i, n) -> mu[i])
    mean_v = field((i, n) -> mv[i])
    mean_w = field((i, n) -> mw[i])
    reynolds_uu = field((i, n) -> max(m2u[i] / n, 0.0))
    reynolds_vv = field((i, n) -> max(m2v[i] / n, 0.0))
    reynolds_ww = field((i, n) -> max(m2w[i] / n, 0.0))
    reynolds_uv = field((i, n) -> c2uv[i] / n)
    reynolds_uw = field((i, n) -> c2uw[i] / n)
    reynolds_vw = field((i, n) -> c2vw[i] / n)
    rms_u, rms_v, rms_w = sqrt.(reynolds_uu), sqrt.(reynolds_vv), sqrt.(reynolds_ww)
    return (; x = r1.x, y = r1.y, z = r1.z, mean_u, mean_v, mean_w,
            rms_u, rms_v, rms_w, reynolds_uu, reynolds_vv, reynolds_ww,
            reynolds_uv, reynolds_uw, reynolds_vw, count)
end

"""
    validate_temporal!(results; threshold = 3, epsilon = 0.1) -> results

Temporal normalized-median test across a sequence of same-grid `PIVResult`s:
at each grid point the median and median absolute deviation (MAD) of the
valid samples over time form a robust reference, and samples with
`|component − median| / (MAD + epsilon) > threshold` in either component are
marked in each result's `outliers`. Displacement arrays are not changed and
existing flags are not cleared. Masked, non-finite, and already-flagged samples
are excluded; points with fewer than three remaining samples are untouched.
`epsilon` is a positive noise floor in the stored components' units;
`threshold` is positive and dimensionless. Scaling metadata is not applied.

The reference spans the whole supplied sequence, rather than a moving time
window. Check newly flagged values against the time history: a real transient
can also differ from the sequence median.
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
    validate_temporal!(results::AbstractVector{<:StereoPIVResult};
                       threshold = 3, epsilon = 0.1) -> results

Stereo form of the temporal normalized-median test. The robust reference is
built independently for all three components and a sample is flagged when
any component exceeds `threshold`. Masked, non-finite, already flagged, and
short (fewer than three valid samples) histories are unchanged. Only the
`outliers` arrays are mutated. `epsilon` uses the stored components' units,
which are calibration length units for unconverted stereo results.
"""
function validate_temporal!(results::AbstractVector{<:StereoPIVResult};
                            threshold::Real = 3, epsilon::Real = 0.1)
    r1 = check_same_grid(results)
    threshold > 0 || throw(ArgumentError("threshold must be positive, got $threshold"))
    epsilon > 0 || throw(ArgumentError("epsilon must be positive, got $epsilon"))
    bufs = (Float64[], Float64[], Float64[])
    for i in eachindex(r1.u)
        foreach(empty!, bufs)
        for r in results
            sample_valid(r, i, false) || continue
            push!(bufs[1], Float64(r.u[i]))
            push!(bufs[2], Float64(r.v[i]))
            push!(bufs[3], Float64(r.w[i]))
        end
        length(bufs[1]) >= 3 || continue
        meds = map(median, bufs)
        mads = ntuple(k -> median(abs.(bufs[k] .- meds[k])), 3)
        for r in results
            sample_valid(r, i, false) || continue
            vals = (r.u[i], r.v[i], r.w[i])
            if any(k -> abs(vals[k] - meds[k]) / (mads[k] + epsilon) > threshold, 1:3)
                r.outliers[i] = true
            end
        end
    end
    return results
end

"""
    error_statistics(result, u_ref, v_ref; include_invalid = false) -> NamedTuple

Compare a planar field with reference components in the same units.
`u_ref`/`v_ref` are grid-sized arrays or functions `(x, y) -> value`
evaluated at the interrogation grid points. Returns
`(; err_u, err_v, bias_u, bias_v, rms_u, rms_v, n)`: signed error fields
(`NaN` where invalid) plus mean (bias) and RMS errors over the `n` valid
vectors. Errors are measured minus reference. Masked or non-finite measured
vectors are always excluded; `include_invalid=true` admits flagged vectors.
Reference values must be finite at admitted locations, or they propagate
into the error statistics. Attached physical scaling is not applied.
"""
function error_statistics(result::PIVResult, u_ref, v_ref; include_invalid::Bool = false)
    as_field(f) = f isa AbstractMatrix ? Float64.(f) :
                  [Float64(f(xj, yi)) for yi in result.y, xj in result.x]
    ur = as_field(u_ref)
    vr = as_field(v_ref)
    size(ur) == size(result.u) && size(vr) == size(result.u) ||
        throw(ArgumentError("reference fields must match the $(size(result.u)) grid"))
    err_u = fill(NaN, size(result.u))
    err_v = fill(NaN, size(result.u))
    su = sv = suu = svv = 0.0
    n = 0
    for i in eachindex(result.u)
        sample_valid(result, i, include_invalid) || continue
        eu = Float64(result.u[i]) - ur[i]
        ev = Float64(result.v[i]) - vr[i]
        err_u[i] = eu
        err_v[i] = ev
        su += eu
        sv += ev
        suu += eu^2
        svv += ev^2
        n += 1
    end
    scalar(x) = n > 0 ? x : NaN
    return (; err_u, err_v,
            bias_u = scalar(su / max(n, 1)), bias_v = scalar(sv / max(n, 1)),
            rms_u = scalar(sqrt(suu / max(n, 1))), rms_v = scalar(sqrt(svv / max(n, 1))),
            n)
end

"""
    peak_locking(displacements; nbins = 21) -> (fractions, counts, index)

Summarize fractional displacements in pixels, `f = x − round(x) ∈
[−0.5, 0.5)` of a displacement sample (any array; non-finite entries are
skipped). Returns the histogram bin centers and counts, plus a locking
index `(n_center - n_edge) / (n_center + n_edge)`, where `n_center` counts
`|f| ≤ 0.1` and `n_edge` counts `|f| ≥ 0.4`. Positive values indicate more
samples near integers; negative values indicate more near half-integers.
The index is `NaN` if neither band contains samples, even if other bins do.

Clustering near integers can indicate peak locking, but the flow's actual
displacement distribution also affects this diagnostic. Pass only the samples
you intend to assess, for example `result.u[.!result.mask .& .!result.outliers]`.
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

Estimate a one-sided power spectral density from at least two uniformly
spaced samples. The mean is removed before applying `window=:hann` or `:none`.
Supply a finite, positive sampling interval `dt`. Frequencies are in cycles
per time unit; PSD has units of signal squared per frequency unit.

With `:none`, `sum(psd) * Δf` equals the population variance (division by
the sample count). With `:hann`, it equals the power-normalized mean square
of the tapered, demeaned signal, which can differ from the untapered variance.
For two samples, `:hann` uses an untapered window.

Non-finite samples are not removed or filled. For a sequence of PIV fields,
use [`result_spectrum`](@ref) to handle validation flags explicitly. Its `dt`
is the interval between fields, separate from the delay within an image pair.
"""
function power_spectrum(signal::AbstractVector{<:Real};
                        dt::Real = 1.0, window::Symbol = :hann)
    n = length(signal)
    n >= 2 || throw(ArgumentError("signal must have at least 2 samples, got $n"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive, got $dt"))
    x = Float64.(signal)
    x .-= sum(x) / n
    if window === :hann && n > 2
        x .*= [0.5 * (1 - cospi(2 * (k - 1) / (n - 1))) for k in 1:n]
        wnorm = sum(abs2, 0.5 * (1 - cospi(2 * (k - 1) / (n - 1))) for k in 1:n)
    elseif window === :none || (window === :hann && n == 2)
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
