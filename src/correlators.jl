# Correlators hold preallocated buffers and cached FFTW plans so that the
# expensive plan creation is paid once per window size, not once per window.
#
# Sign convention (used throughout the package): `correlate(c, A, B)` returns
# the displacement of B relative to A, as `(du, dv)` with `u` along columns (x)
# and `v` along rows (y). A particle at `(row, col)` in A found at
# `(row + dv, col + du)` in B yields positive `(du, dv)`.

"""
    Correlator

Abstract supertype for window correlators. Concrete subtypes:
[`CrossCorrelator`](@ref), [`PhaseCorrelator`](@ref).
"""
abstract type Correlator end

# Apodization weights applied to the mean-subtracted windows; ones(...) for :none.
function apodization_window(::Type{T}, wsize::Dims{2}, apodization::Symbol) where {T}
    apodization === :none && return ones(T, wsize)
    apodization === :gauss ||
        throw(ArgumentError("apodization must be :none or :gauss, got :$apodization"))
    cr, cc = (wsize .+ 1) ./ 2
    sr, sc = wsize ./ 4
    return T[exp(-((i - cr)^2 / (2sr^2) + (j - cc)^2 / (2sc^2))) for i in 1:wsize[1], j in 1:wsize[2]]
end

"""
    CrossCorrelator{T}(window_size::Dims{2}; padding=false, apodization=:none)
    CrossCorrelator(window_size; kwargs...)  # T = Float32

FFT-based cross-correlation of two interrogation windows, with preallocated
buffers and cached in-place FFTW plans.

- `padding = true` zero-pads the FFT to twice the window size, replacing
  circular with true linear correlation; this removes the wrap-around noise
  that biases subpixel estimates toward zero, at roughly 4× the FFT cost.
- `apodization = :gauss` applies a Gaussian window (σ = window/4) to the
  mean-subtracted inputs to reduce edge effects and spectral leakage.
"""
struct CrossCorrelator{T<:AbstractFloat,FP,IP} <: Correlator
    C1::Matrix{Complex{T}}
    C2::Matrix{Complex{T}}
    R::Matrix{T}
    apod::Matrix{T}
    gain::Matrix{T}  # overlap normalization for padded correlation; empty if unpadded
    wsize::Dims{2}
    fp::FP
    ip::IP
end

function CrossCorrelator{T}(window_size::Dims{2};
                            padding::Bool = false, apodization::Symbol = :none) where {T<:AbstractFloat}
    fft_size = padding ? 2 .* window_size : window_size
    C1 = zeros(Complex{T}, fft_size)
    fp = plan_fft!(C1)
    ip = inv(fp)
    apod = apodization_window(T, window_size, apodization)
    gain = padding ? overlap_gain!(C1, fp, ip, apod) : Matrix{T}(undef, 0, 0)
    CrossCorrelator(C1, zeros(Complex{T}, fft_size), zeros(T, fft_size),
                    apod, gain, window_size, fp, ip)
end

CrossCorrelator(window_size::Dims{2}; kwargs...) = CrossCorrelator{Float32}(window_size; kwargs...)

"""
    PhaseCorrelator{T}(window_size::Dims{2}; padding=false, apodization=:none,
                       filter_sigma=min(fft_size...) / 8)
    PhaseCorrelator(window_size; kwargs...)  # T = Float32

Filtered phase correlation (normalized cross-power spectrum) of two
interrogation windows, with preallocated buffers and cached in-place FFTW
plans. More robust to illumination differences than plain cross-correlation.
`padding` and `apodization` behave as for [`CrossCorrelator`](@ref).

Whitening the spectrum gives noise-only high-frequency bins the same weight as
signal-bearing ones, which destroys the peak for low-frequency content like
particle images; the whitened spectrum is therefore weighted by a Gaussian
low-pass with standard deviation `filter_sigma` (in FFT frequency bins of the
possibly padded transform).
"""
struct PhaseCorrelator{T<:AbstractFloat,FP,IP} <: Correlator
    C1::Matrix{Complex{T}}
    C2::Matrix{Complex{T}}
    R::Matrix{T}
    W::Matrix{T}  # Gaussian spectral filter, in FFT (unshifted) bin order
    apod::Matrix{T}
    gain::Matrix{T}  # overlap normalization for padded correlation; empty if unpadded
    wsize::Dims{2}
    fp::FP
    ip::IP
end

function PhaseCorrelator{T}(window_size::Dims{2};
                            padding::Bool = false, apodization::Symbol = :none,
                            filter_sigma::Real = min((padding ? 2 .* window_size : window_size)...) / 8) where {T<:AbstractFloat}
    filter_sigma > 0 || throw(ArgumentError("filter_sigma must be positive, got $filter_sigma"))
    fft_size = padding ? 2 .* window_size : window_size
    C1 = zeros(Complex{T}, fft_size)
    fp = plan_fft!(C1)
    ip = inv(fp)
    fr = fftfreq(fft_size[1], fft_size[1])
    fc = fftfreq(fft_size[2], fft_size[2])
    W = T[exp(-(fi^2 + fj^2) / (2 * filter_sigma^2)) for fi in fr, fj in fc]
    apod = apodization_window(T, window_size, apodization)
    gain = padding ? overlap_gain!(C1, fp, ip, apod) : Matrix{T}(undef, 0, 0)
    PhaseCorrelator(C1, zeros(Complex{T}, fft_size), zeros(T, fft_size), W,
                    apod, gain, window_size, fp, ip)
end

PhaseCorrelator(window_size::Dims{2}; kwargs...) = PhaseCorrelator{Float32}(window_size; kwargs...)

for CT in (CrossCorrelator, PhaseCorrelator)
    @eval Base.show(io::IO, c::$CT) =
        print(io, $(string(nameof(CT))), "{", real(eltype(c.C1)), "}(",
              c.wsize[1], "×", c.wsize[2],
              size(c.R) == c.wsize ? "" : ", padded to $(size(c.R, 1))×$(size(c.R, 2))", ")")
end

window_size(c::Correlator) = c.wsize

make_correlator(params::PIVParameters, ::Type{T}) where {T} =
    params.correlation_method === :cross ?
        CrossCorrelator{T}(params.window_size; params.padding, params.apodization) :
        PhaseCorrelator{T}(params.window_size; params.padding, params.apodization)

# Private correlation-engine hook. The CPU engine is deliberately just a thin
# wrapper around the existing FFTW correlator so current results and workspace
# reuse are unchanged, while future backends can dispatch at the chunk/window
# processing level without changing the public correlator API.
struct _CPUCorrelationEngine{C<:Correlator}
    correlator::C
end

_make_correlation_engine(params::PIVParameters, ::Type{T}) where {T} =
    _CPUCorrelationEngine(make_correlator(params, T))
_make_correlation_engine(c::Correlator) = _CPUCorrelationEngine(c)
_correlation_apod(engine::_CPUCorrelationEngine) = engine.correlator.apod
_correlation_plane!(engine::_CPUCorrelationEngine, subA::AbstractMatrix, subB::AbstractMatrix,
                    mask::Union{Nothing,AbstractMatrix{Bool}} = nothing) =
    correlation_plane!(engine.correlator, subA, subB, mask)

# Padded (linear) correlation weights each lag by the shrinking window overlap
# — the autocorrelation of the apodization window (a triangle for :none) —
# which biases the peak toward zero. Precompute the inverse as a gain plane in
# the fftshifted layout of R, zeroed below 50% of the zero-lag weight: this
# avoids amplifying noise at large lags and enforces the usual quarter-window
# displacement limit. Uses C1 as scratch (zeroed on exit).
function overlap_gain!(C1::Matrix{Complex{T}}, fp, ip, apod::Matrix{T}) where {T}
    fill!(C1, zero(eltype(C1)))
    C1[axes(apod, 1), axes(apod, 2)] .= apod
    mul!(C1, fp, C1)
    @. C1 = abs2(C1)
    mul!(C1, ip, C1)
    weight = zeros(T, size(C1))
    fftshift_abs!(weight, C1)
    fill!(C1, zero(eltype(C1)))
    wmax = maximum(weight)
    return T[w >= wmax / 2 ? wmax / w : zero(T) for w in weight]
end

# Copy the mean-subtracted, apodized windows into the (possibly padded) FFT
# buffers. The in-place FFT overwrites the whole buffer, so the padding region
# must be re-zeroed on every call. Masked pixels (submask true) are loaded at
# the valid-pixel mean, so they contribute exactly zero after mean subtraction
# — no intensity step at the mask edge to bias the correlation peak.
function load_windows!(c::Correlator, subA::AbstractMatrix, subB::AbstractMatrix,
                       submask::Union{Nothing,AbstractMatrix{Bool}} = nothing)
    T = eltype(c.R)
    wr, wc = c.wsize
    if size(c.C1) != c.wsize
        fill!(c.C1, zero(eltype(c.C1)))
        fill!(c.C2, zero(eltype(c.C2)))
    end
    if submask === nothing
        meanA = T(sum(subA) / length(subA))
        meanB = T(sum(subB) / length(subB))
        @inbounds for j in 1:wc, i in 1:wr
            c.C1[i, j] = c.apod[i, j] * (T(subA[i, j]) - meanA)
            c.C2[i, j] = c.apod[i, j] * (T(subB[i, j]) - meanB)
        end
    else
        size(submask) == c.wsize ||
            throw(DimensionMismatch("mask must match the window size $(c.wsize), got $(size(submask))"))
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
        @inbounds for j in 1:wc, i in 1:wr
            if submask[i, j]
                c.C1[i, j] = zero(eltype(c.C1))
                c.C2[i, j] = zero(eltype(c.C2))
            else
                c.C1[i, j] = c.apod[i, j] * (T(subA[i, j]) - meanA)
                c.C2[i, j] = c.apod[i, j] * (T(subB[i, j]) - meanB)
            end
        end
    end
    return c
end

# Write the (cross-power) spectrum into c.C1, given FFTs of A in C1 and B in C2.
function spectrum!(c::CrossCorrelator)
    @inbounds for i in eachindex(c.C1)
        c.C1[i] = conj(c.C1[i]) * c.C2[i]
    end
    return c
end

function spectrum!(c::PhaseCorrelator{T}) where {T}
    @inbounds for i in eachindex(c.C1)
        s = conj(c.C1[i]) * c.C2[i]
        c.C1[i] = c.W[i] * s / (abs(s) + eps(T))
    end
    return c
end

# fftshift the magnitude of complex matrix C into real matrix R (zero lag at center).
function fftshift_abs!(R::AbstractMatrix{T}, C::AbstractMatrix{<:Complex}) where {T}
    nr, nc = size(C)
    sr, sc = nr ÷ 2, nc ÷ 2
    @inbounds for j in 1:nc, i in 1:nr
        R[mod1(i + sr, nr), mod1(j + sc, nc)] = abs(C[i, j])
    end
    return R
end

"""
    correlate(c::Correlator, subA, subB; subpixel=:gauss3, mask=nothing)

Correlate two interrogation windows and locate the displacement peak.
`mask` is an optional window-sized `Bool` matrix; `true` pixels are excluded
from the correlation (they enter at the valid-pixel mean, contributing zero
after mean subtraction).

Returns a named tuple `(du, dv, peak, peakloc, refined_peakloc, correlation)`:
- `du`, `dv`: subpixel displacement of `subB` relative to `subA` along columns
  (x) and rows (y) respectively;
- `peak`: correlation value at the integer peak;
- `peakloc`: integer `(row, col)` peak location in the correlation plane;
- `refined_peakloc`: subpixel `(row, col)` peak location;
- `correlation`: the real correlation plane (zero lag at `size .÷ 2 .+ 1`).

`subpixel` is one of `:gauss3`, `:gauss2d`, or `:none`.

!!! warning
    `correlation` aliases an internal buffer of `c` and is overwritten by the
    next `correlate` call. Copy it if you need to keep it.
"""
function correlate(c::Correlator, subA::AbstractMatrix, subB::AbstractMatrix;
                   subpixel::Symbol = :gauss3,
                   mask::Union{Nothing,AbstractMatrix{Bool}} = nothing)
    R = correlation_plane!(c, subA, subB, mask)
    res = locate_displacement(R, subpixel)
    return (; res.du, res.dv, res.peak, res.peakloc, res.refined_peakloc, correlation = R)
end

# Compute the (fftshifted, overlap-normalized) correlation plane of two
# windows into c.R and return it, without peak detection — the building block
# shared by `correlate` and ensemble (sum-of-correlation) PIV. The returned
# plane aliases c's internal buffer.
function correlation_plane!(c::Correlator, subA::AbstractMatrix, subB::AbstractMatrix,
                            mask::Union{Nothing,AbstractMatrix{Bool}} = nothing)
    size(subA) == size(subB) == window_size(c) ||
        throw(DimensionMismatch("expected windows of size $(window_size(c)), got $(size(subA)) and $(size(subB))"))
    # Mean subtraction removes the DC pedestal from the correlation plane; it
    # doesn't move the integer peak but biases the subpixel fit.
    load_windows!(c, subA, subB, mask)
    mul!(c.C1, c.fp, c.C1)
    mul!(c.C2, c.fp, c.C2)
    spectrum!(c)
    mul!(c.C1, c.ip, c.C1)
    fftshift_abs!(c.R, c.C1)
    isempty(c.gain) || (c.R .*= c.gain)
    return c.R
end

"""
    find_peaks(R, k; exclusion_radius = 2, peak_finder = :regionalmax) -> Vector of (value, location)

Locate up to `k` correlation peaks in descending order. With the default
`peak_finder = :regionalmax`, candidates are 8-connected local maxima, so
nearby but distinct secondary peaks are not hidden by a fixed exclusion box.
With `peak_finder = :exclusion`, this returns the global maximum, then
repeatedly the largest value outside square exclusion zones of
`±exclusion_radius` pixels around every peak already found. Peaks after the
first must be positive.
"""
function find_peaks(R::AbstractMatrix{T}, k::Int; exclusion_radius::Int = 2,
                    peak_finder::Symbol = :regionalmax) where {T<:AbstractFloat}
    k >= 1 || throw(ArgumentError("k must be at least 1, got $k"))
    vals = Vector{T}(undef, k)
    locs = Vector{NTuple{2,Int}}(undef, k)
    found = find_peaks!(vals, locs, R, k; exclusion_radius, peak_finder)
    return [(value = vals[m], location = locs[m]) for m in 1:found]
end

# Allocation-free core: fills the scratch vectors `vals`/`locs` (length ≥ k)
# and returns how many peaks were found.
function find_peaks!(vals, locs, R::AbstractMatrix{T}, k::Int;
                     exclusion_radius::Int = 2,
                     peak_finder::Symbol = :regionalmax) where {T<:AbstractFloat}
    peak_finder === :exclusion &&
        return find_peaks_exclusion!(vals, locs, R, k; exclusion_radius)
    peak_finder === :regionalmax &&
        return find_peaks_regionalmax!(vals, locs, R, k)
    throw(ArgumentError("peak_finder must be :exclusion or :regionalmax, got :$peak_finder"))
end

# Classic path: k plane scans, no branches beyond the exclusion test.
function find_peaks_exclusion!(vals, locs, R::AbstractMatrix{T}, k::Int;
                               exclusion_radius::Int = 2) where {T<:AbstractFloat}
    nr, nc = size(R)
    found = 0
    for _ in 1:k
        best = T(-Inf)
        bi = 0
        bj = 0
        @inbounds for j in 1:nc, i in 1:nr
            excluded = false
            for p in 1:found
                if abs(i - locs[p][1]) <= exclusion_radius &&
                   abs(j - locs[p][2]) <= exclusion_radius
                    excluded = true
                    break
                end
            end
            excluded && continue
            if R[i, j] > best
                best = R[i, j]
                bi = i
                bj = j
            end
        end
        bi == 0 && break                  # everything excluded (degenerate plane)
        found > 0 && best <= 0 && break   # secondary peaks must be positive
        found += 1
        vals[found] = best
        locs[found] = (bi, bj)
    end
    return found
end

# Regional-max path: one plane scan, retaining the strongest k local maxima in
# caller-owned scratch. The primary may be non-positive on a degenerate plane;
# secondary peaks must be positive, matching the exclusion finder.
function find_peaks_regionalmax!(vals, locs, R::AbstractMatrix{T}, k::Int) where {T<:AbstractFloat}
    nr, nc = size(R)
    nfound = 0
    @inbounds for j in 1:nc, i in 1:nr
        is_local_maximum(R, i, j) || continue
        val = R[i, j]
        (nfound < k || val > vals[nfound]) || continue

        pos = nfound
        while pos >= 1 && val > vals[pos]
            pos -= 1
        end
        ins = pos + 1
        ins <= k || continue
        nfound < k && (nfound += 1)
        for m in nfound:-1:(ins + 1)
            vals[m] = vals[m - 1]
            locs[m] = locs[m - 1]
        end
        vals[ins] = val
        locs[ins] = (i, j)
    end
    nfound == 0 && return 0
    found = 1
    while found < nfound && vals[found + 1] > 0
        found += 1
    end
    return found
end

# Analyze one correlation plane using scratch vals/locs (length ≥
# max(params.n_peaks, 2)): subpixel primary displacement, peak ratio from the
# top two peaks, correlation moment, and the peak list for alternatives.
function analyze_plane!(vals, locs, R::AbstractMatrix{T}, params::PIVParameters) where {T}
    found = find_peaks!(vals, locs, R, max(params.n_peaks, 2);
                        peak_finder = params.peak_finder)
    refined = refine_peak(R, locs[1], params.subpixel_method)
    center = size(R) .÷ 2 .+ 1
    ratio = found >= 2 ? vals[1] / vals[2] : (vals[1] > 0 ? T(Inf) : T(NaN))
    return (; du = refined[2] - center[2], dv = refined[1] - center[1],
            ratio, moment = T(calculate_correlation_moment(R, refined)),
            found, center)
end

# Locate the displacement peak of a correlation plane (zero lag at center).
function locate_displacement(R::AbstractMatrix{<:AbstractFloat}, subpixel::Symbol)
    peakidx = argmax(R)
    peak = R[peakidx]
    peakloc = Tuple(peakidx)
    refined = refine_peak(R, peakloc, subpixel)
    center = size(R) .÷ 2 .+ 1
    return (; du = refined[2] - center[2], dv = refined[1] - center[1],
            peak, peakloc, refined_peakloc = refined)
end

# All subpixel paths return coordinates in the correlation plane's precision T,
# so du/dv/peak share the correlator's element type end to end.
function refine_peak(R::AbstractMatrix{T}, peakloc::Tuple{Int,Int}, method::Symbol) where {T<:AbstractFloat}
    method === :gauss3 && return subpixel_gauss3(R, peakloc)
    method === :gauss9 && return subpixel_gauss9(R, peakloc)
    method === :gauss2d && return subpixel_gauss2d(R, peakloc)
    method === :none && return (T(peakloc[1]), T(peakloc[2]))
    throw(ArgumentError("unknown subpixel method :$method (expected :gauss3, :gauss9, :gauss2d, or :none)"))
end

"""
    subpixel_gauss3(R, peakloc::Tuple{Int,Int}) -> (row, col)

Refine an integer correlation peak to subpixel precision with independent
3-point Gaussian fits along rows and columns. Falls back to a zero offset along
any axis where the fit is undefined (peak on the matrix edge, or non-positive
neighbor values).
"""
function subpixel_gauss3(R::AbstractMatrix{T}, peakloc::Tuple{Int,Int}) where {T<:AbstractFloat}
    nr, nc = size(R)
    i, j = peakloc
    di = zero(T)
    dj = zero(T)
    I0 = R[i, j]
    if 1 < i < nr
        Im, Ip = R[i-1, j], R[i+1, j]
        if Im > 0 && I0 > 0 && Ip > 0
            denom = log(Im) - 2log(I0) + log(Ip)
            di = denom != 0 ? (log(Im) - log(Ip)) / (2denom) : zero(T)
        end
    end
    if 1 < j < nc
        Im, Ip = R[i, j-1], R[i, j+1]
        if Im > 0 && I0 > 0 && Ip > 0
            denom = log(Im) - 2log(I0) + log(Ip)
            dj = denom != 0 ? (log(Im) - log(Ip)) / (2denom) : zero(T)
        end
    end
    return (i + di, j + dj)
end

"""
    subpixel_gauss9(R, peakloc::Tuple{Int,Int}) -> (row, col)

Refine an integer correlation peak with a closed-form least-squares 2D
Gaussian regression over the 3×3 neighborhood (log-paraboloid including the
cross term, Nobach & Honkanen 2005). Unlike the two independent 1D fits of
[`subpixel_gauss3`](@ref), this is exact for an elliptical — including
rotated — Gaussian peak and exhibits less peak locking, at the cost of nine
`log`s and a hardcoded 2×2 solve. Falls back to `subpixel_gauss3` at plane
edges, on non-positive samples, or when the fitted surface has no interior
maximum.
"""
function subpixel_gauss9(R::AbstractMatrix{T}, peakloc::Tuple{Int,Int}) where {T<:AbstractFloat}
    nr, nc = size(R)
    pr, pc = peakloc
    (1 < pr < nr && 1 < pc < nc) || return subpixel_gauss3(R, peakloc)
    sL = zero(T); sxL = zero(T); syL = zero(T)
    sxxL = zero(T); syyL = zero(T); sxyL = zero(T)
    @inbounds for dj in -1:1, di in -1:1
        val = R[pr + di, pc + dj]
        val > 0 || return subpixel_gauss3(R, peakloc)
        L = log(val)
        sL += L
        sxL += dj * L
        syL += di * L
        sxxL += dj * dj * L
        syyL += di * di * L
        sxyL += di * dj * L
    end
    # Least-squares paraboloid a1·x + a2·y + a3·x² + a4·xy + a5·y² (+ a0) on
    # the 3×3 stencil; the normal equations reduce to these closed forms.
    a1 = sxL / 6
    a2 = syL / 6
    a3 = sxxL / 2 - sL / 3
    a4 = sxyL / 4
    a5 = syyL / 2 - sL / 3
    D = 4 * a3 * a5 - a4^2
    (a3 < 0 && D > 0) || return subpixel_gauss3(R, peakloc)  # not a maximum
    dx = (a4 * a2 - 2 * a5 * a1) / D
    dy = (a4 * a1 - 2 * a3 * a2) / D
    (abs(dx) <= 1 && abs(dy) <= 1) || return subpixel_gauss3(R, peakloc)
    return (pr + dy, pc + dx)
end

# 2D Gaussian peak model: p = [amplitude, col0, sigma_col, row0, sigma_row, offset];
# xy is an N×2 matrix of (col, row) coordinates.
function gauss2d_model(xy, p)
    x = view(xy, :, 1)
    y = view(xy, :, 2)
    return p[1] .* exp.(-((x .- p[2]) .^ 2 ./ (2p[3]^2) .+ (y .- p[4]) .^ 2 ./ (2p[5]^2))) .+ p[6]
end

"""
    subpixel_gauss2d(R, peakloc::Tuple{Int,Int}) -> (row, col)

Refine an integer correlation peak by least-squares fitting a 2D Gaussian to
the 3×3 neighborhood around it. Falls back to [`subpixel_gauss3`](@ref) when
the peak sits on the matrix edge or the fit fails to converge. The fit itself
runs in `Float64` (LsqFit on 9 points, CPU); the result is returned in the
precision of `R`.
"""
function subpixel_gauss2d(R::AbstractMatrix{T}, peakloc::Tuple{Int,Int}) where {T<:AbstractFloat}
    nr, nc = size(R)
    pr, pc = peakloc
    (1 < pr < nr && 1 < pc < nc) || return subpixel_gauss3(R, peakloc)

    rows = (pr-1):(pr+1)
    cols = (pc-1):(pc+1)
    z = Float64[R[r, c] for r in rows for c in cols]
    xy = hcat([Float64(c) for _ in rows for c in cols],
              [Float64(r) for r in rows for _ in cols])

    lo, hi = extrema(z)
    amplitude = hi > lo ? hi - lo : one(Float64)
    p0 = [amplitude, Float64(pc), 1.0, Float64(pr), 1.0, lo]
    lower = [0.0, pc - 1.0, 0.1, pr - 1.0, 0.1, -Inf]
    upper = [Inf, pc + 1.0, 3.0, pr + 1.0, 3.0, Inf]

    try
        fit = curve_fit(gauss2d_model, xy, z, p0; lower, upper)
        return (T(fit.param[4]), T(fit.param[2]))
    catch
        return subpixel_gauss3(R, peakloc)
    end
end

"""
    correlate_deformable(c::Correlator, subA, subB;
                         iterations=3, subpixel=:gauss3, tol=1e-3)

Iteratively correlate `subA` against a back-warped `subB`: after each pass the
accumulated displacement is used to translate `subB` into alignment with
`subA`, and the residual displacement is re-estimated. Stops after
`iterations` passes or when the residual falls below `tol` pixels.

Returns the same named tuple as [`correlate`](@ref), with `du`/`dv` holding the
accumulated displacement and the remaining fields describing the final pass.
"""
function correlate_deformable(c::Correlator, subA::AbstractMatrix, subB::AbstractMatrix;
                              iterations::Int = 3, subpixel::Symbol = :gauss3, tol::Real = 1e-3)
    iterations >= 1 || throw(ArgumentError("iterations must be at least 1, got $iterations"))
    du = zero(eltype(c.R))
    dv = zero(eltype(c.R))
    res = correlate(c, subA, subB; subpixel)
    du += res.du
    dv += res.dv
    for _ in 2:iterations
        (abs(res.du) < tol && abs(res.dv) < tol) && break
        # Translate subB back by the accumulated displacement to align it with subA.
        tform = AffineTransform([1.0 0.0; 0.0 1.0], [-du, -dv])
        res = correlate(c, subA, warp_image(subB, tform); subpixel)
        du += res.du
        dv += res.dv
    end
    return (; du, dv, peak = res.peak, peakloc = res.peakloc,
            refined_peakloc = res.refined_peakloc, correlation = res.correlation)
end
