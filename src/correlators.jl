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

"""
    CrossCorrelator{T}(window_size::Dims{2})
    CrossCorrelator(window_size)  # T = Float32

FFT-based circular cross-correlation of two interrogation windows, with
preallocated buffers and cached in-place FFTW plans.
"""
struct CrossCorrelator{T<:AbstractFloat,FP,IP} <: Correlator
    C1::Matrix{Complex{T}}
    C2::Matrix{Complex{T}}
    R::Matrix{T}
    fp::FP
    ip::IP
end

function CrossCorrelator{T}(window_size::Dims{2}) where {T<:AbstractFloat}
    C1 = zeros(Complex{T}, window_size)
    fp = plan_fft!(C1)
    CrossCorrelator(C1, zeros(Complex{T}, window_size), zeros(T, window_size), fp, inv(fp))
end

CrossCorrelator(window_size::Dims{2}) = CrossCorrelator{Float32}(window_size)

"""
    PhaseCorrelator{T}(window_size::Dims{2}; filter_sigma = min(window_size...) / 8)
    PhaseCorrelator(window_size)  # T = Float32

Filtered phase correlation (normalized cross-power spectrum) of two
interrogation windows, with preallocated buffers and cached in-place FFTW
plans. More robust to illumination differences than plain cross-correlation.

Whitening the spectrum gives noise-only high-frequency bins the same weight as
signal-bearing ones, which destroys the peak for low-frequency content like
particle images; the whitened spectrum is therefore weighted by a Gaussian
low-pass with standard deviation `filter_sigma` (in FFT frequency bins).
"""
struct PhaseCorrelator{T<:AbstractFloat,FP,IP} <: Correlator
    C1::Matrix{Complex{T}}
    C2::Matrix{Complex{T}}
    R::Matrix{T}
    W::Matrix{T}  # Gaussian spectral filter, in FFT (unshifted) bin order
    fp::FP
    ip::IP
end

function PhaseCorrelator{T}(window_size::Dims{2};
                            filter_sigma::Real = min(window_size...) / 8) where {T<:AbstractFloat}
    filter_sigma > 0 || throw(ArgumentError("filter_sigma must be positive, got $filter_sigma"))
    C1 = zeros(Complex{T}, window_size)
    fp = plan_fft!(C1)
    fr = fftfreq(window_size[1], window_size[1])
    fc = fftfreq(window_size[2], window_size[2])
    W = T[exp(-(fi^2 + fj^2) / (2 * filter_sigma^2)) for fi in fr, fj in fc]
    PhaseCorrelator(C1, zeros(Complex{T}, window_size), zeros(T, window_size), W, fp, inv(fp))
end

PhaseCorrelator(window_size::Dims{2}; kwargs...) = PhaseCorrelator{Float32}(window_size; kwargs...)

for CT in (CrossCorrelator, PhaseCorrelator)
    @eval Base.show(io::IO, c::$CT) =
        print(io, $(string(nameof(CT))), "{", real(eltype(c.C1)), "}(", size(c.R, 1), "×", size(c.R, 2), ")")
end

window_size(c::Correlator) = size(c.R)

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
    correlate(c::Correlator, subA, subB; subpixel=:gauss3)

Correlate two interrogation windows and locate the displacement peak.

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
                   subpixel::Symbol = :gauss3)
    size(subA) == size(subB) == window_size(c) ||
        throw(DimensionMismatch("expected windows of size $(window_size(c)), got $(size(subA)) and $(size(subB))"))

    # Subtract window means to remove the DC pedestal from the correlation
    # plane; it doesn't move the integer peak but biases the subpixel fit.
    c.C1 .= subA
    c.C2 .= subB
    c.C1 .-= sum(c.C1) / length(c.C1)
    c.C2 .-= sum(c.C2) / length(c.C2)
    mul!(c.C1, c.fp, c.C1)
    mul!(c.C2, c.fp, c.C2)
    spectrum!(c)
    mul!(c.C1, c.ip, c.C1)
    fftshift_abs!(c.R, c.C1)

    peakidx = argmax(c.R)
    peak = c.R[peakidx]
    peakloc = Tuple(peakidx)
    refined = refine_peak(c.R, peakloc, subpixel)
    center = size(c.R) .÷ 2 .+ 1
    dv = refined[1] - center[1]
    du = refined[2] - center[2]
    return (; du, dv, peak = Float64(peak), peakloc, refined_peakloc = refined, correlation = c.R)
end

function refine_peak(R::AbstractMatrix, peakloc::Tuple{Int,Int}, method::Symbol)
    method === :gauss3 && return subpixel_gauss3(R, peakloc)
    method === :gauss2d && return subpixel_gauss2d(R, peakloc)
    method === :none && return float.(peakloc)
    throw(ArgumentError("unknown subpixel method :$method (expected :gauss3, :gauss2d, or :none)"))
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
the peak sits on the matrix edge or the fit fails to converge.
"""
function subpixel_gauss2d(R::AbstractMatrix{<:AbstractFloat}, peakloc::Tuple{Int,Int})
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
        return (fit.param[4], fit.param[2])
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
    du = 0.0
    dv = 0.0
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
