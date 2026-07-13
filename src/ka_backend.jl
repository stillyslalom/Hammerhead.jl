# Portable KernelAbstractions correlation kernels and the built-in
# `backend = :ka` engine that runs them on KernelAbstractions' CPU backend — a
# hardware-free proving tier for the kernels the GPU device extensions
# (AMDGPU, CUDA) reuse with their own array types, FFT plans, and device
# backends. The math mirrors the CPU FFTW correlator (`load_windows!`,
# `spectrum!`, `fftshift_abs!`, overlap gain) so planes match `backend = :cpu`
# within FFT round-off, and the plane-analysis kernel (`_ka_analyze!`) ports
# `analyze_plane!` so peak outputs match within `log`-intrinsic round-off.

# Shared option-scope guard for the KA-derived backends (:ka, :amdgpu, :cuda).
# The initial device scope is the plan's Phase 2 slice; the excluded options
# stay CPU-first (or land in a later phase).
function _ka_scope_check(passes, name::Symbol, fp64::Bool = true)
    for p in passes
        p.correlation_method === :cross ||
            throw(ArgumentError("backend :$name supports correlation_method = :cross only " *
                                "(got :$(p.correlation_method)); use backend = :cpu"))
        p.subpixel_method in (:gauss3, :gauss9) ||
            throw(ArgumentError("backend :$name supports subpixel_method :gauss3 or :gauss9 " *
                                "only (got :$(p.subpixel_method)); use backend = :cpu"))
        p.uncertainty && !fp64 &&
            throw(ArgumentError("backend :$name cannot preserve Float64 uncertainty " *
                                "accumulation; use backend = :cpu"))
        p.keep_correlation_planes &&
            throw(ArgumentError("backend :$name does not support keep_correlation_planes yet; " *
                                "use backend = :cpu"))
    end
    return nothing
end

# Device UQ (Phase 4b). Each work-item owns one (component, statistic, window)
# scalar, exposing enough parallelism even on modest vector grids. Recomputing
# the small smoothed dC stencil avoids window-sized device scratch, while the
# Float64 statistics remain resident and additive across ensemble pairs.
@inline function _ka_uq_dc(A, B, r, c, k, transposed)
    if transposed
        a0 = real(A[c, r, k]); a1 = real(A[c + 1, r, k])
        b0 = real(B[c, r, k]); b1 = real(B[c + 1, r, k])
    else
        a0 = real(A[r, c, k]); a1 = real(A[r, c + 1, k])
        b0 = real(B[r, c, k]); b1 = real(B[r, c + 1, k])
    end
    return a0 * b1 - a1 * b0
end

@inline function _ka_uq_dcs(A, B, r, c, k, nr, m, transposed)
    T = typeof(real(A[1, 1, k]))
    cl = max(c - 1, 1)
    cr = min(c + 1, m)
    return (T(1) / 4) * (_ka_uq_dc(A, B, r, cl, k, transposed) +
                         2 * _ka_uq_dc(A, B, r, c, k, transposed) +
                         _ka_uq_dc(A, B, r, cr, k, transposed))
end

@kernel function _ka_uq_means!(means, @Const(A), @Const(B), wr, wc)
    comp, k = @index(Global, NTuple)
    transposed = comp == 2
    nr = transposed ? wc : wr
    m = (transposed ? wr : wc) - 1
    mu = 0.0
    @inbounds for c in 1:m, r in 1:nr
        mu += Float64(_ka_uq_dcs(A, B, r, c, k, nr, m, transposed))
    end
    @inbounds means[comp, k] = mu / (nr * m)
end

@kernel function _ka_uq_stats!(stats, @Const(means), @Const(A), @Const(B),
                               wr, wc, nreal, job0, add)
    comp, si, k = @index(Global, NTuple)
    @inbounds begin
        transposed = comp == 2
        nr = transposed ? wc : wr
        nc = transposed ? wr : wc
        m = nc - 1
        value = 0.0
        if si <= 3
            for c in 1:m, r in 1:nr
                if transposed
                    a0 = real(A[c, r, k]); a1 = real(A[c + 1, r, k])
                    b0 = real(B[c, r, k]); b1 = real(B[c + 1, r, k])
                else
                    a0 = real(A[r, c, k]); a1 = real(A[r, c + 1, k])
                    b0 = real(B[r, c, k]); b1 = real(B[r, c + 1, k])
                end
                value += si == 1 ? Float64(a0 * b0) :
                         (si == 2 ? Float64(a0 * b1) : Float64(a1 * b0))
            end
        else
            mu = means[comp, k]
            if si == 4
                for c in 1:m, r in 1:nr
                    d = Float64(_ka_uq_dcs(A, B, r, c, k, nr, m, transposed)) - mu
                    value += d * d
                end
            else
                odr = 0
                odc = 0
                oi = 4
                for ring in 1:UQ_MAX_OFFSET, dc in 0:UQ_MAX_OFFSET,
                    dr in -UQ_MAX_OFFSET:UQ_MAX_OFFSET
                    (dc == 0 && dr <= 0) && continue
                    max(abs(dr), abs(dc)) == ring || continue
                    oi += 1
                    if oi == si
                        odr = dr
                        odc = dc
                    end
                end
                dr, dc = odr, odc
            for c in max(1, 1 - dc):min(m, m - dc),
                r in max(1, 1 - dr):min(nr, nr - dr)
                d1 = Float64(_ka_uq_dcs(A, B, r, c, k, nr, m, transposed)) - mu
                d2 = Float64(_ka_uq_dcs(A, B, r + dr, c + dc, k, nr, m,
                                        transposed)) - mu
                        value += d1 * d2
                    end
                end
            end
        q = job0 + k
        stats[comp, si, q] = (add ? stats[comp, si, q] : 0.0) + value
    end
end

# Per-window means over the valid pixels — the reduction half of the CPU
# `load_windows!`, one work-item per window, with the exact accumulation order
# (`i` inner, `j` outer) so the means are bit-identical to the CPU path.
@kernel function _ka_window_means!(meanA, meanB, @Const(imgA), @Const(imgB),
                                   @Const(origins), @Const(mask), hasmask, wr, wc)
    k = @index(Global)
    T = eltype(meanA)
    @inbounds begin
        rs = origins[k, 1]
        cs = origins[k, 2]
        sA = zero(T)
        sB = zero(T)
        n = 0
        for j in 1:wc, i in 1:wr
            (hasmask && mask[rs + i - 1, cs + j - 1]) && continue
            sA += T(imgA[rs + i - 1, cs + j - 1])
            sB += T(imgB[rs + i - 1, cs + j - 1])
            n += 1
        end
        meanA[k] = n > 0 ? sA / n : zero(T)
        meanB[k] = n > 0 ? sB / n : zero(T)
    end
end

# Gather each window into the (padded) complex batch, mean-subtracted and
# apodized — the elementwise half of `load_windows!`. One work-item per window
# *pixel* (`ndrange = (wr, wc, nbatch)`) so adjacent work-items touch adjacent
# addresses (coalesced), unlike a work-item-per-window loop whose accesses sit
# a full plane apart. The padding region of each slice must already be zero.
@kernel function _ka_gather!(CA, CB, @Const(imgA), @Const(imgB), @Const(origins),
                             @Const(apod), @Const(meanA), @Const(meanB),
                             @Const(mask), hasmask)
    i, j, k = @index(Global, NTuple)
    T = eltype(apod)
    @inbounds begin
        r = origins[k, 1] + i - 1
        c = origins[k, 2] + j - 1
        if hasmask && mask[r, c]
            CA[i, j, k] = 0
            CB[i, j, k] = 0
        else
            a = apod[i, j]
            CA[i, j, k] = a * (T(imgA[r, c]) - meanA[k])
            CB[i, j, k] = a * (T(imgB[r, c]) - meanB[k])
        end
    end
end

# Cross-power spectrum in place: conj(F{A}) .* F{B}.
@kernel function _ka_crosspower!(CA, @Const(CB))
    I = @index(Global)
    @inbounds CA[I] = conj(CA[I]) * CB[I]
end

# fftshift + magnitude (+ overlap gain when padded) into the real plane batch.
# The output is *batch-major* (`Rt[k, ip, jp]`): the analysis kernel below runs
# one work-item per window, so putting the window index first makes its plane
# scans coalesce across the wavefront (for a fixed (i, j) the wave reads
# consecutive addresses). Writing the permuted layout here costs nothing extra
# — one side of the shift is strided either way.
@kernel function _ka_shiftgain!(Rt, @Const(CA), @Const(gain), padded, sr, sc, nr, nc)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ip = mod1(i + sr, nr)
        jp = mod1(j + sc, nc)
        val = abs(CA[i, j, k])
        padded && (val *= gain[ip, jp])
        Rt[k, ip, jp] = val
    end
end

# Ensemble variant of `_ka_shiftgain!`: add this pair's planes into the
# batch-major cross-pair accumulator at row offset `k0` instead of overwriting
# a scratch batch. Each work-item owns one (window, pixel) address, so the
# unsynchronized `+=` is race-free.
@kernel function _ka_shiftgain_accum!(Racc, @Const(CA), @Const(gain), padded, sr, sc,
                                      nr, nc, k0)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ip = mod1(i + sr, nr)
        jp = mod1(j + sc, nc)
        val = abs(CA[i, j, k])
        padded && (val *= gain[ip, jp])
        Racc[k0 + k, ip, jp] += val
    end
end

# ---------------------------------------------------------------------------
# Device-side symmetric image deformation (Phase 3): one work-item per output
# pixel evaluates the predictor displacement — bilinear over the coarse vector
# grid with flat extrapolation, the Gridded(Linear())/Flat() semantics of the
# CPU path — and resamples both images with cubic B-splines at ∓ half the
# displacement. The expensive prefilter stays on the host; only the padded
# coefficient arrays come to the device. Matches `deform_images` to
# floating-point round-off (not bitwise: evaluation order differs from
# Interpolations.jl).

# Flat-extrapolated linear index/weight split along one axis of the coarse
# predictor grid. Knots are the uniformly spaced window centers `q0 +
# (i-1)*qstep`; clamping the fractional index reproduces Flat() outside the
# knot span.
@inline function _pk_lin_axis(q0::T, qstep::T, n::Int, q::T) where {T}
    n == 1 && return 1, zero(T)
    t = clamp((q - q0) / qstep, zero(T), T(n - 1))
    i = unsafe_trunc(Int, floor(t)) + 1   # guarded: t ∈ [0, n-1]
    i > n - 1 && (i = n - 1)
    return i, t - T(i - 1)
end

# Uniform cubic B-spline basis weights at fractional offset δ ∈ [0, 1].
@inline _pk_cubw(δ::T) where {T} =
    ((1 - δ)^3 / 6, (4 - 6δ^2 + 3δ^3) / 6, (1 + 3δ + 3δ^2 - 3δ^3) / 6, δ^3 / 6)

# Cubic B-spline sample from the padded prefiltered coefficients. `C` is the
# `parent` of the interpolant's offset-axed (0:n+1) coefficient array, so
# logical index k lives at parent index k+1; zero outside the image domain,
# exactly like the CPU path's `extrapolate(…, 0)`. The support of a sample at
# `yy ∈ [1, nr]` is logical `iy-1:iy+2` with `iy ≤ nr-1`, i.e. parent
# `iy:iy+3 ⊆ 1:nr+2` — always in bounds. Verified against Interpolations.jl
# to ~9e-16.
@inline function _pk_cubic(C, yy::T, xx::T, nr, nc) where {T}
    (yy < 1 || yy > nr || xx < 1 || xx > nc) && return zero(T)
    iy = unsafe_trunc(Int, floor(yy))     # guarded: yy ∈ [1, nr]
    iy > nr - 1 && (iy = nr - 1)          # boundary: weights at δ = 1 are exact
    ix = unsafe_trunc(Int, floor(xx))
    ix > nc - 1 && (ix = nc - 1)
    wy = _pk_cubw(yy - T(iy))
    wx = _pk_cubw(xx - T(ix))
    s = zero(T)
    @inbounds for n in 0:3
        col = ix + n                      # parent index of logical ix-1+n
        s += wx[n + 1] * (wy[1] * C[iy, col] + wy[2] * C[iy + 1, col] +
                          wy[3] * C[iy + 2, col] + wy[4] * C[iy + 3, col])
    end
    return s
end

@kernel function _ka_deform!(warpA, warpB, @Const(CA), @Const(CB),
                             @Const(pu), @Const(pv), y0, ysp, x0, xsp,
                             gny, gnx, nr, nc)
    r, c = @index(Global, NTuple)
    T = eltype(warpA)
    @inbounds begin
        iy, ly = _pk_lin_axis(y0, ysp, gny, T(r))
        ix, lx = _pk_lin_axis(x0, xsp, gnx, T(c))
        iy2 = gny == 1 ? iy : iy + 1
        ix2 = gnx == 1 ? ix : ix + 1
        a = pu[iy, ix] + lx * (pu[iy, ix2] - pu[iy, ix])
        b = pu[iy2, ix] + lx * (pu[iy2, ix2] - pu[iy2, ix])
        du2 = (a + ly * (b - a)) / 2
        a = pv[iy, ix] + lx * (pv[iy, ix2] - pv[iy, ix])
        b = pv[iy2, ix] + lx * (pv[iy2, ix2] - pv[iy2, ix])
        dv2 = (a + ly * (b - a)) / 2
        warpA[r, c] = _pk_cubic(CA, T(r) - dv2, T(c) - du2, nr, nc)
        warpB[r, c] = _pk_cubic(CB, T(r) + dv2, T(c) + du2, nr, nc)
    end
end

# Per-call deformation context for the KA-family backends: the prefiltered
# padded B-spline coefficients staged where `_ka_deform!` runs, the resident
# warp output buffers, and the (lazily resized) coarse predictor-grid arrays.
# Staging happens once per `run_piv` call (`_ka_deform_context` below); every
# deformation sweep of every pass then reuses the resident coefficients, and
# on a device backend the warped images never visit the host — the
# correlation engines use them in place.
mutable struct _KADeformContext{T,D,M<:AbstractMatrix{T}}
    dev::D
    coefA::M
    coefB::M
    warpA::M
    warpB::M
    pu::M
    pv::M
end

# Build (or draw from the workspace pool) a deform context on KA backend `dev`
# and stage this call's coefficients into it. The pooled buffers are pure
# scratch — fully rewritten here and by each sweep — so cross-call reuse via
# the workspace leaves results identical.
function _ka_deform_context(dev, workspace, key, itpA, itpB,
                            imgsize::Dims{2}, ::Type{T}) where {T}
    nr, nc = imgsize
    make = () -> _KADeformContext(dev,
        KernelAbstractions.allocate(dev, T, nr + 2, nc + 2),
        KernelAbstractions.allocate(dev, T, nr + 2, nc + 2),
        KernelAbstractions.allocate(dev, T, nr, nc),
        KernelAbstractions.allocate(dev, T, nr, nc),
        KernelAbstractions.allocate(dev, T, 0, 0),
        KernelAbstractions.allocate(dev, T, 0, 0))
    ctx = workspace === nothing ? make() : pooled_engines(make, workspace, key, 1)[1]
    # The padded parent of the offset-axed coefficient array — exactly what
    # `_pk_cubic` indexes (logical k at parent k+1).
    copyto!(ctx.coefA, parent(itpA.itp.coefs))
    copyto!(ctx.coefB, parent(itpB.itp.coefs))
    return ctx
end

# One deformation sweep against a staged context: evaluate the pass-grid
# predictor values on the host (tiny), upload the coarse predictor grids
# (vector-grid sized, the only per-sweep transfer), and launch `_ka_deform!`
# into the context's resident warp buffers.
function _ka_apply_predictor_ctx(ctx::_KADeformContext{T}, predictor,
                                 x::AbstractVector, y::AbstractVector) where {T}
    u, v = predictor_node_values(predictor, x, y, T)
    py, px = predictor.y, predictor.x
    gny, gnx = length(py), length(px)
    if size(ctx.pu) != (gny, gnx)
        ctx.pu = KernelAbstractions.allocate(ctx.dev, T, gny, gnx)
        ctx.pv = KernelAbstractions.allocate(ctx.dev, T, gny, gnx)
    end
    copyto!(ctx.pu, convert(Matrix{T}, predictor.u))
    copyto!(ctx.pv, convert(Matrix{T}, predictor.v))
    nr, nc = size(ctx.warpA)
    y0 = T(first(py))
    x0 = T(first(px))
    ysp = gny > 1 ? T(py[2] - py[1]) : one(T)
    xsp = gnx > 1 ? T(px[2] - px[1]) : one(T)
    _ka_deform!(ctx.dev)(ctx.warpA, ctx.warpB, ctx.coefA, ctx.coefB,
                         ctx.pu, ctx.pv, y0, ysp, x0, xsp, gny, gnx, nr, nc;
                         ndrange = (nr, nc))
    KernelAbstractions.synchronize(ctx.dev)
    return ctx.warpA, ctx.warpB, u, v
end

# ---------------------------------------------------------------------------
# Device-side plane analysis: peak finding, subpixel refinement, peak ratio,
# correlation moment, and alternative-peak refinement — the batched analogue
# of `analyze_plane!`. These are line-for-line ports of the CPU routines in
# `correlators.jl`/`quality.jl` (same scan order, same T/Float64 arithmetic),
# operating on window `k` of the batch-major plane batch `Rt[k, i, j]`, so
# on-device analysis matches the host within `log`-intrinsic round-off
# (identical on the KA-CPU backend). Keeping analysis on the device means only
# a handful of scalars per window cross back to the host, not whole planes.

# `find_peaks_exclusion!` port. The exclusion radius is fixed at 2, the
# `find_peaks!` default `analyze_plane!` relies on.
@inline function _pk_find_exclusion!(vals, locs, Rt, k, K, nr, nc)
    T = eltype(Rt)
    found = 0
    @inbounds for _ in 1:K
        best = T(-Inf)
        bi = 0
        bj = 0
        for j in 1:nc, i in 1:nr
            excluded = false
            for p in 1:found
                if abs(i - locs[1, p, k]) <= 2 && abs(j - locs[2, p, k]) <= 2
                    excluded = true
                    break
                end
            end
            excluded && continue
            if Rt[k, i, j] > best
                best = Rt[k, i, j]
                bi = i
                bj = j
            end
        end
        bi == 0 && break                  # everything excluded (degenerate plane)
        found > 0 && best <= 0 && break   # secondary peaks must be positive
        found += 1
        vals[found, k] = best
        # `% Int32` wraps instead of convert-checking: plane indices always
        # fit, and the checked conversion's throw path would force a GPU
        # malloc hostcall that serializes the whole kernel.
        locs[1, found, k] = bi % Int32
        locs[2, found, k] = bj % Int32
    end
    return found
end

# `is_local_maximum` port (strict/non-strict split by the *plane's*
# column-major order, `r + (c - 1) * nr`, independent of the batch layout).
@inline function _pk_localmax(Rt, k, r, c, nr, nc)
    @inbounds begin
        I0 = Rt[k, r, c]
        isnan(I0) && return false
        lin0 = r + (c - 1) * nr
        for jj in max(1, c - 1):min(nc, c + 1), ii in max(1, r - 1):min(nr, r + 1)
            (ii == r && jj == c) && continue
            In = Rt[k, ii, jj]
            if ii + (jj - 1) * nr < lin0
                In > I0 && return false
            else
                In >= I0 && return false
            end
        end
    end
    return true
end

# `find_peaks_regionalmax!` port.
@inline function _pk_find_regionalmax!(vals, locs, Rt, k, K, nr, nc)
    nfound = 0
    @inbounds for j in 1:nc, i in 1:nr
        _pk_localmax(Rt, k, i, j, nr, nc) || continue
        val = Rt[k, i, j]
        (nfound < K || val > vals[nfound, k]) || continue

        pos = nfound
        while pos >= 1 && val > vals[pos, k]
            pos -= 1
        end
        ins = pos + 1
        ins <= K || continue
        nfound < K && (nfound += 1)
        for m in nfound:-1:(ins + 1)
            vals[m, k] = vals[m - 1, k]
            locs[1, m, k] = locs[1, m - 1, k]
            locs[2, m, k] = locs[2, m - 1, k]
        end
        vals[ins, k] = val
        locs[1, ins, k] = i % Int32   # wrapping store, see _pk_find_exclusion!
        locs[2, ins, k] = j % Int32
    end
    nfound == 0 && return 0
    found = 1
    @inbounds while found < nfound && vals[found + 1, k] > 0
        found += 1
    end
    return found
end

# `subpixel_gauss3` port.
@inline function _pk_gauss3(Rt, k, i, j, nr, nc)
    T = eltype(Rt)
    di = zero(T)
    dj = zero(T)
    @inbounds begin
        I0 = Rt[k, i, j]
        if 1 < i < nr
            Im, Ip = Rt[k, i - 1, j], Rt[k, i + 1, j]
            if Im > 0 && I0 > 0 && Ip > 0
                denom = log(Im) - 2log(I0) + log(Ip)
                di = denom != 0 ? (log(Im) - log(Ip)) / (2denom) : zero(T)
            end
        end
        if 1 < j < nc
            Im, Ip = Rt[k, i, j - 1], Rt[k, i, j + 1]
            if Im > 0 && I0 > 0 && Ip > 0
                denom = log(Im) - 2log(I0) + log(Ip)
                dj = denom != 0 ? (log(Im) - log(Ip)) / (2denom) : zero(T)
            end
        end
    end
    return (i + di, j + dj)
end

# `subpixel_gauss9` port (falls back to the 3-point fit exactly like the CPU).
@inline function _pk_gauss9(Rt, k, pr, pc, nr, nc)
    T = eltype(Rt)
    (1 < pr < nr && 1 < pc < nc) || return _pk_gauss3(Rt, k, pr, pc, nr, nc)
    sL = zero(T); sxL = zero(T); syL = zero(T)
    sxxL = zero(T); syyL = zero(T); sxyL = zero(T)
    @inbounds for dj in -1:1, di in -1:1
        val = Rt[k, pr + di, pc + dj]
        val > 0 || return _pk_gauss3(Rt, k, pr, pc, nr, nc)
        L = log(val)
        sL += L
        sxL += dj * L
        syL += di * L
        sxxL += dj * dj * L
        syyL += di * di * L
        sxyL += di * dj * L
    end
    a1 = sxL / 6
    a2 = syL / 6
    a3 = sxxL / 2 - sL / 3
    a4 = sxyL / 4
    a5 = syyL / 2 - sL / 3
    D = 4 * a3 * a5 - a4^2
    (a3 < 0 && D > 0) || return _pk_gauss3(Rt, k, pr, pc, nr, nc)  # not a maximum
    dx = (a4 * a2 - 2 * a5 * a1) / D
    dy = (a4 * a1 - 2 * a3 * a2) / D
    (abs(dx) <= 1 && abs(dy) <= 1) || return _pk_gauss3(Rt, k, pr, pc, nr, nc)
    return (pr + dy, pc + dx)
end

# `calculate_correlation_moment` port over the fixed 3×3 neighborhood
# (`neighborhood_size = 3`, the `analyze_plane!` default), Float64 sums like
# the CPU's deliberate Float64 island, converted to T on return.
@inline function _pk_moment(Rt, k, pr, pc, nr, nc)
    T = eltype(Rt)
    (isfinite(pr) && isfinite(pc)) || return T(NaN)
    # `round(Int, x)` carries an InexactError throw path, which on the GPU
    # forces a malloc hostcall; the values are finite (guarded above) and
    # plane-sized, so the unchecked truncation of the rounded value is exact.
    r0 = unsafe_trunc(Int, round(pr))
    c0 = unsafe_trunc(Int, round(pc))
    rlo = max(1, r0 - 1); rhi = min(nr, r0 + 1)
    clo = max(1, c0 - 1); chi = min(nc, c0 + 1)
    (rlo > rhi || clo > chi) && return T(NaN)
    sumC = 0.0
    sum_dx2 = 0.0
    sum_dy2 = 0.0
    @inbounds for c in clo:chi, r in rlo:rhi
        w = max(Float64(Rt[k, r, c]), 0.0)
        sumC += w
        sum_dx2 += (c - pc)^2 * w
        sum_dy2 += (r - pr)^2 * w
    end
    sumC < CORR_MOMENT_EPSILON && return T(NaN)
    return T(sqrt((sum_dx2 + sum_dy2) / sumC))
end

# Analyze every plane of the batch, one work-item per window. Results are
# packed per window into `out` (a `(5 + 2*(K-1), batch)` matrix, K =
# max(n_peaks, 2)) so one small copy returns everything to the host:
#   row 1..4  du, dv, peak ratio, correlation moment
#   row 5     number of peaks found (integral, stored in T)
#   rows 6..5+(K-1)      alt-peak du for peaks 2..K (gauss3-refined)
#   rows 6+(K-1)..5+2(K-1) alt-peak dv for peaks 2..K
# `vals` (K, batch) and `locs` (2, K, batch) are per-window device scratch.
@kernel function _ka_analyze!(out, vals, locs, @Const(Rt), use_regionalmax,
                              use_gauss9, npeaks, nr, nc)
    k = @index(Global)
    T = eltype(Rt)
    K = size(vals, 1)
    @inbounds begin
        found = use_regionalmax ? _pk_find_regionalmax!(vals, locs, Rt, k, K, nr, nc) :
                                  _pk_find_exclusion!(vals, locs, Rt, k, K, nr, nc)
        out[5, k] = T(found)
        if found == 0
            # Degenerate all-NaN plane; the CPU path reads uninitialized
            # scratch here, so any well-defined value is acceptable.
            out[1, k] = T(NaN)
            out[2, k] = T(NaN)
            out[3, k] = T(NaN)
            out[4, k] = T(NaN)
        else
            cr = nr ÷ 2 + 1
            cc = nc ÷ 2 + 1
            pr = Int(locs[1, 1, k])
            pc = Int(locs[2, 1, k])
            ri, rj = use_gauss9 ? _pk_gauss9(Rt, k, pr, pc, nr, nc) :
                                  _pk_gauss3(Rt, k, pr, pc, nr, nc)
            out[1, k] = rj - cc
            out[2, k] = ri - cr
            v1 = vals[1, k]
            out[3, k] = found >= 2 ? v1 / vals[2, k] : (v1 > 0 ? T(Inf) : T(NaN))
            out[4, k] = _pk_moment(Rt, k, ri, rj, nr, nc)
            # Alternatives use the cheap 3-point fit regardless of the
            # primary's subpixel method — they are fallback candidates.
            for m in 2:min(found, npeaks)
                ar, ac = _pk_gauss3(Rt, k, Int(locs[1, m, k]), Int(locs[2, m, k]), nr, nc)
                out[5 + (m - 1), k] = ac - cc
                out[5 + (K - 1) + (m - 1), k] = ar - cr
            end
        end
    end
end

# ---------------------------------------------------------------------------
# The built-in `backend = :ka` engine: the kernels above on KernelAbstractions'
# CPU backend. The plane-level numerics are matched to the CPU FFTW path
# exactly (the same apodization window and overlap-gain plane are reused), so
# results differ from `backend = :cpu` only by FFT/reduction round-off; on a
# GPU device only a handful of scalars per window return to the host, never
# whole correlation planes — here the same kernels guard that shared code in
# CI without hardware.

struct _KABackend <: _AbstractHammerheadBackend end

_resolve_backend(::Val{:ka}) = _KABackend()
_supports_fft(::_KABackend) = true
_supports_batched_fft(::_KABackend) = true
_supports_fp64(::_KABackend) = true

# Device engines run the whole window grid as one logical batch (tiled
# internally into memory-bounded sub-batches) rather than fanning out across
# host threads.
_engine_nchunks(::_KABackend, ::Int) = 1

# Windows processed per device sub-batch. Bounds the FFT-buffer footprint:
# materializing every overlapping window of a large image at once can exceed
# device memory.
const _KA_BATCH = 512

_check_backend_params(b::_KABackend, passes) =
    _ka_scope_check(passes, :ka, _supports_fp64(b))

# Per-chunk (here: single) correlation engine. Buffers and FFT plans are cached
# and reused across a pass's deformation sweeps; they are rebuilt only when the
# sub-batch size changes.
mutable struct _KACorrelationEngine{T,KB}
    ka::KB
    wsize::NTuple{2,Int}
    fft_size::NTuple{2,Int}
    padded::Bool
    kpk::Int                        # peaks located per window, max(n_peaks, 2)
    apod::Matrix{T}
    gain::Matrix{T}                 # fftshifted overlap gain; empty if unpadded
    bs::Int
    CA::Array{Complex{T},3}
    CB::Array{Complex{T},3}
    Rt::Array{T,3}                  # (bs+1, nr, nc) batch-major plane batch
    meanA::Vector{T}                # per-window means (device reduction)
    meanB::Vector{T}
    origins::Matrix{Int}
    vals::Matrix{T}                 # (kpk, bs) peak-finder scratch
    locs::Array{Int32,3}            # (2, kpk, bs) peak-finder scratch
    out::Matrix{T}                  # (5 + 2*(kpk-1), bs) packed analysis output
    uqstats::Array{Float64,3}       # (2, UQ_NSTATS, bs), device-UQ scalars
    uqmeans::Matrix{Float64}        # (2, bs), smoothed dC means
    fwd::Any
    bwd::Any
end

_correlation_apod(e::_KACorrelationEngine) = e.apod

function _make_ka_engine(params::PIVParameters, ::Type{T}) where {T}
    # Steal the exact apodization window and overlap-gain plane from a CPU
    # correlator so those factors are bit-identical to the FFTW path.
    cpu = make_correlator(params, T)
    apod = Matrix{T}(cpu.apod)
    fft_size = size(cpu.R)
    padded = !isempty(cpu.gain)
    gain = padded ? Matrix{T}(cpu.gain) : Matrix{T}(undef, 0, 0)
    return _KACorrelationEngine{T,typeof(CPU())}(
        CPU(), params.window_size, fft_size, padded, max(params.n_peaks, 2),
        apod, gain, 0,
        Array{Complex{T},3}(undef, 0, 0, 0), Array{Complex{T},3}(undef, 0, 0, 0),
        Array{T,3}(undef, 0, 0, 0), Vector{T}(undef, 0), Vector{T}(undef, 0),
        Matrix{Int}(undef, 0, 2),
        Matrix{T}(undef, 0, 0), Array{Int32,3}(undef, 2, 0, 0),
        Matrix{T}(undef, 0, 0), Array{Float64,3}(undef, 0, 0, 0),
        Matrix{Float64}(undef, 0, 0), nothing, nothing)
end

# Engines are pooled per window configuration (everything baked in at
# construction: sizes, padding/apodization planes, and the kpk-dependent
# scratch shapes; peak finder and subpixel method are per-call kernel
# arguments, not engine state).
piv_correlation_engines(::_KABackend, workspace, params::PIVParameters,
                        ::Type{T}, nchunks::Int) where {T} =
    pooled_engines(() -> _make_ka_engine(params, T), workspace,
                   (:ka, T, params.window_size, params.padding,
                    params.apodization, max(params.n_peaks, 2)), nchunks)

function _ensure_buffers!(engine::_KACorrelationEngine{T}, bs::Int) where {T}
    if engine.bs != bs
        nr, nc = engine.fft_size
        engine.CA = zeros(Complex{T}, nr, nc, bs)
        engine.CB = zeros(Complex{T}, nr, nc, bs)
        engine.Rt = zeros(T, bs + 1, nr, nc)   # +1: GPU channel-conflict pad, see AMDGPU ext
        engine.meanA = Vector{T}(undef, bs)
        engine.meanB = Vector{T}(undef, bs)
        engine.origins = Matrix{Int}(undef, bs, 2)
        engine.vals = Matrix{T}(undef, engine.kpk, bs)
        engine.locs = Array{Int32,3}(undef, 2, engine.kpk, bs)
        engine.out = Matrix{T}(undef, 5 + 2 * (engine.kpk - 1), bs)
        engine.uqstats = zeros(Float64, 2, UQ_NSTATS, bs)
        engine.uqmeans = zeros(Float64, 2, bs)
        engine.fwd = plan_fft!(engine.CA, (1, 2))
        engine.bwd = inv(engine.fwd)
        engine.bs = bs
    end
    return engine
end

function process_windows!(u, v, peak_ratio, correlation_moment, alt_u, alt_v,
                          uncertainty_u, uncertainty_v, jobs,
                          imgA::AbstractMatrix, imgB::AbstractMatrix, params::PIVParameters,
                          engine::_KACorrelationEngine{T},
                          mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                          planes = nothing) where {T}
    planes === nothing ||
        throw(ArgumentError("backend :ka does not support correlation-plane storage yet; " *
                            "use backend = :cpu"))
    jobvec = jobs isa AbstractVector ? jobs : collect(jobs)
    njobs = length(jobvec)
    njobs == 0 && return nothing

    wr, wc = engine.wsize
    nr, nc = engine.fft_size
    sr, sc = nr ÷ 2, nc ÷ 2
    bs = min(_KA_BATCH, njobs)
    _ensure_buffers!(engine, bs)
    ka = engine.ka
    hasmask = mask !== nothing
    themask = hasmask ? mask : similar(imgA, Bool, 0, 0)
    gainarg = engine.padded ? engine.gain : engine.apod   # dummy when unpadded (unread)
    use_regionalmax = params.peak_finder === :regionalmax
    use_gauss9 = params.subpixel_method === :gauss9
    nalt = engine.kpk - 1

    for start in 1:bs:njobs
        nreal = min(bs, njobs - start + 1)
        @inbounds for m in 1:nreal
            job = jobvec[start + m - 1]
            engine.origins[m, 1] = job[3]   # rs
            engine.origins[m, 2] = job[4]   # cs
        end
        fill!(engine.CA, 0)
        fill!(engine.CB, 0)
        _ka_window_means!(ka)(engine.meanA, engine.meanB, imgA, imgB, engine.origins,
                              themask, hasmask, wr, wc; ndrange = nreal)
        KernelAbstractions.synchronize(ka)
        _ka_gather!(ka)(engine.CA, engine.CB, imgA, imgB, engine.origins, engine.apod,
                        engine.meanA, engine.meanB, themask, hasmask;
                        ndrange = (wr, wc, nreal))
        if uncertainty_u !== nothing
            _ka_uq_means!(ka)(engine.uqmeans, engine.CA, engine.CB, wr, wc;
                              ndrange = (2, nreal))
            _ka_uq_stats!(ka)(engine.uqstats, engine.uqmeans, engine.CA, engine.CB,
                              wr, wc, nreal, 0, false;
                              ndrange = (2, UQ_NSTATS, nreal))
        end
        KernelAbstractions.synchronize(ka)
        mul!(engine.CA, engine.fwd, engine.CA)
        mul!(engine.CB, engine.fwd, engine.CB)
        _ka_crosspower!(ka)(engine.CA, engine.CB; ndrange = length(engine.CA))
        KernelAbstractions.synchronize(ka)
        mul!(engine.CA, engine.bwd, engine.CA)
        _ka_shiftgain!(ka)(engine.Rt, engine.CA, gainarg, engine.padded, sr, sc, nr, nc;
                           ndrange = (nr, nc, bs))
        _ka_analyze!(ka)(engine.out, engine.vals, engine.locs, engine.Rt,
                         use_regionalmax, use_gauss9, params.n_peaks, nr, nc;
                         ndrange = nreal)
        KernelAbstractions.synchronize(ka)

        # Scatter the packed per-window outputs into the vector grids (see the
        # `_ka_analyze!` row layout).
        for m in 1:nreal
            job = jobvec[start + m - 1]
            gi, gj = job[1], job[2]
            u[gi, gj] = engine.out[1, m]
            v[gi, gj] = engine.out[2, m]
            peak_ratio[gi, gj] = engine.out[3, m]
            correlation_moment[gi, gj] = engine.out[4, m]
            if uncertainty_u !== nothing
                uncertainty_u[gi, gj] = finalize_uncertainty(T, view(engine.uqstats, 1, :, m))
                uncertainty_v[gi, gj] = finalize_uncertainty(T, view(engine.uqstats, 2, :, m))
            end
            if alt_u !== nothing
                found = Int(engine.out[5, m])   # small integer, exact in T
                for mm in 2:min(found, params.n_peaks)
                    alt_u[gi, gj, mm - 1] = engine.out[5 + (mm - 1), m]
                    alt_v[gi, gj, mm - 1] = engine.out[5 + nalt + (mm - 1), m]
                end
            end
        end
    end
    return nothing
end

function uncertainty_sweep!(uncertainty_u, uncertainty_v, jobs, imgA, imgB,
                            params::PIVParameters, apod, mask,
                            engine::_KACorrelationEngine{T}) where {T}
    jobvec = jobs isa AbstractVector ? jobs : collect(jobs)
    njobs = length(jobvec)
    njobs == 0 && return nothing
    wr, wc = engine.wsize
    bs = min(_KA_BATCH, njobs)
    _ensure_buffers!(engine, bs)
    hasmask = mask !== nothing
    themask = hasmask ? mask : similar(imgA, Bool, 0, 0)
    for start in 1:bs:njobs
        nreal = min(bs, njobs - start + 1)
        for m in 1:nreal
            job = jobvec[start + m - 1]
            engine.origins[m, 1] = job[3]
            engine.origins[m, 2] = job[4]
        end
        _ka_window_means!(engine.ka)(engine.meanA, engine.meanB, imgA, imgB,
            engine.origins, themask, hasmask, wr, wc; ndrange = nreal)
        _ka_gather!(engine.ka)(engine.CA, engine.CB, imgA, imgB, engine.origins,
            engine.apod, engine.meanA, engine.meanB, themask, hasmask;
            ndrange = (wr, wc, nreal))
        _ka_uq_means!(engine.ka)(engine.uqmeans, engine.CA, engine.CB, wr, wc;
                                 ndrange = (2, nreal))
        _ka_uq_stats!(engine.ka)(engine.uqstats, engine.uqmeans,
                                 engine.CA, engine.CB, wr, wc, nreal, 0, false;
                                 ndrange = (2, UQ_NSTATS, nreal))
        KernelAbstractions.synchronize(engine.ka)
        for m in 1:nreal
            gi, gj = jobvec[start + m - 1][1:2]
            uncertainty_u[gi, gj] = finalize_uncertainty(T, view(engine.uqstats, 1, :, m))
            uncertainty_v[gi, gj] = finalize_uncertainty(T, view(engine.uqstats, 2, :, m))
        end
    end
    return nothing
end

# Deformation on the KA backend: the prefiltered coefficient arrays already
# live where the kernel runs (host memory), so this is the pure proving tier
# for `_ka_deform!` and — via the `:ka` `_deform_context` below — for the
# staged-context path the device extensions run with device arrays. The
# predictor values at the pass-grid nodes are evaluated on the host exactly
# like the CPU path (a tiny Gridded(Linear()) job), so vector attribution is
# unchanged. Without a context (direct calls) the kernel reads the
# interpolants' coefficients in place.
function apply_predictor(::_KABackend, imgA::AbstractMatrix, imgB::AbstractMatrix,
                         itpA, itpB, predictor, x::AbstractVector, y::AbstractVector,
                         ::Type{T}; threaded::Bool = false,
                         warpA::Union{Nothing,Matrix{T}} = nothing,
                         warpB::Union{Nothing,Matrix{T}} = nothing,
                         ctx = nothing) where {T}
    ny, nx = length(y), length(x)
    predictor === nothing && return imgA, imgB, zeros(T, ny, nx), zeros(T, ny, nx)
    ctx === nothing || return _ka_apply_predictor_ctx(ctx, predictor, x, y)
    u, v = predictor_node_values(predictor, x, y, T)

    nr, nc = size(imgA)
    warpA === nothing && (warpA = Matrix{T}(undef, nr, nc))
    warpB === nothing && (warpB = Matrix{T}(undef, nr, nc))
    py, px = predictor.y, predictor.x
    gny, gnx = length(py), length(px)
    y0 = T(first(py))
    x0 = T(first(px))
    ysp = gny > 1 ? T(py[2] - py[1]) : one(T)
    xsp = gnx > 1 ? T(px[2] - px[1]) : one(T)
    ka = CPU()
    _ka_deform!(ka)(warpA, warpB, parent(itpA.itp.coefs), parent(itpB.itp.coefs),
                    predictor.u, predictor.v, y0, ysp, x0, xsp, gny, gnx, nr, nc;
                    ndrange = (nr, nc))
    KernelAbstractions.synchronize(ka)
    return warpA, warpB, u, v
end

# The :ka context runs the identical staged-context machinery the device
# extensions use — coefficient copy, pooled buffers, per-sweep predictor
# upload — on host arrays, guarding that shared path in CI without hardware.
_deform_context(::_KABackend, workspace, itpA, itpB,
                imgsize::Dims{2}, ::Type{T}) where {T} =
    _ka_deform_context(CPU(), workspace, (:ka_deform, T, imgsize),
                       itpA, itpB, imgsize, T)

# ---------------------------------------------------------------------------
# Ensemble (sum-of-correlation) support: the whole window grid's summed planes
# stay resident where the engine computes — for a device backend that means
# planes never return to the host, matching the plan's ensemble dataflow
# (image pair in, device-side accumulate, final vector grid out).

# Batch-major cross-pair plane accumulator, `Racc[j, i, j']` = window j's
# summed plane. Rows beyond `njobs` are stride padding (the leading dimension
# avoids a power-of-two byte stride, which funnels a GPU wave's writes into a
# single memory channel — see the `Rt` pad in `_ensure_buffers!`).
struct _KAPlaneAccumulator{A<:AbstractArray}
    Racc::A
    njobs::Int
end

struct _KAUQAccumulator{A<:AbstractArray}
    stats::A
end

_uncertainty_accumulator(engine::_KACorrelationEngine, ::Type, njobs::Int) =
    _KAUQAccumulator(zeros(Float64, 2, UQ_NSTATS, njobs))
_uncertainty_scratch(::_KACorrelationEngine, ::Type) = nothing

function _plane_accumulator(engine::_KACorrelationEngine{T}, params::PIVParameters,
                            ::Type{T}, njobs::Int) where {T}
    nr, nc = engine.fft_size
    # An odd leading dimension keeps the plane-to-plane byte stride off
    # power-of-two multiples (the memory-channel conflict the Rt pad avoids).
    ld = njobs + (iseven(njobs) ? 1 : 0)
    return _KAPlaneAccumulator(zeros(T, ld, nr, nc), njobs)
end

# Ensemble accumulation for one pair: identical staging to `process_windows!`
# up to the inverse FFT, then `_ka_shiftgain_accum!` adds the shifted planes
# straight into the cross-pair accumulator. Requires a contiguous job range
# (device engines always run the grid as a single chunk).
function accumulate_planes!(acc::_KAPlaneAccumulator, jobrange::AbstractUnitRange,
                            engine::_KACorrelationEngine{T},
                            imgA::AbstractMatrix, imgB::AbstractMatrix, jobs,
                            params::PIVParameters, mask,
                            uacc = nothing, uscratch = nothing) where {T}
    njobs = length(jobrange)
    njobs == 0 && return nothing

    wr, wc = engine.wsize
    nr, nc = engine.fft_size
    sr, sc = nr ÷ 2, nc ÷ 2
    bs = min(_KA_BATCH, njobs)
    _ensure_buffers!(engine, bs)
    ka = engine.ka
    hasmask = mask !== nothing
    themask = hasmask ? mask : similar(imgA, Bool, 0, 0)
    gainarg = engine.padded ? engine.gain : engine.apod   # dummy when unpadded (unread)

    for start in 1:bs:njobs
        nreal = min(bs, njobs - start + 1)
        @inbounds for m in 1:nreal
            job = jobs[jobrange[start + m - 1]]
            engine.origins[m, 1] = job[3]   # rs
            engine.origins[m, 2] = job[4]   # cs
        end
        fill!(engine.CA, 0)
        fill!(engine.CB, 0)
        _ka_window_means!(ka)(engine.meanA, engine.meanB, imgA, imgB, engine.origins,
                              themask, hasmask, wr, wc; ndrange = nreal)
        KernelAbstractions.synchronize(ka)
        _ka_gather!(ka)(engine.CA, engine.CB, imgA, imgB, engine.origins, engine.apod,
                        engine.meanA, engine.meanB, themask, hasmask;
                        ndrange = (wr, wc, nreal))
        if uacc !== nothing
            _ka_uq_means!(ka)(engine.uqmeans, engine.CA, engine.CB, wr, wc;
                              ndrange = (2, nreal))
            _ka_uq_stats!(ka)(uacc.stats, engine.uqmeans, engine.CA, engine.CB,
                              wr, wc, nreal, first(jobrange) + start - 2, true;
                              ndrange = (2, UQ_NSTATS, nreal))
        end
        KernelAbstractions.synchronize(ka)
        mul!(engine.CA, engine.fwd, engine.CA)
        mul!(engine.CB, engine.fwd, engine.CB)
        _ka_crosspower!(ka)(engine.CA, engine.CB; ndrange = length(engine.CA))
        KernelAbstractions.synchronize(ka)
        mul!(engine.CA, engine.bwd, engine.CA)
        # Only the nreal live slices accumulate — the tail of the last tile
        # holds stale spectra that must not pollute the sums.
        _ka_shiftgain_accum!(ka)(acc.Racc, engine.CA, gainarg, engine.padded, sr, sc,
                                 nr, nc, first(jobrange) + start - 2;
                                 ndrange = (nr, nc, nreal))
        KernelAbstractions.synchronize(ka)
    end
    return nothing
end

# Ensemble finalize: run the device analysis kernel over the summed planes in
# tiles and scatter the packed scalars into the vector grids — the ensemble
# analogue of the `process_windows!` scatter, except residuals *add to* the
# shared predictor held in `u`/`v` and alternatives are predictor-relative,
# exactly like the host `ensemble_analyze!` loop.
function ensemble_analyze!(acc::_KAPlaneAccumulator, engine::_KACorrelationEngine{T},
                           u, v, peak_ratio, correlation_moment,
                           uncertainty_u, uncertainty_v, planes, alt_u, alt_v,
                           jobs, params::PIVParameters, uacc) where {T}
    planes === nothing ||
        throw(ArgumentError("KA-family backends do not support correlation-plane " *
                            "storage yet; use backend = :cpu"))
    njobs = acc.njobs
    njobs == 0 && return nothing
    nr, nc = engine.fft_size
    bs = min(_KA_BATCH, njobs)
    _ensure_buffers!(engine, bs)
    ka = engine.ka
    use_regionalmax = params.peak_finder === :regionalmax
    use_gauss9 = params.subpixel_method === :gauss9
    nalt = engine.kpk - 1
    for start in 1:bs:njobs
        nreal = min(bs, njobs - start + 1)
        Rv = view(acc.Racc, start:(start + nreal - 1), :, :)
        _ka_analyze!(ka)(engine.out, engine.vals, engine.locs, Rv,
                         use_regionalmax, use_gauss9, params.n_peaks, nr, nc;
                         ndrange = nreal)
        KernelAbstractions.synchronize(ka)
        for m in 1:nreal
            gi, gj, _, _ = jobs[start + m - 1]
            if alt_u !== nothing
                found = Int(engine.out[5, m])   # small integer, exact in T
                for mm in 2:min(found, params.n_peaks)
                    # Total alternative displacement = shared predictor + residual.
                    alt_u[gi, gj, mm - 1] = u[gi, gj] + engine.out[5 + (mm - 1), m]
                    alt_v[gi, gj, mm - 1] = v[gi, gj] + engine.out[5 + nalt + (mm - 1), m]
                end
            end
            u[gi, gj] += engine.out[1, m]
            v[gi, gj] += engine.out[2, m]
            peak_ratio[gi, gj] = engine.out[3, m]
            correlation_moment[gi, gj] = engine.out[4, m]
            if uacc !== nothing
                j = start + m - 1
                uncertainty_u[gi, gj] = finalize_uncertainty(T, view(uacc.stats, 1, :, j))
                uncertainty_v[gi, gj] = finalize_uncertainty(T, view(uacc.stats, 2, :, j))
            end
        end
    end
    return nothing
end
