# Portable KernelAbstractions correlation kernels, shared by the device
# extensions (KA-CPU, AMDGPU, and future CUDA). This file is `include`d into
# each extension module after `using KernelAbstractions`, so the kernels are
# defined once per backend module against that module's array types. The math
# mirrors the CPU FFTW correlator (`load_windows!`, `spectrum!`,
# `fftshift_abs!`, overlap gain) so planes match `backend = :cpu` within FFT
# round-off, and the plane-analysis kernel (`_ka_analyze!`) ports
# `analyze_plane!` so peak outputs match within `log`-intrinsic round-off.

# Shared option-scope guard for the KA-derived backends (:ka, :amdgpu, and a
# future :cuda). The initial device scope is the plan's Phase 2 slice; the
# excluded options stay CPU-first (or land in a later phase).
function _ka_scope_check(passes, name::Symbol)
    for p in passes
        p.correlation_method === :cross ||
            throw(ArgumentError("backend :$name supports correlation_method = :cross only " *
                                "(got :$(p.correlation_method)); use backend = :cpu"))
        p.subpixel_method in (:gauss3, :gauss9) ||
            throw(ArgumentError("backend :$name supports subpixel_method :gauss3 or :gauss9 " *
                                "only (got :$(p.subpixel_method)); use backend = :cpu"))
        p.uncertainty &&
            throw(ArgumentError("backend :$name does not support uncertainty quantification " *
                                "yet; run UQ on backend = :cpu"))
        p.keep_correlation_planes &&
            throw(ArgumentError("backend :$name does not support keep_correlation_planes yet; " *
                                "use backend = :cpu"))
    end
    return nothing
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
    sumC < Hammerhead.CORR_MOMENT_EPSILON && return T(NaN)
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
