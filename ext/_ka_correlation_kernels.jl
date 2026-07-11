# Portable KernelAbstractions correlation kernels, shared by the device
# extensions (KA-CPU, AMDGPU, and future CUDA). This file is `include`d into
# each extension module after `using KernelAbstractions`, so the kernels are
# defined once per backend module against that module's array types. The math
# mirrors the CPU FFTW correlator (`load_windows!`, `spectrum!`,
# `fftshift_abs!`, overlap gain) so planes match `backend = :cpu` within FFT
# round-off.

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

# Gather each window into the (padded) complex batch, mean-subtracted over its
# valid pixels and apodized — the batched analogue of `load_windows!`. One
# work-item per window; the padding region of each slice must already be zero.
@kernel function _ka_gather!(CA, CB, @Const(imgA), @Const(imgB), @Const(origins),
                             @Const(apod), @Const(mask), hasmask, wr, wc)
    k = @index(Global)
    T = eltype(apod)
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
        meanA = n > 0 ? sA / n : zero(T)
        meanB = n > 0 ? sB / n : zero(T)
        for j in 1:wc, i in 1:wr
            if hasmask && mask[rs + i - 1, cs + j - 1]
                CA[i, j, k] = 0
                CB[i, j, k] = 0
            else
                a = apod[i, j]
                CA[i, j, k] = a * (T(imgA[rs + i - 1, cs + j - 1]) - meanA)
                CB[i, j, k] = a * (T(imgB[rs + i - 1, cs + j - 1]) - meanB)
            end
        end
    end
end

# Cross-power spectrum in place: conj(F{A}) .* F{B}.
@kernel function _ka_crosspower!(CA, @Const(CB))
    I = @index(Global)
    @inbounds CA[I] = conj(CA[I]) * CB[I]
end

# fftshift + magnitude (+ overlap gain when padded) into the real plane batch.
@kernel function _ka_shiftgain!(R, @Const(CA), @Const(gain), padded, sr, sc, nr, nc)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ip = mod1(i + sr, nr)
        jp = mod1(j + sc, nc)
        val = abs(CA[i, j, k])
        padded && (val *= gain[ip, jp])
        R[ip, jp, k] = val
    end
end
