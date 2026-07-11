module HammerheadKAExt

# Portable batched interrogation-window correlation via KernelAbstractions.
#
# This extension registers the `backend = :ka` selector, which runs the window
# correlation on KernelAbstractions' CPU backend — a hardware-free proving tier
# for the portable kernels. A GPU device extension (CUDA/AMDGPU) reuses the same
# kernels with its own array type, FFT plan, and device backend.
#
# The plane-level numerics are matched to the CPU FFTW path exactly (the same
# apodization window and overlap-gain plane are reused), so results differ from
# `backend = :cpu` only by FFT/reduction round-off. Peak finding, subpixel
# refinement, the correlation moment, and the peak ratio are the shared CPU
# routines (`analyze_plane!`) run on the host-side plane batch — identical given
# the same plane — so only the FFT + elementwise stages are new device kernels.

using Hammerhead
using KernelAbstractions
using AbstractFFTs: plan_fft!
using LinearAlgebra: mul!

import Hammerhead: _AbstractHammerheadBackend, _resolve_backend, _check_backend_params,
    _engine_nchunks, piv_correlation_engines, process_windows!, _correlation_apod,
    make_correlator, analyze_plane!, subpixel_gauss3, PIVParameters

struct _KABackend <: _AbstractHammerheadBackend end

_resolve_backend(::Val{:ka}) = _KABackend()

# Device engines run the whole window grid as one logical batch (tiled
# internally into memory-bounded sub-batches) rather than fanning out across
# host threads.
_engine_nchunks(::_KABackend, ::Int) = 1

# Windows processed per device sub-batch. Bounds the FFT-buffer footprint:
# materializing every overlapping window of a large image at once can exceed
# device memory.
const _KA_BATCH = 512

# The initial :ka scope is the plan's Phase 2 slice; other options remain
# CPU-first (or land in a later phase).
function _check_backend_params(::_KABackend, passes)
    for p in passes
        p.correlation_method === :cross ||
            throw(ArgumentError("backend :ka supports correlation_method = :cross only " *
                                "(got :$(p.correlation_method)); use backend = :cpu"))
        p.subpixel_method in (:gauss3, :gauss9) ||
            throw(ArgumentError("backend :ka supports subpixel_method :gauss3 or :gauss9 " *
                                "only (got :$(p.subpixel_method)); use backend = :cpu"))
        p.uncertainty &&
            throw(ArgumentError("backend :ka does not support uncertainty quantification " *
                                "yet; run UQ on backend = :cpu"))
        p.keep_correlation_planes &&
            throw(ArgumentError("backend :ka does not support keep_correlation_planes yet; " *
                                "use backend = :cpu"))
    end
    return nothing
end

# Per-chunk (here: single) correlation engine. Buffers and FFT plans are cached
# and reused across a pass's deformation sweeps; they are rebuilt only when the
# sub-batch size changes.
mutable struct _KACorrelationEngine{T,KB}
    ka::KB
    wsize::NTuple{2,Int}
    fft_size::NTuple{2,Int}
    padded::Bool
    apod::Matrix{T}
    gain::Matrix{T}                 # fftshifted overlap gain; empty if unpadded
    bs::Int
    CA::Array{Complex{T},3}
    CB::Array{Complex{T},3}
    R3::Array{T,3}
    origins::Matrix{Int}
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
        CPU(), params.window_size, fft_size, padded, apod, gain, 0,
        Array{Complex{T},3}(undef, 0, 0, 0), Array{Complex{T},3}(undef, 0, 0, 0),
        Array{T,3}(undef, 0, 0, 0), Matrix{Int}(undef, 0, 2), nothing, nothing)
end

piv_correlation_engines(::_KABackend, workspace, params::PIVParameters,
                        ::Type{T}, nchunks::Int) where {T} =
    [_make_ka_engine(params, T) for _ in 1:nchunks]

function _ensure_buffers!(engine::_KACorrelationEngine{T}, bs::Int) where {T}
    if engine.bs != bs
        nr, nc = engine.fft_size
        engine.CA = zeros(Complex{T}, nr, nc, bs)
        engine.CB = zeros(Complex{T}, nr, nc, bs)
        engine.R3 = zeros(T, nr, nc, bs)
        engine.origins = Matrix{Int}(undef, bs, 2)
        engine.fwd = plan_fft!(engine.CA, (1, 2))
        engine.bwd = inv(engine.fwd)
        engine.bs = bs
    end
    return engine
end

# Gather each window into the (padded) complex batch, mean-subtracted over its
# valid pixels and apodized — the batched analogue of `load_windows!`. One
# work-item per window.
@kernel function _gather!(CA, CB, @Const(imgA), @Const(imgB), @Const(origins),
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
@kernel function _crosspower!(CA, @Const(CB))
    I = @index(Global)
    @inbounds CA[I] = conj(CA[I]) * CB[I]
end

# fftshift + magnitude (+ overlap gain when padded) into the real plane batch.
@kernel function _shiftgain!(R, @Const(CA), @Const(gain), padded, sr, sc, nr, nc)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ip = mod1(i + sr, nr)
        jp = mod1(j + sc, nc)
        val = abs(CA[i, j, k])
        padded && (val *= gain[ip, jp])
        R[ip, jp, k] = val
    end
end

function process_windows!(u, v, peak_ratio, correlation_moment, alt_u, alt_v,
                          uncertainty_u, uncertainty_v, jobs,
                          imgA::AbstractMatrix, imgB::AbstractMatrix, params::PIVParameters,
                          engine::_KACorrelationEngine{T},
                          mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                          planes = nothing) where {T}
    (uncertainty_u === nothing && planes === nothing) ||
        throw(ArgumentError("backend :ka does not support uncertainty or correlation-plane " *
                            "storage yet; use backend = :cpu"))
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

    kpk = max(params.n_peaks, 2)
    vals = Vector{T}(undef, kpk)
    locs = Vector{NTuple{2,Int}}(undef, kpk)

    for start in 1:bs:njobs
        nreal = min(bs, njobs - start + 1)
        @inbounds for m in 1:nreal
            job = jobvec[start + m - 1]
            engine.origins[m, 1] = job[3]   # rs
            engine.origins[m, 2] = job[4]   # cs
        end
        fill!(engine.CA, 0)
        fill!(engine.CB, 0)
        _gather!(ka)(engine.CA, engine.CB, imgA, imgB, engine.origins, engine.apod,
                     themask, hasmask, wr, wc; ndrange = nreal)
        KernelAbstractions.synchronize(ka)
        mul!(engine.CA, engine.fwd, engine.CA)
        mul!(engine.CB, engine.fwd, engine.CB)
        _crosspower!(ka)(engine.CA, engine.CB; ndrange = length(engine.CA))
        KernelAbstractions.synchronize(ka)
        mul!(engine.CA, engine.bwd, engine.CA)
        _shiftgain!(ka)(engine.R3, engine.CA, gainarg, engine.padded, sr, sc, nr, nc;
                        ndrange = (nr, nc, bs))
        KernelAbstractions.synchronize(ka)

        for m in 1:nreal
            job = jobvec[start + m - 1]
            gi, gj = job[1], job[2]
            Rk = view(engine.R3, :, :, m)
            res = analyze_plane!(vals, locs, Rk, params)
            u[gi, gj] = res.du
            v[gi, gj] = res.dv
            peak_ratio[gi, gj] = res.ratio
            correlation_moment[gi, gj] = res.moment
            if alt_u !== nothing
                for mm in 2:min(res.found, params.n_peaks)
                    aref = subpixel_gauss3(Rk, locs[mm])
                    alt_u[gi, gj, mm - 1] = aref[2] - res.center[2]
                    alt_v[gi, gj, mm - 1] = aref[1] - res.center[1]
                end
            end
        end
    end
    return nothing
end

end # module
