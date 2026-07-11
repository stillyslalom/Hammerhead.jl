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
# refinement, the correlation moment, and the peak ratio run in the shared
# `_ka_analyze!` kernel — a faithful port of `analyze_plane!` — so on a GPU
# device only a handful of scalars per window return to the host, never whole
# correlation planes; here on the CPU backend the same kernel guards that
# shared code in CI.

using Hammerhead
using KernelAbstractions
using AbstractFFTs: plan_fft!
using LinearAlgebra: mul!

import Hammerhead: _AbstractHammerheadBackend, _resolve_backend, _check_backend_params,
    _engine_nchunks, piv_correlation_engines, process_windows!, _correlation_apod,
    make_correlator, PIVParameters

# Portable correlation kernels shared with the GPU device extensions.
include("_ka_correlation_kernels.jl")

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

_check_backend_params(::_KABackend, passes) = _ka_scope_check(passes, :ka)

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
        Matrix{T}(undef, 0, 0), nothing, nothing)
end

piv_correlation_engines(::_KABackend, workspace, params::PIVParameters,
                        ::Type{T}, nchunks::Int) where {T} =
    [_make_ka_engine(params, T) for _ in 1:nchunks]

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

end # module
