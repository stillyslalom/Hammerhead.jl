module HammerheadAMDGPUExt

# AMD ROCm/HIP device backend (`backend = :amdgpu`) for interrogation-window
# correlation, built on AMDGPU.jl + KernelAbstractions. It reuses the portable
# correlation kernels shared with the KA-CPU extension (`_ka_correlation_kernels.jl`)
# and the CPU peak/subpixel/moment routines; the only device-specific work here
# is host<->device staging and the rocFFT batched transform.
#
# STATUS: written against the AMDGPU/KernelAbstractions APIs but NOT yet
# validated on hardware — needs an RX 6800 XT (gfx1030) with the ROCm/HIP SDK
# and a working rocFFT for batched region transforms (the central portability
# risk). Numerics mirror the CPU FFTW path (shared apod/gain planes), so results
# should match `backend = :cpu` within FFT round-off; confirm with test_ka-style
# comparisons once the device is available.

using Hammerhead
using AMDGPU
using KernelAbstractions
using AbstractFFTs: plan_fft!

import Hammerhead: _AbstractHammerheadBackend, _resolve_backend, _check_backend_params,
    _engine_nchunks, piv_correlation_engines, process_windows!,
    make_correlator, analyze_plane!, subpixel_gauss3, PIVParameters

# Portable correlation kernels + scope guard, shared with HammerheadKAExt.
include("_ka_correlation_kernels.jl")

struct _AMDGPUBackend <: _AbstractHammerheadBackend end

_resolve_backend(::Val{:amdgpu}) = _AMDGPUBackend()

# Run the whole window grid as one logical batch, tiled internally.
_engine_nchunks(::_AMDGPUBackend, ::Int) = 1

_check_backend_params(::_AMDGPUBackend, passes) = _ka_scope_check(passes, :amdgpu)

# Windows per device sub-batch (bounds device-memory footprint).
const _AMDGPU_BATCH = 512

mutable struct _AMDGPUCorrelationEngine{T}
    wsize::NTuple{2,Int}
    fft_size::NTuple{2,Int}
    padded::Bool
    apod_d::ROCArray{T,2}                 # window-sized (device)
    gain_d::ROCArray{T,2}                 # fftshifted gain (device); empty if unpadded
    empty_mask::ROCArray{Bool,2}          # 0×0 dummy passed when there is no mask
    # image-sized device staging (lazily (re)sized to the run's image)
    img_size::NTuple{2,Int}
    imgA_d::ROCArray{T,2}
    imgB_d::ROCArray{T,2}
    mask_d::ROCArray{Bool,2}
    # sub-batch device/host buffers and plans (lazily (re)sized to `bs`)
    bs::Int
    CA::ROCArray{Complex{T},3}
    CB::ROCArray{Complex{T},3}
    R3::ROCArray{T,3}
    R3_host::Array{T,3}
    origins_host::Matrix{Int}
    origins_d::ROCArray{Int,2}
    fwd::Any
    bwd::Any
end

function _make_amdgpu_engine(params::PIVParameters, ::Type{T}) where {T}
    # Reuse the CPU correlator's apodization window and overlap-gain plane so
    # those factors are bit-identical to the FFTW path, then upload them once.
    cpu = make_correlator(params, T)
    apod_d = ROCArray{T}(Matrix{T}(cpu.apod))
    fft_size = size(cpu.R)
    padded = !isempty(cpu.gain)
    gain_d = padded ? ROCArray{T}(Matrix{T}(cpu.gain)) : AMDGPU.zeros(T, 0, 0)
    return _AMDGPUCorrelationEngine{T}(
        params.window_size, fft_size, padded, apod_d, gain_d, AMDGPU.zeros(Bool, 0, 0),
        (0, 0), AMDGPU.zeros(T, 0, 0), AMDGPU.zeros(T, 0, 0), AMDGPU.zeros(Bool, 0, 0),
        0, AMDGPU.zeros(Complex{T}, 0, 0, 0), AMDGPU.zeros(Complex{T}, 0, 0, 0),
        AMDGPU.zeros(T, 0, 0, 0), Array{T,3}(undef, 0, 0, 0),
        Matrix{Int}(undef, 0, 2), AMDGPU.zeros(Int, 0, 0), nothing, nothing)
end

piv_correlation_engines(::_AMDGPUBackend, workspace, params::PIVParameters,
                        ::Type{T}, nchunks::Int) where {T} =
    [_make_amdgpu_engine(params, T) for _ in 1:nchunks]

function _ensure_batch!(engine::_AMDGPUCorrelationEngine{T}, bs::Int) where {T}
    if engine.bs != bs
        nr, nc = engine.fft_size
        engine.CA = AMDGPU.zeros(Complex{T}, nr, nc, bs)
        engine.CB = AMDGPU.zeros(Complex{T}, nr, nc, bs)
        engine.R3 = AMDGPU.zeros(T, nr, nc, bs)
        engine.R3_host = Array{T,3}(undef, nr, nc, bs)
        engine.origins_host = Matrix{Int}(undef, bs, 2)
        engine.origins_d = AMDGPU.zeros(Int, bs, 2)
        engine.fwd = plan_fft!(engine.CA, (1, 2))
        engine.bwd = inv(engine.fwd)
        engine.bs = bs
    end
    return engine
end

function _ensure_image!(engine::_AMDGPUCorrelationEngine{T}, sz::NTuple{2,Int},
                        hasmask::Bool) where {T}
    if engine.img_size != sz
        engine.imgA_d = AMDGPU.zeros(T, sz...)
        engine.imgB_d = AMDGPU.zeros(T, sz...)
        engine.img_size = sz
    end
    if hasmask && size(engine.mask_d) != sz
        engine.mask_d = AMDGPU.zeros(Bool, sz...)
    end
    return engine
end

function process_windows!(u, v, peak_ratio, correlation_moment, alt_u, alt_v,
                          uncertainty_u, uncertainty_v, jobs,
                          imgA::AbstractMatrix, imgB::AbstractMatrix, params::PIVParameters,
                          engine::_AMDGPUCorrelationEngine{T},
                          mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                          planes = nothing) where {T}
    (uncertainty_u === nothing && planes === nothing) ||
        throw(ArgumentError("backend :amdgpu does not support uncertainty or " *
                            "correlation-plane storage yet; use backend = :cpu"))
    jobvec = jobs isa AbstractVector ? jobs : collect(jobs)
    njobs = length(jobvec)
    njobs == 0 && return nothing

    wr, wc = engine.wsize
    nr, nc = engine.fft_size
    sr, sc = nr ÷ 2, nc ÷ 2
    bs = min(_AMDGPU_BATCH, njobs)
    _ensure_batch!(engine, bs)
    hasmask = mask !== nothing
    _ensure_image!(engine, size(imgA), hasmask)
    ka = ROCBackend()

    # Upload this sweep's (deformed) images once; apod/gain live on the device.
    copyto!(engine.imgA_d, imgA)
    copyto!(engine.imgB_d, imgB)
    maskarg = engine.empty_mask
    if hasmask
        # A BitMatrix (e.g. from `falses`) or a view can't memcpy to the device;
        # materialize a dense Array{Bool} first (identity when already dense).
        copyto!(engine.mask_d, convert(Array{Bool}, mask))
        maskarg = engine.mask_d
    end
    gainarg = engine.padded ? engine.gain_d : engine.apod_d   # dummy when unpadded (unread)

    kpk = max(params.n_peaks, 2)
    vals = Vector{T}(undef, kpk)
    locs = Vector{NTuple{2,Int}}(undef, kpk)

    for start in 1:bs:njobs
        nreal = min(bs, njobs - start + 1)
        @inbounds for m in 1:nreal
            job = jobvec[start + m - 1]
            engine.origins_host[m, 1] = job[3]   # rs
            engine.origins_host[m, 2] = job[4]   # cs
        end
        copyto!(engine.origins_d, engine.origins_host)
        fill!(engine.CA, 0)
        fill!(engine.CB, 0)
        _ka_gather!(ka)(engine.CA, engine.CB, engine.imgA_d, engine.imgB_d,
                        engine.origins_d, engine.apod_d, maskarg, hasmask, wr, wc;
                        ndrange = nreal)
        KernelAbstractions.synchronize(ka)
        # In-place plan application: `p * x` mutates x and returns it, the
        # idiom shared by FFTW and rocFFT in-place plans (rocFFT does not
        # implement the 3-arg `mul!(x, p, x)` that FFTW does).
        engine.fwd * engine.CA
        engine.fwd * engine.CB
        _ka_crosspower!(ka)(engine.CA, engine.CB; ndrange = length(engine.CA))
        KernelAbstractions.synchronize(ka)
        engine.bwd * engine.CA
        _ka_shiftgain!(ka)(engine.R3, engine.CA, gainarg, engine.padded, sr, sc, nr, nc;
                           ndrange = (nr, nc, bs))
        KernelAbstractions.synchronize(ka)
        copyto!(engine.R3_host, engine.R3)   # device -> host for CPU peak finding

        for m in 1:nreal
            job = jobvec[start + m - 1]
            gi, gj = job[1], job[2]
            Rk = view(engine.R3_host, :, :, m)
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
