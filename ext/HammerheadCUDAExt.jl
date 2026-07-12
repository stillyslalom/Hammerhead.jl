module HammerheadCUDAExt

# NVIDIA CUDA device backend (`backend = :cuda`) for interrogation-window
# correlation, built on CUDA.jl + KernelAbstractions. It reuses the portable
# correlation and plane-analysis kernels shared with the KA-CPU and AMDGPU
# extensions (`_ka_correlation_kernels.jl`); the only device-specific work here
# is host<->device staging and the cuFFT batched transform. The whole pipeline —
# gather, FFTs, cross-power, shift/gain, peak finding + subpixel + moment —
# runs on the device; only the packed per-window scalars come back to the host.
#
# STATUS: a line-for-line mirror of the hardware-validated AMDGPU extension,
# written against the CUDA.jl API (v5/v6 — the 6.0 subpackage split keeps the
# `using CUDA` surface, incl. the exported KernelAbstractions `CUDABackend`)
# but NOT yet validated on an NVIDIA device. Expected to match `backend = :cpu`
# within FFT round-off like the other device backends; confirm with
# `bench/gpu_validate.jl`/`bench/gpu_benchmarks.jl` on hardware first.
#
# CUDA-specific notes mirrored from the AMDGPU bring-up:
# - in-place plans apply via `p * x`: like rocFFT, cuFFT does not implement
#   the 3-arg `mul!(x, p, x)` that FFTW does (JuliaGPU/CUDA.jl#1311);
# - masks are materialized as dense `Array{Bool}` before `copyto!` to the
#   device (a BitMatrix or view would fall back to scalar indexing);
# - the shared kernels avoid throwing ops (checked conversions, `round(Int,`)
#   — on device those compile to runtime exception support and, on AMDGPU,
#   malloc hostcalls that serialize the kernel.

using Hammerhead
using CUDA
using KernelAbstractions
using AbstractFFTs: plan_fft!

import Hammerhead: _AbstractHammerheadBackend, _resolve_backend, _check_backend_params,
    _engine_nchunks, piv_correlation_engines, process_windows!,
    make_correlator, PIVParameters

# Portable correlation kernels + scope guard, shared with HammerheadKAExt.
include("_ka_correlation_kernels.jl")

struct _CUDABackend <: _AbstractHammerheadBackend end

_resolve_backend(::Val{:cuda}) = _CUDABackend()

# Run the whole window grid as one logical batch, tiled internally.
_engine_nchunks(::_CUDABackend, ::Int) = 1

_check_backend_params(::_CUDABackend, passes) = _ka_scope_check(passes, :cuda)

# Windows per device sub-batch (bounds device-memory footprint; at 2048 the
# Float64 complex buffers total ~340 MB, and the analysis kernel — one
# work-item per window — gets enough parallelism to hide memory latency).
const _CUDA_BATCH = 2048

mutable struct _CUDACorrelationEngine{T}
    wsize::NTuple{2,Int}
    fft_size::NTuple{2,Int}
    padded::Bool
    kpk::Int                              # peaks located per window, max(n_peaks, 2)
    apod_d::CuArray{T,2}                  # window-sized (device)
    gain_d::CuArray{T,2}                  # fftshifted gain (device); empty if unpadded
    empty_mask::CuArray{Bool,2}           # 0×0 dummy passed when there is no mask
    # image-sized device staging (lazily (re)sized to the run's image)
    img_size::NTuple{2,Int}
    imgA_d::CuArray{T,2}
    imgB_d::CuArray{T,2}
    mask_d::CuArray{Bool,2}
    # sub-batch device/host buffers and plans (lazily (re)sized to `bs`)
    bs::Int
    CA::CuArray{Complex{T},3}
    CB::CuArray{Complex{T},3}
    Rt::CuArray{T,3}                      # (bs+1, nr, nc) batch-major plane batch
    meanA_d::CuArray{T,1}                 # per-window means (device reduction)
    meanB_d::CuArray{T,1}
    vals_d::CuArray{T,2}                  # (kpk, bs) peak-finder scratch
    locs_d::CuArray{Int32,3}              # (2, kpk, bs) peak-finder scratch
    out_d::CuArray{T,2}                   # (5 + 2*(kpk-1), bs) packed analysis output
    out_host::Matrix{T}
    origins_host::Matrix{Int}
    origins_d::CuArray{Int,2}
    fwd::Any
    bwd::Any
end

function _make_cuda_engine(params::PIVParameters, ::Type{T}) where {T}
    # Reuse the CPU correlator's apodization window and overlap-gain plane so
    # those factors are bit-identical to the FFTW path, then upload them once.
    cpu = make_correlator(params, T)
    apod_d = CuArray{T}(Matrix{T}(cpu.apod))
    fft_size = size(cpu.R)
    padded = !isempty(cpu.gain)
    gain_d = padded ? CuArray{T}(Matrix{T}(cpu.gain)) : CUDA.zeros(T, 0, 0)
    return _CUDACorrelationEngine{T}(
        params.window_size, fft_size, padded, max(params.n_peaks, 2),
        apod_d, gain_d, CUDA.zeros(Bool, 0, 0),
        (0, 0), CUDA.zeros(T, 0, 0), CUDA.zeros(T, 0, 0), CUDA.zeros(Bool, 0, 0),
        0, CUDA.zeros(Complex{T}, 0, 0, 0), CUDA.zeros(Complex{T}, 0, 0, 0),
        CUDA.zeros(T, 0, 0, 0), CUDA.zeros(T, 0), CUDA.zeros(T, 0),
        CUDA.zeros(T, 0, 0), CUDA.zeros(Int32, 2, 0, 0), CUDA.zeros(T, 0, 0),
        Matrix{T}(undef, 0, 0),
        Matrix{Int}(undef, 0, 2), CUDA.zeros(Int, 0, 0), nothing, nothing)
end

piv_correlation_engines(::_CUDABackend, workspace, params::PIVParameters,
                        ::Type{T}, nchunks::Int) where {T} =
    [_make_cuda_engine(params, T) for _ in 1:nchunks]

function _ensure_batch!(engine::_CUDACorrelationEngine{T}, bs::Int) where {T}
    if engine.bs != bs
        nr, nc = engine.fft_size
        engine.CA = CUDA.zeros(Complex{T}, nr, nc, bs)
        engine.CB = CUDA.zeros(Complex{T}, nr, nc, bs)
        # Leading dimension padded by one row: the shift/gain kernel's writes
        # stride by the leading dimension, and at a power-of-two batch size an
        # exact power-of-two byte stride funnels a whole warp into one memory
        # channel (measured ~20x slowdown on the AMDGPU backend). Row bs+1 is
        # never touched.
        engine.Rt = CUDA.zeros(T, bs + 1, nr, nc)
        engine.meanA_d = CUDA.zeros(T, bs)
        engine.meanB_d = CUDA.zeros(T, bs)
        engine.vals_d = CUDA.zeros(T, engine.kpk, bs)
        engine.locs_d = CUDA.zeros(Int32, 2, engine.kpk, bs)
        engine.out_d = CUDA.zeros(T, 5 + 2 * (engine.kpk - 1), bs)
        engine.out_host = Matrix{T}(undef, 5 + 2 * (engine.kpk - 1), bs)
        engine.origins_host = Matrix{Int}(undef, bs, 2)
        engine.origins_d = CUDA.zeros(Int, bs, 2)
        engine.fwd = plan_fft!(engine.CA, (1, 2))
        engine.bwd = inv(engine.fwd)
        engine.bs = bs
    end
    return engine
end

function _ensure_image!(engine::_CUDACorrelationEngine{T}, sz::NTuple{2,Int},
                        hasmask::Bool) where {T}
    if engine.img_size != sz
        engine.imgA_d = CUDA.zeros(T, sz...)
        engine.imgB_d = CUDA.zeros(T, sz...)
        engine.img_size = sz
    end
    if hasmask && size(engine.mask_d) != sz
        engine.mask_d = CUDA.zeros(Bool, sz...)
    end
    return engine
end

function process_windows!(u, v, peak_ratio, correlation_moment, alt_u, alt_v,
                          uncertainty_u, uncertainty_v, jobs,
                          imgA::AbstractMatrix, imgB::AbstractMatrix, params::PIVParameters,
                          engine::_CUDACorrelationEngine{T},
                          mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                          planes = nothing) where {T}
    (uncertainty_u === nothing && planes === nothing) ||
        throw(ArgumentError("backend :cuda does not support uncertainty or " *
                            "correlation-plane storage yet; use backend = :cpu"))
    jobvec = jobs isa AbstractVector ? jobs : collect(jobs)
    njobs = length(jobvec)
    njobs == 0 && return nothing

    wr, wc = engine.wsize
    nr, nc = engine.fft_size
    sr, sc = nr ÷ 2, nc ÷ 2
    bs = min(_CUDA_BATCH, njobs)
    _ensure_batch!(engine, bs)
    hasmask = mask !== nothing
    _ensure_image!(engine, size(imgA), hasmask)
    ka = CUDABackend()

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
    use_regionalmax = params.peak_finder === :regionalmax
    use_gauss9 = params.subpixel_method === :gauss9
    nalt = engine.kpk - 1

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
        _ka_window_means!(ka)(engine.meanA_d, engine.meanB_d, engine.imgA_d,
                              engine.imgB_d, engine.origins_d, maskarg, hasmask,
                              wr, wc; ndrange = nreal)
        _ka_gather!(ka)(engine.CA, engine.CB, engine.imgA_d, engine.imgB_d,
                        engine.origins_d, engine.apod_d, engine.meanA_d,
                        engine.meanB_d, maskarg, hasmask; ndrange = (wr, wc, nreal))
        KernelAbstractions.synchronize(ka)
        # In-place plan application: `p * x` mutates x and returns it — cuFFT,
        # like rocFFT, does not implement the 3-arg `mul!(x, p, x)` that FFTW
        # does (JuliaGPU/CUDA.jl#1311).
        engine.fwd * engine.CA
        engine.fwd * engine.CB
        _ka_crosspower!(ka)(engine.CA, engine.CB; ndrange = length(engine.CA))
        KernelAbstractions.synchronize(ka)
        engine.bwd * engine.CA
        _ka_shiftgain!(ka)(engine.Rt, engine.CA, gainarg, engine.padded, sr, sc, nr, nc;
                           ndrange = (nr, nc, bs))
        # Same stream as the shift/gain kernel, so no synchronize between them.
        _ka_analyze!(ka)(engine.out_d, engine.vals_d, engine.locs_d, engine.Rt,
                         use_regionalmax, use_gauss9, params.n_peaks, nr, nc;
                         ndrange = nreal)
        KernelAbstractions.synchronize(ka)
        # Device -> host: only the packed per-window scalars, never the planes.
        copyto!(engine.out_host, engine.out_d)

        # Scatter the packed outputs into the vector grids (see the
        # `_ka_analyze!` row layout).
        for m in 1:nreal
            job = jobvec[start + m - 1]
            gi, gj = job[1], job[2]
            u[gi, gj] = engine.out_host[1, m]
            v[gi, gj] = engine.out_host[2, m]
            peak_ratio[gi, gj] = engine.out_host[3, m]
            correlation_moment[gi, gj] = engine.out_host[4, m]
            if alt_u !== nothing
                found = Int(engine.out_host[5, m])   # small integer, exact in T
                for mm in 2:min(found, params.n_peaks)
                    alt_u[gi, gj, mm - 1] = engine.out_host[5 + (mm - 1), m]
                    alt_v[gi, gj, mm - 1] = engine.out_host[5 + nalt + (mm - 1), m]
                end
            end
        end
    end
    return nothing
end

end # module
