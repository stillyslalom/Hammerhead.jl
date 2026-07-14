module HammerheadCUDAExt

# NVIDIA CUDA device backend (`backend = :cuda`) for interrogation-window
# correlation, built on CUDA.jl + KernelAbstractions. It reuses the portable
# correlation and plane-analysis kernels defined in the core package
# (`src/ka_backend.jl`, shared with the `:ka` proving tier and the AMDGPU
# extension); the only device-specific work here is host<->device staging and
# the cuFFT batched transform. The whole pipeline —
# predictor deformation (from once-per-call staged B-spline coefficients),
# gather, FFTs, cross-power, shift/gain, peak finding + subpixel + moment —
# runs on the device; the warped images stay device-resident between
# deformation and correlation, and only the packed per-window scalars come
# back to the host.
#
# STATUS: a line-for-line mirror of the hardware-validated AMDGPU extension,
# written against the CUDA.jl API (v5/v6 — the 6.0 subpackage split keeps the
# `using CUDA` surface, incl. the exported KernelAbstractions `CUDABackend`).
# Hardware-validated on an RTX 2000 Ada (CUDA.jl 6.2, driver CUDA 12.8): all
# paths match `backend = :cpu` to ~1e-15 (Float64) via `bench/gpu_validate.jl`,
# 1.7–2.8× threaded-CPU speed via `bench/gpu_benchmarks.jl`.
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
# Strong deps of the core package, usable here without being ext triggers.
using KernelAbstractions
using AbstractFFTs: plan_fft!

import Hammerhead: _AbstractHammerheadBackend, _resolve_backend, _check_backend_params,
    _engine_nchunks, piv_correlation_engines, pooled_engines, process_windows!,
    make_correlator, PIVParameters, _supports_fft, _supports_batched_fft,
    _supports_fp64, finalize_uncertainty, UQ_NSTATS, uncertainty_sweep!,
    _correlation_apod

# Portable correlation kernels + scope guard from the core (src/ka_backend.jl).
import Hammerhead: _ka_scope_check, _ka_window_means!, _ka_gather!,
    _ka_crosspower!, _ka_phasepower!, _ka_shiftgain!, _ka_analyze!
import Hammerhead: _ka_uq_fill!, _ka_uq_stats!

# Device-resident deformation (Phase 3b): the staged-context machinery is
# portable core code (proven on `:ka`); this extension only points it at the
# CUDA backend.
import Hammerhead: apply_predictor, _deform_context, _ka_deform_context,
    _ka_apply_predictor_ctx

struct _CUDABackend <: _AbstractHammerheadBackend end

_resolve_backend(::Val{:cuda}) = _CUDABackend()

# Run the whole window grid as one logical batch, tiled internally.
_engine_nchunks(::_CUDABackend, ::Int) = 1
_supports_fft(::_CUDABackend) = true
_supports_batched_fft(::_CUDABackend) = true
_supports_fp64(::_CUDABackend) = true

_check_backend_params(b::_CUDABackend, passes) =
    _ka_scope_check(passes, :cuda, _supports_fp64(b))

# Windows per device sub-batch (bounds device-memory footprint). The analysis
# kernel launches one *workgroup* per window, so the sub-batch size *is* that
# kernel's launch parallelism (in workgroups): at 2048 windows an 8-tile 2048²
# grid starves the device (8 sequential launches, ~4.7 ms each), while a single
# well-occupied launch of the whole grid runs the same work in ~6 ms. 8192 gives
# near-peak occupancy on this class of GPU while keeping the Float64 complex
# buffers bounded (~1.3 GB at 8192; the RTX 2000 Ada has 16 GB). Measured: 2048²
# Float32 single-pass 104 ms → 76 ms versus a 2048 sub-batch.
const _CUDA_BATCH = 8192

# KernelAbstractions' occupancy-based default workgroup size leaves the
# elementwise cross-power far off peak bandwidth on NVIDIA; an explicit
# warp-multiple block recovers ~8x on `_ka_crosspower!` (RTX 2000 Ada, 2048²
# windows). Workgroup size never affects results — every work-item is
# independent — so this is pure tuning. (The analysis kernel is cooperative:
# its groupsize is fixed by `Val{TPW}`, not tuned here.)
const _WG_ELEMENTWISE = 256   # cross-power over the whole complex batch

# Threads per window for the cooperative `_ka_analyze!` (one workgroup per
# window; see the core kernel). The launch groupsize must equal the kernel's
# compile-time `Val{TPW}`; the core sets `_KA_TPW = 128`, fastest across sizes.
const _WG_ANALYZE = Hammerhead._KA_TPW

mutable struct _CUDACorrelationEngine{T}
    wsize::NTuple{2,Int}
    fft_size::NTuple{2,Int}
    padded::Bool
    kpk::Int                              # peaks located per window, max(n_peaks, 2)
    apod_d::CuArray{T,2}                  # window-sized (device)
    gain_d::CuArray{T,2}                  # fftshifted gain (device); empty if unpadded
    phase_filter_d::CuArray{T,2}          # Gaussian FFT weights; empty for cross-correlation
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
    Rt::CuArray{T,3}                      # (nr, nc, bs+1) plane-major plane batch
    meanA_d::CuArray{T,1}                 # per-window means (device reduction)
    meanB_d::CuArray{T,1}
    vals_d::CuArray{T,2}                  # (kpk, bs) peak-finder scratch
    locs_d::CuArray{Int32,3}              # (2, kpk, bs) peak-finder scratch
    out_d::CuArray{T,2}                   # (5 + 2*(kpk-1), bs) packed analysis output
    out_host::Matrix{T}
    uqstats_d::CuArray{Float64,3}
    uqmeans_d::CuArray{Float64,2}
    uqdcs_d::CuArray{T,4}                 # (bs+1, 2, mm, mm) cached smoothed ΔC
    uqstats_host::Array{Float64,3}
    origins_host::Matrix{Int}
    origins_d::CuArray{Int,2}
    fwd::Any
    bwd::Any
end

_correlation_apod(engine::_CUDACorrelationEngine) = engine.apod_d

function _make_cuda_engine(params::PIVParameters, ::Type{T}) where {T}
    # Reuse the CPU correlator's apodization window and overlap-gain plane so
    # those factors are bit-identical to the FFTW path, then upload them once.
    cpu = make_correlator(params, T)
    apod_d = CuArray{T}(Matrix{T}(cpu.apod))
    fft_size = size(cpu.R)
    padded = !isempty(cpu.gain)
    gain_d = padded ? CuArray{T}(Matrix{T}(cpu.gain)) : CUDA.zeros(T, 0, 0)
    phase_filter_d = params.correlation_method === :phase ?
        CuArray{T}(Matrix{T}(cpu.W)) : CUDA.zeros(T, 0, 0)
    return _CUDACorrelationEngine{T}(
        params.window_size, fft_size, padded, max(params.n_peaks, 2),
        apod_d, gain_d, phase_filter_d, CUDA.zeros(Bool, 0, 0),
        (0, 0), CUDA.zeros(T, 0, 0), CUDA.zeros(T, 0, 0), CUDA.zeros(Bool, 0, 0),
        0, CUDA.zeros(Complex{T}, 0, 0, 0), CUDA.zeros(Complex{T}, 0, 0, 0),
        CUDA.zeros(T, 0, 0, 0), CUDA.zeros(T, 0), CUDA.zeros(T, 0),
        CUDA.zeros(T, 0, 0), CUDA.zeros(Int32, 2, 0, 0), CUDA.zeros(T, 0, 0),
        Matrix{T}(undef, 0, 0), CUDA.zeros(Float64, 0, 0, 0),
        CUDA.zeros(Float64, 0, 0), CUDA.zeros(T, 0, 0, 0, 0),
        Array{Float64,3}(undef, 0, 0, 0),
        Matrix{Int}(undef, 0, 2), CUDA.zeros(Int, 0, 0), nothing, nothing)
end

# Engines are pooled per window configuration via the workspace (see the core
# `pooled_engines`), so device buffers and cuFFT plans are paid once per
# configuration for a whole sequence/ensemble batch.
piv_correlation_engines(::_CUDABackend, workspace, params::PIVParameters,
                        ::Type{T}, nchunks::Int) where {T} =
    pooled_engines(() -> _make_cuda_engine(params, T), workspace,
                   (:cuda, T, params.correlation_method, params.window_size, params.padding,
                    params.apodization, max(params.n_peaks, 2)), nchunks)

function _device_spectrum!(engine::_CUDACorrelationEngine, params::PIVParameters, ka)
    if params.correlation_method === :phase
        _ka_phasepower!(ka, _WG_ELEMENTWISE)(
            engine.CA, engine.CB, engine.phase_filter_d, length(engine.phase_filter_d);
            ndrange = length(engine.CA))
    else
        _ka_crosspower!(ka, _WG_ELEMENTWISE)(engine.CA, engine.CB;
                                             ndrange = length(engine.CA))
    end
    return nothing
end

function _ensure_batch!(engine::_CUDACorrelationEngine{T}, bs::Int) where {T}
    if engine.bs != bs
        nr, nc = engine.fft_size
        engine.CA = CUDA.zeros(Complex{T}, nr, nc, bs)
        engine.CB = CUDA.zeros(Complex{T}, nr, nc, bs)
        # Trailing dimension padded by one plane: at a power-of-two batch size
        # an exact power-of-two plane-to-plane byte stride funnels a whole warp
        # into one memory channel (measured ~20x slowdown on the AMDGPU
        # backend). Plane bs+1 is never touched.
        engine.Rt = CUDA.zeros(T, nr, nc, bs + 1)
        engine.meanA_d = CUDA.zeros(T, bs)
        engine.meanB_d = CUDA.zeros(T, bs)
        engine.vals_d = CUDA.zeros(T, engine.kpk, bs)
        engine.locs_d = CUDA.zeros(Int32, 2, engine.kpk, bs)
        engine.out_d = CUDA.zeros(T, 5 + 2 * (engine.kpk - 1), bs)
        engine.out_host = Matrix{T}(undef, 5 + 2 * (engine.kpk - 1), bs)
        engine.uqstats_d = CUDA.zeros(Float64, 2, UQ_NSTATS, bs)
        engine.uqmeans_d = CUDA.zeros(Float64, 2, bs)
        mm = max(engine.wsize...)
        engine.uqdcs_d = CUDA.zeros(T, bs + 1, 2, mm, mm)  # +1: channel-conflict pad
        engine.uqstats_host = Array{Float64,3}(undef, 2, UQ_NSTATS, bs)
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

# Deformation staging: the prefiltered B-spline coefficients go to the device
# once per `run_piv` call (per pair in the ensemble driver) and the deform
# context's warp buffers stay resident, so per-sweep deformation transfers
# only the coarse predictor grid (see `_ka_deform_context` in the core).
_deform_context(::_CUDABackend, workspace, itpA, itpB,
                imgsize::Dims{2}, ::Type{T}) where {T} =
    _ka_deform_context(CUDABackend(), workspace, (:cuda_deform, T, imgsize),
                       itpA, itpB, imgsize, T)

function apply_predictor(::_CUDABackend, imgA::AbstractMatrix, imgB::AbstractMatrix,
                         itpA, itpB, predictor, x::AbstractVector, y::AbstractVector,
                         ::Type{T}; threaded::Bool = false,
                         warpA = nothing, warpB = nothing, ctx = nothing) where {T}
    predictor === nothing &&
        return imgA, imgB, zeros(T, length(y), length(x)), zeros(T, length(y), length(x))
    # Without a staged context (direct calls outside the drivers), deform on
    # the CPU as before.
    ctx === nothing &&
        return apply_predictor(imgA, imgB, itpA, itpB, predictor, x, y, T;
                               threaded, warpA, warpB)
    return _ka_apply_predictor_ctx(ctx, predictor, x, y)
end

# Resolve this sweep's device images and mask. Warped images arriving from the
# deform context are already device-resident and are used in place; host
# images (a non-deforming pass, or direct calls) are uploaded to the engine's
# staging buffers as before.
function _stage_pair!(engine::_CUDACorrelationEngine{T}, imgA, imgB, mask) where {T}
    hasmask = mask !== nothing
    if imgA isa CuArray
        A_d, B_d = imgA, imgB
        hasmask && size(engine.mask_d) != size(imgA) &&
            (engine.mask_d = CUDA.zeros(Bool, size(imgA)...))
    else
        _ensure_image!(engine, size(imgA), hasmask)
        # Upload this sweep's images once; apod/gain live on the device.
        copyto!(engine.imgA_d, imgA)
        copyto!(engine.imgB_d, imgB)
        A_d, B_d = engine.imgA_d, engine.imgB_d
    end
    maskarg = engine.empty_mask
    if hasmask
        # A BitMatrix (e.g. from `falses`) or a view can't memcpy to the device;
        # materialize a dense Array{Bool} first (identity when already dense).
        copyto!(engine.mask_d, convert(Array{Bool}, mask))
        maskarg = engine.mask_d
    end
    return A_d, B_d, maskarg, hasmask
end

function process_windows!(u, v, peak_ratio, correlation_moment, alt_u, alt_v,
                          uncertainty_u, uncertainty_v, jobs,
                          imgA::AbstractMatrix, imgB::AbstractMatrix, params::PIVParameters,
                          engine::_CUDACorrelationEngine{T},
                          mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                          planes = nothing) where {T}
    planes === nothing ||
        throw(ArgumentError("backend :cuda does not support correlation-plane storage yet; " *
                            "use backend = :cpu"))
    jobvec = jobs isa AbstractVector ? jobs : collect(jobs)
    njobs = length(jobvec)
    njobs == 0 && return nothing

    wr, wc = engine.wsize
    nr, nc = engine.fft_size
    sr, sc = nr ÷ 2, nc ÷ 2
    bs = min(_CUDA_BATCH, njobs)
    _ensure_batch!(engine, bs)
    A_d, B_d, maskarg, hasmask = _stage_pair!(engine, imgA, imgB, mask)
    ka = CUDABackend()

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
        _ka_window_means!(ka)(engine.meanA_d, engine.meanB_d, A_d,
                              B_d, engine.origins_d, maskarg, hasmask,
                              wr, wc; ndrange = nreal)
        _ka_gather!(ka)(engine.CA, engine.CB, A_d, B_d,
                        engine.origins_d, engine.apod_d, engine.meanA_d,
                        engine.meanB_d, maskarg, hasmask; ndrange = (wr, wc, nreal))
        if uncertainty_u !== nothing
            _ka_uq_fill!(ka)(engine.uqdcs_d, engine.uqmeans_d, engine.CA, engine.CB,
                             wr, wc; ndrange = (2, nreal))
            _ka_uq_stats!(ka)(engine.uqstats_d, engine.uqmeans_d, engine.uqdcs_d,
                              engine.CA, engine.CB, wr, wc, nreal, 0, false;
                              ndrange = (2, UQ_NSTATS, nreal))
        end
        KernelAbstractions.synchronize(ka)
        # In-place plan application: `p * x` mutates x and returns it — cuFFT,
        # like rocFFT, does not implement the 3-arg `mul!(x, p, x)` that FFTW
        # does (JuliaGPU/CUDA.jl#1311).
        engine.fwd * engine.CA
        engine.fwd * engine.CB
        _device_spectrum!(engine, params, ka)
        KernelAbstractions.synchronize(ka)
        engine.bwd * engine.CA
        _ka_shiftgain!(ka)(engine.Rt, engine.CA, gainarg, engine.padded, sr, sc, nr, nc;
                           ndrange = (nr, nc, bs))
        # Same stream as the shift/gain kernel, so no synchronize between them.
        # Cooperative analysis: one workgroup (_WG_ANALYZE threads) per window.
        _ka_analyze!(ka, _WG_ANALYZE)(engine.out_d, engine.vals_d, engine.locs_d, engine.Rt,
                         use_regionalmax, use_gauss9, params.n_peaks, nr, nc, Val(_WG_ANALYZE);
                         ndrange = _WG_ANALYZE * nreal)
        KernelAbstractions.synchronize(ka)
        # Device -> host: only the packed per-window scalars, never the planes.
        copyto!(engine.out_host, engine.out_d)
        uncertainty_u === nothing || copyto!(engine.uqstats_host, engine.uqstats_d)

        # Scatter the packed outputs into the vector grids (see the
        # `_ka_analyze!` row layout).
        for m in 1:nreal
            job = jobvec[start + m - 1]
            gi, gj = job[1], job[2]
            u[gi, gj] = engine.out_host[1, m]
            v[gi, gj] = engine.out_host[2, m]
            peak_ratio[gi, gj] = engine.out_host[3, m]
            correlation_moment[gi, gj] = engine.out_host[4, m]
            if uncertainty_u !== nothing
                uncertainty_u[gi, gj] = finalize_uncertainty(T,
                    view(engine.uqstats_host, 1, :, m))
                uncertainty_v[gi, gj] = finalize_uncertainty(T,
                    view(engine.uqstats_host, 2, :, m))
            end
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

function uncertainty_sweep!(uncertainty_u, uncertainty_v, jobs, imgA, imgB,
                            params::PIVParameters, apod, mask,
                            engine::_CUDACorrelationEngine{T}) where {T}
    jobvec = jobs isa AbstractVector ? jobs : collect(jobs)
    isempty(jobvec) && return nothing
    wr, wc = engine.wsize
    bs = min(_CUDA_BATCH, length(jobvec))
    _ensure_batch!(engine, bs)
    A_d, B_d, maskarg, hasmask = _stage_pair!(engine, imgA, imgB, mask)
    ka = CUDABackend()
    for start in 1:bs:length(jobvec)
        nreal = min(bs, length(jobvec) - start + 1)
        for m in 1:nreal
            job = jobvec[start + m - 1]
            engine.origins_host[m, 1] = job[3]
            engine.origins_host[m, 2] = job[4]
        end
        copyto!(engine.origins_d, engine.origins_host)
        _ka_window_means!(ka)(engine.meanA_d, engine.meanB_d, A_d, B_d,
            engine.origins_d, maskarg, hasmask, wr, wc; ndrange = nreal)
        _ka_gather!(ka)(engine.CA, engine.CB, A_d, B_d, engine.origins_d,
            engine.apod_d, engine.meanA_d, engine.meanB_d, maskarg, hasmask;
            ndrange = (wr, wc, nreal))
        _ka_uq_fill!(ka)(engine.uqdcs_d, engine.uqmeans_d, engine.CA, engine.CB,
                         wr, wc; ndrange = (2, nreal))
        _ka_uq_stats!(ka)(engine.uqstats_d, engine.uqmeans_d, engine.uqdcs_d,
                          engine.CA, engine.CB, wr, wc, nreal, 0, false;
                          ndrange = (2, UQ_NSTATS, nreal))
        KernelAbstractions.synchronize(ka)
        copyto!(engine.uqstats_host, engine.uqstats_d)
        for m in 1:nreal
            gi, gj = jobvec[start + m - 1][1:2]
            uncertainty_u[gi, gj] = finalize_uncertainty(T, view(engine.uqstats_host, 1, :, m))
            uncertainty_v[gi, gj] = finalize_uncertainty(T, view(engine.uqstats_host, 2, :, m))
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Ensemble (sum-of-correlation): the cross-pair plane accumulator is a single
# plane-major device array resident for the whole ensemble pass — each pair's
# planes are added in place by `_ka_shiftgain_accum!` and only the packed
# per-window scalars of the final analysis return to the host (the plan's
# ensemble dataflow: image pair in, device-side accumulate, vector grid out).
# Device memory holds all `njobs` summed planes: ~njobs*nr*nc*sizeof(T).

import Hammerhead: _KAPlaneAccumulator, _plane_accumulator, accumulate_planes!,
    ensemble_analyze!, _ka_shiftgain_accum!, _KAUQAccumulator,
    _uncertainty_accumulator, _uncertainty_scratch

_uncertainty_accumulator(::_CUDACorrelationEngine, ::Type, njobs::Int) =
    _KAUQAccumulator(CUDA.zeros(Float64, 2, UQ_NSTATS, njobs))
_uncertainty_scratch(::_CUDACorrelationEngine, ::Type) = nothing

function _plane_accumulator(engine::_CUDACorrelationEngine{T}, params::PIVParameters,
                            ::Type{T}, njobs::Int) where {T}
    nr, nc = engine.fft_size
    # Odd trailing dimension: keeps the plane-to-plane byte stride off
    # power-of-two multiples (the memory-channel conflict the Rt pad avoids).
    ld = njobs + (iseven(njobs) ? 1 : 0)
    return _KAPlaneAccumulator(CUDA.zeros(T, nr, nc, ld), njobs)
end

# One pair's contribution: identical staging to `process_windows!` up to the
# inverse FFT, then the shifted planes add straight into the accumulator.
function accumulate_planes!(acc::_KAPlaneAccumulator, jobrange::AbstractUnitRange,
                            engine::_CUDACorrelationEngine{T},
                            imgA::AbstractMatrix, imgB::AbstractMatrix, jobs,
                            params::PIVParameters, mask,
                            uacc = nothing, uscratch = nothing) where {T}
    njobs = length(jobrange)
    njobs == 0 && return nothing

    wr, wc = engine.wsize
    nr, nc = engine.fft_size
    sr, sc = nr ÷ 2, nc ÷ 2
    bs = min(_CUDA_BATCH, njobs)
    _ensure_batch!(engine, bs)
    A_d, B_d, maskarg, hasmask = _stage_pair!(engine, imgA, imgB, mask)
    ka = CUDABackend()

    gainarg = engine.padded ? engine.gain_d : engine.apod_d   # dummy when unpadded (unread)

    for start in 1:bs:njobs
        nreal = min(bs, njobs - start + 1)
        @inbounds for m in 1:nreal
            job = jobs[jobrange[start + m - 1]]
            engine.origins_host[m, 1] = job[3]   # rs
            engine.origins_host[m, 2] = job[4]   # cs
        end
        copyto!(engine.origins_d, engine.origins_host)
        fill!(engine.CA, 0)
        fill!(engine.CB, 0)
        _ka_window_means!(ka)(engine.meanA_d, engine.meanB_d, A_d,
                              B_d, engine.origins_d, maskarg, hasmask,
                              wr, wc; ndrange = nreal)
        _ka_gather!(ka)(engine.CA, engine.CB, A_d, B_d,
                        engine.origins_d, engine.apod_d, engine.meanA_d,
                        engine.meanB_d, maskarg, hasmask; ndrange = (wr, wc, nreal))
        if uacc !== nothing
            _ka_uq_fill!(ka)(engine.uqdcs_d, engine.uqmeans_d, engine.CA, engine.CB,
                             wr, wc; ndrange = (2, nreal))
            _ka_uq_stats!(ka)(uacc.stats, engine.uqmeans_d, engine.uqdcs_d,
                              engine.CA, engine.CB, wr, wc, nreal,
                              first(jobrange) + start - 2, true;
                              ndrange = (2, UQ_NSTATS, nreal))
        end
        KernelAbstractions.synchronize(ka)
        engine.fwd * engine.CA
        engine.fwd * engine.CB
        _device_spectrum!(engine, params, ka)
        KernelAbstractions.synchronize(ka)
        engine.bwd * engine.CA
        # Only the nreal live slices accumulate — the last tile's tail holds
        # stale spectra that must not pollute the sums.
        _ka_shiftgain_accum!(ka)(acc.Racc, engine.CA, gainarg, engine.padded, sr, sc,
                                 nr, nc, first(jobrange) + start - 2;
                                 ndrange = (nr, nc, nreal))
        KernelAbstractions.synchronize(ka)
    end
    return nothing
end

# Ensemble finalize: analyze the summed planes in tiles on the device and
# scatter the packed scalars — residuals *add to* the shared predictor in
# `u`/`v` and alternatives are predictor-relative, exactly like the host loop.
function ensemble_analyze!(acc::_KAPlaneAccumulator, engine::_CUDACorrelationEngine{T},
                           u, v, peak_ratio, correlation_moment,
                           uncertainty_u, uncertainty_v, planes, alt_u, alt_v,
                           jobs, params::PIVParameters, uacc) where {T}
    planes === nothing ||
        throw(ArgumentError("backend :cuda does not support correlation-plane storage yet; " *
                            "use backend = :cpu"))
    njobs = acc.njobs
    njobs == 0 && return nothing
    nr, nc = engine.fft_size
    bs = min(_CUDA_BATCH, njobs)
    _ensure_batch!(engine, bs)
    ka = CUDABackend()
    use_regionalmax = params.peak_finder === :regionalmax
    use_gauss9 = params.subpixel_method === :gauss9
    nalt = engine.kpk - 1
    uqhost = uacc === nothing ? nothing : Array(uacc.stats)
    for start in 1:bs:njobs
        nreal = min(bs, njobs - start + 1)
        Rv = view(acc.Racc, :, :, start:(start + nreal - 1))
        _ka_analyze!(ka, _WG_ANALYZE)(engine.out_d, engine.vals_d, engine.locs_d, Rv,
                         use_regionalmax, use_gauss9, params.n_peaks, nr, nc, Val(_WG_ANALYZE);
                         ndrange = _WG_ANALYZE * nreal)
        KernelAbstractions.synchronize(ka)
        copyto!(engine.out_host, engine.out_d)
        for m in 1:nreal
            job = jobs[start + m - 1]
            gi, gj = job[1], job[2]
            if alt_u !== nothing
                found = Int(engine.out_host[5, m])   # small integer, exact in T
                for mm in 2:min(found, params.n_peaks)
                    # Total alternative displacement = shared predictor + residual.
                    alt_u[gi, gj, mm - 1] = u[gi, gj] + engine.out_host[5 + (mm - 1), m]
                    alt_v[gi, gj, mm - 1] = v[gi, gj] + engine.out_host[5 + nalt + (mm - 1), m]
                end
            end
            u[gi, gj] += engine.out_host[1, m]
            v[gi, gj] += engine.out_host[2, m]
            peak_ratio[gi, gj] = engine.out_host[3, m]
            correlation_moment[gi, gj] = engine.out_host[4, m]
            if uqhost !== nothing
                j = start + m - 1
                uncertainty_u[gi, gj] = finalize_uncertainty(T, view(uqhost, 1, :, j))
                uncertainty_v[gi, gj] = finalize_uncertainty(T, view(uqhost, 2, :, j))
            end
        end
    end
    return nothing
end

end # module
