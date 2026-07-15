module HammerheadAMDGPUExt

# AMD ROCm/HIP device backend (`backend = :amdgpu`) for interrogation-window
# correlation, built on AMDGPU.jl + KernelAbstractions. It reuses the portable
# correlation and plane-analysis kernels defined in the core package
# (`src/ka_backend.jl`, the `backend = :ka` proving tier); the only
# device-specific work here is host<->device staging and the rocFFT batched
# transform. The whole pipeline —
# predictor deformation (from once-per-call staged B-spline coefficients),
# gather, FFTs, cross-power, shift/gain, peak finding + subpixel + moment —
# runs on the device; the warped images stay device-resident between
# deformation and correlation, and only the packed per-window scalars come
# back to the host.
#
# STATUS: validated on an RX 6800 XT (gfx1030) against `backend = :cpu`
# (max deviation ~1e-14 in Float64, single-pass/multipass/masked), and 1.3-2.4x
# faster than the 4-thread CPU path on 1024²-2048² single-pass benches
# (`bench/gpu_benchmarks.jl`). Requires
# ROCm 6.4 — ROCm/HIP 7.1 dropped RDNA2 support on Windows, so point
# HIP_PATH/ROCM_PATH/PATH at the 6.4 install. rocFFT in-place plans apply via
# `p * x` (they lack FFTW's 3-arg `mul!`), and masks must be materialized as
# dense `Array{Bool}` before `copyto!` to the device.
#
# Batch buffers use a byte-budgeted workspace cache: configurations share the
# available budget, retain buffers when they fit, and evict least-recently-used
# buffers when they do not. Calls without a workspace release every stage at
# pass completion. This policy is driven by FFT size, precision, free/total
# VRAM, job count, and the discovered configuration count rather than a fixed
# GPU model or pass schedule.

using Hammerhead
using AMDGPU
# Strong deps of the core package, usable here without being ext triggers.
using KernelAbstractions
using AbstractFFTs: plan_fft!, plan_bfft!, ScaledPlan

import Hammerhead: _AbstractHammerheadBackend, _resolve_backend, _check_backend_params,
    _engine_nchunks, piv_correlation_engines, pooled_engines, process_windows!,
    make_correlator, PIVParameters, _supports_fft, _supports_batched_fft,
    _supports_fp64, finalize_uncertainty, UQ_NSTATS, uncertainty_sweep!,
    _correlation_apod, release_piv_engines!

# Portable correlation kernels + scope guard from the core (src/ka_backend.jl).
import Hammerhead: _ka_scope_check, _ka_window_means!, _ka_gather!,
    _ka_crosspower!, _ka_phasepower!, _ka_shiftgain!, _ka_analyze!
import Hammerhead: _ka_uq_fill!, _ka_uq_stats!

# Device-resident deformation (Phase 3b): the staged-context machinery is
# portable core code (proven on `:ka`); this extension only points it at the
# ROC backend.
import Hammerhead: apply_predictor, _deform_context, _ka_deform_context,
    _ka_apply_predictor_ctx

struct _AMDGPUBackend <: _AbstractHammerheadBackend end

_resolve_backend(::Val{:amdgpu}) = _AMDGPUBackend()

# Run the whole window grid as one logical batch, tiled internally.
_engine_nchunks(::_AMDGPUBackend, ::Int) = 1
_supports_fft(::_AMDGPUBackend) = true
_supports_batched_fft(::_AMDGPUBackend) = true
_supports_fp64(::_AMDGPUBackend) = true

_check_backend_params(b::_AMDGPUBackend, passes) =
    _ka_scope_check(passes, :amdgpu, _supports_fp64(b))

# Windows per device sub-batch. The analysis kernel launches one *workgroup*
# per window, so the sub-batch size *is* that kernel's launch parallelism (in
# workgroups): bigger is better for occupancy until the batch buffers spill
# VRAM to shared/system memory. The spill driver is the correlation buffers at
# the first pass's fft_size, not rocFFT workspace (zero bytes for the
# power-of-two transforms used here). `_amdgpu_batch_cap` therefore sizes the
# batch from the engine's exact per-window footprint and currently free VRAM,
# preserving the occupancy-optimal default when it fits.
const _AMDGPU_BATCH_DEFAULT = 8192
const _AMDGPU_MEM_FRACTION = 0.7    # total VRAM available to cached batch buffers
const _AMDGPU_MIN_BATCH = 256       # floor: keep enough windows for occupancy
const _AMDGPU_BATCH_QUANTUM = 512   # avoid reallocations from free-VRAM wobble

# KernelAbstractions' occupancy-based default workgroup size leaves the
# elementwise cross-power off peak bandwidth. An explicit wavefront-multiple
# block recovers large factors on NVIDIA; workgroup size never affects results
# — every work-item is independent — so this is pure tuning. Re-tune on AMD
# hardware if a sweep shows a better size. (The analysis kernel is cooperative:
# its groupsize is fixed by `Val{TPW}`, not tuned here.)
const _WG_ELEMENTWISE = 256   # cross-power over the whole complex batch

# Threads per window for the cooperative `_ka_analyze!` (one workgroup per
# window; see the core kernel). The launch groupsize must equal the kernel's
# compile-time `Val{TPW}`; the core sets `_KA_TPW = 128`.
const _WG_ANALYZE = Hammerhead._KA_TPW

# Batch buffers are the dominant allocation and differ by window configuration.
# A workspace-level LRU lets configurations that fit coexist, while immediately
# releasing cold buffers when the next stage needs their VRAM. Engine metadata,
# fixed window arrays, and rocFFT handles remain cached.
mutable struct _AMDGPUBatchCache
    engines::Vector{Any}
    clock::UInt64
end

_AMDGPUBatchCache() = _AMDGPUBatchCache(Any[], 0)

mutable struct _AMDGPUCorrelationEngine{T}
    cache::_AMDGPUBatchCache
    stamp::UInt64
    wsize::NTuple{2,Int}
    fft_size::NTuple{2,Int}
    padded::Bool
    kpk::Int                              # peaks located per window, max(n_peaks, 2)
    apod_d::ROCArray{T,2}                 # window-sized (device)
    gain_d::ROCArray{T,2}                 # fftshifted gain (device); empty if unpadded
    phase_filter_d::ROCArray{T,2}         # Gaussian FFT weights; empty for cross-correlation
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
    Rt::ROCArray{T,3}                     # (nr, nc, bs+1) plane-major plane batch
    meanA_d::ROCArray{T,1}                # per-window means (device reduction)
    meanB_d::ROCArray{T,1}
    vals_d::ROCArray{T,2}                 # (kpk, bs) peak-finder scratch
    locs_d::ROCArray{Int32,3}             # (2, kpk, bs) peak-finder scratch
    out_d::ROCArray{T,2}                  # (5 + 2*(kpk-1), bs) packed analysis output
    out_host::Matrix{T}
    uqstats_d::ROCArray{Float64,3}
    uqmeans_d::ROCArray{Float64,2}
    uqdcs_d::ROCArray{T,4}                # (bs+1, 2, mm, mm) cached smoothed ΔC
    uqstats_host::Array{Float64,3}
    origins_host::Matrix{Int}
    origins_d::ROCArray{Int,2}
    fwd::Any
    bwd::Any
end

_correlation_apod(engine::_AMDGPUCorrelationEngine) = engine.apod_d

# Exact dominant per-window device-byte footprint of the sub-batch buffers:
# CA + CB (Complex{T}), Rt (T), and the UQ ΔC cache uqdcs (2·mm²·T, where mm
# is the larger window side). Small per-window scalar buffers are covered by
# the memory-fraction headroom.
_amdgpu_bytes_per_window(engine::_AMDGPUCorrelationEngine{T}) where {T} =
    prod(engine.fft_size) * (2 * sizeof(Complex{T}) + sizeof(T)) +
    2 * max(engine.wsize...)^2 * sizeof(T)

# Immediately release an engine's batch-dependent buffers. The fixed window
# arrays and engine object stay cached, and AMDGPU's rocFFT handle cache can
# reuse the released plans when this configuration becomes hot again.
function _amdgpu_release_batch!(engine::_AMDGPUCorrelationEngine{T}) where {T}
    engine.bs == 0 && return engine
    for a in (engine.CA, engine.CB, engine.Rt, engine.meanA_d, engine.meanB_d,
              engine.vals_d, engine.locs_d, engine.out_d, engine.uqstats_d,
              engine.uqmeans_d, engine.uqdcs_d, engine.origins_d)
        AMDGPU.unsafe_free!(a)
    end
    engine.fwd === nothing || AMDGPU.unsafe_free!(engine.fwd)
    engine.bwd === nothing || AMDGPU.unsafe_free!(engine.bwd.p)
    engine.CA = AMDGPU.zeros(Complex{T}, 0, 0, 0)
    engine.CB = AMDGPU.zeros(Complex{T}, 0, 0, 0)
    engine.Rt = AMDGPU.zeros(T, 0, 0, 0)
    engine.meanA_d = AMDGPU.zeros(T, 0)
    engine.meanB_d = AMDGPU.zeros(T, 0)
    engine.vals_d = AMDGPU.zeros(T, 0, 0)
    engine.locs_d = AMDGPU.zeros(Int32, 2, 0, 0)
    engine.out_d = AMDGPU.zeros(T, 0, 0)
    engine.out_host = Matrix{T}(undef, 0, 0)
    engine.uqstats_d = AMDGPU.zeros(Float64, 0, 0, 0)
    engine.uqmeans_d = AMDGPU.zeros(Float64, 0, 0)
    engine.uqdcs_d = AMDGPU.zeros(T, 0, 0, 0, 0)
    engine.uqstats_host = Array{Float64,3}(undef, 0, 0, 0)
    engine.origins_host = Matrix{Int}(undef, 0, 2)
    engine.origins_d = AMDGPU.zeros(Int, 0, 0)
    engine.fwd = nothing
    engine.bwd = nothing
    engine.bs = 0
    return engine
end

function release_piv_engines!(::_AMDGPUBackend, engines, workspace)
    workspace === nothing || return nothing
    AMDGPU.synchronize()
    foreach(_amdgpu_release_batch!, engines)
    AMDGPU.reclaim()
    return nothing
end

# Size the current stage against one aggregate cache budget. Cached engines
# count as reclaimable memory; least-recently-used batch buffers are evicted
# until the requested stage fits. `HAMMERHEAD_AMDGPU_BATCH` still forces the
# per-stage target for benchmarking, but does not disable eviction of cold
# configurations. `njobs` avoids reserving buffers for windows that do not
# exist in this stage.
function _amdgpu_batch_cap(engine::_AMDGPUCorrelationEngine{T}, njobs::Int) where {T}
    cache = engine.cache
    bpw = _amdgpu_bytes_per_window(engine)
    cached = sum((e.bs * _amdgpu_bytes_per_window(e) for e in cache.engines); init = 0)
    reserve = (1 - _AMDGPU_MEM_FRACTION) * AMDGPU.total()
    pool_budget = max(0, AMDGPU.free() + cached - reserve)
    # After a workspace has discovered its window configurations, divide the
    # aggregate pool fairly so cyclic multipass schedules can retain all of
    # them instead of having LRU thrash the first (usually largest) stage.
    # Single-pass workloads still receive the entire pool; stages whose job
    # count/default ceiling needs less than their share leave unused headroom.
    stage_budget = pool_budget / max(1, length(cache.engines))

    ov = get(ENV, "HAMMERHEAD_AMDGPU_BATCH", "")
    if isempty(ov)
        cap = floor(Int, stage_budget / bpw)
        cap = (cap ÷ _AMDGPU_BATCH_QUANTUM) * _AMDGPU_BATCH_QUANTUM
        target = min(njobs, clamp(cap, _AMDGPU_MIN_BATCH, _AMDGPU_BATCH_DEFAULT))
    else
        target = min(njobs, parse(Int, ov))
    end

    target_bytes = target * bpw
    others = [e for e in cache.engines if e !== engine && e.bs > 0]
    active = sum((e.bs * _amdgpu_bytes_per_window(e) for e in others); init = 0)
    to_evict = Any[]
    for cold in sort(others; by = e -> e.stamp)
        active + target_bytes <= pool_budget && break
        active -= cold.bs * _amdgpu_bytes_per_window(cold)
        push!(to_evict, cold)
    end
    replacing = engine.bs != 0 && engine.bs != target
    if !isempty(to_evict) || replacing
        AMDGPU.synchronize()
        foreach(_amdgpu_release_batch!, to_evict)
        replacing && _amdgpu_release_batch!(engine)
        # Return evicted pool blocks to HIP so subsequent free-memory queries
        # reflect them; the next allocation can still reuse cached rocFFT handles.
        AMDGPU.reclaim()
    end
    cache.clock += 1
    engine.stamp = cache.clock
    return target
end

function _make_amdgpu_engine(params::PIVParameters, ::Type{T},
                             cache::_AMDGPUBatchCache = _AMDGPUBatchCache()) where {T}
    # Reuse the CPU correlator's apodization window and overlap-gain plane so
    # those factors are bit-identical to the FFTW path, then upload them once.
    cpu = make_correlator(params, T)
    apod_d = ROCArray{T}(Matrix{T}(cpu.apod))
    fft_size = size(cpu.R)
    padded = !isempty(cpu.gain)
    gain_d = padded ? ROCArray{T}(Matrix{T}(cpu.gain)) : AMDGPU.zeros(T, 0, 0)
    phase_filter_d = params.correlation_method === :phase ?
        ROCArray{T}(Matrix{T}(cpu.W)) : AMDGPU.zeros(T, 0, 0)
    engine = _AMDGPUCorrelationEngine{T}(
        cache, 0, params.window_size, fft_size, padded, max(params.n_peaks, 2),
        apod_d, gain_d, phase_filter_d, AMDGPU.zeros(Bool, 0, 0),
        (0, 0), AMDGPU.zeros(T, 0, 0), AMDGPU.zeros(T, 0, 0), AMDGPU.zeros(Bool, 0, 0),
        0, AMDGPU.zeros(Complex{T}, 0, 0, 0), AMDGPU.zeros(Complex{T}, 0, 0, 0),
        AMDGPU.zeros(T, 0, 0, 0), AMDGPU.zeros(T, 0), AMDGPU.zeros(T, 0),
        AMDGPU.zeros(T, 0, 0), AMDGPU.zeros(Int32, 2, 0, 0), AMDGPU.zeros(T, 0, 0),
        Matrix{T}(undef, 0, 0), AMDGPU.zeros(Float64, 0, 0, 0),
        AMDGPU.zeros(Float64, 0, 0), AMDGPU.zeros(T, 0, 0, 0, 0),
        Array{Float64,3}(undef, 0, 0, 0),
        Matrix{Int}(undef, 0, 2), AMDGPU.zeros(Int, 0, 0), nothing, nothing)
    push!(cache.engines, engine)
    return engine
end

# Engines are pooled per window configuration via the workspace (see the core
# `pooled_engines`), so device buffers and rocFFT plans are paid once per
# configuration for a whole sequence/ensemble batch.
function piv_correlation_engines(::_AMDGPUBackend, workspace, params::PIVParameters,
                                 ::Type{T}, nchunks::Int) where {T}
    cachekey = (:amdgpu_batch_cache, T)
    cache = if workspace === nothing
        _AMDGPUBatchCache()
    else
        pool = get!(() -> Any[], workspace.engines, cachekey)
        isempty(pool) && push!(pool, _AMDGPUBatchCache())
        pool[1]
    end
    return pooled_engines(() -> _make_amdgpu_engine(params, T, cache), workspace,
                          (:amdgpu, T, params.correlation_method, params.window_size,
                           params.padding, params.apodization, max(params.n_peaks, 2)),
                          nchunks)
end

function _device_spectrum!(engine::_AMDGPUCorrelationEngine, params::PIVParameters, ka)
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

function _ensure_batch!(engine::_AMDGPUCorrelationEngine{T}, bs::Int) where {T}
    if engine.bs != bs
        nr, nc = engine.fft_size
        engine.CA = AMDGPU.zeros(Complex{T}, nr, nc, bs)
        engine.CB = AMDGPU.zeros(Complex{T}, nr, nc, bs)
        # Trailing dimension padded by one plane: at a power-of-two batch size
        # an exact power-of-two plane-to-plane byte stride funnels a whole
        # wavefront into one memory channel (measured ~20x slowdown). Plane
        # bs+1 is never touched.
        engine.Rt = AMDGPU.zeros(T, nr, nc, bs + 1)
        engine.meanA_d = AMDGPU.zeros(T, bs)
        engine.meanB_d = AMDGPU.zeros(T, bs)
        engine.vals_d = AMDGPU.zeros(T, engine.kpk, bs)
        engine.locs_d = AMDGPU.zeros(Int32, 2, engine.kpk, bs)
        engine.out_d = AMDGPU.zeros(T, 5 + 2 * (engine.kpk - 1), bs)
        engine.out_host = Matrix{T}(undef, 5 + 2 * (engine.kpk - 1), bs)
        engine.uqstats_d = AMDGPU.zeros(Float64, 2, UQ_NSTATS, bs)
        engine.uqmeans_d = AMDGPU.zeros(Float64, 2, bs)
        mm = max(engine.wsize...)
        engine.uqdcs_d = AMDGPU.zeros(T, bs + 1, 2, mm, mm)  # +1: channel-conflict pad
        engine.uqstats_host = Array{Float64,3}(undef, 2, UQ_NSTATS, bs)
        engine.origins_host = Matrix{Int}(undef, bs, 2)
        engine.origins_d = AMDGPU.zeros(Int, bs, 2)
        engine.fwd = plan_fft!(engine.CA, (1, 2))
        # AMDGPU's `inv(::cROCFFTPlan)` allocates a full-size temporary array
        # only to compute this normalization. At production batch sizes that
        # unreachable array can occupy several GiB until Julia GC runs. Build
        # the backward plan directly and apply the known FFT scale instead.
        engine.bwd = ScaledPlan(plan_bfft!(engine.CA, (1, 2)),
                                inv(T(prod(engine.fft_size))))
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

# Deformation staging: the prefiltered B-spline coefficients go to the device
# once per `run_piv` call (per pair in the ensemble driver) and the deform
# context's warp buffers stay resident, so per-sweep deformation transfers
# only the coarse predictor grid (see `_ka_deform_context` in the core).
_deform_context(::_AMDGPUBackend, workspace, itpA, itpB,
                imgsize::Dims{2}, ::Type{T}) where {T} =
    _ka_deform_context(ROCBackend(), workspace, (:amdgpu_deform, T, imgsize),
                       itpA, itpB, imgsize, T)

function apply_predictor(::_AMDGPUBackend, imgA::AbstractMatrix, imgB::AbstractMatrix,
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
function _stage_pair!(engine::_AMDGPUCorrelationEngine{T}, imgA, imgB, mask) where {T}
    hasmask = mask !== nothing
    if imgA isa ROCArray
        A_d, B_d = imgA, imgB
        hasmask && size(engine.mask_d) != size(imgA) &&
            (engine.mask_d = AMDGPU.zeros(Bool, size(imgA)...))
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
                          engine::_AMDGPUCorrelationEngine{T},
                          mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                          planes = nothing) where {T}
    planes === nothing ||
        throw(ArgumentError("backend :amdgpu does not support correlation-plane storage yet; " *
                            "use backend = :cpu"))
    jobvec = jobs isa AbstractVector ? jobs : collect(jobs)
    njobs = length(jobvec)
    njobs == 0 && return nothing

    wr, wc = engine.wsize
    nr, nc = engine.fft_size
    sr, sc = nr ÷ 2, nc ÷ 2
    bs = _amdgpu_batch_cap(engine, njobs)
    _ensure_batch!(engine, bs)
    A_d, B_d, maskarg, hasmask = _stage_pair!(engine, imgA, imgB, mask)
    ka = ROCBackend()

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
        # In-place plan application: `p * x` mutates x and returns it, the
        # idiom shared by FFTW and rocFFT in-place plans (rocFFT does not
        # implement the 3-arg `mul!(x, p, x)` that FFTW does).
        engine.fwd * engine.CA
        engine.fwd * engine.CB
        _device_spectrum!(engine, params, ka)
        KernelAbstractions.synchronize(ka)
        engine.bwd * engine.CA
        _ka_shiftgain!(ka)(engine.Rt, engine.CA, gainarg, engine.padded, sr, sc, nr, nc;
                           ndrange = (nr, nc, bs))
        # Same queue as the shift/gain kernel, so no synchronize between them.
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
                            engine::_AMDGPUCorrelationEngine{T}) where {T}
    jobvec = jobs isa AbstractVector ? jobs : collect(jobs)
    isempty(jobvec) && return nothing
    wr, wc = engine.wsize
    bs = _amdgpu_batch_cap(engine, length(jobvec))
    _ensure_batch!(engine, bs)
    A_d, B_d, maskarg, hasmask = _stage_pair!(engine, imgA, imgB, mask)
    ka = ROCBackend()
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

_uncertainty_accumulator(::_AMDGPUCorrelationEngine, ::Type, njobs::Int) =
    _KAUQAccumulator(AMDGPU.zeros(Float64, 2, UQ_NSTATS, njobs))
_uncertainty_scratch(::_AMDGPUCorrelationEngine, ::Type) = nothing

function _plane_accumulator(engine::_AMDGPUCorrelationEngine{T}, params::PIVParameters,
                            ::Type{T}, njobs::Int) where {T}
    nr, nc = engine.fft_size
    # Odd trailing dimension: keeps the plane-to-plane byte stride off
    # power-of-two multiples (the memory-channel conflict the Rt pad avoids).
    ld = njobs + (iseven(njobs) ? 1 : 0)
    return _KAPlaneAccumulator(AMDGPU.zeros(T, nr, nc, ld), njobs)
end

# One pair's contribution: identical staging to `process_windows!` up to the
# inverse FFT, then the shifted planes add straight into the accumulator.
function accumulate_planes!(acc::_KAPlaneAccumulator, jobrange::AbstractUnitRange,
                            engine::_AMDGPUCorrelationEngine{T},
                            imgA::AbstractMatrix, imgB::AbstractMatrix, jobs,
                            params::PIVParameters, mask,
                            uacc = nothing, uscratch = nothing) where {T}
    njobs = length(jobrange)
    njobs == 0 && return nothing

    wr, wc = engine.wsize
    nr, nc = engine.fft_size
    sr, sc = nr ÷ 2, nc ÷ 2
    bs = _amdgpu_batch_cap(engine, njobs)
    _ensure_batch!(engine, bs)
    A_d, B_d, maskarg, hasmask = _stage_pair!(engine, imgA, imgB, mask)
    ka = ROCBackend()

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
function ensemble_analyze!(acc::_KAPlaneAccumulator, engine::_AMDGPUCorrelationEngine{T},
                           u, v, peak_ratio, correlation_moment,
                           uncertainty_u, uncertainty_v, planes, alt_u, alt_v,
                           jobs, params::PIVParameters, uacc) where {T}
    planes === nothing ||
        throw(ArgumentError("backend :amdgpu does not support correlation-plane storage yet; " *
                            "use backend = :cpu"))
    njobs = acc.njobs
    njobs == 0 && return nothing
    nr, nc = engine.fft_size
    bs = _amdgpu_batch_cap(engine, njobs)
    _ensure_batch!(engine, bs)
    ka = ROCBackend()
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
