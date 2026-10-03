# Ensemble (sum-of-correlation) PIV: average the correlation planes of
# corresponding windows across many image pairs before peak detection
# (Meinhart, Wereley & Santiago 2000). For statistically stationary flow
# whose individual pairs are too noisy for reliable peaks — micro-PIV and
# other low-SNR recordings.

# Saved ensemble replay uses contribution-boundary hooks. They carry no result
# arrays and do not change the public driver meter or ordinary arithmetic.
struct _EnsembleCancelled <: Exception end
Base.showerror(io::IO,::_EnsembleCancelled)=print(io,"ensemble replay cancelled before pooled publication")
function _ensemble_replay_hooks(on_contribution,cancel_requested)
    on_contribution===nothing || on_contribution isa Function || throw(ArgumentError("_on_contribution must be a function or nothing"))
    cancel_requested===nothing || cancel_requested isa Function || throw(ArgumentError("_cancel_requested must be a function or nothing"))
    nothing
end
function _ensemble_cancel_check(cancel_requested)
    cancel_requested===nothing && return nothing
    requested=cancel_requested()
    requested isa Bool || throw(ArgumentError("cancel_requested must return Bool"))
    requested && throw(_EnsembleCancelled())
    nothing
end

"""
    run_piv_ensemble(pairs, params = PIVParameters(); kwargs...) -> PIVResult
    run_piv_ensemble(pairs; effort = :low/:medium/:high, kwargs...) -> PIVResult

Ensemble PIV over a sequence of image pairs: each interrogation window's
correlation planes are summed across all pairs and the displacement peak is
located once on the ensemble plane. Combining pairs can reveal a peak that
is too weak to detect in individual pairs. Use a statistically stationary
interval and inspect the combined peak: if displacements vary widely, its
position need not equal the arithmetic mean of separately measured vectors.

`pairs` is as in [`run_piv_sequence`](@ref) (2-tuples of file paths and/or
matrices; paths are reloaded once per pass). `params` may be a single
[`PIVParameters`](@ref), an explicit multi-pass schedule, or omitted in favor
of `effort = :low`, `:medium`, or `:high`; effort keywords follow
[`run_piv`](@ref), except that ensemble `:high` repeats the final window size
because this driver ignores per-pass `max_iterations`. With a multi-pass
schedule the previous pass's ensemble field acts as a shared deformation
predictor for every pair. `peak_ratio` and `correlation_moment` describe the
ensemble planes.

With `uncertainty = true` the correlation-statistics estimator (Wieneke 2015,
see [`PIVParameters`](@ref)) pools its per-window sums over all pairs.
`uncertainty_u` / `uncertainty_v` describe random uncertainty in the combined
displacement estimate. Adding pairs can reduce this uncertainty; a decrease
at every sample count is not guaranteed. The estimator assumes a common
displacement across pairs and does not quantify flow fluctuations. Use
[`field_statistics`](@ref) over single-pair results for those statistics.

Keyword arguments `threaded`, `predictor_smoothing`, `mask`,
`mask_threshold`, `backend`, and `scale` follow [`run_piv`](@ref).
`preprocess` is applied to each loaded frame before analysis; `image_type`
sets the loaded image precision, and `progress` controls the progress display,
as in [`run_piv_sequence`](@ref). `subpixel_method = :gauss2d` and
`keep_correlation_planes = true` require `backend = :cpu`.

`on_diagnostics(d)` observes a separate immutable ensemble execution packet;
`output=path, record_diagnostics=true` saves it beside one native result. Capture
supports CPU/KA and reports one pooled sweep/pass, ignored iteration settings,
numerical plane/source-support populations and pre-addition pooled residuals.
It certifies neither stationarity nor independent samples or convergence.
Known input aliases and invalid recording options reject before image loading.
"""
function run_piv_ensemble(pairs::AbstractVector,
                          params::Union{PIVParameters,AbstractVector{PIVParameters}};
                          effort::Union{Nothing,Symbol} = nothing,
                          backend::Symbol = :cpu,
                          threaded::Bool = Threads.nthreads() > 1,
                          predictor_smoothing::Bool = true,
                          mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                          mask_threshold::Real = 0.5,
                          preprocess = nothing,
                          image_type::Type{<:AbstractFloat} = Float64,
                          progress::Bool = true,
                          scale::Union{Nothing,PhysicalScale} = nothing,
                          on_diagnostics = nothing, output = nothing,
                          record_diagnostics = false,
                          _on_contribution = nothing, _cancel_requested = nothing)
    _ensemble_replay_hooks(_on_contribution,_cancel_requested)
    options=_ensemble_options(pairs,backend,image_type,on_diagnostics,output,record_diagnostics)
    effort === nothing ||
        throw(ArgumentError("effort cannot be combined with explicit PIVParameters or pass schedules"))
    be = _resolve_backend(backend)
    passes = params isa PIVParameters ? [params] : params
    _check_backend_params(be, passes)
    isempty(pairs) && throw(ArgumentError("pairs must not be empty"))
    isempty(passes) && throw(ArgumentError("at least one pass is required"))
    0 < mask_threshold <= 1 ||
        throw(ArgumentError("mask_threshold must be in (0, 1], got $mask_threshold"))

    if options.capture || output!==nothing || _on_contribution!==nothing || _cancel_requested!==nothing
        # Freeze selection/container metadata, not mutable pixel contents.
        pairs=[(pair[1],pair[2]) for pair in pairs]
        mask=mask===nothing ? nothing : BitMatrix(mask)
        passes=copy(passes)
    end
    _ensemble_cancel_check(_cancel_requested)
    total_contributions=_on_contribution===nothing ? nothing : _ensemble_mul(length(passes),length(pairs))
    reports=options.capture ? EnsemblePassDiagnostics[] : nothing
    source_before=options.capture ? _experiment_software()["core_source_sha256"] : nothing

    meter = Progress(length(passes) * length(pairs);
                     desc = "Ensemble PIV: ", enabled = progress)
    # Reuse the image-interpolant and deformation buffers across every pair and
    # pass (the pairs share one image size). The ensemble path already reuses
    # its correlators per pass, so only the interpolant/warp scratch is routed
    # through the workspace here.
    workspace = piv_workspace(; backend)
    result = nothing
    for (k, p) in enumerate(passes)
        predictor = result === nothing ? nothing :
                    build_predictor(result, predictor_smoothing)
        settings=(;threaded,mask,mask_threshold,preprocess,image_type,
            force_replace=k<length(passes),meter,workspace,backend=be,
            on_contribution=_on_contribution,cancel_requested=_cancel_requested,
            contribution_pass=k,scheduled_passes=length(passes),total_contributions)
        result = options.capture ? ensemble_pass(pairs,p,predictor;settings...,diagnostics=reports,pass_index=k) :
            ensemble_pass(pairs,p,predictor;settings...)
    end
    result=scale === nothing ? result : with_scale(result, scale)
    diagnostics=options.capture ? _ensemble_finish(reports,backend,image_type,length(pairs),result,source_before) : nothing
    on_diagnostics===nothing || on_diagnostics(diagnostics)
    diagnostics===nothing || _ensemble_check_result(diagnostics,result)
    _ensemble_cancel_check(_cancel_requested)
    if output!==nothing
        path=_ensemble_output_guard(output,options.inputs)
        diagnostics===nothing || _ensemble_check_result(diagnostics,result)
        jldopen(path,"w") do file
            file["format_version"]=RESULTS_FORMAT_VERSION
            file[result_key(1)]=result
            record_diagnostics && _write_ensemble_execution_diagnostics(file,result_key(1),diagnostics,result)
        end
    end
    return result
end

function run_piv_ensemble(pairs::AbstractVector; effort::Union{Nothing,Symbol} = nothing,
                          backend::Symbol = :cpu,
                          threaded::Bool = Threads.nthreads() > 1,
                          predictor_smoothing::Bool = true,
                          mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                          mask_threshold::Real = 0.5,
                          preprocess = nothing,
                          image_type::Type{<:AbstractFloat} = Float64,
                          progress::Bool = true,
                          scale::Union{Nothing,PhysicalScale} = nothing,
                          on_diagnostics = nothing, output = nothing,
                          record_diagnostics = false,
                          _on_contribution = nothing, _cancel_requested = nothing,
                          kwargs...)
    _ensemble_replay_hooks(_on_contribution,_cancel_requested)
    options=_ensemble_options(pairs,backend,image_type,on_diagnostics,output,record_diagnostics)
    if options.capture || output!==nothing || _on_contribution!==nothing || _cancel_requested!==nothing
        # Effort discovers image size via the public preprocessor too: freeze
        # frame selection and mask before that first load/callback boundary.
        pairs=[(pair[1],pair[2]) for pair in pairs]
        mask=mask===nothing ? nothing : BitMatrix(mask)
    end
    _ensemble_cancel_check(_cancel_requested)
    if effort === nothing
        isempty(kwargs) ||
            throw(ArgumentError("unsupported run_piv_ensemble keyword(s): " *
                                join(string.(keys(kwargs)), ", ")))
        return run_piv_ensemble(pairs, PIVParameters(); backend, threaded, predictor_smoothing,
                                mask, mask_threshold, preprocess, image_type, progress,
                                scale,on_diagnostics,output,record_diagnostics,_on_contribution,_cancel_requested)
    end
    piv_kwargs, driver_kwargs = split_effort_kwargs(kwargs)
    !isempty(driver_kwargs) &&
        throw(ArgumentError("unsupported run_piv_ensemble keyword(s): " *
                            join(string.(keys(driver_kwargs)), ", ")))
    imgsize = first_pair_image_size(pairs; preprocess, image_type)
    passes = effort_schedule(effort; ensemble = true, image_size = imgsize, piv_kwargs...)
    return run_piv_ensemble(pairs, passes; backend, threaded, predictor_smoothing, mask,
                            mask_threshold, preprocess, image_type, progress, scale,
                            on_diagnostics,output,record_diagnostics,_on_contribution,_cancel_requested)
end

function first_pair_image_size(pairs; preprocess, image_type)
    isempty(pairs) && throw(ArgumentError("pairs must not be empty"))
    frameA, frameB = first(pairs)
    imgA = load_frame(frameA, image_type)
    imgB = load_frame(frameB, image_type)
    if preprocess !== nothing
        imgA = preprocess(imgA)
        imgB = preprocess(imgB)
    end
    size(imgA) == size(imgB) ||
        throw(DimensionMismatch("images must have the same size, got $(size(imgA)) and $(size(imgB))"))
    return size(imgA)
end

# One ensemble pass: deform every pair by the shared predictor, sum each
# window's correlation planes across pairs, then peak-find and validate once.
function ensemble_pass(pairs, params::PIVParameters, predictor;
                       threaded::Bool, mask, mask_threshold, preprocess,
                       image_type, force_replace::Bool, meter, workspace = nothing,
                       backend::_AbstractHammerheadBackend = _DEFAULT_BACKEND,
                       diagnostics = nothing, pass_index = 1,
                       on_contribution = nothing, cancel_requested = nothing,
                       contribution_pass = 1, scheduled_passes = 1, total_contributions = nothing)
    local T, grid, accum, chunks, engines, u, v, imgsize, uacc, uscratch
    first_pair = true
    source_gate = nothing
    observation = nothing
    for (pair_index,pair) in enumerate(pairs)
        _ensemble_cancel_check(cancel_requested)
        frameA, frameB = pair
        imgA = load_frame(frameA, image_type)
        imgB = load_frame(frameB, image_type)
        if preprocess !== nothing
            imgA = preprocess(imgA)
            imgB = preprocess(imgB)
        end
        size(imgA) == size(imgB) ||
            throw(DimensionMismatch("images must have the same size, got $(size(imgA)) and $(size(imgB))"))
        if first_pair
            imgsize = size(imgA)
            mask === nothing || size(mask) == imgsize ||
                throw(DimensionMismatch("mask must have the same size as the images, got $(size(mask))"))
            T = float(promote_type(eltype(imgA), eltype(imgB)))
            grid = pass_grid(T, imgsize, params, mask, mask_threshold)
            if diagnostics!==nothing
                T in (Float32,Float64) || _execution_error("actual ensemble processing precision must be Float32/64 for capture")
                observation=_ensemble_observation(grid,T,length(pairs))
            end
            nchunks = _engine_nchunks(backend,
                                      threaded ? min(Threads.nthreads(), length(grid.jobs)) : 1)
            chunk_size = max(cld(length(grid.jobs), max(nchunks, 1)), 1)
            chunks = collect(Iterators.partition(1:length(grid.jobs), chunk_size))
            # One correlation engine per chunk, reused across pairs: CPU
            # engines wrap FFTW correlators whose plans are paid once per
            # configuration (pooled in the workspace), and each accumulator is
            # written by exactly one task per pair, so threaded results match
            # serial exactly. Device engines collapse to a single chunk and
            # tile the window grid internally.
            engines = piv_correlation_engines(backend, workspace, params, T, length(chunks))
            # Plane accumulators live where the engine computes: per-window
            # host matrices for the CPU path, one batch-major device array for
            # KA-family engines — so device backends accumulate in place
            # instead of copying planes back every pair.
            accum = isempty(engines) ? Matrix{T}[] :
                    _plane_accumulator(engines[1], params, T, length(grid.jobs))
            # Uncertainty statistics pool across pairs (per-window Float64
            # accumulators; each is written by exactly one task per pair).
            # As in the single-pair path, only the final pass estimates them.
            unc = params.uncertainty && !force_replace
            # `engines` (and `grid.jobs`) are empty only when every window is
            # masked out; guard the accumulator the same way as `accum` above so
            # that degenerate case returns an all-NaN grid instead of a
            # BoundsError. `uacc === nothing` then short-circuits
            # `accumulate_planes!`, and `ensemble_analyze!` is skipped on the
            # empty grid, so no window ever indexes it.
            uacc = unc && !isempty(engines) ?
                   _uncertainty_accumulator(engines[1], T, length(grid.jobs)) : nothing
            uscratch = unc ? [_uncertainty_scratch(e, T) for e in engines] : nothing
            workspace === nothing || ws_prepare!(workspace, imgsize, T)
            first_pair = false
        else
            size(imgA) == imgsize ||
                throw(DimensionMismatch("all pairs must share the image size $imgsize, got $(size(imgA))"))
        end
        # Images differ per pair, so the deformation interpolants are rebuilt
        # each pair — but only when there is a predictor to deform by. With a
        # workspace, the padded coefficient and deformation buffers are reused
        # across pairs (refilled in place); results are unchanged.
        if predictor === nothing
            itpA = itpB = nothing
            warpbufs = (nothing, nothing)
            dctx = nothing
        elseif workspace === nothing
            itpA = image_interpolant(imgA, T)
            itpB = image_interpolant(imgB, T)
            # A backend with a deform context stages this pair's coefficients
            # where its engine computes and keeps the warped pair resident
            # there — image pair in, device-side accumulate, vector grid out.
            dctx = _deform_context(backend, nothing, itpA, itpB, imgsize, T)
            warpbufs = (nothing, nothing)
        else
            itpA, workspace.itpA_coefs = image_interpolant!(workspace.itpA_coefs, imgA, T)
            itpB, workspace.itpB_coefs = image_interpolant!(workspace.itpB_coefs, imgB, T)
            dctx = _deform_context(backend, workspace, itpA, itpB, imgsize, T)
            if dctx === nothing
                workspace.warpA === nothing && (workspace.warpA = Matrix{T}(undef, imgsize))
                workspace.warpB === nothing && (workspace.warpB = Matrix{T}(undef, imgsize))
                warpbufs = (workspace.warpA, workspace.warpB)
            else
                warpbufs = (nothing, nothing)
            end
        end
        warpA, warpB, pu, pv = apply_predictor(backend, imgA, imgB, itpA, itpB, predictor,
                                               grid.x, grid.y, T; threaded,
                                               warpA = warpbufs[1], warpB = warpbufs[2],
                                               ctx = dctx)
        source_context = _original_support_context(imgA, imgB, mask, T, workspace)
        source_gate = _original_source_gate(source_context, predictor, grid, params, mask;
                                           gate = source_gate, threaded)
        observation===nothing || _ensemble_source!(observation,source_context,predictor,source_gate,grid.jobs)
        u, v = pu, pv  # identical for every pair (shared predictor)
        observing=observation===nothing ? (;source_gate) : (;source_gate,diagnostics=observation)
        if length(chunks) == 1
            accumulate_planes!(accum, chunks[1], engines[1], warpA, warpB,
                               grid.jobs, params, mask, uacc,
                               uscratch === nothing ? nothing : uscratch[1]; observing...)
        elseif !isempty(chunks)
            @sync for (ci, cr) in enumerate(chunks)
                Threads.@spawn accumulate_planes!(accum, cr, engines[ci],
                                                  warpA, warpB, grid.jobs, params, mask,
                                                  uacc,
                                                  uscratch === nothing ? nothing : uscratch[ci]; observing...)
            end
        end
        next!(meter)
        if on_contribution!==nothing
            event=(pass_index=contribution_pass,pair_index=pair_index,
                completed_contributions=_ensemble_add(_ensemble_mul(contribution_pass-1,length(pairs)),pair_index),
                total_contributions=total_contributions,input_pairs=length(pairs),scheduled_passes=scheduled_passes,
                completed_pools=0,published_results=0)
            on_contribution(event)
        end
        _ensemble_cancel_check(cancel_requested)
    end

    ny, nx = length(grid.y), length(grid.x)
    peak_ratio = zeros(T, ny, nx)
    correlation_moment = zeros(T, ny, nx)
    uncertainty_u = fill(T(NaN), ny, nx)
    uncertainty_v = fill(T(NaN), ny, nx)
    # Opt-in full-plane storage: the summed ensemble plane per window.
    planes = params.keep_correlation_planes ?
             fill!(Matrix{Union{Nothing,Matrix{T}}}(undef, ny, nx), nothing) : nothing
    n_alt = params.n_peaks - 1
    alt_u = n_alt > 0 ? fill(T(NaN), ny, nx, n_alt) : nothing
    alt_v = n_alt > 0 ? fill(T(NaN), ny, nx, n_alt) : nothing
    observing=observation===nothing ? (;) : (;diagnostics=observation)
    isempty(grid.jobs) ||
        ensemble_analyze!(accum, engines[1], u, v, peak_ratio, correlation_moment,
                          uncertainty_u, uncertainty_v, planes, alt_u, alt_v,
                          grid.jobs, params, uacc;observing...)
    if any(grid.grid_mask)
        for f in (u, v, peak_ratio, correlation_moment)
            f[grid.grid_mask] .= T(NaN)
        end
    end

    result = PIVResult(grid.x, grid.y, u, v, peak_ratio, correlation_moment,
                       uncertainty_u, uncertainty_v,
                       falses(ny, nx), grid.grid_mask, params, planes)
    observation===nothing || push!(diagnostics,_ensemble_pass_finish(observation,pass_index,params,T,imgsize,grid,predictor,force_replace,uacc,uncertainty_u,uncertainty_v))
    return validate_and_replace!(result, params, force_replace;
                                 alternatives = alt_u === nothing ? nothing : (alt_u, alt_v))
end

_uncertainty_accumulator(::_CPUCorrelationEngine, ::Type, njobs::Int) =
    new_uncertainty_stats(njobs)
_uncertainty_scratch(engine::_CPUCorrelationEngine, ::Type{T}) where {T} =
    uncertainty_scratch(T, size(engine.correlator.apod))

# Per-window ensemble plane accumulators for the CPU engine: plain host
# matrices, one per window. KA-family engines supply their own batch-major
# accumulator (see `_KAPlaneAccumulator` in ka_backend.jl).
_plane_accumulator(::_CPUCorrelationEngine, params::PIVParameters,
                   ::Type{T}, njobs::Int) where {T} =
    [zeros(T, params.padding ? 2 .* params.search_area_size : params.search_area_size)
     for _ in 1:njobs]

# Peak-find and post-process every summed ensemble plane on the host: subpixel
# refinement, alternative-peak refinement, quality metrics, and (final pass)
# pooled-uncertainty finalization. `u`/`v` hold the shared predictor on entry
# and the total displacement on exit.
function ensemble_analyze!(accum::AbstractVector, engine, u, v, peak_ratio,
                           correlation_moment, uncertainty_u, uncertainty_v,
                           planes, alt_u, alt_v, jobs, params::PIVParameters, uacc;
                           diagnostics = nothing)
    T = eltype(u)
    k = max(params.n_peaks, 2)
    vals = Vector{T}(undef, k)
    locs = Vector{NTuple{2,Int}}(undef, k)
    for (j, (gi, gj, _, _)) in enumerate(jobs)
        R = accum[j]
        planes === nothing || (planes[gi, gj] = copy(R))
        res = analyze_plane!(vals, locs, R, params)
        diagnostics===nothing || _ensemble_residual!(diagnostics,gi,gj,res.du,res.dv)
        if alt_u !== nothing
            # Total alternative displacement = shared predictor + residual.
            for m in 2:min(res.found, params.n_peaks)
                aref = subpixel_gauss3(R, locs[m])
                alt_u[gi, gj, m - 1] = u[gi, gj] + (aref[2] - res.center[2])
                alt_v[gi, gj, m - 1] = v[gi, gj] + (aref[1] - res.center[1])
            end
        end
        u[gi, gj] += res.du
        v[gi, gj] += res.dv
        peak_ratio[gi, gj] = res.ratio
        correlation_moment[gi, gj] = res.moment
        if uacc !== nothing
            uncertainty_u[gi, gj] = finalize_uncertainty(T, view(uacc[j], 1, :))
            uncertainty_v[gi, gj] = finalize_uncertainty(T, view(uacc[j], 2, :))
        end
    end
    return nothing
end

# Add the correlation planes of the windows in jobrange to their accumulators,
# and (when enabled) pool each window's uncertainty statistics across pairs.
function accumulate_planes!(accum, jobrange, engine, imgA, imgB, jobs,
                            params::PIVParameters, mask,
                            uacc = nothing, uscratch = nothing; source_gate = nothing,
                            diagnostics = nothing)
    wr, wc = params.window_size
    sr, sc = params.search_area_size
    mr, mc = div.(params.search_area_size .- params.window_size, 2)
    for j in jobrange
        gi, gj, rs, cs = jobs[j]
        _source_informative(source_gate, gi, gj) || continue
        subA = @view imgA[rs:(rs + wr - 1), cs:(cs + wc - 1)]
        subB_uq = @view imgB[rs:(rs + wr - 1), cs:(cs + wc - 1)]
        srs, scs = rs - mr, cs - mc
        subB = @view imgB[srs:(srs + sr - 1), scs:(scs + sc - 1)]
        submaskA = mask === nothing ? nothing :
                  view(mask, rs:(rs + wr - 1), cs:(cs + wc - 1))
        submaskB = mask === nothing ? nothing :
                   (params.search_area_size == params.window_size ? submaskA :
                    view(mask, srs:(srs + sr - 1), scs:(scs + sc - 1)))
        # Fully clean windows take the unmasked fast path.
        submaskA !== nothing && !any(submaskA) && (submaskA = nothing)
        submaskB !== nothing && !any(submaskB) && (submaskB = nothing)
        R=_correlation_plane!(engine, subA, subB, (submaskA, submaskB))
        diagnostics===nothing || _ensemble_plane!(diagnostics,j,_ensemble_plane_category(R))
        accum[j] .+= R
        uacc === nothing ||
            accumulate_uncertainty!(uacc[j], uscratch, subA, subB_uq, submaskA,
                                    _correlation_apod(engine))
    end
    return nothing
end
