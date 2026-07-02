# Ensemble (sum-of-correlation) PIV: average the correlation planes of
# corresponding windows across many image pairs before peak detection
# (Meinhart, Wereley & Santiago 2000). For statistically stationary flow
# whose individual pairs are too noisy for reliable peaks — micro-PIV and
# other low-SNR recordings.

"""
    run_piv_ensemble(pairs, params = PIVParameters(); kwargs...) -> PIVResult

Ensemble PIV over a sequence of image pairs: each interrogation window's
correlation planes are summed across all pairs and the displacement peak is
located once on the ensemble plane, so a peak too weak to detect in any
single pair emerges from the average. Assumes statistically stationary flow;
the result is the ensemble-mean displacement field.

`pairs` is as in [`run_piv_sequence`](@ref) (2-tuples of file paths and/or
matrices; paths are reloaded once per pass). With a multi-pass schedule the
previous pass's ensemble field acts as a shared deformation predictor for
every pair. `peak_ratio` and `correlation_moment` describe the ensemble
planes.

Keyword arguments: `threaded`, `predictor_smoothing`, `mask`,
`mask_threshold` as in [`run_piv`](@ref); `preprocess`, `image_type`,
`progress` as in [`run_piv_sequence`](@ref).
"""
function run_piv_ensemble(pairs::AbstractVector,
                          params::Union{PIVParameters,AbstractVector{PIVParameters}} = PIVParameters();
                          threaded::Bool = Threads.nthreads() > 1,
                          predictor_smoothing::Bool = true,
                          mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                          mask_threshold::Real = 0.5,
                          preprocess = nothing,
                          image_type::Type{<:AbstractFloat} = Float64,
                          progress::Bool = true)
    passes = params isa PIVParameters ? [params] : params
    isempty(pairs) && throw(ArgumentError("pairs must not be empty"))
    isempty(passes) && throw(ArgumentError("at least one pass is required"))
    0 < mask_threshold <= 1 ||
        throw(ArgumentError("mask_threshold must be in (0, 1], got $mask_threshold"))

    meter = Progress(length(passes) * length(pairs);
                     desc = "Ensemble PIV: ", enabled = progress)
    result = nothing
    for (k, p) in enumerate(passes)
        predictor = result === nothing ? nothing :
                    build_predictor(result, predictor_smoothing)
        result = ensemble_pass(pairs, p, predictor; threaded, mask, mask_threshold,
                               preprocess, image_type,
                               force_replace = k < length(passes), meter)
    end
    return result
end

# One ensemble pass: deform every pair by the shared predictor, sum each
# window's correlation planes across pairs, then peak-find and validate once.
function ensemble_pass(pairs, params::PIVParameters, predictor;
                       threaded::Bool, mask, mask_threshold, preprocess,
                       image_type, force_replace::Bool, meter)
    local T, grid, accum, chunks, correlators, u, v, imgsize
    first_pair = true
    for pair in pairs
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
            plane = params.padding ? 2 .* params.window_size : params.window_size
            accum = [zeros(T, plane) for _ in grid.jobs]
            nchunks = threaded ? min(Threads.nthreads(), length(grid.jobs)) : 1
            chunk_size = max(cld(length(grid.jobs), max(nchunks, 1)), 1)
            chunks = collect(Iterators.partition(1:length(grid.jobs), chunk_size))
            # One correlator per chunk, reused across pairs: FFTW plans are
            # paid once per pass, and each accumulator is written by exactly
            # one task per pair, so threaded results match serial exactly.
            correlators = [make_correlator(params, T) for _ in chunks]
            first_pair = false
        else
            size(imgA) == imgsize ||
                throw(DimensionMismatch("all pairs must share the image size $imgsize, got $(size(imgA))"))
        end
        warpA, warpB, pu, pv = apply_predictor(imgA, imgB, predictor, grid.x, grid.y, T)
        u, v = pu, pv  # identical for every pair (shared predictor)
        if length(chunks) == 1
            accumulate_planes!(accum, chunks[1], correlators[1], warpA, warpB,
                               grid.jobs, params, mask)
        elseif !isempty(chunks)
            @sync for (ci, cr) in enumerate(chunks)
                Threads.@spawn accumulate_planes!(accum, cr, correlators[ci],
                                                  warpA, warpB, grid.jobs, params, mask)
            end
        end
        next!(meter)
    end

    ny, nx = length(grid.y), length(grid.x)
    peak_ratio = zeros(T, ny, nx)
    correlation_moment = zeros(T, ny, nx)
    n_alt = params.n_peaks - 1
    alt_u = n_alt > 0 ? fill(T(NaN), ny, nx, n_alt) : nothing
    alt_v = n_alt > 0 ? fill(T(NaN), ny, nx, n_alt) : nothing
    k = max(params.n_peaks, 2)
    vals = Vector{T}(undef, k)
    locs = Vector{NTuple{2,Int}}(undef, k)
    for (j, (gi, gj, _, _)) in enumerate(grid.jobs)
        R = accum[j]
        res = analyze_plane!(vals, locs, R, params)
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
    end
    if any(grid.grid_mask)
        for f in (u, v, peak_ratio, correlation_moment)
            f[grid.grid_mask] .= T(NaN)
        end
    end

    result = PIVResult(grid.x, grid.y, u, v, peak_ratio, correlation_moment,
                       falses(ny, nx), grid.grid_mask, params)
    return validate_and_replace!(result, params, force_replace;
                                 alternatives = alt_u === nothing ? nothing : (alt_u, alt_v))
end

# Add the correlation planes of the windows in jobrange to their accumulators.
function accumulate_planes!(accum, jobrange, correlator, imgA, imgB, jobs,
                            params::PIVParameters, mask)
    wr, wc = params.window_size
    for j in jobrange
        gi, gj, rs, cs = jobs[j]
        subA = @view imgA[rs:(rs + wr - 1), cs:(cs + wc - 1)]
        subB = @view imgB[rs:(rs + wr - 1), cs:(cs + wc - 1)]
        submask = mask === nothing ? nothing :
                  view(mask, rs:(rs + wr - 1), cs:(cs + wc - 1))
        # Fully clean windows take the unmasked fast path.
        submask !== nothing && !any(submask) && (submask = nothing)
        accum[j] .+= correlation_plane!(correlator, subA, subB, submask)
    end
    return nothing
end
