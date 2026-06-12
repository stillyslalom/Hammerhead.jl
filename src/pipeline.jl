"""
    run_piv(imgA, imgB, passes::AbstractVector{PIVParameters};
            threaded = Threads.nthreads() > 1,
            predictor_smoothing = true) -> PIVResult
    run_piv(imgA, imgB, params::PIVParameters = PIVParameters(); kwargs...)

Run a (multi-pass) PIV analysis on an in-memory image pair. `imgA` and `imgB`
must be equally sized real-valued matrices (load and convert image files with
your preferred image package).

Each pass tiles the images into interrogation windows (`window_size`,
`overlap`), correlates each window pair (`correlation_method`, optionally
zero-padded and apodized), refines the peak to subpixel precision
(`subpixel_method`), and validates the resulting field (universal outlier
detection, peak-ratio threshold, and any additional `validation` pipeline)
with local-median replacement of invalid vectors.

From the second pass on, the previous pass's validated field is used as a
predictor: it is smoothed (`predictor_smoothing`), interpolated to pixel
resolution, and both images are symmetrically deformed by ±half the predictor
displacement (central-difference image deformation, cubic B-spline
resampling). The pass then measures only the small residual displacement, so
window sizes can shrink across passes — e.g. `multipass_parameters([64, 32,
16])` — without violating the quarter-window displacement limit. Repeating the
final window size adds convergence sweeps.

With `threaded = true` (the default on multithreaded sessions) the window grid
of each pass is split across tasks; results are identical to the serial path.

Returns the [`PIVResult`](@ref) of the final pass.
"""
function run_piv(imgA::AbstractMatrix{<:Real}, imgB::AbstractMatrix{<:Real},
                 passes::AbstractVector{PIVParameters};
                 threaded::Bool = Threads.nthreads() > 1,
                 predictor_smoothing::Bool = true)
    isempty(passes) && throw(ArgumentError("at least one pass is required"))
    size(imgA) == size(imgB) ||
        throw(DimensionMismatch("images must have the same size, got $(size(imgA)) and $(size(imgB))"))

    result = nothing
    for (k, params) in enumerate(passes)
        predictor = if result === nothing
            nothing
        else
            (x = result.x, y = result.y,
             u = predictor_smoothing ? smooth_field(result.u) : result.u,
             v = predictor_smoothing ? smooth_field(result.v) : result.v)
        end
        # Intermediate passes always replace invalid vectors: the predictor
        # field must stay well behaved for the deformation to converge.
        result = piv_pass(imgA, imgB, params, predictor;
                          threaded, force_replace = k < length(passes))
    end
    return result
end

run_piv(imgA::AbstractMatrix{<:Real}, imgB::AbstractMatrix{<:Real},
        params::PIVParameters = PIVParameters(); kwargs...) =
    run_piv(imgA, imgB, [params]; kwargs...)

"""
    multipass_parameters(window_sizes; overlap_fraction = 0.5, kwargs...) -> Vector{PIVParameters}

Build a multi-pass schedule with one `PIVParameters` per entry of
`window_sizes` (integers or `(rows, cols)` tuples), each with `overlap =
floor(window_size * overlap_fraction)`. All remaining keyword arguments are
forwarded to every pass.

```julia
passes = multipass_parameters([64, 32, 16]; padding = true, apodization = :gauss)
result = run_piv(imgA, imgB, passes)
```
"""
function multipass_parameters(window_sizes::AbstractVector;
                              overlap_fraction::Real = 0.5, kwargs...)
    0 <= overlap_fraction < 1 ||
        throw(ArgumentError("overlap_fraction must be in [0, 1), got $overlap_fraction"))
    isempty(window_sizes) && throw(ArgumentError("window_sizes must not be empty"))
    return [begin
                ws = w isa Integer ? (Int(w), Int(w)) : (Int(w[1]), Int(w[2]))
                PIVParameters(; window_size = ws,
                              overlap = floor.(Int, ws .* overlap_fraction), kwargs...)
            end
            for w in window_sizes]
end

# One interrogation pass: deform by the predictor, correlate the residual on
# the pass grid, add the predictor back, then validate and (maybe) replace.
function piv_pass(imgA::AbstractMatrix, imgB::AbstractMatrix, params::PIVParameters,
                  predictor; threaded::Bool, force_replace::Bool)
    nr, nc = size(imgA)
    wr, wc = params.window_size
    (wr <= nr && wc <= nc) ||
        throw(ArgumentError("window size $(params.window_size) exceeds image size $((nr, nc))"))

    step_r = wr - params.overlap[1]
    step_c = wc - params.overlap[2]
    row_starts = 1:step_r:(nr - wr + 1)
    col_starts = 1:step_c:(nc - wc + 1)
    ny = length(row_starts)
    nx = length(col_starts)

    x = [cs + (wc - 1) / 2 for cs in col_starts]
    y = [rs + (wr - 1) / 2 for rs in row_starts]

    if predictor === nothing
        warpA, warpB = imgA, imgB
        u = zeros(ny, nx)
        v = zeros(ny, nx)
    else
        itp_u = extrapolate(interpolate((predictor.y, predictor.x), predictor.u,
                                        Gridded(Linear())), Flat())
        itp_v = extrapolate(interpolate((predictor.y, predictor.x), predictor.v,
                                        Gridded(Linear())), Flat())
        warpA, warpB = deform_images(imgA, imgB, itp_u, itp_v)
        u = [Float64(itp_u(yi, xj)) for yi in y, xj in x]
        v = [Float64(itp_v(yi, xj)) for yi in y, xj in x]
    end

    residual_u = zeros(ny, nx)
    residual_v = zeros(ny, nx)
    peak_ratio = zeros(ny, nx)
    correlation_moment = zeros(ny, nx)

    jobs = vec([(gi, gj, rs, cs) for (gi, rs) in enumerate(row_starts),
                                     (gj, cs) in enumerate(col_starts)])
    nchunks = threaded ? min(Threads.nthreads(), length(jobs)) : 1
    if nchunks == 1
        process_windows!(residual_u, residual_v, peak_ratio, correlation_moment,
                         jobs, warpA, warpB, params)
    else
        chunk_size = cld(length(jobs), nchunks)
        @sync for chunk in Iterators.partition(jobs, chunk_size)
            Threads.@spawn process_windows!(residual_u, residual_v, peak_ratio,
                                            correlation_moment, chunk, warpA, warpB, params)
        end
    end
    u .+= residual_u
    v .+= residual_v

    result = PIVResult(x, y, u, v, peak_ratio, correlation_moment, falses(ny, nx), params)

    if params.uod_enable
        apply_validator!(result, UniversalOutlierValidator(params.uod_threshold;
            neighborhood_size = params.uod_neighborhood))
    end
    if params.min_peak_ratio > 1
        apply_validator!(result, PeakRatioValidator(params.min_peak_ratio))
    end
    validate_vectors!(result, params.validation)

    if force_replace || params.replace_outliers
        replace_vectors!(result.u, result.v, result.outliers)
    end

    return result
end

"""
    deform_images(imgA, imgB, itp_u, itp_v) -> (warpedA, warpedB)

Symmetric (central-difference) image deformation: each image is resampled with
cubic B-spline interpolation, `imgA` shifted by −d/2 and `imgB` by +d/2, where
`d = (itp_u, itp_v)` is the displacement field evaluated at each pixel. After
deformation, content displaced by exactly `d` is aligned in both outputs, so
correlating them measures the residual displacement. Out-of-image samples are
zero-filled.
"""
function deform_images(imgA::AbstractMatrix{<:Real}, imgB::AbstractMatrix{<:Real},
                       itp_u, itp_v)
    T = float(promote_type(eltype(imgA), eltype(imgB)))
    itpA = extrapolate(interpolate(T.(imgA), BSpline(Cubic(Line(OnGrid())))), zero(T))
    itpB = extrapolate(interpolate(T.(imgB), BSpline(Cubic(Line(OnGrid())))), zero(T))
    warpA = Matrix{T}(undef, size(imgA))
    warpB = Matrix{T}(undef, size(imgB))
    @inbounds for c in axes(imgA, 2), r in axes(imgA, 1)
        du2 = itp_u(r, c) / 2
        dv2 = itp_v(r, c) / 2
        warpA[r, c] = itpA(r - dv2, c - du2)
        warpB[r, c] = itpB(r + dv2, c + du2)
    end
    return warpA, warpB
end

make_correlator(params::PIVParameters, ::Type{T}) where {T} =
    params.correlation_method === :cross ?
        CrossCorrelator{T}(params.window_size; params.padding, params.apodization) :
        PhaseCorrelator{T}(params.window_size; params.padding, params.apodization)

# Each call gets its own correlator, so chunks can run on concurrent tasks.
# Every (gi, gj) is written by exactly one job and the per-window math doesn't
# depend on chunking, so threaded results match serial results exactly.
function process_windows!(u, v, peak_ratio, correlation_moment, jobs,
                          imgA::AbstractMatrix, imgB::AbstractMatrix, params::PIVParameters)
    wr, wc = params.window_size
    T = float(promote_type(eltype(imgA), eltype(imgB)))
    correlator = make_correlator(params, T)
    for (gi, gj, rs, cs) in jobs
        subA = @view imgA[rs:(rs + wr - 1), cs:(cs + wc - 1)]
        subB = @view imgB[rs:(rs + wr - 1), cs:(cs + wc - 1)]
        res = correlate(correlator, subA, subB; subpixel = params.subpixel_method)
        u[gi, gj] = res.du
        v[gi, gj] = res.dv
        peak_ratio[gi, gj] = calculate_peak_ratio(res.correlation, res.peakloc)
        correlation_moment[gi, gj] = calculate_correlation_moment(
            res.correlation, res.refined_peakloc)
    end
    return nothing
end
