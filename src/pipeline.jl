"""
    run_piv(imgA, imgB, params::PIVParameters = PIVParameters();
            threaded = Threads.nthreads() > 1) -> PIVResult

Run a PIV analysis on an in-memory image pair. `imgA` and `imgB` must be
equally sized real-valued matrices (load and convert image files with your
preferred image package).

The images are tiled into interrogation windows of `params.window_size` with
`params.overlap`; each window pair is correlated (`params.correlation_method`,
optionally zero-padded and apodized), refined to subpixel precision
(`params.subpixel_method`), and optionally re-correlated against a back-warped
second window (`params.deformation_iterations > 0`). Peak ratio and correlation
moment are recorded per window, and universal outlier detection is applied to
the resulting field when `params.uod_enable` is set.

With `threaded = true` (the default on multithreaded sessions) the window grid
is split across tasks, each with its own correlator; results are identical to
the serial path.

See [`PIVResult`](@ref) for the output layout and sign convention.
"""
function run_piv(imgA::AbstractMatrix{<:Real}, imgB::AbstractMatrix{<:Real},
                 params::PIVParameters = PIVParameters();
                 threaded::Bool = Threads.nthreads() > 1)
    size(imgA) == size(imgB) ||
        throw(DimensionMismatch("images must have the same size, got $(size(imgA)) and $(size(imgB))"))
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

    u = zeros(ny, nx)
    v = zeros(ny, nx)
    peak_ratio = zeros(ny, nx)
    correlation_moment = zeros(ny, nx)

    jobs = vec([(gi, gj, rs, cs) for (gi, rs) in enumerate(row_starts),
                                     (gj, cs) in enumerate(col_starts)])
    nchunks = threaded ? min(Threads.nthreads(), length(jobs)) : 1
    if nchunks == 1
        process_windows!(u, v, peak_ratio, correlation_moment, jobs, imgA, imgB, params)
    else
        chunk_size = cld(length(jobs), nchunks)
        @sync for chunk in Iterators.partition(jobs, chunk_size)
            Threads.@spawn process_windows!(u, v, peak_ratio, correlation_moment,
                                            chunk, imgA, imgB, params)
        end
    end

    outliers = params.uod_enable ?
        universal_outlier_detection(u, v, params.uod_threshold;
                                    neighborhood_size = params.uod_neighborhood) :
        falses(ny, nx)

    return PIVResult(x, y, u, v, peak_ratio, correlation_moment, outliers, params)
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
        res = if params.deformation_iterations > 0
            correlate_deformable(correlator, subA, subB;
                                 iterations = params.deformation_iterations,
                                 subpixel = params.subpixel_method)
        else
            correlate(correlator, subA, subB; subpixel = params.subpixel_method)
        end
        u[gi, gj] = res.du
        v[gi, gj] = res.dv
        peak_ratio[gi, gj] = calculate_peak_ratio(res.correlation, res.peakloc)
        correlation_moment[gi, gj] = calculate_correlation_moment(
            res.correlation, res.refined_peakloc)
    end
    return nothing
end
